# arXiv:2505.19147v3[cs.CL]12Oct2025

## Shifting AI Efficiency From Model-Centric to Data-Centric Compression

Xuyang Liu1,2* Zichen Wen1,3,4∗ Shaobo Wang1∗ Junjie Chen1 Zhishan Tao1 Yubo Wang1 Tailai Chen1 Xiangqi Jin1,3 Chang Zou1,3 Yiyu Wang1 Chenfei Liao6

Xu Zheng6 Honggang Chen2 Weijia Li4,5 Xuming Hu6 Conghui He4 Linfeng Zhang1

1EPIC Lab, Shanghai Jiao Tong University 2Sichuan University 3University of Electronic Science & Technology of China 4Shanghai AI Laboratory 5Sun Yat-sen University 6Hong Kong University of Science and Technology (Guangzhou)

#### Project: Awesome-Token-level-Model-Compression Abstract

model scale, with larger models consistently showing superior performance in reasoning, knowledge acquisition, and task generalization. The evolution from early models like BERT (117M) (Devlin et al., 2018) to today’s state-of-the-art LLMs such as DeepSeek-R1 (Guo et al., 2025a) and Qwen3 (Yang et al., 2025a) (100B+) illustrates how scale has delivered substantial performance improvements. Nevertheless, this pursuit of performance through larger models incurs ever-increasing computational costs. As a result, by early 2024, the dominant source of computational overhead was primarily attributed to the linear growth in parameter count and associated memory requirements.

The advancement of large language models (LLMs) and multi-modal LLMs (MLLMs) has historically relied on scaling model parameters. However, as hardware limits constrain further model growth, the primary computational bottleneck has shifted to the quadratic cost of self-attention over increasingly long sequences by ultra-long text contexts, high-resolution images, and extended videos. In this position paper, we argue that the focus of research for efficient artificial intelligence (AI) is shifting from model-centric compression to datacentric compression. We position data-centric compression as the emerging paradigm, which improves AI efficiency by directly compressing the volume of data processed during model training or inference. To formalize this shift, we establish a unified framework for existing efficiency strategies and demonstrate why it constitutes a crucial paradigm change for longcontext AI. We then systematically review the landscape of data-centric compression methods, analyzing their benefits across diverse scenarios. Finally, we outline key challenges and promising future research directions. Our work aims to provide a novel perspective on AI efficiency, synthesize existing efforts, and catalyze innovation to address the challenges posed by ever-increasing context lengths.

In response to this scaling trend, the research community has developed numerous model-centric compression techniques, including model quantization (Yang et al., 2019; Rokh et al., 2023), network pruning (Han et al., 2016; Cheng et al., 2024), knowledge distillation (Hinton et al., 2015; Gou et al., 2021), and low-rank decomposition (Yu et al., 2017; Idelbayev and Carreira-Perpinán, 2020). These methods reduce computational overhead by decreasing model size and were a natural response to the 2022–2024 era, when scaling model size was the primary driver of performance gains.

As model sizes approach hardware limits, the pace of parameter growth is flattening. Meanwhile, a new computational challenge has emerged: the exponential growth in context sequence lengths. Figure 1 (left) shows that from 2022 to 2024, model size primarily drove computational costs, reaching around 1T parameters before stagnating. Since then, the dominant factor has shifted dramatically to the staggering number of processed tokens, which continues to grow exponentially. This trend spans multiple domains: language models now handle context lengths orders of magnitude longer than before (Meta, 2025; Yang et al., 2025a), especially with long chain-of-thought reasoning (Guo et al., 2025a) and multi-agent systems (Han et al.,

#### 1 Introduction

The explosive growth of large language models (LLMs) (OpenAI, 2023; Touvron et al., 2023; Grattafiori et al., 2024; Dong et al., 2024; Yang et al., 2025a; Guo et al., 2025a) and their multimodal extensions (MLLMs) (Liu et al., 2024c; Chen et al., 2024d; Zhu et al., 2025; Wang et al., 2024a; Bai et al., 2025) over the past few years has driven remarkable gains in AI capabilities. This progress has been largely achieved by increasing

*Equal contribution: liuxuyang@stu.scu.edu.cn Corresponding author: zhanglinfeng@sjtu.edu.cn

[Figure 1]

[Figure 2]

a paradigm shift toward optimizing context length via token compression

ContextLength

[Figure 3]

[Figure 4]

the growth of model size has slowed down

7B 20B 20B 235B 456B Model Size

Year

- Figure 1: The evolution of AI efficiency: from model-centric to data-centric compression. From 2022 to 2024, AI model performance gains were primarily driven by scaling model size, directing efficiency research toward model-centric compression. By 2024, with model sizes approaching 1T parameters, their growth has slowed down. Consequently, the focus has shifted to expanding context length to enhance model capabilities, necessitating a transition to data-centric compression that reduces context length for efficiency.

2024a); vision models process increasingly highresolution images (Zhu et al., 2025; Bai et al., 2025) and longer videos (Qin et al., 2025); and generative models create higher-resolution images (Labs,

various domains, revealing a critical transition from parameter-centric to context-centric computational bottlenecks that necessitate a paradigm shift in efficiency optimization.

- 2024) and hour-long videos (Brooks et al., 2024), all requiring more tokens and causing substantial computational overhead. Consequently, by late

- • Unified Formulation of Model Efficiency: We establish a comprehensive mathematical formulation that unifies architectural design, modelcentric compression, and data-centric compression within a single expression.
- • Systematic Review of Data-centric Compression: We present a thorough investigation of data-centric compression methods, constructing a unified framework to categorize diverse approaches while analyzing their benefits across different scenarios and tasks.
- • Challenges and Future Directions: We provide an in-depth analysis of current challenges in data-centric compression research and propose promising future directions, aiming to catalyze research efforts toward more efficient and effective compression methods.

- 2024, the primary bottleneck has clearly shifted to the quadratic cost of the attention mechanism over these extremely long context sequences.

This unprecedented growth in sequence lengths has shifted the computational bottleneck from model size to the quadratic cost of attention over long context sequences. Based on this observation, as illustrated in Figure 1 (right), we propose a critical position: the AI community should shift its efficiency optimization paradigm from modelcentric to data-centric compression. We advocate for data-centric compression that directly reduces the volume of data processed during model training or inference (Jiang et al., 2023; Bolya

- et al., 2023b; Bolya and Hoffman, 2023; Chen
- et al., 2024b; Lin et al., 2024b). These methods address computational overhead by removing lowinformation content during processing, typically without modifying model architectures or requiring retraining. Our analysis in Section 3.3 shows that they offer compelling advantages in universality, efficiency, and compatibility, positioning data-centric compression as a promising solution for efficient next-generation LLMs and MLLMs.

#### 2 Background

##### 2.1 Token Overhead across Various Domains

The field of AI has witnessed remarkable advancements across multiple domains, including natural language processing, computer vision, and content generation. These developments have been largely driven by the introduction of the Transformer architecture (Vaswani et al., 2017), which has spawned a wide variety of models. As these domains evolve, we observe a significant increase in token sequence lengths across three main areas:

Building upon these analyses, we make four key contributions in this position paper:

• Evolution of AI Efficiency: We analyze recent developments in long-context AI across

##### (I) Longer Context Length in Language Models:

Large language models (LLMs) (OpenAI, 2023; Touvron et al., 2023; Grattafiori et al., 2024; Liu

- et al., 2024b; Bai et al., 2023a; Yang et al., 2025a) have demonstrated remarkable capabilities in natural language understanding and generation. The context length LLMs can handle has expanded dramatically from 2,048 tokens in early models like Llama 1 (Touvron et al., 2023) to 10M tokens in recent iterations like Llama 4 Scout (Meta, 2025). This expansion has led to the emergence of large reasoning models (Guo et al., 2025a; Yang et al.,

2025a), which focus on complex multi-step problem solving through techniques like long chain-ofthought reasoning (Liu et al., 2024a) and multiagent collaboration (Han et al., 2024a).

- (II) Higher Resolution and Longer Video Understanding: Building on the success of LLMs, multimodal large language models (MLLMs) (Liu et al.,

- 2024c; Li et al., 2024a; Bai et al., 2023b, 2025; Chen et al., 2024d; Zhu et al., 2025; Guo et al.,

2025b) extend these capabilities by integrating vision and text processing (Wu et al., 2023). Visual inputs processed by MLLMs have evolved from basic 224 × 224 resolution images in early models like LLaVA (Liu et al., 2023) to 4K ultra-highresolution images in InternVL3 (Zhu et al., 2025) and 10K-frame videos in Video-XL-2 (Qin et al., 2025), enabling strong performance on image (Bai et al., 2025), video (Yang et al., 2025b), and multimodal reasoning tasks (Shen et al., 2025).

- (III) More Complex Content in Generation Tasks: AI content generation has advanced significantly with the application of Transformers to generative domains (Peebles and Xie, 2023; Brooks et al., 2024; Li et al., 2024f). Early diffusion models like Stable Diffusion (Rombach et al., 2022) generated only 512 × 512 resolution images. With Transformers now successfully applied to generation (Peebles and Xie, 2023; Gao et al., 2023; Brooks et al., 2024; Li et al., 2024f), DiT-based models produce high-quality 4K images in PixArtΣ (Chen et al., 2024a) and hour-long videos in Sora (Brooks et al., 2024). These models capture complex spatiotemporal dependencies, enabling high-fidelity content generation (Labs, 2024; Yang

- et al., 2024c; Wan et al., 2025a; Kang et al., 2025). While these advancements across domains have

demonstrated outstanding performance, they now face significant efficiency challenges due to the quadratic cost of attention mechanisms over extremely long token sequences. The growing trend toward longer contexts, whether in complex rea-

soning chains for language tasks, high-resolution images and longer videos for understanding, or high-fidelity content for generation, necessitates prioritizing research on model efficiency, particularly in mitigating the computational overhead of increasing context lengths. Detailed statistical analysis of this trend is provided in Appendix A.

##### 2.2 AI Efficiency from Different Perspectives

Improving model efficiency has been a key goal in deep learning research. Given input data X and network parameters W, a neural network F produces output Y through the transformation:

##### Y

###### = F

##### ( W

##### , X

output

network

weights

input

) (1)

Model efficiency can be optimized from three perspectives: (I) Efficient Computation Architecture designs efficient neural architectures F (Shen et al., 2021; Peng et al., 2023; Gu and Dao, 2023), (II) Model-centric Compression reduces model weights W (Hinton et al., 2015; Yang et al., 2019; Li et al., 2017; Yu et al., 2017), and (III) Data-centric Compression compresses token sequences from input data X (Rao et al., 2021; Jiang et al., 2023; Chen et al., 2024b; Zou et al., 2025).

- (I) Efficient Computation Architecture (F): Since computational efficiency is determined by architectural design, optimizing F is fundamental. Unlike Transformers with quadratic attention complexity O(n2) (Vaswani et al., 2017), recent methods achieve linear or sub-quadratic scaling: (i) linear attention reformulates attention for O(n) complexity (Katharopoulos et al., 2020; Shen et al., 2021); (ii) RWKV combines RNN-like O(n) scaling with transformer parallelism (Peng et al., 2023; Duan et al., 2024); (iii) State Space Models like Mamba use structured state spaces for O(n) complexity (Gu and Dao, 2023; Zhu et al.). These require retraining, motivating alternative approaches.
- (II) Model-centric Compression (W): Reducing parameter complexity lowers computational and memory costs. Model compression is modelcentric, transforming W to a smaller W′:

##### W′ = Γ(W), where |W′| < |W| (2)

with Γ as the compression operator. Key methods include: (i) network pruning (Liu et al., 2019; Cheng et al., 2024); (ii) quantization (Yang et al., 2019; Rokh et al., 2023); (iii) knowledge distillation (Hinton et al., 2015; Gou et al., 2021); (iv)

low-rank decomposition (Yu et al., 2017; Idelbayev and Carreira-Perpinán, 2020). As model sizes plateau and context lengths grow, research is shifting toward data-centric compression.

(III) Data-centric Compression (X): Datacentric compression is a data-centric paradigm that improves efficiency by directly reducing the volume of data processed during training or inference. It encompasses two primary strategies: (i) dataset compression, which selects or distills informative subsets from the training corpus, and (ii) token compression, which directly reduces the length of input sequences during inference. Given an input sequence X, data-centric compression yields a compressed representation X′:

##### X′ = Φ(X), where |X′| < |X| (3)

with Φ as the data compression operator. This approach complements model-centric compression and has shown strong effectiveness in vision (Rao et al., 2021; Bolya et al., 2023b) and language domains (Kim and Cho, 2021; Jiang et al., 2023).

#### 3 How Data-centric Compression Drives Efficient and Effective Models

In this section, we begin with the research roadmap of data-centric compression in Section 3.1. Then, we comprehensively analyze the benefits of datacentric compression methods during both training and inference stages in Section 3.2. Finally, we summarize their advantages in Section 3.3.

- 3.1 Research Roadmap - What Makes Data-centric Compression Work?

Existing data-centric compression methods (i.e., token compression and dataset compression) fundamentally operate through a two-stage process (see Figure 2): first, identifying tokens eligible for compression within the existing token sequence X = [x1,x2,...,xT] using carefully designed compression criteria through a scoring function E : X → {st}Tt=1, and then determining the precise handling of these tokens through specific compression strategies P : (X,{st}Tt=1) → X′ that transform the original sequence into a compressed one where |X′| < |X|. Given that existing research primarily revolves around these two key components, we next systematically analyze their designs and review representative approaches.

Compression Criteria (E) To determine which tokens should be compressed in sequence X =

[x1,x2,...,xT], compression criteria employ scoring functions E to evaluate each token’s importance or redundancy. Based on whether additional parameters are introduced into original models, these criteria can be categorized into two main approaches:

##### (I) Parametric Methods employ auxiliary net-

works as scoring functions E∆θ : X → {st}Tt=1, introducing additional parameters ∆θ beyond the original model parameters θ. These methods include: (i) training-aware approaches (Rao et al., 2021; You et al., 2024; Li et al., 2024d; Kim et al., 2022) that optimize ∆θ through training to learn scoring function S∆θ : X → {st}Tt=1, and (ii) training-free approaches (Mahmud et al., 2024; Zhao et al., 2024a) that directly employ pre-trained networks as scoring function Sfixed : X → {st}Tt=1 without updating ∆θ.

###### (II) Non-parametric Methods utilize parameterfree heuristics for token scoring without introducing extra parameters. These approaches can be categorized into: (i) inherent computation methods (Liang et al., 2022; Zou et al., 2025; Chen et al., 2024b; Xiao et al., 2024; Ge et al., 2024) that leverage model’s internal calculations for token scoring Sin : A → {st}Tt=1, such as using

attention weights (st = Tj=1 ajt, where ajt represents attention score between tokens), and (ii) ex-

ternal computation methods (Bolya et al., 2023a; Zhang et al., 2025a; Devoto et al., 2024; Wang et al., 2024c; Liu et al., 2025a) that design additional metrics Sex : Z → RT×T to evaluate token relationships. For external methods, an additional function g : X → Z is introduced to compute intermediate features, where Z = g(X). The scoring function then operates on these features: si,j = f(zi,zj), where f is a custom pairwise scoring function. A typical example is using cosine similarity, where g is an identity function and si,j = ∥x⟨xi,xj⟩

i∥2∥xj∥2.

Compression Strategies (P) To reduce sequence length while preserving critical information, compression strategies P transform the original sequence based on token scores {st}Tt=1. These strategies can be categorized into two approaches: (I) Token Pruning directly discards less important tokens from the sequence based on their scores. These methods (Rao et al., 2021; Goyal et al., 2020; Jiang et al., 2023; Chen et al., 2024b) typically remove tokens with scores below a threshold, producing a compressed sequence:

###### X′ = X \ {xt | st < τ} (4)

### Compression Criteria (𝓔)

Compression Strategies Input

(𝓟)

[Figure 5]

[Figure 6]

Compressed

||x1|
|---|
<br><br>|x3|
|---|
|
|---|

Tokens

Tokens

|x2|
|---|

### Scores

Parametric

Token Pruning

|X′|
|---|

|X|
|---|

||s1|
|---|
<br><br>|s2|
|---|
<br><br>|s3|
|---|
<br><br>|s4|
|---|
|
|---|

|x4|
|---|

||x1′|
|---|
<br><br>| |
|---|
|x3′|
| |
|
|---|

|| |
|---|
|x1|
<br><br>| |
|---|
|x2|
<br><br>| |
|---|
|x3|
<br><br>| |
|---|
|x4|
|
|---|

[Figure 7]

[Figure 8]

||x1| |x2|
|---|---|---|
| | | |
|
|---|

Non-Parametric

Token Merging

[Figure 9]

[Figure 10]

[Figure 11]

||x3|
|---|
<br><br>| |
|---|
|x4|
|
|---|

|X′ < X|
|---|

- Figure 2: Overview of the data-centric compression paradigm. Given an input token sequence X = [x1,...,xT], data-centric compression first computes importance scores via a scoring function E : X → {st}Tt=1, then generates a compressed sequence X′ through a compression strategy P : (X,{st}Tt=1) → X′, where |X′| < |X|.

where τ is a threshold determining token removal. Token pruning reduces computation through direct elimination but risks information loss, particularly for fine-grained tasks (Xie et al., 2021).

troducing variability that enhances robustness and informativeness (Cubuk et al., 2018). In computer vision, mixing or combining image tokens creates novel representations that elevate training effectiveness (Zhang et al., 2018; Yun et al., 2019). This strategy has also been extended to synthetic datasets, where adaptive augmentation controls the informativeness of generated images (Zhao and Bilen, 2021; Lee et al., 2022b; Wang et al., 2025d,c, 2024b). Analogously, in natural language processing, augmenting text tokens through synonym replacement (Wei and Zou, 2019), contraction expansion (Coulombe, 2018), back-translation (Chen et al., 2020), and reformulation (Hao et al., 2025), supporting better generalization.

(II) Token Merging preserves information by combining semantically similar tokens (Bolya et al., 2023a; Zhang et al., 2025b; Bolya and Hoffman, 2023). Given an input sequence X = {x1,...,xT} and a mapping π : {1,...,T} → {1,...,M} that assigns tokens to M merge groups based on their semantic relationships, this approach generates a compressed sequence X′ = {x′1,...,x′M} through weighted aggregation:

st t′:π(t′)=m st′

x′m =

(5)

wtxt, wt =

t:π(t)=m

(ii) Token selection focuses on filtering out lowquality tokens to refine training data quality (Lin et al., 2024b; Lee et al., 2022a; Penedo et al., 2023; Wenzek et al., 2019; Gao et al., 2020; Li et al., 2024b; Wang et al., 2025b). Common approaches include rule-based heuristics (Raffel et al., 2020; Penedo et al., 2024), deduplication methods (Lee et al., 2022a; Penedo et al., 2023; Abbas et al., 2023), and scoring strategies leveraging large language models (Wenzek et al., 2019; Gao et al., 2020; Li et al., 2024b; Wettig et al., 2024; Sachdeva et al., 2024; Wang et al., 2025a).

where wt represents importance weights. Token merging preserves information through weighted combinations of tokens, offering a more nuanced approach than direct elimination.

- 3.2 Training and Inference Targets - How Data-centric Compression Benefits?

Training Stage Data-centric compression methods contribute to improving both the quality and efficiency of model training. These benefits can be broadly categorized into two aspects: enhancing training quality and increasing training efficiency.

- (I) Enhancing Training Quality Improvement in training quality can be achieved through methods such as data augmentation and token selection, which serve to increase data diversity and emphasize the most informative content, respectively. (i) Data augmentation techniques have been widely adopted to enrich training datasets by in-

Formally, consider a training batch B = {Xi}Ni=1, where each Xi = [xi,1,xi,2,...,xi,T] is a token sequence of length T. A quality scoring function q : T → R assigns each token xi,j ∈ T a score reflecting its informativeness or relevance. Using a threshold τ, tokens with scores below τ are filtered out via a mask mi:

1, q(xi,j) ≥ τ 0, otherwise

. The filtered batch B˜ consists of sequences:

mi,j =

X˜ i = {xi,j | mi,j = 1, j = 1,...,T}. (6)

Training on these curated, high-quality tokens enables the model to focus on the most relevant information, reducing noise and redundancy, thereby improving generalization and learning efficiency.

- (II) Increasing Training Efficiency Data-centric compression methods directly reduce the sequence length processed during training, addressing critical challenges associated with scaling large models (Bolya et al., 2023b; Choudhury et al., 2024; Shang et al., 2024; Xing et al., 2025). For Transformer architectures with sequence length reduced from n to m (m < n), the computational and memory benefits can be quantified as:

Ω(X′) Ω(X)

= O(m2d) O(n2d)

- m2

- n2

= O

,

(7)

M(X′) M(X) ≈

md nd

- m

- n

=

.

where d is the embedding dimension, Ω(·) represents the computational measure, and M(·) denotes the memory measure. This quadratic reduction in computation and linear reduction in memory enables faster training iterations and larger batch sizes on fixed hardware resources.

Inference Stage Data-centric compression methods can also enhance model inference efficiency through two key aspects: decreasing computational complexity and reducing memory usage.

- (I) Decreasing Computational Complexity: Following patterns established in training, data-centric compression achieves quadratic speedup in inference computations. Notably, many non-parametric compression methods (Bolya et al., 2023b; Chen et al., 2024b) can be directly integrated into inference without additional training or architectural modifications, enabling immediate benefits across domains (Zhang et al., 2025b; Wen et al., 2025b).
- (II) Reducing Memory Usage: Data-centric compression optimizes memory efficiency through two mechanisms: (i) computing memory reduction following the linear scaling pattern shown in training, and (ii) KV cache optimization for large language models (Li et al., 2024e; Cai et al., 2024; Wan et al., 2024, 2025b). During autoregressive generation, each layer caches key and value states

for attention computation, with memory growing with sequence length. For a sequence of length n compressed to length m, with L layers and hidden dimension d, the KV cache memory reduction is:

MKV(X′) MKV(X)

2Lmd 2Lnd

- m

- n

, (8)

=

=

where factor 2 accounts for both key and value states per layer.

These benefits are particularly crucial for realtime interactive systems, including UI agents (Tang

- et al., 2025b,a), autonomous driving (Gao et al., 2021), and embodied AI (Duan et al., 2022; Yang
- et al., 2025c), where efficient processing of continuous inputs under resource constraints is essential.

##### 3.3 Compelling Advantages - Why Data-centric Compression Matters?

Based on comprehensive analysis of data-centric compression, we identify five compelling advantages that makes them particularly promising:

- 1. Universal Applicability: Token redundancy is consistent across modalities and tasks, enabling data-centric compression in diverse settings.
- 2. Dual-phase Efficiency: Data-centric compression accelerates both training and inference with minimal accuracy loss.
- 3. Architectural Compatibility: Data-centric compression is orthogonal to compression methods and integrates seamlessly with them. It is also hardware and system friendly.
- 4. Low Implementation Costs: Modern architectures like transformers support variable-length inputs, allowing data-centric compression without retraining or data overhead.
- 5. Quadratic Gains: The O(n2) complexity of self-attention ensures data-centric compression yields substantial computational savings. As AI development enters a new phase where

context length becomes the primary bottleneck, the research focus of AI efficiency should shift towards data-centric compression, enabling more efficient and scalable AI systems.

4 Current Challenges

##### 4.1 Performance Degradation

Methodological Bottlenecks. Attention scores are central to most data-centric compression approaches. For example, [CLS] token attention scores are used to select key visual tokens (Haurum

Attention Knorm Vnorm Similarity Random

FastV SparseVLM PDrop Random

H2O SnapKV Knorm Random

FastV SparseVLM Pooling Random

90

65

60

100

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

80

Performance(%)

Performance(%)

Performance(%)

Performance(%)

60

70

55

98

60

55

50

50

96

50

40

30

45

94

45

20

10

40

40

92

MATH AIME24 GSM8K

GQA MMB MMB-CN

MVBench MLVU VideoMME

Image Reward

(a) Complex Reasoning

(b) Image Understanding

(c) Video Understanding

(d) Image Generation

- Figure 3: Empirical comparison of carefully designed data-centric compression methods and random token dropping. Results demonstrate that in multiple scenarios (e.g., LLMs, MLLMs, and DiTs), some carefully designed methods surprisingly underperform compared to random token selection.

- et al., 2023; Zhang et al., 2024c; Han et al., 2024b; Yang et al., 2024a; Liu et al., 2025b), while crossmodal guidance (Chen et al., 2024c; Xing et al.,

merging them disrupts this continuity, leading to fragmented recognition or translation. Similarly, cross-lingual text translation may suffer significant degradation under high compression ratios.

- 2025; Zhang et al., 2025b) relies on text-vision attention scores. But are attention scores truly reliable for deciding which data to retain? Recent work (Zhang et al., 2024c; Wen et al., 2025a) reveals that attention scores can suffer from position bias. For instance, when using text-vision scores in LLMs to retain visual tokens, those near the sequence end often receive higher weights. In 2D image space, this biases retention toward the lower half or bottom-right corner. Clearly, it is unrealistic to assume the lower half of all images is universally more important. Such bias can significantly hurt compression performance. In Figure 3, this phenomenon is consistently observed across multiple tasks and models. Recent studies (Jiang et al., 2025; Wen et al., 2025b,a; Liu et al., 2025a) have also confirmed that even well-crafted attention-based methods may underperform simple random dropping. Detailed analysis is in Appendix B.

##### 4.2 Suboptimal Data Representation

Most existing data-centric compression methods fall into two categories: redundancy-based approaches that maximize information preservation between original (X) and compressed data (X′) via maxC I(X;X′), and importance-based methods that ensure predictive sufficiency through I(X′;Y) ≥ I(X;Y)−ϵ, where ϵ denotes bearable information loss. While effective for their respective objectives, we argue that these paradigms share a critical limitation: neither guarantees that the compressed data X′ forms an optimal representation for downstream modeling. The redundancybased framework, despite preserving maximal mutual information with X, often retains tokens with reconstructive but low discriminative value. The importance-based framework, on the other hand, prioritizes maintaining predictive performance with respect to the target variable Y, but often at the cost of introducing task-specific biases. By focusing solely on information relevant to a predefined label, these methods may overlook the need to maintain stable structural and semantic patterns across the sequence that could enhance generalization. Consequently, both approaches risk producing representations misaligned with the ultimate goal of effective and generalizable downstream modeling.

Inherent Limitations of Data-Centric Compression. Beyond methodological design, does datacentric compression face inherent limitations? Is it universally applicable across tasks? For MLLMs, Wen et al. shows most existing methods underperform on visual grounding tasks, with significant drops on benchmarks like RefCOCO (Yu et al., 2016). In OCR-related parsing (Yang et al., 2024b; Ouyang et al., 2024), documents with dense layouts yield highly information-rich visual tokens. Compressing these risks severe information loss and degraded performance. Beyond vision, current methods also face inherent limits in other modalities. In automatic speech recognition (ASR) and automatic speech translation (AST) (Ardila et al., 2019; Conneau et al., 2023), audio is encoded and then decoded into text using an MLLM (Abouelenin et al., 2025; Chu et al., 2024). Audio tokens are dense and temporally continuous; pruning or

##### 4.3 Fair Comparison

Rethinking FLOPs and Compression Ratios as Efficiency Metrics. Many data-centric compression methods report speedup by estimating FLOPs reductions or directly using token compression ratios. But do FLOPs or compression ratios truly reflect real acceleration? Our analysis shows that, even with similar compression ratios or FLOPs,

methods often vary significantly in runtime latency. Investigating further, we find: (i) Importance-based compression often uses attention scores (Chen

- et al., 2024b; Zhang et al., 2025b), but this can limit compatibility with efficient attention operators (e.g., Flash Attention (Dao, 2024)), contributing to the discrepancy between theoretical FLOPs and actual latency. (ii) Some methods pursue high compression via progressive compression across layers (Xing et al., 2025), adding computational overhead that offsets the gains from token reduction. Thus, we argue that runtime latency should be prioritized in evaluations, as FLOP or token count reductions do not always yield real-world speedup. Data-centric Evaluation: The Benchmarking Gap. Current data-centric compression methods are mostly evaluated on general-purpose benchmarks that are not designed to capture compression-specific challenges. Consequently, benchmarks such as ScienceQA (Lu et al., 2022) and VizWiz (Bigham et al., 2010) sometimes show improved or stable performance under compression, contradicting expectations. This suggests these benchmarks may not adequately reflect the trade-offs inherent in compression. The issue undermines the reliability of current evaluations. Benchmarks insensitive to information loss can mask real differences between methods. Moreover, limited task diversity and the absence of compression-sensitive metrics hinder understanding of method behavior in practical settings. Without dedicated benchmarks, it is unclear whether reported gains reflect true progress or artifacts of misaligned evaluation. 5 Future Works

5.1 Data-Model Centric Compression Co-Development

As AI systems continue to scale in both model complexity and context length, a promising direction for future research lies in the co-development of data-centric and model-centric compression strategies. Instead of treating these approaches independently, integrating them can yield synergistic benefits—enhancing overall efficiency while maintaining, or even improving, model performance. The most straightforward form of integration adopts a staged approach, where model-centric compression is applied first, followed by data-centric methods. For example, token compression techniques can be employed on models that have already under-

gone quantization, pruning, or distillation. More advanced approaches aim for mutual reinforcement between the two paradigms. From a data-centric perspective, analyzing the layer-wise evolution of token representations may reveal that certain layers contribute minimal changes. This insight can inform model-centric compression by identifying layers suitable for removal or more aggressive quantization. Conversely, gradient information or attention scores associated with the critical neurons retained after model pruning can also guide token selection in data-centric compression, helping to preserve only the most informative tokens.

5.2 Dedicated Benchmarks for Data-centric Compression

Given the current limitations in evaluating datacentric compression methods using generalpurpose benchmarks, we envision the development of a dedicated benchmark specifically designed to evaluate them. Such a benchmark should comprehensively span diverse domains—including natural language processing, computer vision, and multi-modal tasks—and incorporate task-specific challenges particularly relevant to token compression, such as optical character recognition (OCR) parsing (Ouyang et al., 2024; Zhang et al., 2024a) and automatic speech recognition (ASR) (Conneau et al., 2023; Park et al., 2024). Furthermore, it is essential that this benchmark jointly considers both task performance and latency, as both are critical for real-world deployment. A well-rounded benchmark would enable a more rigorous, fair, and holistic evaluation of data-centric compression methods, ultimately driving progress in this area.

#### 6 Conclusion

In this position paper, we propose repositioning AI efficiency research by advocating a shift from model-centric to data-centric compression to address long-context processing challenges. We first examine recent advances in long-context capabilities across downstream scenarios, showing that performance scaling has shifted from model size to context length, underscoring the need for datacentric compression to mitigate the overhead of growing context lengths. We then review model efficiency approaches, with emphasis on the research roadmap of data-centric compression and its potential benefits. After analyzing current challenges in this area, we outline promising future directions

to inspire innovation. Our work aims to advance AI efficiency by offering a fresh perspective and catalyzing new research.

#### 7 Limitations

In this work, we review data-centric compression methods and analyze their benefits and limitations. Due to space constraints, our analysis focuses on token overhead and compression techniques in several prominent domains (e.g., LLMs, MLLMs, and AIGC). We acknowledge that other application areas, such as computer vision, autonomous driving, embodied intelligence, and audio/speech processing, also face growing efficiency challenges and may benefit from data-centric compression. More comprehensive cross-domain review and analysis are left for our future work.

#### References

Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, and Ari S Morcos. 2023. Semdedup: Dataefficient learning at web-scale through semantic deduplication. arXiv preprint arXiv:2303.09540.

Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, and 1 others. 2025. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv preprint arXiv:2503.01743.

Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer, Reuben Morais, Lindsay Saunders, Francis M Tyers, and Gregor Weber. 2019. Common voice: A massivelymultilingual speech corpus. arXiv preprint arXiv:1912.06670.

Abdul Hameed Azeemi, Ihsan Qazi, and Agha Ali Raza. 2023. Data pruning for efficient model pruning in neural machine translation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 236–246.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, and 29 others. 2023a. Qwen technical report. arXiv preprint arXiv:2309.16609.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023b. Qwen-VL: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, and 1 others. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.

Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, and 1 others. 2010. Vizwiz: nearly real-time answers to visual questions. In Proceedings of the 23nd annual ACM symposium on User interface software and technology, pages 333–342.

Daniel Bolya, Xingyu Dai, Tianyu Dai, Yinpeng Zhou, and Vladlen Koltun. 2023a. Token merging for fast and accurate attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. 2023b. Token merging: Your ViT but faster. In Proceedings of the International Conference on Learning Representations.

Daniel Bolya and Judy Hoffman. 2023. Token merging for fast stable diffusion. In CVPRW.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. 2024. Video generation models as world simulators.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and 1 others. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, and 1 others. 2024. Pyramidkv: Dynamic kv caching compression based on pyramidal information funneling. arXiv preprint arXiv:2406.02069.

Jiaao Chen, Zichao Yang, and Diyi Yang. 2020. Mixtext: Linguistically-informed interpolation of hidden space for semi-supervised text classification. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2147–2157.

Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. 2024a. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024b. An image is worth 1/2 tokens after layer 2: Plug-andplay inference acceleration for large vision-language models. In Proceedings of the European Conference on Computer Vision.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024c. An image is worth 1/2 tokens after layer 2: Plug-andplay inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. 2024d. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198.

Hongrong Cheng, Miao Zhang, and Javen Qinfeng Shi. 2024. A survey on deep neural network pruning: Taxonomy, comparison, analysis, and recommendations. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Rohan Choudhury, Guanglei Zhu, Sihan Liu, Koichiro Niinuma, Kris Kitani, and László Jeni. 2024. Don’t look twice: Faster video transformers with run-length tokenization. In Proceedings of the Advances in Neural Information Processing Systems, volume 37, pages 28127–28149.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, and 1 others. 2024. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. 2023. Fleurs: Few-shot learning evaluation of universal representations of speech. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 798–805. IEEE.

Claude Coulombe. 2018. Text data augmentation made simple by leveraging nlp cloud apis. arXiv preprint arXiv:1812.04718.

Ian Connick Covert, Chanwoo Kim, and Su-In Lee. 2022. Learning to estimate shapley values with vision transformers. In The Eleventh International Conference on Learning Representations.

Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V Le. 2018. Autoaugment: Learning augmentation policies from data. arXiv preprint arXiv:1805.09501.

Tri Dao. 2024. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR).

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Alessio Devoto, Yu Zhao, Simone Scardapane, and Pasquale Minervini. 2024. A simple and effective l_2 norm-based strategy for kv caching compression. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 18476–18499, Miami, Florida, USA. Association for Computational Linguistics.

Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, and 1 others. 2024. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. Advances in Neural Information Processing Systems, 37:42566–42592.

Jiafei Duan, Samson Yu, Hui Li Tan, Hongyuan Zhu, and Cheston Tan. 2022. A survey of embodied ai: From simulators to research tasks. IEEE Transactions on Emerging Topics in Computational Intelligence, 6(2):230–244.

Yuchen Duan, Weiyun Wang, Zhe Chen, Xizhou Zhu, Lewei Lu, Tong Lu, Yu Qiao, Hongsheng Li, Jifeng Dai, and Wenhai Wang. 2024. Vision-rwkv: Efficient and scalable visual perception with rwkv-like architectures. arXiv preprint arXiv:2403.02308.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022. GPTQ: Accurate post-training quantization for generative pre-trained transformers. In Proceedings of the International Conference on Learning Representations.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, and 1 others. 2024. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075.

Cong Gao, Geng Wang, Weisong Shi, Zhongmin Wang, and Yanping Chen. 2021. Autonomous driving security: State of the art and challenges. IEEE Internet of Things Journal, 9(10):7572–7595.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, and 1 others. 2020. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027.

Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. 2023. Masked diffusion transformer is a strong image synthesizer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23164–23173.

Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. 2024. Model tells you

what to discard: Adaptive kv caching compression for llms. In The Twelfth International Conference on Learning Representations.

Daniel Goldstein, Fares Obeid, Eric Alcaide, Guangyu Song, and Eugene Cheah. 2024. Goldfinch: High performance rwkv/transformer hybrid with linear pre-fill and extreme kv-caching compression. arXiv preprint arXiv:2407.12077.

Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. 2021. Knowledge distillation: A survey. International Journal of Computer Vision, 129(6):1789–1819.

Saurabh Goyal, Anamitra Roy Choudhury, Saurabh Raje, Venkatesan Chakaravarthy, Yogish Sabharwal, and Ashish Verma. 2020. Power-bert: Accelerating bert inference via progressive word-vector elimination. In International Conference on Machine Learning, pages 3690–3699. PMLR.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.

Albert Gu, Karan Goel, and Christopher Ré. 2021. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025a. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, and 1 others. 2025b. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062.

Shanshan Han, Qifan Zhang, Yuhang Yao, Weizhao Jin, Zhaozhuo Xu, and Chaoyang He. 2024a. Llm multiagent systems: Challenges and open problems. arXiv preprint arXiv:2402.03578.

Song Han, Huizi Mao, and William J. Dally. 2016. Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding. In Proceedings of the International Conference on Learning Representations.

Yuhang Han, Xuyang Liu, Pengxiang Ding, Donglin Wang, Honggang Chen, Qingsen Yan, and Siteng Huang. 2024b. Rethinking token reduction in mllms: Towards a unified paradigm for training-free acceleration. arXiv preprint arXiv:2411.17686.

Xintong Hao, Ke Shen, and Chenggang Li. 2025. Maga: Massive genre-audience reformulation to pretraining corpus expansion. arXiv preprint arXiv:2502.04235.

Joakim Bruslund Haurum, Sergio Escalera, Graham W Taylor, and Thomas B Moeslund. 2023. Which tokens to use? investigating token reduction in vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 773–783.

Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean.

2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

Junzhou Huang, Tong Zhang, and Dimitris Metaxas.

2011. Learning with structured sparsity. Journal of Machine Learning Research, 12(103):3371–3412.

Drew A Hudson and Christopher D Manning. 2019. GQA: A new dataset for real-world visual reasoning and compositional question answering. Conference on Computer Vision and Pattern Recognition (CVPR).

Yerlan Idelbayev and Miguel A Carreira-Perpinán. 2020. Low-rank compression of neural nets: Learning the rank of each layer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8049–8059.

Eric Jang, Shixiang Gu, and Ben Poole. 2017. Categorical reparameterization with gumbel-softmax. In Proceedings of the International Conference on Learning Representations.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. Llmlingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13358–13376.

Yutao Jiang, Qiong Wu, Wenhao Lin, Wei Yu, and Yiyi Zhou. 2025. What kind of visual tokens do we need? training-free visual token pruning for multi-modal large language models from the perspective of graph. arXiv preprint arXiv:2501.02268.

Hengrui Kang, Siwei Wen, Zichen Wen, Junyan Ye, Weijia Li, Peilin Feng, Baichuan Zhou, Bin Wang, Dahua Lin, Linfeng Zhang, and 1 others. 2025. Legion: Learning to ground and explain for synthetic image detection. arXiv preprint arXiv:2503.15264.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR.

Gyuwan Kim and Kyunghyun Cho. 2021. Lengthadaptive transformer: Train once with length drop, use anytime with search. In Proceedings of the 59th Annual Meeting of the Association for Computational

Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6501–6511.

Sehoon Kim, Sheng Shen, David Thorsley, Amir Gholami, Woosuk Kwon, Joseph Hassoun, and Kurt Keutzer. 2022. Learned token pruning for transformers. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 784–794.

Humaira Kousar, Hasnain Irshad Bhatti, and Jaekyun Moon. 2025. Pruning-based data selection and network fusion for efficient deep learning. arXiv preprint arXiv:2501.01118.

Black Forest Labs. 2024. Flux. https://github.com/ black-forest-labs/flux.

Dong Hoon Lee and Seunghoon Hong. 2024. Learning to merge tokens via decoupled embedding for efficient vision transformers. In Proceedings of the Advances in Neural Information Processing Systems, volume 37, pages 54079–54104.

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. 2022a. Deduplicating training data makes language models better. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8424–8445.

Saehyung Lee, Sanghyuk Chun, Sangwon Jung, Sangdoo Yun, and Sungroh Yoon. 2022b. Dataset condensation with contrastive signals. In International Conference on Machine Learning, pages 12352–12364. PMLR.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024a. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. 2017. Pruning filters for efficient convnets. In International Conference on Learning Representations (ICLR).

Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Yitzhak Gadre, Hritik Bansal, Etash Guha, Sedrick Scott Keh, Kushal Arora, and 1 others. 2024b. Datacomp-lm: In search of the next generation of training sets for language models. Advances in Neural Information Processing Systems, 37:14200–14282.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, and 1 others. 2024c. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206.

Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. 2024d. Tokenpacker: Efficient visual projector for multimodal llm. arXiv preprint arXiv:2407.02392.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024e. Snapkv: Llm knows what you are looking for before generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, and 1 others. 2024f. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748.

Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. 2022. Not all patches are what you need: Expediting vision transformers via token reorganizations. In Proceedings of the International Conference on Learning Representations.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. arXiv preprint arXiv:2305.20050.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, WeiMing Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024a. AWQ: Activation-aware weight quantization for ondevice LLM compression and acceleration. In Proceedings of the Annual Conference on Machine Learning and Systems, pages 87–100.

Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, Yelong Shen, Ruochen Xu, Chen Lin, Yujiu Yang, Jian Jiao, Nan Duan, and 1 others. 2024b. Rho-1: Not all tokens are what you need. arXiv preprint

- arXiv:2404.07965.

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, and 1 others. 2024a. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint

- arXiv:2405.04434.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, and 1 others. 2024b. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024c. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26286–26296.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In Proceedings of the Advances in Neural Information Processing Systems.

Ting Liu, Xuyang Liu, Siteng Huang, Liangtao Shi, Zunnan Xu, Yi Xin, Quanjun Yin, and Xiaohong Liu. 2024d. Sparse-tuning: Adapting vision transformers with efficient fine-tuning and inference. arXiv preprint arXiv:2405.14700.

Xuyang Liu, Yiyu Wang, Junpeng Ma, and Linfeng Zhang. 2025a. Video compression commander: Plugand-play inference acceleration for video large language models. arXiv preprint arXiv:2505.14454.

Xuyang Liu, Ziming Wang, Yuhang Han, Yingyao Wang, Jiale Yuan, Jun Song, Bo Zheng, Linfeng Zhang, Siteng Huang, and Honggang Chen. 2025b. Global compression commander: Plug-and-play inference acceleration for high-resolution large visionlanguage models. arXiv preprint arXiv:2501.05179.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, and 1 others. 2025c. Mmbench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pages 216–233. Springer.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. 2021. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell. 2019. Rethinking the value of network pruning. In Proceedings of the International Conference on Learning Representations.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521.

Tanvir Mahmud, Burhaneddin Yaman, Chun-Hao Liu, and Diana Marculescu. 2024. Papr: Training-free one-step patch pruning with lightweight convnets for faster inference. In Proceedings of the European Conference on Computer Vision, pages 110– 128. Springer.

Maxwell-Jia. 2024. Aime2024. https: //huggingface.co/datasets/Maxwell-Jia/ AIME_2024.

Meta. 2025. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. https://ai.meta.com/blog/ llama-4-multimodal-intelligence/.

OpenAI. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.

Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, and 1 others. 2024. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. arXiv preprint arXiv:2412.07626.

Bowen Pan, Rameswar Panda, Yifan Jiang, Zhangyang Wang, Rogerio Feris, and Aude Oliva. 2021. Ia-red2: Interpretability-aware redundancy reduction for vision transformers. In Proceedings of the Advances in Neural Information Processing Systems, volume 34, pages 24898–24911.

Se Jin Park, Julian Salazar, Aren Jansen, Keisuke Kinoshita, Yong Man Ro, and RJ Skerry-Ryan. 2024. Long-form speech generation with spoken language models. arXiv preprint arXiv:2412.18603.

Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho. 2019. Relational knowledge distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3967–3976.

William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205.

Guilherme Penedo, Hynek Kydlíˇcek, Anton Lozhkov, Margaret Mitchell, Colin A Raffel, Leandro Von Werra, Thomas Wolf, and 1 others. 2024. The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 37:30811–30849.

Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023. The refinedweb dataset for falcon llm: Outperforming curated corpora with web data only. Advances in Neural Information Processing Systems, 36:79155–79172.

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, and 1 others. 2023. Rwkv: Reinventing rnns for the transformer era. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 14048– 14077.

Minghao Qin, Xiangrui Liu, Zhengyang Liang, Yan Shu, Huaying Yuan, Juenjie Zhou, Shitao Xiao, Bo Zhao, and Zheng Liu. 2025. Video-xl-2: Towards very longvideo understanding through task-aware kv sparsification. arXiv preprint arXiv:2506.19225.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67.

Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. 2021. DynamicViT: Efficient vision transformers with dynamic token sparsification. In Proceedings of the Advances in Neural Information Processing Systems, pages 13937–13949.

Babak Rokh, Ali Azarpeyvand, and Alireza Khanteymoori. 2023. A comprehensive survey on model quantization for deep neural networks in image classification. ACM Transactions on Intelligent Systems and Technology, 14(6):1–50.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695.

Noveen Sachdeva, Benjamin Coleman, Wang-Cheng Kang, Jianmo Ni, Lichan Hong, Ed H Chi, James Caverlee, Julian McAuley, and Derek Zhiyuan Cheng. 2024. arXiv preprint arXiv:2402.09668.

Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. 2024. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, and 1 others. 2025. Vlm-r1: A stable and generalizable r1style large vision-language model. arXiv preprint arXiv:2504.07615.

Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, and Hongsheng Li. 2021. Efficient attention: Attention with linear complexities. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 3531–3539.

Fei Tang, Zhangxuan Gu, Zhengxi Lu, Xuyang Liu, Shuheng Shen, Changhua Meng, Wen Wang, Wenqi Zhang, Yongliang Shen, Weiming Lu, and 1 others. 2025a. Gui-g2: Gaussian reward modeling for gui grounding. arXiv preprint arXiv:2507.15846.

Fei Tang, Haolei Xu, Hang Zhang, Siqi Chen, Xingyu Wu, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Zeqi Tan, Yuchen Yan, and 1 others. 2025b. A survey on (m) llm-based gui agents. arXiv preprint arXiv:2504.13865.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, and Aidan N Gomez. 2017. Attention is all you need. In Proceedings of the Advances in Neural Information Processing Systems, volume 30, pages 5998–6008.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, and 1 others. 2025a. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314.

Zhongwei Wan, Hui Shen, Xin Wang, Che Liu, Zheda Mai, and Mi Zhang. 2025b. Meda: Dynamic kv cache allocation for efficient multimodal longcontext inference. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2485–2497.

Zhongwei Wan, Ziang Wu, Che Liu, Jinfa Huang, Zhihong Zhu, Peng Jin, Longyue Wang, and Li Yuan. 2024. Look-m: Look-once optimization in kv cache for efficient multimodal long-context inference. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 4065–4078.

Kuan Wang, Zhijian Liu, Yujun Lin, Ji Lin, and Song Han. 2019. Haq: Hardware-aware automated quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8612–8620.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, and 1 others. 2024a. Qwen2vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Shaobo Wang, Hongxuan Tang, Mingyang Wang, Hongrui Zhang, Xuyang Liu, Weiya Li, Xuming Hu, and Linfeng Zhang. 2025a. Gnothi seauton: Empowering faithful self-interpretability in black-box transformers. In The Thirteenth International Conference on Learning Representations.

Shaobo Wang, Jiaming Wang, Jiajun Zhang, Cong Wang, Yue Min, Zichen Wen, Fei Huang, Huiqiang Jiang, Junyang Lin, Dayiheng Liu, and Linfeng Zhang. 2025b. Winning the pruning gamble: A unified approach to joint sample and token pruning for efficient supervised fine-tuning. Preprint, arXiv:2509.23873.

Shaobo Wang, Yantai Yang, Qilong Wang, Kaixin Li, Linfeng Zhang, and Junchi Yan. 2025c. Not all samples should be utilized equally: Towards understanding and improving dataset distillation. In Synthetic Data for Computer Vision Workshop @ CVPR 2025.

Shaobo Wang, Yantai Yang, Shuaiyu Zhang, Chenghao Sun, Weiya Li, Xuming Hu, and Linfeng Zhang. 2024b. Drupi: Dataset reduction using privileged information. arXiv preprint arXiv:2410.01611.

Shaobo Wang, Yicun Yang, Zhiyuan Liu, Chenghao Sun, Xuming Hu, Conghui He, and Linfeng Zhang. 2025d. Dataset distillation with neural characteristic function: A minmax perspective. In Proceedings of the IEEE conference on computer vision and pattern recognition.

Zheng Wang, Boxiao Jin, Zhongzhi Yu, and Minjia Zhang. 2024c. Model tells you where to merge: Adaptive kv caching merging for llms on longcontext tasks. arXiv preprint arXiv:2407.08454.

Jason Wei and Kai Zou. 2019. Eda: Easy data augmentation techniques for boosting performance on text classification tasks. arXiv preprint arXiv:1901.11196.

Zichen Wen, Yifeng Gao, Weijia Li, Conghui He, and Linfeng Zhang. 2025a. Token pruning in multimodal large language models: Are we solving the right problem? arXiv preprint arXiv:2502.11501.

Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. 2025b. Stop looking for important tokens in multimodal language models: Duplication matters more. arXiv preprint arXiv:2502.11494.

Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave. 2019. Ccnet: Extracting high quality monolingual datasets from web crawl data. arXiv preprint arXiv:1911.00359.

Alexander Wettig, Aatmik Gupta, Saumya Malik, and Danqi Chen. 2024. Qurating: Selecting high-quality data for training language models. arXiv preprint arXiv:2402.09739.

Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. 2024. Longvideobench: A benchmark for longcontext interleaved video-language understanding. In Proceedings of the Advances in Neural Information Processing Systems, volume 37, pages 28828–28857.

Jiayang Wu, Wensheng Gan, Zefeng Chen, Shicheng Wan, and Philip S Yu. 2023. Multimodal large language models: A survey. In 2023 IEEE International Conference on Big Data (BigData), pages 2247–2256. IEEE.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations.

Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. 2021. Segformer: Simple and efficient design for semantic segmentation with transformers. In Proceedings of the Advances in Neural Information Processing Systems, volume 34, pages 12077–12090.

Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, and 1 others. 2025. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025a. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Biao Yang, Bin Wen, Boyang Ding, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, and 1 others. 2025b. Kwai keye-vl 1.5 technical report. arXiv preprint arXiv:2509.01563.

Jiwei Yang, Xu Shen, Jun Xing, Xinmei Tian, Houqiang Li, Bing Deng, Jianqiang Huang, and Xian-sheng Hua. 2019. Quantization networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7308–7316.

Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. 2024a. Visionzip: Longer is better but not necessary in vision language models. arXiv preprint arXiv:2412.04467.

Yantai Yang, Yuhao Wang, Zichen Wen, Luo Zhongwei, Chang Zou, Zhipeng Zhang, Chuan Wen, and Linfeng Zhang. 2025c. Efficientvla: Training-free acceleration and compression for vision-languageaction models. arXiv preprint arXiv:2506.10100.

Zhibo Yang, Jun Tang, Zhaohai Li, Pengfei Wang, Jianqiang Wan, Humen Zhong, Xuejing Liu, Mingkun Yang, Peng Wang, Yuliang Liu, and 1 others. 2024b. Cc-ocr: A comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy. arXiv preprint arXiv:2412.02210.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, and 1 others. 2024c. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072.

Haoran You, Connelly Barnes, Yuqian Zhou, Yan Kang, Zhenbang Du, Wei Zhou, Lingzhi Zhang, Yotam Nitzan, Xiaoyang Liu, Zhe Lin, and 1 others. 2024. Layer-and timestep-adaptive differentiable token compression ratios for efficient diffusion transformers. arXiv preprint arXiv:2412.16822.

Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. 2016. Modeling context in referring expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer.

Xiyu Yu, Tongliang Liu, Xinchao Wang, and Dacheng Tao. 2017. On compressing deep models by low rank and sparse decomposition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7370–7379.

Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. 2019. Cutmix: Regularization strategy to train strong

classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6023–6032.

Evelyn Zhang, Jiayi Tang, Xuefei Ning, and Linfeng Zhang. 2025a. Training-free and hardware-friendly acceleration for diffusion models via similarity-based token pruning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 9878–9886.

Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. 2018. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations.

Junyuan Zhang, Qintong Zhang, Bin Wang, Linke Ouyang, Zichen Wen, Ying Li, Ka-Ho Chow, Conghui He, and Wentao Zhang. 2024a. Ocr hinders rag: Evaluating the cascading impact of ocr on retrieval-augmented generation. arXiv preprint arXiv:2412.02592.

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and 1 others. 2024b. Lmms-eval: Reality check on the evaluation of large multimodal models. arXiv preprint arXiv:2407.12772.

Linfeng Zhang, Chenglong Bao, and Kaisheng Ma. 2021. Self-distillation: Towards efficient and compact neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8):4388– 4403.

Linfeng Zhang, Jiebo Song, Anni Gao, Jingwei Chen, Chenglong Bao, and Kaisheng Ma. 2019. Be your own teacher: Improve the performance of convolutional neural networks via self distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3713–3722.

Qizhe Zhang, Aosong Cheng, Ming Lu, Zhiyong Zhuo, Minqi Wang, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. 2024c. [cls] attention is all you need for training-free visual token pruning: Make vlm inference faster. arXiv preprint arXiv:2412.01818.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and 1 others. 2025b. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In Proceedings of the International Conference on Machine Learning.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, Zhangyang Wang, and Beidi Chen. 2023. H2o: Heavy-hitter oracle for efficient generative inference of large language models. In Thirty-seventh Conference on Neural Information Processing Systems.

Bo Zhao and Hakan Bilen. 2021. Dataset condensation with differentiable siamese augmentation. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 12674–12685. PMLR.

Wangbo Zhao, Yizeng Han, Jiasheng Tang, Zhikai Li, Yibing Song, Kai Wang, Zhangyang Wang, and Yang You. 2024a. A stitch in time saves nine: Small vlm is a precise guidance for accelerating large vlms. arXiv preprint arXiv:2412.03324.

Wangbo Zhao, Jiasheng Tang, Yizeng Han, Yibing Song, Kai Wang, Gao Huang, Fan Wang, and Yang You. 2024b. Dynamic tuning towards parameter and inference efficiency for vit adaptation. In Proceedings of the Advances in Neural Information Processing Systems.

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. 2024. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264.

Yiren Zhou, Seyed-Mohsen Moosavi-Dezfooli, NgaiMan Cheung, and Pascal Frossard. 2018. Adaptive quantization for deep neural network. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, and 1 others. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. In Proceedings of the International Conference on Machine Learning.

Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. 2025. Accelerating diffusion transformers with token-wise feature caching. In Proceedings of the International Conference on Learning Representations.

In the appendix, we provide additional statistical analysis of token overhead in Section A, details of the empirical experiments in Figure 3 in Section B, and analysis of alternative positions in Section C.

#### A Trends in LLM Scaling: Parameters vs. Context Length

In this section, we provide a comprehensive analysis of the temporal progression of mainstream LLMs, documenting the growth trends in both parameter counts and context lengths. This analysis provides empirical support for our central thesis regarding the shift in computational bottlenecks from model parameters to context processing. As shown in Tables below, both in text and vision domains, model size growth has significantly slowed, while context length continues to increase. This trend indicates that the focus of research for efficient AI is shifting from model-centric compression to data-centric compression.

#### B Comparison of Token Compression Methods and Random Token Dropping

This section presents a detailed comparison between designed token compression methods and random token dropping, the simplest baseline. The analysis supports arguments in Section 4.1, showing that current token compression techniques have performance limitations. Experiments span multiple domains: complex reasoning in language, image and video understanding in vision, and textto-image generation in AI content creation. This broad evaluation assesses token compression effectiveness across diverse tasks and modalities. Results highlight the need for more robust approaches in LLMs, MLLMs, VideoLLMs, and DiTs, emphasizing the importance of universally applicable compression strategies.

LLMs: Complex Reasoning We evaluated DeepSeek-R1-Distill-Llama-8B (Guo et al., 2025a) on complex reasoning tasks: MATH-500 (Lightman et al., 2023), AIME24 (Maxwell-Jia, 2024), and GSM8K (Cobbe et al., 2021). During decoding, we enforced a fixed KV cache budget (i.e., 1024 tokens) and applied KV cache evition methods: H2O (Zhang et al., 2023), SnapKV (Li et al., 2024e), KNorm (Devoto et al., 2024), and random eviction at regular intervals (i.e., every 512 tokens).

Figure 3 (a) shows a counterintuitive result: H2O, SnapKV, and KNorm consistently underperform random dropping on math reasoning tasks.

On AIME24, random dropping surpasses SnapKV by 10% accuracy. These findings yield two insights: (i) Random dropping should be a standard baseline in KV cache studies, as it is often overlooked despite strong performance; (ii) Its effectiveness may stem from preserving token distribution uniformity during auto-regressive decoding, better maintaining semantic coherence than deterministic strategies. This challenges the assumption that complex policies are inherently superior and reveals gaps in current token importance modeling.

MLLMs: Image Understanding We tested LLaVA-1.5-7B (Liu et al., 2023) on image benchmarks including GQA (Hudson and Manning, 2019) and MMB (Liu et al., 2025c). We retained 25% of visual tokens, following official LLaVA evaluation scripts1. We compared FastV (Chen et al., 2024b), SparseVLM (Zhang et al., 2025b), random dropping, and pooling.

Figure 3 (b) shows that random dropping and pooling outperform some designed methods. We attribute this to their shared property of spatial uniformity, which mitigates position bias (Sec. 4) in attention-based methods like FastV. This also highlights the negative impact of position bias in attention scores. We thus advocate for spatial uniformity as a key design principle in token compression.

VideoLLMs: Video Understanding We evaluated LLaVA-OneVision-7B (Li et al., 2024a) on video benchmarks: MVBench (Li et al., 2024c), LongVideoBench (Wu et al., 2024), MLVU (Zhou et al., 2024), and VideoMME (Fu et al., 2024). We compared FastV (Chen et al., 2024b), SparseVLM (Zhang et al., 2025b), and PDrop (Xing et al., 2025) with 15% visual tokens, using the LMMsEval framework (Zhang et al., 2024b)2.

Figure 3 (c) shows that even with only 15% tokens, random dropping outperforms designed methods. This implies: (i) Random dropping must be included as a baseline; (ii) VideoLLM compression should prioritize uniform spatial and temporal token distribution for comprehensive video representation. We hypothesize that random dropping succeeds by inherently maintaining this uniformity.

DiTs: Image Generation We tested the DiTbased model FLUX.1-dev (Labs, 2024) with the ToCa (Zou et al., 2025) method, setting cache cy-

- 1https://github.com/haotian-liu/LLaVA
- 2https://github.com/LLaVA-VL/LLaVA-NeXT/blob/

main/docs/LLaVA_OneVision.md

cle length N = 4 and ratio R = 90% (10% tokens computed per step). We compared attention-based, Key Norm (Knorm), Value Norm (Vnorm), and random selection. Surprisingly, all characteristicbased strategies yielded lower Image Reward scores than random selection.

Figure 3 (d) confirms random selection performs well in image generation. To investigate, we designed a similarity-based strategy: select 1% tokens randomly as base, then choose 9% most similar to them. This led to clustered, homogeneous tokens and the worst generation quality. This suggests redundancy among similar tokens, while random selection benefits from diversity, enabling richer information representation.

#### C Alternative Positions

While this paper promotes data-centric compression as a key strategy for advancing Efficient AI, it is equally important to recognize and engage with alternative viewpoints that challenge the feasibility, necessity, or overall effectiveness of this approach.

C.1 Model-Centric Compression as a Superior Alternative

Model-centric compression methods, such as pruning (Liu et al., 2019; Huang et al., 2011; Han et al., 2016; Rao et al., 2021; Jang et al., 2017; Pan et al., 2021), quantization (Rokh et al., 2023; Wang et al., 2019; Zhou et al., 2018; Yang et al., 2019; Lin et al., 2024a; Frantar et al., 2022), and knowledge distillation (Hinton et al., 2015; Zhang et al., 2019, 2021; Park et al., 2019), have long been established as effective techniques for reducing model size and computational cost. Proponents argue that this paradigm is reliable for deployment in resource-constrained environments and maintains performance consistency. For example, pruning techniques such as DynamicViT (Rao et al., 2021) dynamically remove uninformative tokens during inference, reducing the computational load by up to 30–40% with minimal impact on accuracy. Proponents of this view claim that this approach achieves substantial speedups without discarding any original data. In contrast, data-centric methods that prune input tokens risk removing critical contextual information, which may degrade performance. Counterargument. Although model-centric compression is effective, it faces scalability issues as models and datasets grow, requiring costly full retraining and processing of entire inputs. In contrast,

data-centric compression reduces input complexity upfront, easing computational burdens. Some data-centric methods update only a small parameter subset (Zhao et al., 2024b; Liu et al., 2024d; Lee and Hong, 2024), while others enable trainingfree deployment (Bolya et al., 2023a; Chen et al., 2024b; Li et al., 2024e). Combining both approaches can improve efficiency without sacrificing accuracy (Azeemi et al., 2023; Kousar et al., 2025), making data-centric methods a complement to model-centric techniques.

C.2 Advanced Model Architectures as a more Promising Direction

Another argument against data-centric compression is the continued advancement of model architectures that can inherently handle large datasets and long sequences more efficiently (Gu et al., 2021; Goldstein et al., 2024; Peng et al., 2023). The development of transformer-based architectures, such as Vision Transformers (Covert et al., 2022), Swin Transformers (Liu et al., 2021), and large language models like GPT-3 (Brown et al., 2020), has shown significant improvements in both accuracy and scalability. These architectures integrate advanced techniques, such as hierarchical processing, self-attention mechanisms, and dynamic sparsity, enabling them to process large amounts of data efficiently. For example, Swin Transformers (Liu et al., 2021) utilize a window-based self-attention mechanism, which reduces the computational complexity of the standard attention mechanism, making it feasible to scale models to much larger datasets and sequences. Proponents of this view argue that as these advanced models continue to evolve, there may be less need for aggressive input compression, as these models are inherently better equipped to handle large-scale data directly.

Counterargument. Advanced model architectures offer strong performance but demand substantial computational resources, especially during training (Meta, 2025; Yang et al., 2025a; OpenAI, 2023). Data-centric compression reduces computational load early by simplifying input data, enabling more efficient training and inference without sacrificing accuracy. Techniques like token pruning and augmentation preserve or improve performance by focusing on informative data. Combined with advanced architectures, data-centric methods enhance efficiency and maintain high performance (Gao et al., 2020; Wettig et al., 2024), making them complementary rather than competitive.

###### Model Name Release Date Parameters Maximum Context Length Model Link

Qwen-1.8B Nov 30, 2023 1.8B 32K link Qwen-7B Aug 3, 2023 7B 2K (Original), 8K (Updated) link Qwen-14B Sep 25, 2023 14B 8K link Qwen-72B Nov 30, 2023 72B 32K link

- Qwen1.5-0.5B Early 2024 0.5B 32K link
- Qwen1.5-1.8B Early 2024 1.8B 32K link

- Qwen1.5-4B Early 2024 4B 32K link

- Qwen1.5-7B Early 2024 7B 32K link

- Qwen1.5-14B Early 2024 14B 32K link

- Qwen1.5-32B Early 2024 32B 32K link

- Qwen1.5-72B Early 2024 72B 32K link

- Qwen1.5-110B Early 2024 110B 32K link

- Qwen1.5-MoE-A2.7B Mar 28, 2024 14B 32K link

- Qwen2-0.5B Jun 6, 2024 0.5B 32K link
- Qwen2-1.5B Jun 6, 2024 1.5B 32K link

- Qwen2-7B Jun 6, 2024 7B 32K (Base), 131K (Instruct) link

- Qwen2-57B-A14B Jun 6, 2024 57B 32K (Base), 64K (Instruct) link

- Qwen2-72B Jun 6, 2024 72B 32K (Base), 131K (Instruct) link

Qwen2.5-0.5B Sep 19, 2024 0.5B 32K link Qwen2.5-1.5B Sep 19, 2024 1.5B 32K link Qwen2.5-3B Sep 19, 2024 3B 32K link Qwen2.5-7B Sep 19, 2024 7B 128K link Qwen2.5-14B Sep 19, 2024 14B 128K link Qwen2.5-32B Sep 19, 2024 32B 128K link Qwen2.5-72B Sep 19, 2024 72B 128K link Qwen2.5-7B-Instruct-1M Jan 2025 7B 1M link Qwen2.5-14B-Instruct-1M Jan 2025 14B 1M link

- Qwen3-0.6B Apr 29, 2025 0.6B 32K link

- Qwen3-1.7B Apr 29, 2025 1.7B 32K link

- Qwen3-4B Apr 29, 2025 4B 32K link Qwen3-8B Apr 29, 2025 8B 131K link Qwen3-14B Apr 29, 2025 14B 131K link Qwen3-32B Apr 29, 2025 32B 131K link Qwen3-30B-A3B Apr 29, 2025 30B 131K link Qwen3-235B-A22B Apr 29, 2025 235B 131K link

- Table 1: Qwen series model specifications. Details include release dates, parameter counts, maximum context lengths, and Hugging Face links.

Model Name Release Date Parameters Context Length Model Link

DeepSeek-Coder November 2, 2023 1.3B/6.7B/33B 16K tokens link DeepSeek-LLM November 29, 2023 7B 4096 tokens link DeepSeek-LLM November 29, 2023 67B 4096 tokens link DeepSeekMoE January 11, 2024 16B total, 2.7B activated 4096 tokens link DeepSeek-Math April 2024 7B 4096 tokens link DeepSeek-V2 May 6, 2024 236B total, 21B activated 128K tokens link DeepSeek-V2-Lite May 16, 2024 16B total, 2.4B activated 32K tokens link DeepSeek-Coder-V2 June 17, 2024 236B total, 21B activated 128K tokens link DeepSeek-Coder-V2-Lite June 17, 2024 16B total, 2.4B activated 128K tokens link DeepSeek-V2.5 September 2024 236B total, 21B activated 128K tokens link DeepSeek-V3 December 26, 2024 671B total, 37B activated 128K tokens link DeepSeek-R1-Zero January 20, 2025 671B total, 37B activated 128K tokens link DeepSeek-R1 January 20, 2025 671B total, 37B activated 128K tokens link DeepSeek-R1-Distill January 20, 2025 1.5B, 7B, 8B, 14B, 32B, 70B 32K tokens link DeepSeek-V3-0324 March 2025 671B total, 37B activated 128K tokens link

- Table 2: DeepSeek series model specifications. Details include release dates, parameter counts, and maximum context lengths.

Model Name Release Date Parameters Context Length Model Link Llama 1 7B February 24, 2023 7B 2,048 tokens link

- Llama 1 13B February 24, 2023 13B 2,048 tokens link

- Llama 1 33B February 24, 2023 33B 2,048 tokens link

- Llama 1 65B February 24, 2023 65B 2,048 tokens link

- Llama 2 7B July 18, 2023 7B 4,096 tokens link

- Llama 2 13B July 18, 2023 13B 4,096 tokens link

Llama 2 70B July 18, 2023 70B 4,096 tokens link Llama 3 8B April 18, 2024 8B 8,192 tokens link

- Llama 3 70B April 18, 2024 70B 8,192 tokens link Llama 3.1 8B July 23, 2024 8B 128,000 tokens link Llama 3.1 70B July 23, 2024 70B 128,000 tokens link Llama 3.1 405B July 23, 2024 405B 128,000 tokens link

- Llama 4 Scout April 5, 2025 109B total / 17B active 10M tokens link

- Llama 4 Maverick April 5, 2025 400B total / 17B active 1M tokens link

- Table 3: Llama series model specifications. Details include release date, parameter count, context length, and Hugging Face model link.

Model Name Release Date Parameters Context Length Model Link

GLM-130B August 2022 130B 2,048 tokens link ChatGLM-6B March 14, 2023 6.2B 2,048 tokens link ChatGLM2-6B June 25, 2023 6.2B 32,768 tokens link ChatGLM2-6B-32K July 2023 6.2B 32,768 tokens link ChatGLM3-6B October 2023 6.2B 8,192 tokens link ChatGLM3-6B-32K October 2023 6.2B 32,768 tokens link ChatGLM3-6B-128K November 2023 6.2B 131,072 tokens link GLM-4-9B May 2024 9B 8,192 tokens link GLM-4-9B-Chat May 2024 9B 131,072 tokens link GLM-4-9B-Chat-1M May 2024 9B 1,048,576 tokens link

- Table 4: GLM series model specifications. Details include release dates, parameter counts, maximum context lengths, and Hugging Face links.

Model Name Release Date Parameters Context Length Model Link

InternLM-7B July 2023 7B 8,000 tokens link InternLM-7B-Chat v1.1 August 22, 2023 7B 8,000 tokens link InternLM-20B September 20, 2023 20B 16,000 tokens link InternLM-20B-Chat September 20, 2023 20B 16,000 tokens link InternLM2-7B January 17, 2024 7B 200,000 tokens link InternLM2-20B January 17, 2024 20B 200,000 tokens link InternLM2.5-7B July 3, 2024 7B 200,000 tokens link InternLM2.5-7B-Chat-1M July 2024 7B 1,000,000 tokens link InternLM2.5-1.8B August 1, 2024 1.8B 200,000 tokens link InternLM2.5-20B August 1, 2024 20B 200,000 tokens link InternLM3-8B-Instruct January 15, 2025 8B 32,768 tokens link

- Table 5: InternLM series model specifications. Details include release dates, parameter counts, maximum context lengths, and Hugging Face links.

Model Name Release Date LLM Backbone Max Context Image Resolution Max Tokens Model Link

LLaVA-7B April 2023 Vicuna-7B 2K 224×224 256 link LLaVA-13B April 2023 Vicuna-13B 2K 224×224 256 link LLaVA-1.5-7B October 2023 Vicuna-7B-v1.5 4K 336×336 576 link LLaVA-1.5-13B October 2023 Vicuna-13B-v1.5 4K 336×336 576 link LLaVA-NeXT-7B January 2024 Mistral-7B 8K 336x{2x2,1x{2,3,4}, {2,3,4}x1} 2880 link LLaVA-NeXT-7B January 2024 Vicuna-7B-v1.5 4K 336x{2x2,1x{2,3,4}, {2,3,4}x1} 2880 link LLaVA-NeXT-13B January 2024 Vicuna-13B-v1.5 4K 336x{2x2,1x{2,3,4}, {2,3,4}x1} 2880 link LLaVA-NeXT-34B January 2024Nous-Hermes-2-Yi-34B 4K 336x{2x2,1x{2,3,4}, {2,3,4}x1} 2880 link LLaVA-OneVision-0.5B August 2024 Qwen2-0.5B 32K 336×336×[6,6] 7290(si), 8748(mi), 6272(vid) link LLaVA-OneVision-7B August 2024 Qwen2-7B 32K 336×336×[6,6] 7290(si), 8748(mi), 6272(vid) link LLaVA-OneVision-72B August 2024 Qwen2-72B 32K 336×336×[6,6] 7290(si), 8748(mi), 6272(vid) link

- Table 6: LLaVA series model specifications. Details include release dates, backbone models, context lengths, and multimodal capabilities. si: single image; mi: multiple images; vid: video.

###### Model Name Release Date LLM Backbone Max Context Image Resolution Max Tokens Model Link

InternVL-21B Dec 2023 Vicuna-7B 2K 224×224, 336×336, 448×448 1,024 link InternVL-27B Dec 2023 Vicuna-13B 2K 224×224, 336×336, 448×448 1,024 link InternVL1.5-26B Apr 2024 InternLM2-20B 200K 2688×2688 8,192 link

- InternVL2.5-1B Dec 2024 Qwen2.5-0.5B-Instruct 32K 2688×2688 8,192 link
- InternVL2.5-2B Dec 2024 Internlm2.5-1.8B-chat 200K 2688×2688 8,192 link InternVL2.5-4B Dec 2024 Qwen2.5-3B-Instruct 32K 2688×2688 8,192 link InternVL2.5-8B Dec 2024 Internlm2.5-7B-chat 200K 2688×2688 8,192 link InternVL2.5-26B Dec 2024 Internlm2.5-20B-chat 200K 2688×2688 8,192 link InternVL2.5-38B Dec 2024 Qwen2.5-32B-Instruct 128K 2688×2688 8,192 link InternVL2.5-78B Dec 2024 Qwen2.5-72B-Instruct 128K 2688×2688 8,192 link

- InternVL3-1B Apr 2025 Qwen2.5-0.5B 32K 2688×2688 32K link
- InternVL3-2B Apr 2025 Qwen2.5-1.5B 32K 2688×2688 32K link

- InternVL3-8B Apr 2025 Qwen2.5-7B 128K 2688×2688 32K link
- InternVL3-9B Apr 2025 InternLM3-8B 32K 2688×2688 32K link InternVL3-14B Apr 2025 Qwen2.5-14B 128K 2688×2688 32K link InternVL3-38B Apr 2025 Qwen2.5-32B 128K 2688×2688 32K link InternVL3-78B Apr 2025 Qwen2.5-72B 128K 2688×2688 32K link

- Table 7: InternVL series model specifications. Details include release dates, backbone architectures, and multimodal capabilities.

Model Name Release Date LLM Backbone Max Context Image Resolution Max Tokens Model Link

Qwen-VL-9.6B Aug 2023 Qwen-7B 2K 448×448 1,024 link Qwen2-VL-2B Sep 2024 Qwen2-1.5B 32K native resolution (max=2048×2048) 16,384 link Qwen2-VL-7B Sep 2024 Qwen2-7B 32K native resolution (max=2048×2048) 16,384 link Qwen2-VL-72B Sep 2024 Qwen2-72B 32K native resolution (max=2048×2048) 16,384 link Qwen2.5-VL-3B Feb 2025 Qwen2.5-3B 32K native resolution (max=2048×2048) 24,576 link Qwen2.5-VL-7B Feb 2025 Qwen2.5-7B 128K native resolution (max=2048×2048) 24,576 link Qwen2.5-VL-72B Feb 2025 Qwen2.5-72B 128K native resolution (max=2048×2048) 24,576 link

- Table 8: Qwen-VL series model specifications. Includes release dates, backbone architectures, and multimodal capabilities.

