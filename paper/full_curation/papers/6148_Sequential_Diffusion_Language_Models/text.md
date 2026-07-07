# arXiv:2509.24007v1[cs.CL]28Sep2025

## SEQUENTIAL DIFFUSION LANGUAGE MODELS

Yangzhou Liu2,1∗, Yue Cao2,1∗, Hao Li1∗, Gen Luo1∗, Zhe Chen2,1, Weiyun Wang4,1, Xiaobo Liang6, Biqing Qi1, Lijun Wu1, Changyao Tian5,1, Yanting Zhang7, Yuqiang Li1, Tong Lu2, Yu Qiao1, Jifeng Dai3,1, Wenhai Wang5,1 1Shanghai AI Laboratory, 2Nanjing University, 3Tsinghua University, 4Fudan University, 5The Chinese University of Hong Kong, 6Soochow University, 7Donghua University

ABSTRACT

Diffusion language models (DLMs) have strong theoretical efficiency but are limited by fixed-length decoding and incompatibility with key-value (KV) caches. Block diffusion mitigates these issues, yet still enforces a fixed block size and requires expensive training. We introduce Next Sequence Prediction (NSP), which unifies next-token and next-block prediction, enabling the model to adaptively determine the generation length at each step. When the length is fixed to 1, NSP reduces to standard next-token prediction. Building on NSP, we propose Sequential Diffusion Language Model (SDLM), which can retrofit pretrained autoregressive language models (ALMs) at minimal cost. Specifically, SDLM performs diffusion inference within fixed-size mask blocks, but dynamically decodes consecutive subsequences based on model confidence, thereby preserving KV-cache compatibility and improving robustness to varying uncertainty and semantics across the sequence. Experiments show that SDLM matches or surpasses strong autoregressive baselines using only 3.5M training samples, while achieving 2.1× higher throughput than Qwen-2.5. Notably, the SDLM32B model delivers even more pronounced efficiency gains, demonstrating the strong scalability potential of our modeling paradigm. Project page and codes: https://github.com/OpenGVLab/SDLM

1 INTRODUCTION

In recent years, diffusion models have made significant progress in computer vision, dominating various fields such as image generation (Ho et al., 2020; Rombach et al., 2022) and robot control (Chi et al., 2023; Kapelyukh et al., 2023). This successful paradigm has recently emerged as a potential solution for language modeling, i.e., diffusion language models (DLMs). Compared to autoregressive language models (ALMs), DLMs generate tokens in parallel through a denoising process, showing superior theoretical efficiency. However, DLMs are also criticized for its fixed decoding length and inability to use KV cache (Radford et al., 2019).

To address these limitations, it is a natural thought to combine the benefit of DLM and ALM, similar to existing efforts like Block Diffusion (Arriola et al., 2025). Specifically, Block Diffusion reformulate the next token prediction of ALM as the next block prediction, where tokens in each block are decoded in a diffusion manner. In this case, Block Diffusion not only preserve the autoregressive property for flexible and robust prediction, but also benefit from the diffuse nature in efficiency.

Despite the effectiveness, Block Diffusion models still remain two practical limitations. Firstly, the block size is fixed in block diffusion models, which means that the model should predict a constant number of tokens in each step. However, the distribution of certainty and semantics varies across the entire sequence, typically requiring adjusting the suitable block size in predicting different subsequences. As shown in Figure 1(b), a fixed block size easily fails in token prediction that requires previous context information. Secondly, both the DLM and the block diffusion model require training from scratch and cannot be easily developed from a pre-trained ALM. This not only leads to significant training costs but also creates obstacles for developing larger models.

∗Equal contribution. Corresponding author: wangwenhai@pjlab.org.cn

(a) Autoregression Language Model (b) Block Diffusion Language Model

<mask> <mask> <mask> <mask>

Success starts with every

Success starts with every single

Success <mask> with <mask>

fixed size

Success starts with every single step

Success starts with every

<mask> <mask> <mask> <mask>

[Figure 1]

(c) Sequential Diffusion Language Model

(d) Faster Decoding vs. Performance across Models

|<mask>|
|---|

Step 𝑡𝑡

Success <mask> <mask>

|the|
|---|

Success starts with single

dynamic size

|<mask>|
|---|

Step 𝑡𝑡 + 1

Success starts with <mask> <mask>

|a|
|---|

.

Success starts with single step

dynamic size

Mask Token Decoded Token

Discarded Token

Cached Token

- Figure 1: Comparison of decoding paradigms. (a) ALMs: decode one token at a time. (b) DLMs (e.g. Block Diffusion): decode all tokens in a fixed block before moving to the next. (c) SDLM (Ours): dynamically predicts a contiguous subsequence within a fixed block. (d) Performance vs. Speed: MATH-500 results showing trade-off between speed (TPS) and accuracy.

In this paper, we introduce Next Sequence Prediction (NSP), a general form of next token prediction and next block prediction. Specifically, NSP defines an autoregressive probability distribution for sequences of discrete random variables. As shown in Figure 1(c), NSP predicts future sequences of variable length, where a sequence can be either one token or a block of tokens. At each step, NSP decodes the tokens in the sequence in a diffusion manner. Therefore, NSP can dynamically adjust its decoding sequence size according to the difficulty and semantics of future sequences. When the length of the prediction sequence is always 1, NSP degenerates to next-token prediction. This property allows NSP to seamlessly adapt to existing pre-trained ALMs with cheap costs.

Based on the principle of NSP, we propose Sequential Diffusion Language Models (SDLMs) with innovative training and inference strategies. As shown in Figure 2, SDLMs are developed based on a pre-trained ALMs, employing a novel parallel block training approach to extend next token prediction to next sequence prediction. In parallel block training, we use a custom attention mask that makes the prefix and the current block visible to each prediction window, enabling parallel training over multiple future blocks. During inference, SDLM predicts a fixed-length block at each step and then dynamically decodes a continuous subsequence via a confidence scheme based on threshold or verification. Unlike our concurrent work, Samragh et al. (2025) employs gated LoRA with nexttoken and multi-token prediction losses, whereas we use NTP cross-entropy loss for full supervised fine-tuning. For sampling, we apply bidirectional attention with confidence-based decoding, without extra sampling heads.

To validate our approach, we construct different scales of LLMs and conduct extensive experiments on 13 benchmarks across general, math, knowledge and coding tasks. Experiments show that our SDLMs achieve on-par performance with existing ALMs with much faster speed, e.g., 2.1× faster than Qwen-2.5-3B (Team, 2024). Compared to existing DLMs, our SDLMs demonstrate comprehensive advantages in performance, efficiency, and training costs. For example, SDLM-3B significantly outperforms DLMs like Dream-7B (Ye et al., 2025) and LLaDA-8B (Nie et al., 2025b) across multiple benchmarks, while requiring far less training compute and yielding substantially higher inference speed. More importantly, the scalability of SDLMs is validated on the larger models, i.e., Qwen-2.5-32B, requiring only 3.5M training data. In summary, our contributions are three-folds:

- • We introduce Next Sequence Prediction (NSP) as a general form of next token prediction and next block prediction. NSP not only combines the advantage of autoregressive models and diffusion models, but also overcomes the limitations in existing block diffusion models, i.e., the fixed block size and scalability.
- • Based on NSP, we deploy Sequential Diffusion Language Models (SDLMs) through a novel parallel block training method. SDLMs employ a customized attention mask where each

- block is visible to its prefix and itself, enabling parallel training and dynamic variablelength sequence generation via threshold- or verification-based selection.
- • Extensive experiments not only demonstrate the effectiveness and efficiency against existing ALMs and DLMs, but also confirm its scalability on large-scale models. In particular, with only 3.5M training data, our SDLMs achieves comparable performance and nearly 2× speedup against Qwen-2.5-32B-SFT.

2 RELATED WORK

- 2.1 AUTOREGRESSIVE LLMS AND MULTI-TOKEN PREDICTION Autoregressive large language models (ALMs), such as GPT (Radford et al., 2018; OpenAI, 2022;

- 2024), LLaMA series (Touvron et al., 2023a;b; Grattafiori et al., 2024; Chiang et al., 2023), Qwen series (Bai et al., 2023; Yang et al., 2024; Team, 2024; Yang et al., 2025) and other advanced LLMs (DeepMind, 2025; Team et al., 2025; xAI, 2025; Anthropic, 2023; Liu et al., 2024), generate text in a token-by-token manner and have demonstrated strong performance across a wide range of language tasks, including question answering, code generation, mathematical problem solving and dialogue systems. However, this strictly sequential decoding process limits generation speed. To mitigate this, KV Cache has been introduced to store previously computed attention keys and values, avoiding redundant computations and significantly improving inference efficiency.

To address the limitations of serial decoding, multi-token prediction (MTP) (Cai et al., 2024; Gloeckle et al., 2024; Liu et al., 2024) enables the model predict multiple future tokens in parallel via multiple output heads. These parallel predictions can be used with speculative decoding (Xia et al., 2022; Stern et al., 2018) to validate multiple candidates and greatly reduce forward steps. For example, DeepSeek-V3 shows up to 3× faster inference with MTP with speculative decoding.

2.2 DIFFUSION LANGUAGE MODELS

Recent diffusion models have shown increasing potential in language tasks. Masked discrete diffusion models (MDMs) (Zheng et al., 2023; Gong et al., 2024; Ou et al., 2024; Nie et al., 2024) have achieved perplexity comparable to ALMs. LLaDA (Nie et al., 2025a) further scales MDMs to 8B parameters, matching state-of-the-art ALMs. Dream (Ye et al., 2025) adopts shifted prediction and autoregressive initializes, effectively reducing training costs while also delivering strong performance. Block Diffusion (Arriola et al., 2025) introduces block-level generation for variable-length decoding with KV cache reuse. Gemini Diffusion (Google DeepMind, 2025) and Seed Diffusion (Song et al.,

- 2025) further improve speed while narrowing the gap with ALMs.

- 3 METHODS

Although recent acceleration technologies such as dKV-Cache (Ma et al., 2025), Fast-dLLM (Wu et al., 2025), and dLLM-Cache (Liu et al., 2025) attempt to use approximate KV caching mechanisms to accelerate DLM inference, these methods still suffer from substantial computational overhead caused by padding the sequence to the maximum sequence length for each forward computation.

- 3.1 PRELIMINARY AND NOTATION

In autoregressive large language models (ALMs) (OpenAI, 2024; Grattafiori et al., 2024; Yang et al., 2025; DeepMind, 2025), text generation is typically modeled as a conditional probability chain, referred to as the next-token prediction paradigm. Given a sequence of input tokens {x1,...,xL}, the objective is to minimize the cross-entropy loss:

LALM(x;θ) = −Ex

L

log Pθ(xi | x<i) , (1)

i=1

where the model Pθ(· | x<i) aims to maximizes the conditional probability of the current word by leveraging the preceding context x<i = x0,...,xi−1.

In contrast, diffusion language models (DLMs) (Ou et al., 2024; Nie et al., 2025a; Ye et al., 2025) generate outputs by progressively denoising from a fully noisy state in parallel. Block Diffusion (Arriola et al., 2025) are a specialized DLM variant that constrains the diffusion operation to proceed

sequentially in blocks. At each time step t, the model receives a noisy block Xti = xiDt :(i+1)D and predicts all masked tokens (denoted as [m]) within a block of length D, formally defined as:

L/D

αt′ 1 − αt

log Pθ(Xi|X<i,Xti), (2)

Et∼[0,1]Eq

LBD(X;θ) = −

i=1

where X denotes the ground truth, the probability that a token is masked at time t under the noising function q is 1 − αt, and αt′ is the instantaneous rate of change of αt in continuous time.

- 3.2 SEQUENTIAL DIFFUSION LANGUAGE MODELS

In DLMs, the entire sequence is predicted in parallel based on confidence scores. This can result in premature and inaccurate predictions for later tokens, imposing greater demands on the model’s robustness. But predictions for tokens at lower position indices generally benefit from more reliable contextual information and introduce less bias (Wang et al., 2024). Meanwhile, the distribution of certainty and semantics varies across the entire sequence. To this end, we introduce the Next Sequence Prediction (NSP) paradigm, which aims to dynamically adjust the size of the decoding sequence at each step based on the difficulty and semantics of the future sequence.

Based on the above understanding, we propose the Sequential Diffusion Language Models (SDLM) to reduce error accumulation in diffusion-based generation and improve parallel prediction efficiency. As shown in Figure 1(c), the model adopts bidirectional attention similar to Block Diffusion to understand the semantic information in the future fixed-length noise block XTi . Differently, SDLM masks all tokens in the prediction block (masking probability = 1) and is trained by minimizing the cross entropy of all masked tokens. The training objective is formalized as:

L(X;θ) = −EX,X

T

1 D i

log Pθ Xi | x<(i−1),XTi ,

Xi = xi:(i+D), XTi = [xi−1,[m],...,[m]

D−1

],

(3)

- where i denotes a random index within the target sequence, since dynamic length inference makes the decoding start position non-fixed. To better unify next token prediction and block prediction, we continue to employ a shifted-prediction objective.

During inference, we introduce Longest Prefix Decoding, which uses low-order position priors, to decode the next sequence based on model’s confidence. Specifically, at each step, the model

perceives history x<(i−1) and produces fixed-length future logits Zi = [zi1,...,ziD] ∈ RD×|V| over vocabulary V, ultimately decoding only the first γτ(Zi) tokens. In the next step, predictions are repeated starting from the previous step’s end position. The formalization is as follows:

Xˆi = Decode Zi,γτ(Zi) (4)

where γτ(Zi) determines the adaptive sequence length to be decoded (with 1 ≤ γτ(Zi) ≤ D), and Decode(·) denotes extracting the next γτ(Zi) contiguous tokens from Zi. The maximum sequence length function γτ(·) is detailed in Section 3.4. This adaptive length mechanism can effectively balance generation efficiency and quality based on text’s semantic richness and uncertainty.

- 3.3 TRAINING

As noted in Section 3.2, when the block size is 1 our model reduces to the autoregressive paradigm, allowing reuse of pretrained ALM weights and cut training costs. From the perspective of instruction fine-tuning, we define the input as S = [X;Y ], where X is the prefix and Y the response.

Training. During training, we partition Y into blocks at random positions to train the model’s prediction capabilities at different starting positions. As shown in the Equation 3, for a starting position i, we construct a noise block YTi = [yi−1,[m],...,[m]] and predict the next fixed-length block

| |
|---|

| |
|---|

| |
|---|

|𝑥𝑥0|
|---|

|𝑦𝑦1|
|---|

Decoded token 𝑚𝑚 Mask Token |E| EOS Token Discarded Token

Causal Attention Bidirectional Attention Not Visible

Prompt Token

Cache Cached K/V

Last Decoded Token

𝑥𝑥𝑡𝑡

Prompt Response

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

- 𝑥𝑥1
- 𝑥𝑥2

……

|𝑥𝑥𝑢𝑢|
|---|

|𝑚𝑚|
|---|

|𝑚𝑚|
|---|

𝑥𝑥0

𝑥𝑥𝑢𝑢

𝑚𝑚

- 𝑦𝑦1
- 𝑦𝑦2

|E|

- 𝑦𝑦3
- 𝑦𝑦4
- 𝑦𝑦5
- 𝑦𝑦6

|𝑦𝑦3|
|---|

| |
|---|

𝑦𝑦1 𝑦𝑦2

Prefill

Decode

𝑦𝑦1 𝑦𝑦2 𝑦𝑦3 𝑦𝑦3 𝑚𝑚 𝑚𝑚 𝑚𝑚 𝑦𝑦5

K/V Cache

|𝑦𝑦4|
|---|

𝑦𝑦4 𝑦𝑦5 𝑦𝑦5 𝑚𝑚 𝑚𝑚 𝑚𝑚 𝑦𝑦6

K/V Cache

𝑥𝑥2

𝑚𝑚 𝑚𝑚 𝑚𝑚 𝑦𝑦4 𝑚𝑚 𝑚𝑚

|K/V Cache|
|---|

𝑦𝑦6 𝑦𝑦6 𝑚𝑚 𝑚𝑚 𝑦𝑦6

𝑚𝑚 𝑦𝑦7

…

…

…

|𝑚𝑚|
|---|

𝑦𝑦𝑣𝑣−2 𝑦𝑦𝑣𝑣−2 𝑚𝑚 𝑚𝑚 𝑦𝑦𝑣𝑣−1

K/V Cache

𝑚𝑚

𝑦𝑦𝑣𝑣 |E|

Inputs

𝑥𝑥1 𝑥𝑥2 𝑦𝑦1 𝑦𝑦2 𝑦𝑦3 𝑦𝑦4 𝑦𝑦5 𝑦𝑦6 |E| 𝑥𝑥2 𝑚𝑚 𝑚𝑚 𝑚𝑚 𝑦𝑦4 𝑚𝑚 𝑚𝑚

𝑚𝑚

𝑦𝑦1 𝑦𝑦2 𝑦𝑦3 𝑦𝑦4 𝑦𝑦5 |E|

𝑦𝑦6

Labels

Position

0 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7

8

(a) (b)

- Figure 2: Structured attention mask for parallel block training and sampling. (a) Reordered input yields a mask with causal prefix (top-left), visible cross-block prefix (bottom-left), and intrablock bidirectional attention (bottom-right). (b) Confidence-based next sequence prediction with KV reuse. A block of D tokens is predicted with D−1 masks. The longest high-confidence subsequence is selected as dynamic output. Cached KV states enable efficient decoding.

Y i = yi:(i+D) (simplified as Y i = [yi1,...,yiD]) by shifting. A bidirectional attention mechanism is used within the block for feature information, which serves as the basis for decoding dynamic-

length sequences. For historical information, we maintain casual attention as ALMs. Therefore, for a single noise block YTi, we can construct a custom attention mask A ∈ {0,1}(i+D)×(i+D):

Auv = 1v≤u ⊕ 1u≥i∩v≥i (5) This enforces strict causality for u < i and full mutual attention for u,v ≥ i.

Parallel Training. To enable efficient parallel training, we construct the sequence by interleaving noise blocks and target blocks as:

##### ST = concat(X, I1 · YT1, Y 1

##### , ..., Ii · YTi, Y i

##### , ...) (6)

Block 1

Block i

- where Ii ∈ {0,1} is a random indicator variable that controls whether a noise block YTi is inserted

at the current starting position i to predict the ground-truth block Y i. Each noise block YTi attends only within itself, while Y i is visible as prefix to later blocks but not vice versa, ensuring causality through attention constraints and positional encodings.

Since transformers rely on positional encodings, by rearranging S, the attention mask forms three parts as shown in Figure 2: (1) causal attention (top-left), (2) visible prefixes for each block (bottomleft), and (3) bidirectional attention within blocks (bottom-right). To improve training efficiency, we can concatenate any number of noise blocks after the target sequence within max sequence length. The sparse attention structure allows flex attention (Dong et al., 2024) to accelerate training.

- 3.4 INFERENCE

As described in Equation 4, we introduce the Longest Prefix Decoding method for dynamic length decoding based on low-order position priors. We primarily rely on the model’s confidence in its inferences as the basis to refine the length function γ(·), and design two types of decoding strategies:

Greedy Decoding. We implement γτ through a confidence-based stopping rule that identifies the longest prefix satisfying:

γτ(Zi) = max j ∈ {1,2,...,D} |

- j
- k=1

p(zik) ≥ τ ∪ {1} (7)

where p(zik) quantifies confidence at position k (where zik ∈ R|V| is the position-k logit vector), and τ is a predefined threshold. This approach greedily decodes at most j tokens (j ≥ 1) whose cumu-

lative product of confidence scores is greater than τ. We explore two distinct confidence functions:

- (1) Logit Value Confidence. This metric uses the softmax probability of the decoded token v at position k, to capture the model’s per-token confidence in its top prediction:

plogit(zik) = softmax(zik)v (8)

- (2) Entropy-Normalized Confidence. While plogit provides a pointwise confidence signal, it overlooks distributional ambiguity. Inspired by Wang et al. (2025) that higher predictive entropy correlates with forking behavior during generation, we employ an entropy-based confidence score:

|V|

Hp log |V|

pentropy(zik) = 1 −

, where Hp = −

pn log pn (9)

n

Here, pn is the softmax probability of the n-th word. Then, the entropy Hp is normalized by log |V|. Lower entropy indicates higher confidence, while higher entropy reflects more uncertainty.

Self-Speculative Decoding. Following the speculative decoding (Stern et al., 2018), we decode multiple tokens in parallel and verify their correctness through self-consistency checks. In each step,

the model produces D speculative tokens Yˆi = [ˆyi1,...,yˆiD] (where yˆik denote the k-th decoded token of block i) in an initial forward pass. To validate them, D verification inputs are constructed

by progressively extending prefixes of the sampled tokens, appending mask [m] at the first unverified position and padding to form a batch. A second forward pass then yields corresponding predictions

Y i = [ yi1,..., yiD]. The decoding sequence length is determined by the consistency-driven function: γverify(Zi) = max j ∈ {1,2,...,D} | yˆij = yij ∪ {1} (10)

Compared to confidence-based truncation via γτ, which relies on local heuristics, self-speculative decoding performs explicit consistency checks for self-verification without external models, offering greater reliability at the cost of an additional forward pass.

- 4 EXPERIMENTS

- 4.1 SETTING

To ensure a fair comparison, we fine-tune the Qwen-2.5 base model (Team, 2024) with all opensource instruction datasets (3.5 million samples, 2.3 billion tokens), covering math, code, and instruct-following. We compare SDLM against same-scale ALMs (Qwen2.5-3B/32B-Instruct, finetuned verison of Qwen2.5-3B/32B under the same setting), and larger DLMs like Dream-7B-Instruct and LLaDA-8B-Instruct across benchmarks spanning general, mathematics, science, and coding tasks. All evaluated with OpenCompass (Contributors, 2023) under standardized settings. Details about training and evaluating can be found in Appendix A.

- 4.2 MAIN RESULTS

Table 1 shows the performance and inference efficiency of our SDLM, trained in a single epoch on only 3.5M samples. SDLM-32B attains 92.4 on GSM8K, 74.2 on MATH-500, and 78.6 on IFEval, while remaining competitive on coding tasks. The SDLM-3B performs on par with or even surpasses Qwen-2.5-3B-SFT, and significantly outperforms larger DLMs such as LLaDA-8B and Dream-7B.

Table 1: Performance of instruct models across 8 long-form tasks. Numbers in parentheses (#) denote the speedup ratio: average tokens per pass vs. ALMs (1 token per pass). Results marked by † and ¶ are from Team (2024) and Ye et al. (2025) respectively. “–” indicates unknown data.

Model Name GSM8K MATH GPQA HumanEval HumanEval+ MBPP MBPP+ IFEval Avg. ALMs

Qwen-2.5-3B† 86.7 65.9 30.3 74.4 – 72.7 – 58.2 – Qwen-2.5-3B-SFT 85.8 59.8 27.8 73.8 60.4 68.5 42.6 62.1 60.1 Qwen-2.5-32B † 95.9 83.1 49.5 88.4 – 84.0 – 79.5 – Qwen-2.5-32B-SFT 93.2 74.8 33.8 82.9 76.2 82.1 59.0 76.5 72.3

DLMs

LLaDA-8B¶ 78.6 26.6 31.8 47.6 – 34.2 – 59.9 – Dream-7B¶ 81.0 39.2 33.0 55.5 – 58.8 – 62.5 –

84.6 60.8 28.3 67.1 59.8 65.4 40.5 57.1 57.9 (2.15) (2.18) (2.26) (1.91) (1.76) (1.66) (1.78) (1.38) (1.89) τ = .82

τ = .98

SDLM-3B (D = 4)

84.5 57.8 28.3 66.5 60.4 65.0 40.0 55.8 52.8 (2.75) (2.73) (2.66) (2.53) (2.25) (2.30) (2.29) (1.58) (2.39)

92.4 74.2 36.4 81.1 73.8 80.9 58.2 78.6 71.9 (2.15) (2.35) (2.34) (2.05) (2.29) (1.56) (1.51) (1.25) (1.94) τ = .82

τ = .98

SDLM-32B (D = 4)

92.3 73.0 36.9 79.9 73.2 80.9 57.1 78.2 71.4 (2.71) (2.88) (2.61) (2.82) (2.72) (2.17) (2.25) (1.43) (2.45)

In terms of generation efficiency, SDLM generate about 2 tokens per forward pass, reducing latency to about two-thirds of comparable ALMs. Taking GSM8K as an example, SDLM-32B at τ = 0.98 achieves accuracy 92.4 (vs. 93.2 for its same-scale SFT counterpart) while generating 2.15 tokens per step. Lowering τ to 0.82 further increases token output to 2.71 with only a 0.1 pct accuracy drop, highlighting an attractive speed-accuracy tradeoff. SDLM-3B follows a similar trend on GSM8K with minimal performance drop as τ is lowered. This trend holds across all benchmarks, where lowering τ consistently increases token generation while maintaining competitive performance. The effect and robustness of different τ values are ablated in Section 4.3.

In terms of short-answer benchmarks shown in Table 2, SDLM-32B performs within 1 ptc of its autogressive counterpart across MMLU, Winogrande, and Hellaswag, while SDLM-3B matches Qwen-2.5-3BSFT on these benchmarks. This demonstrates that SDLM retains the semantic and reasoning abilities of the base ALMs while enabling more efficient parallel decoding, confirming that our diffusion training preserves the base model’s NTP capability.

#### Table 2: Performance of instruct models across 5 general mutiple-choice tasks.

Model Name MMLU Winogrande Hellaswag ARC-C ARC-E ALMs

Qwen-2.5-3B-SFT 67.6 60.8 75.3 83.1 91.4 Qwen-2.5-32B-SFT 83.7 78.0 92.4 94.2 99.1

DLMs

LLaDA-8B 65.5 – 74.6 88.5 – Dream-7B 67.0 – – – –

SDLM-3B (D = 4) 66.3 60.2 74.2 82.7 92.0 SDLM-32B (D = 4) 82.8 79.2 92.0 94.9 98.9

Overall, SDLM delivers “near-SFT accuracy with significant inference acceleration” at both 3B and 32B scales, proving that NSP generation can stably converge in large-model regimes and providing a solid foundation for future work with larger parameters, longer training, and wider blocks.

- 4.3 TRADE-OFF BETWEEN SPEEDUP AND PERFORMANCE

Existing DLMs (Nie et al., 2025a; Ye et al., 2025) exploit parallel token generation but face a key trade-off: generating one token per step maintains quality, while producing multiple tokens often degrades it. Moreover, the reliance on fixed-length noise sequences constrains flexibility and limits practical efficiency gains over ALMs. In contrast, SDLM only concatenate a block-length masks per step, incurring minimal overhead compared to NTP inference.

- Figure 3 shows the speed-performance trade-off with varying confidence threshold τ across GSM8K, MATH-500 and HumanEval+. As τ decreases, SDLM generates more tokens per step,

[Figure 2]

Figure 3: Trade-off between performance and speed under different inference setting for SDLM-3B (D = 4) and SDLM-3B (D = 8). Adjusting τ allows a controllable trade-off between speed and performance. SpeedUp denotes the average number of tokens output per forward pass.

Table 3: SDLM-3B (D = 8) with larger block size and sampling with self-speculative decoding. (a) Larger blocks yield higher throughput with only minimal performance degradation. (b) With self-speculative decoding, the average accepted tokens per step (in green) significantly exceeds greedy decoding with greedy decoding threshold (Conf. τ).

Model Name GSM8K MATH HumanEval+ MBPP MBPP+ Avg. Qwen-2.5-3B-SFT (AR) 85.8 59.8 60.4 68.5 42.6 63.4

84.6 60.8 59.8 65.4 40.5 62.2

Conf. τ = .98

- (2.15) (2.18) (1.76) (1.66) (1.78) (1.91)

Speculative.

85.1 61.2 58.4 65.8 40.5 62.2

- (3.62) (3.54) (3.40) (3.29) (3.23) (3.42)

SDLM-3B (D = 4)

83.3 58.4 59.2 64.2 39.7 61.0 (2.52) (2.51) (2.01) (1.71) (2.16) (2.18) Speculative.

Conf. τ = .98

SDLM-3B (D = 8)

83.6 60.2 57.3 64.2 39.4 60.9 ((5.99) (5.73) (5.18)) (4.84) (5.33) (5.41)

achieving up to 3.5× speed-up. On math tasks like MATH-500, accuracy remains stable (61.4 → 59.2) as long as tokens per step stay under 3. Code tasks like HumanEval+ are more sensitive, with performance remaining high at around 1.7 tokens per step (60.4 → 59.8).

Furthermore, we compare the effects of generation block size D and confidence functions (Logit vs. Entropy). Results show that D = 4 generally yields slightly better accuracy, while the new trained model SDLM-3B (D = 8) enables greater acceleration due to larger parallel generation capacity. Both confidence function-based schemes maintain good performance. The threshold τ provides a flexible balance between speed and performance across various tasks and configurations.

- 4.4 ABLATION STUDY

Block Size. We investigate the impact of larger block sizes on SDLM-3B in Table 3, focusing on the new trained model SDLM-3B (D = 8). Compared to D = 4, the D = 8 configuration delivers substantially higher throughput with comparable model performance. Under Conf. τ = 0.98, the average number of output tokens per step increases from 1.9 (D = 4) to 2.2 (D = 8), with only a 1.2-point drop in overall accuracy. increasing D from 4 to 8 boosts the accepted tokens with only a small quality drop, suggesting potential for further throughput gains.

Self-Speculative Decoding. We further evluate self-speculative decoding in Table 3. In the Speculative rows, with D = 4 and D = 8, SDLM accepts an average of 3.4 and 5.4 tokens per step, corresponding to roughly 85% and 68% of the proposal budget, respectively. Model performance remains comparable across settings. However, this method incurs additional validation overhead, differing in experimental setup compared to the other two decoding methods. Despite this, it substantially enhances the model’s responsiveness, demonstrating its potential under specific conditions.

No Shift Prediction. To verify the effectiveness of shift prediction, we employ a method similar to LLaDA to directly predict the original tokens at the mask location. As shown in Figure 4, under

Under review as a conference paper at ICLR 2026

- 432
- 433
- 434
- 435
- 436
- 437
- 438
- 439
- 440
- 441
- 442
- 443
- 444
- 445
- 446
- 447
- 448
- 449
- 450
- 451
- 452
- 453
- 454
- 455
- 456
- 457
- 458
- 459
- 460
- 461
- 462
- 463
- 464
- 465
- 466
- 467
- 468
- 469
- 470
- 471
- 472
- 473
- 474
- 475
- 476
- 477
- 478
- 479
- 480
- 481
- 482
- 483
- 484
- 485

[Figure 3]

Figure 4: Ablation on attention mask type and prediction shift strategy. We ablate (1) Casual Attention: replace with causal mask, and (2) No shift: predicte xt instead of xt+1).

2025/9/25 18:41 bar-label-rotation (1).html

2025/9/25 18:41 bar-label-rotation (2).html

[Figure 4]

No Shift Causal Attention SDLM-3B (D=4)

Performance

90

885.85.8

884.24.2

884.24.2

80

6688

70

665.45.4

664.64.6

660.80.8 559.89.8

6600 559.89.8

558.88.8

60

50

445.75.7

440.50.5

338.68.6

338.48.4

40

30

GSM8K MATH HumanEval+ MBPP MBPP+

#### Figure 4: Ablation on attention mask type and prediction shift strategy. We conduct the follow-

Figure 5:

ing ablation experiments: (1) No shift: predicting xt instead of xt+1; (2) Leisure precautions: using a causal mask instead. The left image shows its model performance, while the right image shows the acceleration ratio.

User:

What fraction of 2 feet is 3 inches? Express your answer as a common fraction. Please reason step by step, and put your final answer within \boxed{}.

ﬁle:///Users/yueyin/Downloads/bar-label-rotation (1).html 1/1

ﬁle:///Users/yueyin/Downloads/bar-label-rotation (2).html 1/1

SDLM:

To find the fraction of 2 feet that is 3 inches , we need to convert the feet to inches and then divide the by the total number.

User:

What fraction of 2 feet is 3 inches? Express your answer as a common fraction. Please reason step by step, and put your final answer within \boxed{}.

- 1 foot = 12 inches
- 2 feet = 2 * 12 = 24 inches Now, we can the 3 inches by the fraction of 2 4 inches: \[ \frac{3}{24} = \frac{1}{8} \] So, the answer is 1/8. \[ \boxed{\frac{1}{8}} \]<|im_end|>

SDLM:

To find the fraction of 2 feet that is 3 inches , we need to convert the feet to inches and then divide the by the total number.

- 1 foot = 12 inches
- 2 feet = 2 * 12 = 24 inches Now, we can the 3 inches by the fraction of 2 4 inches: \[ \frac{3}{24} = \frac{1}{8} \] So, the answer is 1/8. \[ \boxed{\frac{1}{8}} \]<|im_end|>

38 step with 113 tokens

User: Write a Fibonacci function in Python. SDLM:

def fibonacci(n): 12 step with 44 tokens if n == 0:

38 step with 113 tokens

- return 0

elif n == 1:

- return 1

User: Write a Fibonacci function in Python. SDLM: def fibonacci(n): 12 step with 44 tokens

if n

| | | | |
|---|---|---|---|
|==| |0:| |

else: return fibonacci(n-1) + fibonacci(n-2)<|im_end|>#(fib

- return 0

elif n == 1:

- return 1

else: return fibonacci(n-1) + fibonacci(n-2)<|im_end|>#(fib

Figure 6: Visualization of the sampling process. Where each blue block indicates a subsequence generated in a single decoding step.

###### Figure 5: Visualization of the sampling process. Where each blue block indicates a subsequence generated in a single decoding step.

No Shift Prediction. To verify the effectiveness of shift prediction, we employ a method similar to LLaDA to directly predict the original tokens at the mask location. As shown in Figure 4, under the same training cost, this method leads to a noticeable decline in model performance, with HumanEval and HumanEval+ scores dropping by approximately 14 points. After log analysis, we find that the model has more repeated outputs. This indicates that the shift prediction method exploits the strong ability of ALMs to predict the first token and provides a stable starting point for diffusion decoding.

the same training cost, this method leads to a noticeable decline in model performance, with HumanEval+ scores dropping by approximately 14 points. After log analysis, we find that the model has more repeated outputs. This indicates that the shift prediction method exploits the strong ability of ALMs to predict the first token and provides a stable starting point for diffusion decoding.

9

Causal Attention. As shown in Figure 4, we replace bidirectional attention inside each block with a causal (unidirectional) masking. With a block size D = 4, the two variants obtain almost identical scores on some benchmarks and exhibit comparable training difficulty. However, the average number of tokens generated per step decreases from 1.88 to 1.82, indicating that bidirectional attention enlarges the local receptive field during decoding and improves parallel generation efficiency.

Case Study. Figure 5 illustrates SDLM’s flexible decoding, where the generated sequence length adapts to local context. In fluent or structured regions (e.g. math expressions, structured code segments, common phrases), it confidently emits longer sequences at once. While facing uncertainty or forking, it slows down with shorter outputs. This adaptive strategy balances speed with precision.

5 CONCLUSION

In conclusion, we propose Next Sequence Prediction (NSP), a unified framework bridging autoregressive and diffusion decoding. Building on NSP, we develop Sequential Diffusion Language Models (SDLMs) that adapt pretrained ALMs via parallel block training and dynamic decoding. SDLM matches SFT-tuned ALMs in performance while decoding faster, offering a stronger speed–performance trade-off. We hope this work inspires further exploration of unified sequence generation.

REFERENCES

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Mart´ın Bl´azquez, Guilherme Penedo, Lewis Tunstall, Andr´es Marafioti, Hynek Kydl´ıˇcek, Agust´ın Piqueres Lajar´ın, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Cl´ementine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, and Thomas Wolf. Smollm2: When smol goes big – data-centric training of a small language model, 2025. URL https://arxiv.org/abs/2502.02737.

Anthropic. Introducing Claude, 2023. URL https://www.anthropic.com/index/ introducing-claude.

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. arXiv preprint arXiv:2401.10774, 2024.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, pp. 02783649241273668, 2023.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing GPT-4 with 90%* ChatGPT quality, March 2023. URL https: //lmsys.org/blog/2023-03-30-vicuna/.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/opencompass, 2023.

Google DeepMind. Gemini 2.5, 2025. URL https://blog.google/technology/ google-deepmind/gemini-model-thinking-updates-march-2025/.

Yuyang Ding, Xinyu Shi, Xiaobo Liang, Juntao Li, Qiaoming Zhu, and Min Zhang. Unleashing reasoning capability of llms via scalable question synthesis from scratch. arXiv preprint arXiv:2410.18693, 2024.

Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, and Horace He. Flex attention: A programming model for generating optimized attention kernels. arXiv preprint arXiv:2412.05496, 2024.

Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozi`ere, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737, 2024.

Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. arXiv preprint arXiv:2410.17891, 2024.

Google DeepMind. https://blog.google/technology/google-deepmind/gemini-diffusion/. https: //blog.google/technology/google-deepmind/gemini-diffusion/, 2025. Accessed: 2024-07-24.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In ICLR. OpenReview.net, 2021a.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In NeurIPS Datasets and Benchmarks, 2021b.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Siming Huang, Tianhao Cheng, Jason Klein Liu, Jiaran Hao, Liuyihan Song, Yang Xu, J Yang, Jiaheng Liu, Chenchen Zhang, Linzheng Chai, et al. Opencoder: The open cookbook for top-tier code large language models. arXiv preprint arXiv:2411.04905, 2024.

Ivan Kapelyukh, Vitalis Vosylius, and Edward Johns. Dall-e-bot: Introducing web-scale diffusion models to robotics. IEEE Robotics and Automation Letters, 8(7):3956–3963, 2023.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Peng Li, Yeye He, Dror Yashar, Weiwei Cui, Song Ge, Haidong Zhang, Danielle Rifinski Fainman, Dongmei Zhang, and Surajit Chaudhuri. Table-gpt: Table-tuned gpt for diverse table tasks. arXiv preprint arXiv:2310.09263, 2023.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in Neural Information Processing Systems, 36:21558–21572, 2023.

Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyan Wei, Shaobo Wang, and Linfeng Zhang. dllm-cache: Accelerating diffusion large language models with adaptive caching,

2025. URL https://github.com/maomaocun/dLLM-cache. Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion language models. arXiv preprint arXiv:2505.15781, 2025.

Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling up masked diffusion models on text. arXiv preprint arXiv:2410.18514, 2024.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint

- arXiv:2502.09992, 2025a.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint

- arXiv:2502.09992, 2025b.

OpenAI. Introducing ChatGPT, 2022. URL https://openai.com/index/chatgpt/. OpenAI. Hello GPT-4o, 2024. URL https://openai.com/index/hello-gpt-4o/. Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan

Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. Technical report, OpenAI, 2018.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An adversarial winograd schema challenge at scale. Commun. ACM, 64(9):99–106, 2021.

Mohammad Samragh, Arnav Kundu, David Harrison, Kumari Nishu, Devang Naik, Minsik Cho, and Mehrdad Farajtabar. Your llm knows the future: Uncovering its multi-token prediction potential. arXiv preprint arXiv:2507.11851, 2025.

Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, Zheng Li, Yuehang Yang, Hongli Yu, Xingwei Qu, et al. Seed diffusion: A large-scale diffusion language model with high-speed inference. arXiv preprint arXiv:2508.02193, 2025.

Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. Blockwise parallel decoding for deep autoregressive models. Advances in Neural Information Processing Systems, 31, 2018.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ram´e, Morgane Rivi`ere, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm. github.io/blog/qwen2.5/.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

David Wadden, Kejian Shi, Jacob Morrison, Aakanksha Naik, Shruti Singh, Nitzan Barzilay, Kyle Lo, Tom Hope, Luca Soldaini, Shannon Zejiang Shen, et al. Sciriff: A resource to enhance language model instruction-following over scientific literature. arXiv preprint arXiv:2406.07835, 2024.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025.

Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, et al. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025.

xAI. Grok 3 beta — the age of reasoning agents, 2025. URL https://x.ai/news/grok-3. Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, and Zhifang Sui. Speculative de-

coding: Exploiting speculative execution for accelerating seq2seq generation. arXiv preprint arXiv:2203.16487, 2022.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b, 2025. URL https://hkunlp.github.io/blog/2025/dream.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Lin Zheng, Jianbo Yuan, Lei Yu, and Lingpeng Kong. A reparameterized discrete diffusion model for text generation. arXiv preprint arXiv:2302.05737, 2023.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

### A DETAILS OF TRAINING We show the training hyperparameters in Table 4.

Table 4: Training Hyperparameters for SDLM. Parameter SDLM-3B SDLM-32B Max sequence length 5,632 Epochs 1 Batch size (global) 256 464 Training steps 13,699 7,558 Learning rate 5 × 10−6 (constant) ZeRO stage 1 3

The training corpus comprises with: Tulu-3-SFT-Mixture (Lambert et al., 2024), Table-GPT (Li

- et al., 2023), SciRIFF (Wadden et al., 2024), SmolTalk (Allal et al., 2025), OPC-SFT-Stage2 (Huang
- et al., 2024), and ScaleQuest-Math (Ding et al., 2024), with a combined total of 3.5 million samples (∼ 2.3 billion tokens).

To comprehensively evaluate the capabilities of SDLM, we conduct evaluations across a diverse set of benchmarks encompassing:

General Tasks. MMLU (Hendrycks et al., 2021a)(5-shot), Winogrande (Sakaguchi et al., 2021)(0shot), Hellaswag (Zellers et al., 2019)(10-shot), ARC-C/E (Clark et al., 2018)(0-shot), IFEval (Zhou

- et al., 2023)(0-shot).

Mathematics & Science Tasks. GSM8K (Cobbe et al., 2021) (0-shot), MATH-500 (Hendrycks et al., 2021b)(0-shot), GPQA (Rein et al., 2024) (0-shot).

Coding Tasks. HumanEval (Chen et al., 2021) (0-shot), Humaneval+ (Liu et al., 2023) (0-shot), MBPP (Austin et al., 2021) (3-shot), MBPP+ (Liu et al., 2023) (3-shot).

B COMPARE WITH MULTI-TOKEN PREDICTION

SDLM can be viewed through the lens of MTP as well. Both SDLM and MTP parallelize autoregressive generation by predicting multiple tokens in a single forward pass. For a prediction horizon of D tokens, MTP use D separate output heads, with the i-th head predicting the token at position m + i. Similarly, SDLM uses D positions in the input sequence: the last token (at position m) and D−1 mask tokens. The prediction at the last token position corresponds to the token at t+1 (equivalent to MTP’s first head), and the prediction at the j-th mask token (1 ≤ j ≤ D − 1) corresponds to the token at m + 1 + j (equivalent to MTP’s (j + 1)-th head).

However, SDLM introduces two advantages. First, the predictions are generated within a local bidirectional attention window, enabling joint context utilization across the predicted tokens. This contrasts with MTP’s isolated head (Cai et al., 2024; Gloeckle et al., 2024) or left-to-right attention (Liu

- et al., 2024). Second, extending the prediction horizon requires no architectural modification: appending additional mask tokens suffices, while MTP necessitates adding new output heads.

