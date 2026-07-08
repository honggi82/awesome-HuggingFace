[Figure 1]

[Figure 2]

[Figure 3]

### InfiniteVL: Synergizing Linear and Sparse Attention for Highly-Efficient, Unlimited-Input Vision-Language Models

Hongyuan Tao1,⋄ Bencheng Liao1 Shaoyu Chen2 Haoran Yin2 Qian Zhang2 Wenyu Liu1 Xinggang Wang1,

1Huazhong University of Science and Technology 2Horizon Robotics Code & Model & Demo: hustvl/InfiniteVL

## arXiv:2512.08829v2[cs.CV]31Mar2026

#### Abstract

Vision-Language Models (VLMs) are increasingly tasked with ultra-long multimodal understanding. While linear architectures offer constant computation and memory footprints, they often struggle with high-frequency visual perception compared to standard Transformers. To bridge this gap, we introduce InfiniteVL. We first develop a hybrid base model called InfiniteVL-Base that interleaves a small fraction of Full Attention layers with Gated DeltaNet. Empowered by a tailored distillation and fine-tuning strategy, InfiniteVL-Base matches the fundamental multimodal performance of equivalent Transformers while achieving a 1.7× decoding speedup. However, the quadratic complexity of the retained Full Attention inevitably becomes an efficiency bottleneck when scaling to ultra long context. To break this barrier, we propose a novel Long-Sequence Architectural Fine-Tuning strategy that seamlessly transforms the dense attention into vision-specific sparse mechanisms. This yields two specialized variants: InfiniteVLOffline for offline retrieval and InfiniteVL-Online for online streaming. By eliminating the computation explosion of global attention without sacrificing high-frequency visual recall, InfiniteVL-Offline achieves Transformer-level length generalization with a 5x prefill acceleration at 256K context. Concurrently, InfiniteVL-Online delivers robust streaming perception with a constant memory footprint and a real-time throughput of 25 FPS.

Qwen2.5VL-3B InfiniteVL-Base

74.2

[Figure 4]

InfiniteVL-Online InfiniteVL-Offline

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

25

118.7

80.9

56.4

68.0 53.553.052.7

67.767.6

43.9

79.8

93.9

6.2

24.7

2.8

better

84.0

Long Video Performance

Streaming Performance

Prefill Speed (256K) (s)

Streaming FPS

Figure 1. Performance and Efficiency Overview. (Left) InfiniteVL-Base fully matches the foundational multimodal performance of leading 2B–4B Transformer VLMs. (Right) In ultra-long scenarios, our variants achieve a superior balance between efficiency and longcontext understanding.

⋄ Intern of Horizon Robotics; Corresponding author: Xinggang Wang (xgwang@hust.edu.cn).

#### 1 Introduction

VLMs [3, 11, 22, 30, 38, 39, 74, 80] are rapidly evolving from static image-text comprehension toward continuous, realworld multimodal understanding [4, 14, 24, 33, 82]. However, scaling these models to handle long multimodal inputs is severely challenged by the inherent computational constraints of the standard Transformer architecture [60]. The quadratic complexity of self-attention with respect to sequence length, coupled with a dynamically growing key-value (KV) cache during autoregressive inference, creates unsustainable computational and memory demands. This inefficiency becomes a critical bottleneck in long-context scenarios, such as extended video understanding or continuous agent interaction. It also prohibits deployment on resource-constrained edge devices, where real-time performance and a strictly low memory footprint are paramount.

Recently, linear attention mechanisms [21, 23, 25, 35, 52, 69–71, 81] have emerged as a promising avenue to address the long-context efficiency bottlenecks in VLMs. Benefiting from constant-size state compression and recent advances in hardware-aware efficient computation, linear architectures can process unbounded multimodal sequences with a strictly constant computation and memory footprint, circumventing the quadratic scaling of standard Transformers. However, this inherent state compression comes at a cost: it inevitably discards high-frequency visual details. Consequently, existing linear VLMs [31, 32, 34, 49, 77, 83] suffer severe performance degradation on information-dense multimodal tasks, such as Optical Character Recognition (OCR) and high-resolution document comprehension, where localized precise perception is indispensable. As a result, current solutions are caught in a dilemma between long-term computational efficiency and precise visual performance.

Hybrid architectures which interleave a small fraction of Full Attention layers with linear modules have shown promise in Large Language Models (LLMs) by restoring precise retrieval capabilities [5, 36, 50]. We hypothesize that extending this hybrid paradigm to multimodal domains could effectively recover the high-frequency visual details typically discarded by linear VLMs. Building on this insight, we first construct a foundational hybrid base for InfiniteVL which is called InfiniteVLBase. This architecture allocates approximately 75% of the Gated DeltaNet layers [69] for efficient long-term memory compression, while retaining 25% as Full Attention layers to preserve precise visual perception. To effectively optimize this hardware-friendly architecture under limited computational resources, we implement a tailored training strategy utilizing a carefully curated multimodal corpus. This involves Distillation Pretraining to transfer robust multimodal knowledge from advanced Transformers, followed by Continuous Supervised Fine-tuning (Continuous SFT) to enhance component alignment and instruction following. As a preliminary exploration, InfiniteVL-Base successfully achieves precise multimodal understanding on par with mainstream Transformers, delivering over a 1.7× inference speedup and a 3.1× expansion in context length processing capabilities.

However, the quadratic memory growth of the retained Full Attention layers inevitably triggers severe bottlenecks when scaling to ultra-long multimodal sequences. To overcome this, we propose a novel Long-Sequence Architectural FineTuning strategy that adaptively converts the dense attention into vision-specific sparse mechanisms, yielding two variants tailored for the two primary scenarios of multimodal long-sequence understanding: online streaming and offline comprehension. For online streaming, the model faces an unbounded influx of frames, strict constant-memory constraints, and an inherent recency bias. We introduce InfiniteVL-Online, which integrates a sliding window to capture recent high-frequency details and attention sinks [67] as anchors to stabilize computation. Combined with the intact DeltaNet layers, it achieves Transformer-level performance on StreamingBench [37] while sustaining a robust, real-time throughput of 25 FPS. Alternatively, offline comprehension ingests all frames simultaneously, incurring exorbitant quadratic prefill costs and demanding exact retrieval. Here, we propose InfiniteVL-Offline which utilizes dynamic top-k routing mechanism that performs chunklevel retrieval, mitigating the prefill burden while preserving precise visual recall. Evaluations on Video-MME [17] and LongVideoBench [63] show that InfiniteVL-Offline matches the length generalization of mainstream Transformers, delivering comparable performance alongside a 5× prefill acceleration at a 256K context length.

In summary, our main contributions are as follows:

- • We propose InfiniteVL, systematically integrating Gated DeltaNet with Full Attention to bridge the precise visual perception gap of linear VLMs. To fundamentally break the quadratic efficiency bottleneck in extended contexts, we introduce a Long-Sequence Architectural Fine-Tuning strategy, adaptively converting the retained attention into vision-specific sparse variants (InfiniteVL-Online and InfiniteVL-Offline) tailored for online and offline scenarios.
- • We design a highly effective, tailored training pipeline encompassing distillation pretraining, Continuous SFT, and LongSFT. This strategy successfully overcomes the inherent training instability of linear layers, seamlessly transferring robust multimodal knowledge from mainstream Transformers.
- • Extensive experiments demonstrate that InfiniteVL-Base achieves Transformer-level fundamental multimodal understand-

ing with over a 1.7× decoding speedup. In ultra-long multimodal scenarios, InfiniteVL-Offline delivers a 5× prefill acceleration at a 256K context length without sacrificing retrieval accuracy. Concurrently, InfiniteVL-Online sustains a robust real-time throughput of 25 FPS with a strictly constant O(1) memory footprint.

- 2 Related Work

- 2.1 Vision-Language Models

Modern VLMs typically integrate visual encoders with pre-trained LLMs [6, 59] through large-scale image-text pretraining. By scaling up data and model capacity, this paradigm achieves strong performance on fundamental vision-language tasks, such as Visual Question Answering (VQA), image captioning, and visual grounding [4, 14, 24, 33, 82]. Additionally, many contemporary VLMs extend their capabilities to video understanding by incorporating temporal modeling mechanisms, allowing them to process short to medium length video clips [3, 10, 11, 43, 54, 55]. As the field advances, a key direction for VLMs is continuous and long-term multimodal understanding particularly for embodied AI and physical scene perception [72]. This requires models to process extensive spatiotemporal contexts. However, using the standard Transformer architecture for such long sequences is severely constrained by its quadratic computational complexity and the continuous growth of the KV cache during inference. While recent methods attempt to extend context length through cache dropping or sliding windows, these approaches often compromise long-term memory and situational coherence [9, 76]. Therefore, it is necessary to explore more memory and computation efficient architectures.

- 2.2 Linear Attention

Linear attention [25] and State Space Models (SSMs) [21] have emerged as pivotal architectures to address the quadratic complexity of standard softmax attention. By decomposing similarity computation and utilizing recurrent state updates, these methods compress historical information into a fixed size state, reducing computational complexity to linear scale while enabling inference with constant memory. To mitigate the capacity limits and memory collisions inherent in vanilla linear attention, recent advances have introduced forgetting mechanisms and delta rule-based updates. This evolution has yielded highly efficient architectures, such as Mamba [21], Gated Linear Attention (GLA) [70], and Gated DeltaNet [69], which effectively enhance memory compaction and serve as robust repositories for long-term context.

Motivated by these computational benefits, several prior works have explored VLMs build with linear attention or SSMs [31,

32, 34, 77]. However, applying those architectures to multimodal tasks exposes a critical limitation. The inherent state compression mechanism inevitably causes the loss of high-frequency visual details. Consequently, existing linear VLMs often underperform on information-dense perception tasks, such as OCR and high-resolution document understanding when compared to Transformer baselines. This performance gap indicates that relying solely on compressed recurrent states is insufficient for comprehensive visual perception, highlighting the necessity for architectures that can balance efficient memory compression with precise local detail retention.

- 2.3 Hybrid Architecture

The weakness of linear attention models in exact recall tasks inherently stems from their low-rank state compression nature. While effectively retaining macroscopic long-term memory, they inevitably discard high-frequency visual details. To mitigate this, some works have explored hybrid architectures by reintroducing a small fraction of Full Attention layers, specifically tasked with recovering the precise features lost during state compression [19, 36, 50]. Empirical studies in LLMs have demonstrated that interleaving even a minimal proportion of Full Attention can effectively bridge the retrieval capability gap of linear architectures [36], thereby driving the design and scaling of large-scale hybrid LLMs [5, 7, 57, 58].

Recently, this hybrid paradigm has been extended to VLMs [32, 34]. However, they still exhibit a noticeable performance gap compared to leading Transformer-based VLMs. Concurrently, the reintroduced Full Attention layers reintroduces an efficiency bottleneck for multimodal long-sequence understanding. Therefore, exploring a highly efficient multimodal hybrid architecture that can simultaneously bridge the visual perception gap and demonstrate compelling efficiency advantages has become a critical open challenge.

Hybrid Block FFN

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Gated DeltaNet

| |
|---|

KV Cache Query

[Figure 15]

Mem: T.C:

[Figure 16]

| | |
|---|---|
| | |

State

[Figure 17]

[Figure 18]

Defined Chunks: k=64 C=32 Mem: T.C:

×3

(b) InfiniteVL-Offline

FFN

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Architecture Full Attention Finetuning

[Figure 29]

Mem: T.C:

[Figure 30]

Query

KV Cache ×1

×9

Sinks: 256 Evicted KV Cache Window KV Cache: 2048

Vision Encoder & MLP Text Tokenizer

[Figure 31]

[Figure 32]

Mem: T.C:

(a) InfiniteVL-Base

(c) InfiniteVL-Online

- Figure 2. Overall Architecture of InfiniteVL. (a) InfiniteVL-Base hybridizes Gated DeltaNet with a small fraction of Full Attention. Through our Long-Sequence Architectural Fine-Tuning, we yield two highly efficient variants: (b) InfiniteVL-Offline for efficient blocklevel routing in offline long-contexts, and (c) InfiniteVL-Online for constant-memory online streaming.

#### 3 Method

As illustrated in Fig. 2, InfiniteVL is a high-performance, long-context hybrid VLM. We first construct InfiniteVL-Base (Sec. 3.1) by integrating a vision encoder with an LLM backbone that interleaves full and linear attention, optimized via Distillation Pretraining and Continuous SFT (Sec. 3.2). For ultra-long sequences, we introduce a long-sequence architectural fine-tuning stage (Sec. 3.3). By adaptively converting global attention into vision-specific sparse mechanisms, we derive two variants: InfiniteVL-Online for online streaming and InfiniteVL-Offline for offline contexts, achieving a better balance between context efficiency and precise visual perception.

#### 3.1 InfiniteVL-Base

InfiniteVL-Base comprises a Qwen2.5-VL [3] vision encoder, a projection MLP, and a decoder-only LLM backbone containing 9 hybrid blocks (d = 2048). To balance precise retrieval and efficiency, each block interleaves one Full Attention layer with three consecutive Gated DeltaNet layers, connected via residual pathways and Pre-Layer Normalization with SiLU activations. The Full Attention layer uses Grouped-Query Attention (GQA) [1] with 16 Query and 2 KV heads. Spatial-temporal semantics are injected via 3D Rotary Position Embedding (3D RoPE) [56]. The standard attention is computed as:

QKT √dk

OFA = Softmax

V. (1)

Gated DeltaNet layers[69] provide constant-memory state compression. Since Gated DeltaNet intrinsically avoids KV cache explosion, we can afford to symmetrically configure it with 16 Query and 16 KV heads, thereby maximizing its outerproduct memory capacity. We further integrate a 1D convolution (window size 4) and an output gate for expressiveness. Operating without explicit positional encodings or weight biases, the recurrent memory update and retrieval are formulated as:

St = γtSt−1 + βtKt ⊗ (Vt − St−1Kt), (2)

OLinear,t = QtSt, (3) where St is the fixed-size hidden state, γt serves as a data-dependent decay gate, and βt controls the update rate.

##### Distillation Pretraining Continuous SFT Long-Seq Architectural SFT

[Figure 33]

Inputs ≈ 800K Multimodal SFT data

###### Inputs ≈ 1M Multimodal QA/Caption data Inputs ≈ 8M Multimodal SFT data

+ 200K Long-Sequence Data

Inherited Params

[Figure 34]

Text-only

[Figure 35]

New Initialized Params

[Figure 36]

General VQA

16.4% 11.9%

[Figure 37]

Trainable Params

Caption & Knowledge

Teacher Student

11.9%

Chart & Table

11.6%

&Layer-to-Layerdistillation End-to-End

Vision encoder/ Tokenizer

Vision encoder/ Tokenizer

OCR & Document

High Resolution Images

Parameterinheritance

Long Documents & Articles

Video Caption Full Attention & QA

Mathematics & Reasoning

10.5%

35.4%

Full Attention

Science & Education

Full Attention

Code & Programmatic

FFN

FFN

0.78%

1.56%

Full Attention

Gated DeltaNet

Topk sparse Attn Sink SWA

[Figure 38]

[Figure 39]

FFN

FFN

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
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
| | | | | | | |

ℒ𝑚𝑚𝑚𝑚𝑚𝑚

Lm head

Lm head

Visionencoder/

GatedDeltaNet

Toekenizer

Lmhead

Inputs

Inputs

SWA

FFN

FFN

distillation

Teacher model Student model

ℒ𝑘𝑘𝑘𝑘

Online Streaming Video Understanding

Offline Long Video Understanding

Outputs Outputs

- Figure 3. Progressive Multi-Stage Training Strategy. Stage I employs layer-to-layer and end-to-end distillation to align the newly initialized linear modules; Stage II enhances component alignment via Continuous SFT; Stage III specifically adapts the dense attention into vision-specific sparse mechanisms for extended contexts.

#### 3.2 Efficient Multi-Stage Training Strategy

Training a hybrid VLM from scratch is computationally prohibitive and demands massive multimodal corpora. To circumvent this, we aim to leverage the robust pre-trained knowledge in leading Transformer VLMs. Consequently, we adopt a minimal-replacement strategy, preserving the majority of the pre-trained weights while selectively substituting specific dense attention layers with randomly initialized linear modules. However, this structural modification introduces severe architectural heterogeneity, rendering direct optimization highly unstable and prone to catastrophic forgetting. To efficiently align these components and stabilize the training process, we design a progressive two-stage training pipeline for InfiniteVL-Base, as illustrated in Fig. 3. By decoupling the learning objectives, this paradigm first employs Distillation Pretraining to stabilize the newly initialized linear modules, followed by Continuous SFT to achieve seamless component alignment and unlock complex multimodal interactions. This phased approach guarantees the seamless transfer of robust pre-trained priors while strictly constraining the overall training budget.

###### 3.2.1 Training Data.

We curate a comprehensive multimodal corpus integrating FineVision [62], LLaVA-OneVision-1.5 [2], PixMo [13], The Cauldron [28], and Docmatix [27]. Spanning tasks from OCR to mathematics, its scale and diversity match leading VLMs like InternVL2.5 [11].

###### 3.2.2 Distillation Pre-training. We initialize the student by replacing specific Qwen2.5-VL attention layers with Gated DeltaNet. To align these modules, we first perform layerwise feature distillation, minimizing the Mean Squared Error (MSE) between the teacher’s (h(Transi) ) and student’s (h(GDNi) ) i-th layer outputs given the same input:

L(layeri) = h(GDNi) − h(Transi) 22. (4) This “same-input, output-matching” design accelerates alignment. Next, an end-to-end distillation phase minimizes the Kullback-Leibler (KL) divergence between token-level logits:

T

1 T

KL Softmax(ztT) Softmax(ztS) , (5)

Llogit =

t=1

where ztT and ztS are the teacher and student logits. Here, we cap image resolution at 512 × 512 and input length to 8192 tokens.

###### 3.2.3 Continuous SFT.

To unlock the linear modules’ full potential and mitigate exposure bias [48], we fine-tune the distilled model using a standard Cross-Entropy (CE) loss:

T

1 T

CE qt,Softmax(ztS) , (6)

LSFT =

t=1

where qt is the target distribution. To improve precise visual detail perception, we increase maximum resolution to 1344 × 1344 while maintaining the 8192-token limit.

#### 3.3 Long-Sequence Architectural Fine-Tuning Strategy

By interleaving a small fraction of Full Attention layers, InfiniteVL-Base effectively mitigates the detail forgetting issues inherent in linear architectures. However, when scaling to ultra-long multimodal sequences of length L, the retained Full Attention layers inevitably dominate the computational and memory footprint, scaling quadratically O(L2) in prefill time and linearly O(L) in decoding memory. We identify that realworld long-context multimodal understanding predominantly unfolds in two distinct paradigms: offline comprehension and online streaming, each presenting unique efficiency bottlenecks. To overcome these without compromising the model’s precise visual perception or causing catastrophic forgetting of pre-trained knowledge, we propose a Stage III Long-Sequence Architectural Fine-Tuning strategy. Tailored to the specific characteristics of the two paradigms, we introduce two variants: InfiniteVL-Offline and InfiniteVL-Online. These variants adaptively transform the dense global attention into vision-specific sparse mechanisms, acting as highly efficient retrieval complements to the linear states while fundamentally shifting the complexity paradigm.

###### 3.3.1 InfiniteVL-Offline for Offline Comprehension.

In offline comprehension tasks (e.g., long-video analysis), all multimodal tokens are ingested simultaneously. This upfront loading triggers the O(L2d) quadratic prefill complexity of the retained Full Attention, leading to a severe computational cost that dominates overall latency. Furthermore, these tasks often require exact retrieval across the massive context (e.g., pinpointing a specific visual event).

To address this severe prefill bottleneck while satisfying the demand for exact retrieval, we introduce InfiniteVL-Offline, which reuses dense attention parameters through a parameter-free architecture modification. Inspired by InfLLM-V2 [78], we implement a coarse-to-fine compression paradigm to perform efficient block-level routing. Specifically, the historical sequence is partitioned into fixed-size chunks of 32 tokens. For a given query block i, we first compute coarse-grained attention scores and dynamically route queries to the top-k (k = 64) most relevant blocks. The final attended indices are mathematically defined as the union of three critical components:

I(i) = Iinit ∪ Ilocal(i) ∪ Itopk(i). (7)

Here, Iinit protects the initial system prompts, Ilocal(i) preserves local temporal coherence, and Itopk(i) enables global precise retrieval. By restricting the attention computation to only these retrieved blocks (totaling length C ≪ L), the prefill time complexity is drastically reduced from O(L2d) to O(L · Cd), effectively achieving linear scaling. This hybrid dense-sparse design relies on the linear layers to compress the macroscopic long-term history, while the sparse attention efficiently routes and recalls specific high-frequency visual details.

###### 3.3.2 InfiniteVL-Online for Online Streaming.

Conversely, online streaming (e.g., continuous agent perception) is characterized by an unbounded, continuous influx of frames. In this paradigm, standard attention requires O(Ld) memory for the KV cache, which eventually exceeds GPU VRAM capacity and inevitably leads to Out-of-Memory (OOM) failures. Therefore, the memory footprint must be strictly bounded to a constant O(1) scale.

While reducing Full Attention to standard Sliding Window Attention (SWA) successfully bounds memory and aligns well with the inherent recency bias of streaming perception, naive SWA suffers from ”attention collapse”. It indiscriminately

###### Model MME MMStar MMBenchtest en SeedBenchimage ScienceQAval RealworldQA AI2Dw/o M Average

TinyLLaVA-3B [79] 1733 37.9 69.5 70.2 68.7 55.0 61.8 61.8 PaliGemma2-3B [20] 1658 52.7 60.7 71.6 94.3 58.3 72.2 68.0 Phi-3.5-Vision-4B [47] 1846 47.5 76.0 71.2 92.2 57.9 77.8 70.9 SmolVLM2-2B [44] 1764 46.0 43.0 70.9 90.0 58.4 74.9 64.8 InternVL2.5-4B [11] 2338 58.3 81.1 74.1 97.0 64.3 81.4 78.5 Qwen2.5VL-3B [3] 2171 54.3 78.2 73.3 81.4 65.4 81.6 74.4 Qwen2.5VL*(4B) 2089 55.7 78.7 72.9 93.1 65.8 77.1 75.3 InfiniteVL-Base 2126 55.6 79.0 72.9 93.4 67.3 77.2 75.8

(a) Results on General Benchmarks

###### Model ChartQAtest TextVQAval DocVQAtest OCRBench MMMUval MathVistamini Average

TinyLLaVA-3B [79] 21.2 55.3 34.7 36.0 36.2 28.3 35.3 PaliGemma2-3B [20] 33.6 63.0 71.6 60.1 30.3 27.7 47.7 Phi-3.5-Vision-4B [47] 81.8 72.0 69.3 59.9 43.0 43.9 61.7 SmolVLM2-2B [44] 68.8 73.2 80.0 72.9 42.0 51.5 64.7 InternVL2.5-4B [11] 84.0 76.8 91.6 82.8 52.3 60.5 74.7 Qwen2.5VL-3B [3] 84.0 79.6 93.9 79.7 49.6 62.3 74.9 Qwen2.5VL*(4B) 82.5 79.1 92.2 80.5 43.7 65.2 73.8 InfiniteVL-Base 82.0 78.5 91.7 79.8 44.0 65.4 73.6

(b) Results on Chart, Document, and Math Benchmarks

- Table 1. Foundational Multimodal Performance. Comparison of InfiniteVL-Base against representative 2B–4B VLMs across (a) general multimodal benchmarks and (b) visually-intensive tasks requiring precise recall (e.g., OCR, Chart, Document). InfiniteVL successfully bridges the performance gap inherent in linear architectures.

discards early historical tokens (e.g., system instructions and initial scene setups) which absorb massive attention scores and are paramount for maintaining the stable distribution of the softmax function [66, 68].

To achieve continuous streaming without performance degradation, InfiniteVL-Online utilizes a sliding window to capture recent high-frequency details while designating a fixed length of initial tokens as “attention sinks” serving as anchors to stabilize the computation. The attention mask matrix M is formulated as:

Mi,j =

1, if j < Nsink or i − j < W, 0, otherwise,

(8)

where Nsink = 256 is the number of sink tokens and W = 2048 is the window size. Because the retained KV cache size is strictly bounded to Nsink +W, the memory complexity drops from O(Ld) to O(1), and the decoding time per token becomes constant. Combined with the intact Gated DeltaNet layers that continuously compress distant history into a fixed-size state, InfiniteVL-Online guarantees sustainable processing, completely eliminating OOM risks.

- 3.3.3 Long-Sequence Fine-Tuning.

Both variants seamlessly inherit the pre-trained weights of InfiniteVL-Base, optimizing long-sequence efficiency without harming foundational multimodal comprehension. To further enhance length generalization up to 32,768 tokens, we conduct a third stage of fine-tuning as shown in Fig. 3. The training data blends 800,000 samples from the Stage II SFT corpus with 200,000 video QA pairs uniformly sampled from LLaVA-Video-178K [75] (10 FPS, up to 224 frames, max 256 tokens/frame). The optimization objective strictly mirrors the Cross-Entropy loss of the SFT stage, smoothly adapting the model to the newly introduced sparse attention masks.

- 4 Experiments

#### 4.1 Experimental Setup

###### 4.1.1 Baselines.

To isolate architectural effects, we construct a controlled Transformer baseline. Using identical training pipelines and data, we train a pure Transformer VLM (initialized from Qwen2.5-VL) equivalent in scale to our model, denoted as Qwen2.5-VL*.

High Resolution Long Video Streaming Video Vstar BLINK HRBench4K Video-MME LongVideoBench StreamingBench

Model

Qwen2.5-VL* 64.9 47.2 63.5 56.0 50.7 67.7

InfiniteVL-Base (wo LSFT) 62.8 46.4 53.3 45.3 42.6 64.9 InfiniteVL-Base 64.5 47.6 62.8 55.7 50.3 67.6 InfiniteVL-SWA 56.5 46.1 57.0 51.0 47.2 66.7

InfiniteVL-Online 58.0 46.5 59.0 52.1 48.0 68.0 InfiniteVL-Offline 64.1 47.4 62.5 55.3 50.1 67.6

- Table 2. Multimodal Long-Context Performance. Comprehensive evaluation of different architectural variants on high-resolution image understanding, offline long-video comprehension, and online streaming perception benchmarks.

###### (a) VideoMME Performance Comparison (b) LongVideoBench Performance Comparison

- Figure 4. Length Generalization on Long Videos. Performance scaling across different frame counts (up to 1,024 frames) on (a) Video-MME and (b) LongVideoBench. InfiniteVL-Offline perfectly matches the robust length extrapolation trajectory of the Hybrid and Transformer baseline.

To contextualize our performance, we also compare InfiniteVL-Base against leading 2B–4B VLMs: TinyLLaVA-3B [79], PaliGemma2-3B [20], Phi-3.5-Vision-4B [47], SmolVLM2-2B [44], InternVL2.5-4B [11], and the official Qwen2.5-VL3B [3]. For long-context tasks, we apply an identical long-sequence fine-tuning recipe to both Qwen2.5-VL* and InfiniteVLBase to establish rigorous comparative baselines.

###### 4.1.2 Benchmarks.

We assess foundational multimodal performance using VLMEvalKit [15] across 13 diverse public benchmarks covering general understanding, OCR, charts, documents, and reasoning: MME[16], MMStar[8], MMBenchtest en[40], SeedBenchIMG[29], ScienceQAval[51], RealworldQA[65], AI2Dw/o M[26], TextVQAval[53], OCRBench[41], ChartQAtest[45], DocVQAtest[46], MMMUval[73] and MathVista[42]. For extended-context evaluation, we target three demanding scenarios: 1) High-resolution images: Vstar [64], BLINK [18], and HRBench4K [61]; 2) Offline long-videos: Video-MME [17] and LongVideoBench [63]; and 3) Online streaming: StreamingBench [37].

#### 4.2 Main Results of InfiniteVL-Base

###### 4.2.1 Foundational Multimodal Performance.

As shown in Tab. 1(a-b), InfiniteVL-Base achieves consistent parity with our strictly controlled Transformer baseline (Qwen2.5VL*) and leading models of similar scale. Notably, it excels in visually intensive tasks (OCR, Chart, Document understanding) that historically challenge linear VLMs. This confirms that our hybrid design successfully leverages Gated DeltaNet for macroscopic compression while retaining a fraction of Full Attention for precise detail recall, ensuring robust general-purpose perception without task-specific inductive biases.

OOM 3.1× OOM

OOM

OOM

1.7× 1.6×

2.7×

[Figure 40]

[Figure 41]

###### (a) Decode Speed Comparison (b) Decode Memory Comparison

- Figure 5. Autoregressive Decoding Efficiency. Comparison of (a) per-token decode latency and (b) allocated memory. InfiniteVL-Base successfully extends the maximum decoding length to 3.1× that of the standard Qwen2.5-VL-3B. Furthermore, empowered by its strictly constant O(1) memory footprint, InfiniteVL-Online unlocks theoretically infinite decoding lengths.

- 4.2.2 Efficiency Analysis.

Evaluations are conducted on a single NVIDIA RTX 4090 with Flash Attention 2 [12]. Fig. 5 highlights the decoding advantage: while the official Qwen2.5-VL-3B hits OOM at 454K context due to KV cache accumulation, InfiniteVL-Base scales to 1,589K tokens before OOM, owing to Gated DeltaNet’s constant memory state. Furthermore, its per-token latency grows significantly slower, achieving a 1.7× speedup at 450K context. For prefilling at 256K context, InfiniteVL-Base delivers a 2.8× acceleration over Qwen2.5-VL-3B (Fig. 6(a)).

- 4.3 Scaling to Extended Contexts

- 4.3.1 Multimodal Long-context Performance.

We compare the performance of Qwen2.5-VL*, InfiniteVL-Base, InfiniteVL-SWA, InfiniteVL-Online, and InfiniteVL-Offline on high-resolution image understanding, offline long-video comprehension, and online streaming perception tasks after identical long-context fine-tuning. Qwen2.5-VL* and InfiniteVL-Base demonstrate comparable performance across multiple benchmarks. As shown in Tab. 2, while InfiniteVL-SWA suffers performance degradation due to its strictly restricted attention window, InfiniteVL-Online effectively mitigates this long-context degradation. Specifically, InfiniteVL-Online matches or even exceeds the performance of InfiniteVL-Base on the StreamingBench online video task. Meanwhile, InfiniteVL-Offline maintains strict performance parity with InfiniteVL-Base across all evaluated benchmarks. Furthermore, Fig. 4 provides a detailed comparison of the length generalization capabilities across these models. The evaluation is conducted on the VideoMME and LongVideoBench datasets, configured at 1 fps with 256 tokens per frame, and sampling from 8 up to 1024 frames (corresponding to context lengths ranging from 2K to 256K). The experimental results demonstrate that InfiniteVL-Online consistently exhibits significantly milder performance degradation than InfiniteVL-SWA as the sequence grows. Concurrently, InfiniteVL-Offline delivers equivalent performance to InfiniteVL-Base across all evaluated context lengths.

- 4.3.2 Offline and Online Video Understanding Efficiency.

While Sec. 4.2.2 demonstrates the InfiniteVL-Base’s efficiency gains over standard Transformer VLMs, InfiniteVL-Online and InfiniteVL-Offline further eliminate the computational and memory bottlenecks of the remaining Full Attention layers. As shown in Fig. 6(b), in online streaming scenarios, InfiniteVL-Online achieves a stable inference speed of 8 FPS. Crucially, its constant memory footprint enables the use of CUDA Graph acceleration, which propels the throughput to a real-time streaming rate of 25 FPS. Conversely, InfiniteVL-Base and Qwen2.5-VL* suffer from steadily decreasing FPS and eventual OOM errors due to their unconstrained KV cache growth. Fig. 6(a) shows that InfiniteVL-Offline drastically reduces the prefill computational load for offline processing. At a 256K input length, it demonstrates a 1.8× and 5× prefill speedup over InfiniteVL-Base and Qwen2.5-VL* respectively.

≈ 25 FPS

16×

5×

###### (a) Long Sequence Prefill Speed Comparison (b) Streaming Prefill Speed Comparison

- Figure 6. Long-Sequence Prefill and Streaming Efficiency. (a) Time-to-First-Token (TTFT) for offline prefilling, where InfiniteVLOffline achieves a 5× acceleration at 256K context. (b) Sustained throughput (FPS) in online streaming, where InfiniteVL-Online (with CUDA Graph) maintains a robust 25 FPS without OOM degradation.

General Text-rich Averages ↑

RealWorldQA

image

TextVQAChartQAtest OCRBenchDocVQAtest Gen-Avg Text-Avg

SeedBench

MMEMMStar

Settings

(a) Effect of Ratio

Ratio = 0 1986 44.4 60.5 68.5 71.0 73.1 72.6 74.2 63.2 72.7 Ratio = 1/8 2008 44.1 59.3 69.1 73.7 75.8 76.4 84.0 63.2 77.5 Ratio = 1/4 2010 45.0 60.0 70.6 76.0 77.5 78.6 87.8 64.2 80.0 Ratio = full attn 2008 45.1 59.8 70.5 76.2 77.6 78.8 88.1 63.9 80.2

###### (b) Effect of Architecture

Linear attention NAN NAN NAN NAN NAN NAN NAN NAN NAN NAN Mamba 1686 40.1 58.2 63.1 34.3 43.2 49.0 15.3 57.2 35.5 GLA 1712 40.3 58.5 64.3 35.0 46.6 51.0 19.0 57.9 37.9 Gated DeltaNet 1986 44.4 60.5 68.5 71.0 73.1 72.6 74.2 63.2 72.7

###### (c) Effect of Training Stages

None NAN NAN NAN NAN NAN NAN NAN NAN NAN NAN

- Stage 1 2010 45.0 60.0 70.6 76.0 77.5 78.6 87.8 64.0 80.0

- Stage 2 1713 39.6 54.1 65.7 73.4 73.0 74.6 84.1 57.0 76.3 Stage 1 + 2 2120 54.7 66.3 72.7 78.3 82.4 79.8 91.1 69.6 82.9

- Table 3. Ablation Studies. Quantitative analysis validating the design choices of InfiniteVL, including (a) the optimal mixing ratio of Full Attention, (b) the selection of the core linear modeling architecture, and (c) the necessity of the multi-stage training strategy.
- 4.4 Ablation Studies

We systematically analyze the design choices and training strategies of InfiniteVL-Base through ablation studies, focusing on the linear layer selection, mixing ratios, and the multi-stage training strategy (summarized in Tab. 3).

###### 4.4.1 Linear Layer Selection.

We evaluate the impact of different linear sequence modeling modules on VLM performance. Vanilla Linear Attention exhibits severe training instability [25] and fails to inherit the teacher model’s pre-trained knowledge. While models incor-

porating scalar gating mechanisms (e.g., Mamba [21] and GLA [70]) achieve convergence, they show limited performance on information-dense document and text understanding tasks. In contrast, Gated DeltaNet [69], with its highly efficient state compression mechanism, yields significant improvements in both training stability and downstream performance (particularly on DocVQA and OCRBench), proving that efficient state compression is crucial for precise visual tasks.

###### 4.4.2 Mixing Strategy.

We investigate the impact of the Full Attention layer ratio within the architecture. Even a minimal proportion of Attention layers significantly boosts performance on visual intensive benchmarks. When the ratio reaches 1:3 (one Attention layer for every three linear layers), the hybrid architecture achieves performance parity with the Transformer baseline across comprehensive benchmarks.

###### 4.4.3 Training Strategy.

Direct initialization of the linear layers fails to produce usable capabilities. Stage I (Distillation Pre-training) successfully imparts foundational conversational abilities. Following Stage II (Continuous SFT), the model surpasses the teacher model across multiple metrics, indicating superior generalization. Notably, skipping Stage I and training from scratch using only SFT yields extremely poor results, underscoring the absolute necessity of the distillation phase. Finally, as illustrated in Tab. 2, InfiniteVL-Base without long-context fine-tuning (InfiniteVL-Base (wo LSFT)) falls significantly behind its finetuned counterparts on multimodal long-context tasks, firmly validating the effectiveness of the Stage III fine-tuning for extended multimodal understanding.

#### 5 Conclusion

In this paper, we introduced InfiniteVL, a novel hybrid vision-language architecture designed to reconcile the conflict between precise visual perception and long-context efficiency. By strategically interleaving constant-memory Gated DeltaNet layers with a small fraction of Full Attention layers, InfiniteVL-Base effectively overcomes the detail-degradation issues inherent in linear models. To further scale to extremely long or continuous multimodal contexts, we proposed a seamless long-sequence architectural fine-tuning strategy. This yields two specialized variants: InfiniteVL-Offline for precise offline long-video comprehension, and InfiniteVL-Online for constant-memory online streaming. Experiments demonstrate that InfiniteVL matches the capabilities of strictly controlled Transformer baselines, whilst delivering a 5× prefill speedup at 256K context lengths and enabling stable, real-time streaming at 24 FPS. Ultimately, InfiniteVL paves a highly efficient and scalable path for deploying robust, continuous multimodal agents in real-world environments.

#### References

- [1] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895–4901, 2023. 4
- [2] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661,

2025. 5

- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 3, 4, 7, 8
- [4] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. arXiv preprint ARXIV.2410.24164. 2, 3
- [5] Aaron Blakeman, Aarti Basant, Abhinav Khattar, Adithya Renduchintala, Akhiad Bercovich, Aleksander Ficek, Alexis Bjorlin, Ali Taghibakhshi, Amala Sanjay Deshmukh, Ameya Sunil Mahabaleshwarkar, et al. Nemotron-h: A family of accurate and efficient hybrid mamba-transformer models. arXiv preprint arXiv:2504.03624, 2025. 2, 3
- [6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020. 3
- [7] Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025. 3

- [8] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37: 27056–27087, 2024. 8
- [9] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer, 2024. 3
- [10] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. In The Thirteenth International Conference on Learning Representations. 3
- [11] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 2, 3, 5, 7, 8
- [12] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023. 9
- [13] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv e-prints, pages arXiv–2409, 2024. 5
- [14] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. In International Conference on Machine Learning, pages 8469–8488. PMLR, 2023. 2, 3
- [15] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198–11201, 2024. 8
- [16] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. 8
- [17] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108–24118, 2025. 2, 8
- [18] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024. 8
- [19] Paolo Glorioso, Quentin Anthony, Yury Tokpanov, James Whittington, Jonathan Pilault, Adam Ibrahim, and Beren Millidge. Zamba: A compact 7b ssm hybrid model. arXiv preprint arXiv:2405.16712, 2024. 3
- [20] Google DeepMind and Google Research. Paligemma 2 — google ai for developers (model page). https://ai.google.dev/ gemma/docs/paligemma, 2025. Accessed 2025-11-12. 7, 8
- [21] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First conference on language modeling. 2, 3, 11
- [22] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025. 2
- [23] Haowen Hou, Peigen Zeng, Fei Ma, and Fei Richard Yu. Visualrwkv: Exploring recurrent neural networks for visual language models. In Proceedings of the 31st International Conference on Computational Linguistics, pages 10423–10434, 2025. 2
- [24] Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Senna: Bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313, 2024. 2, 3
- [25] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR, 2020. 2, 3, 10
- [26] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer, 2016. 8
- [27] Hugo Lauren¸con, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding vision-language models: insights and future directions., 2024. 5
- [28] Hugo Lauren¸con, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?, 2024. 5
- [29] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024. 8
- [30] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023. 2
- [31] Kunchang Li, Xinhao Li, Yi Wang, Yinan He, Yali Wang, Limin Wang, and Yu Qiao. Videomamba: State space model for efficient video understanding. In European conference on computer vision, pages 237–255. Springer, 2024. 2, 3

- [32] Yingyue Li, Bencheng Liao, Wenyu Liu, and Xinggang Wang. Matvlm: Hybrid mamba-transformer for efficient vision-language modeling. arXiv preprint arXiv:2503.13440, 2025. 2, 3
- [33] Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052, 2025. 2, 3
- [34] Bencheng Liao, Hongyuan Tao, Qian Zhang, Tianheng Cheng, Yingyue Li, Haoran Yin, Wenyu Liu, and Xinggang Wang. Multimodal mamba: Decoder-only multimodal state space model via quadratic to linear distillation. arXiv preprint arXiv:2502.13145, 2025. 2, 3
- [35] Bencheng Liao, Xinggang Wang, Lianghui Zhu, Qian Zhang, and Chang Huang. Vig: Linear-complexity visual sequence learning with gated linear attention. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 5182–5190, 2025. 2
- [36] Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, et al. Jamba: A hybrid transformer-mamba language model. arXiv preprint arXiv:2403.19887, 2024. 2, 3
- [37] Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. arXiv preprint arXiv:2411.03628, 2024. 2, 8
- [38] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26679–26689. IEEE, 2024. 2
- [39] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 2
- [40] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 8
- [41] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102,

2024. 8

- [42] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. CoRR,

2023. 8

- [43] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12585–12602, 2024. 3
- [44] Andr´es Marafioti, Orr Zohar, Miquel Farr´e, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Vaibhav Srivastav, Joshua Lochner, Hugo Larcher, Mathieu Morlon, Lewis Tunstall, Leandro von Werra, and Thomas Wolf. Smolvlm: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299, 2025. 7, 8
- [45] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263–2279,

2022. 8

- [46] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 8
- [47] Microsoft. Phi-3.5-vision-instruct (model card). https://huggingface.co/microsoft/Phi-3.5-visioninstruct, 2025. Lightweight multimodal model with 128K context; official model card. Accessed 2025-11-12. 7, 8
- [48] Andrea Pozzi, Alessandro Incremona, Daniele Tessera, and Daniele Toti. Mitigating exposure bias in large language model distillation: an imitation learning approach. Neural Computing and Applications, pages 1–17, 2025. 6
- [49] Yanyuan Qiao, Zheng Yu, Zijia Zhao, Sihan Chen, Mingzhen Sun, Longteng Guo, Qi Wu, and Jing Liu. Vl-mamba: Exploring state space models for multimodal learning. In NeurIPS Efficient Natural Language and Speech Processing Workshop, pages 102–113. PMLR, 2024. 2
- [50] Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, and Weizhu Chen. Samba: Simple hybrid state space models for efficient unlimited context language modeling. arXiv preprint arXiv:2406.07522, 2024. 2, 3
- [51] Tanik Saikh, Tirthankar Ghosal, Amish Mittal, Asif Ekbal, and Pushpak Bhattacharyya. Scienceqa: A novel resource for question answering on scholarly articles. International Journal on Digital Libraries, 23(3):289–301, 2022. 8
- [52] Imanol Schlag, Kazuki Irie, and J¨urgen Schmidhuber. Linear transformers are secretly fast weight programmers. In International conference on machine learning, pages 9355–9366. PMLR, 2021. 2
- [53] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326,

2019. 8

- [54] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024. 3
- [55] Enxin Song, Wenhao Chai, Shusheng Yang, Ethan Armand, Xiaojun Shan, Haiyang Xu, Jianwen Xie, and Zhuowen Tu. Videonsa: Native sparse attention scales video understanding. arXiv preprint arXiv:2510.02295, 2025. 3

- [56] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 4
- [57] Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, et al. Kimi linear: An expressive, efficient attention architecture. arXiv preprint arXiv:2510.26692, 2025. 3
- [58] MiniCPM Team, Wenhao An, Yingfa Chen, Yewei Fang, Jiayi Li, Xin Li, Yaohui Li, Yishan Li, Yuxuan Li, Biyuan Lin, et al. Minicpm-sala: Hybridizing sparse and linear attention for efficient long-context modeling. arXiv preprint arXiv:2602.11761, 2026. 3
- [59] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971,

2023. 3

- [60] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2
- [61] Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7907–7915, 2025. 8
- [62] Luis Wiedmann, Orr Zohar, Amir Mahla, Xiaohan Wang, Rui Li, Thibaud Frere, Leandro von Werra, Aritra Roy Gosthipaty, and Andr´es Marafioti. Finevision: Open data is all you need. arXiv preprint arXiv:2510.17269, 2025. 5
- [63] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024. 2, 8
- [64] Penghao Wu and Saining Xie. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13084–13094, 2024. 8
- [65] xAI. Realworldqa, 2024. 8
- [66] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations. 7
- [67] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023. 2
- [68] Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. Streamingvlm: Real-time understanding for infinite video streams. arXiv preprint arXiv:2510.09608, 2025. 7
- [69] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations, . 2, 3, 4, 11
- [70] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardwareefficient training. In Forty-first International Conference on Machine Learning, . 3, 11
- [71] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. Advances in neural information processing systems, 37:115491–115522, 2024. 2
- [72] Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, et al. Cambrian-s: Towards spatial supersensing in video. arXiv preprint arXiv:2511.04670, 2025. 3
- [73] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024. 8
- [74] Lunbin Zeng, Jingfeng Yao, Bencheng Liao, Hongyuan Tao, Wenyu Liu, and Xinggang Wang. Diffusionvl: Translating any autoregressive models into diffusion vision language models. arXiv preprint arXiv:2512.15713, 2025. 2
- [75] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 7
- [76] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher R´e, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36:34661–34710, 2023. 3
- [77] Han Zhao, Min Zhang, Wei Zhao, Pengxiang Ding, Siteng Huang, and Donglin Wang. Cobra: Extending mamba to multi-modal large language model for efficient inference. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 10421–10429,

2025. 2, 3

- [78] Weilin Zhao, Zihan Zhou, Zhou Su, Chaojun Xiao, Yuxuan Li, Yanghao Li, Yudi Zhang, Weilun Zhao, Zhen Li, Yuxiang Huang, et al. Infllm-v2: Dense-sparse switchable attention for seamless short-to-long adaptation. arXiv preprint arXiv:2509.24663, 2025. 6
- [79] Baichuan Zhou, Ying Hu, Xi Weng, Junlong Jia, Jie Luo, Xien Liu, Ji Wu, and Lei Huang. Tinyllava: A framework of small-scale large multimodal models. arXiv preprint arXiv:2402.14289, 2024. 7, 8
- [80] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations. 2
- [81] Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, and Xinggang Wang. Dig: Scalable and efficient diffusion models with gated linear attention. arXiv, 2024. 2

- [82] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023. 2, 3
- [83] Jialv Zou, Bencheng Liao, Qian Zhang, Wenyu Liu, and Xinggang Wang. Omnimamba: Efficient and unified multimodal understanding and generation via state space models. arXiv preprint arXiv:2503.08686, 2025. 2

# Appendix

In this part, we provide additional details about InfiniteVL, which are omitted due to the 14 page limit of the main paper. Specifically, Section 6 analyzes the cache evolution of linear layers in InfiniteVL under long-term streaming inputs, illustrating its length generalization capability. Section 7 supplements comprehensive training and evaluation setup details to facilitate the reproduction of our experimental results. Section 8 presents a series of visualization cases demonstrating InfiniteVL’s performance in general VQA, text-rich understanding, and ultra-long streaming understanding scenarios.

#### 6 Analysis of Cache Evolution under Streaming Inputs

In the linear layers, each output is computed by multiplying the query with a memory cache formed through the cumulative outer product of keys and values. This memory cache, a matrix of size 16×128×256, serves as one of the core components influencing the output, and its stability directly affects output stability. To long-term behavior in streaming scenarios, we track the evolution of the L2 norm of this memory cache as the number of input frames increases, using it as a metric for long-term stability.

[Figure 42]

- Figure 7. L2 norm of the Linear-layer memory cache versus input frame index: the norm increases rapidly at the beginning and then stabilizes.

As shown in Fig. 7, the norm of the memory cache rises rapidly during the initial frames and then produce convergence, demonstrating stable behavior without exhibiting unbounded growth. This indicates that the cache management mechanism remains effective and that the linear layers maintain stable comprehension capability when processing unlimited streaming input sequences.

#### 7 Comprehensive Training and Evaluation Setup Details

All trainings were conducted on NVIDIA H20 GPUs using bfloat16 (BF16) precision. The training corpus comprises exclusively publicly available, open-source datasets totaling ∼10M samples. For each stage, we uniformly subsample the required portion of data. We use AdamW with β = (0.9,0.999) and weight decay 0.01, along with a cosine-annealing learning-rate schedule and a 5% linear warm-up.

In Stage I, we use a learning rate of 2 × 10−4, batch size of 64, with both layer-wise and end-to-end distillation trained on 1M multimodal Caption & QA pairs each. Stage II employs a learning rate of 5 × 10−5, batch size of 256, and trains on

- 8M multimodal QA pairs. For Stage III, the configuration includes a learning rate of 2 × 10−5, batch size of 64, and training on 0.25M multimodal long-sequence Caption & QA pairs together with 0.75M multimodal SFT data from Stage II’s training corpora to prevent excessive distribution shift.

We adopt VLMEvalKit as our multimodal evaluation framework and utilize its default prompt configurations for all benchmarks. During evaluation, the maximum generation length for InfiniteVL is set to 256 tokens, with image resolutions ranging from 128 × 128 to 1344 × 1344. By prepending all visual tokens to the prompt, we force the model to compress the full image without any textual priors, strictly validating InfiniteVL’s highly efficient and lossless-like compression. For specialized tasks such as MMMU and MathVista-MINI, we employ Qwen3-4B-Instruct as the evaluation model to ensure consistent and reliable assessment of reasoning capabilities.

#### 8 Case Study across multiple scenarios

We present several examples to demonstrate the capabilities of InfiniteVL. As shown in Figure 8, InfiniteVL exhibits strong comprehension across high-resolution, text-intensive, and complex structural scenarios. Beyond fundamental visionlanguage understanding, we further investigate the model’s performance in long-term streaming scenarios. Using two extended videos from street-view recordings and dashcam footage, we sample frames at 1 FPS and continuously feed them to InfiniteVL to update its memory cache, while posing questions at randomly selected timestamps. As shown in Figures 9 and 10, InfiniteVL maintains stable comprehension and reasoning capabilities even under extremely long contexts exceeding 512K tokens.

InfiniteVL: ‘The image depicts an outdoor scene in what appears to be a public park or plaza. The setting is characterized by a mix of natural and urban elements, with several people engaged in various activities.\n\nIn the foreground, there is a man seated on a stone bench, playing an electric guitar. He is dressed casually in a mustard-colored jacket, blue jeans, and dark shoes. His focus is entirely on his instrument, suggesting he is performing for passersby. A black amplifier is placed beside him on the ground, indicating that he might be using it to amplify his sound.\n\nTo the left of the guitarist, there is another individual walking away from the camera. This person is wearing a black T-shirt, red shorts, and sandals, and appears to be carrying a backpack. The attire suggests a casual, possibly warm day.\n\nFurther back, there are several other individuals sitting on benches, some of whom are also engaged in their own activities such as reading or looking at their phones. The benches are made of stone and are arranged in a semicircular fashion around a central area where the guitarist is seated.\n\nBehind the benches, there is a large, ornate fountain with a pedestal and a basin at the top. The fountain is surrounded by lush green trees, adding a touch of nature to the urban environment. The trees have vibrant foliage, indicating that the season could be spring or summer.\n\nIn the background, there are more people walking or standing, contributing to the lively atmosphere of the scene. Some of these individuals are also carrying bags or backpacks, suggesting they might be tourists or locals out for leisure. …

User: Please describe the content in the

image in detail.

[Figure 43]

User: Extract all text content from this image.

InfiniteVL: 'Mr. Stephen Schmidt, Editor\nNutrition Action Healthletter\nCenter for Science in the Public Interest\n1875 Connecticut Avenue, N.W., Suite 300\nWashington, DC 20009\n\nDear Mr. Schmidt:\n\nThe review of artificial sweeteners in “Sweet Nothings,” Nutrition Action Healthletter, May 2004, provides some helpful information for readers but causes two areas of concern.\n\nFirst, the article pictures a glass of Coca-Cola in the section on saccharin, mistakenly indicating that the beverage contains this sweetener. However, CocaCola does not contain saccharin, and readers should be informed of this error.\n\nSecond, discussions of the safety of saccharin and acesulfame potassium present a selective and biased review of the data raising unsubstantiated fears among consumers. According to the National Cancer Institute, “There is no evidence that the regulated artificial sweeteners on the market in the United States are related to cancer risk in humans.” The U.S. National Toxicology Program has removed saccharin from the list of established human carcinogens. And the World Health Organization’s International Agency for Research on Cancer confirmed that the evidence is inadequate to link saccharin with human cancer. More than 90 countries approve the use of saccharin.\n\nRegarding acesulfame potassium, about 90 scientific studies, versus three mentioned in the story, have thoroughly established the safety of acesulfame potassium. These studies are the basis for a number of government authorities, including the U.S. Food and Drug Administration and the Commission of the European Communities, to conclude that acesulfame potassium is safe. The American Dietetic Association has also affirmed the safety of acesulfame potassium, even during pregnancy.\n\nMany individuals count on CSPI for nutrition and health advice. But when the scientific data and consensus of experts overwhelmingly differ from that reported in the Healthletter, readers are misinformed.\n\nSincerely,\n\nBarbara L. Beck, M.S., R.D.\nPrincipal Manager, Scientific and Regulatory Affairs'

[Figure 44]

User: Please interpret this image.

InfiniteVL: "The image is a pie chart that shows the percentage of STEM-trained college graduates who are employed in different types of jobs. The chart indicates that 52% of STEMtrained college graduates are currently employed in the STEM workforce, while 17% are employed in management, business, and finance, 11% are employed in social services, legal, and education, and 20% are employed in other non-STEM fields. The chart also includes a note stating that the data is based on employed adults ages 25 and older completing a bachelor's degree in STEM major field of study. The source of the data is the Pew Research Center Demographic Analysis of the American Community Survey (IPUMS). The chart also mentions that women and men in STEM often face challenges in the workplace equity."

[Figure 45]

Figure 8. Examples of fundamental Visual-Language Understanding

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

User: Summarize the main events that have occurred in the video so far? InfiniteVL: A man is walking down the street, passing by various shops and people. A woman is also seen crossing the street at an intersection with traffic lights. The scene changes to show a busy city street with many pedestrians and vehicles moving around. People are seen walking on sidewalks, crossing streets, and riding bicycles. There are also several cars driving through the area. Overall, it appears to be a bustling urban environment with lots of activity happening all around.

Frames: 210 Timestamp: 3:30 Cum Tokens: ~62.8K

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

User: Describe the people in the scene and what they are doing. InfiniteVL: The image shows a bustling urban street scene with several individuals walking along the sidewalk, engaged in various activities such as talking to each other or looking at their phones. The presence of multiple pedestrians suggests that this is a popular area for both locals and tourists alike. In addition to the people, there are also some notable features visible on the buildings lining the street. These include large billboards displaying advertisements, which contribute to the vibrant atmosphere typical of city centers. There's also a red backpack worn by one individual, adding another layer of detail to the everyday life captured in the

Frames: 1020 Timestamp: 17:00 Cum Tokens: ~305K

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Frames: 1680 Timestamp: 28:00 Cum Tokens: ~502K

User: What is the main text appears in the current scene? InfiniteVL: NBC STUDIOS

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

User: What are the most salient objects or elements in the scene right now?

InfiniteVL: The image depicts a large, ornate cathedral with intricate Gothic architectural details and multiple spires reaching towards the sky. The cathedral is adorned with stained glass windows that allow natural light to filter through, illuminating the interior space. In front of the cathedral stands a statue of a religious figure, possibly a saint, which adds to the grandeur of the structure. In addition to the cathedral itself, there are several other buildings visible in the background, suggesting an urban setting.

Frames: 2120 Timestamp: 35:20 Cum Tokens: ~634K

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

User: What is currently happening in the scene right now? InfiniteVL: A person wearing dark clothing and carrying a green bag walks down a city street, passing by various storefronts and pedestrians. The environment appears to be urban with tall buildings on either side of the street.

Frames: 3100 Timestamp: 51:40 Cum Tokens: ~927K

Figure 9. Examples of Long-Term Streaming Understanding Capability

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

User: Describe what is happening on the road right now. InfiniteVL: The scene shows a busy highway with multiple lanes of traffic moving in both directions under an overpass or bridge structure. There are several cars and trucks visible, some traveling at high speeds while others appear to be slowing down or waiting for signals. The scene suggests that this might be during peak travel times when there's heavy traffic due to commuting hours. In terms of specific details:

Frames: 540 Timestamp: 9:00 Cum Tokens: ~161K

- - Traffic lights can't be seen clearly from this angle.
- - Road signs are not distinctly readable without closer inspection.
- - The sky appears clear which could indicate good weather conditions but doesn’t provide any additional context about time of day or other environmental factors affecting traffic flow. Given these observations, it seems like a typical scenario one would encounter on a major roadway during rush hour. However, without more information such as timestamps, speed limits, or signage indicating current events (like construction zones), we cannot make detailed predictions about future traffic patterns based solely on this snapshot.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

User: What potential hazards do you see in the current scene? InfiniteVL: In this image, there is a busy street with multiple vehicles and pedestrians. Potential hazards include:

- 1. **Vehicle Traffic**: The presence of numerous cars suggests that traffic could be heavy at any given time.
- 2. **Pedestrian Safety**: There appears to be a significant number of people crossing the street or walking along the sidewalk. This can lead to accidents if drivers are not attentive.
- 3. **Crosswalk Hazards**: Some crosswalks seem less well-defined than others, which might pose risks for both drivers and pedestrians who may misjudge distances.

Frames: 1945 Timestamp: 32:25 Cum Tokens: ~582K

These elements highlight areas where safety measures such as improved signage, better road markings, and increased police patrols should be considered to mitigate these hazards.

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

User: What traffic signs, signals, or road markings are currently visible? InfiniteVL: The image shows several traffic signs and signals that are commonly found in urban areas:

Frames: 2305 Timestamp: 38:25 Cum Tokens: ~689K

- 1. **Traffic Lights**: There is at least one red light visible on the left side of the image.
- 2. **Crosswalk Markings**: The white crosswalk lines indicate where pedestrians should wait to cross the street safely.

Additionally, there are no visible road markings such as lane dividers or parking restrictions from this angle. However, these elements would typically be present but not clearly discernible due to the resolution and lighting conditions of the photograph.

Figure 10. Examples of Long-Term Streaming Understanding Capability

