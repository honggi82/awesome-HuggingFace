## arXiv:2601.21420v1[cs.LG]29Jan2026

[Figure 1]

### ConceptMoE: Adaptive Token-to-Concept Compression for Implicit Compute Allocation

Zihao Huang∗, Jundong Zhou∗, Xingwei Qu∗, Qiyang Min∗, Ge Zhang† ByteDance Seed ∗Main authors, †Corresponding authors

Abstract

Large language models allocate uniform computation across all tokens, ignoring that some sequences are trivially predictable while others require deep reasoning. We introduce ConceptMoE, which dynamically merges semantically similar tokens into concept representations, performing implicit token-level compute allocation. A learnable chunk module identifies optimal boundaries by measuring inter-token similarity, compressing sequences by a target ratio R before they enter the compute-intensive concept model. Crucially, the MoE architecture enables controlled evaluation: we reallocate saved computation to match baseline activated FLOPs (excluding attention map computation) and total parameters, isolating genuine architectural benefits. Under these conditions, ConceptMoE consistently outperforms standard MoE across language and vision-language tasks, achieving +0.9 points on language pretraining, +2.3 points on long context understanding, and +0.6 points on multimodal benchmarks. When converting pretrained MoE during continual training with layer looping, gains reach +5.5 points, demonstrating practical applicability. Beyond performance, ConceptMoE reduces attention computation by up to R2× and KV cache by R×. At R = 2, empirical measurements show prefill speedups reaching 175% and decoding speedups up to 117% on long sequences. The minimal architectural modifications enable straightforward integration into existing MoE, demonstrating that adaptive concept-level processing fundamentally improves both effectiveness and efficiency of large language models.

Date: January 30, 2026 Correspondence: huangzihao.notabot@bytedance.com Code Page: https://github.com/ZihaoHuang-notabot/ConceptMoE

#### 1 Introduction

Large language models (LLMs) process text uniformly at the token level, allocating equal computation to every position in the sequence. Yet not all tokens carry equal semantic weight: while some represent pivotal concepts requiring deep reasoning, others are trivially predictable from context. This one-size-fits-all approach wastes substantial computation on routine predictions while potentially under-resourcing semantically dense content. The natural question emerges: can we move beyond fixed token-level processing toward adaptive concept-level computation?

Traditional approaches to enriching semantic content while reducing token count focus on vocabulary expansion. Recent work [33] shows that larger vocabularies consistently improve LLM performance by compressing text into fewer, information-denser tokens. However, the gains are limited: a 100× vocabulary expansion yields

only 1.3× compression. The computational cost of massive vocabularies during training and inference makes further scaling impractical.

An alternative paradigm is dynamic token merging within the model. Several studies [7, 30] merge consecutive tokens into higher-level concepts without vocabulary expansion, but use fixed-length or rule-based strategies that cannot adapt to varying information density. Byte-level transformers [31, 36, 40] explore adaptive chunking [19, 26], yet lack precise parameter control and introduce confounding factors through input representation changes. Moreover, most evaluations on dense models compensate reduced token count by scaling depth or width, conflating architectural benefits with parameter increases.

We introduce ConceptMoE, which elevates LLM processing from the token level to the concept level through learnable adaptive chunking. The core insight is straightforward: consecutive tokens with high semantic similarity merge into unified concept representations, while semantically distinct tokens maintain fine-grained granularity. A learnable chunk module identifies optimal boundaries by measuring inter-token similarity, compressing sequences before they enter the compute-intensive concept model. This naturally performs implicit compute allocation - predictable token sequences merge and process efficiently, while complex tokens preserve detailed computation. Crucially, the MoE architecture enables controlled evaluation: by reallocating saved computation to match baseline FLOPs while maintaining identical total parameters, we isolate genuine architectural benefits from concept-level processing.

Our key contributions are:

- • Fair architectural comparison under controlled conditions. Unlike dense models, MoE allows adjusting activated parameters independently of total parameters. By reallocating computation saved from token reduction, we compare ConceptMoE against standard MoE under identical total parameters and average per-token FLOPs, isolating genuine architectural benefits. We explore three compute reallocation strategies and demonstrate consistent improvements across all configurations.
- • Versatile application across training paradigms. We demonstrate ConceptMoE’s effectiveness across diverse settings: small-scale language model pretraining (Section 4.1), vision-language model training with dual-modality compression (Section 4.2), and lossless conversion from pretrained MoE during continual training (Section 4.3). The CT conversion achieves +5.5 points improvement while enabling direct inference speedup, with from-scratch training reaching +6.4 points overall.
- • Inherent efficiency advantages under compute matching. Even after reallocating saved FLOPs, given compression ratio R, ConceptMoE reduces attention map computation by up to R2× and KV cache by up to R× (Table 1). Empirical results show prefill speedups up to 175% and decoding speedups up to 117% on long sequences under R = 2, validating both superior performance and efficiency.
- • Minimal architectural intrusion for practical adoption. ConceptMoE requires only a lightweight chunk module and minor decoder modifications (additional QKV projectors in the last 4 layers), making it straightforward to integrate into existing MoE architectures for both pretraining and continual training scenarios.

ConceptMoE demonstrates a paradigm shift from uniform token-level to adaptive concept-level processing, fundamentally improving both effectiveness and efficiency of LLMs.

#### 2 Related Work

Vocabulary size determines the information capacity of each token in LLMs, typically ranging from 32K to 256K. Larger vocabularies enable higher compression efficiency and fewer tokens per sequence. Tao et al. [34] demonstrate that vocabulary size should scale with model parameters, while Takase et al. [33] show that increasing vocabulary from 5K to 500K achieves a compression ratio of 1.3, yielding substantial downstream improvements when training on fixed data or token budgets. However, a 100× vocabulary expansion produces only 1.3× compression, revealing that further compression gains require exponential vocabulary growth. Moreover, excessively large vocabularies become inference bottlenecks [3, 24]. This motivates an alternative approach: dynamic compression within the model itself, which forms the foundation of ConceptMoE.

Token-level chunking compresses tokens internally within the model. Existing approaches include fixed-length merging [7, 30] and heuristic or rule-based compression [1, 12]. These methods merge multiple tokens into single concepts to achieve higher compression ratios without expanding vocabulary size, but lack adaptive compression strategies. Token information density varies significantly: information-sparse tokens should merge aggressively, while information-rich tokens should maintain finer granularity. Recent concurrent work DLCM[28] introduces dynamic compression to merge tokens into concepts, but when comparing against FLOPs-matched baselines, the model parameters actually double, undermining the fairness of ablation studies. ConceptMoE addresses these through a learnable chunk module that adaptively identifies optimal merge boundaries based on semantic similarity. Crucially, we leverage MoE properties to ensure all comparisons maintain identical total parameters and average per-token FLOPs, enabling rigorous ablation studies.

Byte-level models require more sophisticated merging strategies due to the granularity of byte tokens, which necessitates aggressive compression to maintain manageable computational costs. AU-Net [36] introduces multi-level compression but relies on rule-based strategies. BLT [26] leverages a pretrained auxiliary model to compute token entropy, merging low-entropy tokens more aggressively. However, this non-end-to-end approach may encounter difficulties during downstream fine-tuning. H-Net [19] proposes an end-to-end dynamic chunking module, achieving 9× compression at the byte level. While this represents approximately 2× compression at the token level, H-Net’s experiments control only FLOPs while allowing total parameters to vary, introducing confounding factors. Additionally, the byte-level input representation itself constitutes an experimental variable. ConceptMoE builds on insights from H-Net but operates at the token level with higher compression efficiency and provides fair comparison by controlling both FLOPs and total parameters. We validate effectiveness and efficiency at significantly larger model scales across diverse training scenarios, including language-only pretraining, vision-language training, and continual training from pretrained checkpoints.

#### 3 Approach

##### 3.1 Overview

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

^

^

𝒄^

+𝒉

𝒉

𝒉 𝒉 𝒉 𝒉 𝒉 𝒉 𝒉

𝒄

𝒉 𝒉 𝒉 𝒉 𝒉 𝒉 𝒉

^

^

+𝒉

𝒉

^

^

𝒄^

+𝒉

𝒄

𝒉

[Figure 7]

chunk module

^

^

[Figure 8]

[Figure 9]

[Figure 10]

+𝒉

𝒉

encoder concept model decoder

^

^

[Figure 11]

+𝒉

𝒉

dechunk module

^

^

𝒄^

+𝒉

𝒉

𝒄

^

^

+𝒉

𝒉

|[Figure 12]<br><br>𝒄^ 𝒄^<br><br>𝒄^<br><br>𝒄^<br><br>𝒄^<br><br>𝒄^<br><br>𝑝<br><br>𝑝<br><br>1 − 𝑝<br><br>1 − 𝑝<br><br>𝑝<br><br>+𝒉<br><br>^<br><br>+𝒉<br><br>^<br><br>+𝒉<br><br>^<br><br>+𝒉<br><br>^<br><br>+𝒉<br><br>^<br><br>+𝒉<br><br>^<br><br>+𝒉<br><br>^|
|---|

|[Figure 13]<br><br>q<br><br>k<br><br>sim_func<br><br>q<br><br>k<br><br>sim_func<br><br>q<br><br>k<br><br>sim_func<br><br>𝑝 = 1.0<br><br>𝑝 = 0.2<br><br>𝑝 = 0.9<br><br>𝑝 = 0.1<br><br>𝒄 = 𝒉<br><br>^<br><br>𝒄 = 𝒉<br><br>^<br><br>+ 𝒉<br><br>^<br><br>or 𝒉<br><br>^<br><br>q query linear layer k key linear layer sim_func similarity-based function<br><br>𝒉<br><br>^<br><br>𝒉<br><br>^<br><br>𝒉<br><br>^<br><br>𝒉<br><br>^|
|---|

Figure 1 Overview of ConceptMoE, with details of chunk and dechunk modules.

###### Just like byte level models, our ConceptMoE is also composed of 5 modules, namely the encoder E, chunk

module Chunk, concept model C, dechunk module DeChunk, and decoder D, as shown in Figure 1. E, C and D are all composed of multiple stacked MoE modules. Given a sequence of input hidden state H = {h1,h2,...,hn,...,hN}, hn ∈ Rd,

Hˆ = E(H), C,P = Chunk(Hˆ ), Cˆ = C(C), Z = DeChunk(Cˆ,P),

(1)

Zˆ = D(Cˆ,Z),

where C = {c1,c2,...,cm,...,cM}, cm ∈ Rd1 is the concept embeddings, M ≤ N, Hˆ , Z, Zˆ and H have the same dimension. P = {p1,p2,...,pn,...,pN} is the probability that the token is a chunk boundary. Note that in the D input, each token is guaranteed to have an associated concept for joint decoding. This significantly enhances model capability by ensuring that information in the concept is fully utilized across multiple subsequent tokens.

In general, the computational proportion of E and D is relatively small, and the main FLOPs come from the C. The Chunk will select to merge multiple consecutive hˆ into single c to enter C. We regard the merged multiple tokens as a concept, hence the intermediate model is called the concept model. We provide PyTorch-like code in the Appendix B for better understanding.

##### 3.2 Chunk module

The chunk module aims to identify optimal chunk boundaries for a given input token sequence under a specified compression ratio. Similar to H-Net[19], it selects which consecutive tokens are easily predictable and merges them into chunks, while keeping hard-to-predict tokens unmerged or merged into smaller chunks. This approach performs implicit token-level compute allocation, in contrast to explicit compute allocation methods like MoE (e.g., Zero Expert[21]) that activate different amounts of computation for each token.

Specifically, for the input hn, we calculate the cosine similarity between the embeddings of two adjacent tokens after linear transformation to determine if this token serves as a chunk boundary:

qn⊤kn−1 ∥ qn ∥ · ∥ kn−1 ∥

- 1

- 2

n≥0.5, (2)

), bn = 1p

(1 −

qn = Wqhn, kn = Wkhn, pn =

where Wq,Wk ∈ Rd×d are learnable parameters, and we set p1 = 1.0 to ensure that the first token in the decoder D always has a concept for joint decoding. If the probability pn exceeds 0.5, the token is identified as a chunk boundary. This design is natural and intuitive: when several consecutive tokens exhibit high similarity but the current token shows low similarity to its predecessor, it indicates a significant semantic shift. Such tokens typically carry substantially different information and require more careful processing.

Auxiliary loss Given a target compression ratio, we introduce an auxiliary loss to constrain the token compression ratio on the training set, where the compression ratio is defined as R = N/M ≥ 1. Inspired by the load balancing loss in MoE, we treat boundary and non-boundary selections as two experts and constrain their activation frequencies to achieve the target compression ratio. Let the average probabilities and average selection ratios for boundary and non-boundary be:

1 N

G1 =

N

n

1 N

pn, G2 =

N

n

1 N

1 − pn = 1 − G1, F1 =

N

n

1 N

bn, F2 =

N

n

1 − bn = 1 − F1, (3)

then the auxiliary loss is

R R − 1

R R − 1

((R − 1)F1G1 + (1 − F1)(1 − G1)). (4)

Laux = RF1G1 +

F2G2 =

1For the sake of simplicity, the dimensions of c and h are kept consistent here, both being d; in fact, the dimension of c can be larger than that of h.

Note that when computing G1 and F1, we aggregate statistics across all samples in the current devices rather than averaging per-sample statistics. This enables sample-level compute allocation: when a batch contains both difficult and easy samples, the model can reduce the compression ratio for difficult samples while increasing it for easy samples. In practical use, we use λ to control the weight of this auxiliary loss.

Random flip boundary When we constrain the compression ratio on the training set, empirical observations show that distribution shifts in the evaluation set can lead to excessively high compression ratio, which speeds up inference but significantly degrades model performance. We address this by adding random perturbations to boundaries during training to simulate such scenarios, thereby mitigating the over-compression issue caused by distribution shifts at inference time. Specifically, after computing pn, we sharpen the probability:

1

nτ pn ≥ 0.5 1 − (1 − pn)τ1 pn < 0.5

psharpn = p

(5)

Where τ is a hyperparameter to control sharpness. The final boundary is obtained by sampling from a Bernoulli distribution parameterized by psharpn :

psharpn if k = 1 1 − psharpn if k = 0

btrainn ∼ Bernoulli(psharpn ), btrainn ∈ {0,1}, Pr(btrainn = k) =

(6)

The sharpened probability distribution is more likely to flip boundaries with low confidence (close to 0.5) during sampling, while probabilities with high confidence (close to 0 or 1) are less likely to be flipped. This ensures stable training convergence.

Merging strategy We can either sum all tokens within a chunk to obtain the concept, which maximally preserves the information of each token, or use only the last token as the concept, since the self-attention mechanism in the encoder E enables the last token to already aggregate the information of the entire chunk.

##### 3.3 Dechunk module

The dechunk module remaps concepts back to tokens and ensures that no information leakage occurs. Before dechunk, we first apply exponential moving average (EMA) on concepts. Let I = {n | bn = 1,1 ≤ n ≤ N} with |I| = M. Define the index mapping

ϕ : {1,2,...,m,...,M} → I, ϕ(m) = nm, (7) then for each cˆm, there is

cˆemam = pϕ(m)cˆm + (1 − pϕ(m))cˆm−1. (8) This mechanism is illustrated in the green part of Figure 1. The EMA accelerates chunking convergence through the following process: consider an example where "Simple and easy-to-" and "understand picture" are initially split into two concepts, with p = 0.5 for "understand picture". Through EMA, if the model discovers that the concept of "Simple and easy-to-" effectively aids in predicting tokens within "understand picture", it will reduce the boundary probability p for "understand picture". Once p falls below 0.5, the boundary is eliminated, merging "Simple and easy-to-" and "understand picture" into a unified concept.

After EMA, based on the given index mapping, dechunk the concept to the token-level embedding. Specifically, we define the mapping from concept index to token index:

ψ : {1,2,...,n,...,N} → {1,2,...,m...,M}, ψ(n) = m if ϕ(m) ≤ n < ϕ(m + 1). (9) Finally we can obtain the D input zn as

zn = hˆn + cˆemaψ(n). (10)

##### 3.4 Joint decoding

During token decoding, multiple tokens share the same concept in the dechunking process. To fully exploit the information in the concept, which accounts for most of the model’s computation and contains rich information, we perform joint decoding of zn and cˆemaψ(n). Specifically, in each self-attention layer of the decoder D, we have:

Attention(zn,cˆemaψ(n)) = softmax

(znWq + cˆemaψ(n)Wqc)(znWk + cˆemaψ(n)Wkc)T √dhead

+ M (znWv + cˆemaψ(n)Wvc), (11)

where M is the causal attention mask with Mij = −∞ · 1i<j. The highlighted terms show how we augment standard attention by incorporating concept information into the query, key, and value. Given the shallow depth of decoder D, the additional parameters are negligible while yielding substantial performance gains. Moreover, this design maintains compatibility with existing architectures, enabling straightforward application to continue training (CT) from pretrained models.

##### 3.5 Compute reallocation strategies for fair comparison

Given a standard MoE model, we decompose it into encoder, concept model, and decoder with layer depths LE, LC, and LD respectively. Temporarily disregarding attention map computation, let Cattn and Cmoe denote the FLOPs per token in self-attention and MoE layers. The concept model incurs LC(Cattn + Cmoe) FLOPs per token. With the chunk module at compression ratio R, the encoder outputs R tokens on average before the concept model processes one token, reducing its computation to LC(Cattn+Cmoe)

R (an R-fold reduction). This freed computation can be reallocated by increasing LC, Cattn, or Cmoe to maintain total FLOPs. Unlike dense architectures, MoE allows adjusting activated parameters while fixing total parameters. This property enables a fair comparison: under identical total parameters and per-token FLOPs, we can isolate the true gains of ConceptMoE over standard MoE by varying only the activated parameters. This paper explores three parameter allocation schemes:

- 1. Increasing Cmoe: We match the compute by increasing the number of activated experts in the MoE layers. This is the simplest approach and can be applied to both pretraining and continual training.
- 2. Increasing LC and Cmoe: Building on the previous approach, we additionally increase LC by looping through intermediate layers. This method is CT-friendly and introduces no additional parameters.
- 3. Increasing Cattn and Cmoe: We scale up both self-attention and MoE computation while keeping total parameters fixed. This is achieved by proportionally enlarging the hidden size of LC while reducing the total number of MoE experts. This requires two additional linear projectors for the mappings hˆ → c and cˆ → z. This approach is less suitable for CT and better suited for pretraining.

We now quantify the reduction in attention map computation and kv cache for the three schemes, which represents an inherent advantage of ConceptMoE. In standard MoE, the attention map computation for the concept model is LCdN2, where d denotes the hidden size and N the sequence length. Assume using multi-head attention, the kv cache is 2LCdN. The corresponding computations and kv cache for the three schemes are shown in Table 1, where Lloop is the number of loop layers. Any of these schemes can significantly reduce the computation of the attention map and the kv cache.

Table 1 Attention map computation and KV cache comparison

Method Attn Map FLOPs Reduction KV Cache Reduction

Baseline (MoE) LCdN2 1× 2LCdN 1× Increasing Cmoe LCdN2

R2 R2× 2L

CdN R R×

2 R2

R2LC

Increasing LC and Cmoe (LC+Lloop)dN

LC+Lloop × 2(LC+LRloop)dN RL

C LC+Lloop× Increasing Cattn and Cmoe LCdN2

√

R1.5 R1.5× 2L

√CdN R

R×

#### 4 Experiments

We conduct comprehensive experiments to validate ConceptMoE’s effectiveness and efficiency across diverse settings. Our evaluation encompasses four main aspects: (1) Small-scale language model pretraining (Section 4.1) at 12B and 24B parameters to establish core benefits under controlled conditions. (2) Vision-language model training (Section 4.2) at 60B parameters, demonstrating dual-modality compression and achieving strong gains on long context tasks. (3) Continual training conversion (Section 4.3) from pretrained MoE at 90B parameters, validating practical deployment through lossless integration. (4) Inference speedup analysis (Section 4.4) on 300B parameters models, measuring actual latency improvements across diverse compression ratios and layer configurations. Additionally, we perform extensive ablation studies examining auxiliary loss weight, chunking strategies, router design, joint decoding, boundary noise, and target compression ratios. All experiments maintain identical total parameters and per-token FLOPs (excluding attention maps) between ConceptMoE and MoE, ensuring fair architectural comparison.

Common model configuration The MoE baseline activates 8 experts. ConceptMoE uses an auxiliary loss weight of λ = 0.03. For token merging strategy, models with CT integration use only the last token in each chunk as the concept to minimize structural modifications when converting MoE to ConceptMoE and keep the initial loss as low as possible. Other models sum tokens within each chunk to form concepts. The hyperparameter for boundary random flipping is set to τ = 6, under which approximately 4% of tokens are flipped. This configuration keeps the evaluation compression ratio close to the training compression ratio without affecting training performance.

Evaluation Benchmarks We conduct extensive evaluations on both open-source and proprietary benchmarks. The evaluation suite covers text reasoning, mathematics, code generation, knowledge retrieval, needlein-haystack, long context summarization and understanding, as well as multimodal tasks including visual localization, visual reasoning, hallucination detection, visual question answering, and chart extraction. Detailed benchmark specifications are provided in the Appendix A.

##### 4.1 Small-scale language model pretraining

Model configurations We evaluate ConceptMoE on two MoE configurations: a 0.5B FLOPs model with 12B total parameters (MoE-A0.5B-12B), and a 1B FLOPs model with 24B total parameters(MoE-A1B-24B). Both baselines activate 8 experts per token. For ConceptMoE, we set LE = LD = 4 and scale the hidden size of C to 4/3 that of E and D. This configuration increases the per-token compute of C by a factor of 16/9 relative to the baseline. We therefore set the compression ratio to R = 16/9 to match the total compute budget 2. This setup corresponds to the third reallocation strategy in Section 3.5, which jointly increases Cattn and Cmoe.

Training Protocol We define tokens per parameter (TPP) as the ratio of training tokens to total model parameters. All models are trained at TPP=400: MoE-A0.5B-12B on 243B tokens and MoE-A1B-24B on 559B tokens. We use the AdamW optimizer with cosine learning rate decay.

Results Table 2 presents the main results comparing ConceptMoE against standard MoE baselines. Both model pairs maintain identical total parameters and per-token FLOPs, differing only in compute allocation strategy under concept or token level process. ConceptMoE consistently outperforms the baseline across most metrics.

These results validate that concept-level processing provides genuine benefits beyond simple token-level processing, suggesting that adaptive chunking effectively allocates more compute to semantically complex token sequences while efficiently processing predictable patterns.

##### 4.2 Train a vision-language model

Model configurations We investigate the potential of concept representations as inputs for vision-language model (VLM). We conduct experiments on MoE-A2.5B-60B and ConceptMoE-A2.5B-60B. The vision encoder uses a small ViT[8] for image feature extraction, and a linear projection to align the dimensions of the visual

2Note that all FLOPs comparisons exclude attention map computation, meaning ConceptMoE actually consumes fewer FLOPs than the baseline when compute-matched.

- Table 2 Model performance comparison. Comp.: Comprehensive Evaluation, Reason.: Reasoning, Know.: Knowledge. Details see Table 8.

Training Eval Openbench Easy

Model

Tokens Loss↓ Loss↓ Comp.↑ Reason.↑ Math↑ Code↑ Know.↑ All↑

- MoE-A0.5B-12B 243B 1.852 1.992 46.2 37.6 27.8 30.7 26.2 35.6 ConceptMoE-A0.5B-12B 243B 1.849 1.990 47.3 39.1 28.8 30.7 26.1 36.4

- MoE-A1B-24B 559B 1.717 1.851 57.6 54.4 47.2 42.3 41.6 50.0 ConceptMoE-A1B-24B 559B 1.711 1.844 57.4 56.8 50.0 42.4 41.5 50.9

token with the text tokens. On the LLM side, we maintain LE = LD = 4 and apply the third reallocation strategy from Section 3.5. We scale the hidden size of C to 1.5ÃŮ while slightly reducing the MoE inner dimension, and set the compression ratio to R = 2 to preserve total activated compute. Note that in the VLM setting, we apply compression to both visual and textual tokens.

Training Protocol We first pretrain the LLM on 500B tokens, then integrate a pretrained NaViT and a randomly initialized linear projection layer. The VLM continues training for 200B tokens on a mixture of high-quality image-text pairs and pure text data with 32K sequence length. Training uses the AdamW optimizer with cosine learning rate decay throughout.

0.01

2.05

0.010

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |MoE| | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | |oncept|M|oE| | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

MoE

0.00

ConceptMoE-mmloss ConceptMoE-textloss

2.00

CompressRatio

LossDifference

LossDifference

- -0.06
- -0.05
- -0.04
- -0.03
- -0.02
- -0.01

0.000

1.95

1.90

- -0.020

- -0.017

- -0.012

- -0.010

1.85

total R

image R

1.80

text R

10 20 50 100 200 500

10 20 50 100 200

10 20 50 100 200

Consumed tokens (B)

Consumed tokens (B)

Consumed tokens (B)

(a) PT Loss Diff. Curve (Smoothed)

(b) CT Loss Diff. Curve (Smoothed)

(c) Compression Ratio

- Figure 2 Training dynamics of loss and compression ratio. (a) Loss difference between ConceptMoE and MoE during language model pretraining (PT). (b) Loss difference during multimodal continue training(CT), separated into image-text data (mmloss) and text-only data(textloss). (c) Compression ratio evolution for image tokens and text tokens during multimodal training.

Results Figure 2 shows training dynamics. During PT (a), ConceptMoE achieves 0.01 lower loss than MoE. In multimodal CT (b), this gap increases to 0.017 for text data and 0.012 for image-text data. This difference reflects adaptive compression (c): while maintaining overall R = 2, the model compresses text less (using more compute) and images more, suggesting higher visual redundancy.

Tables 3, 4, and 5 present downstream results with average compression ratios of 2.27, 2.0, and 2.0 respectively. ConceptMoE outperforms MoE by 0.9 points on text benchmarks, 2.3 on long context, and 0.6 on multimodal tasks. Long context shows particularly strong gains across most subtasks, as expected from reduced sequence length alleviating degradation on long documents. The Needle task improvement confirms that concept merging preserves information. Multimodal results show gains in reasoning and understanding, validating stronger concept-level reasoning. However, fine-grained visual tasks (location, chart text, image Q&A) decline slightly, likely because treating image tokens sequentially disrupts spatial relationships critical for localization.

##### 4.3 Train from CT

In this subsection, we investigate converting MoE to ConceptMoE during continual training and evaluate the performance gap compared to training from scratch. We observe three key findings:

- Table 3 Text benchmark results on OpenBench. Comp.: Comprehensive Evaluation, Spec.: Specialized Knowledge, Know.: Knowledge. Details see Table 9.

Model Comp.↑ Reasoning↑ Math↑ Code↑ Spec.↑ Know.↑ All↑ MoE-A2.5B-60B 41.7 35.0 16.3 34.6 28.8 35.0 33.5 ConceptMoE-A2.5B-60B 41.8 35.3 16.2 36.9 30.8 36.8 34.4

Table 4 Long context evaluation results. LC: Long context, Sum.: Summary, Underst: Understanding.

Model Needle↑ LC Learn.↑ LC Reason.↑ LC Sum./Q&A↑ LC Underst.↑ All↑ MoE-A2.5B-60B 78.9 34.7 10.3 63.7 40.9 49.4 ConceptMoE-A2.5B-60B 80.7 38.8 17.1 59.0 45.1 51.7

- • CT conversion is lossless and highly beneficial. ConceptMoE-top15 maintains baseline performance, while adding layer loops yields substantial gains of +5.5 points on Open Benchmark. Training from scratch achieves an additional +0.9 points, reaching +6.4 overall improvement.
- • PT evaluation shows distribution shift effects. During PT, downstream metrics may appear lower due to misalignment between evaluation and pretraining data distributions, resulting in higher compression ratios (R = 1.81). However, evaluation and CT data distributions align closely, stabilizing compression at the expected ratio (R = 1.5).
- • CT-converted models enable direct inference speedup. The stabilized compression ratio of R = 1.5 in CT allows lossless conversion of existing MoE models to ConceptMoE while obtaining significant inference gains. Section 4.4 presents detailed speedup analysis showing prefill acceleration up to 43.6% and decoding acceleration up to 53.3% at R = 1.5.

# 5 …… L-3 L-1 L

[Figure 14]

[Figure 15]

[Figure 16]

|L-2|[Figure 17]<br><br>[Figure 18]|
|---|---|
| | |

|[Figure 19]<br><br>[Figure 20]<br><br>dechunk module|
|---|

[Figure 21]

[Figure 22]

[Figure 23]

|[Figure 24]<br><br>[Figure 25]<br><br>chunk module|
|---|

[Figure 26]

[Figure 27]

| |L-4|[Figure 28]<br><br>[Figure 29]|
|---|---|---|
| | | |

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

1 2 3 4

+

- Figure 3 From MoE to ConceptMoE illustration. The blue blocks denote the original MoE components. We add a chunk module and a dechunk module. In addition, in the last four self-attention layers, we insert an extra QKV projector initialized to zeros to enable joint decoding of concepts and tokens.

As illustrated in Figure 3, we start with MoE-A2.5B-90B pretrained on 700B tokens and convert it to ConceptMoE by adding a chunk module (query and key linear layers with random initialization), a dechunk module, and additional QKV projectors in the last 4 self-attention layers (zero-initialized). We then perform 400B tokens of 32k CT, followed by 40B tokens of 128k CT, and finally 3B tokens of in-house SFT with a small fraction of math and reasoning problems accompanied by long chain-of-thought. Given the CT setting, we conservatively set R = 1.5 and explore two compute allocation schemes:

- • ConceptMoE-top153: Increase activated experts from 8 to 15 (Strategy 1 in Section 3.5).
- • ConceptMoE-top11-loop8: Increase activated experts from 8 to 11 while looping 8 intermediate layers (Strategy 2 in Section 3.5).

These configurations ensure nearly identical average FLOPs and total parameters, enabling fair comparison. Additionally, we train ConceptMoE-top11-loop8 from scratch to quantify the performance gap introduced by CT conversion.

3We omit activated parameters and total parameters for brevity.

- Table 5 Vision-language benchmark results. Reason.: Reasoning, Desc. Halluc.: Description Hallucinations, Q&A: Image Question and Answer, Comp.: Comprehensive Bench.

Model Location↑ Reason.↑ Desc. Halluc.↑ Q&A↑ Chart Text↑ Comp. ↑ All↑ MoE-A2.5B-60B 82.2 53.5 35.8 61.4 34.7 53.9 53.6 ConceptMoE-A2.5B-60B 81.9 57.9 35.3 58.0 33.6 58.7 54.2

Figure 4 shows downstream benchmark evolution during training. During CT (800B to 1100B tokens), ConceptMoE-top15 matches MoE performance closely, with only a 0.3 point gap on Open Benchmark. ConceptMoE-top11-loop8 shows clear improvements, indicating that reallocating compute to LC is more efficient than increasing Cmoe. Comparing CT conversion to from-scratch training, ConceptMoE-top11-loop8 (from scratch) performs similarly on Open Benchmark but shows substantial gains on in-house and long context benchmarks, particularly the latter. This highlights the value of from-scratch training while demonstrating that CT conversion preserves and moderately improves existing model capabilities. Evaluation compression ratios remain stable at R = 1.5 throughout CT.

During PT (0 to 700B tokens), ConceptMoE-top11-loop8 and MoE exhibit comparable fluctuations, with evaluation compression ratio at R = 1.81, reducing FLOPs by 12.5%. This discrepancy between PT (R = 1.81) and CT (R = 1.5) compression ratios reveals distribution shift: evaluation and CT data are more aligned, both representing higher quality than PT data.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

200 400 600 800 1000

Consumed tokens (B)

0.28

0.30

0.32

0.34

0.36

0.38

0.40

0.42

Scores

(a) Open Benchmark

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

200 400 600 800 1000

Consumed tokens (B)

0.10

0.12

0.14

0.16

0.18

0.20

(b) In-house Benchmark

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

750 800 850 900 950 1000 1050 1100

Consumed tokens (B)

0.22

0.24

0.26

0.28

0.30

0.32

(c) In-house Long-context Benchmark

MoE ConceptMoE-top15 ConceptMoE-top11-loop8 ConceptMoE-top11-loop8 (from scratch)

Figure 4 Evolution of evaluation metrics across three benchmarks throughout the training process.

- Table 6 presents post-SFT performance. ConceptMoE-top15 achieves a modest 0.4 point improvement over MoE, while ConceptMoE-top11-loop8 shows substantial gains of 5.5 points overall, with particularly strong improvements in reasoning (+8.3), math (+12.2), and code (+6.4). While layer looping partly contributes to these gains, ConceptMoE critically achieves them without increasing FLOPs. The from-scratch trained model further improves by 0.9 points over the CT-converted version, demonstrating significant benefits. Evaluation compression ratios stabilize at R = 1.5 across all configurations.

- Table 6 CT results comparison on Open Benchmark. Comp. Eval.: Comprehensive Evaluation, FS: From Scratch.

Category MoE ConceptMoE-top15 ConceptMoE-top11-loop8 ConceptMoE-top11-loop8(FS)

Overall 40.9 41.3 46.4 47.3 Comp. Eval. 49.4 50.5 54.4 53.9 Reasoning 30.3 28.4 38.6 38.5 Math 38.1 39.7 50.3 52.8 Code 20.0 20.9 26.4 30.1 Instruction 54.7 52.9 55.1 54.6 Knowledge 28.2 26.2 27.5 27.3 Multilingual 71.7 75.3 75.3 80.3

##### 4.4 Significant inference speedup

We evaluated the inference latency of ConceptMoE on Hopper GPUs. Using MoE-A10B-300B as the baseline, we assessed 5 ConceptMoE configurations, ranging from efficiency-oriented to quality-oriented setups. Our results show that quality-oriented ConceptMoE achieves comparable speed to MoE on short sequences despite doubling the number of layers, while maintaining increasing speedup advantages on long sequences. Efficiency-oriented ConceptMoE delivers speedups of 32.1% to 117.1% in decoding and 24.7% to 175.1% in prefill.

ConceptMoE-2L-top8-R2 ConceptMoE-1.5L-top13-R2 ConceptMoE-1L-top24-R2 ConceptMoE-1L-top16-R1.5 ConceptMoE-1.25L-top11-R1.5

(a) Prefill

+175.1

+152.1

200

+121.6

+109.0

+95.4

+88.7

+76.8

+68.7

+61.1

+59.6

+55.3

100

+46.8

+43.6

+42.3

+40.5

+36.4

+36.0

+32.7

+31.6

+31.4

+27.6

+26.6

+24.7

+22.9

+22.3

+20.7

+19.4

+19.1

+18.4

+18.0

+17.2

+16.2

+14.4

+14.1

+12.1

+10.4

+10.4

+9.2

+7.8

+7.4

SpeedupvsMoE(%)

+6.7

+5.2

+0.4

-5.1

-2.9

0

4K 8K 16K 32K 64K 128K 256K 512K 1024K

(b) Decoding

200

+117.1

150

+77.5

+77.3

+72.5

100

+55.2

+53.3

+50.6

+38.3

+37.5

+37.4

+34.1

+34.1

+32.1

+25.2

+21.7

+16.9

+16.1

+13.5

50

+8.9

+7.3

+5.7

-14.3

-10.9

+3.1

-2.9

0

4K 8K 16K 32K 64K

KV Size

- Figure 5 Inference latency speedup over MoE for prefill and decoding. The prefill plot uses sequence length on the x axis, and the decoding plot uses KV cache length on the x axis with batch size 256. The y axis reports speedup relative to MoE in percent. ConceptMoE-xL-topy-Rz matches MoE in FLOPs and total parameters, where x is the layer multiplier, y is the number of activated experts per MoE block with baseline 8 and increased to match FLOPs, and z is the compression ratio.

Specifically, we configure models in the form ConceptMoE-xL-topy-Rz, where x represents the layer multiplication ratio, y denotes the number of activated experts (baseline is 8), and z indicates the compression ratio. For quality-oriented configurations, we use ConceptMoE-2L-top8-R2. For efficiency-oriented setups, we employ ConceptMoE-1L-top24-R2 and ConceptMoE-1L-top16-R1.5. For balanced configurations, we adopt ConceptMoE-1.5L-top13-R2 and ConceptMoE-1.25L-top11-R1.5. The R = 1.5 configurations are similar to the models discussed in Section 4.3.

- Figure 5 shows the speedup ratio of ConceptMoE over MoE. For the prefill stage, we evaluate speedup variations across input sequences ranging from 4K to 1024K. For the decoding stage, we assess speedup changes for kv cache lengths from 4K to 64K at batch size 256. Figure 6 presents the actual latency. We observe that even when doubling the number of layers at R = 2, ConceptMoE still achieves substantial speedupâĂŤthis stems from the quadratic reduction in attention map computation and the linear reduction in kv cache. When keeping the layer count unchanged, efficiency-oriented configurations achieve prefill speedups up to 175%, which continue to increase with longer sequences, and decoding speedups up to 117.1%. At R = 1.5, we also observe prefill speedups up to 43.6% and decoding speedups up to 53.3%. Section 4.3 validates the feasibility of integrating ConceptMoE in CT at R = 1.5, which allows us to directly convert existing MoE models to ConceptMoE losslessly while obtaining significant inference gains.

##### 4.5 Structure ablation

###### 4.5.1 Auxiliary loss weight

We ablate the auxiliary loss weight λ on ConceptMoE-A0.5B-12B trained for 243B tokens with target compression ratio R = 2. Following H-Net[19]’s auxiliary loss weight of 0.03, we evaluate λ ∈ {0.03,0.1,0.5,1.0} and examine their effects on training loss and achieved compression ratio.

MoE

ConceptMoE-1.5L-top13-R2

ConceptMoE-1L-top16-R1.5

ConceptMoE-2L-top8-R2

ConceptMoE-1L-top24-R2

ConceptMoE-1.25L-top11-R1.5

250000

Prefill

Decoding

90

200000

80

DecodingTime(ms)

PrefillTime(ms)

70

150000

| |
|---|

| |
|---|

| |
|---|

60

100000

| |
|---|

| |
|---|

50

| |
|---|
| |

| |
|---|
| |

| |
|---|

| |
|---|

50000

40

| |
|---|

| |
|---|

| |
|---|
| |
| |

| |
|---|

| |
|---|

30

| |
|---|

| |
|---|

| |
|---|

0

| |
|---|

4K 8K 16K 32K 64K 128K 256K 512K 1024K

KV Size

- Figure 6 Inference latency for prefill and decoding. The prefill plot uses sequence length on the x axis, and the decoding plot uses KV cache length on the x axis with batch size 256. The y axis reports the measured end to end inference latency on Hopper GPUs. ConceptMoE-xL-topy-Rz matches MoE in FLOPs and total parameters, where x is the layer multiplier, y is the number of activated experts per MoE block with baseline 8 and increased to match FLOPs, and z is the compression ratio.

0.01 0.1 1 10 100

Consumed tokens (B)

0.02

0.00

0.02

0.04

0.06

TrainingLossDifference

(a) Training Loss Diff

=0.03

=0.1

- =0.5

- =1

0.1 1 10 100

Consumed tokens (B)

- 2

- 3

- 4

- 5

- 6

CompressionRatio

(b) Compression Ratio

=0.03

=0.1

- =0.5

- =1

Figure 7 Impact of auxiliary loss weight λ on training loss and compression ratio.

- Figure 7 presents the results. As λ increases, training loss degrades while all configurations achieve compression ratios close to 2. Based on these findings, we set λ = 0.03 for all other experiments.

###### 4.5.2 Chunking strategy

We compare two chunking strategies on ConceptMoE-A0.5B-12B trained for 243B tokens with target compression ratio R = 2: (1) Dynamic Chunk: our learnable adaptive chunking based on token similarity, (2) Fixed Chunk: merging every consecutive R tokens into one concept. We use No Chunk(MoE-A0.5B-12B) as our baseline.

- Figure 8(a) shows training dynamics. Dynamic Chunk maintains a consistent advantage throughout training and ultimately achieves 0.004 lower loss than No Chunk, demonstrating that adaptive compression with proper compute reallocation improves optimization. In contrast, Fixed Chunk degrades by 0.01 relative to No Chunk, indicating that uniform merging disrupts the model’s learning dynamics. Downstream evaluation (Figure 8(b)) confirms this pattern: Dynamic Chunk achieves the highest average score (36.4), outperforming No Chunk (35.6) and Fixed Chunk (34.2). These results validate that adaptive boundary identification preserves semantic coherence while enabling efficient computation.

50

Dynamic Chunk

Dynamic Chunk

47.3

46.6

46.2

45

0.20

Fixed Chunk

Fixed Chunk

LossDifference

No Chunk

No Chunk

40

0.15

39.1

38.0

Score

37.6

36.4

35.6

35

34.2

0.10

30.7

30.7

30

28.8

0.05

27.8

27.8

26.2

26.1

26.0

25

0.00

21.9

1 10 100

Avg Comp. Reason. Math Code Know.

Consumed tokens (B)

(a) Training Loss Diff (Smoothed)

(b) Downstream Metrics

###### Figure 8 Comparison of chunking strategies. (a) Training loss difference relative to No Chunk baseline (i.e. MoE) during pretraining. Dynamic Chunk consistently achieves lower loss than Fixed Chunk. (b) Downstream benchmark scores after training. Comp.: Comprehensive Evaluation, Reason.: Reasoning, Know.: Knowledge.

0.00

Baseline

47.3

45

- - cos + linear router

- - joint decoding

LossDifference

44.2

43.3

0.01

40

39.1

Score

37.8

0.02

36.4

35.5

35.1

35

34.4

- - cos + linear router

- - joint decoding

31.8

30.7

0.03

30.3

30

28.8

28.5

28.1

Baseline

26.7

26.1

25.6

25

10 100

Avg Comp. Reason. Math Code Know.

Consumed tokens (B)

(a) Training Loss Diff (Smoothed)

(b) Downstream Metrics

Figure 9 Impact of router type and joint decoding on training and downstream performance.

###### 4.5.3 Router design

Section 3.2 introduces our cosine similarity-based score for identifying chunk boundaries. A natural alternative is to directly predict boundary scores using a linear layer, analogous to MoE routers. We compare these two designs: cosine router (our approach) and linear router, on ConceptMoE-A0.5B-12B trained for 243B tokens with target compression ratio R = 2.

- Figure 9(a) shows that the linear router achieves 0.003 lower training loss at convergence. However, Figure 9(b) reveals a substantial gap in downstream performance: the linear router scores 34.4 average, significantly underperforming the cosine router at 36.4. This train-eval discrepancy suggests that the linear router overfits to training data patterns. In contrast, the cosine router’s explicit modeling of inter-token similarity provides better generalization by capturing semantic relationships rather than memorizing dataset-specific boundary patterns.

###### 4.5.4 Joint decoding ablation

Section 3.4 introduces joint decoding of concepts and tokens in the decoder’s final layers, adding negligible computation. We ablate this component on ConceptMoE-A0.5B-12B trained for 243B tokens with target compression ratio R = 2.

- Figure 9(a) shows that removing joint decoding yields 0.002 lower training loss at convergence. However,
- Figure 9(b) reveals substantial downstream degradation: the model without joint decoding scores 35.1 average compared to 36.4 with joint decoding. This pattern mirrors the router ablation: improvements in training loss do not guarantee better generalization. We hypothesize that joint decoding acts as implicit regularization during training. By forcing the decoder to

explicitly attend to concept information through additional QKV projections, the model learns more robust representations that transfer better to downstream tasks. Without joint decoding, the decoder may overfit to residual token-level patterns in the training data while underutilizing the semantic information encoded in concepts. This validates that leveraging concept representations throughout the decoding process is essential for realizing the full benefits of adaptive compression.

##### 4.6 Boundary Noise for Robustness

We observe a discrepancy between training and evaluation compression ratios: the model achieves lower compression during training than during evaluation. Analysis of the boundary probability distribution reveals that a substantial portion of probabilities cluster around 0.5. During evaluation, these borderline cases can easily flip, causing unintended compression ratio drift.

Theoretically, for target compression ratio R, the mean probability should satisfy E[pn] ≈ 1/R. However, without noise, we observe significantly lower probability means, indicating that many boundaries hover near the 0.5 threshold. During evaluation, slight perturbations or distribution shifts push these marginal probabilities over the boundary, leading to higher compression than intended.

To improve robustness, we introduce noise during training to simulate evaluation conditions. We evaluate Bernoulli noise with τ ∈ {4,6} for psharpn and Gaussian noise with σ = 0.1. All models train with target R = 1.5.

- Figure 10 shows training dynamics. Bernoulli noise strategies yield higher training loss at convergence, with smaller τ producing larger degradation, confirming that noise perturbs chunk boundaries. Gaussian noise maintains comparable training loss. Critically, noise regularization normalizes the probability mean closer to 1/R (e.g. 0.667), stabilizing evaluation compression ratios.

- Table 7 demonstrates that noise improves downstream performance despite higher training loss. ConceptMoE with τ = 4 achieves 30.3 average score (+1.4 over baseline), with stronger Bernoulli noise proving most effective. This validates that boundary noise strengthens chunk module robustness by preventing probability collapse around 0.5, ensuring consistent compression behavior between training and evaluation. To avoid excessive impact on training loss, we use Bernoulli noise with τ = 6 in all other experiments.

0.66

Bernoulli =4 Bernoulli =6 Gaussian =0.1

baseline

0.0125

Bernoulli =4 Bernoulli =6 Gaussian =0.1

0.64

Loss-Baseline

0.0100

ProbMean

baseline

0.62

0.0075

0.60

0.0050

0.58

0.0025

0.56

0.54

0.0000

10 20 50 100 200 500

10 20 50 100 200 500

Consumed tokens (B) (a) Training Loss (Smoothed)

Consumed tokens (B) (b) Mean Probability (Smoothed)

- Figure 10 Training loss diff and mean probability of chunk module for different noise strategies. We do smoothing and remove some spike data for better visualization, which doesn’t affect the conclusion.

###### 4.6.1 Target compression ratio

We investigate the impact of compression ratio by training models with R = 2 and R = 4, using MoE-A1B-24B as baseline with compute reallocation strategy 3 (increasing Cattn and Cmoe from Section 3.5) over 559B tokens.

- Table 7 Model performance comparison on OpenBench for different noise strategies. Comp.: Comprehensive Evaluation, Spec.: Specialized Knowledge, Know.: Knowledge.. Comp.: Comprehensive Evaluation, Reason.: Reasoning, Know.: Knowledge.

###### Category ConceptMoE ConceptMoE-Gaussian ConceptMoE-τ = 4 ConceptMoE-τ = 6

All 28.9 29.6 30.3 30.0 Comp. Eval. 38.9 40.0 40.1 40.3 Reasoning 22.9 23.5 24.7 24.0 Math 9.1 8.7 9.4 9.5 Code 36.0 37.5 38.8 38.7 Spec. 25.0 28.4 27.6 26.1 Know. 30.2 28.6 30.3 29.6

60

Baseline

R=4

57.6

57.4

56.8

55.9

R=2 R=4

Baseline

LossDifference

55

0.15

54.4

R=2

51.3

50.8

50.0

50.0

Score

50

0.10

47.7

47.2

46.3

45

0.05

42.4

42.3

41.6

41.5

40.7

40

0.00

37.4

1 10 100

Avg Comp. Reason. Math Code Know.

Consumed tokens (B)

(a) Training Loss Diff (Smoothed)

(b) Downstream Metrics

- Figure 11 Impact of target compression ratio on training and downstream performance. (a) Training loss difference relative to baseline. R = 2 converges close to baseline, while R = 4 shows persistent degradation. (b) Downstream benchmark scores. R = 2 outperforms the baseline across most metrics, while R = 4 underperforms substantially, particularly on reasoning and math tasks. Results indicate that excessive compression (e.g., R = 4) degrades performance despite successful compression ratio control.

- Figure 11 reveals that higher compression does not guarantee better performance. While R = 2 achieves comparable training loss to the baseline and improves downstream scores to 50.8 average, R = 4 shows substantial degradation in both training loss (0.013 gap at convergence) and downstream performance (47.7 average). The gap is particularly pronounced on reasoning (51.3 vs 56.8) and math (46.3 vs 50.0), suggesting that aggressive compression disrupts complex reasoning patterns.

We hypothesize that each dataset has an optimal compression ratio determined by its semantic redundancy distribution. At R = 4, the model is forced to merge tokens with significant semantic differences, losing critical information for downstream tasks. The auxiliary loss successfully constrains the compression ratio during training, but the resulting 4× compression fundamentally exceeds the natural redundancy level in the data. This indicates that compression ratio should be calibrated to dataset characteristics rather than maximized unconditionally. For typical pretraining corpora, R = 1.5 to R = 2 appears to strike an effective balance between efficiency and information preservation.

#### 5 Conclusion

We introduce ConceptMoE, a framework that dynamically merges semantically similar tokens into concepts through learnable chunking. By operating within the MoE architecture, we enable fair comparison under controlled conditions: reallocating saved computation to match baseline FLOPs and total parameters isolates genuine architectural benefits. Experiments demonstrate consistent improvements across language pretraining (+0.9 points), vision-language training(+0.6 points), long context (+2.3 points), and continual training conversion (+5.5 points with layer looping).

Beyond performance gains, ConceptMoE reduces attention computation by up to R2× and KV cache by R×,

###### achieving prefill speedups up to 175% and decoding speedups up to 117% at R = 2. The minimal architectural changes enable straightforward integration into existing systems. ConceptMoE demonstrates that adaptive concept-level processing fundamentally improves both the effectiveness and efficiency of LLMs, opening new directions for compute allocation in sparse architectures.

#### References

- [1] Sravan Kumar Ankireddy, Nikita Seleznev, Nam H Nguyen, Yulun Wu, Senthil Kumar, Furong Huang, and C Bayan Bruss. Timesqueeze: Dynamic patching for efficient time series forecasting. In Recent Advances in Time Series Foundation Models Have We Reached the’BERT Moment’?

- [2] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

- [3] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. arXiv preprint arXiv:2401.10774, 2024.

- [4] Linzheng Chai, Shukai Liu, Jian Yang, Yuwei Yin, Ke Jin, Jiaheng Liu, Tao Sun, Ge Zhang, Changyu Ren, Hongcheng Guo, et al. Mceval: Massively multilingual code evaluation. arXiv e-prints, pages arXiv–2406, 2024.

- [5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, et al. Evaluating large language models trained on code. 2021.
- [6] François Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019.

- [7] Beiya Dai, Yuliang Liu, Daozheng Xue, Qipeng Guo, Kai Chen, Xinbing Wang, Bowen Zhou, and Zhouhan Lin. Context-level language modeling by learning predictive context embeddings. arXiv preprint arXiv:2510.20280, 2025.

- [8] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch nâĂŹpack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:2252–2274, 2023.

- [9] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proc. of NAACL, 2019.

- [10] Ippei Fujisawa, Sensho Nobe, Hiroki Seto, Rina Onda, Yoshiaki Uchida, Hiroki Ikoma, Pei-Chun Chien, and Ryota Kanai. Procbench: Benchmark for multi-step reasoning and following procedure. arXiv preprint arXiv:2410.03117, 2024.

- [11] Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. Omni-math: A universal olympiad level mathematic benchmark for large language models, 2024. URL https://arxiv.org/abs/2410.07985.
- [12] Saibo Geng, Nathan Ranchin, Maxime Peyrard, Chris Wendler, Michael Gastpar, Robert West, et al. zip2zip: Inference-time adaptive vocabularies for language models via token compression. arXiv preprint arXiv:2506.01084, 2025.

- [13] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024.
- [14] Yancheng He, Shilong Li, Jiaheng Liu, Yingshui Tan, Weixun Wang, Hui Huang, Xingyuan Bu, Hangyu Guo, Chengwei Hu, Boren Zheng, Zhuoran Lin, Xuepeng Liu, Dekai Sun, Shirong Lin, Zhicheng Zheng, Xiaoyong Zhu, Wenbo Su, and Bo Zheng. Chinese simpleqa: A chinese factuality evaluation for large language models, 2024. URL https://arxiv.org/abs/2411.07140.
- [15] Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. Aligning ai with shared human values. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

- [16] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

- [17] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

- [18] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. In Advances in Neural Information Processing Systems, 2023.

- [19] Sukjun Hwang, Brandon Wang, and Albert Gu. Dynamic chunking for end-to-end hierarchical sequence modeling. arXiv preprint arXiv:2507.07955, 2025.

- [20] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint, 2024.

- [21] Peng Jin, Bo Zhu, Li Yuan, and Shuicheng Yan. Moe++: Accelerating mixture-of-experts methods with zero-computation experts. arXiv preprint arXiv:2410.07348, 2024.

- [22] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. triviaqa: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. arXiv e-prints, art. arXiv:1705.03551, 2017.

- [23] Bill Yuchen Lin, Ronan Le Bras, and Yejin Choi. Zebralogic: Benchmarking the logical reasoning ability of language models, 2024. URL https://huggingface.co/spaces/allenai/ZebraLogic.
- [24] Dong Liu, Yanxuan Yu, and Ben Lengerich. Csv-decode: Certifiable sub-vocabulary decoding for efficient large language model inference. arXiv preprint arXiv:2511.21702, 2025.

- [25] Kaijing Ma, Xinrun Du, Yunran Wang, Haoran Zhang, Zhoufutu Wen, Xingwei Qu, Jian Yang, Jiaheng Liu, Minghao Liu, Xiang Yue, Wenhao Huang, and Ge Zhang. Kor-bench: Benchmarking language models on knowledge-orthogonal reasoning tasks, 2024. URL https://arxiv.org/abs/2410.06526.
- [26] Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, et al. Byte latent transformer: Patches scale better than tokens,

2024. URL https://arxiv. org/abs/2412.09871, 2412, 2024.

- [27] Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

- [28] Xingwei Qu, Shaowen Wang, Zihao Huang, Kai Hua, Fan Yin, Rui-Jie Zhu, Jundong Zhou, Qiyang Min, Zihao Wang, Yizhi Li, et al. Dynamic large concept models: Latent reasoning in an adaptive semantic space. arXiv preprint arXiv:2512.24617, 2025.

- [29] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=Ti67584b98.

- [30] Chenze Shao, Darren Li, Fandong Meng, and Jie Zhou. Continuous autoregressive language models. arXiv preprint arXiv:2510.27688, 2025.

- [31] Kevin Slagle. Spacebyte: Towards deleting tokenization from large language modeling. Advances in Neural Information Processing Systems, 37:124925–124950, 2024.

- [32] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

- [33] Sho Takase, Ryokan Ri, Shun Kiyono, and Takuya Kato. Large vocabulary size improves large language models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 1015–1026, 2025.

- [34] Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, and Ngai Wong. Scaling laws with vocabulary: Larger models deserve larger vocabularies. Advances in Neural Information Processing Systems, 37:114147–114179, 2024.

- [35] M-A-P Team, Xinrun Du, Yifan Yao, et al. Supergpqa: Scaling llm evaluation across 285 graduate disciplines,

2025. URL https://arxiv.org/abs/2502.14739.

- [36] Mathurin Videau, Badr Youbi Idrissi, Alessandro Leite, Marc Schoenauer, Olivier Teytaud, and David Lopez-Paz. From bytes to ideas: Language modeling with autoregressive u-nets. arXiv preprint arXiv:2506.14761, 2025.

- [37] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024.

- [38] Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, and William Fedus. Measuring short-form factuality in large language models. arXiv preprint arXiv:2411.04368,

- 2024.

[39] Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Benjamin Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey, Shubh-Agrawal, Sandeep Singh Sandha, Siddartha Venkat Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, and Micah Goldblum. Livebench: A challenging, contamination-free LLM benchmark. In The Thirteenth International Conference on Learning Representations,

- 2025.

- [40] Lili Yu, Dániel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, and Mike Lewis. Megabyte: Predicting million-byte sequences with multiscale transformers. Advances in Neural Information Processing Systems, 36:78808–78823, 2023.

- [41] Albert S. Yue, Lovish Madaan, Ted Moskovitz, DJ Strouse, and Aaditya K. Singh. HARP: A challenging human-annotated math reasoning benchmark, 2024. URL https://github.com/aadityasingh/HARP.
- [42] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models, 2023.

### Appendix

#### A Evaluation benchmark

We provide the evaluation datasets included in each category of OpenBench. Table 8 details the relatively easy evaluation sets used for models with 12B and 24B total parameters. Table 9 includes both easy and hard evaluation sets used for models with 60B and 90B total parameters.

Table 8 Open benchmarks Easy across different domains

Comprehensive Evaluation Reasoning Math Code Knowledge MMLU[15, 16] BBH[32] MATH[17] HumanEval[5] TriviaQA[22] C-Eval[18] DROP[9] MBPP+[2] ChineseSimpleQA[14] MMLU-Pro[37] McEval[4] AGIEval[42]

Table 9 Open benchmarks across different domains

Comprehensive Evaluation

Specialized Knowledge

Reasoning Math Code Knowledge

MMLU[15, 16] BBH[32] MATH[17] HumanEval[5] TriviaQA[22] GPQA[29] C-Eval[18] DROP[9] AIME2024 MBPP+[2] SimpleQA[38] MMLU-Pro[37] ARC_AGI[6] AIME2025 LiveCodeBench[20] ChineseSimpleQA[14] AGIEval[42] ProcBench[10] HARP[41] McEval[4] HLE[27] ZebraLogic[23] Omni-MATH[11] SuperGPQA[35] KOR-Bench[25] OlympiadBench[13] LiveBench[39]

#### B Code

###### Below we present the forward function in PyTorch format. In this example, the encoder has 2 layers, the concept model has 23 layers, and the decoder has 2 layers. The hidden size is 2048. The input consists of 1024 tokens, which are compressed to 701 tokens before entering the concept model.

import torch import torch.nn.functional as F from torch import nn

class ChunkModule(nn.Module): def __init__(self, config): super().__init__()

self.q_proj_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False) self.k_proj_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

def forward(self, hidden_states):

cos_sim = torch.einsum( "l d, l d -> l", F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1), F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),

) # shape [1023,] boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0) # shape [1023,] # Force boundary probability of the first element to 1.0 PAD_PROB = 1.0 boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB) # shape [1024,]

selected_idx = torch.zeros_like(boundary_prob, dtype=torch.long)

boundary_mask = boundary_prob >= 0.5 selected_idx[..., boundary_mask] = 1 boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

selected_probs = boundary_prob.gather(

dim=-1, index=selected_idx.unsqueeze(-1) ) # (shape hidden_states.shape[:-1], 1)

return boundary_prob, boundary_mask, selected_probs

class DechunkModule(nn.Module): def __init__(self, config):

super().__init__()

def forward(self, concept, boundary_prob, boundary_mask): concept_prob = boundary_prob[boundary_mask] # shape [701,] concept_merge = torch.zeros_like(concept)

# For ease of understanding, this is written in for-loop form. In practice, it can be

accelerated through parallel scan. concept_merge[0] = concept[0] # shape [701, 2048] for i in range(1, concept.shape[0]):

concept_merge[i] = concept_merge[i-1]*(1-concept_prob[i]) + concept[i] * concept_prob[ i]

plug_back_idx = boundary_mask.cumsum(dim=0) - 1 concept_merge = torch.gather(

concept_merge, dim=0, index=plug_back_idx.expand(-1, 2048) ) # concept_merge shape [701,2048] -> [1024,2048]

return concept_merge

class STE(torch.autograd.Function): @staticmethod def forward(ctx, x):

return torch.ones_like(x) @staticmethod def backward(ctx, grad_output):

grad_x = grad_output return grad_x

def ste_func(x): return STE.apply(x)

class ConceptMoE(nn.Module):

def __init__(self, config): super().__init__() self.encoder = nn.Modulelist(transformer_layer(layer_id=i) for i in range(2)) self.concept_model = nn.Modulelist(transformer_layer(layer_id=i) for i in range(2,25)) self.decoder = nn.Modulelist(transformer_layer(layer_id=i) for i in range(25,27))

self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

self.chunk_module = ChunkModule(config) self.dechunk_module = DechunkModule(config)

def forward(self, input_ids): # encoder hidden_state = self.embedding(input_ids) # shape [1024, 2048] hidden_state = self.encoder(hidden_state)

# chunk boundary_prob, boundary_mask, selected_probs = self.chunk_module(hidden_state)

# main network concept = hidden_state[boundary_mask] # shape [701, 2048]

concept = self.concept_model(concept) # dechunk concept_merge = self.dechunk_module(concept, boundary_prob, boundary_mask) # decoder

hidden_state = hidden_state + concept_merge * ste_func(selected_probs) hidden_state = self.decoder(hidden_stateïijˇN concept_merge) # joint decoding

logits = self.lm_head(hidden_state) return logits

