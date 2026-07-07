# arXiv:2502.09245v2[cs.LG]28May2025

## You Do Not Fully Utilize Transformer’s Representation Capacity

Gleb Gerasimov1,2,3† Yaroslav Aksenov1† ∗

Nikita Balagansky1,2 Viacheslav Sinii1

Daniil Gavrilov1 †Equal contribution 1T-Tech 2Moscow Institute of Physics and Technology 3HSE University

### Abstract

In contrast to RNNs, which compress their history into a single hidden state, Transformers can attend to all past tokens directly. However, standard Transformers rely solely on the hidden state from the previous layer to represent the entire context. We show that this design choice induces representation collapse and degrades performance. To address this issue, we introduce Layer-Integrated Memory (LIMe), a lightweight extension that leverages existing key–value buffers and learns perhead, per-layer routing weights to integrate representations from all previous layers with negligible overhead. Through extensive experiments—including language modeling, synthetic reasoning benchmarks, and very deep architectures—LIMe consistently achieves faster convergence, lower perplexity per FLOP, and substantial accuracy improvements on synthetic tasks while preserving higher value–vector entropy and improved token separability. Finally, our analysis of the learned routing weights reveals systematic reuse of both local and long-distance features, demonstrating how LIMe mitigates collapse, unlocks richer representations without increasing hidden-state size, and points to promising directions for future research.

### 1 Introduction

Transformers [Vaswani et al., 2017] have become a central architecture in modern machine learning, powering state-of-the-art solutions in language modeling, computer vision, and beyond. Their ability to capture complex patterns arises from deeply stacked layers that refine contextual representations. However, despite their success, standard Transformer decoders maintain a single residual stream per layer, forcing the model to compress all previously learned features into the immediately preceding hidden state [Srivastava et al., 2015, He et al., 2015]. This design choice can lead to representation collapse—a phenomenon in which different tokens or features become indistinguishable in deeper layers [Voita et al., 2019, Barbero et al., 2024, Arefin et al., 2024]. The problem is particularly pronounced when learning from lengthy sequences, where subtle token distinctions risk being squeezed out by limited floating-point precision and finite hidden-state capacity.

In this paper, we propose Layer-Integrated Memory (LIMe) 2, a lightweight extension to multi-head self-attention that enables each attention head to retrieve and integrate representations from all

∗Corresponding author: y.o.aksenov@tbank.ru 2Code available at https://github.com/corl-team/lime

Preprint. Under review.

preceding layers—rather than relying solely on the most recent hidden state. LIMe accomplishes this by learning a per-layer, per-head routing mechanism that efficiently blends multi-layer Key–Value features, all while preserving the core Transformer structure and adding negligible overhead by reusing already allocated Key–Value buffers.

Our key contributions are:

- • Layer-Integrated Routing. A trainable router that, for each head at every layer, dynamically weights and mixes buffered Key–Value representations from all earlier layers, without increasing hidden-state dimensions or memory footprint.
- • Strong Empirical Gains. LIMe converges 15.3% (8.9% with GQA) faster in FLOPs and achieves 1.15% (0.91% with GQA) lower perplexity than 1B-parameter LLaMa-based [Grattafiori et al., 2024] transformer, yields up to +8% on ProsQA [Hao et al., 2024] and

+30% on arithmetic reasoning benchmarks [Arefin et al., 2024, Feng et al., 2023]. In deep settings (32, 64, 128 layers), a 64-layer LIMe matches a 128-layer baseline, indicating superior scaling behavior.

- • Mitigating Collapse. An empirical analysis showing that LIMe preserves higher R´enyi entropy [Arefin et al., 2024] and better token separability [Voita et al., 2019] in value spaces, effectively alleviating representation collapse.

Together, these results confirm that by distributing representational burden across persistent Key–Value buffers and learning to route information across layers, LIMe substantially improves both optimization efficiency and representational capacity, especially in tasks requiring long-range or multi-step reasoning, opening the door of utilizing LIMe for cutting-edge area of latent-space reasoning.

### 2 Related Work

Early works on training very deep networks highlighted the need for mechanisms to ease gradient flow and information propagation. Highway Networks introduce gated skip connections to regulate information flow across layers [Srivastava et al., 2015]. Deep Residual Networks further simplify this by adding identity shortcuts, enabling networks to exceed a hundred layers without suffering from vanishing gradients [He et al., 2015]. Transformers adopt a similar residual-plus-normalization design, which underpins their success in language and vision tasks [Vaswani et al., 2017, Grattafiori et al., 2024, Jiang et al., 2023, Qwen et al., 2024, DeepSeek-AI et al., 2024].

Although residual streams facilitate training, they still force each layer to compress all prior features into a single vector, which can lead to representation collapse—distinct inputs becoming indistinguishable in deeper layers. Tenney et al. [2019] found that BERT’s deeper layers refine earlier predictions using higher-level context. Voita et al. [2019] empirically demonstrated that Transformers’ top layers lose fine-grained token distinctions. Theoretically, Barbero et al. [2024] proved that decoder-only Transformers can exhibit arbitrarily close final-token representations for different inputs, a phenomenon akin to over-squashing. Building on this, Hahn and Rofin [2024] showed that the loss landscape of Transformers biases them toward low-sensitivity functions, exacerbating collapse. Recently, Arefin et al. [2024] introduced Seq-VCR, a variance–covariance regularizer that preserves intermediate representation diversity and significantly improves multi-step reasoning performance.

To mitigate collapse, several works have explored aggregating information across layers. CrossLayer Retrospective Retrieving learns dynamic attention weights over prior layer outputs for each head [Fang et al., 2023]. Hyper-Connections augment Transformers with multiple residual streams that interact via learned projections, preventing collapse at the cost of increased hidden-state size [Zhu et al., 2024]. Although Mixture-of-Depths [Raposo et al., 2024] focuses on reducing FLOPs by skipping token computations layer-wise, its dynamic routing approach resonates with our per-head, per-layer routing mechanism; unlike MoD, LIMe retains full dense computation while enriching representational capacity through routing over pre-allocated key–value buffers. Different architectures based on usage of previous representations were proposed in [Huang et al., 2018, Bapna et al., 2018, Wu et al., 2023]. Despite these advances, most methods require substantial architectural changes or extra memory. Our method, Layer-Integrated Memory (LIMe), instead reuses existing key–value buffers and learns per-head, per-layer routing to mix multi-layer representations with negligible memory and speed overhead (see Appendix F).

Training Loss

2.75

LLaMa GQA

LIMe GQA

2.70

2.65

2.60

Loss

2.55

2.50

2.45

0.4 0.6 0.8 1.0 1.2 1.4 1.6

FLOPs

1e20

Figure 1: Training loss per FLOPs for LLaMa and LIMe. LIMe has a substantially lower loss with a similar amount of FLOPs. See Section 5.1 for more details.

### 3 Preliminaries

Notation. Let t denote the sequence length (temporal dimension), d the model dimension, H the number of attention heads, dhead = d/H the dimension of each head, and L the total number of layers. We denote by Xℓ−1 ∈ Rt×d the residual stream entering layer ℓ, with ℓ = 1,...,L.

Causal Self-Attention. Let Q = XW(Q), K = XW(K), V = XW(V ),

with W(Q),W(K),W(V ) ∈ Rd×d. Splitting into H heads of dimension dh = d/H yields {Qi,Ki,Vi}Hi=1. For head i,

⊤

headi = softmax Qi K

√dhi + M Vi ∈ Rt×d

,

h

where M masks future positions. The heads are concatenated across the last dimension and projected: MultiHeadAttn(X) = Concat(head1,...,headH)W(O), W(O) ∈ Rd×d.

Residual connections. Denoting a sub-layer function F(·) and input X, the pre-norm residual update is

X′ = X + F RMSNorm(X) .

### 4 Method

We introduce Layer-Integrated Memory (LIMe), a lightweight mechanism to augment a decoderonly Transformer with inter-layer, learnable information flow. Unlike standard multi-head attention (MHA), which attends only to the current layer’s residual stream, LIMe enables each head to retrieve and fuse Key–Value representations from all earlier layers. This enriches the model’s representation capacity without increasing memory use, since we reuse the Key–Value buffers already allocated by vanilla Transformers.

At a high level, each LIMe attention layer performs three steps:

- 1. Compute and buffer per-head Key–Value projections from the current residual stream.
- 2. Route by forming a learned mixture of all buffered Key and Value heads’ states up to the current layer.
- 3. Compute attention between the current layer’s Queries and the routed Key–Value mixture.

Visualisation of the architecture can be found in Appendix G.

- 1. Key–Value Buffering. At layer ℓ, we compute per-head Key and Value tensors in the usual way:

Kℓ = Xℓ−1 Wℓ(K), Vℓ = Xℓ−1 Wℓ(V ), Kℓ, Vℓ ∈ Rt×H×d

h

. (1) We then store these in the pre-allocated buffers

B(K), B(V ) ∈ RL×H×t×d

h

,

for Keys and Values respectively. No extra memory is required, since vanilla Transformers already maintain all per-layer Key–Value states for training and cache them during inference for generation efficiency. See Appendix F for details.

- 2. Inter-Layer Routing. To enable each head at layer ℓ to mix information from all previous layers,

we introduce a trainable router tensor R(ℓ) ∈ Rℓ×H×H, where R(ℓℓ′),h′,h is a weight from head h′ at layer ℓ′ into head h at layer ℓ.

Using buffer we route keys and values for each head h:

Kℓ,h =

ℓ

ℓ′=1

H

h′=1

Rℓ(ℓ′,h) ′,h Bℓ(′K,h)′, and Vℓ,h =

ℓ

ℓ′=1

H

h′=1

Rℓ(ℓ′,h) ′,h Bℓ(′V,h)′. (2)

- 3. Attention with Layer-Integrated Memory. We compute the usual per-head Queries,

Qℓ,h = Xℓ−1 Wℓ,h(Q), Qℓ,h ∈ Rt×d

,

h

and then perform scaled dot-product attention for each head between Qℓ,h and the routed Kℓ,h, Vℓ,h. LIMe Advantages. By routing through all prior layers, LIMe endows each head with a learnable, layer-wise memory. Unlike fixed skip connections or naive averaging, LIMe learns per-head, per-layer weightings, enabling selective retrieval and forgetting of past representations. Despite this added flexibility, the extra computation is only linear in sequence length. Crucially, LIMe is fully compatible with efficient MHA implementations such as FlashAttention [Dao, 2024], and it introduces negligible additional memory footprint by reusing existing Key–Value buffers (see Appendix F for details). In Appendix D, we include an ablation study on restricted router weights, demonstrating the importance of the trained router in LIMe.

### 5 Experiments

#### 5.1 Language Modeling

We evaluate the effectiveness of LIMe against two baselines: LLaMa [Grattafiori et al., 2024] and Hyper Connections [Zhu et al., 2024]. All models have approximately 1B parameters and share the same underlying transformer architecture (see Table 3). We trained each model from scratch on the FineWeb Edu [Penedo et al., 2024] subset with about 50B tokens. The full training setup can be found in Appendix A.

- Figure 1 displays the iso-flops training loss curves, demonstrating that LIMe converges more rapidly and achieves lower perplexities than LLaMa, indicating improved parameter efficiency. Details on model efficiency and FLOPs calculations can be found in Appendix F. Table 1 presents results on the 3-shot LM Eval Harness benchmarks Wang et al. [2018, 2019], Srivastava et al. [2023], further highlighting the advantages conferred by LIMe on language modeling over baseline models. For more benchmarks see Appendix C. In the next section, we go deeper into the factors driving these gains.

#### 5.2 Measuring Representation Collapse

Recent work has shown that large language models (LLMs) can suffer from representation collapse when representing long sequences, thereby forcing subtle token distinctions to become inseparable in deeper layers [Voita et al., 2019, Arefin et al., 2024]. We investigate this phenomenon by comparing LLaMa [Grattafiori et al., 2024] and LIMe via two complementary approaches: (i) quantifying

Model MultiRC WiC QNLI ARC-E ARC-C KV Induction LD-3 Avg LLaMA 43.24 50.00 49.49 70.45 38.70 45.94 54.20 33.60 48.20

HC 54.34 49.72 49.43 71.15 37.63 51.68 51.59 33.87 49.93 LIMe 56.15 50.44 51.43 71.15 39.30 55.64 55.36 34.47 51.74

Table 1: LM Evaluation Harness benchmarks results on 1B models with GQA in 3-shot setup. LIMe outperforms both LLaMA and Hyper-Connections baselines. See details in Section 5.1.

Accuracy of Values Classification

Entropy of Values

1.00

LLaMa

LIMe

0.99

6.6

0.98

6.4

0.97

Accuracy

Entropy

6.2

0.96

0.95

6.0

0.94

5.8

0.93

LLaMa

LIMe

5.6

0.92

2 4 6 8 10 12 14 16

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

Layer

Layer

(a)

(b)

- Figure 2: (a) Matrix entropy of values on the FineWeb Edu subset by layer. LIMe has more diverse values than LLaMa, which indicates that more information is stored in its hidden states. (b) Values’ classification accuracy, with standard deviation over five cross-validation folds. Values in later layers obtained from LIMe can be linearly separated with nearly 1.0 accuracy, whereas the accuracy for values from LLaMa is much lower. See Section 5.2 for more details.

the diversity of hidden states and values with matrix-based R´enyi entropy [Arefin et al., 2024] and (ii) measuring and visualizing the linear separability of layer-wise embeddings of closely related tokens (is, are, was, were) [Voita et al., 2019]. These two methodologies directly measure representation collapse in language models.

Unlike Arefin et al. [2024], we evaluate both residual-stream hidden states and value representations. We expect weaker linear separability in hidden states (because the model need not pack all information there) and stronger separation in value vectors. For matrix entropy, we anticipate little change at the hidden-state level but a clear difference for value representations. At each layer ℓ, we record value

states (i.e., the output of the Wℓ(V ) linear projection) and hidden states (i.e., the residual stream Xℓ). Matrix-Based R´enyi Entropy. Following Arefin et al. [2024], we measure the diversity of representations at layer ℓ by forming the Gram matrix K = Z(ℓ) Z(ℓ)⊤ ∈ Rt×t, where Z(ℓ) contains the d-dimensional representations of t tokens. Let {λi(K)}ti=1 be the eigenvalues of K. We define the α-order R´enyi entropy as Sα Z(ℓ) = 1−1α log ti=1 λ

α

i(K) tr(K)

. Each eigenvalue is normal-

ized by tr(K), ensuring the probabilities sum to 1. Higher Sα indicates greater variance (i.e., lower collapse).

Figure 2(a) shows that LIMe yields significantly higher matrix entropy of gathered MHA values compared with LLaMa and shows no significant difference when evaluating hidden states (see Figure 7(a)).

Layer-Wise Token Separability. To more directly evaluate the level of representation collapse, we replicate the methodology of Voita et al. [2019], extracting 1668 occurrences each of is, are, was, were from the FineWeb Edu corpus. To quantify information collapse, we train a linear fourway classifier (for is, are, was, were) on layer-wise representations. Figure 2(b) shows mean classification accuracies (with five-fold cross-validation) for value representations layer by layer. We observe that LIMe consistently exhibits higher classification accuracy than LLaMa, confirming that LIMe’s value representations avoid collapse. As hypothesized, hidden states became less separable

Hidden States Representations

Layer 2 Layer 4 Layer 7 Layer 12 Layer 15 Layer 16

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

LIMe(Ours)LLaMa

| | |
|---|---|
| | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |

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

| | | |
|---|---|---|
| | | |
| | | |

is are were was

(a)

Values Representations

Layer 2 Layer 4 Layer 7 Layer 12 Layer 15 Layer 16

| | | | |
|---|---|---|---|
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

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

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

LIMe(Ours)LLaMa

| | | | |
|---|---|---|---|
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

is are were was

(b)

- Figure 3: (a) t-SNE of similar tokens’ hidden states among layers. Although hidden states are not separable in later layers for both models, unlike LLaMA, LIMe can make updates attending to the previous representations, which leads to high values’ separability. (b) t-SNE of similar tokens’ values among layers shows higher separability for LIMe’s representations. See Section 5.2 for more details.

for LIMe, indicating that there was no need to store all necessary information in a single hidden state (see Figure 7(b)).

Additionally, we project representations into a two-dimensional space via t-SNE and visualize how well value states and hidden states can be clustered (Figure 3). In contrast to LIMe, deeper-layer representations in LLaMa for such similar tokens often collapse into overlapping regions, reflecting the inclination of the vanilla transformer to heavily compress relevant information into a single representation and therefore blur small yet important differences.

Discussion. Together, these results corroborate our theoretical motivation: by allowing each head to attend directly to earlier-layer representations, LIMe expands the overall representational capacity. This multi-layer routing reduces collapse in the values while freeing deeper hidden states from the burden of storing all lexical nuances—leading to higher overall entropy on values (Figure 2(a)) and improved model performance (Table 1). In the next section, we evaluate LIMe on synthetic benchmarks where the model’s ability to store complex information in limited state capacity is crucial.

Entropy of Values, 6 Operands

Accuracy on Arithmetic Expressions Task

1.0

LLaMa

LLaMa

2.8

LIMe

LIMe

0.8

2.7

2.6

0.6

Accuracy

Entropy

2.5

0.4

2.4

0.2

2.3

2.2

0.0

4 5 6 Num Operands

1 2 3 4

Layer

(a)

(b)

- Figure 4: (a) LIMe exhibits consistently higher entropy of value vectors across layers, particularly in the final layer, indicating reduced representation collapse compared to LLaMa. (b) On the Arithmetic Expressions task, LIMe significantly outperforms the LLaMa baseline, maintaining high accuracy even as the number of operands increases, while LLaMa’s performance deteriorates. For details, see Section 5.3.2.

- 5.3 Evaluating Representation Collapse on Synthetic Tasks

- 5.3.1 Planning and Search Capabilities

We fine-tune models on ProsQA (Proof with Search Question-Answering) [Hao et al., 2024]. Each ProsQA instance presents a set of fictional concepts described via natural-language conditions arranged in a DAG, requiring models to determine the veracity of a target statement by exploring multiple reasoning paths over the graph (examples in Appendix B). Unlike linear chain-of-thought methods [Wei et al., 2022], ProsQA demands maintaining and evaluating parallel hypothesis streams akin to breadth-first search in latent reasoning [Hao et al., 2024]. In our experiments we evaluate both fine-tuned models on ProsQA task via open-ended reasoning generation. LLaMA achieves 69.4% accuracy, meanwhile LIMe achieves 77.8% accuracy, outperforming LLaMA by 8.4%. Since correct prediction requires searching over paths in the graph of input statements, baseline transformers suffer representation collapse from storing multiple reasoning chains in their hidden states, particularly for longer inference sequences. LIMe mitigates this by distributing the reasoning process across layers early layers may store primitive inferences while deeper layers compose them, maintaining better separation between similar reasoning paths.

- 5.3.2 Arithmetic Expression Benchmark

Standard one-shot QA benchmarks mainly test final-token prediction, which can often be solved via shallow pattern matching or retrieval, masking the role of intermediate representation quality in reasoning. To isolate the impact of multi-step computation, we adopt the Arithmetic Expression Task (AET) [Arefin et al., 2024, Feng et al., 2023], a synthetic benchmark presenting expressions over integer operands with operators +,−,×,÷, along with solution steps and requiring the exact integer result. See examples in Appendix B.

Following Arefin et al. [2024], we generate 3 difficulty tiers comprising expressions with 4, 5, and

- 6 operands, accompanied by step-by-step solutions (details in Appendix A). While performing similarly to LLaMa on 4 operands, LIMe achieves significantly higher accuracy after increasing number of operands to 5 and 6 (Figure 4(b)). LIMe (71.6%) outperforms LLaMa (41.3%) by over 30% in accuracy on 6 operands. These results go along with lower representation collapse which is illustrated by higher entropy of value representations shown in Figure 4(a). Also, LIMe exhibits better separability of close numbers which leads to lower error rate in intermediate calculations, see Figure 8 in Appendix.

Arithmetic Expressions Task requires intermediate calculations to be performed correctly in order to get the correct final answer. The problem of representation collapse results in representations of close numbers being similar which leads to incorrect intermediate results, and thus the wrong final answer. Since LIMe has access to previous representations at each layer, it preserves finer numerical distinctions in comparison with standard transformer architectures like LLaMa. Moreover, LIMe has

Normalized Absolute Mean Weights of Representations

|[Figure 1]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

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

0.6

2345678910111213141516

0.5

0.4

Layer

0.3

0.2

0.1

0.0

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

Buffered Representation

- Figure 5: Mean retrieval weight for each buffered representation across subsequent layers. Larger diagonal values confirm reliance on the current residual stream, while the pronounced off-diagonal weights for the earliest buffers and the repeated reuse of intermediate ones show that the model systematically retrieves earlier features, providing auxiliary memory and helping to mitigate representation collapse. See Section 5.4 for more details.

ability to store information in earlier representations, i.e. performing computations at some early or intermediate layer, but using it further only in later layers, which also boosts its reasoning capabilities and leads to better results on tasks that require intermediate steps.

#### 5.4 Analyzing Learned Routings in LIMe

To understand how LIMe routes information across layers and thereby mitigates representation collapse, we inspect the learned router weights. Since the router weights can be both positive and negative—and because random initialization of the key, value, and output projections renders their sign semantically ambiguous—we analyze the absolute magnitudes of these weights to quantify each buffered representation’s relative contribution in a sign-agnostic manner.

For each layer ℓ ≥ 2, we take the absolute magnitude of its router weights, average over heads for each buffered representation j ≤ ℓ, and then normalize these averages per layer. The resulting heatmap in Figure 5 shows the normalized mean weight: cell (ℓ,j) measures the average contribution of the keys and values generated at layer j to the attention computation in layer ℓ. In a standard Transformer without routing, each layer would attend solely to its own keys and values, yielding a heatmap with ones on the diagonal and zeros elsewhere; LIMe departs markedly from this behavior.

Several clear patterns emerge:

- • Strong reliance on embeddings in early layers: Layers 2-4 allocate much of their attention to the buffered representations from the embedding layer. This corroborates the view that the initial attention layers focus on capturing local and morphological relationships among tokens, and that LIMe grants additional flexibility in reusing these low-level features.
- • Auxiliary memory via neighboring layers: Early and middle layers place a share of attention on the buffered KV states of its immediate predecessor. This indicates that they can treat them as an auxiliary memory bank, effectively extending the subspace of features it can manipulate by leveraging projections made by other heads.
- • Long-distance retrieval from early buffers: Higher layers also attend nontrivially to the first two buffered representations. The effect is especially pronounced in the final layers, suggesting that late-stage prediction benefits from revisiting the original token embeddings and shallow features.

By allowing flexible retrieval of features from arbitrarily distant layers, LIMe relieves each residual stream from having to carry the entire contextual signal forward. Instead, information can be distributed across a set of persistent buffers, preserving a richer and more diverse feature set throughout the network’s depth and thereby mitigating representation collapse. For the full, detailed set of normalized router weights, see Appendix Figure 9.

#### 5.5 Deep Networks Performance

Training Loss

LLaMa 32L

LIMe 32L

3.6

LLaMa 64L

LIMe 64L

LLaMa 128L

3.5

LIMe 128L

3.4

Loss

3.3

3.2

3.1

3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0

Tokens

1e9

- Figure 6: Training losses for deep architectures. The LIMe models consistently outperform their LLaMA counterparts across all depths, with LIMe with 64 layers outperforming LLaMa with 128 layers. See Section 5.5 for details.

Transformers scaled to increasing depths often suffer from representation collapse, which motivates our evaluation of LIMe in 32-, 64-, and 128-layer configurations. We compare LIMe against the baseline LLaMA, each using 8 attention heads per layer, and observe that LIMe outperforms LLaMA at every tested depth (Fig. 6). Furthermore, LIMe exhibits superior scaling behavior: as depth increases, its loss decreases more rapidly than LLaMA’s, implying that direct routing of earlier-layer features enhances the model’s effective representational capacity, whereas LLaMA’s single-stream residual architecture struggles to preserve fine-grained features across layers. Notably, a 64-layer LIMe model outperforms a 128-layer LLaMA model, despite the latter requiring roughly twice the FLOPs and parameters. This suggests that the optimal scaling strategy for transformers may deviate from conventional practice, potentially favoring much deeper models with smaller hidden dimensions. We leave further investigation of these scaling dynamics to future work.

### 6 Conclusion and Future Work

In this paper, we proposed Layer-Integrated Memory (LIMe), a lightweight extension to multi-head self-attention that enables each attention head to retrieve and integrate representations from all preceding layers. Through extensive experiments on language modeling, synthetic reasoning benchmarks, and deep transformer configurations, we demonstrated that LIMe (i) accelerates convergence in FLOPs by up to 15.3% and reduces perplexity by up to 1.15% compared to standard Transformer decoders, yields improvements of up to +8% on the challenging ProsQA task and +30% on Arithmetic Reasoning Task; (ii) mitigates representation collapse by preserving higher entropy in value vectors and maintaining token separability in deeper layers; and (iii) enables shallower models to match or exceed the performance of double-sized deeper baselines. Our analysis of the learned routing weights further revealed that LIMe systematically leverages both local and long-distance feature reuse, effectively distributing contextual information across layers without increasing the hidden-state size.

Limitations. While our method consistently yields better results on both benchmarks and language modeling tasks, it could lead to additional communication between GPUs in pipeline parallel setup. Also, vanilla implementation of the method has O(L2) asymptotic, and some heuristics proposed in Appendix D might be useful for scaling.

Looking forward, two research directions emerge as particularly promising. First, a comprehensive exploration of the width–depth trade-off in LIMe architectures could unveil optimal scaling regimes tailored to diverse tasks and computational budgets. Second, a rigorous theoretical analysis of the routing mechanism may inform principled designs for multi-layer memory, thereby enabling models to perform advanced latent-space reasoning grounded in Layer-Integrated Memory.

### References

Quentin Anthony, Stella Biderman, and Hailey Schoelkopf. Transformer math 101. blog.eleuther.ai/,

2023. URL https://blog.eleuther.ai/transformer-math/.

Md Rifat Arefin, Gopeshh Subbaraj, Nicolas Gontier, Yann LeCun, Irina Rish, Ravid Shwartz-Ziv, and Christopher Pal. Seq-vcr: Preventing collapse in intermediate transformer representations for enhanced reasoning. arXiv preprint arXiv: 2411.02344, 2024.

Ankur Bapna, Mia Chen, Orhan Firat, Yuan Cao, and Yonghui Wu. Training deeper neural machine translation models with transparent attention. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3028–3033, 2018. doi: 10.18653/v1/ D18-1338. URL https://arxiv.org/abs/1808.07561. arXiv:1808.07561.

Federico Barbero, Andrea Banino, Steven Kapturowski, Dharshan Kumaran, Jo˜ao G. M. Ara´ujo, Alex Vitvitskyi, Razvan Pascanu, and Petar Velickovi´c. Transformers need glasses! information over-squashing in language tasks. arXiv preprint arXiv: 2406.04267, 2024.

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.

DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Size Zheng, T. Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Liu, Xin Xie, Xingkai Yu, Xinnan Song, Xinyi Zhou, Xinyu Yang, Xuan Lu, Xuecheng Su, Y. Wu, Y. K. Li, Y. X. Wei, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Zheng, Yichao Zhang, Yiliang Xiong, Yilong Zhao, Ying He, Ying Tang, Yishi Piao, Yixin Dong, Yixuan Tan, Yiyuan Liu, Yongji Wang, Yongqiang Guo, Yuchen Zhu, Yuduan Wang, Yuheng Zou, Yukun Zha, Yunxian Ma, Yuting Yan, Yuxiang You, Yuxuan Liu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhewen Hao, Zhihong Shao, Zhiniu Wen, Zhipeng Xu, Zhongyu Zhang, Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, and Ziwei Xie. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv: 2405.04434, 2024.

Yanwen Fang, Yuxi Cai, Jintai Chen, Jingyu Zhao, Guangjian Tian, and Guodong Li. Cross-layer retrospective retrieving via layer attention. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=pvgEL1yS3Ql.

Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: A theoretical perspective, 2023. URL https://arxiv.org/ abs/2305.15408.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan,

Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzm´an, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur C¸elebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, V´ıtor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson,

Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models. arXiv preprint arXiv: 2407.21783, 2024.

Michael Hahn and Mark Rofin. Why are sensitive functions hard for transformers? In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14973–15008, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.acl-long.800. URL https://aclanthology.org/2024.acl-long.800/.

Shibo Hao, Sainbayar Sukhbaatar, Dijia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. arXiv preprint arXiv: 1512.03385, 2015.

Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks, 2018. URL https://arxiv.org/abs/1608.06993.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b. arXiv preprint arXiv: 2310.06825, 2023.

Guilherme Penedo, Hynek Kydl´ıcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale, 2024. URL https://arxiv.org/abs/2406.17557.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv: 2412.15115, 2024.

David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. Mixture-of-depths: Dynamically allocating compute in transformer-based language models, 2024. URL https://arxiv.org/abs/2404.02258.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adri`a Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, Alicia Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ambrose Slone, Ameet Rahane, Anantharaman S. Iyer, Anders Andreassen, Andrea Madotto, Andrea Santilli, Andreas Stuhlm¨uller, Andrew M. Dai, Andrew La, Andrew K. Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karakas, B. Ryan Roberts, Bao Sheng Loe, Barret Zoph, Bartlomiej Bojanowski, Batuhan Ozyurt,¨ Behnam Hedayatnia, Behnam Neyshabur, Benjamin Inden, Benno Stein, Berk Ekmekci, Bill Yuchen Lin, Blake Howald, Bryan Orinion, Cameron Diao, Cameron Dour, Catherine Stinson, Cedrick Argueta, C`esar Ferri Ram´ırez, Chandan Singh, Charles Rathkopf, Chenlin Meng, Chitta Baral, Chiyu Wu, Chris Callison-Burch, Chris Waites, Christian Voigt, Christopher D. Manning, Christopher Potts, Cindy Ramirez, Clara E. Rivera, Clemencia Siro, Colin Raffel, Courtney Ashcraft, Cristina Garbacea, Damien Sileo, Dan Garrette, Dan Hendrycks, Dan Kilman, Dan Roth, Daniel Freeman, Daniel Khashabi, Daniel Levy, Daniel Mosegu´ı Gonz´alez, Danielle Perszyk, Danny Hernandez, Danqi Chen, Daphne Ippolito, Dar Gilboa, David Dohan, David Drakard, David Jurgens, Debajyoti Datta, Deep Ganguli, Denis Emelin, Denis Kleyko, Deniz Yuret, Derek Chen, Derek Tam, Dieuwke Hupkes, Diganta Misra, Dilyar Buzan, Dimitri Coelho Mollo, Diyi Yang, Dong-Ho Lee, Dylan Schrader, Ekaterina Shutova, Ekin Dogus Cubuk, Elad Segal, Eleanor Hagerman, Elizabeth Barnes, Elizabeth Donoway, Ellie Pavlick, Emanuele Rodol`a, Emma Lam, Eric Chu, Eric Tang, Erkut Erdem, Ernie Chang, Ethan A. Chi, Ethan Dyer, Ethan J. Jerzak, Ethan Kim, Eunice Engefu Manyasi, Evgenii Zheltonozhskii, Fanyue Xia, Fatemeh Siar, Fernando Mart´ınez-Plumed, Francesca Happ´e, Franc¸ois Chollet, Frieda Rong, Gaurav Mishra, Genta Indra Winata, Gerard de Melo, Germ´an Kruszewski, Giambattista Parascandolo, Giorgio Mariani, Gloria Wang, Gonzalo Jaimovitch-L´opez, Gregor Betz, Guy Gur-Ari, Hana Galijasevic, Hannah Kim, Hannah Rashkin, Hannaneh Hajishirzi, Harsh Mehta, Hayden Bogar, Henry Shevlin, Hinrich Sch¨utze, Hiromu Yakura, Hongming Zhang, Hugh Mee Wong, Ian Ng, Isaac Noble, Jaap Jumelet, Jack Geissinger, Jackson Kernion, Jacob Hilton, Jaehoon Lee, Jaime Fern´andez Fisac, James B. Simon, James Koppel, James Zheng, James Zou, Jan Kocon, Jana Thompson, Janelle Wingfield, Jared Kaplan, Jarema Radom, Jascha Sohl-Dickstein, Jason Phang, Jason Wei, Jason Yosinski, Jekaterina Novikova, Jelle Bosscher, Jennifer Marsh, Jeremy Kim, Jeroen Taal, Jesse H. Engel, Jesujoba Alabi, Jiacheng Xu, Jiaming Song, Jillian Tang, Joan Waweru, John Burden, John Miller, John U. Balis, Jonathan Batchelder, Jonathan Berant, J¨org Frohberg, Jos Rozen, Jos´e Hern´andez-Orallo, Joseph Boudeman, Joseph Guerr, Joseph Jones, Joshua B. Tenenbaum, Joshua S. Rule, Joyce Chua, Kamil Kanclerz, Karen Livescu, Karl Krauth, Karthik Gopalakrishnan, Katerina Ignatyeva, Katja Markert, Kaustubh D. Dhole, Kevin Gimpel, Kevin Omondi, Kory W. Mathewson, Kristen Chiafullo, Ksenia Shkaruta, Kumar Shridhar, Kyle McDonell, Kyle Richardson, Laria Reynolds, Leo Gao, Li Zhang, Liam Dugan, Lianhui Qin, Lidia Contreras Ochando, Louis-Philippe Morency, Luca Moschella, Lucas Lam, Lucy Noble, Ludwig Schmidt, Luheng He, Luis Oliveros Col´on, Luke Metz, L¨utfi Kerem Senel, Maarten Bosma, Maarten Sap, Maartje ter Hoeve, Maheen Farooqi, Manaal Faruqui, Mantas Mazeika, Marco Baturan, Marco Marelli, Marco Maru, Mar´ıa Jos´e Ram´ırez-Quintana, Marie Tolkiehn, Mario Giulianelli, Martha Lewis, Martin Potthast, Matthew L. Leavitt, Matthias Hagen, M´aty´as Schubert, Medina Baitemirova, Melody Arnaud, Melvin McElrath, Michael A. Yee, Michael Cohen, Michael Gu, Michael I. Ivanitskiy, Michael Starritt, Michael Strube, Michal Swedrowski, Michele Bevilacqua, Michihiro Yasunaga, Mihir Kale, Mike Cain, Mimee Xu, Mirac Suzgun, Mitch Walker, Mo Tiwari, Mohit Bansal, Moin Aminnaseri, Mor Geva, Mozhdeh Gheini, Mukund Varma T., Nanyun Peng, Nathan A. Chi, Nayeon Lee, Neta Gur-Ari Krakover, Nicholas Cameron, Nicholas Roberts, Nick Doiron, Nicole Martinez, Nikita Nangia, Niklas Deckers, Niklas Muennighoff, Nitish Shirish Keskar, Niveditha Iyer, Noah Constant, Noah Fiedel, Nuan Wen, Oliver Zhang, Omar Agha, Omar Elbaghdadi, Omer Levy, Owain Evans, Pablo Antonio Moreno Casares, Parth Doshi, Pascale Fung, Paul Pu Liang, Paul Vicol, Pegah Alipoormolabashi, Peiyuan Liao, Percy Liang, Peter Chang, Peter Eckersley, Phu Mon Htut, Pinyu Hwang, Piotr Milkowski, Piyush Patil, Pouya Pezeshkpour,

Priti Oli, Qiaozhu Mei, Qing Lyu, Qinlang Chen, Rabin Banjade, Rachel Etta Rudolph, Raefer Gabriel, Rahel Habacker, Ramon Risco, Rapha¨el Milli`ere, Rhythm Garg, Richard Barnes, Rif A. Saurous, Riku Arakawa, Robbe Raymaekers, Robert Frank, Rohan Sikand, Roman Novak, Roman Sitelew, Ronan LeBras, Rosanne Liu, Rowan Jacobs, Rui Zhang, Ruslan Salakhutdinov, Ryan Chi, Ryan Lee, Ryan Stovall, Ryan Teehan, Rylan Yang, Sahib Singh, Saif M. Mohammad, Sajant Anand, Sam Dillavou, Sam Shleifer, Sam Wiseman, Samuel Gruetter, Samuel R. Bowman, Samuel S. Schoenholz, Sanghyun Han, Sanjeev Kwatra, Sarah A. Rous, Sarik Ghazarian, Sayan Ghosh, Sean Casey, Sebastian Bischoff, Sebastian Gehrmann, Sebastian Schuster, Sepideh Sadeghi, Shadi Hamdan, Sharon Zhou, Shashank Srivastava, Sherry Shi, Shikhar Singh, Shima Asaadi, Shixiang Shane Gu, Shubh Pachchigar, Shubham Toshniwal, Shyam Upadhyay, Shyamolima (Shammie) Debnath, Siamak Shakeri, Simon Thormeyer, Simone Melzi, Siva Reddy, Sneha Priscilla Makini, Soo-Hwan Lee, Spencer Torene, Sriharsha Hatwar, Stanislas Dehaene, Stefan Divic, Stefano Ermon, Stella Biderman, Stephanie Lin, Stephen Prasad, Steven T. Piantadosi, Stuart M. Shieber, Summer Misherghi, Svetlana Kiritchenko, Swaroop Mishra, Tal Linzen, Tal Schuster, Tao Li, Tao Yu, Tariq Ali, Tatsu Hashimoto, Te-Lin Wu, Th´eo Desbordes, Theodore Rothschild, Thomas Phan, Tianle Wang, Tiberius Nkinyili, Timo Schick, Timofei Kornev, Titus Tunduny, Tobias Gerstenberg, Trenton Chang, Trishala Neeraj, Tushar Khot, Tyler Shultz, Uri Shaham, Vedant Misra, Vera Demberg, Victoria Nyamai, Vikas Raunak, Vinay V. Ramasesh, Vinay Uday Prabhu, Vishakh Padmakumar, Vivek Srikumar, William Fedus, William Saunders, William Zhang, Wout Vossen, Xiang Ren, Xiaoyu Tong, Xinran Zhao, Xinyi Wu, Xudong Shen, Yadollah Yaghoobzadeh, Yair Lakretz, Yangqiu Song, Yasaman Bahri, Yejin Choi, Yichi Yang, Yiding Hao, Yifu Chen, Yonatan Belinkov, Yu Hou, Yufang Hou, Yuntao Bai, Zachary Seid, Zhuoye Zhao, Zijian Wang, Zijie J. Wang, Zirui Wang, and Ziyi Wu. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Trans. Mach. Learn. Res., 2023, 2023. URL https://openreview.net/forum?id=uyTL5Bvosj.

Rupesh Kumar Srivastava, Klaus Greff, and J¨urgen Schmidhuber. Highway networks. arXiv preprint arXiv: 1505.00387, 2015.

Ian Tenney, Dipanjan Das, and Ellie Pavlick. BERT rediscovers the classical NLP pipeline. In Anna Korhonen, David Traum, and Llu´ıs M`arquez, editors, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4593–4601, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1452. URL https:

##### //aclanthology.org/P19-1452/.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. NEURIPS, 2017.

Elena Voita, Rico Sennrich, and Ivan Titov. The bottom-up evolution of representations in the transformer: A study with machine translation and language modeling objectives. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4396–4406, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1448. URL https://aclanthology.org/D19-1448/.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. Ws, 2018.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. Neural Information Processing Systems, 2019.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.

Xixin Wu, Hui Lu, Kun Li, Zhiyong Wu, Xunying Liu, and Helen Meng. Hiformer: Sequence modeling networks with hierarchical attention mechanisms. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:3993–4003, 2023. doi: 10.1109/TASLP.2023.3313428. URL https://doi.org/10.1109/TASLP.2023.3313428.

###### Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, and Xun Zhou. Hyper-connections. arXiv preprint arXiv: 2409.19606, 2024.

### A Experimental Setup Details

Language Modeling. We observe that omitting weight decay on the LIMe router weights enjoys better performance and setting the router’s learning rate to 1 × 10−2 boosts model performance by speeding up router convergence and circuit formation. To preserve the standard Transformer’s

information flow at the start of the training, we initialize the slice Rℓ,h(ℓ)′,h = δh′,h (identity across heads). Other coefficients are initialized randomly via Kaiming uniform to stabilize mixtures at the start of the training. Random initialization of all weights resulted in worse overall model performance. Hyperparameter values are summarized in Table 2, and the detailed model architecture is given in Table 3. Additional training loss visualizations are available in Figure 11 for full attention and in Figure 10 for Grouped Query Attention.

We used NVIDIA H100 GPUs and spent about 2400 GPU-days on all experiments including preliminary research.

ProsQA Fine-Tuning. We fine-tune pretrained LLaMa 150M and LIMe 150M on approximately 18,000 sequences for 10 epochs. We use learning rate of 1 × 10−4 with linear decay and warmup during the first epoch, effective batch size is 128. Trained models are then evaluated on the test subset via open generation of reasoning steps and answers.

Arithmetic Expression Task. We train models and evaluate them on open-ended generation of solutions given initial expression, from which we extract the answers and calculate accuracy on the test subset. We train 4-layer models (with 4 attention heads and model dim is 32) on datasets with 50,000 samples per each number of operands for 200 epochs. Learning rate is 1 × 10−3 with linear decay.

#### Hyperparameter Value

Optimizer AdamW Learning Rate 0.001 LIMe Router Learning Rate 0.01 Weight Decay 0.1

- β1 0.9
- β2 0.95 ϵ 1 × 10−8 Scheduler cosine Warmup Steps 200 Min LR 1 × 10−6 Mixed Precision bf16 Gradient Clipping 1.0

Sequence Length 2048 Batch Size 1024 Training Steps 20,000

Table 2: Key training hyperparameters used in experiments.

#### Parameter Value

Vocab Size 50,257 Hidden Size 2048 Intermediate Size 8192 Number of Hidden Layers 16 Number of Attention Heads 32 Number of Key-Value Heads 8 (GQA) and 32 (otherwise) Tie Word Embeddings True

Table 3: Base model architecture at 1B scale.

### B Synthetic Benchmarks

ProsQA

Question: "Every shumpus is a rempus. Every shumpus is a yimpus. Every terpus is a fompus. Every terpus is a gerpus. Every gerpus is a brimpus. Alex is a rempus. Every rorpus is a scrompus. Every rorpus is a yimpus. Every terpus is a brimpus. Every brimpus is a lempus. Tom is a terpus. Every shumpus is a timpus. Every yimpus is a boompus. Davis is a shumpus. Every gerpus is a lorpus. Davis is a fompus. Every shumpus is a boompus. Every shumpus is a rorpus. Every terpus is a lorpus. Every boompus is a timpus. Every fompus is a yerpus. Tom is a dumpus. Every rempus is a rorpus. Is Tom a lempus or scrompus?"

Steps: "Tom is a terpus. Every terpus is a brimpus. Every brimpus is a lempus."

Answer: "Tom is a lempus."

Arithmetic Expression Task Input:

(7 + 5) ÷ (6 + 4 × 3 − 2 × 7) = Output:

12 ÷ (6 + 4 × 3 − 2 × 7) = 12 ÷ (6 + 12 − 2 × 7)

= 12 ÷ (18 − 2 × 7)

= 12 ÷ (18 − 14)

= 12 ÷ 4

= 3

Accuracy of Hidden States Classification

Entropy of Hidden States

1.00

LLaMa

LLaMa

LIMe

LIMe

6.8

0.99

0.98

6.6

Accuracy

Entropy

0.97

6.4

0.96

0.95

6.2

0.94

6.0

2 4 6 8 10 12 14 16

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

Layer

Layer

(a)

(b)

- Figure 7: (a) Matrix entropy of the hidden states across layers on the FineWeb Edu subset. We do not observe a significant difference between LIMe and LLaMa in this experiment. (b) Classification accuracy of the hidden states, with standard deviation, measured over five cross-validation folds. Because the hidden states in LIMe do not need to store all the information in the residual stream, they become less separable. See Section 5.2 for more details.

Values Representations

Layer 2 Layer 3 Layer 4

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

LIMe(Ours)LLaMa

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

1 2 3 4

(a)

Values Representations

Layer 2 Layer 3 Layer 4

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

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

LIMe(Ours)LLaMa

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

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

100 101 102 103

(b)

- Figure 8: t-SNE of close numbers’ values representations of models trained on Arithmetic Expressions Task. (a) For 1,2,3,4. (b) For 100,101,102,103. See Section 5.3.2.

Normalized Absolute Router Weights

- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16

[Figure 2]

0.8

0.6

Buffered Representation Layer

0.4

0.2

0.0

1 3 5 7 9 11 13 15

- Figure 9: Magnitudes of router weights averaged among buffered heads and normalized among buffered layers. Each cell represents ratio of attention for each buffered representation in the specific head.

Training Loss

2.75

LLaMa GQA

LIMe GQA

2.70

2.65

2.60

Loss

2.55

2.50

2.45

10 15 20 25 30 35 40

Tokens

1e9

- Figure 10: Training loss per tokens trained on for LLaMa and LIMe with GQA. It shows that LIMe is more data efficient. See Section 5.1 for more details.

0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8

FLOPs

2.40

2.45

2.50

2.55

2.60

2.65

2.70

Loss

1e20

Training Loss

LLaMa

LIMe

(a)

10 15 20 25 30 35 40

Tokens

2.40

2.45

2.50

2.55

2.60

2.65

2.70

Loss

1e9

Training Loss

LLaMa

LIMe

(b)

- Figure 11: Training loss for LLaMa and LIMe without GQA. (a) shows that LIMe has a substantially lower loss with a similar amount of FLOPs. (b) shows that LIMe is more data efficient. See Section 5.1 for more details.

### C Additional Benchmarks

Model COPA MultiRC WiC QNLI WNLI Avg LLaMA 75.80 43.24 50.00 49.49 51.27 53.96

HC 74.00 54.34 49.72 49.43 56.34 56.77 LIMe 75.20 56.15 50.44 51.43 56.06 57.86

- Table 4: GLUE and SuperGLUE benchmarks accuracies (%) on 1B GQA models (3-shot), with average over the five tasks.

Model ARC-E ARC-C HellaSwag OBQA Avg LLaMA 70.45 38.70 52.55 37.68 49.85

HC 71.15 37.63 54.04 40.08 50.73 LIMe 71.15 39.30 52.85 39.68 50.75

- Table 5: QA benchmarks accuracies (%) on 1B GQA models (3-shot), with average over the four tasks.

Model KV Induction DisambQA LD-5 LD-3 IR CO Avg LLaMA 45.94 54.20 30.16 19.44 33.60 12.94 16.97 30.46

HC 51.68 51.59 26.20 19.92 33.87 15.29 18.48 31.00 LIMe 55.64 55.36 30.23 20.72 34.47 14.82 17.39 32.66

- Table 6: Accuracies (%) of 3-shot 1B GQA models on BIG-Bench tasks: Key–Value Maps (KV), Mathematical Induction, Disambiguation QA, Logical Deduction for 3 and 5 objects (LD-3, LD-5), Implicit Relations (IR), and Reasoning About Colored Objects (CO).

### D Router Ablation

We conduct an ablation study to assess the importance of learning full per-layer, per-head router weights in LIMe. Specifically, we compare the standard LIMe routing against several constrained variants on the 150M-parameter model, evaluating their impact on perplexity:

- • Fixed Average (average): Aggregates all buffered Key–Value representations via a uniform average, without any learned head-specific weighting.
- • Recent–j (last-j): Restricts each layer ℓ to attend only to the most recent min(ℓ,j) buffered representations; router weights for these representations are learned.
- • Initial–j (first-j): Restricts each layer ℓ to attend only to the first min(ℓ,j) buffers plus the immediately preceding layer; router weights for these are learned.

Model Perplexity Change to LIMe

LLaMA 16.4611 +3.36% LIMe average 16.4611 +3.36% LIMe last-2 16.2810 +2.22% LIMe last-4 16.1675 +1.51% LIMe last-6 16.1351 +1.31% LIMe first-2 15.9746 +0.30% LIMe first-4 15.9586 +0.20% LIMe first-6 15.9906 +0.40% LIMe 15.9267 —

- Table 7: Impact of constrained routing schemes on validation perplexity for the 150M-parameter model. Table reports perplexity for each scheme and the relative change with respect to the full LIMe model. The average variant fails to improve over the LLaMA baseline, indicating that uniform pooling of past representations is insufficient. Constraining attention to fixed windows of layers (last-j and first-j) yields modest gains but still underperforms the unrestricted router. By contrast, the full LIMe routing achieves the lowest perplexity (15.9267), corresponding to a 3.36% reduction relative to LLaMA, thereby confirming the necessity of learning full, per-head, per-layer router weights for optimal performance.

### E LIMe Pseudocode

- 1 class LIMeRouter(nn.Module):
- 2 def __init__(self , config , layer_idx):
- 3 super(). __init__()
- 4 bound = math.sqrt(
- 5 3 / (layer_idx + 1) * config.num_kv_heads
- 6 )
- 7 weights = torch.empty(
- 8 config.num_kv_heads ,
- 9 (layer_idx + 1) * config.num_kv_heads ,
- 10 ).uniform_(-bound , bound)
- 11 weights[:, -config.num_kv_heads:] = torch.eye(
- 12 config.num_kv_heads
- 13 )
- 14 self.weights = nn.Parameter(weights)
- 15
- 16 def forward(self , kv_buffer):
- 17 # kv_buffer shape = [(layer_idx + 1) * kv_h , 2 * b * t * hd]
- 18 return self.weights.mm(kv_buffer)
- 19
- 20
- 21 class LIMeAttention(LlamaAttention):
- 22 def __init__(self , config , layer_idx):
- 23 super(). __init__(config , layer_idx)
- 24 if layer_idx > 0:
- 25 self.lime_router = LIMeRouter(config , layer_idx)
- 26
- 27 def forward(self , hidden_states , kv_buffer):
- 28 query_states = self.q_proj(hidden_states).reshape(b, h, t, hd)
- 29 key_states = self.k_proj(hidden_states).reshape(b, kv_h , t, hd)
- 30 value_states = self.v_proj(hidden_states).reshape(b, kv_h , t, hd)
- 31 kv_buffer.add_(key_states , value_states)
- 32 if self.layer_idx > 0:
- 33 key_states , value_states = self.lime_router(kv_buffer)
- 34 attn_output = scaled_dot_product_attention(
- 35 query_states , key_states , value_states
- 36 )
- 37 attn_output = self.o_proj(
- 38 attn_output.transpose(1, 2). reshape(b, t, -1)
- 39 )

- 40 return attn_output , kv_buffer
- 41
- 42
- 43 class LIMeLayer(LlamaDecoderLayer):
- 44 def __init__(self , config , layer_idx):
- 45 super(). __init__(config , layer_idx)
- 46 self.self_attn = LIMeAttention(config , layer_idx)
- 47
- 48 def forward(self , hidden_states , kv_buffer):
- 49 residual = hidden_states
- 50 hidden_states = self.input_layernorm(hidden_states)
- 51 attn_out , kv_buffer = self.self_attn(hidden_states , kv_buffer)
- 52 hidden_states = residual + attn_out
- 53
- 54 residual = hidden_states
- 55 hidden_states = self.post_attention_layernorm(hidden_states)
- 56 hidden_states = self.mlp(hidden_states)
- 57 hidden_states = residual + hidden_states
- 58
- 59 return hidden_states , kv_buffer
- 60
- 61
- 62 class LIMeModel(LlamaModel):
- 63 def __init__(self , config):
- 64 super(). __init__(config)
- 65 self.layers = [
- 66 LIMeLayer(config , i) for i in range(config.num_hidden_layers)
- 67 ]
- 68
- 69 def forward(self , input_ids):
- 70 hidden_states = self.embed_tokens(input_ids)
- 71 kv_buffer = init_kv_buffer()
- 72 for layer in self.layers:
- 73 hidden_states , kv_buffer = layer(hidden_states , kv_buffer)
- 74 return hidden_states

### F Efficiency

###### MHA Model # Parameters (B) FLOPs (T)

LLaMa 1.07607 2.7615

GQA

LIMe 1.07608(+0.00%) 2.7638(+0.08%) HC 1.07640(+0.03%) 2.7701(+0.31%)

LLaMa 1.17674 2.9679

Full

LIMe 1.17687(+0.01%) 3.0041(+1.22%) HC 1.17706(+0.03%) 2.9764(+0.29%)

- Table 8: Model size (# parameters, in billions) and forward FLOPs for LIMe and Hyper-connections (HC) relative to LLaMa under grouped-query attention (GQA) and full attention. We used torch.jit.trace to record all operations and estimated FLOPs via the fvcore library, based on tensor shapes and ATen operators. Total training FLOPs are approximated as 3× forward FLOPs, accounting for both forward and backward passes [Anthony et al., 2023].

Train Peak Memory (GB)

MHA RO Model Step Time (ms)

LLaMa 65.770 16.035

+

LIMe 66.533(+1.16%) 16.035(+0.00%) HC 81.003(+23.16%) 16.040(+0.03%)

GQA

LLaMa 66.404 20.489

–

LIMe 67.449(+1.57%) 20.490(+0.00%) HC 83.265(+25.39%) 21.693(+5.88%)

LLaMa 69.776 17.535

+

LIMe 77.093(+10.49%) 17.537(+0.01%) HC 84.990(+21.80%) 17.540(+0.03%)

Full

LLaMa 70.258 22.364

–

LIMe 77.607(+10.46%) 22.367(+0.01%) HC 86.314(+22.85%) 23.007(+2.87%)

- Table 9: Per-step latency and peak GPU memory usage of LIMe and Hyper-connections (HC) in comparison to LLaMa under grouped-query attention (GQA) and full attention (Full), measured with PyTorch Inductor in default (–) and reduced-overhead (+) modes. LIMe incurs only minimal overhead—effectively negligible in reduce-overhead mode—whereas HC exhibits substantially higher increases in both time and memory.

### G LIMe Visualisation

[Figure 3]

[Figure 4]

|[Figure 5]|
|---|

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

|[Figure 14]|
|---|

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

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

Figure 12: LIMe routing scheme.

