# arXiv:2505.18842v6[cs.CL]7May2026

## v1: Learning to Point Visual Tokens for Multimodal Mathematical Grounded Reasoning

##### Jiwan Chung∗♠† Junhyeok Kim♠† Siyeol Kim♠ Jaeyoung Lee♣ Minsoo Kim♠ Youngjae Yu♠

♠ Yonsei University ♣ Seoul National University

jiwan.chung.research@gmail.com

#### Abstract

When thinking with images, humans rarely rely on a single glance: they revisit visual evidence while reasoning. In contrast, most Multimodal Language Models encode an image once to key-value cache and then reason purely in text, making it hard to re-ground intermediate steps. We empirically confirm this: as reasoning chains lengthen, models progressively lose focus on relevant regions. We introduce v1, a lightweight extension for active visual referencing via point-and-copy: the model selects relevant image patches and copies their embeddings back into the reasoning stream. Crucially, our point-and-copy mechanism retrieves patches using their semantic representations as keys, ensuring perceptual evidence remains aligned with the reasoning space. To train this behavior, we build v1g, a dataset of 300K multimodal reasoning traces with interleaved grounding annotations. Across multimodal mathematical reasoning benchmarks, v1 consistently outperforms comparable baselines.

#### 1 Introduction

Multimodal mathematical problem solving often requires repeated access to diagrams: intermediate steps depend on localized cues (e.g., angles, tangency points, symmetries) that are revisited as reasoning unfolds. Cognitive studies report similar visual revisitation (Cox, 1999; Brun et al., 2016; Chu et al., 2017; Kozhevnikov et al., 2002).

Recent advances in Multimodal Large Language Models (MLLMs) (Liu et al., 2023; Bai et al., 2025; Chen et al., 2025) have shifted focus toward complex multi-step reasoning (Xu et al., 2025; Yao et al., 2024; Sun et al., 2025; Huang et al., 2025) rather than direct recognition. Mathematical reasoning (Lu et al., 2024; Zhang et al., 2024a; Wang et al., 2024a) is a standard benchmark for this capability; it requires long-horizon grounding and unambiguous solutions, making it a natural testbed for grounded multimodal inference.

However, current MLLMs process images only once at the start and, due to causal masking, thereafter reason mainly over the frozen key–value cache of visual embeddings. This limits their ability to actively revisit visual context as inference unfolds. In practice, this constraint manifests as two forms of visual grounding decay. First, attention to all image tokens steadily weakens as reasoning chains extend. Second, even the relative weight on relevant tokens declines, reducing the model’s ability to focus on the most informative regions (Section 3). These effects highlight the need for mechanisms that let models explicitly re-access visual information to keep reasoning grounded in the input.

We propose v1, a simple yet effective extension that equips MLLMs with a point-and-copy mechanism for dynamically referencing input visual tokens during multimodal mathematical reasoning (Figure 1). Specifically, we augment the model with an additional pointing head that outputs a probability distribution over the input image token positions, alongside the standard vocabulary logits. When an image token is selected, its embedding is copied

∗† Equal contribution.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

- Figure 1: Pure text-based reasoning vs. v1 during inference. v1 can actively re-access visual context by point-and-copying relevant image regions throughout reasoning process.

and injected as the next-step input, allowing the model to dynamically retrieve and reuse visual information during generation.

Our approach is readily compatible with popular MLLM architectures (Liu et al., 2023; Wang et al., 2024b; Chen et al., 2025) that operate on continuous image embeddings. Unlike methods that attempt to generate new image tokens (Chameleon Team, 2024) which require full-scale pretraining for image generation capacity, our method simply reuses existing input embeddings through pointing and copying. The only additional parameters are lightweight linear heads, incurring minimal computational overhead.

To train v1, we construct v1g, a dataset of 300K multimodal mathematical reasoning paths with interleaved grounding annotations, where each reasoning step is explicitly linked to a corresponding image region. The construction pipeline comprises three stages: (1) oversampling diverse reasoning traces from an MLLM, (2) extracting visual queries and retrieval steps from the traces using an LLM-guided decomposition process, and (3) grounding each visual reference by associating it with a bounding box in the input image. The pipeline is fully automated, leveraging the generative and interpretive capabilities of LLMs to produce high-quality, grounded reasoning trajectories at scale.

We evaluate v1 on three established multimodal mathematical reasoning benchmarks: MathVista (Lu et al., 2024), MathVision (Wang et al., 2024a), and MathVerse (Zhang et al., 2024a), following prior work (Yao et al., 2024; Sun et al., 2025; Huang et al., 2025). v1 demonstrates strong performance across all benchmarks, outperforming existing models of comparable scale and approaching the capabilities of much larger models, particularly on tasks requiring precise visual grounding and iterative reference to localized regions. These results suggest that dynamic access of visual input at inference time can improve multimodal mathematical reasoning capabilities.

Our contributions are:

- • v1 model: a lightweight MLLM extension that helps mitigate visual grounding decay through a point-and-copy mechanism, enabling dynamic visual reference.
- • v1g dataset: a large-scale training set with 300K multimodal mathematical reasoning traces and fine-grained visual grounding.
- • Empirical findings: extensive experiments and ablations on multimodal mathematical reasoning benchmarks, showing that dynamic visual reference and the point-and-copy design both mitigate visual grounding decay and lead to better multimodal reasoning.

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

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

- Figure 2: Inference process of v1. At each step, the MLLM encodes the multimodal context and generation history into token representations. For the last token (e.g., "<region>"), (a) a copy head projects its representation and computes logits against image patch embeddings, (b) a language head produces logits over the vocabulary, and (c) the two are concatenated to form the final distribution. If a patch is chosen, its embedding is copied as the next token input, enabling v1 to reference image regions one patch at a time.

#### 2 Related Work

##### 2.1 Reasoning in Large Language Models

Reasoning in text-only large language models. Recent advances have significantly improved reasoning in text-only LLMs. OpenAI’s o1 model (Jaech et al., 2024) had achieved state-of-the-art performance on mathematical benchmarks (Lightman et al., 2023; Cobbe et al., 2021), motivating follow-up work that induces stronger reasoning through reinforcement learning and reflective Chain-of-Thought (Guo et al., 2025), as well as inference-time scaling (Muennighoff et al., 2025). While effective in text-only settings, extending these approaches to multimodal reasoning remains challenging.

Reasoning in multimodal large language models. Multimodal reasoning poses challenges beyond text-only inference, requiring both visual perception and the integration of visual inputs into multi-step reasoning. Early approaches (Liu et al., 2023; Chen et al., 2024; Zhang et al., 2024b; Yang et al., 2023) typically convert images into textual descriptions for downstream tasks. More recently, inspired by Chain-of-Thought prompting in LLMs, models such as LLaVA-CoT (Xu et al., 2025), TVC (Sun et al., 2025), and Mulberry (Yao et al.,

- 2024), among others (Huang et al., 2025; Deng et al., 2025; Meng et al., 2025; Wu et al., 2025; Wang et al., 2025), extend CoT reasoning to multimodal settings and achieve strong results on benchmarks such as MathVista (Lu et al., 2024) and MathVision (Wang et al., 2024a). However, these models treat visual inputs as fixed context and perform reasoning entirely in the text space, lacking explicit mechanisms to re-access visual representations.

##### 2.2 Implementing Visual Reference

A growing body of work equips MLLMs with mechanisms to interact with images more explicitly during reasoning, beyond treating the visual input as a single, static context. Existing approaches span a range of design choices, including predicting structured spatial references (e.g., boxes or points) to retrieve localized evidence (Gupta & Kembhavi, 2023; Wu & Xie, 2023; Chung et al., 2025; Jiang et al., 2025), generating intermediate visual artifacts (e.g., sketches or diagrams) to externalize intermediate states (Hu et al., 2024; Borazjanizadeh et al., 2025; Li et al., 2025; Ma et al., 2025), and hybrid tool-use pipelines that interleave perception and reasoning steps (Gupta & Kembhavi, 2023; Hu et al., 2024).

Our work is most closely related to methods that enable visual re-access during decoding. We extend the pointer-generator idea (See et al., 2017) from selective copying in text to

Sum of the attention scores of full image

0.4

Layer

layer 2

layer 14 layer 27

0.3

Attentionscore

0.2

0.1

0.0

0 100 200 300 400 500

Generation step

Ratio of bounded region to full image

Layer

1.0

layer 2

layer 14 layer 27

0.9

Ratio

0.8

0.7

0.6

0 100 200 300 400 500

Generation step

- Figure 3: Left: Cumulative attention across all visual tokens, showing a gradual decrease in overall attention to the input image tokens. Right: Attention dynamics during reasoning, showing that semantically important visual regions receive disproportionately low attention, suggesting inefficient grounding.

selectively reusing visual token embeddings, yielding an interpretable and lightweight mechanism to directly point at the visual evidence.

#### 3 Visual Grounding Decays During Reasoning

To examine how visual attention evolves at each generation step, we use RefCOCO (Kazemzadeh et al., 2014), a visual reference expression generation benchmark where the task is to create a caption that uniquely identifies a target region specified by a bounding box. Although RefCOCO generally involves shorter generations than mathematical reasoning, it provides a controlled setting for measuring attention dynamics over visual regions. We analyze the TVC-7B model (Sun et al., 2025) on the RefCOCO testA split, examining attention from the most recent token to all image tokens across early, middle, and late transformer layers (layers 2, 14, and 27).

Figure 3 left tracks total attention to image tokens and shows a steady decline across decoding steps, indicating a reduced reliance on visual grounding as generation proceeds.

- Figure 3 right measures attention to the target region by computing the ratio of mean attention on bounding-box image tokens to that over all image tokens. While layers 14 and 27 initially place greater emphasis on the target region, attention converges to a ratio of ∼0.8 by mid-generation, suggesting weakened relative focus on salient visual tokens.

These results suggest visual grounding decay, where attention to relevant visual content diminishes during extended generation. This limitation is especially relevant for multimodal reasoning tasks where intermediate steps may benefit from revisiting visual evidence, thereby motivating architectures with dynamic visual access during inference.

#### 4 Method

At inference time, v1 operates as a single, self-contained MLLM: all textual reasoning, visual localization, and point-and-copy operations are performed by the same fine-tuned backbone without invoking external models or tools.

##### 4.1 Preliminary: Pointing for Language Generation

Formulation. We formulate a conditional next-token prediction objective, as commonly adopted in modern multimodal large language models (MLLMs). Given a sequence of continuous input representations c (e.g. embedded text tokens or visual features) the model is trained to autoregressively predict the discrete next token xt conditioned on the input c

and previously generated tokens x<t:

p(x1, . . . , xT | c) =

T

### ∏

p(xt | c, x1, . . . , xt−1)

t=1

The continuous input sequence c may include a heterogeneous mixture of modality-specific features, such as embedded discrete text tokens or continuous visual embeddings produced by image encoders (e.g. CLIP (Radford et al., 2021)). This general formulation covers a wide range of multimodal architectures such as LLaVA (Liu et al., 2023) and Qwen-VL (Bai et al.,

- 2025), which use continuous input image representations.

Pointing. For visually grounded reasoning, rather than generating new visual tokens, we instead teach the model to point to its position within the input image sequence it already understands, thereby referencing it explicitly.

The pointing mechanism we examine was first introduced by the pointer-generator network (See et al., 2017) in text summarization research. In the pointer-generator network, the input context sequence c also consists of discrete tokens within the vocabulary space V, unlike our setup. The model dynamically mixes two distributions at each decoding step t: (1) a generation distribution over the vocabulary Pgen(xt), and (2) a copy distribution Pptr(xt) over input tokens. The final output probability is given by a gated mixture:

p(xt | c, x<t) = λt Pgen(xt) + (1 − λt) Pptr(xt), λt ≡ λ(xt | c, x<t),

(1)

where λ ∈ [0,1] is a learnable scalar gate that controls the trade-off between generating a new token and copying one from the input.

The pointer distribution is obtained via attention over the encoder representations:

exp score(ht, ck) ∑k′ exp score(ht, ck′)

α(tk) =

,

(2)

α(tk).

Pptr(xt = w) = ∑

k: wk=w

where ht is the decoder hidden state at step t, wk the token at position k, and score denotes a standard attention scoring function (e.g., dot-product or additive). We generalize this formulation beyond the original implementation to arbitrary autoregressive language models for explanatory purposes.

Discrete targets. The above formulation constrains the pointing targets to be within the discrete vocabulary space V. This prevents application to general MLLMs as the multimodal inputs often consist of continuous feature sequences (Liu et al., 2023; Bai et al., 2025).

##### 4.2 v1: Pointing for Multimodal Grounded Reasoning

To overcome these limitations, we introduce v1, a lightweight extension to autoregressive MLLMs that enables explicit visual grounding by pointing to continuous input embeddings. v1 augments the vocabulary with pointer tokens that allow the model to either generate text or copy visual content during inference. All reasoning and grounding are handled within a single fine-tuned backbone (e.g., Qwen2.5-VL), without external modules or auxiliary grounding networks. This design supports unified inference over discrete and continuous modalities without modifying the core architecture (Figure 2).

Pointing to continuous inputs. The gated mixture formulation of See et al. (2017) is unfit for continuous inputs as image embeddings, as such inputs lack discrete mappings to vocabulary tokens V. To enable pointing in this setting, we extend the output space to include references to positions in the continuous input. Specifically, we define the augmented output space as V¯ = V ∪ C, where C = {c1, c2, . . . , cK} denotes the set of continuous input

Reasoning MathVista MathVision MathVerse Average

Model Size

Only mini mini full mini mini full

Qwen2-VL (Wang et al., 2024b) 7B ✗ 60.9 - 16.3 24.6 - 20.5 Qwen2-VL (Wang et al., 2024b) 72B ✗ 69.7 - 26.6 36.2 - 31.4 Qwen2.5-VL (Bai et al., 2025) 7B ✗ 67.8 23.6 - 44.5 45.3 Qwen2.5-VL (Bai et al., 2025) 72B ✗ 74.8 39.8 - 57.6 57.4 InternVL2.5 (Chen et al., 2025) 8B ✗ 64.4 22.0 19.7 39.5 41.9 29.6 InternVL2.5 (Chen et al., 2025) 78B ✗ 72.3 34.9 32.2 51.7 53.0 42.0 GPT-4o (Hurst et al., 2024) - ✗ 63.8 - 30.4 50.2 - 40.3

LLaVa-CoT (Xu et al., 2025) 11B ✓ 54.8 16.3 - 33.9 35.0 Mulberry (Yao et al., 2024) 7B ✓ 63.1 - - 39.6 - TVC (Sun et al., 2025) 7B ✓ 68.1 - 22.7 38.9 - 30.8 TVC (Sun et al., 2025) 72B ✓ 72.2 - 41.9 48.8 - 45.4 QVQ-72B-preview (Qwen Team, 2024) 72B ✓ 71.4 35.9 - 41.5 49.6 Base (Qwen2.5-VL) 7B ✗ 67.8 23.6 - 44.5 45.3 Text-Only (TVC) 7B ✓ 68.1 - 22.7 38.9 - 30.8 Ours 7B ✓ 68.6 34.5 28.1 48.6 50.6 38.4

###### ↰

Inference w/o Pointing 7B ✓ 60.0 25.3 23.7 33.6 39.6 28.7

Table 1: Results on multimodal mathematical reasoning tasks. We report MathVision on both the mini and full subsets to include additional baselines. Average (full) uses MathVision full in place of MathVision mini. Some scores are absent in the references (“-“).

vectors (e.g. embeddings of the input image patches). This formulation allows the model to generate either a vocabulary token or a pointer to a specific continuous input. We denote a pointer to input vector ck as ⟨ptr : ck⟩, which is treated as a discrete token during decoding. At each decoding step t, the model computes two distributions: (1) a generation distribution over the vocabulary V, producing logits logitgen ∈ R|V|, and (2) a pointing distribution over the input positions C, producing logits logitptr ∈ RK. The final output logits are defined as:

logitt = logitgen ∥ logitptr ∈ R|V|+K

where [· ∥ ·] denotes concatenation. Pointing logits are computed by attending over the image sequence:

Lq(ht) · Lk(ck)⊤ √D

logitptr(k) =

where ht is the decoder hidden state at step t, Lq and Lk are learned linear projections, and the scaling factor √D follows standard attention practice. We omit the gating module λ as the logit types are defined over disjoint spaces and do not require interpolation.

During inference, if the model selects an index in V, the next token xt is emitted as the corresponding vocabulary token. If the model selects an index k ∈ C, the token is represented as

a pointer xt = ⟨ptr : ck⟩. On the subsequent decoding step, the input embedding at position t is replaced with the continuous vector ck, enabling the model to attend directly to the referenced content.

##### 4.3 Annotating Visually-grounded Reasoning Data

To train v1, we require fine-grained multimodal reasoning traces in which each step is grounded to specific visual evidence. To this end, we construct v1g, a dataset of 300K multimodal reasoning paths with interleaved grounding annotations. Each trajectory includes a sequence of reasoning steps, where textual inferences are explicitly linked to corresponding image regions.

We generate the dataset using a fully automated three-stage pipeline: (1) oversampling textual reasoning paths from a pretrained MLLM, (2) decomposing each path into discrete visual queries and retrieval steps via an LLM, and (3) grounding each visual reference by aligning it with a bounding box in the input image (see Section C and Figure 6 for details).

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

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

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

- Figure 4: Qualitative comparison on MathVision. v1’s dynamic grounding helps to solve both bar graph and spatial reasoning tasks, while LLaVA-CoT misinterprets visual context.

Constructing base reasoning traces. As a seed to our grounded corpus, we adopt the training set of TVC (Sun et al., 2025), which consists of reasoning traces generated from the QvQ model (Qwen Team, 2024).

Decomposing reasoning traces into visual reference steps. We use a strong off-the-shelf LLM, Gemini-2.0-flash (Google, 2025), to extract visual grounding cues from text-based reasoning traces. The model collects and rewrites each visual reference as a detect call, which takes a short natural-language description and returns the corresponding image region. Retrieved regions are cached and assigned symbolic identifiers such as <objX> in order of appearance. The LLM also produces a key-value list of visual components, where each key is a unique descriptive reference used in later steps. We guide this process with domain-specific few-shot prompts, described in Appendix C. We then filter out invalid outputs, including mismatches between references and retrieved objects, non-unique labels, insufficient object count (≤ 2), and ill-formed reasoning. After filtering, about 82% of samples are retained.

Grounding visual references to image regions. Visual grounding remains difficult in multimodal reasoning, especially for abstract or unconventional regions such as charts, geometry, and medical scans. Existing grounding models often perform poorly in these settings (Steiner et al., 2024; Xiao et al., 2024), particularly when the target is abstract or symbolic, such as an angle or geometric relation.

To exploit the implicit visual grounding behavior in MLLMs, we build on Qwen2.5-VL’s visual grounding capacity (Bai et al., 2025). However, rather than relying on its coordinate generation interface, we estimate the model’s visual focus using a relative attention mechanism inspired by Zhang et al. (2025), in order to better handle non-natural domains and symbolic cues. The corresponding algorithm is detailed in Appendix E. We validate the quality of this pipeline through human evaluation in Appendix D. This produces a curated training set of ~300K grounded examples.

- 5 Implementation Details

Preprocessing. Given interleaved images, text, and bounding box annotations, we first convert each image into a flattened sequence of patches following the backbone’s patchification scheme (e.g., Qwen2.5-VL (Bai et al., 2025)). Each bounding box is then mapped to a sequence of pointer tokens (e.g., <ptr4>, ..., <ptr32>) that index the enclosed patches. These tokens are added to the tokenizer vocabulary without changing the original embedding

Layer 2

Layer 14

Layer 27

Copy Attention Input Attention Copy Sequence

0.4

0.3

AttentionScore

0.2

0.1

0.0

0 200 400 600 800 1000 Distance From First Copy

0 200 400 600 800 1000 Distance From First Copy

0 200 400 600 800 1000 Distance From First Copy

- Figure 5: Comparison of attention to copy tokens vs. original visual tokens. Layer-wise sum of attention scores directed to copy tokens and their corresponding original visual input tokens from a v1. Copy token intervals are shown in yellow.

table or generation head. Instead, during preprocessing, their embeddings are replaced by the corresponding image patch embeddings before entering the transformer. The resulting input is a unified sequence of text tokens, pointer tokens, and image patch embeddings.

Model. We instantiate v1 with Qwen2.5-VL as the MLLM backbone and add only two lightweight linear layers: a pointing query head Lq ∈ RD×D and a pointing key head Lk ∈ RD×D, where D is the latent dimensionality. Both are initialized as identity matrices scaled by 1/√D, so their initial effect on the output distribution is minimal. This is well suited to our setting because the pretrained backbone already provides meaningful generative likelihoods Pgen, while the pointing module selects at most one position per timestep from Pptr, allowing smooth early training without catastrophic forgetting.

Training. Given the dual-nature output space comprising a generative vocabulary V and a pointing reference set K, we incorporate z-loss regularization to stabilize the softmax partition function, following Chameleon Team (2024). Specifically, we regularize the logpartition function Z = ∑j exj in the softmax σ(x)i = exi/Z by introducing a z-loss term Lz = λ log Z¯, where λ = 10−5. To reduce computational overhead, we approximate Z using a top-k = 40 partition function Z¯ = ∑j∈TopK(x) exj. This approximation enables efficient and numerically stable training in large-output-space settings. Further details are in Section A.

Inference. At each decoding step xt, v1 uses two extra caches: (1) key features Lk(c) at image patch positions to compute pointing logits logitptr, and (2) the corresponding image patch features for copying. We implement these by extending the HuggingFace Transformers key-value cache (Wolf et al., 2020). The overhead is small, since only a subset of tokens from a single layer is stored.

The added model cost is also minimal: v1 introduces only one linear projection over image embeddings, negligible relative to the language head. During inference, copied visual tokens increase sequence length, but with revisit suppression they typically stay below 60% of the text token count, yielding an average total length of about 1.6× text-only reasoning. No external models or executors are used.

#### 6 Experiments

##### 6.1 Downstream Evaluation

Setup. We use three representative multimodal mathematical reasoning benchmarks: MathVista (mini) (Lu et al., 2024), MathVision (mini/full) (Wang et al., 2024a), and MathVerse (mini) (Zhang et al., 2024a).

We compare our method against both general-purpose and reasoning-specialized MLLMs. General MLLMs include Qwen2-VL (Wang et al., 2024b) and Qwen2.5-VL (Bai et al., 2025)

at both 7B and 72B scales, as well as InternVL2.5 (Chen et al., 2025) at 8B and 78B. We also include GPT-4o (Hurst et al., 2024) as a high-performing proprietary baseline. For reasoning-oriented models, we evaluate LLaVa-CoT-11B (Xu et al., 2025), Mulberry-7B (Yao et al., 2024), TVC-7B and 72B (Sun et al., 2025), and QVQ-72B-preview (Qwen Team, 2024).

Results. Quantitative results are presented in Table 1. Our approach yields substantial performance improvements over baseline models. Among 7B-scale models, v1 with full pointing capability outperforms both general-purpose and reasoning-specialized baselines. Notably, despite its smaller size, our 7B model narrows the performance gap with several 72B-scale models. The gains are particularly pronounced on MathVision, a benchmark known for its stronger demand for correct grounding.

##### 6.2 Further Analysis

Qualitative results. Figure 4 compares our method with LLaVA-CoT (Xu et al., 2025) on MathVision short-answer and multiple-choice examples. Our v1 exhibits explicit visual grounding via pointer-based detection and selective copying of relevant image regions. In the bar graph example, v1 correctly identifies the bar for Candy E and computes the correct percentage, while LLaVA-CoT misidentifies the tallest bar. In the hexagon pathfinding task, v1 reasons over spatial connectivity by attending to structural differences among options, whereas LLaVA-CoT fails to eliminate invalid candidates. These examples illustrate how active visual reference supports more precise and interpretable grounded reasoning than text-only chain-of-thought methods.

Ablation study. We report an ablation study in Table 2 to isolate the effect of the pointand-copy mechanism. All variants share the same Qwen2.5-VL-7B backbone: Backbone applies no task-specific fine-tuning, while Ours (no pointing) uses the same training setup as v1 but disables pointing at inference. On MathVision (testmini), v1 improves over the backbone from 23.6 to 34.5 (+10.9), whereas disabling pointing reduces the score to 25.3 (+1.7), indicating that point-and-copy at inference is a key driver of the gains.

Variant Tr. Inf. Score ∆ Backbone ✗ ✗ 23.6 – (no pointing) ✓ ✗ 25.3 +1.7 Ours ✓ ✓ 34.5 +10.9

Table 2: Ablation on MathVision mini. Train: grounded-reasoning training. Infer: point-and-copy at inference. ∆: gain relative to the backbone.

For context, TVC (Sun et al., 2025) serves as a text-only reasoning baseline; our training data extends its reasoning traces with additional visual-grounding annotations.

How does v1 utilize pointed visual regions? We analyze how v1 uses visual regions retrieved via the point-and-copy mechanism. As shown in Figure 5, we compare Input Attention (to original visual tokens) and Copy Attention (to copied tokens) during generation after the first copy operation.

We observe a structured sequence of behaviors. Before copying, attention to original image tokens increases, indicating a localization step in which relevant regions are identified. Immediately after copying, intermediate layers (e.g., layers 2 and 14) exhibit dominant copy attention, reflecting focused post-retrieval processing of the retrieved region. When averaged across layers, copied tokens consistently receive higher attention than original image tokens, suggesting that copied regions serve as stable and accessible references. In higher layers (e.g., layer 27), attention to input and copied tokens becomes more balanced, which may correspond to a late-stage integration of retrieved visual information into the broader reasoning context.

#### 7 Conclusion

We introduced v1, a lightweight extension that enables MLLMs to actively revisit input images via a point-and-copy mechanism, and v1g, a dataset of 300K multimodal reasoning

traces with fine-grained visual grounding. Experiments on established multimodal mathematical reasoning benchmarks show that v1 improves performance, particularly on tasks requiring grounded, multi-step visual reasoning. We hope this work encourages further exploration of dynamic visual access as a core component of multimodal reasoning.

#### References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Nasim Borazjanizadeh, Roei Herzig, Eduard Oks, Trevor Darrell, Rogerio Feris, and Leonid Karlinsky. Visualizing thought: Conceptual diagrams enable robust planning in lmms,

2025. URL https://arxiv.org/abs/2503.11790.

Juliette Brun, Pascal Le Masson, and Benoît Weil. Designing with sketches: the generative effects of knowledge preordering. Design Science, 2, 2016. URL https://api. semanticscholar.org/CorpusID:17302973.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14455–14465, June 2024.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025. URL https://arxiv.org/abs/2412.05271.

Junyi Chu, Emily R. Fyfe, and Bethany Rittle-Johnson. Diagrams benefit symbolic problemsolving. British Journal of Educational Psychology, 87:273–287, 2017. URL https://api. semanticscholar.org/CorpusID:14563301.

Jiwan Chung, Saejin Kim, Yongrae Jo, Jaewoo Park, Dongjun Min, and Youngjae Yu. Teaching metric distance to discrete autoregressive language models, 2025. URL https://arxiv.org/abs/2503.02379.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Richard J. Cox. Representation construction, externalised cognition and individual differences. Learning and Instruction, 9:343–363, 1999. URL https://api.semanticscholar.org/ CorpusID:143780266.

Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative selfimprovement, 2025. URL https://arxiv.org/abs/2503.17352.

Google. Gemini 2.0 flash (gemini-2.0-flash-001), February 2025. URL https://cloud.google. com/vertex-ai/generative-ai/docs/models/gemini/2-0-flash.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14953–14962, June 2023.

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=GNSMl1P5VR.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models, 2025. URL https://arxiv.org/abs/2503.06749.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Qing Jiang, Junan Huo, Xingyu Chen, Yuda Xiong, Zhaoyang Zeng, Yihao Chen, Tianhe Ren, Junzhi Yu, and Lei Zhang. Detect anything via next point prediction, 2025. URL https://arxiv.org/abs/2510.12798.

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in photographs of natural scenes. In Alessandro Moschitti, Bo Pang, and Walter Daelemans (eds.), Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 787–798, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1086. URL https: //aclanthology.org/D14-1086.

Maria Kozhevnikov, Mary Hegarty, and Richard E. Mayer and. Revising the visualizerverbalizer dimension: Evidence for two types of visualizers. Cognition and Instruction, 20(1):47–77, 2002. doi: 10.1207/S1532690XCI2001\_3. URL https://doi.org/10.1207/ S1532690XCI2001_3.

Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vulic´, and Furu Wei. Imagine while reasoning in space: Multimodal visualization-of-thought,

2025. URL https://arxiv.org/abs/2501.07542.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=w0H2xGHlkw.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pp. 38–55. Springer, 2024.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= KUNzEQMWU7.

Tianren Ma, Lingxi Xie, Yunjie Tian, Boyu Yang, and Qixiang Ye. Clawmachine: Learning to fetch visual tokens for referential comprehension. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=TOtk9dTYGG.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning, 2025. URL https://arxiv.org/abs/ 2503.07365.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling, 2025. URL https://arxiv.org/abs/2501.19393.

Qwen Team. Qvq: To see the world with wisdom, December 2024. URL https://qwenlm. github.io/blog/qvq-72b-preview/.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020.

Abigail See, Peter J Liu, and Christopher D Manning. Get to the point: Summarization with pointer-generator networks. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1073–1083, 2017.

Andreas Steiner, André Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, et al. Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555, 2024.

Hai-Long Sun, Zhun Sun, Houwen Peng, and Han-Jia Ye. Mitigating visual forgetting via take-along visual conditioning for multi-modal long cot reasoning, 2025. URL https: //arxiv.org/abs/2503.13360.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024a. URL https://openreview.net/forum?id=QWTCcxMpPA.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution, 2024b. URL https:

//arxiv.org/abs/2409.12191.

Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-efficient visual reasoning self-improvement, 2025. URL https://arxiv.org/abs/ 2504.07934.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45, Online, October 2020. Association for Computational Linguistics. URL https://www.aclweb.org/ anthology/2020.emnlp-demos.6.

Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms, 2023. URL https://arxiv.org/abs/2312.14135.

Qiong Wu, Xiangcong Yang, Yiyi Zhou, Chenxin Fang, Baiyang Song, Xiaoshuai Sun, and Rongrong Ji. Grounded chain-of-thought for multimodal large language models, 2025. URL https://arxiv.org/abs/2503.12799.

Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4818–4829, 2024.

Guowei Xu, Peng Jin, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step, 2025. URL https://arxiv.org/abs/2411.10440.

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023.

Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, and Dacheng Tao. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search, 2024. URL https://arxiv.org/abs/2412.18319.

Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, and Filip Ilievski. MLLMs know where to look: Training-free perception of small visual details with multimodal LLMs. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=DgaY5mDdmT.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pp. 169–186. Springer, 2024a.

Zhuosheng Zhang, Aston Zhang, Mu Li, hai zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. Transactions on Machine Learning Research, 2024b. ISSN 2835-8856. URL https://openreview.net/forum?id=y1pPWFVfvR.

#### Overview of the Appendix

This Appendix is structured as follows:

- • Section A describes implementation details and resources used in the project;
- • Section B discusses limitations and future directions;
- • Section C provides details of the data generation process;
- • Section D reports human evaluation results validating the grounding quality of our training dataset (v1g) and our model (v1);
- • Section E details pseudo-code on the visual grounding pipeline we utilized in the data generation process;
- • Section F presents the use of Large Language Models (LLMs);
- • Section G presents additional qualitative results.

#### A Implementation Details & Resources

Training Details All models are trained under uniform settings: a base learning rate of 3 × 10−5, per-device batch size of 2, and gradient accumulation over 4 steps. We leverage DeepSpeed for distributed training across 8 NVIDIA A100 GPUs. Optimization uses AdamW with β1 = 0.9, β2 = 0.95, and training is performed for 5 epochs.

Training Duration Our training schedule of 5 epochs follows the setup used for the original text-only reasoning trace dataset, which we extend to the grounded reasoning setup (Sun et al., 2025). Because the reasoning traces contain substantially longer token sequences than typical MLLM data, shorter training runs produced unstable behaviors such as repetition and incomplete reasoning without a final answer. In contrast, the point-andcopy behavior required relatively little data and typically saturated within the first epoch, as indicated by an early plateau in copy-token accuracy. The longer schedule therefore reflects the requirements of the inherited reasoning-trace setup rather than the needs of the pointer mechanism itself.

#### B Limitations and Future Work

This work focuses on demonstrating the effectiveness of active visual reference in structured multimodal reasoning via a simple point-and-copy mechanism. While v1 shows strong performance in mathematical domains, several directions remain for broader applicability.

Beyond mathematical domains. Extending v1 to other settings (e.g., scientific diagrams, medical images, or visual commonsense) presents new challenges in representation and supervision. These domains often lack structured reasoning traces, making data collection more difficult. Since v1g relies on a pretrained text-only MLLM to seed reasoning, generalizing to less structured domains will require advances in decomposition, grounding, and alignment.

Weak supervision and reinforcement learning. Recent work in inference-time scaling and alignment has shown the promise of reward-based learning for reasoning. Incorporating such methods into v1 may enable more flexible and efficient visual retrieval strategies without dense supervision. We leave this exploration to future work due to current resource constraints.

#### C Data Generation Details

- Figure 6 illustrates the construction pipeline for our v1g dataset; each stage of this pipeline is described in detail in Section 4.3. The specific prompt template used to decompose text-

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

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

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Figure 6: v1g dataset construction pipeline.

based reasoning paths into visual queries (as outlined in our methodology in Section 4.3) is provided in Table 5.

#### D Human Evaluation of Grounding Quality

##### D.1 Evaluation of v1g Dataset Quality

To validate the quality of our automatically generated visual grounding annotations in the v1g dataset, we conducted a human evaluation comparing our attention-based grounding approach against GroundingDINO (Liu et al., 2024), a widely used open-set object detector.

Methodology. We randomly sampled 100 examples from the v1g dataset, each containing multiple bounding boxes. Three expert annotators independently evaluated each bounding box using binary classification on three criteria:

- • Correctness: Whether the bounding box covers the intended object or region
- • Comprehensiveness: Whether all relevant visual content is included within the box
- • Tightness: Whether the box is well-fitted with minimal extraneous background

We report the average score across annotators, majority vote, and Fleiss’ κ to assess interannotator agreement. Agreement quality follows standard interpretations: Fair (0.21–0.40), Moderate (0.41–0.60), Substantial (0.61–0.80), and Almost Perfect (0.81–1.00).

Results. As shown in Table 3, our attention-based grounding method substantially outperforms GroundingDINO on correctness (83.3% vs. 29.3% average score), demonstrating superior capability in localizing semantically complex and context-dependent entities such as geometric elements (e.g., “angle ABC”), chart components (e.g., “bar for Grace”), and referring expressions (e.g., “the second figure”). While GroundingDINO achieves higher inter-annotator agreement, this primarily reflects consistent failure modes rather than quality, as evidenced by its low absolute performance.

Table 3: Human evaluation of v1g dataset quality.

Method Metric Avg Majority Fleiss’ κ Agreement

Correctness 83.3% 87.0% 0.352 Fair Comprehensive. 55.0% 56.0% 0.582 Moderate Tightness 46.0% 44.0% 0.436 Moderate

Attention-based (Ours)

Correctness 29.3% 30.0% 0.711 Substantial Comprehensive. 47.3% 49.0% 0.906 Almost Perfect Tightness 19.0% 18.0% 0.675 Substantial

GroundingDINO

Table 4: Human evaluation of v1 model pointing quality.

Metric Avg Majority Fleiss’ κ Agreement

Correctness 82.7% 87.0% 0.558 Moderate Comprehensiveness 55.7% 54.0% 0.689 Substantial Tightness 49.3% 40.0% 0.280 Fair Appropriateness 87.7% 90.0% 0.599 Moderate

##### D.2 Evaluation of v1 Pointing Accuracy

We additionally evaluated the pointing accuracy of our trained v1 model to assess how effectively it grounds visual references during inference.

Methodology. Using the same evaluation protocol, we sampled 100 outputs from v1 on the MathVision dataset. Annotators evaluated whether the model’s pointed regions (copied image tokens) correctly corresponded to the referenced objects in the reasoning trace. We added an Appropriateness criterion to assess whether the pointing action was contextually justified.

Results. Table 4 demonstrates that v1 maintains high grounding quality during inference, achieving 82.7% correctness; comparable to the training data quality. The high appropriateness score (87.7%) indicates that the model learns to selectively invoke the pointing mechanism when dynamic visual reference is genuinely beneficial for reasoning.

Discussion. The evaluation reveals that our attention-based grounding excels at capturing semantically rich visual references that are challenging for traditional object detectors. The moderate tightness scores across both methods reflect the inherent ambiguity in defining precise boundaries for abstract concepts (e.g., "angle 2" or "the second figure"), where multiple valid interpretations exist. The consistency between training data quality and model performance suggests that v1 successfully learns robust visual grounding capabilities from our automatically generated supervision.

#### E Bounding-Box Extraction from Cross-Attention

This section provides a high-level pseudocode description of our data annotation method for deriving bounding boxes from cross-attention in Qwen2.5-VL.

#### F Use of LLMs

This work uses LLMs for data generation (Gemini-2.0-Flash for reasoning trace decomposition) and evaluation (Gemini-2.0-Flash for answer matching), as described in Sections 4.3 and 6.1.

Algorithm 1 Bounding-Box Extraction from Cross-Attention (High-Level)

Input: Image I, region description T Output: Bounding box b corresponding to T

- 1. Prepare multimodal input. Concatenate I with a static visual-grounding instruction prompt and feed it to Qwen2.5VL.
- 2. Extract attention with instruction. From the final decoding position, obtain the cross-attention map A over image tokens. Use a predefined set of layers (selected empirically) and average across heads.
- 3. Extract baseline attention. Remove the object name from the prompt, feed the modified prompt with I to the model, and extract the corresponding attention map A′ using the same layers and averaging.
- 4. Compute attention contrast. Compute the contrastive relevance for each image token: R = A/A′.
- 5. Derive bounding region. Identify the peak region in R. Sweep over multiple candidate crop ratios; for each ratio, form a bounding region around the peak. Select the bounding box maximizing contrast sharpness between inside and outside regions. Convert the selected region to image-coordinate bounding box b.
- 6. Return. return b

#### G Additional Qualitative Results

To further illustrate v1’s complex visual reasoning, this section provides additional qualitative examples, complementing Figure 4 from the main text. These examples highlight how v1 leverages the point-and-copy mechanism.

- Figure 7 demonstrates v1 on a synthetic task (CLEVR-like) requiring object counting based on the query: “Subtract all red things, then subtract all tiny matte balls. How many objects are left?”. v1 first localizes objects using its pointer mechanism. It then sequentially reasons, identifying “red” objects before revisiting relevant items, like the “cyan sphere,” to verify the combined “tiny” and “matte” attributes through targeted attention. This process demonstrates v1’s capacity for precise attribute grounding and multi-step compositional reasoning enabled by the point-and-copy mechanism.

In Figure 8, v1 tackles a chart comprehension task: determining if the “Dark Violet” data series has the minimum area under the curve. v1 initially grounds key chart elements, using its pointer to isolate data series such as “Dark Violet,” “Medium Mint,” and “Dark Cyan.” Later in its reasoning, it proactively revisits these series, performing a comparative analysis of their visual trajectories and relative y-axis values to infer their respective areas. Such selective re-focusing showcases its ability to perform nuanced comparisons within dense visual information.

These examples further affirm that v1’s architecture, by supporting active visual reference and precise grounding via its pointing mechanism, achieves robust, interpretable, and accurate multi-step visual reasoning.

[Figure 128]

[Figure 129]

[Figure 130]

###### Figure 7: Qualitative example of v1 tackling an attribute-based counting task in a synthetic domain.

[Figure 131]

[Figure 132]

[Figure 133]

###### Figure 8: Qualitative example of v1 performing comparative reasoning on a chart comprehension task.

Prompt for data generation

You are given text-only reasoning for visual question answering. Your task is to convert this text-only reasoning into visually grounded reasoning. ### STEP-BY-STEP INSTRUCTION Please follow these instructions step-by-step, imitating human visual reasoning behavior by:

- 1) Start from the beginning of the reasoning and read EACH sentence.
- 2) When you think you’d better look at the object or region, use the detect() function.
- 3) Format: ‘detect(query="visual item that you want to find", objects=["<obj#>"])‘
- 4) After detection, reference the visual element with ’<obj#>’ tags every time you need to look at it again immediately after mentioning the item.
- 5) Use NEW object numbers (‘<obj1>‘, ‘<obj2>‘, ‘<obj3>‘...) for EACH new detection.

### EXAMPLE: Original text: "Looking at the graph, I can see the function reaches its maximum at x = 3."

Corrected: “‘ To answer the question, I need to look the graph. detect(query="function graph", objects=["<obj1>"]) Looking at the graph <obj1>, I can see the function reaches its maximum at x = 3. “‘ Later reference: You can skip the <obj#> tag when you think you do not need to look it again. “‘ The slope of the function becomes zero at this point on the graph. “‘

### KEY REQUIREMENTS:

- - Every item in lists MUST have its own ‘detect()‘ statement
- - Put ‘detect()‘ statements on their own lines
- - NEVER skip any visual element mentioned in the reasoning
- - Start object numbering at ‘obj1‘ and increase by 1 for each new object ### <OBJ#> REQUIREMENTS
- - Visual element should be concrete, distinct, and explicit. Later you will localize the element based on the detect(). So make sure that the element not confusing.
- - Use separate tags for each object (write "between the bus <obj1> and the car <obj2>" not "between <obj1 and obj2>").
- - GOOD grounding: "I need to analyze this problem. detect(query="triangle", objects=["<obj1>"]) The triangle <obj1> has a right angle at vertex S."
- - BAD grounding: "detect(query="triangle and rectangular", objects=["<obj1>"]) in the diagram, there are the triangle and rectangular has a right angle." (referring to non-atomic element)
- - BAD grounding: "detect(query="region", objects=["<obj1>"]) The triangle <obj1> has a right angle." (referring to ambiguous element)

After completing the reasoning, list all objects detected: {

- "obj1": {"type": "function_graph", "description": "Graph of a function with maximum at x = 3"},
- "obj2": {"type": "next_item", "description": "Description of next item"} }

- We will localize the element with the open-world detector based on the description, so make sure to include well-described full self-contained description enough to uniquely identify the object.

### FINAL FORMAT: { "reasoning": "Your fully visually-grounded reasoning text", "obj_list": "Your JSON object list" }

Now, strictly following the instruction and the example, please provide the object list and visually grounded reasoning for the following prompt and reasoning:

### Example Original Conversation

HUMAN: [few_shot_question]

GPT: [few_shot_answer]

### Visually Grounded Reasoning GPT: [few_shot_reasoning] ### Object List: [few_shot_objects]

Now, given the conversation, please convert GPT’s text-only reasoning into visually grounded reasoning

Original Conversation:

HUMAN: [question]

GPT: [answer]

### Visually Grounded Reasoning:

Table 5: Prompt used for converting textual reasoning to grounded reasoning annotations in v1g data generation process.

