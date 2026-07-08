# arXiv:2605.15572v2[cs.CL]2Jun2026

## Measuring Maximum Activations in Open Large Language Models

Luxuan Chen1,2∗, Han Tian2,4∗, Xinran Chen2, Rui Kong2, Fang Wang2, Jiamin Chen2, Yuchen Li2†, Jiashu Zhao2, Shuaiqiang Wang2, Haoyi Xiong3, Linghe Kong1, Dawei Yin2† 1Shanghai Jiao Tong University

2Baidu Inc. 3Independent Researcher 4Nankai University yuchenli1230@gmail.com,yindawei@acm.org

### Abstract

The dynamic range of activations is a first-order constraint for low-bit quantization, activation scaling, and stable LLM inference. Prior work characterized outlier features and massive activations on pre-2024 LLaMA-style models, and the downstream activation-quantization stack inherits that picture without revisiting it for the post-LLaMA open-model boom. We ask the deployment-oriented question: how large can activations get in modern open LLMs, and how does this magnitude vary across families, generations, and training stages? Under a unified pipeline (5,000-sample multi-domain corpus, family-specific tokenization, identical hooks across embeddings, hidden states, attention, MLP/MoE, SwiGLU gates, and final norm), we measure global and layerwise maxima on 27 checkpoints from 8 open families spanning dense, MoE, vision-language, intermediate-training, and instruction-tuned variants. We find that (i) global maxima span over nearly four orders of magnitude at comparable parameter counts, with Qwen3.5 and MoE checkpoints in the 102–103 range and Gemma3-27B-it reaching ∼ 7×105; (ii) cross-family and cross-generation comparisons break simple monotonic scaling; and (iii) MoE checkpoints exhibit 14.0–23.4× lower peaks than matched-scale dense counterparts, while the residual stream carries the global maximum in 22/24 checkpoints. A lightweight INT-8 sanity check shows that measured maxima covary with low-bit reconstruction error via activation-scale selection. We conclude that maximum activation magnitude is a model property tied to family, architecture, and training stage—not a simple byproduct of size—and should be measured and reported alongside any open-weight release before low-bit deployment. The code is publicly available at https://github.com/clx1415926/Max_act_llm.

### 1 Introduction

The activation dynamic range of a large language model (LLM) is not merely a descriptive statistic: it determines the numerical range that inference systems, activation quantizers, and scaling rules must accommodate [10]. In low-bit inference, for example, a per-tensor activation scale is often chosen to cover the largest magnitude observed on a calibration set. A small number of extremely large activations can therefore dominate the scale, waste most quantization levels on rarely used values, and amplify reconstruction error for ordinary activations. This paper studies a simple but deployment-critical quantity: the maximum activation magnitude, defined as the largest absolute activation observed across layers and key components under a fixed evaluation protocol.

Extreme activations, massive activations, and outlier features have been studied from several perspectives, including existence tests, token- or feature-level localization, and functional interventions.

∗Co-first authors with equal contributions. †Corresponding author

Preprint.

However, the deployment question remains less systematically mapped: how large can activations become in recent open LLMs, where do the largest values appear, and how do they change with model family, architecture, model generation, and training stage? This question is increasingly important because modern open models no longer differ only in parameter count. They vary in normalization and training recipes, dense versus MoE computation, vision-language adaptation, instruction tuning, and released intermediate training stages. As a result, parameter scale alone may be an unreliable proxy for activation range.

Prior work on extreme activations falls along two largely separate lineages. The interpretability line begins with [11], who defined emergent outlier features via a 6σ rule on OPT/BLOOM, and [29], who introduced massive activations as coordinates that are simultaneously large (|xi| > 100) and locally sparse (≥ 1000× the per-token median); [5] attributed them to attention heads needing a “no-op” route through the residual stream, and very recent work refines this picture—[15] trace when attention sinks emerge during pretraining, and [30] decouple massive-activation “spikes” from sinks and localize them to early-layer step-up blocks under pre-norm transformers. All of these studies treat the phenomenon categorically and analyze a handful of LLaMA-family or singlearchitecture checkpoints. The quantization line treats the same activations as a deployment obstacle: SmoothQuant [33], AWQ [18], GPTQ [12], and Outlier Suppression+ [32] migrate or rescale outlier mass; rotation methods QuaRot [3], SpinQuant [21], and DuQuant [17] remove the outlier basis; FlatQuant [31] learns affine flattening transforms; PrefixQuant [6] and KIVI [22] target the KV cache; and FP8 pretraining pipelines [10, 9] fold analogous mitigations into low-precision training, with [20] arguing that high-sparsity MoE routing further changes the activation regime. All of these mitigations transform away the upper bound rather than measuring how it varies across modern releases.

It remains unclear whether either discovery still holds for the recent wave of post-LLaMA open releases—Qwen2.5/3/3.5 [26, 27], Qwen2.5-VL [4], Gemma 2 and Gemma 3 [13, 14], the Ling-mini series [19], and gpt-oss [24]—which diverge from earlier LLaMA-style models along multiple axes simultaneously: normalization stack, gated MLP variants, MoE routing, multimodal adaptation, intermediate-training releases, and instruction tuning. To our knowledge no prior study reports activation magnitudes across these families under a unified protocol. Our paper is complementary along both axes: we provide the first unified-protocol measurement of the global maximum M = max|a| across 27 post-LLaMA open checkpoints from 8 families, treat M as a continuous releasable model property rather than a binary outlier flag, and connect M directly to per-tensor INT-8 reconstruction error—inputs that the mechanistic and quantization-mitigation lines currently lack.

We address this gap with a unified empirical survey of maximum activations in modern open LLMs. Our main analysis covers 24 checkpoints from 8 model families: Qwen2.5, Qwen2.5-VL, Qwen3, Qwen3.5, Gemma2, Gemma3, Ling, and GPT-OSS. We additionally analyze 3 Qwen2.5-Instruct checkpoints to isolate the effect of supervised fine-tuning. All checkpoints are evaluated on the same 5,000-sample multi-domain corpus, with the text corpus re-tokenized for each model family. During forward inference, we use PyTorch hooks to stream activation statistics from embeddings, layerwise hidden states, attention outputs, MLP or MoE outputs, SwiGLU gate pre-activations, and final normalization outputs. This protocol lets us compare global maxima, layerwise peak trajectories, carrier components, family and generation effects, and matched architectural or training contrasts under the same measurement pipeline.

Contributions. Our work makes the following contributions:

- • Largest-to-date cross-family activation survey. We measure global and layerwise maximum activations on 24 checkpoints from 8 modern open families (Qwen2.5/2.5-VL/3/3.5, Gemma2/3, Ling, GPT-OSS)—spanning dense, MoE, vision-language, intermediatetraining, and instruction-tuned variants—under a single unified pipeline (5,000-sample multi-domain corpus, family-specific tokenization, identical hooks across embeddings, hidden states, attention/MLP/MoE outputs, SwiGLU gates, and final norm), moving beyond the LLaMA-derivative monoculture of [11, 29].
- • Continuous magnitude reformulation of “massive activation.” We replace the binary same-token criterion of [29] with the deployment-relevant statistic M = max|a|, and show the two views can disagree—some checkpoints failing the binary criterion are the easiest to quantize, while some passing it are the hardest.
- • Five matched-design comparisons. We isolate (i) within-family scaling, (ii) same-scale MoE-vs-dense, (iii) same-family vision-language-vs-text-only, (iv) same-backbone Base-

- vs-Instruct, and (v) same-family training-stage effects on the same measurement substrate, providing the first observational decomposition of scale, family, generation, architecture, modality, and training progress for activation peaks.
- • Empirical findings with deployment implications. (a) Global maxima vary by orders of magnitude across families at comparable parameter counts and break simple monotonic scaling; (b) the residual stream carries the global maximum in 22/24 main checkpoints; (c) MoE reduces peak magnitudes by 14.0–23.4× relative to nearby dense counterparts; (d) SFT mainly compresses late-layer peaks; (e) training progress can monotonically increase the global maximum at fixed family and architecture; and (f) a lightweight per-tensor INT-8 probe shows higher M correlates with substantially lower SQNR.
- • Open pipeline and per-checkpoint statistics. We release the hook-based measurement code and per-checkpoint activation statistics to support reproducibility and future quantization, scaling, and architecture research. We do not claim a causal mechanism for the observed differences (see Section D).

### 2 Measurement Protocol

Stage 3 ANALYSIS & REPORTING

Stage 2 ACTIVATION MEASUREMENT

Stage 1 DATA PREPARATION

✤

Shared Evaluation Corpus

Model Suite

{ }

↻

Per-Model Statistics

PyTorch Forward Hooks

27 checkpoints from 8 open families

5,000 multi-domain text samples

JSON

Saved as JSON files

 ️

Embedding Outputs

- Qwen2.5 Qwen2.5-VL Qwen3

- Qwen3.5 Gemma2 Gemma3

Post-Residual Hidden States

Math / Science (850) Code (850) English Web (850) Knowledge / QA / Books (850) Chinese (400) Other Low-Resource Languages (300) Extra English / Mixed Web (900)

Attention Outputs

###### …

###### Downstream Analyses

Ling GPT-OSS

MLP / MoE Outputs

◎ Global maxima

+ Qwen2.5-Instruct for SFT comparison

SwiGLU Gate Pre-Activations

▧ Layerwise heatmaps / trajectories ⌁ Familyscale comparisons/ generation / ⚖ MoESFT contrastsvs. dense and

Final LayerNorm Outputs

###### Forward-Only Inference

Length buckets: 256 / 512 / 1024 / 2048 / 4096 tokens

Frozen weights; No parameter updates

###### Recorded Statistics

☑ INT-8 sanity check

Family-Specific Tokenization

⚙️▢▢▢

- • Mean
- • Std
- • RMS
- • Mean |a|
- • Max / Min
- • Absolute-value Quantiles

Same text, re-tokenized for each model family

▁▃▆

Streaming GPU Accumulation

▣

###### Tables

###### Figures

Online collection without per-sample synchronization

Token Sequences

| | | |
|---|---|---|
| | | |
| | | |
| | | |

⌁ ▃▅▇

- Family A
- Family B

101 2087 374 … 5021 102

151 992 63 … 8421 102 ... ... ... ... ... ...

★ Primarymetric:M=max|a|

Log-histogram-based quantile estimation

All paper figures and tables are generated from the saved statistics.

...

Figure 1: Overview of the pipeline.

The overall pipeline is shown in Figure 1, which consists of three steps: Data Preparation, Activation Measurement, and Analysis. We use a unified offline evaluation protocol. We first construct a multi-domain text corpus, tokenize the same text with each model family’s tokenizer, run forward inference on each checkpoint, and record layerwise activation statistics. All figures and tables are generated from the resulting per-model statistics.

#### 2.1 Corpus construction

The target evaluation corpus contains 5,000 samples. The data are sampled from RedPajama [28] sources and bucketed by content type. The target category counts are 850 mathematical or scientific samples, 850 code samples, 850 English web samples, 850 knowledge-oriented samples such as encyclopedic, book, or Q&A text, 400 Chinese samples, 300 samples in other low-resource languages, and 900 additional English or mixed web samples. This design reduces the risk that maximumactivation statistics are dominated by a single domain and ensures that the corpus covers formal text, natural web text, knowledge-intensive text, code, and multilingual content.

The corpus also controls sequence-length diversity. Samples are randomly truncated to 256, 512, 1024, 2048, or 4096 tokens with target proportions of 1%, 1%, 2%, 3%, and 93%, respectively. The corpus is therefore dominated by long-context inputs while retaining a small number of short and

medium-length sequences. The resulting corpus has an average length of approximately 3899 tokens, corresponding to roughly 19.5M tokens in total.

To avoid tokenizer mismatch artifacts, the text corpus is held fixed while tokenization is performed separately for each model family. Thus, models receive semantically identical text but token sequences aligned with their own tokenizer, reducing activation-statistics bias caused by tokenizer incompatibility.

#### 2.2 Model suite and instrumentation

We select models according to three principles. First, we cover recent mainstream open LLM families rather than restricting the study to earlier LLaMA-style models. Second, we include multiple parameter scales and architectural forms, allowing us to separate the effects of scale, family, and architecture. Third, we include special variants such as MoE models, vision-language models, intermediate training checkpoints, and instruction-tuned models, so that we can examine whether maximum activations change with routing, modality adaptation, training progress, or supervised fine-tuning (SFT).

The main experiment contains 24 checkpoints from 8 families: Qwen2.5, Qwen2.5-VL, Qwen3, Qwen3.5, Gemma2, Gemma3, Ling, and GPT-OSS. Except for the publicly released Gemma3 checkpoints, which are instruction-tuned models, we treat the main-analysis checkpoints as base or intermediate-training checkpoints, as Shown in Table 2. Therefore, the Gemma2/Gemma3 comparison should be interpreted as a public-checkpoint family-level contrast rather than a strict base-to-base generational ablation.

#### 2.3 Recorded statistics and peak stability

The statistics pipeline has three stages. First, the shared text corpus is converted into token sequences with each model family’s tokenizer. Second, each checkpoint is loaded with its full weights and evaluated with forward inference only; no parameters are modified. During inference, PyTorch forward hooks collect six classes of activation tensors: embedding outputs, layerwise hidden states after residual updates, layerwise attention outputs, layerwise MLP outputs or MoE block outputs, MLP gate pre-activations in SwiGLU-style architectures, and final LayerNorm outputs. Third, permodel JSON statistics are used to generate all figures and tables. For each captured component, we record the mean, standard deviation, RMS, mean absolute value, maximum value, minimum value, and streaming estimates of absolute-value quantiles.

Because the global maximum activation is an extreme statistic, we first verify that it is not triggered by a small number of accidental samples. For four representative models, we construct categoryproportional subsamples of 1,000 and 2,000 examples from the original 5,000-example corpus. Each subsample size is repeated 5 times, and each repeat runs the full activation scan. The resulting peaks consistently reproduce the order of magnitude of the 5k reference run. The largest coefficient of variation across 1k repeats is 10.1% for Qwen3-30B-A3B, and the largest coefficient of variation across 2k repeats is 8.2%. These results indicate that the reported maximum activations are not accidental single-sample artifacts and that the measurements are statistically robust at the scale studied here.

### 3 From Binary Massive Activations to Continuous Peaks

The empirical story proceeds from definition to mechanism to comparison. We first connect our deployment-oriented maximum M to the binary massive-activation criterion used in prior work, then ask where the largest values are carried, and finally compare families, generations, architectures, and training stages under matched designs.

#### 3.1 Relationship to the Sun criterion

Our main metric is the global maximum activation, M = max|a|, taken across all six hooked component classes (embeddings, layerwise hidden states, attention outputs, MLP/MoE outputs, SwiGLU gate pre-activations, final LayerNorm) and all layers; this is the value plotted in every bar chart and used in every matched-pair ratio in Sections 5 and C. The Top-k values reported in

Table 1: Representative activation locations for the 24 main-analysis checkpoints. We report the five largest absolute activations, Top-10, Top-100, Top 1%, Top 10%, and median values. The representative location is selected from the full layerwise scan: for passing models, we use a hidden layer containing a same-token sparse extreme value; for failing models, we use the full-model peak location. The median is the median absolute value across the d dimensions of the single token activation vector carrying the reported peak. Models marked with × fail the Sun criterion due to insufficient local ratio or insufficient absolute magnitude.

Model Top 1 Top 2 Top 3 Top 4 Top 5 Top-10 Top-100 Top 1% Top 10% median Qwen2.5-1.5B× 7,968 7,968 7,968 7,968 7,968 7,968 7,776 3.72 1.37 13.9

- Qwen2.5-7B 13,248 13,248 13,248 13,248 13,248 13,184 12,224 3.94 1.37 5.00

- Qwen2.5-32B 22,144 21,888 21,760 21,760 21,760 21,632 21,120 12.9 3.94 8.62

- Qwen2.5-VL-3B 3,248 3,248 3,248 3,248 3,088 3,056 2,944 6.05 1.97 1.84

- Qwen2.5-VL-7B 8,256 8,256 8,256 8,256 8,256 8,256 7,552 5.57 1.91 3.23

- Qwen2.5-VL-32B 22,144 22,144 22,144 22,144 22,144 22,144 21,248 14.6 4.37 9.31

Qwen3-1.7B 14,208 14,144 14,144 14,144 14,144 14,080 13,952 103.9 52.5 4.19 Qwen3-8B 17,664 17,664 17,664 17,664 17,664 17,664 16,896 4.05 1.32 3.14 Qwen3-30B-A3B 1,512 1,512 1,512 1,512 1,512 1,512 1,392 0.348 0.099 0.578 Qwen3-32B 35,328 35,328 35,328 35,328 35,328 35,328 34,304 11.7 3.61 12.4

- Qwen3.5-0.8B× 122.0 122.0 114.5 114.5 114.0 22.5 3.23 12.2 6.28 2.20

- Qwen3.5-9B× 956.0 920.0 920.0 912.0 908.0 848.0 87.5 12.1 5.48 3.42

- Qwen3.5-27B 167.0 167.0 167.0 167.0 160.0 152.0 127.0 0.965 0.445 0.225

- Qwen3.5-35B-A3B× 132.0 131.0 129.0 126.5 125.0 121.0 40.8 1.37 0.595 0.719 gemma-2-2b 2,992 2,992 2,976 2,944 2,944 2,848 884.0 6.71 2.34 50.5

- gemma-2-9b 1,656 1,560 1,160 1,112 1,040 808.0 294.0 16.4 6.11 1.50

- gemma-2-27b 157,696 156,672 155,648 153,600 151,552 149,504 83,456 564.7 298.5 122.0

- gemma-3-4b-it 245,760 243,712 242,688 241,664 241,664 237,568 165,888 828.7 461.2 86.5

- gemma-3-27b-it 696,320 696,320 696,320 696,320 696,320 696,320 667,648 602.4 316.6 31.2

Ling-mini-5T 7,648 7,648 7,648 7,648 7,648 7,648 1,976 16.8 7.03 3.59 Ling-mini-10T 9,024 9,024 9,024 9,024 9,024 9,024 3,296 9.59 3.14 3.39 Ling-mini-15T 9,600 9,600 9,600 9,600 9,600 9,600 8,192 10.4 3.31 4.06 Ling-mini-20T 10,240 10,048 9,984 9,920 9,856 9,664 8,512 3.52 1.05 1.32

gpt-oss-20b 43,008 42,496 42,496 42,496 42,496 42,496 42,240 48.3 12.0 5.94

Table 1 are drawn from a single representative layer chosen for the local-sparsity diagnostic and may therefore differ from M when a different layer carries the global peak. Before turning to this macro-level magnitude analysis, we first drill down into whether the global extrema also satisfy a commonly used local sparsity definition from prior work. We adopt the same-token criterion of [29]: given a hidden state vector x ∈ Rd for one token, a coordinate xi is counted as a massive activation if it simultaneously satisfies |xi| > 100 and |x

i|

mediandj=1 |xj| > 1000. At the model level, a checkpoint passes the criterion if any hidden layer contains at least one token-feature coordinate satisfying both thresholds.

3.2 Overall existence and failure mechanisms

- Table 1 summarizes representative activation locations for each checkpoint based on the full layerwise scan. Overall, 20 of the 24 main-analysis checkpoints pass the Sun criterion, indicating that massive activations remain widespread in recent open LLMs.

The four failing checkpoints reveal two distinct failure mechanisms. Qwen2.5-1.5B reaches an absolute peak of 7,968, but the median absolute value within the peak token is 13.9, giving a local ratio of roughly 574 and falling below the 1000× threshold. This model therefore exhibits large but relatively dense activations rather than locally sparse massive activations. In contrast, Qwen3.5-0.8B,

- Qwen3.5-9B, and Qwen3.5-35B-A3B fail because their overall activation scale is systematically suppressed. Figure 2 shows that all failing points avoid the upper-right passing region. This diagnosis confirms that local ratio alone does not fully characterize activation-range risk, motivating our use of the global absolute maximum as the primary metric for quantization and deployment analysis.

We retain the absolute peak, the local-ratio scatter (Figure 2), and the dual reporting in Table 1 as the diagnostic for these failures rather than introducing a normalization-stack-specific re-analysis. The diagnostic is consistent with the architectural account of [30], in which residual-stream spikes are generated by a small number of early-layer step-up blocks and shaped by the pre-norm normalization stack; for the deployment-oriented question of this paper—“how large can the activation magnitude become”—the absolute peak M is the directly relevant quantity. We therefore treat the binary criterion mainly as a descriptive bridge to prior work and report M as the primary metric throughout the paper.

[Figure 1]

- Figure 2: Failure modes for the four checkpoints that do not satisfy the Sun criterion. Colors indicate models; circles denote each layer’s peak token, and triangles denote each layer’s highest local-ratio token. Dashed lines mark the absolute-magnitude and local-ratio thresholds; the upper-right region is the passing region. No point overlaps this region.

4 Where Maximum Activations Form

4.1 Layerwise intensity distribution

- Figure 3 shows the normalized-depth heatmap of hidden-state peak magnitudes for all main-analysis checkpoints. Peak depth has no universal location across architectures: even within the same family, maxima can occur in shallow, middle, or deep layers. Therefore, reporting only the peak layer index is less informative than characterizing the full layerwise trajectory, namely how peak magnitudes accumulate, jump, plateau, or decay with network depth.

#### 4.2 Two layerwise patterns

The layerwise trajectories broadly fall into two patterns, illustrated in Figure 4. The first is a jumpand-plateau pattern: activation magnitude rises sharply in early or middle layers and then remains high over a long layer interval, as in Qwen2.5 and GPT-OSS. The second is a gradual-accumulation pattern: activation magnitude increases more smoothly with depth and often reaches its maximum in later layers, as in Qwen3.5 and Gemma. This distinction indicates that maximum activations are governed not only by the physical depth of the peak layer, but also by the dynamics through which the peak forms. The pattern is strongly associated with model family and architecture rather than being a monotonic function of parameter scale.

We treat this dichotomy as a qualitative description of the depth-normalized layerwise heatmap in Figure 3 and the representative trajectories in Figure 4, not as a quantitative classifier. The

[Figure 2]

##### Figure 3: Layerwise heatmap of hidden-state peak magnitudes. The horizontal axis is normalized depth, and color indicates layerwise absolute peak magnitude on a log scale. White hollow markers indicate the peak depth bin for each checkpoint.

two patterns are consistent with the architectural account of [30], in which spike formation is concentrated in a small number of early-layer step-up blocks under pre-norm transformers, and the cross-family heatmap already shows the relevant separation; introducing a numerical classifier on n = 24 trajectories would not change the matched-design contrasts in Sections 5–C, which use M rather than the trajectory shape.

[Figure 3]

##### Figure 4: Representative layerwise trajectories for the two main emergence patterns. Left: jumpand-plateau models rise sharply in early or middle layers and remain high afterward. Right: gradualaccumulation models grow more smoothly with depth and often peak in later layers. Each curve corresponds to one representative checkpoint; markers indicate the peak layer.

#### 4.3 Carrier components

Across the 24 main-analysis checkpoints, 22 global maxima occur in layerwise hidden states. GPTOSS-20B is a component-level exception whose global maximum comes from the MLP output, and the failing Qwen3.5-0.8B checkpoint peaks at the final LayerNorm output. If we restrict attention to the 20 checkpoints that satisfy the local sparsity criterion, all qualifying coordinates occur in layerwise hidden states. The residual stream is therefore the dominant carrier through which extreme activation magnitudes are propagated and preserved.

- 5 What Controls Peak Magnitude?

#### 5.1 Within-family scaling

- Figure 5 compares checkpoints of different sizes within the same family and model form. Most families, including Qwen2.5, Qwen3.5, and Gemma3, show a stable within-family scale effect: the global maximum activation increases with parameter count. Gemma2 is the main local non-monotonic exception, with the 9B checkpoint peaking below the 2B checkpoint before the 27B checkpoint rises again. These results suggest that, when model family and training form are fixed, model size often amplifies activation extremes, although individual checkpoints can still deviate due to training or recipe differences.

Within-family scaling of global activation maxima

| | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |Qwe|n2.|5|Qwen3|G|emma2| | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | |Qwe|n2.|5-V|L Qwen3.5|G|emma3| | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |G3-2|7B-it| |
| | | | | | | | | | | | | | | |G2-|27B| |
| | | | | | | |G3-4B-i|t| | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |3-32B| | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | |3-8|B| | | |2.5-|32B|
| | | | | |3-1.7B| | | | |2.5-|7B<br><br>| | | | | | |
| | | | | |2.5-1.5B| | | | | | | | | | |VL-32B| |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | |G2-2B| | | | | | | | | | | | |
| | | | | | | | | | |VL-|7B| | | | | | |
| | | | | | | | | | | | |G2-|9B| | | | |
| | | | | | |VL-3B| | | | | | | | | | | |
| | | | | | | | | | | | | | | |3.5-|27B| |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | |3.5-|9B| | | | |
| | | | | | | | | | | | | | | | | | |
| |3.5-|0.8B| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |

- 102
- 103
- 104
- 105
- 106

Globalmaximum|activation|

100 101

Model size within the same family (B parameters)

- Figure 5: Within-family scaling effects. The figure compares size changes only within the same family and model form; crosses mark local non-monotonic cases. The vertical axis is global maximum activation magnitude on a log scale.

5.2 Cross-family magnitude differences

- Figure 6 summarizes the global maximum activation magnitudes of all 24 main-analysis checkpoints. Cross-family variation is much larger than within-family scaling variation, spanning several orders of magnitude. For example, Qwen3.5 is concentrated in a low-magnitude regime around hundreds to low thousands, whereas Gemma3-27B-it reaches a global maximum of 696,320. This directly shows that the severity of maximum activations is strongly reshaped by family-level architecture and training choices.

#### 5.3 Non-monotonic generational evolution

Appendix Figure 7 compares generational trends at similar model sizes. Maximum activation magnitude does not monotonically shrink or grow with release time; instead, it is highly familydependent. Across three size groups, Qwen exhibits an inverted-V trajectory: maximum activations

Global maximum activation landscape across 24 base checkpoints

- 102

- 103

- 104

- 105

- 106

Qwen2.5 Qwen2.5-VL

Qwen3

- Gemma2

- Gemma3

Ling

Qwen3.5

GPT-OSS

| |
|---|

×: no Sun-pass hidden state

Globalmax|activation|(logscale)

Qwen3.5-0.8BQwen3.5-35B-A3BQwen3.5-27BQwen3.5-9BQwen3-30B-A3Bgemma-2-9bgemma-2-2bQwen2.5-VL-3BLing-mini-5TQwen2.5-1.5BQwen2.5-VL-7BLing-mini-10TLing-mini-15TLing-mini-20TQwen2.5-7BQwen3-1.7BQwen3-8BQwen2.5-32BQwen2.5-VL-32BQwen3-32Bgpt-oss-20bgemma-2-27bgemma-3-4b-itgemma-3-27b-it

- Figure 6: Global maximum activation magnitudes for the 24 main-analysis checkpoints. The vertical axis is on a log scale. A × marker indicates a model that does not satisfy the massive-activation existence criterion.

increase from Qwen2.5 to Qwen3 and are then strongly suppressed in Qwen3.5. In contrast, Gemma shows a sharp increase from Gemma2 to Gemma3 across both matched size groups.

The detailed same-scale generational plot is moved to Appendix Figure 7 to keep the main-text submission within the nine-page content budget.

#### 5.4 Empirical rule of thumb

Taken together, these measurements suggest a practical rule of thumb. Within a fixed family, increasing parameter count often increases maximum activation magnitude. Once family or generation changes, however, design and training differences can override this monotonicity. For inference and quantization boundaries, family identity and generation are therefore at least as important as parameter count. The appendix extends this logic to matched MoE-vs-dense, vision-language-vs-text, Base-vs-Instruct, and training-stage contrasts.

### 6 Deployment Takeaways and Conclusion

The preceding sections establish a compact logic for deployment: define the relevant upper-bound statistic, verify that it is not reducible to a binary outlier label, locate its carrier, and then compare matched model factors. This logic supports three takeaways. First, maximum activation magnitude is a family- and architecture-dependent model property: MoE routing, modality adaptation, instruction tuning, and training stage all shift either the global peak or its layerwise carrier. Second, the residual stream is the dominant carrier of extreme values, so activation quantization and scaling policies should inspect hidden-state peaks rather than only attention or MLP outputs. Third, the INT-8 probe in Appendix D shows that larger measured peaks can translate into lower reconstruction SQNR through scale selection, making M a practical model-card statistic rather than only a descriptive outlier measure.

In summary, M = max|a| is not predicted by parameter count alone: within-family scaling often increases M, but cross-family, cross-generation, and cross-architecture comparisons break monotonic trends by orders of magnitude. Detailed matched-factor analyses, quantization checks, limitations, and supplementary figures are provided in the appendix.

### References

- [1] Yongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang. Systematic outliers in large language models. In International Conference on Learning Representations (ICLR), 2025.
- [2] Anand, Umberto Cappellazzo, Stavros Petridis, and Maja Pantic. Mitigating attention sinks and massive activations in audio-visual speech recognition with LLMs. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026.
- [3] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-free 4-bit inference in rotated llms, 2024.
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL technical report, 2025.
- [5] Yelysei Bondarenko, Markus Nagel, and Tijmen Blankevoort. Quantizable transformers: Removing outliers by helping attention heads do nothing. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
- [6] Mengzhao Chen, Yuxuan Liu, Jiahao Wang, Yi Bin, Wenqi Shao, and Ping Luo. PrefixQuant: Static quantization beats dynamic through prefixed outliers in LLMs, 2024.
- [7] Yinjie Chen, Zipeng Yan, Chong Zhou, Bo Dai, and Andrew F. Luo. Vision transformers with self-distilled registers. In Advances in Neural Information Processing Systems (NeurIPS), 2025.
- [8] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In International Conference on Learning Representations (ICLR), 2024.
- [9] DeepSeek-AI. Insights into DeepSeek-V3: Scaling challenges and reflections on hardware for AI architectures, 2025.
- [10] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang,

- X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei,
- Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou,
- Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu,

Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. Deepseek-v3 technical report, 2025.

- [11] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. LLM.int8(): 8-bit matrix multiplication for transformers at scale. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [12] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization for generative pre-trained transformers. In ICLR, 2023.
- [13] Gemma Team. Gemma 2: Improving open language models at a practical size, 2024.
- [14] Gemma Team. Gemma 3 technical report, 2025.
- [15] Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min Lin. When attention sink emerges in language models: An empirical view. In International Conference on Learning Representations (ICLR), 2025.
- [16] Prannay Kaul, Chengcheng Ma, Ismail Elezi, and Jiankang Deng. From attention to activation: Unravelling the enigmas of large language models. In International Conference on Learning Representations (ICLR), 2025.
- [17] Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song, Zhenan Sun, and Ying Wei. DuQuant: Distributing outliers via dual transformation makes stronger quantized LLMs. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [18] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware weight quantization for on-device llm compression and acceleration. In MLSys, 2024.
- [19] Ling Team. Every FLOP counts: Scaling a 300b mixture-of-experts LING llm without premium gpus, 2025.
- [20] Ling Team. Towards greater leverage: Scaling laws for efficient mixture-of-experts language models, 2025.
- [21] Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. SpinQuant: Llm quantization with learned rotations, 2024.
- [22] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In International Conference on Machine Learning (ICML), 2024.
- [23] Iuri Macocco, Nora Graichen, Gemma Boleda, and Marco Baroni. Not a nuisance but a useful heuristic: Outlier dimensions favor frequent tokens in language models. In Proceedings of the 8th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, 2025.
- [24] OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025.
- [25] Enrique Queipo-de Llano, Álvaro Arroyo, Federico Barbero, Xiaowen Dong, Michael Bronstein, Yann LeCun, and Ravid Shwartz-Ziv. Attention sinks and compression valleys in LLMs are two sides of the same coin, 2026.
- [26] Qwen Team. Qwen2.5 technical report, 2024.
- [27] Qwen Team. Qwen3 technical report, 2025.
- [28] Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Zhengzhong Liu, Hongyi Wang, Bowen Tan, Joel Hestness, Natalia Vassilieva, Daria Soboleva, and Eric Xing. Slimpajama-dc: Understanding data combinations for llm training, 2024.
- [29] Mingjie Sun, Xinlei Chen, J. Zico Kolter, and Zhuang Liu. Massive activations in large language models. In Conference on Language Modeling (COLM), 2024.

- [30] Shangwen Sun, Alfredo Canziani, Yann LeCun, and Jiachen Zhu. The spike, the sparse and the sink: Anatomy of massive activations and attention sinks, 2026.
- [31] Yuxuan Sun, Ruikang Liu, Haoli Bai, Han Bao, Kang Zhao, Tiancheng Li, Chenghua Chen, Xin Hu, Chen Yu, Lu Hou, Chun Yuan Tu, Yuen-Hin Yeung, Yu Xu, Qi Tian, and Wulong Liu. FlatQuant: Flatness matters for LLM quantization. In International Conference on Machine Learning (ICML), 2025.
- [32] Xiuying Wei, Yunchen Zhang, Yuhang Li, Xiangguo Zhang, Ruihao Gong, Jinyang Guo, and Xianglong Liu. Outlier suppression+: Accurate quantization of large language models by equivalent and effective shifting and scaling. In EMNLP, 2023.
- [33] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning (ICML), 2023.

### A Supplementary Experiments Model Details

Table 2: The 24 checkpoints included in the main analysis. Gemma3 uses publicly released instructiontuned checkpoints; therefore, the Gemma2/Gemma3 comparison is not interpreted as a strict base-tobase ablation.

Model family Checkpoints Role in analysis Qwen2.5 1.5B, 7B, 32B Dense scaling baseline

- Qwen2.5-VL 3B, 7B, 32B Vision-language adaptation Qwen3 1.7B, 8B, 30B-A3B, 32B Dense/MoE comparison
- Qwen3.5 0.8B, 9B, 27B, 35B-A3B Low-outlier counterexample family

- Gemma2 2B, 9B, 27B Non-monotonic scaling case
- Gemma3 4B-it, 27B-it High-magnitude family Ling 5T, 10T, 15T, 20T Training-stage evolution GPT-OSS 20B Component-level special case

### B Supplementary Main-Text Figures

Matched-scale generation trends in global activation maxima

Qwen generations at matched scales

Gemma generations at matched scales

- 102

- 103

- 104

- 105

- 106

696K

Qwen2.5 Qwen3 Qwen3.5

Gemma2 Gemma3

246K

158K

Globalmaximum|activation|

35.3K

30.8K

17.7K

14.2K

13.2K

8.0K

1.8K

2K

956

122

~1B ~7-9B ~27-32B

mid scale large scale

Figure 7: Generational evolution at similar sizes. Left: Qwen shows a Qwen2.5→Qwen3 increase followed by a Qwen3→Qwen3.5 decrease across three size groups. Right: Gemma2→Gemma3 increases in both size groups. The vertical axis is global maximum activation magnitude on a log scale.

### C Architectural and Training Factors at Matched Scale

This section provides controlled comparisons at identical or similar model scales. We compare MoE versus dense models, vision-language versus text-only checkpoints, base versus instruction-tuned checkpoints, and different Ling-mini training stages.

#### C.1 MoE vs. Dense Models

The MoE-versus-dense comparison is the clearest matched-design contrast we have for an architectural axis. Figure 8 compares two pairs of same-family checkpoints near the 30B scale. Qwen330B-A3B has a global peak of 1,512, which is 23.4× lower than Qwen3-32B at 35,328. Qwen3.535B-A3B has a global peak of 132, which is 14.0× lower than the Qwen3.5-27B dense counterpart (the value plotted is the global M across all hooked components and layers, which differs from the representative-layer Top-1 reported in Table 1; see Section 3). The compared checkpoints have similar total parameter counts but MoE models activate fewer parameters per token and route tokens through sparse expert paths. The matched-pair gap is therefore unlikely to be explained by active-parameter count alone; it is consistent with an architectural effect of sparse routing and expert structure. We

caution, however, that with only n = 2 matched pairs and possible unobserved differences in training recipe between paired checkpoints, this contrast is observational rather than causal.

[Figure 4]

- Figure 8: Matched-scale comparison of MoE and dense checkpoints. Each bar group fixes model family and approximate total parameter scale while changing only the dense/MoE form. The vertical axis is global maximum activation magnitude on a log scale.

#### C.2 Vision-language vs. Text-only models

Vision-language adaptation does not eliminate extreme activations, but matched-scale checkpoints differ measurably in peak magnitude. Figure 9 compares Qwen2.5-VL with text-only Qwen2.5 at 7B and 32B. Qwen2.5-VL-7B reaches a global peak of 8,256, 1.6× lower than Qwen2.5-7B at 13,248. At 32B, Qwen2.5-VL reaches 22,144, 1.4× lower than the text-only Qwen2.5-32B counterpart, whose global M is 30,848 across all hooked components and layers (the corresponding Top-1 entry in Table 1, 22,144, is the value at the representative passing-criterion layer; see the Section 3 convention note). Vision-language checkpoints therefore remain in a high-magnitude regime comparable to their text-only backbones, but modality adaptation and training-recipe changes co-occur with a moderate same-scale shift in the maximum activation. As with the MoE contrast, this is observational evidence based on n = 2 matched pairs within a single family.

#### C.3 Base vs. Instruct

Instruction tuning affects maximum activations differently from model scaling. Figure 10 compares Base and Instruct checkpoints at the same Qwen2.5 backbone and parameter scale. At 1.5B, the global peak remains unchanged at 7,968. At 7B, it increases slightly from 13,248 to 13,312 (1.005×). At 32B, it decreases from 30,848 to 22,144, a 1.4× reduction. The 32B Base value of 30,848 is the global M across all hooked components and layers; the Top-1 entry of 22,144 that appears for Qwen2.5-32B in Table 1 is taken at the representative passing-criterion layer, and is by construction ≤ M (see the Section 3 convention note). A closer inspection of layerwise hidden-state peaks shows that SFT mainly changes late layers rather than reorganizing the mid-layer structure. For 1.5B, 7B, and 32B, the stable high-peak middle-layer regions remain at 1.000×, 1.005×, and 1.000× of their base values, respectively. The final-layer peaks decrease from 1,536 to 840 (−45%), from 4,864 to 2,528 (−48%), and from 30,848 to 21,248 (−31%), with only mild compensation in the penultimate layer. The global-peak decrease in Qwen2.5-32B-Instruct is therefore best understood as late-layer compression that exposes an already existing mid-layer high-peak region as the global maximum, rather than the creation of a new activation structure.

Text-only checkpoints vs vision-language adaptation

Qwen2.5 text Qwen2.5-VL

Globalmaximum|activation|

1.4× lower

30.8K

22.1K

1.6× lower

13.2K

104

8.3K

7B 32B

- Figure 9: Matched-scale comparison between Qwen2.5-VL and text-only Qwen2.5 checkpoints. The two bar groups correspond to 7B and 32B scales. The vertical axis is global maximum activation magnitude on a log scale.

1.5B 7B 32B

104

Globalmaximum|activation|

8.0K

13.2K

30.8K

8.0K

13.3K

22.1K

unchanged

unchanged

1.4× lower

Base checkpoints vs instruction-tuned checkpoints

Base Instruct

- Figure 10: Matched-backbone comparison of Qwen2.5 Base and Instruct checkpoints. Each bar group fixes parameter scale and changes only whether the checkpoint has undergone instruction tuning. The vertical axis is global maximum activation magnitude on a log scale.

#### C.4 Training-stage evolution

Ling-mini provides a training-stage comparison with fixed family and approximately fixed model scale. Figure 11 shows that as training increases from 5T to 20T tokens, the global maximum activation rises monotonically from 7,648 to 9,024, then to 9,600, and finally to 10,240, for an overall increase of 1.34×. In this training sequence, longer training is associated with larger activation peaks. Even when family, architecture, and parameter scale are approximately held fixed, training progress itself co-occurs with a gradual increase in extreme activation magnitudes. Training stage should therefore be recorded as a separate observation dimension alongside model size and architecture, although the magnitude of this effect (1.34×) is small relative to the cross-family span.

Ling training-stage evolution

2 × 104

5T→20T: 1.54× higher

Globalmaximum|activation|

11.8K

10.1K

104

9.0K

7.6K

6 × 103

5T 10T 15T 20T

Training tokens

- Figure 11: Global maximum activation across Ling-mini training stages. The horizontal axis is training tokens, and the vertical axis is global maximum activation magnitude on a log scale.

Overall, MoE, vision-language adaptation, SFT, and training stage are all associated with measurable differences in maximum activations, but the contrasts differ in magnitude and in the strength of evidence available. The MoE-vs-dense gap is large (an order of magnitude or more) but rests on n = 2 matched pairs. The SFT contrast is the strongest matched design (n = 3 same-backbone sizes) and reveals a layerwise-localized effect (late-layer compression with preserved middle-layer high-peak regions). The vision-language and training-stage contrasts produce moderate within-family shifts. None of these comparisons constitute causal evidence; each could in principle reflect unobserved training-recipe differences between the paired checkpoints.

### D Discussion

#### D.1 Distinction from outlier features

The object of this study is maximum activation magnitude: the largest absolute activation observed across layers and key components under a unified evaluation corpus. This is a model-level statistic oriented toward dynamic range and deployment risk. It asks how large an activation upper bound a checkpoint may produce during actual forward inference. By contrast, outlier-feature studies often focus on whether certain feature dimensions remain persistently abnormal across many tokens or samples. We therefore do not interpret maximum activation magnitude as a fixed outlier-feature existence test. Instead, we treat it as a direct risk indicator for quantization scales, activation rescaling, and inference stability.

#### D.2 Implications for quantization and deployment

Maximum activation magnitude directly constrains the upper bound of activation scale, and therefore affects scale selection and reconstruction error in low-bit activation quantization. Higher peaks make per-tensor quantization more likely to be dominated by a few extreme values; however, quantization error is also shaped by the full distribution at the peak layer, the effective signal magnitude, and the clipping strategy. Maximum activation magnitude should therefore be treated as an important prior for quantization risk, rather than as the sole determinant of end-to-end quantization quality. Appendix Figure 13 provides a deployment-oriented grouping based on global maximum activation magnitude.

To validate this deployment relevance, we conduct a lightweight INT-8 activation quantization sanity check. The experiment covers eight representative models spanning Qwen2.5, Qwen3, Qwen3.5, Gemma3, and MoE/dense contrasts. For each model, we use 128 samples for calibration and 256 samples for evaluation at the peak hidden layer, and compare two per-tensor symmetric quantization strategies: max-abs scaling and 99.9% clipping. Figure 12 shows that Qwen3.5-0.8B maintains the highest SQNR under both strategies, at 29.1 dB and 26.3 dB. Most medium- and high-peak models obtain roughly 10–14 dB under max-abs scaling and further drop to approximately 0.2–0.4 dB under 99.9% clipping. Qwen3.5-9B and Qwen3.5-35B-A3B have relatively low absolute peaks but still show substantially lower SQNR than Qwen3.5-0.8B, indicating that within-layer distributional structure beyond peak magnitude also affects quantization error. Overall, this sanity check demonstrates that maximum activation magnitude can translate into observable activation reconstruction error through scale selection, while also suggesting that end-to-end quantization should be evaluated with richer distributional and task-level measurements.

We deliberately keep this experiment as a deployment-relevance sanity check rather than a calibrated dose–response curve. The eight checkpoints are chosen to span the main regimes identified by the survey—low-peak dense (Qwen3.5-0.8B), low-peak Qwen3.5 variants, medium-baseline Qwen2.5, high-peak dense Qwen3 checkpoints, a MoE checkpoint (Qwen3-30B-A3B), and a high-peak Gemmafamily checkpoint (Gemma3-4B-it)—so that any M-dependence of SQNR has a chance to manifest at the same layer hooks used in the rest of the study. The two recipes (max-abs and 99.9% clipping) bracket common per-tensor symmetric scale-selection strategies and are directly affected by the activation upper bound; advanced mitigations such as rotation-based quantization [3, 21] or KV-cache prefixing [6] are designed to neutralize M rather than to expose it, so they are intentionally not part of this probe. We therefore read the M-versus-SQNR relationship as qualitative evidence that maximum activation magnitude translates into observable scale-selection error, not as a generalized scaling law over all 24 checkpoints.

Relation to 2024–2026 quantization advances. A wave of post-2024 quantization techniques—dualrotation DuQuant [17], learned affine flattening FlatQuant [31], and KV-cache-targeted KIVI [22], alongside the earlier rotation-based QuaRot/SpinQuant [3, 21] and prefix-based PrefixQuant [6]—has substantially narrowed the SQNR gap that motivates max-abs scaling, and a parallel low-precisionpretraining line [10, 9] folds analogous mitigations into FP8 training itself. These methods transform away or absorb the activation upper bound; none of them removes it. Our contribution is orthogonal: we measure the upper bound itself across a far wider family/architecture/training-stage span than any of these papers calibrates against, and we show that M varies by nearly four orders of magnitude across modern open releases. The cost of any of the above mitigations—rotation rank, prefix-token budget, KV-cache bit width, or FP8 block-scale granularity—is itself a monotone function of the dynamic range we report, so the per-checkpoint M values remain a deployment-relevant model card entry even when none of the mitigations are bypassed.

#### D.3 Threats to validity and limitations

The conclusions of this paper should be interpreted within the boundaries of an observational study. We measure maximum activations across open checkpoints under a unified evaluation protocol and analyze their relationships with family, architecture, and training stage. These results reveal stable empirical differences but do not by themselves establish causal training mechanisms. Second, the study covers open LLMs only; closed models or models with unreleased training recipes may exhibit different activation dynamics. Third, maximum activation is an extreme statistic and could be affected by longer contexts, rare inputs, or larger evaluation corpora. Nevertheless, our subsampling experiment shows that representative models reproduce the same peak orders of magnitude across repeated subsamples. Finally, the INT-8 experiment is a deployment-relevance sanity check rather than

[Figure 5]

- Figure 12: INT-8 activation quantization sanity check for eight representative models. Grouped horizontal bars show SQNR at the peak hidden layer under max-abs scaling and 99.9% clipping. The dashed line marks 20 dB; higher values indicate lower quantization error. Right-side annotations report the ratio between the global peak and the 99.9% clipping threshold.

a complete end-to-end quantization evaluation. Future work could combine intervention experiments, token/feature localization, and end-to-end quantized tasks to explain the mechanisms that create these peaks and determine how controllable they are.

Several further limitations are worth flagging explicitly; these are items that the present manuscript does not address but that we view as the natural next iteration. (i) Corpus coverage. Our 5,000sample evaluation mixture is restricted to English, Chinese, and code domains; activations on long-tail languages, mathematical reasoning chains, or tool-use traces may differ. (ii) Context length. All measurements use sequences up to 4,096 tokens, so we cannot speak to whether maxima grow, saturate, or shift carriers under 32k–128k contexts that several of these checkpoints support. (iii) Training-stage labels. Our Base-vs-Instruct contrast treats “instruction-tuned” as a single bucket, but the public Instruct checkpoints conflate supervised fine-tuning with downstream RLHF or DPO stages whose training data we do not control; a finer-grained decomposition would require checkpoint releases at each stage. (iv) Quantization scope. The INT-8 sanity check is restricted to 8 checkpoints, 1 layer each, and per-tensor recipes only; a 24-point regression of SQNR against log10 M, and a comparison against rotation-based [3, 21] or prefix-based [6] mitigations, would be required to turn the link from M to deployment cost into a calibrated dose–response curve. (v) Layerwise-pattern classification. The two-pattern dichotomy in Section 4 is identified qualitatively from the depthnormalized heatmap; a quantitative jump-score / clustering diagnostic would be required to convert it into a numerical classifier. (vi) Sun-criterion failure mechanisms. Linking each failing checkpoint (Qwen2.5-1.5B, Qwen3.5-0.8B/-9B/-35B-A3B) to specific normalization or attention dynamics—for example via residual-stream RMS profiles or via the early-layer step-up blocks identified by [30]—is left to a mechanistic follow-up. (vii) Reproducibility artifacts. The full reproducibility appendix (HuggingFace repository ID and commit revision per checkpoint, dtype, attention implementation, normalization variant, RoPE base, pinned versions of torch/transformers/accelerate, random seeds, code-availability statement) will accompany the public release; the current text describes the pipeline but does not yet pin its concrete artifacts. (viii) Sample sizes for matched comparisons. The MoE-vs-dense and VL-vs-text contrasts each rest on n = 2 matched pairs within a single family, the training-stage contrast on 1 family with 4 stages, and only the Base-vs-Instruct contrast achieves n = 3; effect-size statements should therefore be read as direction-of-effect rather than as population-level estimates, and additional matched pairs from other families would be required to strengthen them. None of these gaps is fundamental to the measurement framework; they are scoping

choices for the present submission, and each is a candidate for the next iteration of this study rather than a planned addition to this manuscript.

### E Related Work

Outlier features and massive activations. Large activations in transformer language models were first brought into focus by low-bit inference failures. [11] showed that sufficiently large OPT- and BLOOM-style models develop sparse, high-magnitude “emergent features” whose preservation is necessary for lossless INT8 inference. [29] sharpened this observation into the notion of massive activations: token-feature coordinates that are both absolutely large and locally sparse relative to the surrounding hidden-state distribution. This line of work established two facts that are now central to activation-aware deployment: the extreme coordinates occupy a tiny fraction of the representation, yet ablating or quantizing them naively can cause disproportionate degradation. Subsequent analyses extended the taxonomy beyond hidden-state activations. [1] relate activation outliers, weight outliers, and attention outliers, arguing that they are mutually structured rather than independent numerical accidents, while [23] show that last-layer outlier dimensions can implement useful high-frequencytoken prediction heuristics. These studies motivate treating extreme activations as functional model components, not merely as pathological noise.

Attention sinks and mechanistic accounts. A major mechanistic explanation links massive activations to the softmax attention constraint. When an attention head has no useful context to retrieve, the probability simplex still forces it to allocate its mass somewhere; models often learn to route this mass to a beginning-of-sequence token or another low-information token, creating an attention sink. [5] connect such no-op behavior to quantization difficulty, and [15] empirically trace when sink behavior emerges during pretraining. More recent work refines this picture. [16] attribute first-token attention dominance to softmax geometry and large hidden-state kurtosis partly to coordinate-wise adaptive optimization, proposing softmax-1 and OrthoAdam as architectural and optimizer-level remedies. [30] decouple residual-stream activation spikes from local attention sinks, showing that both phenomena co-occur in pre-norm transformers but need not be the same object. [25] further link attention sinks to compression valleys, proving that extreme residual-stream norms induce representational compression and proposing a mix-compress-refine view of depth-wise computation. Together these works explain why massive activations may be useful for information routing and stabilization, while also clarifying why their magnitude can become a deployment bottleneck.

Quantization and numerical mitigation. The practical importance of massive activations is most visible in post-training quantization. Because per-tensor or per-channel scales are often governed by the largest observed magnitude, even a small number of extreme values can waste quantization levels and increase reconstruction error for ordinary activations. LLM.int8() preserves sparse outlier dimensions in higher precision while quantizing the remaining bulk activations [11]. SmoothQuant instead migrates activation difficulty into weights through an algebraically equivalent rescaling [33], while Outlier Suppression+ shifts and rescales activations to reduce quantization sensitivity [32]. Weight-only and activation-aware methods such as GPTQ and AWQ reduce memory and bandwidth pressure while protecting salient parameters [12, 18]. More recent methods transform the representation basis or flatten the activation landscape: QuaRot and SpinQuant use rotations to distribute outlier mass [3, 21], DuQuant applies dual transformations [17], FlatQuant learns affine flattening transforms [31], and PrefixQuant or KIVI target outliers in prefixed activations and KV caches [6, 22]. These methods are complementary to our study: they reduce or absorb the numerical effect of large activations, whereas we measure the upper-bound activation magnitude itself across current open model families.

Architectural and multimodal interventions. A parallel line of work asks whether outlier-like artifacts arise because transformer architectures lack explicit storage locations for global or null information. In vision transformers, high-norm artifact tokens appear in low-information image patches and harm dense prediction; register tokens provide dedicated learned slots that absorb this global computation and remove the artifact pattern [8]. Post-hoc register methods show that similar benefits can be obtained without full retraining by distilling artifact-free dense features into a lightly modified model [7]. In multimodal language models, [2] show that attention sinks and massive activations can emerge not only at the BOS token but also at intermediate low-semantic audio-visual

tokens during fine-tuning, and that decorrelating intermediate tokens from the BOS representation mitigates both the sink and activation effects. These results suggest that maximum activation behavior is sensitive to modality adaptation, token roles, and the availability of architectural “scratch space.”

Position of this work. Existing studies have primarily treated massive activations as a binary phenomenon, a mechanistic artifact, or an obstacle to be removed by a particular quantization method. They also concentrate on a limited set of earlier LLaMA-, OPT-, or BLOOM-style checkpoints. Modern open models now vary along many additional axes, including normalization and training recipes, dense versus MoE computation, multimodal adaptation, instruction tuning, and released intermediate training stages [26, 27, 4, 13, 14, 19, 24]. Our work is therefore complementary to both the interpretability and quantization literatures. Rather than asking only whether a model contains a massive activation under a fixed local criterion, we measure the continuous deployment-relevant statistic M = max|a| under a unified protocol, compare it across recent open families and matched architectural or training contrasts, and connect it to activation-quantization reconstruction error. This framing turns maximum activation magnitude into a model-level property that should be reported alongside open-weight releases.

### F Appendix Summary

We presented a unified measurement study of maximum activations across 24 checkpoints from 8 modern open LLM families. Our central message is empirical and definitional: the maximum activation magnitude M of an open LLM is a continuous, releasable model property that is not predicted by parameter count alone. Within a family, M often grows with model size, but crossfamily, cross-generation, and cross-architecture comparisons break simple monotonic scaling, and matched-design contrasts associate MoE routing, vision-language adaptation, supervised fine-tuning, and training stage with measurable shifts in M or in its layerwise structure. The residual stream carries the global maximum in 22 of 24 checkpoints, with GPT-OSS-20B as a single MLP-output exception that any quantization scheme should accommodate. A lightweight INT-8 sanity check further indicates that M co-varies with low-bit reconstruction error through its effect on activationscale selection. We therefore recommend that M and its layerwise carrier be reported alongside any open-weight release as part of a deployment-oriented model card extension. We deliberately stop short of causal claims about why families differ; explaining the mechanism behind these differences—and intervening on it—is left as future work.

### G Additional Figures

The appendix keeps only figures that complement the main-text conclusions and provide additional verification value. We omit separate plots for local-sparsity pass rates and per-family global peaks because their information overlaps with the main text and with Figures 6 and 5. The retained figures focus on three types of evidence: deployment-oriented magnitude tiers, per-family layerwise trajectories, and component-level carrier differences in representative models.

#### G.1 Deployment-oriented tiers

Figure 13 groups all main-analysis checkpoints into five order-of-magnitude tiers according to global maximum activation magnitude. The figure is not intended to propose a new threshold standard; rather, it provides an intuitive deployment-oriented reference for selecting quantization and activation-scaling strategies.

#### G.2 Per-family layerwise maximum activations

- Figure 14 shows hidden-state layerwise maximum-activation trajectories within each model family. The horizontal axis is normalized depth, and the vertical axis is hidden-state layerwise maximum absolute activation. Because peak magnitudes span several orders of magnitude across families, the vertical axis uses a log scale.

Deployment-oriented tiers by global maximum activation

6

T1 low (<2K)

6

T2 moderate (2K-10K)

8

T3 high (10K-50K)

2

T4 very high (50K-200K)

2

T5 extreme (>200K)

0 2 4 6 8

# checkpoints

- Figure 13: Deployment-oriented tiers based on global maximum activation magnitude. The horizontal bar chart counts checkpoints in each tier. The horizontal layout avoids crowded tier labels. The figure is based only on observed activation magnitudes and is intended to indicate that different models may require different quantization treatment.

- 102

- 103

- 104

Hiddenpeak|activation|

Qwen2.5

2.5-1.5B

2.5-7B

2.5-32B 101

- 102

- 103

- 104

Qwen2.5-VL

VL-3B VL-7B VL-32B

- 101

- 102

- 103

- 104

Qwen3

3-1.7B

3-8B

3-30B-A3B

3-32B

- 100

- 101

- 102

- 103

Hiddenpeak|activation|

Qwen3.5

3.5-0.8B

3.5-9B

3.5-27B

3.5-35B-A3B

- 102

- 103

- 104

- 105

Gemma2

G2-2B G2-9B G2-27B

- 104

- 105

Gemma3

G3-4B-it

G3-27B-it

0.0 0.2 0.4 0.6 0.8 1.0

Normalized layer depth

- 101

- 102

- 103

- 104

Hiddenpeak|activation|

Ling

5T

10T 15T 20T

0.0 0.2 0.4 0.6 0.8 1.0

Normalized layer depth

- 102

- 103

- 104

Qwen2.5-IT

2.5-1.5B-IT

2.5-7B-IT

2.5-32B-IT

0.0 0.2 0.4 0.6 0.8 1.0

Normalized layer depth

- 103

- 104

- 105

GPT-OSS

GPT-OSS-20B

Per-family hidden-state layerwise maximum activation trajectories

- Figure 14: Hidden-state layerwise maximum-activation trajectories within each model family. Each subplot corresponds to a family or training form, and each curve corresponds to one checkpoint in that group. The vertical axis uses a log scale to accommodate cross-family magnitude differences.

- G.3 Component-level maximum activation distributions
- Figure 15 presents component-level layerwise trajectories for three representative checkpoints, complementing the main-text discussion of carrier components. The three subplots correspond to a low-peak model, a high-peak model, and the GPT-OSS component-level exception. Compared with plotting all checkpoints, these representative panels more clearly show the magnitude differences among hidden states, attention outputs, and MLP/MoE outputs.

Component-wise layer trajectories for representative checkpoints

Qwen3.5-9B (low peak)

Gemma3-27B-it (high peak)

GPT-OSS-20B (component exception)

- 101
- 102
- 103
- 104
- 105
- 106

| | | | | | | |
|---|---|---|---|---|---|---|
|Hidden<br><br>Attention<br><br>MLP/MoE| | | | | | |
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
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

- 101
- 102
- 103

- 102
- 103
- 104
- 105

Componentpeak|activation|

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Normalized layer depth

Normalized layer depth

Normalized layer depth

- Figure 15: Component-level maximum-activation trajectories for representative models. The three subplots show a low-peak model, a high-peak model, and the GPT-OSS component-level exception. Curves correspond to hidden states, attention outputs, and MLP/MoE outputs; the vertical axis is component-level maximum absolute activation on a log scale.

