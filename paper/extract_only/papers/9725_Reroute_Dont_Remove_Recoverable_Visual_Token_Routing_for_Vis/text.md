# arXiv:2606.12412v1[cs.CV]10Jun2026

## Reroute, Don’t Remove: Recoverable Visual Token Routing for Vision-Language Models

Cheng-Yu Yang National Yang Ming Chiao Tung University elm855220@gmail.com

Shao-Yuan Lo National Taiwan University sylo@csie.ntu.edu.tw

Yu-Lun Liu National Yang Ming Chiao Tung University yulunliu@cs.nycu.edu.tw

###### Abstract

Vision-language models (VLMs) project images into hundreds to thousands of visual tokens, making decoder inference expensive in both attention computation and KV-cache memory. Existing visual-token reduction methods largely follow a rank-and-remove paradigm: they score visual tokens, keep a compact subset, and permanently discard the rest. We show that this irreversible action is fragile because visual-token importance changes across decoder depth; tokens ranked low at one stage may become relevant in later layers, especially for grounding-sensitive queries. We propose Reroute, a training-free plug-in that replaces removal with recoverable routing. At each routing stage, selected vision tokens pass through decoder blocks, while deferred tokens bypass the stage and re-enter the candidate pool at the next routing decision. Reroute reuses existing attention-score ranking rules and stage-wise schedules, preserving the theoretical TFLOPs and KV-cache budget class of the pruning method it augments. Across FastV, PDrop, and Nüwa variants on LLaVA-1.5 and Qwen backbones, reroute improves grounding under aggressive token reduction while maintaining general VQA performance. These results suggest that VLM token reduction should not be viewed only as irreversible pruning, but also as recoverable routing. The code can be found here: https:

//github.com/elmma/mllm-reroute/.

###### 1 Introduction

Vision-language models (VLMs) [41, 5] project images into hundreds to thousands of tokens that are fed to a language-model decoder, making inference cost grow quadratically with attention and linearly with KV-cache. Reducing the number of visual tokens that participate in decoder computation has therefore become a key lever for deploying VLMs at scale [12, 71, 26].

Existing methods follow a common rank-and-remove recipe. Vision-encoder-side approaches [73, 58] compress tokens before they enter the decoder. Decoder-side methods intervene inside the LLM: FastV [12] makes a single-shot pruning decision at an early layer using text-to-vision attention; PDrop [71] generalizes this to a progressive multi-stage schedule; FEATHER [17] and Nüwa [26] refine the scoring rule to better preserve spatial evidence. Across these methods, the underlying action is identical. Once a visual token is deemed unimportant, it is removed from the sequence and cannot contribute to any later layer.

This paradigm assumes token importance is stable across decoder depth. Our analysis often shows this assumption is not true. Figure 2(a) tracks attention from query words to visual tokens across depth: shallow layers attend diffusely, while target-relevant regions emerge only in middle and deep

Preprint.

###### LLaVA-1.5-7B

###### Qwen2.5-VL-7B

###### Qwen3.5-9B

avg token: 576 -> 64 (88.9% reduction)

avg token reduction = 88.9%

avg token reduction = 88.9%

Query: “person holding animal”

Query: “man in white shirt”

Query: “man in jeans out of focus”

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

PDropPDrop+Reroute

PDropPDrop+Reroute

PDropPDrop+Reroute

|IoU 0.21 ❌<br><br>[Figure 5]|
|---|

|IoU 0.22 ❌<br><br>[Figure 6]|
|---|

|IoU 0.38 ❌<br><br>[Figure 7]|
|---|

[Figure 8]

[Figure 9]

[Figure 10]

|IoU 0.88 ✅<br><br>[Figure 11]|
|---|

|IoU 0.72 ✅<br><br>[Figure 12]|
|---|

|IoU 0.80 ✅<br><br>[Figure 13]|
|---|

- Figure 1: Reroute, don’t remove. Under aggressive 88.9% visual token reduction (576→64 visual tokens), conventional pruning permanently discards visual evidence and grounding collapses (top rows, IoU < 0.4). Replacing irreversible removal with our recoverable routing, using the same scorer (PDrop) and the same token budget, we can recover grounding accuracy across three backbones spanning early-generation MLLM (LLaVA-1.5-7B), native dynamic resolution (Qwen2.5-VL-7B), and hybrid Mamba-Transformer (Qwen3.5-9B). Reroute is a training-free plug-in that improves grounding at matched theoretical cost.

layers. Figure 2(b) follows one ground-truth token for “man in green shirt”: its attention percentile is

- 0.11 at layer 3 (where FastV decides) and 0.25 at layer 8 (where PDrop decides). Both methods irreversibly remove it, yet it climbs to 0.97 at layer 25. As Figure 1 shows, this irrecoverability causes outright grounding collapse under 88.9% token reduction across three backbones, with IoU falling below 0.4.

We argue the limitation lies not in the scorer but in the action taken after scoring. We propose Reroute, a training-free plug-in that replaces irreversible removal with recoverable routing (Figure 3). At each routing layer, Reroute reuses text-to-vision attention as the ranking signal and selects a top-r fraction of tokens to pass through Attn+FFN. Deferred tokens are not deleted. They bypass the stage via a residual path and re-enter the candidate pool at the next routing layer, evaluated against the full visual-token set. A token deferred at layer 3 can be re-admitted at layer 15 if its score warrants it. Since only selected tokens enter Attn+FFN per stage, Reroute preserves the FLOPs and KV-cache budget of the pruning baseline it augments. From a conditional-computation view, Reroute is a training-free, attention-driven instantiation of mixture-of-depth [56] for VLM visual tokens, in contrast to learned-router approaches like γ-MoD [50]. We validate Reroute as a drop-in augmentation of FastV, PDrop, and Nüwa across LLaVA-1.5-7B, Qwen2.5-VL-7B, and Qwen3.5-9B, with the largest gains in grounding-heavy and aggressive-pruning regimes.

Our contributions are as follows:

- • Recoverable routing formulation. We recast decoder-side visual-token pruning as stagewise routing with deferred-token bypass, making conventional irreversible pruning the degenerate case where re-entry is forbidden. This enables a training-free, attention-driven realization of mixture-of-depth for VLM visual tokens.
- • Plug-in mechanism with matched theoretical efficiency. Reroute reuses existing scorers and schedules with no extra trainable parameters, preserving the FLOPs and KV-cache budget of the pruning method it augments.

[Figure 14]

- (a) Visual attention shifts across decoder depth
- (b) Important tokens are ranked low when pruning decides

[Figure 15]

- Figure 2: Why reroute? Important visual tokens are ranked low when pruning decides. (a) On a single RefCOCO image, attention from each query word (“boy”, “blue”, “jacket”, “gray”) to visual tokens drifts substantially across decoder depth. Shallow layers attend diffusely, while target-relevant regions only emerge at middle and deep layers. (b) Tracking one ground-truth token (⋆) inside the GT bbox of “man in green shirt”: its attention percentile is 0.11 at L3 (where FastV drops) and 0.25 at L8 (where PDrop drops), so both methods irreversibly remove it. Yet its importance climbs to 0.97 by L25, a token that is essential at deep layers but indistinguishable from background at the layers where pruning commits. Reroute keeps such tokens recoverable.

• Consistent gains across methods and backbones. Applied to FastV, PDrop, and Nüwa across three backbones, Reroute improves matched-budget pruning baselines on grounding benchmarks and preserves general VQA performance, with the largest gains under aggressive token reduction.

###### 2 Related Work

Vision-encoder-side visual token reduction. A long line of work compresses visual information before it enters the decoder. ViT pruning supplies the basic primitives, score-and-drop [55, 38] and similarity-based merging [7]. These ideas are imported into VLMs through methods that operate on the projector or [CLS] head: LLaVA-PruMerge [58], VisionZip [73], FasterVLM [82], and the budget-aware HiRED for sub-image partitioning [2]. A complementary thread redesigns the projector itself [35, 10, 23, 8, 77, 11, 63], while DeepStack instead distributes high-resolution tokens across decoder layers without enlarging the context [52]. Other works prune at the patch level with training-free saliency [51] or jointly compress encoder and decoder [22]. All commit to a fixed visual representation before language reasoning begins. In contrast, Reroute intervenes inside the decoder and revisits the visual-token set at every routing stage.

Decoder-side visual token pruning. FastV initiated this line by observing that vision-attention mass collapses after layer 2 and pruning with a single text-to-vision rule [12]. PyramidDrop generalized this to a progressive multi-stage schedule [71]; SparseVLM combined pre-LLM filtering with in-LLM recycling [84]; FEATHER traced FastV’s grounding failures to spatial bias and added

a uniform-coverage safety net [17]; and Nüwa enforces spatial integrity, becoming the strongest grounding-oriented baseline [26]. The same score-then-remove skeleton has been refined along nearly every axis: divergence-based calibration [75], layer-adaptive budgets with KV compression [21], multi-stage drops with a protected key-set [44], terminal visual withdrawal [40, 65], and diversity- or similarity-aware selection [1, 83, 72, 28, 43, 14]. Trained variants predict instance- and layer-specific keep ratios [25, 76, 87]. Across this body of work, the post-selection action is identical: deferred tokens are permanently removed and cannot influence later layers. The closest exception, Recoverable Compression, restores dropped evidence via text-guided similarity at a single recovery point [13]. We instead recast the entire decoder schedule as a sequence of recoverable routing decisions: a token deferred at one stage stays in the sequence and competes for re-entry at every subsequent stage.

Mixture-of-depth and conditional computation. Adaptive Computation Time [20] and early-exit networks [61, 24, 70, 45, 57] let inputs spend variable depth, but exit is terminal. Layer-skipping policies route inputs through a subset of blocks via learned gates [64, 68]. Mixture-of-Depths refines this for transformers: each layer routes only the top-k tokens through its block while the rest take a residual bypass [56]. γ-MoD [50] and p-MoD [81] extend MoD to MLLMs, and Routing Experts [66] and Mixture-of-Recursions [3] explore related dynamic-routing variants; a recent bypass-and-attend study analyzes how skipped positions interact with later attention [32]. Multimodal MoE methods pursue width-based rather than depth-based sparsity [39, 34, 88]. Reroute differs in two ways. First, where these methods train a new router (often the model with it), our router reuses the text-to-vision attention scores already computed by the decoder, making the mechanism training-free. Second, where they target the LLM’s own tokens, we target visual tokens, where layer-wise importance shifts are pronounced and existing pruning practice provides a free routing signal.

Layer-wise attention dynamics in VLM decoders. The phenomenon we exploit, visual-token importance, is unstable across depth. This has been documented from several angles. FastV first noted the post-layer-2 attention drop [12]; FasterVLM and VisPruner identified two failure modes, attention shift and attention dispersion [82, 83]. Massive Activations [60] and the Visual Attention Sink [29] explain part of this through tokens that absorb disproportionate attention without carrying matching semantics; analogous sinks were earlier reported in pure LLMs [69]. Recent VLM analyses argue for a phased shallow/middle/deep view of visual information flow [78], and MCA-LLaVA traces part of the drift to RoPE-induced positional bias [86]. Mechanistic studies of induction and retrieval heads provide broader context [53, 67]. Rather than pruning at a single “best” layer, we treat depth-dependent importance as a feature to be exploited and let deferred tokens re-enter when their score warrants it.

Efficient inference and KV-cache compression. Our work is complementary to KV-cache eviction [69, 85, 47, 19, 37, 9] and its multimodal extensions [62, 48], as well as to system-level optimizations [16, 31]. These methods decide which past states to keep in the cache; we decide which visual tokens deserve full computation in the first place. Reroute can in principle be combined with any of them.

###### 3 Preliminaries

Multimodal decoder notation. Let T ,V index text and vision token positions. At layer ℓ, hidden states Hℓ = {hℓj}j∈T ∪V are updated by a standard decoder block:

Hℓ+1 = H˜ℓ+1 + FFNℓ(LN2(H˜ℓ+1)), H˜ℓ+1 = Hℓ + Attnℓ(LN1(Hℓ)). (1)

Following the visual-token pruning literature, reduction targets only vision tokens; text tokens stay active throughout decoding, isolating the post-ranking action on low-ranked vision tokens.

Stage-wise visual-token selection. We cast decoder-side visual-token reduction as stage-wise selection. The decoder is partitioned into S stages, with stage i starting at routing layer ℓi with keep ratio ri. Let Ci ⊆ V denote the candidate vision-token set at the start of stage i, with C1 = V. At layer ℓi, a scorer (typically text-to-vision attention) ranks Ci and picks Visel = Top-Ki(Ci) with Ki = ⌊ri|V|⌋; the deferred set is Videf = Ci \ Visel. Since text tokens remain active in every stage, the active set is Ai = T ∪ Visel. The reduction method is defined by the action on Videf: pruning removes them, whereas Reroute keeps them recoverable.

Rank-and-remove as irreversible pruning. Conventional pruning removes Videf, contracting the candidate pool to Ciprune+1 = Visel. Once deferred, a vision token cannot be re-scored or re-selected in later stages. This irreversibility defines rank-and-remove pruning.

###### Input vision tokens 𝑽 = {𝒗𝒊}

𝑽𝒅𝒆𝒇 𝑽𝒔𝒆𝒍

###### : deferred tokens : selected tokens

1 2 3 4 5 6 7 8

###### Rank & Reroute mechanism for stage 𝒊

Text Tokens

1 2 3 4 5 6 7 8

###### Stage 1 (select 𝒓𝟏% tokens )

| | |
|---|---|
|Text-to-Vis|ion attention|
| | |
|Top-K (selec|t 𝒓 % tokens)|
| | |

- t1

- t2

Rank & Reroute

1 2 3 4 5 6 7 8

...

Transformer Layer ×𝑵

tm

1 2 3 4 5 6 7 8

###### Stage 2 (select 𝒓𝟐% tokens )

###### Reroute pipeline within stage 𝒊

Rank & Reroute

1 2 3 4 5 6 7 8

1 2 3 4 5 6 7 8

Rank & Reroute

Transformer Layer ×𝑵

deferred selected

1 3 5 8 2 4 6 7

...

Stage 𝑺 (select 𝒓𝑺% tokens )

Attn FFN

Rank & Reroute

×𝑵

1 2 3 4 5 6 7 8

merged

1 2 3 4 5 6 7 8

Transformer Layer ×𝑵

- Figure 3: Overview of Reroute. Reroute organizes decoder-side visual token reduction as a sequence of S routing stages, each consisting of a Rank & Reroute operator followed by N transformer layers. (Left): The full pipeline. Input visual tokens V = {vi} pass through stages 1...S; at each stage i, the operator selects a top-ri% subset to route through the transformer block, while the remaining deferred tokens bypass it via a residual path. Crucially, deferred tokens are not removed from V. They re-enter the next stage’s ranking and may be reselected. (Right): Within one stage, selected and deferred tokens take separate computational paths and merge back at the stage output, restoring the original sequence ordering for the next stage. The Rank & Reroute operator itself reuses the existing text-to-vision attention from the current decoder layer as the ranking signal. No additional

- router is trained. The top-K visual tokens (K = ⌊ri · |V |⌋) are marked selected; the rest are marked deferred. Reroute therefore preserves the FLOPs and KV-cache budget of conventional pruning (only |ri| tokens enter Attn + FFN), while breaking the irreversibility constraint that pruning imposes on V .

4 Method

- 4.1 Reroute: recoverable stage-wise routing. Reroute keeps the stage-wise ranking interface of pruning but replaces the post-selection action. At

routing layer ℓi, the same scorer and keep-ratio schedule yield Visel and Videf: selected tokens traverse the full decoder path, while deferred tokens bypass the current stage yet remain eligible for later

selection. As summarized in Figure 3, each stage ranks vision tokens via text-to-vision attention, routes Visel through Attn+FFN, and defers the rest until the next routing decision. Let Blockℓ denote the full decoder block at layer ℓ (self-attention, FFN, residuals, and normalization). Reroute applies it only to the active subsequence HAℓ

i

, scatters outputs to their original positions, and leaves deferred tokens untouched:

hℓj+1 =

Blockℓ(HAℓ

i

)j, j ∈ Ai, hℓj, j ∈ Videf.

At the next routing layer, the full candidate set is restored by Cireroute+1 = V, so a token deferred in one stage may be re-selected later if its score rises. Reroute thus converts irreversible deletion into

recoverable deferral.

- 4.2 Pruning as a degenerate routing policy. The two methods differ only in the candidate-set transition:

Ciprune+1 = Visel and Cireroute+1 = V. (2) Sharing the same scoring rule, routing layers, and stage budgets, pruning is the degenerate policy that assigns zero re-entry probability to deferred tokens, whereas Reroute keeps them valid for future stages. Figure 4 illustrates this: prior to an additional reroute decision the two policies can produce identical selections, but later stages may re-admit tokens that pruning has already discarded.

[Figure 16]

- Figure 4: Pruning as a degenerate routing policy. We compare FastV and PDrop with their Reroute variants on Qwen2.5-VL-7B. Blue, gray, and orange denote selected, pruned/deferred, and re-admitted tokens. The red box marks the shared selected-token prefix before pruning and reroute diverge: pruning keeps only the previous candidate set, whereas reroute restores the full visual-token candidate set.

- 4.3 Matched-budget efficiency. Reroute is designed for budget-matched comparison with the pruning baseline it augments. Under a shared schedule {(ℓi,ri)}Si=1, both methods activate the same number of vision tokens per stage.

Token identities may diverge once re-entry is allowed. The active set is Ai = T ∪ Visel. With an optimized implementation, only Ai executes the decoder block at stage i, so Reroute targets the same theoretical TFLOPs and KV-cache class as the corresponding pruning schedule. The sole

distinction is the post-selection action: pruning discards Videf, while Reroute retains these tokens in the sequence state for future routing. Practical latency still depends on gather/scatter kernels and

KV-cache management.

- 4.4 Relation to conditional computation.

Reroute can be viewed as a training-free application of Mixture-of-Depth (MoD)-style conditional computation to VLM visual-token pruning. Whereas standard MoD methods learn a router that decides which tokens execute a block and which bypass it, Reroute reuses the same conditionalcomputation pattern but inherits its routing signal, without training, from the text-to-vision attention scores already used by pruning.

This framing clarifies the design motivation. Visual-token pruning already poses a token-wise compute-allocation problem: selected tokens receive decoder computation while low-ranked ones are excluded. Rank-and-remove resolves the allocation by deletion. Reroute instead treats the excluded tokens as a bypass branch. They skip the stage but remain eligible for later selection. Reroute is therefore not a new general-purpose MoD architecture, but a training-free instantiation of the MoD principle in the MLLM visual-token pruning setting, with irreversible removal replaced by recoverable deferral.

- 5 Experiments

We evaluate Reroute under matched visual-token budgets on grounding, general VQA, cross-backbone transfer, and efficiency.

###### 5.1 Experimental setup

Benchmarks. We evaluate grounding on RefCOCO, RefCOCO+, and RefCOCOg [30, 79] using bounding-box accuracy at IoU 0.5 (Acc@0.5). To assess general multimodal performance under visual-token reduction, we also report results on VQA benchmarks [46, 80, 36, 49, 33], including GQA [27], TextVQA [59], and MME [18], depending on model support and completed runs.

Models. We experiment with LLaVA-1.5-7B [42] and Qwen-family models, including Qwen2.5VL-7B [6] and Qwen3.5-9B-Hybrid [54]. LLaVA-1.5-7B is used for comparison with prior pruning

- Table 1: Visual grounding performance on RefCOCO-series benchmarks using LLaVA-1.5-7B. We compare visual-token reduction methods and their Reroute variants under matched averagetoken budgets. Panel (a) reports the original LLaVA-format results, with * marking Nüwa results reproduced in the official Nüwa codebase; panel (b) reports HuggingFace-format reproduced results under the same LLaVA-1.5-7B backbone. When shown, A/B denote testA/testB splits. Avg. Ratio is normalized by the corresponding full-token baseline within each panel. Best and second-best results within each average-token block are shown in bold and underlined, respectively, and unavailable results are denoted by “–”.

(a) Original LLaVA format results.

###### (b) HuggingFace-format reproduced results.

Method Src. RefCOCO RefCOCO+ RefCOCOg Avg. val test val A B val test Ratio

###### Method Src. RefCOCO RefCOCO+ RefCOCOg Avg. val A B val A B val test Ratio

Average Token 576 Vanilla (HF) CVPR’24 54.0 59.4 48.0 46.5 54.1 39.1 46.9 47.6 100.0% Average Token 192 ↓ 66.7%

Average Token 576 Vanilla* CVPR’24 56.2 58.0 50.0 59.3 38.8 48.9 48.4 100.0% Average Token 192 ↓ 66.7% FEATHER ICCV’25 – 27.7 – 2.5 – – 27.2 36.1%

FastV ECCV’24 25.3 27.8 22.3 21.0 23.6 18.0 21.5 20.5 45.5% FastV+Reroute Ours 32.7 37.7 26.8 26.7 32.3 21.3 28.7 29.1 59.2% PDrop CVPR’25 39.8 45.1 33.4 33.5 39.5 25.9 33.2 32.8 71.3% PDrop+Reroute Ours 45.3 49.5 39.0 37.9 44.3 31.1 39.7 39.2 82.3% Average Token 128 ↓ 77.8%

Nüwa* ICLR’26 59.1 60.1 50.5 60.9 40.2 51.8 50.1 103.7% Nüwa+Reroute* Ours 58.6 59.6 50.4 61.3 39.8 50.9 49.9 103.0% Average Token 128 ↓ 77.8%

FastV ECCV’24 10.1 10.3 8.2 8.5 9.8 9.1 8.9 18.4% SparseVLM ICML’25 6.2 6.3 9.9 5.8 4.2 6.5 6.4 12.6% VisionZip CVPR’25 4.1 4.5 3.9 4.1 4.9 3.5 3.5 8.1%

FastV ECCV’24 11.0 12.7 9.1 8.9 10.3 8.0 9.3 9.0 19.8% FastV+Reroute Ours 17.4 20.7 13.9 14.5 17.5 11.6 15.8 15.1 31.8% PDrop CVPR’25 30.7 36.5 23.7 25.2 30.5 18.3 24.4 23.9 53.4% PDrop+Reroute Ours 34.1 39.6 27.2 27.6 33.2 22.0 29.8 29.7 61.2% Average Token 64 ↓ 88.9% FastV ECCV’24 0.3 0.2 0.6 0.3 0.1 0.5 0.1 0.1 0.6% FastV+Reroute Ours 0.4 0.2 0.4 0.3 0.1 0.7 0.1 0.1 0.6% PDrop CVPR’25 12.8 14.8 10.2 10.1 12.1 8.0 10.3 10.1 22.2% PDrop+Reroute Ours 18.9 23.3 14.5 15.4 19.9 11.5 16.5 15.6 34.0%

Nüwa* ICLR’26 52.3 53.5 43.7 53.2 35.6 45.0 43.5 90.9% Nüwa+Reroute* Ours 51.8 53.0 44.3 55.5 35.0 44.3 43.1 90.3% Average Token 64 ↓ 88.9%

FastV ECCV’24 2.0 2.7 2.4 1.2 1.0 2.0 2.2 3.8% SparseVLM ICML’25 1.0 1.0 1.0 1.0 1.3 0.7 0.6 1.9% VisionZip CVPR’25 3.8 4.0 3.5 3.7 3.9 3.2 3.4 7.2%

Nüwa* ICLR’26 32.8 33.2 26.5 34.3 20.5 25.6 24.5 54.6% Nüwa+Reroute* Ours 32.8 33.7 27.2 35.3 21.1 26.3 25.2 55.8%

Vanilla baseline Other methods Base methods Reroute variants .

methods, while Qwen models test cross-backbone transfer. Checkpoint sources and implementation formats are marked separately in the tables.

Baselines. We compare against the full model, FastV [12], PDrop [71], and their budget-matched Reroute variants [83, 84, 73, 17, 26]. We also consider Nüwa [26] as a strong recent baseline: using its official codebase, we augment its FastV-style decoder pruning with Reroute.

###### 5.2 Main results

Visual grounding on LLaVA-1.5. Table 1 reports RefCOCO-series results on LLaVA-1.5 under two checkpoint formats. Panel (a) follows the original LLaVA-1.5 format, combining paper-reported baselines with rows reproduced in the official Nüwa codebase (∗). Panel (b) uses the HuggingFaceformat checkpoint and reports fully reproduced results from our unified pipeline. Results are grouped by average visual-token budget, and average ratios are normalized by the corresponding full-token baseline.

In the HuggingFace-format setting, Reroute consistently improves matched pruning baselines. At 192 average tokens, it raises the average ratio from 45.5% to 59.2% for FastV and from 71.3% to 82.3% for PDrop. At 128 tokens, the gains remain 12.0 and 7.8 points, respectively. At the extreme 64-token budget, one-shot FastV leaves little room for recovery, whereas PDrop+Reroute still improves from 22.2% to 34.0%, indicating that repeated re-entry opportunities are important for effective rerouting.

In the original-format setting, Nüwa is already a strong grounding-oriented baseline. Reroute matches it at moderate budgets and further improves the average ratio at 64 tokens from 54.6% to 55.8%, showing that recoverable deferral remains beneficial even on top of a stronger pipeline.

Overall, these results support our view of Reroute as a change in the post-selection action rather than a new scoring heuristic: replacing irreversible deletion with recoverable deferral improves FastV/PDrop-style pruning, especially under stage-wise schedules, while remaining compatible with stronger grounding-aware pipelines.

General VQA results on LLaVA-1.5. Table 3 reports general VQA results on the HuggingFaceformat LLaVA-1.5 checkpoint. These benchmarks are used as an auxiliary evaluation: our main hypothesis concerns recoverable visual evidence for grounding under aggressive token reduction, while VQA measures whether the same routing mechanism preserves general multimodal capability.

- Table 2: RefCOCO-series visual grounding with Qwen-family VLMs. FastV, PDrop, and their Reroute variants are compared under matched average-token budgets. Panel (a) uses Qwen2.5-VL-7B with decoder-side retained visual tokens; panel (b) uses Qwen3.5-9B-Hybrid with gated-attention routed visual tokens. A/B denote testA/testB. Avg. Ratio is normalized to the corresponding vanilla baseline. Best and second-best results within each budget block are bolded and underlined.

(a) Qwen2.5-VL-7B.

Method Src. RefCOCO RefCOCO+ RefCOCOg Avg. val A B val A B val test Ratio

Average Token 100% Vanilla Qwen’25 87.7 89.8 82.3 80.5 85.9 72.8 82.7 83.3 100.0% Average Token Reduction ↓ 66.7%

FastV ECCV’24 46.4 48.1 44.4 40.5 42.6 39.3 48.4 47.2 53.7% FastV+Reroute Ours 53.4 57.8 49.4 45.0 50.9 40.6 52.8 52.1 60.3% PDrop CVPR’25 55.9 59.2 52.0 48.4 53.0 43.6 53.6 52.9 62.9% PDrop+Reroute Ours 57.9 63.0 53.6 49.9 55.7 44.8 54.7 54.0 65.1% Average Token Reduction ↓ 77.8%

FastV ECCV’24 19.2 21.5 17.8 16.7 18.4 14.6 19.1 18.6 21.9% FastV+Reroute Ours 26.8 31.1 22.9 21.7 26.0 18.0 25.3 24.1 29.3% PDrop CVPR’25 25.9 30.9 24.2 21.2 26.7 19.2 24.1 23.7 29.3% PDrop+Reroute Ours 33.2 37.6 31.5 27.7 33.5 25.1 31.0 29.7 37.4% Average Token Reduction ↓ 88.9%

FastV ECCV’24 1.6 1.6 2.2 1.7 1.7 2.6 2.0 2.3 2.4% FastV+Reroute Ours 6.2 5.5 5.9 4.8 4.2 4.8 6.7 6.5 6.7%

PDrop CVPR’25 10.2 10.9 9.0 7.9 9.8 6.7 9.9 9.5 11.1% PDrop+Reroute Ours 16.8 19.2 15.2 13.9 16.9 12.2 15.0 14.9 18.6%

(b) Qwen3.5-9B-Hybrid.

Method Src. RefCOCO RefCOCO+ RefCOCOg Avg.

val A B val A B val test Ratio Average Gated Attention Token 100% Vanilla Qwen’25 90.4 90.7 85.5 80.0 84.3 75.7 87.3 86.8 100.0% Average Gated Attention Token Reduction ↓ 66.7%

FastV ECCV’24 28.7 32.5 25.6 25.2 28.3 20.3 26.3 25.6 31.1% FastV+Reroute Ours 70.8 75.1 64.8 60.7 67.0 53.9 68.5 68.0 77.5% PDrop CVPR’25 36.4 40.4 32.4 32.7 36.1 27.4 33.7 33.4 40.0% PDrop+Reroute Ours 70.0 76.1 62.4 61.1 68.6 54.2 69.2 68.3 77.7% Average Gated Attention Token Reduction ↓ 77.8%

FastV ECCV’24 18.0 22.2 15.9 14.5 18.3 12.3 16.6 15.9 19.5% FastV+Reroute Ours 51.7 58.4 44.6 42.3 49.7 35.7 50.2 49.6 55.9% PDrop CVPR’25 23.9 28.6 20.4 20.4 25.0 16.7 20.8 20.4 25.8% PDrop+Reroute Ours 54.9 62.7 47.0 46.4 53.5 38.0 52.9 52.2 59.7% Average Gated Attention Token Reduction ↓ 88.9%

FastV ECCV’24 12.4 14.7 12.3 6.8 6.8 5.8 8.3 8.1 10.9% FastV+Reroute Ours 14.2 15.0 13.2 8.8 9.4 7.6 11.9 11.6 13.4% PDrop CVPR’25 19.5 23.8 15.7 14.2 17.8 10.0 15.8 15.5 19.3% PDrop+Reroute Ours 35.0 41.1 27.8 27.8 34.8 22.2 33.7 33.8 37.4%

Vanilla baseline Base methods Reroute variants .

- Table 3: VQA performance on LLaVA-1.5-7B under the HuggingFace model format. Bold and underline denote best and second-best results within each averagetoken block.

Table 4: Efficiency comparison on LLaVA-1.5-7B. TFLOPs and KV-cache memory are normalized against the vanilla full-token baseline.

Method GQA MMB MMMU MME TextVQA POPE SQA SEED Avg. Ratio Average Token 576

Method Avg. TFLOPs KV Cache

Tok. Val. /Van. MB /Van. Vanilla 576 8.67 – 297.5 – FastV 306 4.89 ↓ 44% 162.5 ↓ 45% FastV+Reroute 306 5.02 ↓ 45% 162.5 ↓ 45% FastV 128 2.45 ↓ 72% 73.2 ↓ 75% FastV+Reroute 128 2.58 ↓ 70% 73.2 ↓ 75% PDrop 128 2.47 ↓ 72% 73.3 ↓ 75% PDrop+Reroute 128 2.56 ↓ 75% 73.2 ↓ 75% FastV 64 1.60 ↓ 82% 41.5 ↓ 86% FastV+Reroute 64 1.73 ↓ 80% 41.5 ↓ 86% PDrop 64 1.83 ↓ 79% 49.7 ↓ 83% PDrop+Reroute 64 1.99 ↓ 77% 51.6 ↓ 83%

Vanilla 60.5 62.4 34.4 1782 57.5 85.9 66.1 65.5 100.0% Average Token 192 ↓ 66.7%

FastV 55.0 58.6 34.2 1637 53.7 79.6 66.4 58.2 93.9% FastV+Reroute 57.4 60.7 34.2 1672 56.7 81.0 67.1 60.4 96.5% PDrop 59.2 61.4 34.7 1776 46.9 84.4 65.9 63.4 96.6% PDrop+Reroute 58.9 61.3 34.4 1775 57.4 84.2 66.1 62.6 98.6%

Average Token 128 ↓ 77.8%

FastV 50.5 53.7 34.0 1451 50.4 73.0 65.5 54.3 88.0% FastV+Reroute 54.8 57.0 34.3 1557 53.6 75.1 68.4 55.4 92.2% PDrop 57.0 59.5 34.2 1728 56.2 82.1 66.8 59.2 96.3% PDrop+Reroute 57.3 59.8 34.6 1733 56.5 80.9 66.6 60.8 96.8%

Average Token 64 ↓ 88.9%

FastV 40.0 22.4 32.1 950 41.8 51.1 65.1 36.0 66.8% FastV+Reroute 44.3 23.3 32.2 954 43.5 51.4 65.3 39.8 69.1% PDrop 52.8 55.1 33.1 1486 51.6 74.9 68.2 53.9 89.7% PDrop+Reroute 54.4 57.4 33.8 1542 52.5 74.8 69.2 54.8 91.7%

Vanilla Base Reroute

Vanilla baseline Base methods Reroute variants

Across the 192-, 128-, and 64-token budgets, Reroute remains comparable to or better than its matched FastV and PDrop counterparts. This indicates that the grounding improvements reported above are not obtained by sacrificing general VQA performance. Instead, recoverable routing preserves broad multimodal behavior while providing larger benefits on grounding-sensitive tasks, where irreversible removal is more likely to discard visual evidence needed by later decoder layers.

Qwen-series visual grounding results. We next evaluate whether Reroute transfers beyond LLaVA-

- 1.5. Tables 2 report RefCOCO-series results on Qwen2.5-VL-7B and Qwen3.5-9B-Hybrid, grouped by average visual-token reduction and gated-attention-token reduction, respectively.

On Qwen2.5-VL, Reroute improves matched pruning baselines across all budgets. At 66.7% reduction, FastV+Reroute improves the average ratio from 53.7% to 60.3%, and PDrop+Reroute improves it from 62.9% to 65.1%. The gains become larger under tighter budgets: PDrop+Reroute improves from 29.3% to 37.4% at 77.8% reduction and from 11.1% to 18.6% at 88.9% reduction.

Qwen3.5-9B-Hybrid requires separate interpretation because it interleaves Gated DeltaNet linearattention blocks with softmax-attention blocks, and routing is applied only to the latter. Since Gated DeltaNet follows a recurrent state-update view [74], preserving sequence layout may be important for the remaining hybrid blocks. Reroute is compatible with this setting because deferred tokens bypass routed softmax-attention computation while remaining in the sequence state.

Under this hybrid setting, Reroute yields larger gains. At 66.7% reduction, FastV+Reroute improves the average ratio from 31.1% to 77.5%, and PDrop+Reroute from 40.0% to 77.7%. For PDrop,

reroute vanilla FastV (reroute layers = {2})

FastV (reroute layers = {2}) reroute (reroute layers = {2, X})

###### GQA

###### RefCOCO testA

RefCOCO testB

###### GQA

###### RefCOCO testA

RefCOCO testB

50

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| |est|: e|xt|re|m|e (59.88| |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| |est|: e|xt|re|m|e (53.51| |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| |est|: N|=|8|(4|1.39)| |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

|bes|t: L=5|(59.5|6)|
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

|be|st: L=|7 (51.|79|
|---|---|---|---|
| | | | |
| | | | |
| | | | |

|be|st: L=|7 (41.|18|
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

60

)

)

59.5

40

50

Accuracy

Accuracy

Accuracy

45

55

Accuracy

Accuracy

Accuracy

- 59
- 60

50

59.0

45

40

35

45

35

40

58.5

30

1 2 3 4 6 8 30 (extr.)

1 2 3 4 6 8 30 (extr.)

1 2 3 4 6 8 30 (extr.)

3 11 19 27 31

3 11 19 27 31

3 11 19 27 31

Rerouted layers N

Rerouted layers N

Rerouted layers N

Layer X

Layer X

Layer X

(a) Frequency sweep.

(b) Location sweep.

- Figure 5: Ablation of reroute scheduling. Under a fixed retention ratio ri = 0.5, we vary (a) the number of rerouted layers N and (b) the second reroute location X with the first reroute layer fixed. Reroute benefits from repeated re-evaluation, but the best schedule depends on where re-entry is allowed.

Reroute further improves the ratio from 25.8% to 59.7% at 77.8% reduction and from 19.3% to 37.4% at 88.9% reduction.

These results reinforce the LLaVA-1.5 finding: Reroute is most beneficial when irreversible pruning prevents later recovery of visual evidence. The Qwen3.5 results further suggest that preserving sequence layout may be important for hybrid architectures, although detailed analysis of linearattention state dynamics is left for future work.

Ablation on reroute scheduling. Figure 5 ablates the reroute schedule at fixed ri = 0.5, varying the number of reroute layers and the placement of an added layer. The single-layer case is exactly FastV; additional reroute layers therefore measure the effect of extending FastV to multi-stage recoverable routing.

The frequency sweep shows that adding reroute stages generally improves over the single-layer FastV reference. On GQA and RefCOCO testA, performance increases with more rerouted layers and peaks under the densest schedule. RefCOCO testB improves up to a moderate frequency, peaking around N = 8, then saturates. This suggests that repeated re-evaluation helps until deferred tokens have sufficient re-entry opportunities; beyond that, additional routing decisions provide limited benefit and may introduce unnecessary perturbation.

The location sweep shows that Reroute is also sensitive to where the next routing stage is inserted. With the first routing layer fixed, placing the second reroute layer in the early-to-middle decoder performs best: GQA peaks around layer 5, while RefCOCO testA and testB peak around layer 7. Later insertion provides less benefit and can approach the FastV reference. These trends support our stage-wise design and show that the routing schedule should not be treated as arbitrary.

Efficiency metrics for Reroute. Table 4 reports LLaVA-1.5 efficiency under different average visual-token budgets. We report TFLOPs and KV-cache memory from runtime profiler as primary metrics, with ratios normalized by the full-token vanilla baseline.

Reroute largely preserves the efficiency profile of visual-token reduction. At the representative 64token budget, FastV+Reroute reduces main TFLOPs/KV cache by 80%/86%, while PDrop+Reroute reduces them by 77%/83%, with other budgets remaining close to their matched pruning baselines. The small TFLOPs overhead mainly comes from additional reroute stages relative to one-shot FastV, or from the budget-matched schedule relative to PDrop. Thus, recoverable routing retains the main compute and cache benefits of pruning; practical wall-clock speedup further depends on optimized gather/scatter and KV-cache implementation.

- 6 Conclusion

We revisited visual-token reduction from the perspective of the post-scoring action. Instead of permanently removing low-ranked vision tokens, Reroute defers them: selected tokens receive full Attn+FFN computation, while deferred tokens bypass the current stage but remain eligible for later re-entry. Conventional pruning becomes the special case where deferred tokens never return, yielding a training-free Mixture-of-Depth-style formulation for VLM visual tokens. Empirically, Reroute improves FastV/PDrop-style pruning on LLaVA-1.5 grounding, preserves general VQA, transfers to Qwen-family backbones, and retains the theoretical TFLOPs and KV-cache benefits of token reduction. Gains are largest under aggressive reduction and grounding-oriented evaluation, where irreversible removal is most likely to discard useful evidence. The takeaway is simple: visual-token reduction should preserve the option to recover deferred evidence.

Limitations. Reroute inherits the limitations of its attention-score router: poor ranking or overly aggressive budgets can leave useful evidence inactive for multiple stages. Efficiency gains are primarily theoretical, since practical latency depends on optimized gather/scatter kernels and KVcache management. Reroute targets decoder-side reduction and is complementary to encoder-side compression, token merging, and KV-cache eviction, whose combinations we do not exhaustively evaluate. Finally, in hybrid architectures such as Qwen3.5, routing is applied only to softmax-attention blocks; interaction with linear-attention state dynamics is left for future work.

###### References

- [1] Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversity-based visual token pruning for large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9392–9401, 2025.
- [2] Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S Nikolopoulos, Hans Vandierendonck, Deepu John, and Bo Ji. Hired: Attention-guided token dropping for efficient inference of high-resolution vision-language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 1773–1781, 2025.
- [3] Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, et al. Mixture-of-recursions: Learning dynamic recursive depths for adaptive token-level computation. arXiv preprint arXiv:2507.10524, 2025.
- [4] Natan Bagrov, Eugene Khvedchenia, Borys Tymchenko, Shay Aharon, Lior Kadoch, Tomer Keren, Ofri Masad, Yonatan Geifman, Ran Zilberstein, Tuomas Rintamaki, Matthieu Le, and Andrew Tao. Efficient video sampling: Pruning temporally redundant tokens for faster vlm inference, 2025. URL https://arxiv.org/abs/2510.14624.
- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization. Text Reading, and Beyond, 2(1):1, 2023.
- [6] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report,

2025. URL https://arxiv.org/abs/2502.13923.

- [7] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022.
- [8] Mu Cai, Jianwei Yang, Jianfeng Gao, and Yong Jae Lee. Matryoshka multimodal models. arXiv preprint arXiv:2405.17430, 2024.
- [9] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Junjie Hu, et al. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling. arXiv preprint arXiv:2406.02069, 2024.
- [10] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Localityenhanced projector for multimodal llm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13817–13827, 2024.
- [11] Jieneng Chen, Luoxin Ye, Ju He, Zhao-Yang Wang, Daniel Khashabi, and Alan Yuille. Efficient large multi-modal models via visual context compression. Advances in neural information processing systems, 37:73986–74007, 2024.
- [12] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer, 2024.

- [13] Yi Chen, Jian Xu, Xu-Yao Zhang, Wen-Zhuo Liu, Yang-Yang Liu, and Cheng-Lin Liu. Recoverable compression: A multimodal vision token recovery mechanism guided by text information. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 2293–2301, 2025.
- [14] Yuhao Chen, Bin Shan, Xin Ye, and Cheng Chen. Evoprune: Early-stage visual token pruning for efficient mllms. arXiv preprint arXiv:2603.03681, 2026.
- [15] Tzu-Chun Chien, Chieh-Kai Lin, Shiang-Feng Tsai, Ruei-Chi Lai, Hung-Jen Chen, and Min Sun. Grounding-aware token pruning: Recovering from drastic performance drops in visual grounding caused by pruning, 2025. URL https://arxiv.org/abs/2506.21873.
- [16] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.
- [17] Mark Endo, Xiaohan Wang, and Serena Yeung-Levy. Feather the throttle: Revisiting visual token pruning for vision-language model acceleration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22826–22835, 2025.
- [18] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.
- [19] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive kv cache compression for llms. arXiv preprint arXiv:2310.01801, 2023.
- [20] Alex Graves. Adaptive computation time for recurrent neural networks. arXiv preprint arXiv:1603.08983, 2016.
- [21] Yefei He, Feng Chen, Jing Liu, Wenqi Shao, Hong Zhou, Kaipeng Zhang, and Bohan Zhuang. Zipvl: Efficient large vision-language models with dynamic token sparsification. arXiv preprint arXiv:2410.08584, 2024.
- [22] Lianyu Hu, Liqing Gao, Fanhua Shang, Liang Wan, and Wei Feng. illava: An image is worth fewer than 1/3 input tokens in large multimodal models. arXiv preprint arXiv:2412.06263, 2024.
- [23] Wenbo Hu, Zi-Yi Dou, Liunian H Li, Amita Kamath, Nanyun Peng, and Kai-Wei Chang. Matryoshka query transformer for large vision-language models. Advances in Neural Information Processing Systems, 37:50168–50188, 2024.
- [24] Gao Huang, Danlu Chen, Tianhong Li, Felix Wu, Laurens Van Der Maaten, and Kilian Q Weinberger. Multi-scale dense networks for resource efficient image classification. arXiv preprint arXiv:1703.09844, 2017.
- [25] Wenxuan Huang, Zijie Zhai, Yunhang Shen, Shaosheng Cao, Fei Zhao, Xiangfeng Xu, Zheyu Ye, Yao Hu, and Shaohui Lin. Dynamic-llava: Efficient multimodal large language models via dynamic vision-language context sparsification. arXiv preprint arXiv:2412.00876, 2024.
- [26] Yihong Huang, Fei Ma, Yihua Shao, Jingcai Guo, Zitong Yu, Laizhong Cui, and Qi Tian. N\" uwa: Mending the spatial integrity torn by vlm token pruning. arXiv preprint arXiv:2602.02951, 2026.
- [27] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.
- [28] Lei Jiang, Weizhe Huang, Tongxuan Liu, Yuting Zeng, Jing Li, Lechao Cheng, and Xiaohua Xu. Fopru: Focal pruning for efficient large vision-language models. arXiv preprint arXiv:2411.14164, 2024.

- [29] Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. See what you are told: Visual attention sink in large multimodal models. arXiv preprint arXiv:2503.03321, 2025.
- [30] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014.
- [31] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.
- [32] Tim Lawson and Laurence Aitchison. Learning to skip the middle layers of transformers. arXiv preprint arXiv:2506.21103, 2025.
- [33] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seedbench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.
- [34] Jiachen Li, Xinyao Wang, Sijie Zhu, Chia-Wen Kuo, Lu Xu, Fan Chen, Jitesh Jain, Humphrey Shi, and Longyin Wen. Cumo: Scaling multimodal llm with co-upcycled mixture-of-experts. Advances in Neural Information Processing Systems, 37:131224–131246, 2024.
- [35] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. International Journal of Computer Vision, 133(10):6794–6812, 2025.
- [36] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292–305, 2023.
- [37] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. Advances in Neural Information Processing Systems, 37:22947–22970, 2024.
- [38] Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. Not all patches are what you need: Expediting vision transformers via token reorganizations. arXiv preprint arXiv:2202.07800, 2022.
- [39] Bin Lin, Zhenyu Tang, Yang Ye, Jinfa Huang, Junwu Zhang, Yatian Pang, Peng Jin, Munan Ning, Jiebo Luo, and Li Yuan. Moe-llava: Mixture of experts for large vision-language models. IEEE Transactions on Multimedia, 2026.
- [40] Zhihang Lin, Mingbao Lin, Luxi Lin, and Rongrong Ji. Boosting multimodal large language models with visual tokens withdrawal for rapid inference. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 5334–5342, 2025.
- [41] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [42] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024.
- [43] Jizhihui Liu, Guangdao Zhu, and Feiyi Du. Hiprune: Training-free visual token pruning via hierarchical attention in vision-language models (student abstract). In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 41275–41277, 2026.
- [44] Ting Liu, Liangtao Shi, Richang Hong, Yue Hu, Quanjun Yin, and Linfeng Zhang. Multi-stage vision token dropping: Towards efficient multimodal large language model. arXiv preprint arXiv:2411.10803, 2024.
- [45] Weijie Liu, Peng Zhou, Zhiruo Wang, Zhe Zhao, Haotang Deng, and Qi Ju. Fastbert: a selfdistilling bert with adaptive inference time. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 6035–6044, 2020.

- [46] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.
- [47] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. Scissorhands: Exploiting the persistence of importance hypothesis for llm kv cache compression at test time. Advances in Neural Information Processing Systems, 36:52342–52364, 2023.
- [48] Zuyan Liu, Benlin Liu, Jiahui Wang, Yuhao Dong, Guangyi Chen, Yongming Rao, Ranjay Krishna, and Jiwen Lu. Efficient inference of vision instruction-following models with elastic cache. In European Conference on Computer Vision, pages 54–69. Springer, 2024.
- [49] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems, 35: 2507–2521, 2022.
- [50] Yaxin Luo, Gen Luo, Jiayi Ji, Yiyi Zhou, Xiaoshuai Sun, Zhiqiang Shen, and Rongrong Ji. γ−mod: Exploring mixture-of-depth adaptation for multimodal large language models. arXiv preprint arXiv:2410.13859, 2024.
- [51] Tanvir Mahmud, Burhaneddin Yaman, Chun-Hao Liu, and Diana Marculescu. Papr: Trainingfree one-step patch pruning with lightweight convnets for faster inference. In European Conference on Computer Vision, pages 110–128. Springer, 2024.
- [52] Lingchen Meng, Jianwei Yang, Rui Tian, Xiyang Dai, Zuxuan Wu, Jianfeng Gao, and Yu-Gang Jiang. Deepstack: Deeply stacking visual tokens is surprisingly simple and effective for lmms. Advances in Neural Information Processing Systems, 37:23464–23487, 2024.
- [53] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.
- [54] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https: //qwen.ai/blog?id=qwen3.5.
- [55] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. Advances in neural information processing systems, 34:13937–13949, 2021.
- [56] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. Mixture-of-depths: Dynamically allocating compute in transformer-based language models. arXiv preprint arXiv:2404.02258, 2024.
- [57] Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Tran, Yi Tay, and Donald Metzler. Confident adaptive language modeling. Advances in Neural Information Processing Systems, 35:17456–17472, 2022.
- [58] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22857–22867, 2025.
- [59] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.
- [60] Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. Massive activations in large language models. arXiv preprint arXiv:2402.17762, 2024.
- [61] Surat Teerapittayanon, Bradley McDanel, and Hsiang-Tsung Kung. Branchynet: Fast inference via early exiting from deep neural networks. In 2016 23rd international conference on pattern recognition (ICPR), pages 2464–2469. IEEE, 2016.

- [62] Zhongwei Wan, Ziang Wu, Che Liu, Jinfa Huang, Zhihong Zhu, Peng Jin, Longyue Wang, and Li Yuan. Look-m: Look-once optimization in kv cache for efficient multimodal long-context inference. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 4065–4078, 2024.
- [63] Haicheng Wang, Zhemeng Yu, Gabriele Spadaro, Chen Ju, Victor Quétu, Shuai Xiao, and Enzo Tartaglione. Folder: Accelerating multi-modal large language models with enhanced performance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23614–23625, 2025.
- [64] Xin Wang, Fisher Yu, Zi-Yi Dou, Trevor Darrell, and Joseph E Gonzalez. Skipnet: Learning dynamic routing in convolutional networks. In Proceedings of the European conference on computer vision (ECCV), pages 409–424, 2018.
- [65] Qiong Wu, Wenhao Lin, Yiyi Zhou, Weihao Ye, Zhanpeng Zen, Xiaoshuai Sun, and Rongrong Ji. Accelerating multimodal large language models via dynamic visual-token exit and the empirical findings. arXiv preprint arXiv:2411.19628, 2024.
- [66] Qiong Wu, Zhaoxi Ke, Yiyi Zhou, Xiaoshuai Sun, and Rongrong Ji. Routing experts: Learning to route dynamic experts in existing multi-modal large language models. In The Thirteenth International Conference on Learning Representations, 2025.
- [67] Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality. arXiv preprint arXiv:2404.15574, 2024.
- [68] Zuxuan Wu, Tushar Nagarajan, Abhishek Kumar, Steven Rennie, Larry S Davis, Kristen Grauman, and Rogerio Feris. Blockdrop: Dynamic inference paths in residual networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8817–8826, 2018.
- [69] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.
- [70] Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin. Deebert: Dynamic early exiting for accelerating bert inference. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 2246–2251, 2020.
- [71] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247, 2024.
- [72] Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, et al. Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19803–19813, 2025.
- [73] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19792–19802, 2025.
- [74] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=r8H7xhYPwz.
- [75] Weihao Ye, Qiong Wu, Wenhao Lin, and Yiyi Zhou. Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 22128–22136, 2025.
- [76] Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, and Yansong Tang. Atp-llava: Adaptive token pruning for large vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24972–24982, 2025.

- [77] Xubing Ye, Yukang Gan, Xiaoke Huang, Yixiao Ge, and Yansong Tang. Voco-llama: Towards vision compression with large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29836–29846, 2025.
- [78] Hao Yin, Guangzong Si, and Zilei Wang. Lifting the veil on visual information flow in mllms: unlocking pathways to faster inference. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9382–9391, 2025.
- [79] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In European conference on computer vision, pages 69–85. Springer, 2016.
- [80] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024.
- [81] Jun Zhang, Desen Meng, Zhengming Zhang, Zhenpeng Huang, Tao Wu, and Limin Wang. p-mod: Building mixture-of-depths mllms via progressive ratio decay. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3705–3715, 2025.
- [82] Qizhe Zhang, Aosong Cheng, Ming Lu, Zhiyong Zhuo, Minqi Wang, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. [cls] attention is all you need for training-free visual token pruning: Make vlm inference faster. arXiv e-prints, pages arXiv–2412, 2024.
- [83] Qizhe Zhang, Aosong Cheng, Ming Lu, Renrui Zhang, Zhiyong Zhuo, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20857–20867, 2025.
- [84] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417, 2024.
- [85] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36:34661–34710, 2023.
- [86] Qiyan Zhao, Xiaofeng Zhang, Yiheng Li, Yun Xing, Xiaosong Yuan, Feilong Tang, Sinan Fan, Xuhang Chen, Da-Han Wang, and Xu-Yao Zhang. Mca-llava: Manhattan causal attention for reducing hallucination in large vision-language models. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 3981–3990, 2025.
- [87] Jiedong Zhuang, Lu Lu, Ming Dai, Rui Hu, Jian Chen, Qiang Liu, and Haoji Hu. St3: Accelerating multimodal large language model by spatial-temporal visual token trimming. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 11049–11057, 2025.
- [88] Zhuofan Zong, Bingqi Ma, Dazhong Shen, Guanglu Song, Hao Shao, Dongzhi Jiang, Hongsheng Li, and Yu Liu. Mova: Adapting mixture of vision experts to multimodal context. Advances in Neural Information Processing Systems, 37:103305–103333, 2024.

###### Appendix Overview

This appendix supplements the main paper with reproducibility details, routing-schedule analyses, and additional results. Appendix A specifies the datasets, splits, model-specific inference settings, routing configurations, and position-index handling used in our experiments. Appendix B analyzes the schedule choices behind the main Reroute configurations, including the early-stage PDrop ablation and the frequency/location sweeps for reroute scheduling. Appendix C reports additional VQA preservation results, discusses Qwen3-VL as an architecture-sensitive grounding case study, and provides qualitative grounding and token-selection visualizations. Together, these materials document how the reported results are reproduced and further analyze when recoverable routing is most beneficial.

###### A Experimental Protocol and Configurations

This section provides the implementation and evaluation details needed to reproduce the main results. We first describe the datasets, evaluation splits, model settings, and grounding post-processing conventions. We then specify the routing configurations, cross-model schedule normalization, and position-index handling used throughout the experiments.

###### A.1 Evaluation datasets and splits

For visual grounding, we evaluate on the RefCOCO-family grounding suite in lmms-eval. The evaluated splits include RefCOCO (refcoco_val, refcoco_testA, refcoco_testB), RefCOCO+ (refcoco+_val, refcoco+_testA, refcoco+_testB), and RefCOCOg (refcocog_val, refcocog_test). These benchmarks evaluate whether a model can localize the object referred to by a natural-language expression. The model prediction is parsed as a bounding box and evaluated using the standard IoU-based grounding protocol implemented by lmms-eval.

To verify that visual-token reduction does not improve grounding at the expense of general multimodal understanding, we also evaluate on a general VQA suite. The core suite includes MME, GQA, POPE, TextVQA, ScienceQA, and MMBench. When available, we additionally report results on AI2D, MMMU-Val, VQAv2-Lite, MMStar, and SEEDBench-Lite. These tasks cover general perception, object recognition, hallucination evaluation, OCR-oriented question answering, scientific reasoning, and broader multimodal reasoning.

###### A.2 Model and inference settings We evaluate Reroute and its pruning baselines on LLaVA-1.5-7B, Qwen2.5-VL-7B, and Qwen3.5-

- 9B. We additionally include Qwen3-VL-8B for architecture-level discussion and supplementary comparisons. Unless otherwise noted, each benchmark uses the default generation configuration provided by lmms-eval. All experiments are conducted on a single NVIDIA RTX 4090 GPU.

For visual grounding, prompt and post-processing conventions follow the corresponding model family. For LLaVA-1.5, we use the default lmms-eval prompt and post-processing pipeline. For Qwen-family models, we follow the official/Nuwa prompt format and the community-suggested grounding post-processing convention. In particular, Qwen2.5-VL bounding boxes are normalized by pixel coordinates, which is already handled by lmms-eval, whereas Qwen3-VL and Qwen3.5 use [0,1000] coordinate normalization.

###### A.3 Routing configurations

We report the routing-layer configurations used in our experiments. LLaVA-1.5 contains 32 decoder layers and uses 576 visual tokens under the standard evaluation setting. Table 5 summarizes the LLaVA-1.5 configurations used in the main Avg. = 192 comparison.

PDrop introduces a configuration ambiguity because different routing-layer schedules can target the same average-token budget. We consider two conventions: a standard PDrop-style schedule that distributes routing decisions approximately evenly across decoder depth, e.g., {8,16,24} for LLaVA-1.5, and an early-routing convention that starts near layer 2 and continues at later stages, e.g., {2,7,15,23}. We denote these configurations as PDrop (L8) and PDrop (L2), respectively.

Table 5: LLaVA-1.5 routing configurations at Avg. = 192. All schedules use keep-index positional handling.

Backbone Setting Layers Routing layers

LLaVA-1.5-7B PDrop (L8) 32 {8, 16, 24} LLaVA-1.5-7B PDrop (L2) 32 {2, 7, 15, 23} LLaVA-1.5-7B PDrop (L2) + Reroute 32 {2, 7, 15, 23} LLaVA-1.5-7B FastV (L3) 32 {3} LLaVA-1.5-7B FastV (L3) + Reroute 32 {3, 7, 15, 23}

Table 6: Cross-model routing-layer conventions used for Reroute.

Backbone Setting Layers Routing layers Rationale

LLaVA-1.5-7B PDrop + Reroute 32 {2, 7, 15, 23} Early PDrop-aligned schedule from the LLaVA ablation LLaVA-1.5-7B FastV + Reroute 32 {3, 7, 15, 23} Main-table FastV setting uses K = 3 Qwen2.5-VL-7B PDrop + Reroute 28 {2, 6, 13, 20} Layer-normalized early PDrop-aligned schedule

- Qwen2.5-VL-7B FastV + Reroute 28 {3, 6, 13, 20} Layer-normalized FastV-aligned schedule Qwen3-VL-8B Both 36 {3, 8, 17, 26} Starts after DeepStack injection layers
- Qwen3.5-9B Both 32 {3, 7, 15, 23} Snapped to compatible full-attention layers in the hybrid-attention backbone

Appendix B.1 shows that PDrop (L2) gives stronger grounding results, so we use the early-routing convention as the main PDrop-aligned schedule for Reroute.

For cross-model experiments, we preserve the same scheduling principle while normalizing routing layers to model depth and respecting architecture-specific constraints. Table 6 summarizes the cross-model routing-layer conventions used for Reroute.

###### A.4 Position-index handling for physical pruning

When visual tokens are physically pruned, the remaining tokens can either be re-indexed into a contiguous position sequence or keep their original visual position indices [15, 26, 4]. Table 7 compares these two choices under FastV on LLaVA-1.5-7B. Re-indexing substantially degrades both general VQA and RefCOCO grounding, indicating that spatial position preservation is critical for localization after token removal.

Therefore, for all physical-pruning baselines, we use the keep-index convention: kept visual tokens preserve their original position indices after pruning. This makes the pruning baselines stronger and avoids conflating pruning failures with position-indexing artifacts. Reroute does not physically remove deferred tokens, so it preserves the original sequence layout by construction.

###### B Routing Schedule Analysis

This section analyzes the schedule-design choices behind the main Reroute configurations. Since Reroute is training-free and reuses the same attention-score ranking interface as pruning baselines, its main design degrees of freedom are the routing layers and per-stage retention ratios. We therefore study three related questions: whether PDrop should start early or follow its standard depth-spaced schedule, how many reroute stages are useful, and where these stages should be placed across decoder depth.

###### B.1 Early-stage routing for PDrop

For the full RefCOCO-family evaluation, we report ACC@0.5 as the primary grounding metric in Table 8. Across both evaluated backbones, PDrop (L2) consistently improves over PDrop (L8) at the same Avg. = 192 budget. This result supports using early-stage routing as the main PDrop-aligned schedule when comparing pruning against Reroute.

###### B.2 Frequency and location sweeps for Reroute

We provide additional details for the routing-schedule ablations used in the main paper. Table 9 lists the exact schedules used for the frequency sweep in Figure 5a, where the first routing layer is fixed

- Table 7: Re-indexing vs. preserved indexing under FastV on LLaVA-1.5-7B. Grounding results are reported using ACC@0.5.

Schedule Avg. PE handling GQA MMBench RefCOCO A RefCOCO B

- FastV (K=2) 306 Re-index 5.3 16.6 3.9 3.3

- FastV (K=2) 306 Keep-index 58.9 61.9 45.0 37.0
- FastV (K=3) 192 Re-index 40.8 34.8 1.9 1.1

- FastV (K=3) 192 Keep-index 55.0 58.6 27.8 22.3

- Table 8: Schedule ablation on RefCOCO-family visual grounding at Avg. = 192. All values report ACC@0.5. We compare the standard PDrop schedule, denoted PDrop (L8), with the early-routing schedule, denoted PDrop (L2), on the two backbones used for the main grounding analysis.

###### Model Schedule Avg. RefCOCO RefCOCO+ RefCOCOg

val A B val A B val test LLaVA-1.5-7B PDrop (L8) 192 26.2 31.9 19.9 22.4 27.1 16.5 21.8 20.8 LLaVA-1.5-7B PDrop (L2) 192 39.8 45.1 33.4 33.5 39.5 25.9 33.2 32.8 Qwen2.5-VL-7B PDrop (L8) 192 33.4 39.3 28.5 28.9 34.6 23.3 32.7 33.1 Qwen2.5-VL-7B PDrop (L2) 192 55.9 59.2 52.0 48.4 53.0 43.6 53.6 52.9

and the number of reroute stages is varied. Table 10 reports a complementary location sweep, where the number of reroute stages is fixed but their decoder-layer positions are changed. In both sweeps, each routing stage uses the same per-stage retention ratio, ri = 0.5. The realized average token ratio is reported explicitly because it can vary with the placement of routing layers.

The frequency sweep shows that increasing the number of reroute decisions generally improves both GQA and RefCOCO grounding while keeping the average token ratio fixed. The location sweep further shows that the placement of reroute stages matters: later routing stages retain a larger realized average token ratio and perform strongly on RefCOCO, while early and clustered-early schedules offer a stricter-budget comparison. Together, these ablations indicate that Reroute benefits from repeated re-evaluation, but its schedule should be chosen with both routing frequency and decoder-depth placement in mind.

###### C Supplementary Results and Visualizations

This section provides additional evidence beyond the main grounding tables. We first report general VQA results to verify that Reroute does not sacrifice broad multimodal capability. We then discuss Qwen3-VL as an architecture-sensitive case study, followed by qualitative grounding and tokenselection visualizations.

###### C.1 General VQA preservation

We report general VQA benchmark results in Tables 11, 3, 12, 13, and 14. These results serve as a preservation test rather than the primary source of improvement. Since many VQA benchmarks can be answered from coarse or globally salient visual evidence, the key question is whether Reroute maintains broad multimodal capability under the same average-token budget.

Across LLaVA-1.5-7B and Qwen-family backbones, Reroute is generally comparable to, and often improves over, its corresponding pruning baseline. The trend is clearest for PDrop + Reroute, where multi-stage routing gives deferred tokens later opportunities to re-enter computation; for example, on Qwen2.5-VL-7B, the average-ratio score improves over PDrop at all three reduction levels. FastV + Reroute is less stable under extreme reduction because FastV depends on a single early scoring layer, but remains broadly comparable under moderate budgets. Overall, these results indicate that Reroute’s grounding gains do not come from sacrificing general VQA performance. Instead, recoverable routing provides a robust alternative to irreversible pruning, especially when paired with multi-stage schedules.

###### Table 9: Frequency-sweep configurations for reroute scheduling on LLaVA-1.5-7B. All settings

fix the first routing layer at layer 2 and use per-stage retention ratio ri = 0.5. The sweep varies only the number of reroute stages. RefCOCO A/B denote testA/testB Acc@0.5.

Setting #Stages Routing layers Avg. ratio GQA RefCOCO A RefCOCO B

- N1 1 {2} 0.5312 58.94 44.95 36.90
- N2 2 {2, 16} 0.5312 59.03 44.56 36.31
- N3 3 {2, 11, 20} 0.5312 59.24 45.48 37.49
- N4 4 {2, 9, 16, 23} 0.5312 59.56 48.49 37.45 N6 6 {2, 7, 12, 17, 22, 27} 0.5312 59.72 50.93 39.33 N8 8 {2, 6, 10, 14, 18, 22, 26, 30} 0.5312 59.79 52.22 41.39 Dense 30 {2, 3, . . . , 31} 0.5312 59.88 53.51 41.18

###### Table 10: Location-sweep configurations for reroute scheduling on LLaVA-1.5-7B. All settings

use four reroute stages with per-stage retention ratio ri = 0.5, and vary where these stages are placed across decoder depth. RefCOCO A/B denote testA/testB Acc@0.5.

Placement Routing layers Avg. ratio GQA RefCOCO A RefCOCO B

Early {2, 5, 9, 13} 0.531 60.05 53.07 41.28 Uniform {4, 12, 20, 28} 0.563 59.31 46.88 37.74 Late {9, 15, 21, 27} 0.641 60.61 53.81 40.29 Very late {15, 19, 23, 27} 0.734 60.53 59.41 47.93 Clustered early {2, 4, 7, 11} 0.531 59.89 53.08 41.06 Clustered late {20, 23, 26, 29} 0.813 60.52 59.02 47.91

###### C.2 Qwen3-VL grounding case study

Table 15 reports the Qwen3-VL RefCOCO results. We interpret this experiment as an architecturesensitive diagnostic rather than as a primary success case. Reroute is training-free and does not learn a router; it reuses the same attention-score ranking signal as the corresponding pruning baseline. Therefore, the relevant design variable is the hand-specified routing schedule, including routed layers and keep-ratio budgets, rather than a trained routing policy.

In the moderate-reduction regimes, the early PDrop schedule remains the strongest grounding configuration: at 66.7% token reduction, PDrop (L2) reaches a 69.5% average ratio, while PDrop + Reroute reaches 57.2%; at 77.8% reduction, PDrop reaches 38.9%, while PDrop + Reroute reaches 36.1%. Under the most aggressive 88.9% reduction, however, PDrop + Reroute improves over PDrop from 12.9% to 19.1%, suggesting that recoverable deferral becomes useful when irreversible pruning is most likely to remove essential grounding evidence.

We therefore use Qwen3-VL as a case study showing that Reroute can help under extreme compression, but that training-free schedules should still be chosen with model-specific architectural constraints in mind, especially for backbones with additional visual-feature injection mechanisms.

###### C.3 Additional qualitative results

We provide additional qualitative visualizations for RefCOCO-family grounding in Figures 8, 9, and

- 10. These examples cover LLaVA-1.5, Qwen2.5-VL, and Qwen3.5, and compare pruning baselines with their corresponding Reroute variants under matched token budgets. We further visualize selected visual-token masks for LLaVA-1.5 and Qwen2.5-VL in Figures 6 and 7, showing which image regions remain active under the routing policy.

- Table 11: VQA performance comparison on LLaVA-1.5-7B under the original LLaVA / Nüwa official-codebase format. Prior-method rows are paper-reported results, while rows marked with

* are reproduced in the official Nüwa codebase. Only tasks available in the updated result sheet are reported. Best and second-best results within each average-token block are shown in bold and underlined, respectively.

Method Source GQA MMB MMMU MME TextVQA POPE SQA SEED Avg. Ratio Average Token 576

Vanilla CVPR’24 61.9 64.7 36.3 1862 58.2 85.9 69.5 58.6 100.0% Average Token 192 ↓ 66.7%

FastV ECCV’24 52.7 61.2 34.3 1612 52.5 64.8 67.3 57.1 90.1% PDrop CVPR’25 57.1 63.2 34.1 1766 56.1 82.3 70.2 54.7 95.7% SparseVLM ICML’25 57.6 62.5 33.8 1721 56.1 83.6 69.1 55.8 95.4% VisionZip CVPR’25 59.3 63.0 36.6 1782 57.3 85.3 68.9 56.4 97.9%

Nüwa ICLR’26 60.9 64.3 35.5 1834 57.4 86.4 68.2 59.7 99.2% Nüwa+Reroute* Ours 61.1 63.6 35.7 1848 57.8 86.0 68.6 66.1 100.6%

Average Token 128 ↓ 77.8%

FastV ECCV’24 49.6 56.1 34.9 1490 50.6 59.6 60.2 55.9 85.2% PDrop CVPR’25 56.0 61.1 34.2 1664 55.1 82.3 69.9 53.3 93.8% SparseVLM ICML’25 56.0 60.0 33.8 1696 54.9 80.5 67.1 53.4 92.9% VisionZip CVPR’25 57.6 62.0 37.9 1761 56.8 83.2 68.9 54.9 96.9% PruMerge ICCV’25 57.8 59.6 36.2 1712 54.3 81.5 67.6 – 94.7%

Nüwa ICLR’26 60.2 63.4 35.8 1828 57.0 85.5 67.8 58.7 98.4% Nüwa+Reroute* Ours 60.4 63.2 36.1 1827 57.3 85.3 68.2 64.4 99.8%

Average Token 64 ↓ 88.9%

FastV ECCV’24 46.1 48.0 34.0 1256 47.8 59.6 51.1 51.9 77.9% PDrop CVPR’25 41.9 33.3 26.5 1092 45.9 55.9 69.2 40.0 70.3% SparseVLM ICML’25 53.8 60.1 35.4 1589 53.4 77.5 69.8 51.1 91.5% VisionZip CVPR’25 55.1 60.1 36.2 1690 55.5 77.0 69.0 52.2 93.2% PruMerge ICCV’25 55.4 59.6 35.8 1616 52.0 75.7 69.5 – 92.1%

Nüwa ICLR’26 58.3 62.0 36.4 1706 54.9 83.0 67.5 56.4 95.8% Nüwa+Reroute* Ours 58.5 61.9 35.9 1688 55.1 82.6 67.7 61.6 96.7%

Vanilla baseline ; Other methods ; Base methods ; Reroute variants .

*: reproduced in the official Nüwa codebase with Reroute applied. Unavailable results are denoted by “–”. MMB denotes MMBench; SQA denotes ScienceQA; SEED denotes SeedBench. Avg. Ratio is reported relative to the corresponding full-token baseline.

- Table 12: VQA performance comparison on Qwen2.5-VL-7B. Best and second-best results within each average-token block are shown in bold and underlined, respectively.

Method Source GQA POPE MMB SQA TextVQA AI2D MMMU MMStar SEED VQAv2-lite MME Avg. Ratio Average Token (100%)

Vanilla Qwen’25 60.8 87.8 83.7 87.8 81.0 82.6 48.7 62.3 79.8 78.6 2318 100.0% Average Token Reduction ↓ 66.7%

FastV ECCV’24 54.0 84.6 77.8 82.8 80.1 72.5 49.6 50.3 66.9 72.9 2178 92.0% PDrop CVPR’25 55.7 84.7 78.6 85.0 81.3 75.0 48.4 52.6 68.7 75.8 2213 93.8%

FastV+Reroute Ours 55.9 85.2 77.3 82.5 77.9 73.2 47.3 50.6 68.1 76.3 2184 92.3% PDrop+Reroute Ours 57.0 85.8 79.0 84.8 82.2 74.8 46.7 53.5 69.5 77.6 2243 94.4%

Average Token Reduction ↓ 77.8%

- FastV ECCV’24 48.2 79.5 66.5 77.1 74.2 67.5 44.9 41.0 61.0 64.2 1994 83.0% PDrop CVPR’25 50.7 80.5 72.8 82.0 77.5 71.5 47.0 46.8 62.0 67.6 2056 87.4%

FastV+Reroute Ours 52.1 80.8 68.1 78.8 72.1 68.1 45.0 42.4 59.4 72.4 2002 84.9% PDrop+Reroute Ours 53.8 83.1 75.4 81.2 80.3 71.1 45.8 47.5 62.6 74.1 2078 89.4%

Average Token Reduction ↓ 88.9%

FastV ECCV’24 36.3 49.3 32.0 73.2 13.6 64.7 41.2 30.6 43.0 37.7 1127 56.1% PDrop CVPR’25 44.9 69.3 67.4 79.4 53.5 68.0 45.4 39.7 53.7 58.2 1755 76.9%

FastV+Reroute Ours 42.2 58.3 36.3 74.1 36.7 64.3 41.7 30.8 45.3 52.3 1247 63.5% PDrop+Reroute Ours 49.3 77.0 66.0 77.2 65.0 67.8 44.4 41.0 57.2 68.6 1874 81.3%

Vanilla baseline ; Base methods ; Reroute variants . Avg. Ratio is the mean score ratio against the vanilla baseline, computed over subsets where both scores are available. Schedule details are reported in the appendix.

- Table 13: VQA performance comparison on Qwen3-VL-8B. Best and second-best results within each average-token block are shown in bold and underlined, respectively.

Method Source GQA POPE MMB SQA TextVQA AI2D MMMU MMStar SEED VQAv2-lite MME Avg. Ratio Average Token (100%)

Vanilla Qwen’25 61.7 88.9 84.4 94.5 81.6 83.8 52.2 61.8 80.4 80.0 2374 100.0% Average Token Reduction ↓ 66.7%

FastV ECCV’24 54.1 85.4 76.6 87.3 76.7 74.8 49.2 49.1 72.7 70.6 2012 89.8% PDrop CVPR’25 56.9 87.6 75.4 87.0 71.0 74.4 50.3 50.1 77.6 74.2 2008 90.8%

FastV+Reroute Ours 54.7 85.4 71.3 81.2 73.0 71.0 48.3 43.7 71.7 71.8 1902 86.5% PDrop+Reroute Ours 57.5 87.1 75.8 86.9 77.8 74.6 49.6 48.4 76.8 75.1 2039 91.4%

Average Token Reduction ↓ 77.8%

FastV ECCV’24 49.4 80.7 68.0 83.7 68.5 69.5 48.8 43.0 69.3 63.4 1756 82.7% PDrop CVPR’25 51.7 83.5 67.9 80.8 57.6 68.9 47.9 43.6 70.7 62.5 1812 81.9%

FastV+Reroute Ours 50.4 79.1 62.7 77.3 64.2 67.2 48.0 40.0 64.0 62.1 1659 79.0% PDrop+Reroute Ours 53.6 84.2 70.3 81.6 71.7 70.2 49.1 42.5 71.9 69.3 1839 85.4%

Average Token Reduction ↓ 88.9%

FastV ECCV’24 37.3 56.9 29.6 73.8 16.0 64.9 44.9 30.7 48.9 40.8 1195 57.5% PDrop CVPR’25 45.3 73.8 52.6 75.5 33.5 65.3 46.8 35.4 58.4 51.6 1525 69.6%

FastV+Reroute Ours 39.8 57.9 29.0 73.3 20.0 64.5 44.1 30.1 47.9 42.7 1177 58.1% PDrop+Reroute Ours 49.3 77.5 60.0 77.1 60.6 66.8 46.9 38.6 63.8 60.6 1661 77.4%

Vanilla baseline ; Base methods ; Reroute variants . Avg. Ratio is the mean score ratio against the vanilla baseline, computed over subsets where both scores are available. Schedule details are reported in the appendix.

- Table 14: VQA performance comparison on Qwen3.5-9B-Hybrid. Best and second-best results within each average-token block are shown in bold and underlined, respectively.

Method Source GQA POPE MMB SQA TextVQA AI2D MMMU MMStar SEED VQAv2-lite MME Avg. Ratio Average Gated Attention Token (100%)

Vanilla Qwen’25 61.7 89.6 84.0 93.5 82.8 85.2 48.1 46.8 77.0 79.6 2378 100.0% Average Gated Attention Token Reduction ↓ 66.7%

FastV ECCV’24 52.8 81.4 69.9 82.8 48.0 72.0 45.3 35.4 70.1 62.8 1823 82.5% PDrop CVPR’25 56.6 85.5 74.2 86.7 65.0 74.6 44.3 37.4 72.1 68.6 1924 87.9%

FastV+Reroute Ours 58.4 84.7 76.9 87.2 77.4 75.3 44.3 38.9 71.1 77.2 1929 91.1% PDrop+Reroute Ours 59.6 87.2 80.0 89.9 79.6 77.3 48.1 40.0 72.7 75.4 2042 93.9%

Average Gated Attention Token Reduction ↓ 77.8%

FastV ECCV’24 47.5 74.7 58.9 78.3 28.0 69.2 43.2 31.2 61.8 53.1 1547 72.5% PDrop CVPR’25 51.7 80.1 65.7 79.3 46.2 71.4 44.3 33.2 64.9 58.6 1688 78.9%

FastV+Reroute Ours 56.4 82.5 72.6 82.0 71.7 72.2 45.0 35.1 65.1 72.3 1798 86.2%

- PDrop+Reroute Ours 57.3 83.9 75.6 84.9 74.6 74.3 44.6 37.7 71.7 74.4 1932 89.6% Average Gated Attention Token Reduction ↓ 88.9%

FastV ECCV’24 38.4 55.7 33.2 73.1 9.8 66.8 43.3 20.1 46.9 38.9 1263 57.1% PDrop CVPR’25 44.8 69.0 51.7 75.3 24.3 68.9 42.3 27.7 56.6 47.3 1431 67.5%

FastV+Reroute Ours 51.0 68.7 64.0 78.5 56.0 67.3 42.8 29.5 56.8 64.5 1539 76.1% PDrop+Reroute Ours 54.9 79.7 70.8 80.8 68.3 71.9 45.1 34.7 64.9 70.8 1711 84.4%

Vanilla baseline ; Base methods ; Reroute variants . Avg. Ratio is the mean score ratio against the vanilla baseline, computed over subsets where both scores are available. Schedule details are reported in the appendix.

[Figure 17]

GT FastV PDrop FastV + Reroute PDrop + Reroute

- Figure 6: Selected visual-token masks on LLaVA-1.5. The highlighted regions indicate the visual tokens selected by the routing policy under the corresponding token budget.

- Table 15: Performance comparison on RefCOCO-series visual grounding benchmarks using Qwen3VL-8B. Best and second-best results within each average-token block are shown in bold and underlined, respectively. Method Source RefCOCO RefCOCO+ RefCOCOg Avg.

###### val A B val A B val test Ratio

Average Token (100%) Vanilla Qwen’25 89.3 90.4 84.7 83.3 85.0 76.3 87.2 87.2 100.0% Average Token Reduction ↓ 66.7%

- FastV ECCV’24 49.8 55.1 54.6 42.3 47.6 46.6 53.5 53.6 59.1% PDrop CVPR’25 63.4 64.8 60.6 55.5 57.3 51.2 61.6 61.0 69.5% FastV+Reroute Ours 46.8 50.8 43.0 39.2 43.8 35.5 44.0 43.5 50.6%

- PDrop+Reroute Ours 58.1 53.8 46.8 50.5 47.2 39.0 48.7 48.0 57.2% Average Token Reduction ↓ 77.8%

FastV ECCV’24 27.4 33.9 31.5 22.6 29.3 28.1 31.5 31.9 34.7% PDrop CVPR’25 37.1 40.5 32.0 31.6 34.1 27.4 32.1 31.5 38.9%

FastV+Reroute Ours 27.0 30.4 24.9 22.1 24.8 20.2 24.0 23.6 28.8% PDrop+Reroute Ours 36.3 36.2 29.6 30.3 30.7 24.2 29.8 29.2 36.1%

Average Token Reduction ↓ 88.9%

FastV ECCV’24 6.3 7.6 4.2 4.1 4.1 3.5 5.1 5.3 5.9% PDrop CVPR’25 11.9 15.0 8.9 10.6 12.3 8.1 10.7 10.5 12.9%

FastV+Reroute Ours 6.6 8.2 4.5 3.9 4.4 3.9 5.2 5.5 6.1% PDrop+Reroute Ours 17.8 21.5 15.8 14.1 17.7 12.9 15.9 15.1 19.1%

Vanilla baseline ; Base methods ; Reroute variants . For RefCOCO and RefCOCO+, A/B denote testA/testB. Avg. Ratio is the mean score ratio against the vanilla baseline, computed over subsets where both scores are available. Schedule details are reported in the appendix.

[Figure 18]

GT FastV PDrop FastV + Reroute PDrop + Reroute

- Figure 7: Selected visual-token masks on Qwen2.5-VL. The highlighted regions indicate the visual tokens selected by the routing policy under the corresponding token budget.

[Figure 19]

##### GT PDrop PDrop + Reroute

- Figure 8: Additional RefCOCO-family grounding examples on LLaVA-1.5. We compare pruning baselines with their corresponding Reroute variants under matched token budgets.

[Figure 20]

### GT PDrop PDrop + Reroute

- Figure 9: Additional RefCOCO-family grounding examples on Qwen2.5-VL. Reroute preserves deferred visual evidence and improves grounding over matched pruning baselines.

[Figure 21]

#### GT PDrop PDrop + Reroute

- Figure 10: Additional RefCOCO-family grounding examples on Qwen3.5. We compare pruning baselines and their Reroute variants under matched gated-attention token budgets.

