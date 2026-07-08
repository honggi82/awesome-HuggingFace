## arXiv:2606.11722v1[cs.LG]10Jun2026

# ICA Lens: Interpreting Language Models Without Training Another Dictionary

Sida Liu1† Feijiang Han2† 1Independent Researcher 2University of Maryland

Finding interpretable directions in language-model representations is a critical step toward understanding model behavior and enabling more precise control over model internals. Sparse autoencoders (SAEs) have become the standard tool for this purpose, but using them as the default first lens often requires training, tuning, storing, and evaluating large overcomplete dictionaries for every layer, activation site, and sparsity setting. This computational bottleneck limits rapid exploration and motivates a fundamental question: how much interpretable structure is already visible from activation geometry before training another neural dictionary? Our intuition is simple: many interpretable directions are selective on tokens, and selective directions should look less Gaussian than typical random directions. We therefore revisit independent component analysis (ICA), a classical method for finding non-Gaussian directions, as a compact lens for language-model interpretability. We find that ICA has been underestimated for LLM interpretability because prior uses often relied on off-the-shelf ICA implementations that are brittle on LLM activations and lacked systematic tools for inspecting, annotating, and evaluating the recovered directions. To bridge these gaps, we introduce ICALens, the first practical workflow for stable, efficient, and auditable ICA analysis of LLM representations. ICALens combines an optimized GPU-parallel FastICA pipeline with LLM-specific stability recipes and better fitting diagnostics, enabling efficient and reliable layer-wise analysis across modern LLMs. Across GPT-2 Small, Gemma 2 2B, and Qwen 3.5 2B Base, ICALens efficiently recovers compact, humaninterpretable directions without per-layer gradient-based dictionary training. On SAEBench, ICA is competitive with public SAEs in sparse probing and outperforms them in targeted probe perturbation under small-tomedium component budgets. These results suggest that ICA should not be viewed as a weak classical baseline, but as an efficient and complementary first lens for exploring language-model representations. To support reproducible analysis, we release all fitted ICA checkpoints, the ICA explorer, and human annotations.

Keywords: Mechanistic Interpretability, Independent Component Analysis, Sparse Autoencoders, Language Model Representations

Project Page: ICA Project Website Online Demo: ICA Explorer Checkpoints: Hugging Face Collection Source Code: GitHub Repository

†Corresponding author(s): Sida Liu me@liusida.com, Feijiang Han feijhan@umd.edu

### Contents

- 1 Introduction 4
- 2 Related Work 6
- 3 ICALens: Making ICA Practical for LLM Activations 6

- 3.1 Activation Corpus . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.2 Row-Normalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.3 Robust Convergence Acceptance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 3.4 Adaptive Refit for Difficult Layers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4 What Does ICA Recover? Non-Gaussianity and Context Dependence 10

- 4.1 Non-Gaussianity Separates ICA Directions from Random Projections . . . . . . . . . . . . . . 10
- 4.2 From Non-Gaussianity to Context Dependence . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 5 Human Inspection of ICA Components 14

- 5.1 An Interactive Lens for Component Inspection . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 5.2 Annotation Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 5.3 Random Component Audit . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 5.4 Case Studies: Reading Component Mixtures . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- 5.4.1 Contextual Decomposition of a Polysemous Word . . . . . . . . . . . . . . . . . . . . 18
- 5.4.2 Component Traces Across Sentences . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 5.4.3 Embedding Components Recover Lexical Structure . . . . . . . . . . . . . . . . . . . 21

- 6 Feature Utility under Sparse Probing and Intervention 23

- 6.1 Sparse Probing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- 6.2 Targeted Probe Perturbation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- 7 How ICA Relates to SAEs 26

- 7.1 Directional Overlap . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- 7.2 Activation Pattern Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- 8 Conclusion 29

- 9 Limitations and Future Directions 29

- A Additional FastICA Fitting Diagnostics 34
- B Additional Human Interpretation Results 35
- C Complete Random Annotation Tables 39
- D Secondary Contrastive Label Audit 42
- E Token-Wise Activation Patterns in Gemma and Qwen 45
- F Explorer Interface Screenshots 46

### 1. Introduction

Understanding the internal representations of large language models (LLMs) is a central goal of mechanistic interpretability (Olah et al., 2020, Elhage et al., 2021, Gurnee et al., 2023, Park et al., 2023). Sparse autoencoders have become the dominant approach to this problem by decomposing activations into sparse latents over an overcomplete dictionary of learned directions (Huben et al., 2024, Bricken et al., 2023, Templeton et al., 2024, Gao et al., 2025). SAEs are powerful when a fine-grained feature dictionary is needed, but using them as the default first step comes with substantial practical costs. Training these dictionaries requires substantial data, storage, tuning, and compute. Gemma Scope, for example, required hundreds of SAEs, tens of millions of learned latents, 4–16B training tokens per SAE, roughly 20 PiB of saved activations, and over 20% of GPT-3’s training compute (Lieberum et al., 2024). This cost makes it difficult to train new dictionaries for every model, layer, activation site, and sparsity setting one wants to inspect. Publicly released SAEs partly mitigate the burden, but their coverage and release cadence remain limited compared with the pace and diversity of open model releases.

For many early-stage analyses, however, we may not need a full overcomplete dictionary. A compact set of interpretable directions can already help identify promising layers, inspect concepts, and guide lightweight interventions. This motivates a lighter first lens that can expose useful activation structure without training another large dictionary.

How much interpretable structure is already visible from activation geometry? To answer this question, we go back to a basic intuition about interpretability: many useful interpretable directions are selective. They should not behave like typical random projections over all tokens. Instead, they may activate on a small set of contexts, prefer one sign, or respond to lexical, syntactic, semantic, or discourse patterns. Such selectivity should leave a statistical footprint in activation space.

We study non-Gaussianity as a simple and measurable footprint of this kind. This is motivated by our observation that public SAE decoder directions are substantially more non-Gaussian than random directions, despite being optimized for sparse reconstruction rather than explicit non-Gaussianity. This suggests a simple possibility: by directly searching for highly non-Gaussian directions, we may recover part of the structure that SAEs learn through expensive dictionary training.

Independent component analysis (ICA) gives an intuitive way to test this possibility (Hyvarinen, 1999, Hyvarinen and Oja, 2000). After centering and whitening, ICA finds a linear basis whose one-dimensional projections are as non-Gaussian as possible, returning a small set of statistically exceptional directions. In this sense, ICA directly uses a signal that sparse reconstruction appears to learn only implicitly.

Despite its conceptual elegance, ICA is rarely used as a first lens for layer-wise LLM activation analysis. We argue that ICA has been underestimated for two practical reasons. First, standard ICA does not work reliably out of the box on modern LLM activations. Residual streams are high-dimensional, contain largenorm token outliers, and include a small number of slow or oscillating components that can dominate convergence. As a result, a naive FastICA implementation1 can be slow, fail to converge on many layers, and return few or no usable components. Second, prior evaluations have not systematically tested the interpretability of ICA directions on LLM activations through human annotation or downstream measures of feature utility (Karvonen et al., 2025, Makelov et al., 2025). These implementation and evaluation gaps leave the practical value of ICA unclear.

1For example, prior work used an unoptimized CPU FastICA pipeline from scikit-learn as its ICA baseline (Huben et al., 2024). See https://github.com/HoagyC/sparse_coding/blob/main/autoencoders/ica.py.

We revisit ICA for LLM interpretability and introduce ICALens, a practical workflow for stable, efficient, and auditable ICA analysis of LLM representations.

Making ICA practical for LLM activations (Section 3). We use three recipes to make fitting faster and more stable. Row normalization reduces the influence of activation-norm outliers before whitening. A p95-LIM rule accepts a fit when most components have stabilized but a small tail remains difficult to converge. Adaptive refitting lowers the target component count for difficult layers. On GPT-2 Small with 1M activations, our pipeline increases the number of accepted layers by 400.0% and reduces the total number of FastICA iterations by 21.5%.

Characterizing what ICA recovers (Section 4). ICA does not return an arbitrary rotation of the residual stream. Across GPT-2 Small, Gemma 2 2B, and Qwen 3.5 2B Base, its components are substantially more non-Gaussian than random directions and public SAE decoder directions. To study what textual evidence makes these directions exceptional, we introduce the effective receptive field (ERF), which measures how much textual evidence is needed for a direction to activate. ERF reveals a layer-wise shift from token-local components in shallow layers to broader context-dependent components in deeper layers. It also shows an empirical link between non-Gaussianity and interpretability—more non-Gaussian directions tend to require less context to activate, making them easier to explain from top examples.

Interpreting ICA components (Section 5). We build an interactive ICA explorer and manually annotate components across all three model families. Through random audits, secondary expert checks, controlled prompt tests, and case studies, we find that ICA components expose readable structure in both residual streams and token embeddings. The recovered components capture lexical form, local syntax, phrase templates, discourse context, semantic ambiguity, long-range repetition, and embedding-level lexical organization. These results show that ICA components are not merely high-kurtosis artifacts, but inspectable directions that support testable working hypotheses.

Testing feature utility (Section 6). Using SAEBench Sparse Probing and Targeted Probe Perturbation (TPP), we test whether ICA directions both concentrate concept-relevant information and support selective interventions (Karvonen et al., 2025). We compare ICA with public high-capacity SAE dictionaries, Matryoshka SAE variants with smaller dictionary sizes (Bussmann et al., 2025), ITDA as a training-light sparse-coding alternative (Leask et al., 2025), and PCA as a classical statistical baseline (Abdi and Williams, 2010). Across models, ICA remains highly competitive with SAEs and consistently outperforms ITDA and PCA in sparse probing. It also achieves stronger TPP performance than public SAEs under small-to-medium intervention budgets. These results suggest that ICA directions are useful feature coordinates, and that non-Gaussianity is a more informative interpretability signal than variance.

Analyzing the ICA–SAE Relationship (Section 7). We find that non-Gaussianity maximization and sparse reconstruction recover related yet non-redundant directions. ICA recovers many SAE-aligned directions, while also exposing components that are weakly captured by any single SAE feature. We also find that the two methods exhibit distinct token-level activation patterns. ICA directions often show strong activations across a set of concept-related tokens, whereas SAE features more often activate on localized tokens. These findings position ICA as a complementary first lens for language-model activations, rather than as a replacement for sparse autoencoders. SAEs learn large dictionaries optimized for sparse reconstruction, making them well-suited for high-resolution feature discovery. ICA asks how much interpretable structure is already visible from activation geometry, helping us decide where heavier dictionary learning is worth the cost.

### 2. Related Work

Early work showed that sparse dictionary learning can recover latents that are more monosemantic, interpretable, and causally useful than neurons or classical decompositions (Huben et al., 2024, Bricken et al.,

- 2023, Templeton et al., 2024, Gao et al., 2025). This success has made sparse autoencoders the dominant route for turning language-model activations into inspectable feature coordinates and has led to a broader ecosystem: automated feature labeling (Paulo et al., 2024), standardized evaluation (Karvonen et al., 2025, Makelov et al., 2025), improved architectures and objectives (e.g., Gated, TopK, JumpReLU, BatchTopK, Switch, and Matryoshka SAEs), and open model-wide releases such as Gemma Scope, Llama Scope, and Qwen Scope (Rajamanoharan et al., 2024a, Gao et al., 2025, Rajamanoharan et al., 2024b, Bussmann et al.,
- 2024, Mudide et al., 2025, Bussmann et al., 2025, Lieberum et al., 2024, Deng et al., 2026).

Independent Component Analysis (ICA) is a classical lightweight alternative that searches for statistically independent, non-Gaussian directions in a linear representation space (Hyvarinen, 1999, Hyvarinen and Oja, 2000). This objective has recovered interpretable components in images, fMRI, word embeddings, and multimodal embedding spaces (Bell and Sejnowski, 1997, Daubechies et al., 2009, Yamagiwa et al., 2023, 2024, Musil and Mareček, 2024). However, ICA has received much less attention as a practical tool for analyzing modern LLM activations. Unlike SAEs, it has not been developed into a complete analysis workflow with stable fitting recipes, feature browsers, human annotation protocols, and task-level evaluations.

Our work revisits ICA for modern LLM interpretability. We build a more complete ICA workflow, including a more stable fitting pipeline, an annotation and analysis platform, and a broader evaluation suite. Our results show that ICA has been underestimated, and position ICA as a compact complement to SAEs: lower-capacity than an overcomplete dictionary, but cheaper to fit, easier to browse, and competitive with public SAEs while outperforming the lightweight baselines we evaluate.

### 3. ICALens: Making ICA Practical for LLM Activations

FastICA (Hyvarinen, 1999) is a fixed-point method for independent component analysis. Given cached activations Xℓ ∈ Rn×d from layer ℓ, it first centers and whitens the activations to obtain Z ∈ Rn×m. After whitening removes second-order correlations, ICA searches for a rotation W ∈ Rm×m whose coordinates are as non-Gaussian as possible. With the standard log-cosh contrast G, this can be viewed as

n

m

G((ZW⊤)ij), S = ZW⊤,

1 n

∑

∑

max

WW⊤=I

i=1

j=1

where S contains the signed ICA component scores. In this work, we use the parallel FastICA variant and implement it in PyTorch so fitting can run efficiently on GPU.2

It is conceptually simple, but a direct application to modern LLM activations is brittle. Activation matrices are high-dimensional, often contain large-norm token positions, and can include a small number of slow or oscillating components that dominate worst-case convergence statistics. Thus, a naive FastICA run can reject an entire layer even when most directions have already stabilized and remain useful for analysis.

Algorithm 1 summarizes our layer-wise pipeline. We make FastICA practical with three simple recipes. First, row normalization reduces the influence of activation-norm outliers before whitening (Section 3.2). Second,

2We build on https://github.com/RichieHakim/FastICA_torch. The implementation follows the standard FastICA interface, similar in spirit to scikit-learn’s FastICA, while using PyTorch tensors and GPU-parallel linear algebra. We further modify it for numerical stability on LLM activations.

p95-LIM provides a fallback acceptance rule when most components have stabilized but a small tail remains difficult (Section 3.3). Third, adaptive refitting reduces the target component count only when needed, returning the highest accepted resolution for each layer under the same convergence standard (Section 3.4).

Algorithm 1 ICALens layer-wise ICA fitting pipeline. Highlighted steps are our additions to standard FastICA.

Require: Activation matrix Xℓ, initial component count m0, LIM threshold τ, maximum iterations T

Ensure:1: M ←Artifact(m0,⌊m0A/ℓ2⌋=,⌊(mµ,0K/4,⌋W, . . ., R), D[A3, Sadaptive, Ist, Itailcomponent, status) search]

- 2: for m ∈ M from largest to smallest do
- 3: Normalize rows: x¯iℓ ← xiℓ/max(∥xiℓ∥2, ϵ), for i = 1, . . . , n [A1 row normalization]

- 4: Center activations: µ ← n−1 ∑i x¯iℓ, Xcℓ ← X¯ ℓ − 1µ⊤
- 5: Whiten activations: compute K, Z ← XcℓK⊤
- 6: Run parallel FastICA on Z with m components for at most T iterations
- 7: Obtain rotation W, source scores S, and final component-wise LIM vector λ
- 8: Record max(λ), p95(λ), and the full vector λ [A2 convergence audit]

- 9: if max(λ) < τ then
- 10: status ← Strict-Accept, Ist ← {1, . . . , m}, Itail ← ∅
- 11: else if p95(λ) < τ then
- 12: status ← P95-Accept, Ist ← {j ∶ λj < τ}, Itail ← {j ∶ λj ≥ τ} [A2 robust acceptance]

- 13: else
- 14: continue [A3 reduce component count and refit]

- 15: end if
- 16: R ← WK, D ← R†

- 18:17: endreturnfor Aℓ = (µ, K,W, R, D, S, Ist, Itail, status)
- 19: return Reject-Layer with diagnostics from all attempted component counts [A3 failed fit]

The returned artifact separates two roles. The reading map R = WK projects normalized, centered activations into signed component scores, S = XcℓR⊤. These scores indicate when a component is active. We use them for non-Gaussianity analysis, top-example retrieval, human annotation, activation-pattern visualization, and sparse probing. Since ICA scores are signed, annotation inspects both positive and negative sides when needed. The writing map D = R† maps component coordinates back to the normalized fitting space. Its columns are the activation-space directions associated with ICA components. We use these write directions for comparison with SAE decoder directions and for intervention-style experiments, including targeted probe perturbation and steering-style edits.

###### 3.1. Activation Corpus

We evaluate ICA on GPT-2 Small, Gemma 2 2B, and Qwen 3.5 2B Base. For each model, we fit ICA independently at the embedding layer and at each residual-stream layer. The embedding layer provides a noncontextual view of token-identity structure, while residual-stream layers expose contextual representations formed during autoregressive processing.

We refer to each recorded residual-stream hidden-state vector as an activation. For residual-stream fitting, we collect one million post-block residual-stream activations from randomly sampled token positions in the Pile-10k training split for each model.3 These sites match the residual-stream locations used by the

3https://huggingface.co/datasets/NeelNanda/pile-10k

corresponding public SAE checkpoints. In implementation, we use the hidden states returned by the Hugging Face model interface.4

Documents are truncated to a context length of 1024, and token positions are sampled uniformly without replacement using a fixed random seed. We run the model on the truncated documents and record hidden states at the sampled positions, so each activation is computed with its proper left context rather than as an isolated token.

Let xi(ℓ) ∈ Rd denote the residual-stream activation at sampled token position i and layer ℓ. For each layer, we form

X(ℓ) = [x1(ℓ), . . . , xn(ℓ)]

⊤

∈ Rn×d, n = 106.

This gives a task-agnostic activation corpus that is not tuned to any later case study or benchmark. For the embedding-layer analysis, we replace the contextual activation matrix with the model’s static input embedding matrix and pass it through the same fitting interface. The embedding matrices contain 50,257 rows for GPT-2, 256,000 rows for Gemma 2 2B, and 248,320 rows for Qwen 3.5 2B Base.

100

| | | | | | |
|---|---|---|---|---|---|
|Row-normalized<br><br>Raw max p95<br><br>| | | | | |
| | | | | | |

10−1

10−2

limLimitstatistic

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

FastICA iteration

Figure 1: Layer-3 convergence diagnostics for GPT2 Small. Blue curves use row-normalized activations and orange curves use raw activations. Solid curves report max-LIM, while dashed curves report p95LIM. Full diagnostics are provided in Appendix A.

Max-LIM only Max-LIM + p95 fallback Raw Norm. Raw Norm.

Rows

1k 2 / 3,431 9 / 2,177 2 / 3,431 9 / 2,177 3k 3 / 3,294 5 / 3,155 3 / 3,294 5 / 3,155

10k 2 / 3,499 6 / 3,206 4 / 3,383 6 / 3,206 100k 7 / 3,224 7 / 3,055 9 / 3,010 9 / 2,710

1M 2 / 3,492 8 / 3,107 6 / 2,954 10 / 2,741

Table 1: GPT-2 Small convergence yield. Each cell reports accepted layers / total iterations across 12 residual-stream layers. For the p95 fallback, we use max-LIM whenever it succeeds and p95-LIM only when the strict criterion fails. Bold marks the best setting for each row.

###### 3.2. Row-Normalization

The first practical difficulty is the input geometry seen by the optimizer. When FastICA is applied directly to raw LLM activations, we often observe slow or unstable convergence. Prior work suggests two relevant intuitions. First, raw transformer activation spaces contain systematic outlier dimensions, rare massive activations, and attention-sink tokens, making raw activations highly anisotropic (Dettmers et al., 2022, Sun et al., 2024, Xiao et al., 2024, Han et al., 2025). Second, many activation-steering works treat directions in residual-stream space as meaningful objects for reading or controlling model behavior (Panickssery et al., 2024, You et al., 2026). We therefore normalize activations before fitting ICA, and surprisingly find that this recipe makes the fitting dynamics more stable and efficient.

Concretely, before the usual centering and whitening, each activation vector is normalized by its ℓ2 norm: r(xi) = max(∥xxi

i∥2,ϵ) , where ϵ is a small numerical constant. ICA is then fit on the row-normalized matrix X˜ , whose rows are r(xi)⊤.

4For the final layer of GPT-2 Small, the hidden state returned by Hugging Face is after the model’s final LayerNorm. We therefore use the raw block output before the final LayerNorm, preserving the same residual-stream convention used for earlier layers.

- Figure 1 shows a representative convergence diagnostics example on GPT-2 Small layer 3. The normalized curves stay below the raw-activation curves through most of training. The normalized max-LIM curve reaches the 10−4 threshold within the iteration budget, while the raw max-LIM curve remains above it. Table 1 shows the same pattern at the layer level. Under the strict max-LIM criterion, normalization improves the accepted-layer count from 2 to 9 at 1k fitting rows, from 2 to 6 at 10k rows, and from 2 to 8 at 1M rows. It also usually reduces the total number of fitting iterations. The later annotation and intervention results show that the normalized components remain useful despite the loss of raw norm information.

- 3.3. Robust Convergence Acceptance The second practical difficulty is convergence diagnosis. Standard FastICA uses a worst-case stopping rule:

max

j

LIMj < τ.

This rule accepts a fit only when every component has stabilized up to sign. We keep this strict criterion whenever it succeeds. However, in high-dimensional LLM activation fits, a small number of slow or oscillating components can dominate the maximum and cause the whole layer to be rejected. For exploratory interpretability, this is too conservative: many stable directions may be discarded because of a small unstable tail.

To make convergence less sensitive to the single hardest component, we record the full distribution of per-component LIM values. When the strict rule fails within the iteration budget, we use a softer fallback criterion:

p95(LIMj) ≤ τ. This accepts a layer when almost all components have stabilized, while still flagging the remaining tail as difficult. For layers accepted by this fallback, we retain all components but attach a stability flag to each one. Components within the accepted 95% are marked as stable, while the remaining tail components are marked as unstable. We do not discard the unstable tail from qualitative inspection, since a slow-converging component may still exhibit a recognizable activation pattern. However, we preserve this flag in downstream audits and avoid using unstable components as primary evidence for convergence-sensitive claims.

- Figure 1 illustrates this distinction. The dashed p95-LIM curves decrease more smoothly, while the solid max-LIM curves show sharper spikes and slower worst-case behavior. The aggregate effect appears in Table 1. With 1M fitting rows, raw activations accept only 2 of 12 GPT-2 layers under max-LIM alone, but 6 layers with the p95 fallback. With row normalization, the same comparison improves from 8 to 10 accepted layers and reduces total iterations from 3,107 to 2,741.

3.4. Adaptive Refit for Difficult Layers

Some layers still fail to meet the acceptance criterion within the iteration budget even after row normalization. We handle these cases with an adaptive refit procedure. We start with m = d and halve m until the layer satisfies either the strict max-LIM criterion or the p95-LIM criterion, or until a minimum component count mmin = 16 is reached. This gives each layer the highest accepted resolution under the same convergence standard.

- Figure 2 summarizes the final accepted component counts across layers. Many layers converge at the full hidden dimension, while some layers require reduced component counts. The accepted component count is therefore also a diagnostic of fitting difficulty. A layer that accepts m = d has a full-dimensional ICA basis. A layer that only accepts m < d still provides a valid set of non-Gaussian directions.

components/hiddendim.

###### GPT-2 Small (d=768)

###### Gemma 2 2B (d=2304)

Qwen 3.5 2B Base (d=2048)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
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

1

- 0
- 1/4
- 1/2
- 3/4

0 2 4 6 8 10

0 5 10 15 20 25

0 4 8 12 16 20

layer

- Figure 2: Final ICA component counts selected by the adaptive refit procedure. For each layer, we initially attempt to fit the full hidden dimension and reduce the target component count when convergence fails. Dashed horizontal lines indicate the hidden dimension of each model.

4. What Does ICA Recover? Non-Gaussianity and Context Dependence

The previous section made ICA stable and efficient enough to fit on LLM activations. We now ask what kind of directions it recovers. A useful direction should pass two checks. First, it should be statistically exceptional rather than an arbitrary rotation of the activation space. Second, its exceptionality should correspond to recoverable structure in text, ranging from token-local patterns to broader contextual signals.

This section separates these two steps. We first verify that ICA components are substantially more nonGaussian than random projections and SAE decoder directions (Section 4.1). We then introduce effective receptive field (ERF), a diagnostic for each component (Section 4.2). ERF measures how much context is sufficient to recover a component’s activation at a target token, helping us characterize the context dependence of ICA components across layers and models.

- 4.1. Non-Gaussianity Separates ICA Directions from Random Projections

For a direction v and a set of row-normalized activations {h˜i}in=1, we measure the excess kurtosis of its scalar projection

zi(v) = ⟨v, h˜i⟩ as

κ(v) =

1 n ∑in=1(zi(v) − z¯(v))4

(n1 ∑in=1(zi(v) − z¯(v))2)2

− 3, z¯(v) =

1 n

n

∑

i=1

zi(v).

A Gaussian projection has excess kurtosis close to zero. Larger values indicate that the direction has a heavier-tailed or otherwise more exceptional projection distribution.

We compare three families of directions: random unit directions as a null baseline, SAE decoder directions as learned sparse-dictionary directions, and ICA score directions recovered by our fitted ICA maps. All three direction families are evaluated as projection directions on the same row-normalized activation distribution. This comparison therefore tests whether a direction is statistically exceptional under a common input geometry.

- Figure 3 shows a consistent separation across all three model families. ICA directions are substantially more non-Gaussian than random directions and SAE decoder directions across layers, confirming that our fitted ICA components successfully recover the statistical objective of the method on LLM activations. Interestingly, SAEs are not trained to maximize kurtosis, but their decoder directions already show elevated

non-Gaussianity, suggesting that non-Gaussianity is a common statistical signature of many learned feature directions. This gives us a useful perspective on why ICA can serve as an interpretability lens: ICA makes this inductive bias explicit by directly searching for directions with exceptional projection statistics, rather than learning it implicitly through sparse reconstruction as in SAEs. In Sections 5–7, we show that these non-Gaussian directions from ICA are also human-interpretable and behaviorally useful under downstream feature evaluations.

Random SAE decoder ICA

###### GPT-2 Small

###### Gemma 2 2B

Qwen 3.5 2B Base

- 100

- 101

- 102

excesskurtosis

0

0 2 4 6 8 10

0 5 10 15 20 25

0 4 8 12 16 20

layer

- Figure 3: ICA recovers highly non-Gaussian directions in LLM activation space. For each model and layer, we project row-normalized residual-stream activations onto random unit directions, public SAE decoder directions, and fitted ICA score directions, then compute the excess kurtosis of each projection distribution. Across GPT-2 Small, Gemma 2 2B, and Qwen 3.5 2B Base, random projections remain close to Gaussian, SAE directions show elevated non-Gaussianity, and ICA directions are the most non-Gaussian, validating the statistical objective of the ICA fit. Statistics are computed over one million token positions from Pile-10k; random baselines use 32,768 Gaussian unit directions per layer. Lines show medians across directions, and shaded bands show interquartile ranges.

###### 4.2. From Non-Gaussianity to Context Dependence

Non-Gaussianity tells us that a direction is statistically unusual, but not what textual structure makes it unusual. A high-kurtosis component may correspond to a local token pattern, a short phrase, or a broader contextual signal. We therefore need a way to ask whether ICA is only finding local detectors or also recovering components whose activation depends on context. To separate these cases, we introduce effective receptive field (ERF).

ERF asks a simple question: how much left context is sufficient to recover the same signed component response at a target token? Small-ERF components can be recovered from the token itself or a short suffix, while large-ERF components require a broader preceding context.

Computing ERF. ERF is computed from a set of evidence examples for each component. For a target token position t in a sequence x1∶T, let ht(x1∶t) be the residual-stream activation computed with the full available context, and let sj(ht) be the signed score of ICA component j. We first identify the highest-scoring example of component j in absolute value and record its sign. We then collect all examples with the same sign whose absolute score exceeds half of this maximum absolute score, forming an evidence set ℰj. In practice, this typically yields around 20 evidence examples per component.

For each evidence example (x, t), we construct suffixes ending at the target token with lengths xt−k+1∶t, k = 1,2, . . . , Kmax.

Let h(tk) denote the final-token activation obtained from the length-k suffix. We then recompute ICA scores at this token. A suffix of length k is said to recover component j if the component remains among the top 15 components ranked by absolute score and preserves its original sign. Formally, let

Rj(x, t, k) = 1[j ∈ Top15({∣sr(h(tk))∣}r) ∧ sign(sj(h(tk))) = sign(sj(ht(x1∶t)))].

We search suffix lengths in increasing order and stop at the first successful recovery. The sample-level ERF is the shortest suffix length that recovers the component,

erfj(x, t) = min{k ∈ {1, . . . , Kmax} ∶ Rj(x, t, k) = 1}.

If no suffix of length at most Kmax succeeds, we assign erfj(x, t) = Kmax and mark the example as unrecovered within the tested window. In our experiments, Kmax = 11. The component-level ERF is the mean sample-level ERF over evidence examples:

ERF(j) =

erfj(x, t).

1 ∣ℰj∣

∑

(x,t)∈ℰj

1

2

3-5

6-10

11+ / unrecovered

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

###### GPT-2 Small

Gemma 2 2B

Qwen 3.5 2B Base

1.0

| | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | |

fractionofcomponents

0.8

0.6

0.4

0.2

0.0

0 2 4 6 8 10

0 4 8 12 16 20 24

0 4 8 12 16 20

##### layer

- Figure 4: Layerwise distribution of effective receptive field (ERF). For each ICA component, we average the sample-level ERF over its evidence examples and bucket components by the resulting mean ERF. Each stacked bar shows the fraction of components in one layer. The final bin includes components not recovered within the 11-token window, reported as 11+

.

ERF reveals a local-to-contextual spectrum. Figure 4 summarizes component context dependence as a layerwise distribution, showing many insights. First, ICA components span a broad local-to-contextual

spectrum. Many components are recovered from the target token or a short suffix, but a substantial fraction require longer context. Thus, ICA is not merely finding lexical detectors; it also exposes directions whose activation depends on the surrounding text.

Second, ERF provides a quantitative lens on layer specialization. It shows that component scope changes with depth through a gradual shift in mixture proportions, not through a sharp handoff between layer types. Early layers are dominated by token-local components but already contain context-dependent directions, while later layers contain more medium- and long-context components yet still retain many token-local directions. Thus, layers are better described by the mixture of component types they contain than by a single local-toglobal label. The distribution also shows a non-monotonic pattern. The largest-ERF components, including those recovered only at the maximum tested length or not recovered within the 11-token window, are often most prevalent in middle layers. This makes middle layers a useful target for inspecting context-dependent ICA components.

GPT-2 Small n = 7, 981, ρs = −0.50

Gemma 2 2B n = 16, 703, ρs = −0.41

Qwen 3.5 2B Base n = 40, 447, ρs = −0.46

11

| |[Figure 1]|
|---|---|
| | |
| | |
| | |
| | |

| |[Figure 2]|
|---|---|
| | |
| | |
| | |
| | |

| |[Figure 3]|
|---|---|
| | |
| | |
| | |
| | |

|[Figure 4]<br><br>|
|---|

componentsperbin

9

- 100
- 101
- 102

meanERF

7

5

3

1

100 101 102 103

100 101 102 103

100 101 102 103

excess kurtosis K

- Figure 5: Relationship between effective receptive field (ERF) and excess kurtosis. Each cell counts ICA components with a given excess-kurtosis range and mean ERF. Across all three models, components with larger excess kurtosis tend to have smaller ERFs, with Spearman correlations ranging from −0.41 to −0.50. This indicates that high-kurtosis components are typically more local, while broad-context components tend to have lower kurtosis and larger ERFs.

ERF offers one explanation for the kurtosis–interpretability link. Prior SAE work found that feature kurtosis is positively correlated with automated interpretability scores (Huben et al., 2024). This suggests that heavy-tailed feature activations are often easier for existing auto-interpretation pipelines to explain, but the correlation itself does not explain why. We use ERF to probe this link. Figure 5 shows a consistent negative relationship between ERF and excess kurtosis across all three models. Components with larger excess kurtosis tend to have smaller ERFs. These components are often activated by narrow, recurring patterns such as tokens, surface forms, or short phrases, so their top examples expose the triggering pattern directly and are easier to summarize for both human annotators and LLM-based automatic interpretation methods. By contrast, larger-ERF components depend on broader context. Their activation conditions may not be visible at the target token or in a short local window, making them less sharply kurtotic and harder to explain from standard top-example views. Thus, ERF provides one operational account of why kurtosis can be informative for interpretation—high kurtosis is associated with locality, and locality makes top-example explanations easier to form and verify. We treat this as a correlational diagnostic.

This interpretation also makes ERF useful for organizing annotation. Small-ERF components can often be analyzed from top examples alone, while larger-ERF components typically require context ablations or longer

evidence windows. ERF therefore serves as a proxy for annotation difficulty. We use this ERF-guided strategy in our manual analysis in Section 5.2, and it may also help guide future automatic annotation systems.

### 5. Human Inspection of ICA Components

After quantifying the statistical structure and context dependence of ICA components, we now ask a more direct interpretability question: can these components be inspected, labeled, and reused in a controlled way? This section describes the inspection interface and annotation protocol, audits randomly sampled components, and then uses the resulting labels to analyze contextual and lexical structure.5

###### 5.1. An Interactive Lens for Component Inspection

[Figure 5]

- Figure 6: Interactive inspection view for GPT-2 Small layer 6. For each token, the explorer displays the strongest ICA components, signed scores, working labels, top examples, ERF, kurtosis, and annotation metadata used during component annotation. Additional screenshots for Gemma 2 2B and Qwen 3.5 2B Base are provided in Appendix F.

To support component interpretation, we built an interactive explorer that keeps the relevant evidence for each component visible during analysis. Given an input sequence, the interface displays the strongest ICA components at each token, ranked by absolute signed score. For each selected component, it shows the signed score, current working label, top-scoring examples, original document context, ERF, excess kurtosis,

5We treat component labels as working hypotheses: short descriptions of the dominant pattern captured by a signed ICA direction, not ground-truth semantic definitions.

and annotation metadata. The explorer also links components across adjacent layers using cosine similarity between ICA directions. These links often provide useful clues when related patterns persist across depth.

The explorer is a quality-control tool for annotation. Its main role is to prevent labels from being assigned from a few isolated top examples without checking whether the component behaves consistently under additional contexts. The interface therefore supports both passive inspection of high-scoring examples and active hypothesis testing with targeted prompts.

###### 5.2. Annotation Protocol

Given an activation ht, we rank ICA components by absolute signed score ∣sj(ht)∣. The magnitude indicates how salient component j is at the target token, while the sign selects one side of the ICA direction. Since ICA directions are identifiable only up to sign, annotations are sign-specific. The two sides of the same component may receive different labels when they correspond to different activation patterns.

Our annotation protocol is evidence-controlled and ERF-guided. For each component side, the expert annotator first checks the component’s ERF to estimate the expected scope of evidence. Low-ERF components can often be interpreted from the target token and its local context, while high-ERF components usually require broader document context or explicit context ablations. The annotator then inspects a fixed set of high-scoring examples, including the target token, signed component score, and surrounding text. When the pattern is ambiguous, the annotator also examines middle-range, near-zero, and opposite-sign examples to avoid explaining only the most extreme activations.

The annotation proceeds as an iterative hypothesis-testing process. The annotator first proposes a concise working label from the evidence examples. They then construct targeted prompts that should activate the component and matched counterexamples that should not. The explorer is used to check whether the component appears at the expected token positions with the expected sign. For context-dependent components, the annotator additionally performs context ablations by progressively removing preceding text and monitoring whether the component remains active. The label is revised until it is stable across top examples, targeted tests, and, when needed, context ablations.

Each labeled side receives a working label, a confidence level, and an interpretation type. Confidence is one of high, medium, low, or unclear. A high-confidence label consistently explains the observed behavior; a medium-confidence label captures a clear but harder-to-summarize pattern; a low-confidence label gives only a partial explanation; and an unclear label fails to provide a reliable interpretation. The interpretation type records the scale of the pattern: form components respond to surface features such as capitalization or initial letters, word components are tied mainly to token identity, phrase components respond to short local constructions, sentence components track sentence-level structure or topics, long-range components depend on earlier distant words or repetitions, global components reflect broader genre or discourse setting, position components respond mainly to token position, and sophisticated components appear to implement more conditional patterns.

###### 5.3. Random Component Audit

To assess whether ICA components are practically inspectable, we conduct a random component audit. For each model, we uniformly sample 50 converged ICA components without replacement from the full component inventory, including embedding-layer components. For each sampled component, the expert annotators label the dominant signed side—the sign of the strongest absolute-score example. Empty labels and placeholder labels such as “?” are counted as unclear.

Model High Med. Low Unc. W P F S G/L/Pos/X GPT-2 Small 44 4 1 1 23 8 7 5 6 Gemma 2 2B 43 3 2 2 15 11 3 7 13 Qwen 3.5 2B Base 40 1 4 5 16 14 1 8 6 Total 127 8 7 8 54 33 11 20 25

- Table 2: Random component annotation summary. We sample 50 ICA components per model and annotate the dominant signed side of each component. High/Med./Low/Unc. report annotation confidence. Type columns count the main interpretation type: word (W), phrase (P), form (F), sentence (S), and the union of global, long-range context, position, and sophisticated components (G/L/Pos/X). The complete annotation table appears in Appendix C.

Component Expected Prompt Score Rank L0/C192: After high, Word, negative side, ERF 1.0, κ = 298

L0/C192 activate I went outside to play after I finished my homework. −22.447 1 L0/C192 activate The puppy ran after the ball. −22.700 1 L0/C192 not activate I went outside to play when I finished my homework. −1.017 18 L0/C192 not activate The puppy ran through the ball. +1.503 21

L7/C17: Scientific research / citation high, Sentence, negative side, ERF 2.0, κ = 33

L7/C17 activate Smith, J., & Lee, K. (2021). −9.480 1 L7/C17 activate Vaswani, A., et al. (2017). Attention is all you need. −16.003 1 L7/C17 not activate Smith and Lee (2021) is a fictional patent citation used here for illustration purposes

only.

−2.051 14

L7/C17 not activate In 2017, Ashish Vaswani led the team that published Attention Is All You Need, introducing the Transformer model.

+0.114 639

L10/C368: Gaming language high, Global, positive side, ERF 3.2, κ = 31

L10/C368 activate Don’t fight before dragon unless our jungler has Smite up; save your ult for their engage.

+21.492 1

L10/C368 activate If the tank does not kite during the second phase, the healers get overwhelmed by the damage.

+16.429 1

L10/C368 not activate Don’t leave before dinner unless everyone has arrived; save your announcement for their arrival.

+0.158 634

L10/C368 not activate If the manager does not delegate during the busiest part of the project, the team gets overwhelmed by the workload.

−0.328 471

L8/C738: Repetition of a prior section header medium, Long-range, negative side, ERF 7.7, κ = 23

L8/C738 activate ### Section 1: Discovery ...### Section 2: Analysis −11.160 1 L8/C738 activate Cat: maomao. Dog: wowowow? Cat: mamoa. −7.496 2 L8/C738 not activate ### Discovery ...### Analysis +0.352 502 L8/C738 not activate At 3:30 p.m., the race began, and the score was 3:2 after the first round. −1.737 55

- Table 3: Representative prompt-level cases from the random component audit. Label-consistent prompts usually produce a large signed score and high rank at the target token, while matched label-inconsistent prompts reduce the score or rank. These controlled cases complement top-example inspection and make the working labels easier to falsify.

- Table 2 shows that most sampled components admit usable working labels under this protocol. Across 150 randomly sampled components, 142 receive non-unclear labels and 127 receive high-confidence labels. The labels are not limited to token identity. They include local form patterns, word categories, phrase templates, sentence-level constructions, global topics, position effects, and long-range or conditional patterns. We interpret these results as evidence of practical inspectability, because a randomly sampled ICA component

Model Component ERF Type Working label GPT-2 Small L0/C192 1.0 Word After GPT-2 Small L9/C445 5.3 Long-range Refer to a spokesperson Gemma 2 2B L13/C30 8.6 Sophisticated Refer back to one of two alternatives Gemma 2 2B L10/C261 9.2 Long-range Repetition and number increases Qwen 3.5 2B Base L18/C1794 3.5 Global Study protocol Qwen 3.5 2B Base L22/C511 4.9 Sentence Either/or, whether/or construction

- Table 4: Representative labels from the random component audit. The examples span token-local, sentence-level, global, long-range, and conditional patterns. Labels are working descriptions under the inspection protocol, not ground-truth semantic definitions.

often supports a concise and testable human hypothesis.

- Table 3 shows representative prompt-level cases from the initial audit and how a working label is tested in practice. For each component, label-consistent prompts are paired with matched label-inconsistent prompts. We report the signed component score and its rank among components at the target token. Table 4 gives a compact view of representative audited labels across models.

Model Component Initial label Score Revised label GPT-2 Small L0/C192 After 10 after token GPT-2 Small L7/C17 Scientific research, Citation 9 scientific reference-title final period GPT-2 Small L10/C368 Gaming language 9 video-game discourse GPT-2 Small L9/C445 Refer to a spokesperson 8 spokesperson/spokesman name refer-

ence

Gemma 2 2B L24/C355 Turkish language 10 Turkish-language token/subword Gemma 2 2B L13/C30 Refer back to one of two

recently introduced alternatives

9 reference to one member of a recent pair

Gemma 2 2B L10/C261 Repetition and number increases

8 repeated identifier with increasing nu-

meric suffix Gemma 2 2B L20/C263 Cryptography, Computer Se-

curity

8 cryptography / security context

- Qwen 3.5 2B Base L18/C1794 Study protocol 10 clinical study protocol/approval language
- Qwen 3.5 2B Base L19/C824 Follow, violate rules 10 rule compliance/violation language Qwen 3.5 2B Base L22/C511 Either or, Whether or, One

way or another

9 alternative/concessive construction

Qwen 3.5 2B Base L11/C1211 Because of, As a result of 9 causal preposition phrase

- Table 5: Representative rows from the secondary expert audit of high-confidence labels. The table reports the initial working label assigned in the first-stage expert annotation, the 0–10 audit score, and the revised label suggested by the second expert auditor. The revised labels usually preserve the original interpretation while making it more specific.

We further test whether the first-stage expert labels remain supported under an independent expert audit using a prompt-based validation protocol. For each of the 127 high-confidence labels, a second expert auditor inspected the component evidence and the initial working label. The auditor then constructed matched contrastive prompts: label-consistent prompts intended to activate the component and label-inconsistent

controls intended not to activate it. After running the frozen model and ICA decomposition on these prompts, the auditor assigned a 0–10 support score and wrote a revised label when the original description could be made more precise.

- Table 5 shows representative rows from the secondary expert audit across all three models. Each row includes the initial label, the audit score, and the revised label suggested after the second inspection. The full audit table is provided in Appendix D. This secondary audit serves as a consistency check. Among the 127 audited labels, 121 are supported and the remaining 6 are partially supported; none are rejected. Moreover, 112 of the 127 labels receive a score of at least 8. The audit also shows that many initial labels can be sharpened, for example from “Gaming language” to “video-game discourse” or from “Scientific research, Citation” to “scientific reference-title final period.”

###### 5.4. Case Studies: Reading Component Mixtures

The random audit asks whether ICA components can be labeled in isolation. We now ask how these labels behave when placed back onto concrete inputs. The case studies below use three complementary views. The first fixes a polysemous token and follows how its representation decomposes across layers. The second fixes two labeled components and traces their scores across sentences. The third removes context entirely and studies how embedding-layer ICA decomposes familiar lexical relations.6

###### 5.4.1. Contextual Decomposition of a Polysemous Word

Polysemous words provide a controlled setting for studying contextual decomposition because the surface token is fixed while the surrounding evidence changes. We use a short paragraph containing four occurrences of bank. Two refer to a financial institution, denoted F1 and F2, and two refer to a river edge, denoted R1 and R2.

Maya stopped at the bank [F1] before the trip, waiting in line to deposit a check and withdraw enough cash for the weekend. The teller asked about her travel plans, and Maya said she was driving north to visit the old village by the river. By sunset she was sitting on the grassy bank [R1] with her shoes beside her, watching the water curl around stones and reeds. It amused her that the same word had followed her all day: first a bank [F2] with counters, accounts, and vaults, then a bank [R2] of earth holding the river in place.

- Figure 7 is a component-level decomposition of four occurrences of the same token. Each column corresponds to one bank token in the paragraph. Each row corresponds to a layer. Within each cell, we list the five ICA components with largest absolute score at that token position. The bar length shows relative score magnitude within the cell. The text label gives our working interpretation of the signed component side, and the badge records annotation type and confidence. Equivalent bank walkthroughs for Gemma 2 2B and Qwen 3.5 2B Base are provided in Appendix B.

The embedding row provides a useful control. All four columns are identical because the input embedding depends only on the token identity bank. The strongest embedding components include Bank, Mortgage, A Room or Place, Group, Team, and Letter B. Before contextual processing begins, all four occurrences share the same lexical representation.

The residual-stream layers show how this shared lexical state separates under context. F1 occurs after the prefix “Maya stopped at the.” The later evidence, such as “deposit a check” and “withdraw enough cash,” is

6Readers can reproduce and vary these examples in our interactive demo: ICA Explorer case-study demo.

###### layer F1 financial R1 river edge F2 financial R2 river edge

|W Bank, Mortgage W A Room or Place<br><br>? W Group, Team F Letter B|
|---|

|W Bank, Mortgage W A Room or Place<br><br>? W Group, Team F Letter B|
|---|

|W Bank, Mortgage W A Room or Place<br><br>? W Group, Team F Letter B|
|---|

|W Bank, Mortgage W A Room or Place<br><br>? W Group, Team F Letter B|
|---|

emb

|W Security, Finan... O Early Doc Posit... W School W Company W Government|
|---|

|W Security, Finan... W Town, Village W Material, Manuf... W Water W School|
|---|

|W Security, Finan... W School<br><br>? W Company W Water|
|---|

|W Security, Finan... W School<br><br>? W Company W Water|
|---|

L0

|O Early noun W Organization, U... W Business<br>P Follows "the" ?<br>|
|---|

|W Organization, U... W Water W Sign S Biology, Evolut... W Field|
|---|

|W Business P Follows "a" W Organization, U... W Pay W Water|
|---|

|W Business P Follows "a" W Organization, U... W Pay ?|
|---|

L2

|W Financial W Building, Place<br><br>O Early Document...<br>P Follows "the" W Country, Company<br>|
|---|

|W Place Name W Building, Place<br><br>? ?<br><br>W Financial|
|---|

|W Building, Place<br><br>? W Country, Company W Financial W Company Name|
|---|

|W Building, Place W Country, Company<br><br>? W Financial W Company Name|
|---|

L4

|W Financial<br><br>O Early Doc Posit...<br>P Organization, G... S Place W $ Dollar, Euro,...<br>|
|---|

|W Building, Place S Place W Financial F Letter B W Group, Team|
|---|

|P Follows "a" W Financial W Building, Place W Group, Team P Organization, G...|
|---|

|S Repetition P Follows "a" W Financial W Building, Place<br><br>?|
|---|

L6

|W Financial W Place, Building<br><br>O Early Doc Posit...<br>P Follows "the" W $ Dollar, Euro,...<br>|
|---|

|W Water P Follows "on" W Financial W Place, Building S Place, Location|
|---|

|W Financial P Follows "a" W Place, Building<br><br>?<br><br>W Water|
|---|

|W Place, Building P Follows "a" W Financial L Repetition W Company, Brand|
|---|

L8

|W Department S Investment W $ O Early Doc Posit... S Spacious Room|
|---|

|W Crouch, Sit S Body Movement P On Something W Area, Field S Spacious Room|
|---|

|W Department S Investment W Area, Field<br><br>Household Item... Adjective|
|---|

|W Product, Organi... W Together, Coupl... W Area, Field W Department S Investment|
|---|

L10

confidence: high medium low unclear type: F form W word P phrase S sentence L long-range context G global O position X sophisticated

- Figure 7: Contextual decomposition of a polysemous word in GPT-2 Small. We probe four occurrences of bank in one paragraph, including two financial-bank uses (F1, F2) and two river-edge uses (R1, R2). Each cell lists the five largest-absolute-score ICA components at the target token in a given layer. Bar lengths show relative score magnitude within the cell, and the legend decodes annotation type and confidence. Across layers, the same surface token is represented by different mixtures of lexical, syntactic, semantic, positional, and longer-context components.

unavailable at the target token, yet the prefix already supports a financial reading. Later layers therefore contain components related to finance, organizations, companies, investment, and departments. This shows that the model can begin to disambiguate bank from left context alone.

- R1 has a different mixture because the prefix already points to a physical scene. The paragraph has mentioned a village by the river, and the immediate prefix is “sitting on the grassy.” The active components shift toward labels such as Water, Place Name, Building, Place, Crouch, Sit, Body Movement, and Area, Field. These labels

should be read as scene factors rather than a single river-bank feature. The token representation combines water, location, posture, movement, and outdoor space.

F2 returns to the financial sense, but its context differs from F1. The immediate prefix is “first a,” and the preceding paragraph has already introduced both senses of bank. The model therefore represents F2 as part of a repeated-word contrast while also activating financial components such as Financial, Business, Organization, Pay, Department, and Investment. The same cell also contains phrase-level components such

- as Follows “a”. This illustrates that one token representation can mix semantic content, local syntax, and discourse structure.

- R2 is the easiest column to misread. A human reader identifies it as a river-edge use after seeing “bank of earth holding the river in place.” At the target token, however, the model has access only to the prefix “first a bank with counters, accounts, and vaults, then a.” This prefix explicitly mentions the financial sense and sets up a contrast. The finance-related components in R2, including Financial, Department, and Investment, are therefore expected under autoregressive conditioning. The cell reflects the model state at the target token, not the interpretation available after reading the following words.

This token-centered view gives two main lessons. First, ICA does not compress polysemy into one “financial versus river” direction. Each occurrence is represented by a changing mixture of lexical, phrase-level, semantic, positional, and longer-context components. Second, the mixture must be interpreted relative to the information available at the target position. The embedding row is token-driven, while later residual-stream layers incorporate the available prefix and discourse context.

###### 5.4.2. Component Traces Across Sentences

The previous analysis fixes a token and asks how its representation decomposes across layers. We now take the complementary view. We fix two labeled components and ask how each component behaves across sentence positions. This is necessary because a component that appears in the top list for one token may not be a detector for that token alone. It may track a broader context that persists across several related tokens.

- Sentence 1. She arrived at the library to study for her exam.
- Sentence 2. Maya stopped at the bank before the trip, waiting in line to deposit a check and withdraw enough cash for the weekend.

- Figure 8 shows this distinction directly. C67 is labeled as a financial component. In Sentence 1, which contains no financial context, its score stays close to zero. In Sentence 2, it becomes active around bank and remains active near deposit, check, withdraw, and cash. If C67 were only a detector for the word bank, we would expect a single spike at that token. Instead, the trace suggests a broader financial-context direction.

C273 illustrates the same point for a construction-level pattern. In Sentence 1, it rises around “arrived at the library to study,” which combines motion, destination, and purpose. The highest response occurs near the to study region, where the purpose relation becomes explicit. In Sentence 2, the same component also responds to related structure, including stopping at a place and waiting in line to perform an action. C273 is therefore better understood as an arrival- or purpose-related direction than as a detector for a single token such as arrived or to.

The insight from this trace view is simple. Some ICA components behave like local token or phrase detectors, but others trace contextual states across a span. Such traces help avoid overly narrow labels. A component whose largest score occurs near one token may still be responding to a larger construction. In our annotation

Sentence 1 C67 (Financial)

Sentence 2 C67 (Financial)

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |

5

0

|score|

Sentence 1 C273 (Arriving/Leaving for a purpose)

Sentence 2 C273 (Arriving/Leaving for a purpose)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

10

5

0

Maystoppeda atthebankbeforethetripwaiting, inlinedepositto checkawithdrawandenoughcashforweekendthe .

Shearrived at thelibrary to study for her exam .

- Figure 8: Sentence-level traces for selected GPT-2 Small ICA components. We plot raw absolute ICA scores across two sentences for layer-6 components C67 and C273. C67 tracks a financial-bank context and becomes active across several related tokens in Sentence 2. C273 tracks an arrival- or purpose-related construction and responds strongly to “arrived

- at the library to study” in Sentence 1, while also responding to related movement and purpose structure in Sentence 2. ICA directions can therefore trace context-sensitive signals across spans, rather than firing only at isolated tokens.

workflow, we use these traces as hypotheses and then check them with targeted prompts and context ablations.

- 5.4.3. Embedding Components Recover Lexical Structure

The previous two case studies analyzed contextual representations. We now apply the same inspection protocol to token embeddings. This setting is a useful control because token embeddings and residual-stream activations live in the same d-dimensional representation space, but embeddings are fixed for each token. Embedding-layer ICA therefore asks which non-Gaussian directions organize lexical representations before any context enters the model.

Classical word-embedding analyses often summarize lexical relations through vector offsets, such as king − man + woman ≈ queen. This view is useful, but it compresses a relation into one displacement. ICA gives a more inspectable view. Instead of asking whether an analogy can be solved by one offset, we ask which readable directions are active in each token embedding.

- Figure 9 is a component-level decomposition of static Qwen 3.5 2B token embeddings. Each box corresponds to one token, and each row inside the box is one high-scoring ICA component for that token embedding. Results for GPT-2 small and Gemma 2 2B are provided in Appendix B.

The first pattern is shared structure with extra factors. For king and queen, both tokens activate a Royal component, which recovers the expected royalty-related direction. Yet the two boxes also contain different additional components. Queen strongly activates Woman, Female and a surface-form component such as Letter Q, while king activates Letter K, morphology, and common-token components. Thus, the familiar pair is not represented only as “royalty plus gender.” It is represented as a mixture of semantic, gender-associated, morphological, and surface-form directions.

The second pattern is that different analogy families decompose in different ways. In the man/woman

|king<br><br>W Royal W Reg F Letter K F -ing W Common Words|
|---|

|man<br><br>W Man W Management W Men, Male W People, Person<br><br>?|
|---|

|father<br><br>W Parent W Men, Male W Son, Daughter W Common Words W Child|
|---|

|brother<br><br>W Friend F ClW Son, Daughter W Parent W Men, Male|
|---|

|actor<br><br>W Act W Film, Movie W Builder, Creator W Action W Activity|
|---|

|queen<br><br>W Royal W Woman, Female F Letter Q W Reg W Common Words|
|---|

|woman<br><br>W Woman, Female W Management W People, Person W Common Words W Man|
|---|

|mother<br><br>W Parent W Woman, Female W Common Words W Birth, Pregnancy W Son, Daughter|
|---|

|sister<br><br>W Friend W Woman, Female F ClW Son, Daughter W Parent|
|---|

|actress<br><br>W Woman, Female W Act W Film, Movie W CEO, Professor W Builder, Creator|
|---|

confidence: high medium low unclear type: F form W word P phrase S sentence L long-range context G global O position X sophisticated

- Figure 9: Embedding-layer ICA components for familiar analogy word sets in Qwen 3.5 2B Base. Each box corresponds to one token embedding and lists its five largest-absolute-score ICA components. Columns pair familiar masculine-coded tokens with their feminine-coded counterparts. Bar lengths show relative score magnitude within each box, and the legend decodes annotation type and confidence. The figure shows that familiar analogy pairs are supported by overlapping component directions, including royalty, gender-associated terms, family relations, professions, morphology, and common-token components.

pair, both tokens activate person-related directions, while gender-associated components distinguish the pair. In the father/mother pair, both tokens share a Parent component and child-relation components. Father is associated with Men, Male, while mother is associated with Woman, Female and Birth, Pregnancy. The brother/sister pair shows a different organization. Both tokens activate directions such as Friend, Son, Daughter, and Parent, suggesting that sibling terms are represented through overlapping social and family-relation components rather than one isolated sibling axis.

The profession pair gives a third view of the same decomposition. Actor is dominated by components related to acting, film, creation, action, and activity. Actress shares several of these directions, including Act, Film, Movie, and Builder, Creator, but also prominently activates Woman, Female. This makes the relation more specific than a profession offset. The embedding contains both the shared occupation-related structure and the gender-associated component that distinguishes the pair.

This component view also makes asymmetries visible. Queen and actress prominently activate components labeled Woman, Female, while king and actor emphasize other directions among their top components. Therefore, ICA can reveal asymmetric associations that are easy to hide when a relation is summarized by a single vector offset.

The main lesson is that embedding-layer ICA turns a coarse analogy view into a component-level explanation. Some directions are shared across a pair, such as royalty, parenthood, or occupation. Others distinguish the pair, such as gender-associated, morphological, profession-related, or common-token components. As in the residual-stream case studies, these labels are working descriptions rather than concept definitions. A component labeled Royal or Woman, Female is useful because it summarizes a recurring direction of variation in the inspected examples. This makes embedding-layer ICA a compact diagnostic for lexical structure before context enters the model.

### 6. Feature Utility under Sparse Probing and Intervention

The previous sections show that ICA directions are statistically exceptional and human-inspectable. We now ask whether they are useful as feature coordinates. A useful direction should concentrate task-relevant information in a small number of coordinates. Prior SAEBench evaluations compare SAEs mainly against PCA as the classical linear baseline, but do not systematically evaluate ICA (Karvonen et al., 2025). We therefore evaluate ICA on two SAEBench metrics: sparse probing for predictive feature utility and Targeted Probe Perturbation for selective feature intervention.

###### 6.1. Sparse Probing

We first evaluate sparse probing, which tests whether feature activations contain concept-relevant information. For each concept dataset, SAEBench ranks feature dimensions by how strongly their training-set activations separate positive from negative examples. It then trains a supervised linear probe using only the top-k ranked features.

- As a concrete example, consider an AG News class such as World. SAEBench treats examples from this class as positives and examples from the other classes as negatives. For each article, it runs the language model, collects residual-stream activations at the selected layer, masks special tokens, averages token-level feature activations into one vector for the whole text, and selects the k features whose training-set means best distinguish the positive class from the negatives. A linear probe is then trained on only these k features and evaluated on held-out articles.

For public SAE baselines, feature activations are obtained from the SAE encoder. For ICA and PCA, feature activations are the corresponding component scores. Since ICA and PCA scores are signed, we split each component into two nonnegative sides,

fi+ = max(si,0), fi− = max(−si,0),

where si is the signed component score. The probe can then select the positive and negative sides separately. We choose baselines to test four specific comparisons. Public SAEs, such as Gemma Scope and QwenScope, are the standard high-capacity reference because they use large overcomplete dictionaries trained with reconstruction and sparsity objectives (Gao et al., 2025, Lieberum et al., 2024, Deng et al., 2026). Matryoshka SAEs test whether reducing the effective SAE dictionary size makes SAEs closer to ICA under compact feature budgets (Bussmann et al., 2025). ITDA tests another training-light, non-SAE activation dictionary method (Leask et al., 2025). PCA tests whether non-Gaussianity is more useful than variance as a statistical signal for feature discovery (Abdi and Williams, 2010).

All methods are evaluated with the same sparse-probing datasets, feature budgets, and train/test splits. We average over two representative layers for each model: GPT-2 Small layers 6 and 10, Gemma 2 2B layers 12 and 20, and Qwen 3.5 2B Base layers 12 and 20. Since released Matryoshka SAE checkpoints are available for Gemma 2 2B, we compare them only on Gemma 2 2B layer 12.

###### 6.2. Targeted Probe Perturbation

Sparse probing tests whether ICA coordinates are predictive. We next ask whether they can support selective interventions. We evaluate this with SAEBench Targeted Probe Perturbation (TPP) (Karvonen et al., 2025),

ICA SAE ITDA PCA

###### GPT-2 Small

###### Gemma 2 2B

Qwen 3.5 2B Base

meanprobeaccuracy

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

0.9

0.8

0.7

1 2 5 10 20 50 100

1 2 5 10 20 50 100

1 2 5 10 20 50 100

top-k features used by probe

- Figure 10: SAEBench sparse-probing performance for ICA, public SAE, ITDA, and PCA representations. For each representation, SAEBench ranks features by class contrast on the training set and trains supervised probes using only the top-k ranked feature activations. Each curve reports test accuracy averaged over the eight default sparse-probing datasets and two evaluated layers for that model. SAE features are produced by the SAE encoder, while ICA and PCA use signed component scores split into two nonnegative sides.

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

1 2 5 10 20 50 100

top-k features used by probe

0.65

0.70

0.75

0.80

0.85

0.90

0.95

probeaccuracy

Gemma 2 2B layer 12

ICA

SAE

ITDA

SAE-Matryoshka-512 SAE-Matryoshka-128 PCA

- Figure 11: SAEBench sparse-probing comparison for Gemma 2 2B layer 12. We compare PCA, ICA, ITDA, prefix-restricted Matryoshka SAE variants, and the full Gemma Scope 16k SAE.

Figure 10 shows that ICA is a strong feature representation across all three model families. ICA remains competitive with public SAE dictionaries and consistently outperforms PCA and ITDA. Figure 11 further shows that compact Matryoshka variants with smaller dictionaries stay below ICA. These results show that:

- (1) ICA is competitive with public SAEs at concentrating concept-relevant information,
- (2) smaller SAE dictionaries do not close the gap by themselves, (3) ICA improves over the tested non-SAE alternative, and (4) non-Gaussianity is more useful than variance for sparse concept prediction.

an intervention-based feature-disentanglement metric.7 Since sparse probing shows that ICA and public SAEs are the strongest feature representations in our setting, we focus TPP on comparing ICA with public SAE dictionaries.

7We use TPP as a controlled, probe-based intervention diagnostic rather than as a standalone measure of feature quality. Recent work auditing SAEBench metrics cautions that TPP can be sensitive to benchmark settings and should not be over-interpreted in isolation (Chanin, 2026). We therefore interpret TPP together with sparse probing, human inspection, and ICA–SAE overlap analyses.

- As a concrete example, consider the same AG News setting. SAEBench first trains one-vs-rest probes for all classes. For a target class such as World, it identifies the top-N features most associated with that class, zero-ablates those features, reconstructs the modified activation, and measures how all class probes change. A good feature representation should reduce the target-class probe score more than the non-target probe scores.

To evaluate ICA under the same TPP protocol as SAE features, we use the same two-sign ICA wrapper as in sparse probing. After intervention, the wrapper reconstructs the signed ICA scores as si = fi+ − fi−, maps the modified score vector back through the fitted ICA inverse, and returns the modified activation to the model.

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

0.0

0.1

0.2

0.3

0.4

TPPtotal

GPT-2 Small

| | | |
|---|---|---|
| | | |
| | | |

2 20 500

0.0

0.2

0.4

TPPeffect

intended

| | | |
|---|---|---|
| | | |
| | | |

2 20 500

unintended

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

Gemma 2 2B

| | | |
|---|---|---|
| | | |
| | | |

2 20 500

intended

| | | |
|---|---|---|
| | | |
| | | |

2 20 500

unintended

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

Qwen 3.5 2B Base

| | | |
|---|---|---|
| | | |
| | | |

2 20 500

intended

| | | |
|---|---|---|
| | | |
| | | |

2 20 500

unintended

top-N ablated features

ICA total score

SAE total score

ICA intended/unintended

SAE intended/unintended

Figure 12: SAEBench Targeted Probe Perturbation (TPP) for GPT-2 Small, Gemma 2 2B, and Qwen 3.5 2B Base. The x-axis is the number of top-ranked features ablated for each target class. The top row reports the overall TPP score. The bottom row decomposes the same interventions into intended effects on the target-class probe and unintended effects on non-target probes. Scores are averaged over evaluated layers and the two TPP datasets, with shaded bands indicating interquartile ranges. ICA uses the two-sign SAE-compatible wrapper; SAE results use the public SAE dictionaries from the SAEBench comparison configs.

- Figure 12 shows that ICA is strongest relative to public SAEs at small-to-medium intervention budgets. Across all three model families, the intended effect of ICA rises quickly after ablating only a small number of features.

- At the same time, the unintended effect remains comparatively small. Therefore, when the goal is to inspect, edit, or intervene on a small set of task-relevant directions, a compact ICA basis can be advantageous. A small number of ICA components often captures enough class-relevant information to support selective interventions, without requiring the analyst to search through a large overcomplete dictionary.

- At larger N, public SAEs become more competitive and can become stronger. This is expected, as SAE dictionaries are overcomplete and contain many more features, so increasing N gives SAEs more capacity to accumulate class-relevant latents. For ICA, the same N consumes a much larger fraction of the available

compact basis, making broad disruption harder to avoid.

### 7. How ICA Relates to SAEs

Sparse autoencoders and ICA both expose directions in activation space, but they are designed around different objectives. SAEs learn large overcomplete dictionaries through reconstruction and sparsity losses. ICA instead returns a compact set of statistically non-Gaussian directions after normalization, centering, and whitening. Do they recover related structure, and if so, where do they differ?

###### 7.1. Directional Overlap

For each ICA component, we find the public SAE decoder direction in the same model and layer with the largest absolute cosine similarity. This gives a simple measure of whether an ICA direction has a close counterpart in the SAE dictionary.

###### GPT-2 Small

###### Gemma 2 2B

###### Qwen 3.5 2B Base

8

ICAcomponents(%)

6

4

2

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

nearest absolute SAE cosine

- Figure 13: Nearest-SAE overlap for ICA components. For each ICA component, we report the maximum absolute cosine with any public SAE decoder direction in the same layer. Across all three models, most components lie in a moderate-overlap range, with both weakly matched and strongly matched tails. The distributions show partial agreement between ICA and SAE rather than a one-to-one correspondence.

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0 1 2 3 4 5 6 7 8 9 10 11

0.0

0.2

0.4

0.6

nearestabsoluteSAEcosine

GPT-2 Small

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 5 10 15 20 25

Gemma 2 2B

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 5 10 15 20 23

Qwen 3.5 2B Base

layer

- Figure 14: Layer-wise nearest-SAE overlap. Each point shows the median maximum absolute cosine between ICA components and public SAE decoder directions in one layer. Shaded bands show interquartile ranges. The median overlap remains moderate across depth, showing that partial ICA-SAE alignment is a consistent property of the decompositions rather than an artifact of a few layers.

ICA Label Abs-top context Nearest SAE ∣cos∣ SAE act. SAE rank GPT-2 low L0/C395 Water, Removed ...im and walk into the kitchen. I’ve already

F10102 0.10 0 inactive GPT-2 mid L0/C192 After ...f the tibial tray may remain exposed above

pulled a bottle of water

F10573 0.44 31.9 1 GPT-2 high L10/C142 Conditional repetition string "You’re pretty good.$"

the tibial plateau after

F29658 0.75 46.1 14

Route110_Text_16EA2A:: @ 816EA2

Gemma low L6/C216 71 in No. _ ) ASBCA N°’ 60315 ) ) Under Contract No. HTC71

F9142 0.09 0 inactive Gemma mid L12/C466 Accusation of bad behavior ...nger the lives of our law enforcement officers

F2168 0.43 9.87 33 Gemma high L24/C510 Female-centered narrative ...he referee, Carlos Ramos have wronged Ser-

to promote a radical

F9501 0.73 118 3

ena Williams and took from

Qwen low L1/C834 Certain name 6, 1994. Decided July 8, 1994. Rear F12249 0.08 0 inactive Qwen mid L18/C1794 ? ...ained from the patient included in the study,

F3134 0.42 15.1 1 Qwen high L22/C511 ? ...trigger, here’s a post on how to determine

and the institutional

F8570 0.68 4.88 3

whether they scrolled up

- Table 6: Sampled ICA components and their nearest public SAE features. Rows are drawn from the preselected random-component audit set, choosing the lowest, median, and highest nearest-SAE cosine example for each model. The context column shows the ICA top-absolute-score example ending at the target token. SAE activation and rank measure whether the nearest SAE feature activates on that same target-token activation.

- Figure 13 shows broad nearest-neighbor distributions across all three models. Many ICA components have a nontrivial SAE neighbor, so ICA is not unrelated to SAE. At the same time, many components are only weakly matched by any single SAE direction, so ICA is not merely a compact copy of public SAE dictionaries.
- Figure 14 shows that this pattern is stable across depth.

We also check functional agreement. For each ICA component, we take its top-activation context and test whether the nearest SAE feature also activates on the same target-token activation. Table 6 shows concrete examples. The “SAE act.” and “SAE rank” columns measure whether the nearest SAE feature fires on the ICA component’s own top context. Low-overlap examples usually have inactive SAE neighbors, while mid- and high-overlap examples often activate their nearest SAE feature at the same target token. This suggests that high cosine is not only direction-level similarity; it often corresponds to similar activation behavior on real examples. These results provide evidence that high-overlap ICA components and SAE features can recover related interpretable structure.

A useful byproduct of this comparison is that it also lets us audit public SAE labels. We find that some public Neuronpedia labels are incomplete or overly narrow, and may not fully describe the feature’s behavior.8 For example, GPT-2 L10/C142 is labeled by our audit as conditional repetition, while its nearest SAE feature F29658 is labeled on Neuronpedia as numerical values and symbols in mathematical or programming contexts. However, the feature also fires on non-code contexts that match the ICA component’s repeated or template-like behavior, suggesting that the public label captures only part of the feature’s operational pattern. Similarly, Gemma L24/C510 is labeled by our audit as female-centered narrative, while Neuronpedia labels its nearest SAE feature F9501 as positive affirmations and encouragement. In controlled prompts, F9501 activates most strongly for “She’s passionate about books, travel, beauty, and all things cheese.” (99.5), less strongly for the male-substituted version (77.0), and still less for “He’s passionate about cars, football, and

8We refer to the public feature labels hosted on Neuronpedia, an online platform for browsing and annotating SAE features: https://www.neuronpedia.org/. Since these labels may be updated over time, we treat them as useful public hypotheses rather than fixed ground truth.

video games.” (40.75). This behavior is closer to a gendered narrative direction than to encouragement alone. From this perspective, ICA-SAE overlap can help cross-check and refine feature interpretations. It also highlights a limitation of current automated interpretability labels: many labels should be treated as hypotheses rather than ground truth.

###### 7.2. Activation Pattern Comparison

Direction overlap can reveal whether an ICA component has a nearby SAE feature, but it does not show whether their activations follow the same token-level pattern. To compare token-wise behavior, we use the same sentence for both decompositions. For GPT-2 Small, we select the top-1 ICA component by absolute score and the top-1 SAE feature by activation at each token. We then plot the union of the selected directions across the full sentence. Each row is normalized by its own maximum value in the sentence.

###### ICA scores

SAE activations

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

F31538 F2491 F9643

C335 C621

C52 C173

F29308

F6136 F21524 F27737

C67 C598 C207

F1138 F14010 F24609

C22 C761 C421 C735 C273 C313 C161 C222 C617 C323 C345 C318

F2312 F31794 F11659 F10552 F10422

F7259 F7974

F20832 F23575

F1151 F12706 F30658

C89 C141

F158 F28577

Maystoppeda atthebankbeforethetripwaiting, inlinedepositto checkawithdrawandenoughcashforweekendthe .

Maystoppeda atthebankbeforethetripwaiting, inlinedepositto checkawithdrawandenoughcashforweekendthe .

- Figure 15: Token-wise ICA and SAE responses on the same GPT-2 Small sentence. At each token, we select the strongest ICA component by absolute score and the strongest SAE feature by activation, then plot the union of selected directions across the sentence. Each row is normalized by its maximum value to show within-direction response shape. SAE features mostly appear as localized activations, while ICA directions often vary across short spans, including the financial context around bank, deposit, check, withdraw, and cash. Sentence: “Maya stopped at the bank before the trip, waiting in line to deposit a check and withdraw enough cash for the weekend.”

- Figure 15 shows a clear difference in activation patterns. On the SAE side, selected features often peak at one token or a short span. This is consistent with the sparsity pressure in SAE training. On the ICA side, several directions vary more smoothly across neighboring tokens. In this example, ICA directions track the financial interpretation of bank across related words such as deposit, check, withdraw, and cash. This matches the behavior observed in our case studies, where ICA components often behave less like isolated event detectors and more like compact contextual factors. Full examples for Gemma 2 2B and Qwen 3.5 2B Base are provided in Appendix E.

### 8. Conclusion

We introduced ICALens, a practical workflow for using ICA as a stable, efficient, and auditable lens on LLM representations. Rather than training another overcomplete dictionary, ICALens normalizes activations, fits a compact GPU-parallel FastICA basis, and supports inspection, annotation, and evaluation of signed non-Gaussian directions. Across GPT-2 Small, Gemma 2 2B, and Qwen 3.5 2B Base, this workflow recovers human-interpretable components, links non-Gaussianity to context dependence through ERF, and yields useful feature coordinates under sparse probing and targeted probe perturbation.

Our results position ICA as a compact first lens for language-model activations, rather than as a replacement for sparse autoencoders. SAEs learn large overcomplete dictionaries optimized for sparse reconstruction, making them well suited for high-resolution feature discovery. ICALens asks how much interpretable structure is already visible from activation geometry, helping analysts decide where heavier dictionary learning is worth the cost.

### 9. Limitations and Future Directions

Compact ICA bases. The main algorithmic boundary is that standard FastICA is compact. It returns at most d components in a d-dimensional fitting space, so it cannot provide the large overcomplete vocabulary that modern SAEs are designed to learn. When a difficult layer is accepted with fewer than d components, the reading map remains useful for inspection and probing, while intervention-style edits rely on a pseudoinverse reconstruction. A natural next step is to explore higher-capacity ICA variants, including overcomplete ICA, preconditioned or adaptive FastICA, deflationary ICA, JADE, Infomax, extended Infomax, and heavy-tailaware objectives. These methods may recover additional stable directions while keeping the objective simpler than full SAE training (Podosinnikova et al., 2019, Ablin et al., 2018, Miettinen et al., 2018, Cardoso and Souloumiac, 1993, Bell and Sejnowski, 1995, Lee et al., 1999, Spurek et al., 2018).

From states to transformations. Our experiments focus on residual-stream states, which reveal what information is present at a layer. A natural next step is to analyze what a layer changes. ICA could be fit to MLP outputs, attention outputs, residual updates, or shared bases across multiple layers. Such transformationlevel analyses would provide a lightweight counterpart to model-wide SAE releases and related work on component-level tools for studying model computation (Lieberum et al., 2024, Deng et al., 2026).

Actionable analysis and annotation. Our main goal in this paper is to make ICA useful for human understanding: to help analysts quickly find, inspect, and test compact directions in model activations. Steering and automatic annotation are promising downstream directions. Because ICA is cheap to refit, it may be especially useful for discovering task- or distribution-specific directions across different datasets, which is valuable for actionable mechanistic interpretability. Future steering applications should be evaluated against strong task-specific baselines (Wu et al., 2025, Kantamneni et al., 2025). Future automatic annotation systems could also build on the evidence used in our workflow—ERF, top examples, opposite-side examples, trace plots, prompt tests, and human audits—to produce more grounded component labels and turn automated descriptions into testable hypotheses (Paulo et al., 2024).

We will maintain an updated list of follow-up projects and open questions on the our website.9

9https://liusida.github.io/ica-lens-paper/future-projects.html

### References

Hervé Abdi and Lynne J Williams. Principal component analysis. Wiley interdisciplinary reviews: computational statistics, 2(4):433–459, 2010.

Pierre Ablin, Jean-François Cardoso, and Alexandre Gramfort. Faster independent component analysis by preconditioning with hessian approximations. IEEE Transactions on Signal Processing, 66(15):4040–4049,

2018. doi: 10.1109/TSP.2018.2844203.

Anthony J. Bell and Terrence J. Sejnowski. An information-maximization approach to blind separation and blind deconvolution. Neural Computation, 7(6):1129–1159, 1995. doi: 10.1162/neco.1995.7.6.1129.

Anthony J Bell and Terrence J Sejnowski. The “independent components” of natural scenes are edge filters. Vision research, 37(23):3327–3338, 1997.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. https://transformer-circuits.pub/2023/monosemantic-features/index.html.

Bart Bussmann, Patrick Leask, and Neel Nanda. Batchtopk sparse autoencoders. arXiv preprint arXiv:2412.06410, 2024.

Bart Bussmann, Noa Nabeshima, Adam Karvonen, and Neel Nanda. Learning multi-level features with matryoshka sparse autoencoders. arXiv preprint arXiv:2503.17547, 2025.

Jean-François Cardoso and Antoine Souloumiac. Blind beamforming for non-gaussian signals. IEE Proceedings

F: Radar and Signal Processing, 140(6):362–370, 1993. doi: 10.1049/ip-f-2.1993.0054. David Chanin. Are sparse autoencoder benchmarks reliable? arXiv preprint arXiv:2605.18229, 2026. Ingrid Daubechies, E Roussos, S Takerkart, M Benharrosh, C Golden, K D’ardenne, W Richter, Jonathan D

Cohen, and J Haxby. Independent component analysis for brain fmri does not select for independence. Proceedings of the National Academy of Sciences, 106(26):10415–10422, 2009.

Boyi Deng, Xu Wang, Yaoning Wang, Yu Wan, Yubo Ma, Baosong Yang, Haoran Wei, Jialong Tang, Huan Lin, Ruize Gao, et al. Qwen-scope: Turning sparse features into development tools for large language models. arXiv preprint arXiv:2605.11887, 2026.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale. Advances in neural information processing systems, 35:30318–30332, 2022.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1(1):12, 2021.

Leo Gao, Tom Dupre la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. In International Conference on Learning Representations, volume 2025, pages 26721–26754, 2025.

Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas. Finding neurons in a haystack: Case studies with sparse probing. arXiv preprint arXiv:2305.01610, 2023.

Feijiang Han, Xiaodong Yu, Jianheng Tang, Qingyun Zeng, Licheng Guo, and Lyle Ungar. Zerotuning: Unlocking the initial token’s power to enhance large language models without training. In ICML 2025 Workshop on Methods and Opportunities at Small Scale, 2025. URL https://openreview.net/forum? id=THSbsRWy9v.

Robert Huben, Hoagy Cunningham, Logan Smith, Aidan Ewart, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. In International Conference on Learning Representations, volume 2024, pages 7827–7845, 2024.

Aapo Hyvarinen. Fast and robust fixed-point algorithms for independent component analysis. IEEE transactions on Neural Networks, 10(3):626–634, 1999.

Aapo Hyvarinen and Erkki Oja. Independent component analysis: algorithms and applications. Neural networks, 13(4-5):411–430, 2000.

Subhash Kantamneni, Joshua Engels, Senthooran Rajamanoharan, Max Tegmark, and Neel Nanda. Are sparse autoencoders useful? A case study in sparse probing. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, 2025. URL https://arxiv.org/abs/2502.16681.

Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Bloom, David Chanin, Yeu-Tong Lau, Eoin Farrell, Callum McDougall, Kola Ayonrinde, et al. Saebench: A comprehensive benchmark for sparse autoencoders in language model interpretability. arXiv preprint arXiv:2503.09532, 2025.

Patrick Leask, Neel Nanda, and Noura Al Moubayed. Inference-time decomposition of activations (itda): A scalable approach to interpreting large language models. arXiv preprint arXiv:2505.17769, 2025.

Te-Won Lee, Mark Girolami, and Terrence J. Sejnowski. Independent component analysis using an extended infomax algorithm for mixed sub-gaussian and super-gaussian sources. Neural Computation, 11(2): 417–441, 1999. doi: 10.1162/089976699300016719.

Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, and Neel Nanda. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 278–300, 2024.

Aleksandar Makelov, Georg Lange, and Neel Nanda. Towards principled evaluations of sparse autoencoders for interpretability and control. In International Conference on Learning Representations, volume 2025, pages 33588–33636, 2025.

Jari Miettinen, Klaus Nordhausen, and Sara Taskinen. fica: Fastica algorithms and their improved variants. The R Journal, 10(2):148–158, 2018. doi: 10.32614/RJ-2018-046.

Anish Mudide, Josh Engels, Eric Michaud, Max Tegmark, and Christian Schroeder de Witt. Efficient dictionary learning with switch sparse autoencoders. In International Conference on Learning Representations, volume 2025, pages 101830–101844, 2025.

Tomáš Musil and David Mareček. Exploring interpretability of independent components of word embeddings with automated word intruder test. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, editors, Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 6922–6928, Torino, Italia, May 2024. ELRA and ICCL. URL https://aclanthology.org/2024.lrec-main.605/.

Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. Distill, 5(3):e00024–001, 2020.

Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering llama 2 via contrastive activation addition. In Annual Meeting of the Association for Computational Linguistics, 2024.

Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. arXiv preprint arXiv:2311.03658, 2023.

Gonçalo Paulo, Alex Mallen, Caden Juang, and Nora Belrose. Automatically interpreting millions of features in large language models. arXiv preprint arXiv:2410.13928, 2024.

Anastasia Podosinnikova, Amelia Perry, Alexander S Wein, Francis Bach, Alexandre d’Aspremont, and David Sontag. Overcomplete independent component analysis via sdp. In The 22nd international conference on artificial intelligence and statistics, pages 2583–2592. PMLR, 2019.

Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Tom Lieberum, Vikrant Varma, János Kramár, Rohin Shah, and Neel Nanda. Improving dictionary learning with gated sparse autoencoders. arXiv preprint arXiv:2404.16014, 2024a.

Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, János Kramár, and Neel Nanda. Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders. arXiv preprint arXiv:2407.14435, 2024b.

Przemyslaw Spurek, Pawel Rola, Jacek Tabor, and Adam Czechowski. Ica based on split generalized gaussian. arXiv preprint arXiv:1802.05550, 2018.

Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. Massive activations in large language models. arXiv preprint arXiv:2402.17762, 2024.

Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. Transformer Circuits Thread, 2024. URL https://transformer-circuits.

#### pub/2024/scaling-monosemanticity/index.html.

Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, and Christopher Potts. AxBench: Steering LLMs? even simple baselines outperform sparse autoencoders. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, 2025. URL https://openreview.net/forum?id=K2CckZjNy0.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In International Conference on Learning Representations, 2024.

Hiroaki Yamagiwa, Momose Oyama, and Hidetoshi Shimodaira. Discovering universal geometry in embeddings with ica. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4647–4675, 2023.

Hiroaki Yamagiwa, Yusuke Takase, and Hidetoshi Shimodaira. Axis tour: Word tour determines the order of axes in ica-transformed embeddings. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 477–506, 2024.

Zejia You, Chunyuan Deng, and Hanjie Chen. Spherical steering: Geometry-aware activation rotation for language models. arXiv preprint arXiv:2602.08169, 2026.

### A. Additional FastICA Fitting Diagnostics

100

| | | | | | |
|---|---|---|---|---|---|
|Row-normalized<br><br>Raw max p95<br><br>| | | | | |
| | | | | | |

10−1

10−2

mLmtstatstc

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

(a) Layer 0

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

10−1

10−2

mLmtstatstc

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

FastICA iteration

(g) Layer 6

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

10−1

10−2

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

(b) Layer 1

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

10−1

10−2

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

FastICA iteration

(h) Layer 7

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

10−1

10−2

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

(c) Layer 2

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

10−1

10−2

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

FastICA iteration

(i) Layer 8

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

10−1

10−2

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

(d) Layer 3

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

10−1

10−2

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

FastICA iteration

(j) Layer 9

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

10−1

10−2

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

(e) Layer 4

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

10−1

10−2

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

FastICA iteration

(k) Layer 10

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

10−1

10−2

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

(f) Layer 5

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

10−1

10−2

10−3

10−4

10−5

10−6

10−7

10−8

0 50 100 150 200 250 300

FastICA iteration

(l) Layer 11

- Figure 16: Layer-wise convergence diagnostics for FastICA on GPT-2 Small using 1k fitting rows.

| | | | | | |
|---|---|---|---|---|---|
|Row-normalized<br><br>Raw<br><br>max p95<br><br>| | | | | |
| | | | | | |

0 50 100 150 200 250 300

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

mLmtstatstc

(a) Layer 0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

(b) Layer 1

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

(c) Layer 2

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

(d) Layer 3

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

(e) Layer 4

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

(f) Layer 5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

FastICA iteration

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

mLmtstatstc

(g) Layer 6

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

FastICA iteration

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

(h) Layer 7

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

FastICA iteration

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

(i) Layer 8

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

FastICA iteration

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

(j) Layer 9

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

FastICA iteration

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

(k) Layer 10

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

FastICA iteration

10−5

10−4

10−3

10−2

10−1

100

(l) Layer 11

- Figure 17: Layer-wise convergence diagnostics for FastICA on GPT-2 Small using 100k fitting rows.

| | | | | | |
|---|---|---|---|---|---|
|Row-normalized<br><br>Raw<br><br>max p95<br><br>| | | | | |
| | | | | | |

0 50 100 150 200 250 300

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

mLmtstatstc

(a) Layer 0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

(b) Layer 1

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

(c) Layer 2

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

(d) Layer 3

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

(e) Layer 4

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

(f) Layer 5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

FastICA iteration

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

100

mLmtstatstc

(g) Layer 6

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

FastICA iteration

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

(h) Layer 7

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

FastICA iteration

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

(i) Layer 8

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

FastICA iteration

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

(j) Layer 9

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

FastICA iteration

10−8

10−7

10−6

10−5

10−4

10−3

10−2

10−1

(k) Layer 10

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150 200 250 300

FastICA iteration

10−5

10−4

10−3

10−2

10−1

100

(l) Layer 11

- Figure 18: Layer-wise convergence diagnostics for FastICA on GPT-2 Small using 1M fitting rows.

p05-p95 median max 10−4

GPT-2 Small

Gemma 2 2B

Qwen 3.5 2B Base

| |L0, 768 comps| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| |L0, 288 comps| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| |L0, 512 comps| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

10−1

10−3

10−5

10−7

| |L2, 768 comps| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| |L5, 1152 comps| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| |L5, 2048 comps| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

10−1

10−3

10−5

10−7

| |L4, 768 comps| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| |L10, 576 comps| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| |L9, 1024 comps| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

10−1

10−3

10−5

FastICAlim

10−7

| |L7, 768 comps| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| |L15, 576 comps| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| |L14, 1024 comps| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

10−1

10−3

10−5

10−7

| |L9, 768 comps| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| |L20, 576 comps| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| |L18, 2048 comps| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

10−1

10−3

10−5

10−7

| |L11, 192 comps| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| |L25, 1152 comps| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| |L23, 2048 comps| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

10−1

10−3

10−5

10−7

1 100 200

1 100 200

1 100 200

FastICA iteration

- Figure 19: Representative FastICA convergence curves across model families. Each panel shows component-wise FastICA limit values over iterations for one fitted layer. Solid colored lines show medians, shaded regions show the 5th–95th percentile interval, dashed black lines show the maximum, and dotted horizontal lines mark the 10−4 convergence threshold.

### B. Additional Human Interpretation Results

###### layer F1 financial R1 river edge F2 financial R2 river edge

|W Bank W Common Word W Money W Hotel W Cook|
|---|

|W Bank W Common Word W Money W Hotel W Cook|
|---|

|W Bank W Common Word W Money W Hotel W Cook|
|---|

|W Bank W Common Word W Money W Hotel W Cook|
|---|

emb

|W Government, Dep... O Early Document... W Legal Court<br><br>? ?|
|---|

|W Government, Dep... W Water<br><br>? W Number W Legal Court|
|---|

|W Government, Dep... W Water W Legal Court<br><br>?<br><br>F Letter B|
|---|

|W Government, Dep... W Water W Legal Court<br><br>?<br><br>F Letter B|
|---|

L0

|F Letter B W Car, Train O Early Document... W Money<br><br>?|
|---|

|P Noun Following... W Group<br><br>? F Letter B W Area|
|---|

|F Letter B<br><br>? W Group P Follows "a" W Any|
|---|

|F Letter B ? unlabeled P Follows "a" W Court|
|---|

L3

|W Financial P Facility W Blood, Heart F Letter B S Body Movement|
|---|

|W Water S Body Movement P Facility P Name<br><br>?|
|---|

|P Facility W Financial F Letter B W Company, Busine... P Senate, House,...|
|---|

|? P Facility W Database, SQL F Letter B W Financial|
|---|

L6

|O Early Document...<br><br>? F Letter B W $ Dollar, £ Pou...<br><br>?|
|---|

|S Location S Body Movement W Number 1 W Any W Type, Kind|
|---|

|F Letter B<br><br>? ? W Any ?|
|---|

|? ?<br><br>W Sport, Health<br><br>?<br><br>F Letter B|
|---|

L9

|O Early Document...<br>P Follows "the" F Letter B O Early Document... W World<br>|
|---|

|G Story Telling W Place F Letter B W Forceful<br><br>?|
|---|

|F Letter B<br><br>?<br><br>X Repeat and Cont...<br><br>?<br><br>P Follows "the"|
|---|

|S Repetition S Repetition W Retailer F Letter B W Health, Sports|
|---|

L12

|P Organization P Follows "the" O Early Document...<br><br>?<br><br>W Internet|
|---|

|? S Location S Body Movement<br><br>? ?|
|---|

|P Organization<br><br>? P Follows "a" W Internet W Safe, Clean|
|---|

|P Follows "a" unlabeled<br><br>W Safe, Clean ?<br>X Conditional Rep...<br>|
|---|

L15

|W Location W Department<br><br>? O Early Document... S Building|
|---|

|W Location S Rural Area S Body Movement W Abstract Noun<br><br>?|
|---|

|W Department W One of Many<br><br>? W Field F Letter B|
|---|

|F Letter B W </ P Noun with attri... W Art, Health, Sp... W { }|
|---|

L18

|W Place W Public Announce... S Change or Rever... S Money G Sentence about...|
|---|

|W Location G Story Telling W Place G Poetic W Body Part|
|---|

|W Art, Fashion W Place S One-person desc... S Authorization,... W Police|
|---|

|S Route, Path S Economy, Finance<br><br>? W Art, Fashion P Follows "a"|
|---|

L21

|G Misfortune W Noun S Body Movement O Early Document... S Building|
|---|

|S Body Movement W Rural Area<br><br>?<br><br>W Physical Things F Past Participle|
|---|

|P Describing a pe... W Noun W Noun<br><br>?<br><br>G Political|
|---|

|W Noun P Follows "a" W Compare<br><br>?<br><br>S Body Movement|
|---|

L24

confidence: high medium low unclear type: F form W word P phrase S sentence L long-range context G global O position X sophisticated

- Figure 20: Contextual decomposition of a polysemous word in Gemma 2 2B.

###### layer F1 financial R1 river edge F2 financial R2 river edge

|W Bank W Simulation W Common Words W Money F Letter B|
|---|

|W Bank W Simulation W Common Words W Money F Letter B|
|---|

|W Bank W Simulation W Common Words W Money F Letter B|
|---|

|W Bank W Simulation W Common Words W Money F Letter B|
|---|

emb

|F Letter B P Role or Follows... W Punctuation, St... F Letter K W Legal Court|
|---|

|F Letter B F Letter K W Cost, Money W University, Sch... W \[|
|---|

|F Letter B F Letter K W Company, Busine... W Cost, Money W Legal Court|
|---|

|F Letter B F Letter K W Company, Busine... W Legal Court W Turn, Around|
|---|

L0

|W Bank W Place O Early Document...<br><br>? W Sit|
|---|

|W Bank W Sit W Land, Ground W Mountain, Hill W Side|
|---|

|W Bank W Place W Money, Financial F Letter B<br><br>?|
|---|

|W Bank W Place S Repetition S Repetition F Letter B|
|---|

L3

|W Building, Hall O Early Document... W Tax in Military<br><br>?<br><br>W Password, Authe...|
|---|

|S Body Movement S Meadow, Woods W Building, Hall W Surface, Skin W Ocean, Sea|
|---|

|P Follows "a" W Building, Hall F Letter B<br><br>?<br><br>W Tax in Military|
|---|

|S Repetition W Building, Hall P Follows "a" W Money, Fund F Letter B|
|---|

L6

|O Early Document... O Early Document...<br><br>? W Chair, Room S Stock, Investim...|
|---|

|W Nature, Meadow W Chair, Room<br><br>?<br><br>S Body Movement W Water, Liquid|
|---|

|P Follows "a" F Letter B W Occupation, Job W Industry, Health<br><br>?|
|---|

|S Repetition P Follows "a" S Repetition W Company, Busine...<br><br>?|
|---|

L9

|G Story Telling S Talk to, Write... W Place G Math Word Probl... S Main Topic Item|
|---|

|G Story Telling P Meadow, Woods P Location W Place W Water, River|
|---|

|G Story Telling X Conditional Rep... X Conditional Rep... F Letter B<br><br>?|
|---|

|L Repetition G Story Telling X Homonym<br><br>?<br><br>X Conditional Rep...|
|---|

L12

|S Indicate Change G Story Telling P Travel, Move, C... W Room, Office F Letter B|
|---|

|G Story Telling W Land, Floor, Su... S Body Movement W Relax, Enjoy W Wait, Rest|
|---|

|W Room, Office G Story Telling F Letter B<br>X Conditional Rep... X Conditional Rep...<br>|
|---|

|G Story Telling F Letter B W Room, Office W And, Or W Legal Court|
|---|

L15

|S Checking Bank A... S Location G Story Telling S Shop, Store F Letter B|
|---|

|G Story Telling S Location S Body Movement W Sit, Stand S Beautiful, Enjo...|
|---|

|S Checking Bank A... G Story Telling F Letter B F Letter K X One of choices|
|---|

|S Checking Bank A... G Story Telling F Letter B F Letter K S Explaining a te...|
|---|

L18

|G Story Telling S Payment, Credit...<br><br>? F Letter B S Body Movement|
|---|

|G Story Telling W Sit, Stand S Body Movement W Natural location S Intention|
|---|

|G Story Telling S Payment, Credit... W Team, Group P Place, Hotel F Letter B|
|---|

|G Story Telling S Payment, Credit... F Letter B W Team, Group W Thing|
|---|

L21

confidence: high medium low unclear type: F form W word P phrase S sentence L long-range context G global O position X sophisticated

- Figure 21: Contextual decomposition of a polysemous word in Qwen 3.5 2B Base.

|king<br><br>W King, Royal P Occupations<br><br>? ?<br><br>W Find|
|---|

|man<br><br>W Man, Woman P Occupations W Most Common Words<br><br>? ?|
|---|

|father<br><br>W Parent, Child W Friend, Partner W Parent, Maker P Occupations W Man, Woman|
|---|

|brother<br><br>W Friend, Partner W Parent, Child W Stupid, Disliked W Man, Woman P David, Michael,...|
|---|

|actor<br><br>W Film, Movie P Occupations W Alternative, Ex... W Activity, Action<br><br>?|
|---|

|queen<br><br>W King, Royal W Female Names<br><br>?<br><br>P Occupations<br><br>?|
|---|

|woman<br><br>W Female Names W Man, Woman P Occupations W Sexual W Young, Teen|
|---|

|mother<br><br>W Parent, Child W Friend, Partner W Female Names P CPU, Motherboar... P Occupations|
|---|

|sister<br><br>W Friend, Partner W Female Names W Parent, Child W Initially, Part... F Letter S|
|---|

|actress<br><br>W Film, Movie W Female Names P Occupations W Alternative, Ex...<br><br>?|
|---|

confidence: high medium low unclear type: F form W word P phrase S sentence L long-range context G global O position X sophisticated

- Figure 22: Embedding-layer ICA components for familiar analogy word sets in GPT-2 Small.

|king<br><br>W King, Royal F Letter K W Common Word W Leader, Head<br><br>?|
|---|

|man<br><br>W Man, Human W Male<br><br>? W Common Word W People, Person|
|---|

|father<br><br>W Parent, Father,... W Male W Common Word W Son, Daughter W Brother, Sister|
|---|

|brother<br><br>W Brother, Sister W Friend W Son, Daughter W Parent, Father,... W Male|
|---|

|actor<br><br>W Act W Researcher, Jou... W Player W Theater<br><br>?|
|---|

|queen<br><br>W King, Royal F Letter Q W Woman, Female W Female Name<br><br>?|
|---|

|woman<br><br>W Woman, Female W Man, Human W People, Person W Common Word W Researcher, Jou...|
|---|

|mother<br><br>W Parent, Father,...<br><br>? W Woman, Female W Common Word<br><br>?|
|---|

|sister<br><br>W Brother, Sister W Friend W Parent, Father,... W Son, Daughter W Woman, Female|
|---|

|actress<br><br>W Act W Researcher, Jou... W Theater W Player W Woman, Female|
|---|

confidence: high medium low unclear type: F form W word P phrase S sentence L long-range context G global O position X sophisticated

- Figure 23: Embedding-layer ICA components for familiar analogy word sets in Gemma 2 2B.

### C. Complete Random Annotation Tables

- Table 7: Manual labels for the random ICA component audit. Rows are the selected components from random_components_n50_seed0. For each component, the table reports only the sign corresponding to its strongest absolute-score example, matching the annotation side most visible in the random-component audit. ERF is the mean effective receptive field measured by the short-context probe.

Layer/Component Confidence ERF Type Label Excess kurtosis GPT-2 Small

L0/C6 high 1 Word Arrest, Conviction 46 L0/C192 high 1 Word After 298 L0/C267 high 1 Word Any 194 L0/C366 high 1 Word " 92 L0/C395 high 1 Word Water, Removed 6.7 L1/C11 high 1 Word Be, Been 31 L1/C97 high 1 Word Also, Therefore 15 L1/C109 high 1 Word \t, \r 23

- L2/C302 high 1 Word ] 101

- L2/C303 high 1 Form Letter B 79 L2/C563 high 1 Word En 80 L3/C257 high 1 Form Letter Z 49 L3/C291 high 1 Word Seem, Appear 29 L3/C330 high 1 Word Very 129 L3/C413 high 1 Phrase Inverted Is, Are, ... 64 L3/C449 high 1 Form Letter U 50 L5/C15 high 1 Word Social Media 26 L5/C696 high 1 Word Building, Place 15 L6/C392 high 1 Word Semicolon 92 L7/C140 high 1 Word Dead, Killed 14 L7/C260 high 1 Word With 81 L7/C492 high 1 Word Look, Seem 33 L7/C766 high 1 Word If 134 L8/C552 high 1 Word Out, Down, Up 61 L9/C463 high 1 Position Early Doc Position 11 L9/C621 high 1 Form Mod-, Sub- 4.3 L10/C144 high 1 Form Letter Z 24 L10/C375 high 1 Word /, )/ 15 L10/C576 high 1 Word Try, Hard 9.6 L5/C11 high 1.4 Sentence Sentence start 9.4 L9/C204 high 1.5 Phrase Than, Comparison 30 L7/C254 high 1.5 Phrase Thus, Therefore starting a clause 11 L4/C298 high 1.8 Phrase Ongoing -ing 18 L6/C290 high 1.8 Global Geopolitical conflicts 7.8 L4/C8 high 1.9 Form D- 8.6 L2/C152 high 2 Phrase Code whitespace 19 L7/C17 high 2 Sentence Scientific research, Citation 33 L10/C74 high 2 Form XML tag name 81 L5/C585 high 2.1 Phrase i.e., Mean 12 L6/C164 high 2.1 Sentence Citation 17 L8/C736 high 2.3 Phrase Right, Permission, License to 3.0 L9/C118 high 3.1 Sentence If-condition 7.9 L10/C368 high 3.2 Global Gaming language 31 L9/C445 high 5.3 Long-Range Context Refer to a spokesperson 24 L3/C294 medium 1 Word Happy, Able 13 L6/C246 medium 2.2 Phrase ‘in’, ‘the’, or ‘to’ following ‘tell’ or ‘say’ 9.7 L10/C146 medium 5.4 Global Natural area 11 L8/C738 medium 7.7 Long-Range Context Repetition of a prior xxx: section header 23 L10/C142 low 8.3 Sentence Conditional repetition 18 L2/C648 unclear 11+ ? -1.1

###### Gemma 2 2B

emb/C1964 high 1 Word kick 13 emb/C2034 high 1 Word Smiley emoji 89

L0/C9 high 1 Word Have 163 L1/C82 high 1 Word =" in code 124 L2/C302 high 1 Word Does, Is, Will 134 L2/C341 high 1 Phrase Two leading whitespaces 91 L2/C708 high 1 Word Array 25 L2/C890 high 1 Word Data 234 L5/C404 high 1 Word Non-breaking space, U+00A0 25 L7/C351 high 1 Word Do, Did 80 L16/C495 high 1 Word Get 4.4 L19/C171 high 1 Form Letter X 60

- L20/C79 high 1 Word Change 13

- L20/C183 high 1 Word Out 24 L5/C602 high 1 Word Move 14 L9/C442 high 1 Word Based, In response, Depend 7.1

- L21/C399 high 1.1 Phrase Adj. prefer after “be” 17 L7/C530 high 1.1 Phrase Heart, Liver, Lung 27 L9/C79 high 1.1 Word Director, President 21 L24/C355 high 1.2 Global Turkish language 28

- L21/C575 high 1.2 Phrase Opening parenthesis in code 6.8

- L15/C65 high 1.3 Phrase Anatomical location 8.6 L10/C311 high 1.4 Phrase Be the, Be a 23 L19/C210 high 1.6 Form Letter V 12

- L16/C527 high 1.8 Phrase Family name 35 L18/C19 high 2.2 Sentence Gas, Fluid 6.5 L6/C541 high 2.2 Phrase What, How question 18 L16/C206 high 2.2 Global Low-level programming language, C, ASM, ... 13 L6/C677 high 3.5 Sentence If-condition 17 L9/C516 high 4.2 Sentence Polynomial, Quadratic expression 2.6 L20/C263 high 4.4 Global Cryptography, Computer Security 5.4

- L22/C218 high 4.4 Sentence Persistent deficiency despite expected improvement 5.9

- L12/C466 high 4.6 Global Accusation of bad behavior 4.3

- L24/C372 high 5.2 Global Immigration policy discussion 6.3

- L18/C206 high 5.7 Long-Range Context Same line ending in code 31 L8/C303 high 6 Sentence At/In/During + time, something happened/changed 1.2

- L19/C543 high 6.2 Sentence Expecting additional parallel elements 29

- L24/C510 high 6.6 Global Female-centered narrative 10

- L25/C854 high 7.5 Phrase Citation marker 94

L23/C567 high 7.7 Global UK civic info 6.8 L13/C30 high 8.6 Sophisticated Refer back to one of two recently introduced alternatives 9.8 L10/C261 high 9.2 Long-Range Context Repetition and number increases 35 L6/C775 high 11+ Long-Range Context Repetition of prefix 19 L7/C106 medium 1.1 Sentence Beginning of a new paragraph 7.7

- L20/C101 medium 1.6 Phrase Number >10 40

- L25/C92 medium 2.9 Global Core issue in news 16 L6/C216 low 3 Phrase 71 in No. 0.5 L5/C453 low 3.5 Form To-infinitive 0.4 emb/C59 unclear 1.7 ? 0.2

- L18/C33 unclear 11+ Long-Range Context Apostrophe with unknown pattern 75

###### Qwen 3.5 2B Base

emb/C930 high 1 Word Where 107 emb/C984 high 1 Word Sort 24 L0/C446 high 1 Word Control 43 L1/C694 high 1 Word Google, Big Tech Company 164 L1/C917 high 1 Word Comma + a/the 152 L1/C1096 high 1 Word Down 191 L1/C1604 high 1 Word As (Capitalized) 186 L2/C511 high 1 Word Shoot 138 L4/C1591 high 1 Word Advertise 26 L7/C730 high 1 Position Post-title article start 12 L7/C1009 high 1 Word Without 35 L13/C1634 high 1 Word Create, Develop 51 L16/C1572 high 1 Word Only 150 L16/C1811 high 1 Form Word fragment -e 24

###### L19/C808 high 1 Word Chinese atmospheric literary narrative character 17

- L21/C519 high 1 Word > 49 L3/C508 high 1.1 Word Register 14 L2/C1940 high 1.1 Phrase End 268 L6/C647 high 1.3 Phrase Last as in "last year/week" 44

L20/C189 high 1.3 Phrase Total number, Amount 17 L13/C3 high 1.3 Phrase Sensitive identity 14 L4/C1899 high 1.9 Word Money, Fund 95 L13/C392 high 1.9 Sentence Legal document format 56 L8/C522 high 2 Phrase Is, Was, Has 32 L19/C824 high 2 Phrase Follow, violate rules 23 L19/C999 high 2.4 Phrase Math quantity, Math object 15 L11/C1211 high 2.4 Phrase Because of, As a result of 13 L3/C77 high 2.5 Phrase I feel I, You feel you, Feel like, ... 5.9

- L13/C577 high 2.5 Sentence "In" used in software context 10 L18/C280 high 3.1 Global Reproductive health 20

- L18/C1794 high 3.5 Global Study protocol 46

- L20/C970 high 4 Sentence Motor, Electromechanical device 15

- L18/C745 high 4.3 Global Object in mechanical patent 21 L15/C258 high 4.4 Phrase Age of a person or group 30

L22/C511 high 4.9 Sentence Either or, Whether or, One way or another 5.0 L21/C145 high 4.9 Phrase Citation (pandoc-style with dashes inside the ID) 29 L14/C707 high 5.3 Phrase Legal case citation 16 L8/C189 high 5.3 Sentence Number in elementary arithmetic 25 L9/C47 high 5.6 Sentence Contrast, Opposition 12

- L19/C526 high 8 Global Javascript i18n code 17 L7/C671 medium 6.1 Phrase Closing bracket in code 20 L1/C834 low 3.6 Phrase Certain name 1.0 L13/C1679 low 4 Sentence Context where medical condition or biological process hap-

pens

7.4 L6/C441 low 11+ Long-Range Context Beginning of a repeated labeled/enumerated entry after

intervening context

3.4

L23/C1250 low 11+ Sentence Memorized string in code snippet 184 L7/C406 unclear 4.6 ? 0.5 L23/C541 unclear 5.6 ? 1.2 L5/C126 unclear 7.4 ? 0.5 L3/C1037 unclear 7.5 ? 0.5

- L19/C646 unclear 10.3 ? 11

### D. Secondary Contrastive Label Audit

- Table 8: Secondary audit summary for the high-confidence audited components from the random ICA component annotation study. Rows are drawn from random_components_n50_seed0. Score is the 0–10 secondary audit score.

Layer/Component Initial label Score Suggested label

GPT-2 Small L0/C6 Arrest, Conviction 9 arrest / conviction / guilty legal context L0/C192 After 10 after token L0/C267 Any 10 any token L0/C366 " 8 double quote token L0/C395 Water, Removed 8 water token L1/C11 Be, Been 9 be / been token L1/C97 Also, Therefore 8 also / therefore discourse connective L1/C109 \t, \r 10 tab / carriage-return token

- L2/C302 ] 10 closing square bracket token

- L2/C303 Letter B 10 B / b token L2/C563 En 8 en subword token L3/C257 Letter Z 10 Z / z token L3/C291 Seem, Appear 9 seem / appear token L3/C330 Very 10 very token L3/C413 Inverted Is, Are, ... 8 inverted/question auxiliary L3/C449 Letter U 10 U / u token L5/C15 Social Media 9 social-media platform/context L5/C696 Building, Place 8 building / apartment place token L6/C392 Semicolon 10 semicolon token L7/C140 Dead, Killed 9 died / killed / dead token L7/C260 With 10 with token L7/C492 Look, Seem 9 look / seem / sound evidential verb L7/C766 If 10 if token L8/C552 Out, Down, Up 9 out / down / up particle L9/C463 Early Doc Position 8 early-document opening subtoken L9/C621 Mod-, Sub- 8 mod/sub prefix token L10/C144 Letter Z 10 Z / z token L10/C375 /, )/ 8 slash token in URL/path/markup L10/C576 Try, Hard 8 try / trying / tries token L5/C11 Sentence start 9 sentence-initial capitalized token L9/C204 Than, Comparison 10 than comparison token L7/C254 Thus, Therefore starting a clause 9 thus / therefore consequence connective L4/C298 Ongoing -ing 9 progressive/ongoing -ing form L6/C290 Geopolitical conflicts 9 geopolitical conflict/diplomacy context L4/C8 D- 9 D-initial subword/token L2/C152 Code whitespace 7 code indentation/alignment whitespace L7/C17 Scientific research, Citation 9 scientific reference-title final period L10/C74 XML tag name 10 HTML/XML closing tag name L5/C585 i.e., Mean 8 i.e. / that is to say / mean L6/C164 Citation 8 reference-list title-final period L8/C736 Right, Permission, License to 9 permission/right/license to L9/C118 If-condition 8 then/comma after if-condition L10/C368 Gaming language 9 Video-game discourse L9/C445 Refer to a spokesperson 8 spokesperson/spokesman name reference

Gemma 2 2B emb/C1964 kick 10 Kick / kicks token emb/C2034 Smiley emoji 7 Emoji / emoticon token L0/C9 Have 8 have / has / had auxiliary L1/C82 =" in code 8 Quoted attribute assignment in markup L2/C302 Does, Is, Will 10 WH-question auxiliary does/is/will L2/C341 Two leading whitespaces 10 Two-space indentation token

L2/C708 Array 9 array token L2/C890 Data 10 data token L5/C404 Non-breaking space, U+00A0 10 Non-breaking space (U+00A0) L7/C351 Do, Did 10 do / did auxiliary L16/C495 Get 8 get / got / gotten L19/C171 Letter X 10 X / x token

- L20/C79 Change 9 change / changes / changed

- L20/C183 Out 9 out token L5/C602 Move 8 move / moved / moving L9/C442 Based, In response, Depend 9 based / in response / depend construction

- L21/C399 Adj. prefer after “be” 7 predicate adjective after be L7/C530 Heart, Liver, Lung 7 cardiac / cardiovascular and organ terms L9/C79 Director, President 6 institutional role title L24/C355 Turkish language 10 Turkish-language token/subword

- L21/C575 Opening parenthesis in code 8 opening call/index delimiter in code

- L15/C65 Anatomical location 9 anatomical term/subword L10/C311 Be the, Be a 9 article after be L19/C210 Letter V 10 V / v token

- L16/C527 Family name 10 surname after given name/title L18/C19 Gas, Fluid 8 gas/fluid flow medium L6/C541 What, How question 9 What/How question auxiliary L16/C206 Low-level programming language, C, ASM, ... 8 low-level systems-code token L6/C677 If-condition 9 If-clause predicate after if this/it/that L9/C516 Polynomial, Quadratic expression 8 minus sign in polynomial/algebra expression L20/C263 Cryptography, Computer Security 8 cryptography / security context

- L22/C218 Persistent deficiency despite expected improvement

7 still/remain poor despite expectation

- L12/C466 Accusation of bad behavior 8 accusatory / blame discourse L24/C372 Immigration policy discussion 8 immigration-policy news discourse

- L18/C206 Same line ending in code 7 repeated code/list line-ending pattern L8/C303 At/In/During + time, something hap-

pened/changed

7 temporal transition after time expression

- L19/C543 Expecting additional parallel elements 7 parallel/list element expectation

- L24/C510 Female-centered narrative 8 female-character narrative context

- L25/C854 Citation marker 6 pandoc-style citation marker digit

L23/C567 UK civic info 7 UK public-service/civic information prose

- L13/C30 Refer back to one of two recently introduced alternatives

9 reference to one member of a recent pair

L10/C261 Repetition and number increases 8 repeated identifier with increasing numeric suffix L6/C775 Repetition of prefix 8 repeated identifier prefix

Qwen 3.5 2B Base emb/C930 Where 10 where / Where token emb/C984 Sort 10 sort / Sort token L0/C446 Control 10 control token L1/C694 Google, Big Tech Company 9 Google token / big-tech context L1/C917 Comma + a/the 9 article after comma appositive L1/C1096 Down 10 down token L1/C1604 As (Capitalized) 10 capitalized As token L2/C511 Shoot 10 shot / shoot token L4/C1591 Advertise 9 advertising / ads token L7/C730 Post-title article start 9 article-start token after scientific title L7/C1009 Without 10 without token L13/C1634 Create, Develop 9 create/develop verb L16/C1572 Only 10 only token L16/C1811 Word fragment -e 9 word-final e fragment L19/C808 Chinese atmospheric literary narrative character 8 Chinese literary/poetic atmosphere character

- L21/C519 > 8 literal > in markup/code L3/C508 Register 10 register token

L2/C1940 End 10 end / End token L6/C647 Last as in "last year/week" 9 last in temporal expression L20/C189 Total number, Amount 9 aggregate count/amount expression L13/C3 Sensitive identity 9 sensitive identity term L4/C1899 Money, Fund 9 money / funds token L13/C392 Legal document format 8 legal-document header formatting L8/C522 Is, Was, Has 5 is/was after discourse setup L19/C824 Follow, violate rules 10 rule compliance/violation language L19/C999 Math quantity, Math object 8 scientific/mathematical quantity term L11/C1211 Because of, As a result of 9 causal preposition phrase L3/C77 I feel I, You feel you, Feel like, ... 8 pronoun after feel/felt like

- L13/C577 "In" used in software context 8 in within code/software context L18/C280 Reproductive health 7 reproductive/adolescent health context L18/C1794 Study protocol 10 clinical study protocol/approval language

- L20/C970 Motor, Electromechanical device 9 electromechanical motor/device context

- L18/C745 Object in mechanical patent 9 mechanical patent object term L15/C258 Age of a person or group 9 age number for person/group

L22/C511 Either or, Whether or, One way or another 9 alternative/concessive construction L21/C145 Citation (pandoc-style with dashes inside the ID) 9 dash inside Pandoc citation/key L14/C707 Legal case citation 7 legal case citation whitespace L8/C189 Number in elementary arithmetic 9 digit answer/context in elementary arithmetic L9/C47 Contrast, Opposition 8 contrast continuation token

- L19/C526 Javascript i18n code 7 JavaScript/JSON locale object

### E. Token-Wise Activation Patterns in Gemma and Qwen

###### ICA scores

SAE activations

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

C56 C41

F1741

C415 C430 C126

F11087 F7357 F1178 F6810 F8287 F5277 F5940 F6336

C65 C462

C8 C141 C275 C167 C190 C107 C550 C418

F11634 F5080

C64 C206 C422 C150 C426 C534 C477 C122 C447 C254 C270 C144 C156

F12358 F9492 F4342

F15300 F13479 F15281

F7529 F2583

F16320 F8318 F3375 F8947 F3539 F9283

C7 C137 C547 C181

C4

C85 C205 C402

F10336 F304

<bos>Mayastoppedatthebankbeforethetripwaiting, inlinedepositto checkawithdrawandenoughcashforweekendthe .

<bos>Mayastoppedatthebankbeforethetripwaiting, inlinedepositto checkawithdrawandenoughcashforweekendthe .

- Figure 24: Token-wise ICA and SAE patterns for the same Gemma 2 2B sentence. For each token in the bank sentence, we select the top-2 ICA components by absolute score and the top-2 SAE features by activation, then plot the union of selected directions across all token positions.

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

Maystoppeda atthebankbeforethetripwaiting, inlinedepositto checkawithdrawandenoughcashforweekendthe .

C305

C1778 C504 C488 C186 C633

C1285 C1317

C713 C354

C39

C271 C1740 C1581

C408 C1196

C24 C642 C926

C1054

C638 C1781 C2032 C1644

C772 C258

ICA scores

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

Maystoppeda atthebankbeforethetripwaiting, inlinedepositto checkawithdrawandenoughcashforweekendthe .

F7850

F17391 F1019 F5680

F14546 F26121 F12677 F21663 F18862

F7648 F3074

F25374 F21789 F10849 F32494 F25488 F23581 F30981 F16565 F22498

F5787 F18345 F24251

SAE activations

- Figure 25: Token-wise ICA and SAE patterns for the same Qwen 3.5 2B Base sentence. For each token in the bank sentence, we select the top-2 ICA components by absolute score and the top-2 SAE features by activation, then plot the union of selected directions across all token positions.

### F. Explorer Interface Screenshots

[Figure 6]

[Figure 7]

- Figure 26: Explorer interface screenshots for Gemma 2 2B layer 12. Two screenshots are ICA Explorer and SAE Explorer.

[Figure 8]

[Figure 9]

###### Figure 27: Explorer interface screenshots for Qwen 3.5 2B Base layer 12.

