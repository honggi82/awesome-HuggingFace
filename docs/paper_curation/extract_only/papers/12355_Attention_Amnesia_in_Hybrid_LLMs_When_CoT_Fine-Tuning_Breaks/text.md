## arXiv:2606.11052v1[cs.CL]9Jun2026

[Figure 1]

# Attention Amnesia in Hybrid LLMs: When CoT Fine-Tuning Breaks Long-Range Recall, and How to Fix It

###### Xinyu Zhou∗1, Boyu Zhu∗2, Yi Xu3, Zhiwei Li1, Yingfa Chen4, Huiming Wang†5 and Zhijiang Guo†1,6 1LARK, HKUST(GZ), 2UCL, 3Mistral AI, 4Tsinghua University, 5SUTD, 6HKUST

∗Equal Contribution †Corresponding Author

Github Page: https://github.com/LARK-AI-Lab/QK-Restore

### Abstract

Chain-of-thought (CoT) supervised fine-tuning (SFT) is widely adopted to improve reasoning ability, yet we find that it systematically degrades long-context recall in hybrid linear-attention models. Across architectures including HypeNet and Jet-Nemotron, retrieval performance on Needle-In-A-Haystack (NIAH) deteriorates substantially after CoT-SFT, and the degradation becomes more severe under harder retrieval settings and longer context windows. For example, HypeNet-9B on NIAH-S2@256K decreases from 67.2% to 9.4%. We attribute this to CoT-SFT biasing attention gradients toward short-range patterns, disrupting query-key projections (𝑾𝑄,𝑾𝐾) that are responsible for long-range routing. Motivated by this observation, we propose QK-Restore, a training-free method that restores only 𝑾𝑄 and 𝑾𝐾 from the pre-SFT checkpoint while preserving all other post-SFT parameters. We further introduce a Procrustes variant to balance routing preservation and reasoning adaptation. Across architectures, QKRestore consistently restores long-context capability at zero training cost while preserving reasoning performance; for instance, on HypeNet-5B it improves S3@256K from 65.4% to 76.4% while maintaining strong reasoning performance.

### 1. Introduction

Efficient sequence models, such as linear attention (Katharopoulos et al., 2020; Qin et al., 2024b; Yang et al., 2024) and state-space models (Dao and Gu, 2024; Gu and Dao, 2024), have emerged as attractive alternatives to softmax-attention (Vaswani et al., 2023) for long-context processing, reducing the quadratic cost to linear complexity through compact recurrent or structured states. However, this compression introduces an information bottleneck on recall-intensive tasks that are crucial for long-context understanding (Zhang et al., 2024), such as Needle-in-a-Haystack (NIAH; Hsieh et al. 2024). Hybrid linear-attention models mitigate this tradeoff by retaining a small set of softmax-attention layers for global recall while converting the remaining layers into efficient linearattention layers (NVIDIA et al., 2025). Although native hybrid models such as Qwen3.5 (Qwen Team, 2026) and Kimi Linear (Team et al., 2025) achieve strong performance and efficiency, pretraining them from scratch remains highly resource-intensive. This cost motivates recent work that converts pretrained softmax-attention into hybrid linear-attention models through distillation, using only a small fraction of pretraining tokens (Chen et al., 2026; Hoshino et al., 2025; Li et al., 2025b). Existing conversion work primarily focuses on preserving pretraining capabilities during architectural conversion, while the impact of downstream reasoning post-training on long-context retrieval remains underexplored.

Corresponding author(s): Huiming Wang (huiming_wang@mymail.sutd.edu.sg), Zhijiang Guo (zhijiangguo@hkust-gz.edu.cn) 1

[Figure 2]

- Figure 1 | NIAH retrieval accuracy before and after CoT-SFT on HypeNet-2B. Despite improving reasoning ability, CoT-SFT substantially degrades long-context retrieval accuracy in distilled hybrid models, especially at longer context lengths and harder NIAH settings.

While distilled hybrid models primarily serve as efficient base models, modern LLMs typically rely on post-training to acquire stronger instruction-following and reasoning abilities (Guo et al., 2025; Ouyang et al., 2022; Qwen et al., 2025). In particular, CoT-SFT is widely used to enhance mathematical and multi-step reasoning(Li et al., 2025c; Wei et al., 2023). We observe that applying CoT-SFT on mathematical reasoning data improves reasoning performance, but can substantially degrade the long-context recall ability that distilled hybrid models acquire during pretraining or architectural conversion, particularly in more challenging NIAH settings and extended contexts length (Figure 1). Unlike generic catastrophic forgetting, this degradation is highly structured: it primarily affects long-range retrieval behavior mediated by the retained softmax-attention layers, while leaving the intended reasoning gains largely intact. This reveals a fundamental tension in hybrid models: CoTSFT strengthens local reasoning can simultaneously disrupt the routing mechanisms required for long-range retrieval.

To investigate this phenomenon, we model CoT reasoning traces as a latent Markov process over intermediate reasoning states, reflecting the local step-to-step structure of mathematical derivations (Prystawski et al., 2023; Wang et al., 2026). Under this CoT-Markov assumption, we derive that the expected gradient magnitude on attention logits decays exponentially with token distance. We empirically validate this prediction by measuring token autocorrelation and gradients of attentionlogit. These results suggest that CoT-SFT can improve local multi-step reasoning while eroding the long-range routing behavior needed for retrieval.

To better isolate the source of recall degradation, we analyze the retained softmax-attention layers through a routing-extraction decomposition. We find that CoT-SFT induces locality-biased drift in the query-key projections 𝑾𝑄,𝑾𝐾, which determine the source of information retrieval, whereas value-side extraction can still benefit from post-SFT adaptation. Since long-context recall in distilled hybrid models depends heavily on a small number of retained softmax-attention layers, such query-key drift can disproportionately disrupt retrieval. Motivated by this asymmetry, we propose QK-Restore, a training-free method that restores only 𝑾𝑄 and 𝑾𝐾 in these layers from the pre-SFT checkpoint while preserving all other post-SFT parameters, thereby recovering long-range routing while retaining most reasoning improvements from CoT-SFT. Our contributions can be summarized as:

- • We identify CoT-SFT-induced recall degradation as a structured post-training failure mode of distilled hybrid models.
- • We provide a theoretical and empirical analysis showing that CoT-SFT concentrates training signals on local token interactions and induces locality-biased drift in the query-key routing geometry of retained softmax-attention layers.
- • We introduce QK-Restore, a training-free method that restores only query and key projections,

[Figure 3]

- Figure 2 | Overview of QK-Restore. After CoT-SFT, the retained softmax-attention layers in a hybrid model may lose long-range routing ability. QK-Restore restores only the query and key projections

(𝑾𝑄,𝑾𝐾) from the pre-SFT checkpoint while keeping value-side and other post-SFT parameters unchanged, recovering long-context recall while preserving reasoning performance.

recovering long-context recall while largely preserving reasoning performance.

### 2. Related Work

Efficient Long-Context Models and Hybrid Attention. Recent hybrid attention models interleave softmax-attention layers with efficient recurrent or linear-attention layers, achieving competitive performance on both commonsense reasoning and recall-intensive long-context tasks while improving inference efficiency (NVIDIA et al., 2025; Qwen Team, 2026; Team et al., 2025). Pretraining strong hybrid models from scratch at large scale remains prohibitively expensive, which motivates recent efforts to obtain hybrid models through Transformer-to-hybrid conversion or distillation (Chen et al., 2026; Li et al., 2025b). However, existing work primarily focus to obtaining high-performing converted hybrid checkpoints, while their behavior during subsequent post-training remains underexplored.

Distilling Transformers into Hybrid Models. Transformer-to-hybrid distillation converts selected softmax-attention layers into linear or recurrent mixers, where layer selection critically influences long-context retrieval performance (Chen et al., 2026; Goldstein et al., 2026; Gu et al., 2026; Li et al., 2025b). However, strong recall performance after conversion does not necessarily imply stability after reasoning-oriented post-training; specifically, across multiple distilled hybrid models with strong long-context retrieval, we observe that CoT-SFT can substantially degrade recall capabilities, revealing a critical failure mode that remains unaddressed by existing distillation methods.

CoT Dynamics and Attention Routing. Recent work has studied CoT reasoning as a structured generation process over intermediate reasoning states (Prystawski et al., 2023; Wang et al., 2026). A Markovian view formalizes this structure by modeling reasoning as local transitions between latent states, suggesting that CoT supervision primarily reinforces short-range step-to-step dependencies (Wang et al., 2026). In contrast to prior work that utilizes this locality to explain the efficacy of CoT in reasoning, we connect it to a failure mode in efficient long-context models: locality-biased CoT-SFT can erode long-range attention routing required for recall. This perspective aligns with analyses of attention as an information-routing mechanism, where query-key interactions define routing logits that select positions, while value projections determine the content extracted from those positions (Vaswani et al., 2023).

### 3. Background and Problem Setup

##### 3.1. Hybrid Model

In this work, we primarily focus on the hybrid model. To retain the expressive power of softmaxattention while improving long-context efficiency, hybrid architectures interleave a small set of softmax-attention layers with efficient linear-attention layers (Chen et al., 2026; Yang et al., 2025).

Softmax-Attention Layers For Transformer’s layer ℓ, head ℎ ∈ {1, . . . , 𝐻}, the per-head projection matrices 𝑾𝑄(ℓ,ℎ),𝑾𝐾(ℓ,ℎ),𝑾𝑉(ℓ,ℎ) ∈ ℝ𝑑

ℎ×𝑑 (𝑑ℎ = 𝑑/𝐻) transform hidden state 𝒉𝑡(ℓ) ∈ ℝ𝑑:

𝒒𝑡(ℓ,ℎ) = 𝑾𝑄(ℓ,ℎ)𝒉𝑡(ℓ), 𝒌𝑠(ℓ,ℎ) = 𝑾𝐾(ℓ,ℎ)𝒉𝑠(ℓ), 𝒗𝑠(ℓ,ℎ) = 𝑾𝑉(ℓ,ℎ)𝒉𝑠(ℓ),

(1)

with scalar logit and attention weight

exp 𝑒𝑡𝑠(ℓ,ℎ) 𝑢≤𝑡 exp 𝑒𝑡𝑢(ℓ,ℎ)

𝑒𝑡𝑠(ℓ,ℎ) = (𝒒𝑡(ℓ,ℎ))⊤𝒌𝑠(ℓ,ℎ)

. (2)

, 𝐴𝑡𝑠(ℓ,ℎ) =

√𝑑ℎ

The per-head output vector and layer output are

𝒗¯𝑡 = ∑︁ 𝑠≤𝑡

𝐴𝑡𝑠 𝒗𝑠, 𝒐𝑡(ℓ) = 𝑾𝑂 Concatℎ 𝒗 ¯𝑡(ℎ) , (3)

Linear-Attention Layers For layer ℓ, a recurrent state matrix 𝑺𝑡 ∈ ℝ𝑑

𝑘×𝑑𝑣 evolves via

𝑺𝑡(ℓ) = Φ𝑡(ℓ) 𝑺𝑡(−ℓ)1, 𝒉𝑡(ℓ−1) , 𝒐𝑡(ℓ) = Ψ𝑡(ℓ) 𝑺𝑡(ℓ), 𝒉𝑡(ℓ−1) , (4)

where Φ𝑡, Ψ𝑡 are layer-specific update and readout operators. Taking Lightning Attention (Qin et al., 2024a) as an example, the state update and readout are:

𝑺𝑡(ℓ) = diag(𝜆𝑡(ℓ)) 𝑺𝑡(−ℓ)1 + 𝒌𝑡(ℓ) 𝒗𝑡(ℓ) ⊤, (5) 𝒐𝑡(ℓ) = 𝑺𝑡(ℓ) ⊤𝒒𝑡(ℓ). (6)

We denote the set of softmax-attention layers as 𝐿attn. Empirical studies suggest that long-range recall in hybrid models depends disproportionately on a set of softmax-attention layers, whereas most remaining layers can be replaced by recurrent mechanisms with minimal degradation (Chen et al., 2026; Jelassi et al., 2024; Wang et al., 2025). Thus, preserving the routing behavior of 𝐿attn is critical for long-context recall.

##### 3.2. Attention Routing

√𝑑ℎ, which determines how strongly position 𝑡 attends to position 𝑠. Since the routing pattern is determined solely by 𝑾𝑄,𝑾𝐾, changes to these matrices directly alter long-range retrieval behavior. To understand how CoT-SFT training affects retrieval behavior, we examine the gradients w.r.t QK metrics. The gradient update of 𝑾𝑄 is:

We operationalize attention-routing through the pre-softmax logit 𝑒𝑡𝑠 = 𝒒⊤𝑡 𝒌𝑠/

∑︁

1 √𝑑ℎ

𝜕L 𝜕𝑒𝑡𝑠

𝒌𝑠 𝒉⊤𝑡 . (7)

∇𝑾𝑄L =

𝑡,𝑠

If 𝜕L/𝜕𝑒𝑡𝑠 is large for small distance 𝜏 = 𝑡−𝑠 and negligible for large 𝜏, gradient descent systematically pushes 𝑾𝑄,𝑾𝐾 toward local patterns regardless of context length. Furthermore, we have:

𝜕L 𝜕𝑒𝑡𝑠

= 𝐴𝑡𝑠 · 𝑮⊤𝑡 (𝒗𝑠 − 𝒗¯𝑡), (8) where 𝑮𝑡 = 𝜕L/𝜕𝒐𝑡 ∈ ℝ𝑑

ℎ is the gradient vector. So understanding the distance dependence of 𝜕L/𝜕𝑒𝑡𝑠 becomes central to understanding how CoT-SFT affects retrieval.

### 4. Why CoT-SFT Disrupts Routing?

##### 4.1. CoT Data Assumption

We first characterize CoT data structure via a latent Markov model (Prystawski et al., 2023; Wang et al., 2026), which motivates the gradient analysis that follows.

Assumption 4.1 (CoT-Markov structure). There exist latent reasoning states 𝑧1, . . . , 𝑧𝑇 ∈ Z, |Z| = 𝐾 < ∞, such that:

- 1. Markov transitions: 𝑃(𝑧𝑡 | 𝑧𝑡−1, 𝑧𝑡−2, . . .) = 𝑃(𝑧𝑡 | 𝑧𝑡−1).
- 2. Observation model: 𝑃(𝑥𝑡 | 𝑧𝑡, 𝑥𝑡−1) = 𝑃(𝑥𝑡 | 𝑧𝑡). The token at position 𝑡 is determined by the current reasoning state.
- 3. Ergodicity: The chain is irreducible and aperiodic with stationary distribution 𝜋, 𝜋min := min𝑧∈Z 𝜋(𝑧) > 0.
- 4. Reversibility: Detailed balance holds: 𝜋(𝑧) 𝑃(𝑧, 𝑧′) = 𝜋(𝑧′) 𝑃(𝑧′, 𝑧) for all 𝑧, 𝑧′ ∈ Z.
- 5. Spectral gap: The transition matrix has second-largest eigenvalue magnitude 𝜌 ∈ (0, 1). We include the discussion of the validity of this assumption in Appendix A.

##### 4.2. Gradient Locality Theory

Softmax-attention layers ℓ ∈ 𝐿attn are the sole locus of long-range recall in a hybrid model. Therefore, we analyze how CoT-SFT erodes their routing capacity in these layers through gradient locality.

- 4.2.1. Definition and Assumptions Definition 4.2 (Distance-conditioned gradient magnitude).

𝜕L 𝜕𝑒𝑡, 𝑡(ℓ,ℎ−)𝜏

𝑔(𝜏) := 𝔼𝑡,ℓ∈𝐿attn, ℎ

. (9)

- Assumption 4.3 (Norm bounds). There exist constants 𝐵𝐺, 𝐵𝑉 > 0 such that, almost surely over the training distribution,

∥𝑮𝑡∥2 ≤ 𝐵𝐺 and ∥𝒗𝑠∥2 ≤ 𝐵𝑉. (10)

- Assumption 4.4 (Score-function identity). Let 𝑮𝑡 = 𝜕L/𝜕𝒐𝑡 denote the gradient of the loss with respect to the per-head output at position 𝑡. At approximate optimality under cross-entropy, the model satisfies, for all 𝑡 and almost surely over 𝑥1:𝑇,

𝔼 𝑮𝑡 𝑥1:𝑡 = 0. (11) The justifications for the above assumptions are included in Appendix B and Appendix C.

##### 4.2.2. Gradient Locality Theorem

Lemma 4.5 (Spectral Correlation Decay). For a stationary reversible ergodic Markov chain with spectral gap 1 − 𝜌, and any mean-zero 𝑓, 𝑔 ∈ 𝐿2(𝜋):

𝔼[ 𝑓 (𝑧𝑡) 𝑔(𝑧𝑠)] ≤ 𝜌𝑡−𝑠 ∥ 𝑓 ∥𝐿2(𝜋) ∥𝑔∥𝐿2(𝜋), (12)

Theorem 4.6 (Gradient locality). Let 𝐵𝐺, 𝐵𝑉 > 0 be the constants from Assumption 4.3, and let 𝜌 ∈ (0, 1) be the spectral radius from Assumption 4.1. Under Assumptions 4.3–4.4 and the CoT-Markov model (Assumption 4.1), for all 𝜏 ≥ 0:

|𝑔(𝜏) ≤ 𝐶′ · 𝑒−𝜏/𝑊,|
|---|

(13) where

𝐶′ := 2𝐵𝐺𝐵𝑉 > 0, 𝑊 := −1 log 𝜌

> 0. (14)

The constant 𝐶′ depends only on the model’s norm bounds; the decay rate 𝑊 is set entirely by the spectral gap of the latent transition matrix. The full proof is included in Appendix F.

##### 4.3. Empirical Validation

- Theorem 4.6 shows that CoT training preferentially reinforces nearby token interactions, causing long-range routing signals to decay with distance. The theoretical chain rests on two empirical claims about data and model:

1. Lemma 4.5’s hypothesis: the latent Markov chain governing token generation has a spectral gap

- 1 − 𝜌. However, since the latent states 𝑧𝑡 are unobservable, we instead consider the observed token sequence and use the token autocorrelation 𝜌¯(𝜏) as a proxy for 𝜌. Concretely, 𝜌¯(𝜏) measures how much more likely a token is to reappear at distance 𝜏 than by chance, averaged over all token types weighted by their stationary frequency 𝜋(𝑣). As proved in Appendix G, 𝜌¯(𝜏) ≤ 𝐶 · 𝜌𝜏 under the hidden markov model.
- 2. Theorem 4.6’s prediction: the expected gradient magnitude 𝔼[|𝜕L/𝜕𝑒𝑡𝑠|] decays exponentially in 𝜏, with a smaller effective window for CoT data than for general text. We measure 𝔼[|𝜕L/𝜕𝑒𝑡𝑠|] at each distance 𝜏 directly from the model:

𝜕L 𝜕𝑒𝑡𝑠

= 𝐴𝑡𝑠

𝜕L 𝜕𝐴𝑡𝑠 − ∑︁

𝑠′

𝐴𝑡𝑠′

𝜕L 𝜕𝐴𝑡𝑠′

. (15)

- 3. Curve Fitting. Both 𝜌¯(𝜏) and 𝑔(𝜏) take the form 𝐶 · 𝑒−𝜏/𝑊, where 𝑊 = −1/log 𝜌′ serves as a natural scale of the decay. Therefore, we fit the model 𝑦(𝜏) = 𝐶 · 𝜌𝜏 = 𝐶 ·𝑒−𝜏/𝑊 via log-linear regression. Taking logarithms linearizes the model:

log 𝑦(𝜏) = log𝐶 −

1

𝑊

𝜏. (16)

- 4. Interpretation. As shown in Figure 3, the left subfigure presents the token autocorrelation decay

parameter 𝑊corr, a corpus-level statistic that measures the characteristic distance over which text retains structured, non-random self-similarity. Because mathematical reasoning repeatedly references the same symbols, variables, and formula fragments, CoT text sustains high self-similarity over longer distances (𝑊corr ≈ 172 tok) than general prose (𝑊corr ≈ 115 tok). The right subfigure shows the attention gradient decay parameter 𝑊grad, which characterises the training dynamics by measuring the

[Figure 4]

- Figure 3 | Empirical validation for token autocorrelation decay and attention gradient decay on HypeNet-2B.

effective reach of the gradient signal 𝔼[|𝜕L/𝜕𝑒𝑡𝑠|]. Both decay curves are well described by exponential fits, consistent with the thoerem’s prediction.

When 𝑊grad < 𝑊corr, there exists a band of distances 𝑑 ∈ (𝑊grad, ,𝑊corr] where the data demands long-range attention but the training gradient no longer reinforces it. The discussions about the attention gradient decay also observed in the pre-training stage are included Appendix H.

Key Takeaway-1

CoT-SFT improves short-range reasoning ability in hybrid models, at the cost of long-range retrieval.

### 5. Method

- Theorem 4.6 has established that the gradient on the attention logit 𝑒𝑡𝑠 decays geometrically in distance 𝜏 = 𝑡 − 𝑠. Now we investigate which weight metrics this decay propagates into.

5.1. Routing–Extraction Gradient Decoupling

- Theorem 5.1 (Routing-Extraction Gradient Decoupling). Under CoT-SFT, where the token-generation process is modeled as a stationary reversible ergodic Markov chain with a spectral gap 1 − 𝜌 > 0, gradient updates to the parameters of softmax-attention layer exhibit the following asymmetry:

[Routing] The per-pair contribution to ∇𝑾𝑄

from position pairs at distance 𝜏 = 𝑡 − 𝑠 satisfies:

|𝔼<br><br>𝜕L 𝜕𝑒𝑡𝑠<br><br>𝒌𝑠 𝒉⊤𝑡<br><br>𝐹<br><br>≤ 𝐶𝑅 · 𝜌𝜏|
|---|

for constant 𝐶𝑅 = 𝐶′𝐵𝐾𝐵ℎ > 0. The same bound holds for ∇𝑾𝐾

.

[Extraction] The gradient on the value vector at any position 𝑠 satisfies:

|𝔼<br><br>𝜕L 𝜕𝒗𝑠 ≥ 𝛿𝐴 · 𝑐𝐺 > 0,<br><br>|
|---|

Algorithm 1 QK-Restore Input: Hybrid model’s pre-SFT checkpoint 𝜽pre; Hybrid model’s post-SFT checkpoint 𝜽post; softmaxattention layers 𝐿attn Output: Repaired model 𝜽rep # Initialized from post-SFT weights 𝜽rep ← 𝜽post for ℓ ∈ 𝐿attn do

# Replace with pre-SFT weights 𝑾𝑄ℓ,rep ← 𝑾𝑄ℓ,pre, 𝑾𝐾ℓ,rep ← 𝑾𝐾ℓ,pre

end for return 𝜽rep

where 𝛿𝐴 > 0 satisfies 𝐴𝑡𝑡 ≥ 𝛿𝐴 for all 𝑡 and 𝑐𝐺 > 0 satisfies 𝔼[∥𝑮𝑠∥] ≥ 𝑐𝐺 for all 𝑠.

The detailed proof is in Appendix I. Therefore, CoT-SFT affects routing and knowledge extraction differently. 𝑾𝑄,𝑾𝐾 receive meaningful gradient signal only from short-range pairs, while the gradient reaching 𝑾𝑉 is bounded below uniformly over all positions, independent of context length.

##### 5.2. QK-Restore

Theorem 5.1 shows that CoT-SFT affects the two functional components of a softmax-attention layer in fundamentally different ways.

The routing parameters 𝑾𝑄 and 𝑾𝐾, which control where the model attends, receive gradient signal that decays as 𝜌𝜏 with distance: they are shaped exclusively by short-range pairs and progressively lose long-range routing capacity. The extraction parameters 𝑾𝑉 and 𝑾𝑂, which control what is retrieved, receive a gradient uniformly bounded below for every position: they accumulate value-processing improvements.

In other words, CoT-SFT simultaneously corrupts 𝑾𝑄,𝑾𝐾 and improves 𝑾𝑉,𝑾𝑂. These effects are segregated into disjoint parameter sets. We propose QK-Restore , which eliminates routing corruption and preserves extraction improvement by transplanting 𝑾𝑄,𝑾𝐾 from the pre-SFT checkpoint while retaining post-SFT 𝑾𝑉 and 𝑾𝑂 (Algorithm 1).

6. Experiments

##### 6.1. Setup

Models. HypeNet (Chen et al., 2026) has recently presented strong long-context recall performance, therefore we focus mainly on this model at scales from 2B to 9B. To ensure a comprehensive analysis, we also include Jet-Nemotron-2B1.

Training. To investigate the influence of CoT-SFT, we primarily focus on the math domain, which is widely studied and has sufficient high-quality datasets (Mitra et al., 2024; Yu et al., 2025; Zhou et al., 2026). We train HypeNet in different scales on our own in both pre-training and SFT stage, and apply SFT to Jet-Nemotron from their pre-trained checkpoints. More details of datsets and training configuration are in Appendix J.

Evaluation. We evaluate the model’s performance on long-context recall and math reasoning tasks.

1https://huggingface.co/collections/jet-ai/jet-nemotron

- Table 1 | NIAH and Math performance comparisons across models after CoT-SFT on the Miromind dataset. ↑ (↓) indicates the improvement (degradation) compared to SFT model.

Model Method NIAH-Single-1 NIAH-Single-2 NIAH-Single-3 MATH500 GSM8K AIME24

32K 64K 128K 256K 32K 64K 128K 256K 32K 64K 128K 256K Avg@16 Maj@16 Avg@16 Maj@16 Avg@16 Maj@16

Jet-Nemotron-2B Pre-train 100.0 99.8 100.0 100.0 95.6 91.0 61.0 22.2 57.2 48.0 73.6 40.6 37.5 64.0 52.6 84.3 0.80 3.30 +SFT 100.0 100.0 100.0 98.6 57.8 28.4 21.0 11.6 34.4 33.6 67.2 27.8 49.4 70.6 71.9 88.0 1.50 3.30 +QK-Restore 100.0 100.0 100.0 98.6 64.6 34.0 22.0 7.60 41.8(7.4↑) 39.0(5.4↑) 64.4(2.8↓) 34.0(6.2↑) 49.0 69.4 71.2 88.6 1.90 6.70

HypeNet-2B Pre-train 99.2 97.6 97.2 98.4 75.8 94.0 88.0 56.2 48.8 34.8 30.4 23.4 4.70 25.4 2.20 20.0 0.00 0.00 +SFT 99.8 99.8 99.2 99.6 82.2 81.8 48.4 41.8 47.4 32.6 7.60 10.6 34.4 55.2 40.8 68.1 0.00 0.00

###### +QK-Restore 99.4 99.6 99.4 99.8 97.6 91.2 83.2 40.8 56.2(8.8↑) 52.8(20.2↑) 30.8(23.2↑) 30.2(19.6↑) 33.7 56.6 41.2 69.0 0.20 3.30

HypeNet-5B Pre-train 100.0 100.0 100.0 100.0 99.4 99.8 97.0 93.4 95.0 94.2 87.2 75.2 4.70 33.8 4.10 39.5 0.00 0.00 +SFT 100.0 100.0 100.0 100.0 98.8 100.0 91.4 83.6 96.6 96.6 90.6 65.4 49.3 67.0 61.3 81.4 0.60 0.00

###### +QK-Restore 100.0 100.0 100.0 100.0 99.8 100.0 96.8 85.4 97.0(0.4↑) 96.4(0.2↓) 90.2(0.4↓) 76.4(11.0↑) 48.7 68.4 60.9 81.8 0.40 3.30

HypeNet-9B Pre-train 100.0 100.0 100.0 100.0 64.2 95.8 72.4 67.2 53.0 95.0 83.8 52.0 17.6 57.6 15.3 62.4 0.80 6.70 +SFT 100.0 100.0 100.0 100.0 89.0 64.2 17.4 9.40 98.2 90.0 47.8 22.8 39.8 69.5 62.3 87.8 1.20 10.0 +QK-Restore 100.0 100.0 100.0 100.0 92.6 100.0 44.0 19.6 98.6(0.4↑) 97.0(7.0↑) 66.4(18.6↑) 42.6(19.8↑) 37.6 68.4 59.3 88.4 0.60 3.30

For long-context recall, we report the accuracy on NIAH. To measure math reasoning, we test the model on GSM8K (Cobbe et al., 2021) and MATH500 (Lightman et al., 2023).

- 6.2. Main Results The performance comparisons are in Table 1.

CoT-SFT drives long-context degradation. A key observation is that CoT-SFT often weakens longcontext retrieval abilities acquired during pre-training, particularly under more challenging retrieval settings and longer context windows. While simpler tasks such as NIAH-Single-1 remain nearly saturated, substantial degradation emerges on NIAH-Single-2 and NIAH-Single-3.

For example in Table 1, HypeNet-2B on NIAH-Single-3 at 128K drops from 30.4 to 7.60 after SFT, while HypeNet-9B on NIAH-Single-2 at 256K decreases from 67.2 to 9.40. Moreover, the degradation consistently becomes more severe as context length increases, suggesting that SFT disproportionately disrupts the mechanisms responsible for long-range retrieval.

QK-Restore recovers long-context capability. QK-Restore consistently mitigates the loss of long-context performance incurred during SFT, with the largest gains observed in configurations where the degradation is most severe.

Specifically in Table 1, for HypeNet-2B, NIAH-Single-3 at 256K improves from 10.6 to 30.2 (+19.6), while HypeNet-9B increases from 22.8 to 42.6 (+19.8). Under OpenThoughts-3 CoT-SFT, HypeNet-9B improves from 22.4 to 45.0 on NIAH-Single-2 at 256K and from 40.6 to 61.2 on NIAH-Single-3 at 256K. In several configurations, QK-Restore even surpasses the original pre-training baseline. For example, HypeNet-2B on NIAH-Single-3 at 64K improves from 34.8 (pre-train) to 52.8. This suggests that our approach does not simply recover pre-training behavior; rather, it can preserve retrieval structures from pre-training while still benefiting from task-specific representations introduced during SFT.

QK-Restore preserves downstream reasoning performance. Beyond retrieval, we evaluate whether restoring routing parameters affects downstream reasoning capabilities. Across benchmarks, QK-Restore largely preserves the gains obtained from CoT-SFT while substantially improving long-context retrieval.

On mathematical reasoning tasks, performance remains close to the post-SFT model. For example, on HypeNet-5B, QK-Restore incurs only minor changes of −0.6 and −0.4 points on MATH500 and GSM8K, respectively, while recovering long-context retrieval performance. Similar trends are observed across other model scales. These results suggest that routing parameters can be selectively restored to recover long-context retrieval without significantly affecting the task-specific knowledge and reasoning capabilities acquired during post-training.

- Table 2 | NIAH comparisons across models after Instruction-following (non-CoT) SFT on the Tulu-3 dataset.

Model Method NIAH-Single-1 NIAH-Single-2 NIAH-Single-3 IFEval

32K 64K 128K 256K 32K 64K 128K 256K 32K 64K 128K 256K

HypeNet-5B Pre-train 100.0 100.0 100.0 100.0 99.4 99.8 97.0 93.4 95.0 94.2 87.2 75.2 12.0 +SFT 99.8 100.0 100.0 100.0 99.8 100.0 99.6 93.6 98.6 97.2 85.4 83.8 26.6 +QK-Restore 100.0 100.0 100.0 100.0 99.6 100.0 99.8 93.4 97.4 95.2 85.4 85.2 26.6

HypeNet-9B Pre-train 100.0 100.0 100.0 100.0 64.2 95.8 72.4 67.2 53.0 95.0 83.8 52.0 14.0 +SFT 100.0 100.0 100.0 100.0 86.4 97.8 88.2 78.4 96.2 96.8 95.8 86.8 27.2 +QK-Restore 100.0 99.8 100.0 100.0 88.2 97.4 92.8 78.6 96.0 97.4 95.2 88.2 29.0

Retrieval difficulty amplifies long-context degradation. We further observe that retrieval complexity strongly influences robustness. NIAH-Single-1 remains nearly saturated across training stages, whereas substantially larger gaps emerge in Single-2 and Single-3. This trend indicates that difficult retrieval settings place greater demands on precise long-range token interactions and are therefore more sensitive to the degradation.

### 7. Analysis

##### 7.1. Analysis on Non-CoT SFT

Our theoretical analysis attributes long-context degradation to the localized optimization dynamics induced by CoT supervision. A natural question is whether this phenomenon arises from post-training in general, or whether it is specific to CoT-style reasoning traces. To answer this question, we compare CoT-SFT with conventional instruction-following SFT using the Tulu-3 dataset.

- As shown in Table 2, standard instruction tuning does not exhibit the same degradation pattern observed under CoT-SFT. Across both HypeNet-5B and HypeNet-9B, long-context retrieval performance is largely preserved and frequently improved after Tulu-3 training. For example, HypeNet-5B improves from 75.2 to 83.8 on NIAH-Single-3 at 256K, while HypeNet-9B improves from 52.0 to 86.8. Similar gains are observed on NIAH-Single-2 across multiple context lengths.

Moreover, instruction-following ability improves substantially after Tulu-3 tuning, as measured by IFEval. Despite these improvements, long-context retrieval remains intact. This contrasts sharply with the behavior of CoT-SFT, where retrieval performance deteriorates under identical architectures and context lengths.

These findings suggest that long-context degradation is not an inherent consequence of post-training. Instead, it is closely associated with CoT-style supervision, supporting our hypothesis that the sequential dependency structure of reasoning traces concentrates gradient updates on short-range interactions and disproportionately affects long-range routing behavior.

Key Takeaway-2

Long-context degradation is not an inherent consequence of post-training, where non-CoT instruction-following SFT preserves or even improves retrieval for hybrid models.

##### 7.2. Analysis Beyond Mathetical Reasoning

Our primary experiments focus on mathematical reasoning, where high-quality CoT supervision is readily available. To evaluate whether the observed retrieval–reasoning trade-off extends beyond

- Table 3 | NIAH comparisons across models after CoT-SFT on the OpenThoughts-3 dataset.

Model Method NIAH-Single-1 NIAH-Single-2 NIAH-Single-3 LCB-V5

32K 64K 128K 256K 32K 64K 128K 256K 32K 64K 128K 256K Avg@8 HypeNet-5B Pre-train 100.0 100.0 100.0 100.0 99.4 99.8 97.0 93.4 95.0 94.2 87.2 75.2 0.27

+SFT 100.0 100.0 99.8 100.0 99.8 100.0 99.0 89.6 98.8 97.8 94.2 91.4 6.37

- +QK-Restore 100.0 99.8 99.8 100.0 99.8 100.0 100.0 92.8 98.6 97.4 93.2 79.4 6.92

HypeNet-9B Pre-train 100.0 100.0 100.0 100.0 64.2 95.8 72.4 67.2 53.0 95.0 83.8 52.0 1.66

+SFT 100.0 100.0 100.0 100.0 91.8 100.0 41.8 22.4 96.2 98.6 76.6 40.6 13.11

- +QK-Restore 100.0 100.0 100.0 99.8 94.2 99.8 78.2 45.0 95.0 97.4 90.8 61.2 12.91

mathematics, we conduct additional experiments using OpenThoughts-3 and evaluate downstream coding performance on LiveCodeBench-V5.

- As shown in Table 3, the same qualitative behavior persists in the coding domain. CoT-SFT substantially improves coding capability, increasing LiveCodeBench-V5 from 0.27 to 6.37 on HypeNet-5B and from 1.66 to 13.11 on HypeNet-9B. However, these gains are accompanied by noticeable degradation in long-context retrieval, particularly on challenging NIAH settings and long context lengths.

Applying QK-Restore recovers a large fraction of the lost retrieval capability while preserving downstream coding performance. For example, on HypeNet-5B, LiveCodeBench-V5 further improves from 6.37 to 6.92, while NIAH-Single-3 at 256K recovers from 75.2 to 79.4. Similarly, HypeNet9B improves from 40.6 to 61.2 on NIAH-Single-3 at 256K while maintaining competitive coding performance (13.11 → 12.91).

These results suggest that the retrieval degradation induced by CoT-SFT is not specific to mathematical reasoning. Instead, it appears to arise from a more general optimization phenomenon associated with long reasoning traces. The effectiveness of QK-Restore across both math and code domains further supports our central hypothesis that long-range retrieval is primarily governed by routing parameters, while task-specific reasoning capabilities are largely encoded elsewhere in the network.

##### 7.3. Analysis on Attention Routing Recover

In this section, we visualize the attention map to characterize how CoT supervised fine-tuning alters the long-range routing behavior of the softmax-attention layers in HypeNet-5B. We conduct the analysis on the models trained on Miromind dataset.

In detail, we compare the attention matrices of a pre-SFT and a post-CoT-SFT checkpoint of HypeNet-

- 5B across all softmax-attention layers. We then identify the most affected layer via a per-head mean attention distance metric:

𝑑¯ℎ = 𝔼𝑡 ∑︁ 𝑠≤𝑡

(𝑡−𝑠) 𝐴ℎ,𝑡,𝑠 , (17)

where ℎ indexes the attention head and 𝐴ℎ,𝑡,𝑠 is the post-softmax weight from query 𝑡 to key 𝑠. Ranking layers by the number of heads whose 𝑑¯ℎ decreases after SFT, Layer 33 emerges as the most affected, with 25 of 32 heads shifting toward shorter attention distances. We therefore focus our analysis on this layer.

We visualize the top recall heads on a synthetic NIAH prompt of 8,192 tokens, with the target fact embedded at 50% context depth. For each head, we display a four-panel figure: the pre-SFT attention matrix 𝐴˜pre (blue), the post-SFT matrix 𝐴˜post (red), the QK-Restore matrix 𝐴˜QK-Restore (green), and the element-wise log-ratio log2(𝐴˜QK-Restore/𝐴˜post) (Panel 4), where positive values indicate positions where QK-Restore recovers more long-range attention than post-SFT.

[Figure 5]

- Figure 4 | Attention map from query 𝑡 to key 𝑠 of layer33-head8 in pre-SFT and post-SFT on HypeNet5B. Particularly, panel 4 demonstrates post-SFT assigning higher weights to nearby tokens, which is consistent with our Theorem, and the effectiveness of QK-Restore in restoring the long-range attention behavior.

[Figure 6]

- Figure 5 | Routing geometry drift in HypeNet-2B.

Analysis. Figure 4 shows the pattern of Layer-33 Head-8 of HypeNet-5B. From panel 4, we observe that the off-diagonal lower triangle is predominantly red, confirming that QK-Restore partially recovers the long-range routing suppressed by CoT fine-tuning: for most query positions 𝑡, QK-Restore attends more strongly to distant keys 𝑠 ≪ 𝑡 than post-SFT does. Second, the lower-right block, where both the query and key positions fall after the needle 𝑠∗, shows a blue band along the main diagonal, meaning that post-SFT assigns higher weight to nearby tokens in this region, which matches our hypothesis. These observations demonstrate the effectiveness of QK-Restore in mitigating routing collapse and restoring the long-range attention behavior characteristic.

##### 7.4. Ablation on QK-Restore

QK-Restore transplanting the full 𝑾𝑄,𝑾𝐾 matrices from the pre-SFT checkpoint into the post-SFT model, recovers long-range recall substantially but incurs a math performance drop. The post-SFT 𝑾𝑄 can be decomposed as:

𝑾𝑄post = 𝑾𝑄pre + 𝛿𝑾𝑄route + 𝛿𝑾𝑄math, where 𝛿𝑾route encodes harmful locality drift and 𝛿𝑾math encodes beneficial math ability adaptation. Since both components are entangled within 𝑾𝑄, QK-Restore discards them together, recovering routing at the cost of erasing the math benefit. As expected, Figure 5 and Figure 6 show a nonnegligible drift in 𝑹 = 𝑾𝑄𝑾⊤

𝐾 , where Δ𝑹 = 𝑹post − 𝑹pre. Therefore, we model it as a Procrustes problem:

2

2

𝑾𝑄new − 𝑾𝑄post

+ 𝑾𝐾new − 𝑾𝐾post

min

(18)

𝑾𝑄new,𝑾𝐾new

𝐹

𝐹 s.t. 𝑾𝑄new𝑾𝐾new⊤ = 𝑹pre,

[Figure 7]

- Figure 6 | Routing geometry drift in Jet-Nemotron-2B.

- Table 4 | Ablation of QK-Restore in HypeNet-2B.

Method NIAH-Single-2 NIAH-Single-3 MATH500

64K 128K 64K 128K Avg@16

QK-Restore 91.2 83.2 52.8 30.8 33.7 QK-Pro 94.0 79.0 51.8 26.0 34.2 Q-Restore 78.4 79.0 50.8 24.4 34.2 K-Restore 90.4 71.6 46.2 13.2 34.2 V-Restore 57.2 36.6 24.0 15.2 32.1 QK-Frozen 90.6 76.4 49.4 23.6 33.1

and we call it as QK-Pro variant (more details are in Appendix N).

Furthermore, we evaluate the roles of each components in attention (i.e., 𝑾𝐾,𝑾𝑄,𝑾𝑉), and CoT-SFT with 𝑾𝑄pre,𝑾𝐾pre frozen as a preventive counterpart (denoted as QK-Frozen), to answer a question: is it preferable to prevent routing drift during training, or to correct it post-hoc?

Analysis. Table 4 validates three key claims. V-Restore fails to recover NIAH while degrading MATH500, confirming that 𝑾𝑉 does not contribute to routing recovery. Q-Restore and K-Restore each achieve partial recall recovery but fall significantly short at 128K (e.g., K-Restore collapses on NIAH-Single-3 (13.2 vs. 30.8), indicating that isolated restoration introduces a 𝑾𝑄-𝑾𝐾 mismatch and that joint restoration is necessary for coherent routing geometry. QK-Pro achieves the highest performance on MATH500, but trading long-range routing recovery for math preservation. QKFrozen consistently underperforms QK-Restore in both NIAH and mathematical tasks, indicating that unconstrained SFT is for better 𝑾𝑉 adaptation.

### 8. Conclusion

In this work, we identify CoT-SFT-induced long-context recall degradation as a critical post-training failure mode in distilled hybrid models. We show that reasoning-oriented SFT biases optimization toward short-range attention patterns, perturbing the query-key routing geometry of retained softmaxattention layers and weakening long-range retrieval. Motivated by this observation, we propose QK-Restore, a training-free method that restores only the query and key projections from the pre-SFT checkpoint while preserving all other parameters. Experiments across multiple architectures show that QK-Restore substantially recovers long-context capability while largely preserving SFT reasoning gains. Overall, our findings highlight routing stability as a key factor for efficient long-context models that jointly support retrieval and reasoning.

### References

L. B. Allal, A. Lozhkov, E. Bakouch, G. M. Blázquez, G. Penedo, L. Tunstall, A. Marafioti, H. Kydlíček,

- A. P. Lajarín, V. Srivastav, J. Lochner, C. Fahlgren, X.-S. Nguyen, C. Fourrier, B. Burtenshaw, H. Larcher, H. Zhao, C. Zakka, M. Morlon, C. Raffel, L. von Werra, and T. Wolf. Smollm2: When smol goes big – data-centric training of a small language model, 2025. URL https://arxiv.org/abs/2502.02737.

- S. Arora, B. Yang, S. Eyuboglu, A. Narayan, A. Hojel, I. Trummer, and C. Ré. Language models enable simple systems for generating structured views of heterogeneous data lakes. arXiv preprint arXiv:2304.09433, 2023.

Y. Chen, Z. L. Thai, Z. Zhou, Z. Zhang, X. Shen, S. Wang, C. Xiao, X. Han, and Z. Liu. Hybrid linear attention done right: Efficient distillation and effective architectures for extremely long contexts,

2026. URL https://arxiv.org/abs/2601.22156.

- K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton,

- R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

T. Dao and A. Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality, 2024. URL https://arxiv.org/abs/2405.21060.

- L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602.

- D. Goldstein, E. Alcaide, J. Lu, and E. Cheah. Radlads: Rapid attention distillation to linear attention decoders at scale, 2026. URL https://arxiv.org/abs/2505.03005.

A. Gu and T. Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2024. URL https://arxiv.org/abs/2312.00752.

Y. Gu, Q. Hu, H. Xi, J. Chen, S. Yang, S. Han, and H. Cai. Jet-nemotron: Efficient language model with post neural architecture search. Advances in Neural Information Processing Systems, 38:47191–47218, 2026.

- E. Guha, R. Marten, S. Keh, N. Raoof, G. Smyrnis, H. Bansal, M. Nezhurina, J. Mercat, T. Vu, Z. Sprague, et al. Openthoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178, 2025.

D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Xu, H. Ding, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Chen, J. Yuan, J. Tu, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. You, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Zhou, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan,

- R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou,
- S. Pan, S. S. Li, S. Zhou, S. Wu, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Liu, W. Liang, W. Gao,

- W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie,
- X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song,

- X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun,
- Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo,

- Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Huang,

- Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie,
- Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang,

- Z. Xu, Z. Zhang, and Z. Zhang. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL http://dx.doi.org/10.1038/s41586-025-09422-z.

- Y. Hoshino, H. Tachibana, M. Inahara, and H. Takegawa. Rad: Redundancy-aware distillation for hybrid models via self-speculative decoding, 2025. URL https://arxiv.org/abs/2505.22135.

C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, Y. Zhang, and B. Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https: //github.com/huggingface/open-r1.

- S. Jelassi, D. Brandfonbrener, S. M. Kakade, and E. Malach. Repeat after me: Transformers are better than state space models at copying. arXiv preprint arXiv:2402.01032, 2024.

- A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand,

- G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed. Mistral 7b, 2023. URL https://arxiv.org/abs/2310.06825.

A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention, 2020. URL https://arxiv.org/abs/2006.16236.

- X. Li, Y. Xiao, D. Ng, H. Ye, Y. Deng, X. Lin, B. Wang, Z. Mo, C. Zhang, Y. Zhang, Z. Yang, R. Li, L. Lei, S. Xu, H. Zhao, W. Chen, F. Ji, and L. Bing. Miromind-m1: An open-source advancement in mathematical reasoning via context-aware multi-stage policy optimization, 2025a. URL https: //arxiv.org/abs/2507.14683.
- Y. Li, S. Yang, S. Tan, M. Mishra, R. Panda, J. Zhou, and Y. Kim. Distilling to hybrid attention models via kl-guided layer selection. arXiv preprint arXiv:2512.20569, 2025b.
- Z.-Z. Li, X. Liang, Z. Tang, L. Ji, P. Wang, H. Xu, H. Huang, W. Deng, Y. Gong, Z. Guo, et al. Tl; dr: Too long, do re-weighting for efficient llm reasoning compression. arXiv preprint arXiv:2506.02678, 2025c.

- H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe. Let’s verify step by step, 2023. URL https://arxiv.org/abs/2305.20050.

C. Lockard, P. Shiralkar, and X. L. Dong. Openceres: When open information extraction meets the semi-structured web. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3047–3056, 2019.

A. Lozhkov, L. Ben Allal, L. von Werra, and T. Wolf. Fineweb-edu: the finest collection of educational content, 2024. URL https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu.

J. Mattern, S. Jaghouar, M. Basra, J. Straube, M. D. Ferrante, F. Gabriel, J. M. Ong, V. Weisser, and J. Hagemann. Synthetic-1: Two million collaboratively generated reasoning traces from deepseek-r1,

2025. URL https://www.primeintellect.ai/blog/synthetic-1-release. A. Mitra, H. Khanpour, C. Rosset, and A. Awadallah. Orca-math: Unlocking the potential of slms in grade school math, 2024.

NVIDIA, :, A. Blakeman, A. Grattafiori, A. Basant, A. Gupta, A. Khattar, A. Renduchintala, A. Vavre, A. Shukla, A. Bercovich, A. Ficek, A. Shaposhnikov, A. Kondratenko, A. Bukharin, A. Milesi, A. Taghibakhshi, A. Liu, A. Barton, A. S. Mahabaleshwarkar, A. Klein, A. Zuker, A. Geifman, A. Shen,

- A. Bhiwandiwalla, A. Tao, A. Agrusa, A. Verma, A. Guan, A. Mandarwal, A. Mehta, A. Aithal,

- A. Poojary, A. Ahamed, A. Mishra, A. K. Thekkumpate, A. Dattagupta, B. Zhu, B. Sadeghi, B. Simkin,
- B. Lanir, B. Schifferer, B. Nushi, B. Kartal, B. D. Rouhani, B. Ginsburg, B. Norick, B. Soubasis,

- B. Kisacanin, B. Yu, B. Catanzaro, C. del Mundo, C. Hwang, C. Wang, C.-P. Hsieh, C. Zhang, C. Yu,
- C. Mungekar, C. Patel, C. Alexiuk, C. Parisien, C. Neale, C. Meurillon, D. Mosk-Aoyama, D. Su,
- D. Corneil, D. Afrimi, D. Lo, D. Rohrer, D. Serebrenik, D. Gitman, D. Levy, D. Stosic, D. Mosallanezhad, D. Narayanan, D. Nathawani, D. Rekesh, D. Yared, D. Kakwani, D. Ahn, D. Riach, D. Stosic,
- E. Minasyan, E. Lin, E. Long, E. P. Long, E. Segal, E. Lantz, E. Evans, E. Ning, E. Chung, E. Harper,

- E. Tramel, E. Galinkin, E. Pounds, E. Briones, E. Bakhturina, E. Tsykunov, F. Ladhak, F. Wang, F. Jia,
- F. Soares, F. Chen, F. Galko, F. Sun, F. Siino, G. H. Agam, G. Ajjanagadde, G. Bhatt, G. Prasad,
- G. Armstrong, G. Shen, G. Batmaz, G. Nalbandyan, H. Qian, H. Sharma, H. Ross, H. Ngo, H. Hum,
- H. Sahota, H. Wang, H. Soni, H. Upadhyay, H. Mao, H. C. Nguyen, H. Q. Nguyen, I. Cunningham,
- I. Galil, I. Shahaf, I. Gitman, I. Loshchilov, I. Schen, I. Levy, I. Moshkov, I. Golan, I. Putterman,
- J. Kautz, J. P. Scowcroft, J. Casper, J. Mitra, J. Glick, J. Chen, J. Oliver, J. Zhang, J. Zeng, J. Lou, J. Zhang, J. Choi, J. Huang, J. Conway, J. Guman, J. Kamalu, J. Greco, J. Cohen, J. Jennings, J. Daw, J. V. Vialard, J. Yi, J. Parmar, K. Xu, K. Zhu, K. Briski, K. Cheung, K. Luna, K. Wyss, K. Santhanam, K. Shih, K. Kong, K. Bhardwaj, K. Shankar, K. C. Puvvada, K. Pawelec, K. Anik, L. McAfee, L. Sleiman, L. Derczynski, L. Ding, L. Wei, L. Liebenwein, L. Vega, M. Grover, M. V. Segbroeck, M. R. de Melo, M. Nazemi, M. N. Sreedhar, M. Kilaru, M. Ashkenazi, M. Romeijn, M. Chochowski, M. Cai, M. Kliegl, M. Moosaei, M. Kulka, M. Novikov, M. Samadi, M. Corpuz, M. Wang, M. Price,

- M. Andersch, M. Boone, M. Evans, M. Martinez, M. Khona, M. Chrzanowski, M. Lee, M. Dabbah,

- M. Shoeybi, M. Patwary, N. Mulepati, N. Nabwani, N. Hereth, N. Assaf, N. Habibi, N. Zmora,
- N. Haber, N. Sessions, N. Bhatia, N. Jukar, N. Pope, N. Ludwig, N. Tajbakhsh, N. Ailon, N. Juluru,

- N. Sharma, O. Hrinchuk, O. Kuchaiev, O. Delalleau, O. Olabiyi, O. U. Argov, O. Puny, O. Tropp, O. Xie, P. Chadha, P. Shamis, P. Gibbons, P. Molchanov, P. Morkisz, P. Dykas, P. Jin, P. Xu, P. Januszewski,

- P. P. Thombre, P. Varshney, P. Gundecha, P. Tredak, Q. Miao, Q. Wan, R. K. Mahabadi, R. Garg,

- R. El-Yaniv, R. Zilberstein, R. Shafipour, R. Harang, R. Izzo, R. Shahbazyan, R. Garg, R. Borkar,

- R. Gala, R. Islam, R. Hesse, R. Waleffe, R. Watve, R. Koren, R. Zhang, R. Hewett, R. J. Hewett,

- R. Prenger, R. Timbrook, S. Mahdavi, S. Modi, S. Kriman, S. Lim, S. Kariyappa, S. Satheesh, S. Kaji,
- S. Pasumarthi, S. Muralidharan, S. Narentharen, S. Narenthiran, S. Bak, S. Kashirsky, S. Poulos,

- S. Mor, S. Ramasamy, S. Acharya, S. Ghosh, S. T. Sreenivas, S. Thomas, S. Fan, S. Gopal, S. Prabhumoye, S. Pachori, S. Toshniwal, S. Ding, S. Singh, S. Sun, S. Ithape, S. Majumdar, S. Singhal,

- S. Sergienko, S. Alborghetti, S. Ge, S. D. Devare, S. K. Barua, S. Panguluri, S. Gupta, S. Priyadarshi, S. N. Akter, T. Bui, T.-D. Ene, T. Kong, T. Do, T. Blankevoort, T. Moon, T. Balough, T. Asida, T. B. Natan, T. Ronen, T. Konuk, T. Vashishth, U. Karpas, U. De, V. Noorozi, V. Noroozi, V. Srinivasan,

- V. Elango, V. Cui, V. Korthikanti, V. Rao, V. Kurin, V. Lavrukhin, V. Anisimov, W. Jiang, W. U. Ahmad,
- W. Du, W. Ping, W. Zhou, W. Jennings, W. Zhang, W. Prazuch, X. Ren, Y. Karnati, Y. Choi, Y. Meyer,

- Y.-F. Wu, Y. Zhang, Y. Qin, Y. Lin, Y. Geifman, Y. Fu, Y. Subara, Y. Suhara, Y. Gao, Z. Moshe, Z. Dong,
- Z. Zhu, Z. Liu, Z. Chen, and Z. Yan. Nvidia nemotron 3: Efficient and open intelligence, 2025. URL https://arxiv.org/abs/2512.20856.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama,

- A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe. Training language models to follow instructions with human feedback, 2022. URL https://arxiv.org/abs/2203.02155.
- B. Prystawski, M. Y. Li, and N. D. Goodman. Why think step by step? reasoning emerges from the

locality of experience, 2023. URL https://arxiv.org/abs/2304.03843.

- Z. Qin, W. Sun, D. Li, X. Shen, W. Sun, and Y. Zhong. Lightning attention-2: A free lunch for handling unlimited sequence lengths in large language models, 2024a. URL https://arxiv.org/abs/2401. 04658.

- Z. Qin, W. Sun, D. Li, X. Shen, W. Sun, and Y. Zhong. Various lengths, constant speed: Efficient language modeling with lightning attention, 2024b. URL https://arxiv.org/abs/2405.17381.

Qwen, :, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin,

- J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su,

- Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/ blog?id=qwen3.5.

K. Team, Y. Zhang, Z. Lin, X. Yao, J. Hu, F. Meng, C. Liu, X. Men, S. Yang, Z. Li, W. Li, E. Lu,

- W. Liu, Y. Chen, W. Xu, L. Yu, Y. Wang, Y. Fan, L. Zhong, E. Yuan, D. Zhang, Y. Zhang, T. Y. Liu, H. Wang, S. Fang, W. He, S. Liu, Y. Li, J. Su, J. Qiu, B. Pang, J. Yan, Z. Jiang, W. Huang,

- B. Yin, J. You, C. Wei, Z. Wang, C. Hong, Y. Chen, G. Chen, Y. Wang, H. Zheng, F. Wang, Y. Liu, M. Dong, Z. Zhang, S. Pan, W. Wu, Y. Wu, L. Guan, J. Tao, G. Fu, X. Xu, Y. Wang, G. Lai, Y. Wu,

- X. Zhou, Z. Yang, and Y. Du. Kimi linear: An expressive, efficient attention architecture, 2025. URL https://arxiv.org/abs/2510.26692.

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need, 2023. URL https://arxiv.org/abs/1706.03762.

D. Wang, R.-J. Zhu, S. Abreu, Y. Shan, T. Kergan, Y. Pan, Y. Chou, Z. Li, G. Zhang, W. Huang, and J. Eshraghian. A systematic analysis of hybrid linear attention, 2025. URL https://arxiv.org/ abs/2507.06457.

- Z. Wang, Y. Dong, and Q. Lei. When does chain-of-thought help: A markovian perspective, 2026. URL https://arxiv.org/abs/2603.00306.

- L. Wen, Y. Cai, F. Xiao, X. He, Q. An, Z. Duan, Y. Du, J. Liu, T. Tanglifu, X. Lv, et al. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track), pages 318–327, 2025.
- M. Yang, M. Rezagholizadeh, G. Li, V. Appia, and E. Barsoum. Zebra-llama: Towards extremely efficient hybrid models. arXiv preprint arXiv:2505.17272, 2025.

J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou. Chain-ofthought prompting elicits reasoning in large language models, 2023. URL https://arxiv.org/ abs/2201.11903.

S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim. Gated linear attention transformers with hardwareefficient training, 2024. URL https://arxiv.org/abs/2312.06635.

- Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, X. Liu, H. Lin, Z. Lin, B. Ma, G. Sheng, Y. Tong, C. Zhang, M. Zhang, W. Zhang, H. Zhu, J. Zhu, J. Chen, J. Chen, C. Wang, H. Yu, Y. Song, X. Wei, H. Zhou, J. Liu, W.-Y. Ma, Y.-Q. Zhang, L. Yan, M. Qiao, Y. Wu, and M. Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476.

- M. Zhang, K. Bhatia, H. Kumbong, and C. Ré. The hedgehog & the porcupine: Expressive linear attentions with softmax mimicry, 2024. URL https://arxiv.org/abs/2402.04347.

- C. Zhou, H. Lyu, X. Lin, H. Zhao, J. Guo, X. Zhang, S. Xue, Q. Ma, J. Zhou, Y. Wang, and Z. Liu. Ultradata-math, 2026. URL https://huggingface.co/datasets/openbmb/UltraData-Math.

J. Zhou, T. Lu, S. Mishra, S. Brahma, S. Basu, Y. Luan, D. Zhou, and L. Hou. Instruction-following evaluation for large language models, 2023. URL https://arxiv.org/abs/2311.07911.

### A. Discussion of the CoT-Markov Structure Assumption

- Discussion 1. The 1-step Markov condition is an idealisation for analysis. Any 𝑘-step Markov chain is equivalent to a 1-step chain on a 𝐾𝑘-state product space, yielding the same exponential decay with a larger reasoning horizon 𝑊.
- Discussion 2. Reversibility is invoked in the proof of Lemma 4.5 at one step: it implies that the Markov operator 𝑃 is self-adjoint on 𝐿2(𝜋). We acknowledge that a causally generated sequence such as a CoT reasoning chain does not, in general, satisfy detailed balance: the latent transition probabilities are asymmetric by construction. Reversibility is, however, a sufficient condition rather than a necessary one. For a general ergodic chain (possibly non-reversible), the identical bound holds with 𝜌 replaced by the second-largest singular value of 𝑃, obtained via its singular value decomposition in place of the eigendecomposition. Since singular values are always real and non-negative, the proof proceeds without modification, and the form of the final bound is unchanged. We therefore state Lemma 4.5 under reversibility for clarity of exposition, with the understanding that all subsequent results hold for the general ergodic case by replacing eigenvalues with singular values throughout.

### B. Justification for Norm Bounds Assumption

Bound on ∥𝒗𝑠∥. The value vector is 𝒗𝑠 = 𝑾𝑉 LN(𝒉𝑠), where LN denotes layer normalisation. Layer normalisation standardises each coordinate so that LN(𝒉𝑠)𝑖 = 𝛾𝑖(𝒉𝑠,𝑖 − 𝜇ˆ)/𝜎ˆ + 𝛽𝑖, where 𝜇ˆ and 𝜎ˆ are the empirical mean and standard deviation of the entries of 𝒉𝑠. After normalisation, the pre-affine vector satisfies 𝑖 𝒛2𝑖 = 𝑑, giving

###### ∥ LN(𝒉𝑠)∥2 = ∑︁

(𝛾𝑖𝒛𝑖 + 𝛽𝑖)2 (19)

𝑖

≤ 2 ∥𝛾∥2∞𝑑 + ∥𝛽∥2 (20) ≤ 𝐶LN2 𝑑, (21)

for a constant 𝐶LN depending only on the learned scale 𝛾 and bias 𝛽. The projection matrix 𝑾𝑉 is a fixed trained weight; its operator norm ∥𝑾𝑉∥op < ∞ is a finite model-dependent constant. Combining we have:

∥𝒗𝑠∥ = ∥𝑾𝑉 LN(𝒉𝑠)∥ (22) ≤ ∥𝑾𝑉∥op 𝐶LN

√

𝑑 =: 𝐵𝑉. (23)

Bound on 𝑮𝑡. The gradient 𝑮𝑡 = 𝜕L/𝜕𝒐𝑡 back-propagates through the final layer normalisation (with Jacobian 𝑱LN) and the language-model head 𝑾lm, giving

𝑮𝑡 = 𝑱LN𝑇 𝑾lm𝑇 (𝒑𝑡 − 𝒚𝑡), (24)

where 𝒑𝑡 ∈ Δ|V|−1 is the predicted distribution and 𝒚𝑡 ∈ {0, 1}|V| is the one-hot target. By submultiplicativity of the operator norm:

∥𝑮𝑡∥ ≤ ∥𝑱LN∥op ∥𝑾lm∥op ∥𝒑𝑡 − 𝒚𝑡∥. (25) Since 𝒑𝑡 − 𝒚𝑡 is a difference of probability vectors, ∥𝒑𝑡 − 𝒚𝑡∥ ≤

√2. For a fixed trained model, both ∥𝑱LN∥op and ∥𝑾lm∥op are finite constants. Therefore:

√2 ∥𝑱LN∥op ∥𝑾lm∥op =: 𝐵𝐺. (26)

∥𝑮𝑡∥ ≤

### C. Justification for Score-function Identity Assumption

At the optimum 𝑝𝜽 = 𝑝data, the cross-entropy loss with respect to the attention output 𝒐𝑡 satisfies, for any fixed context 𝑥1:𝑡:

𝜕

𝜕𝒐𝑡 −log 𝑝𝜃(𝑦𝑡 | 𝑥1:𝑡) (27)

𝔼𝑦

𝑡

∑︁

𝜕 𝜕𝒐𝑡

𝑝(𝑦𝑡 | 𝑥1:𝑡) log 𝑝𝜽(𝑦𝑡 | 𝑥1:𝑡)

(28)

= −

𝑝𝜃=𝑝data

𝑦𝑡

= 0, (29)

since the cross-entropy gradient vanishes at its own minimum. Applying the law of total expectation over 𝑥1:𝑡 then yields 𝔼[𝑮𝑡] = 0. More precisely, because the gradient is computed at a given context 𝑥1:𝑡 and then averaged over the target 𝑦𝑡 drawn from the conditional, the identity holds conditionally:

𝔼[𝑮𝑡 | 𝑥1:𝑡] = 0. (30)

#### D. Preliminary Definition and Lemma We list the following necessary definitions and lemmas for the proof.

- Definition D.1 (𝐿2(𝜋) Space). For a Markov chain with finite state space Z and stationary distribution 𝜋, define the weighted function space:

𝐿2(𝜋) = 𝑓 : Z → ℝ (31) with inner product

⟨𝑓, 𝑔⟩𝜋 := ∑︁ 𝑧∈Z

𝜋(𝑧) 𝑓 (𝑧) 𝑔(𝑧) (32)

= 𝔼𝜋[ 𝑓 (𝑍) 𝑔(𝑍)] (33)

and norm ∥ 𝑓 ∥𝐿2(𝜋) := √︁⟨𝑓, 𝑓⟩𝜋.

In addition, mean-zero means ⟨𝑓, 1⟩𝜋 = 𝔼𝜋[ 𝑓 (𝑍)] = 0, i.e., the function has zero expectation under 𝜋.

- Definition D.2 (Operator 𝑃 acting on functions). We define the Markov operator 𝑃 acting on functions:

###### (𝑃 𝑓)(𝑧) := ∑︁

𝑃(𝑧, 𝑧′) 𝑓 (𝑧′) (34)

𝑧′

= 𝔼[ 𝑓 (𝑍𝑡+1) | 𝑍𝑡 = 𝑧] (35)

Lemma D.3. For a reversible chain, detailed balance holds: 𝜋(𝑧)𝑃(𝑧, 𝑧′) = 𝜋(𝑧′)𝑃(𝑧′, 𝑧), which makes 𝑃 symmetric in 𝐿2(𝜋).

Proof. According to the definition of Markov operator 𝑃 in D.2, we have:

⟨𝑃 𝑓, 𝑔⟩𝜋 = ∑︁

𝜋(𝑧)(𝑃 𝑓)(𝑧) 𝑔(𝑧) (36)

= ∑︁

𝑧

𝜋(𝑧) 𝑃(𝑧, 𝑧′) 𝑓 (𝑧′) 𝑔(𝑧) (37)

= ∑︁

𝑧,𝑧′

𝜋(𝑧′) 𝑃(𝑧′, 𝑧) 𝑓 (𝑧′) 𝑔(𝑧) (38)

= ∑︁

###### 𝜋(𝑧′) 𝑓 (𝑧′) ∑︁

𝑧,𝑧′

𝑃(𝑧′, 𝑧) 𝑔(𝑧) (39)

𝑧′

𝑧

= ⟨𝑓, 𝑃𝑔⟩𝜋 (40) □

Since |Z| = 𝐾 < ∞, the finite-dimensional spectral theorem for self-adjoint operators yields a real orthonormal eigenbasis {𝜑𝑘}𝑘𝐾=1 of 𝐿2(𝜋) with eigenvalues 1 = 𝜆1 ≥ |𝜆2| ≥ · · · ≥ |𝜆𝐾|, and spectral gap 1 − 𝜌 where 𝜌 := max𝑘≥2 |𝜆𝑘|:

𝑃𝜑𝑘 = 𝜆𝑘𝜑𝑘, ⟨𝜑𝑘, 𝜑𝑗⟩𝜋 = 𝛿𝑘𝑗 (41) Further, for mean-zero 𝑓:

⟨𝑓, 𝜑1⟩𝜋 = ⟨𝑓, 1⟩𝜋 = 𝔼𝜋[ 𝑓 (𝑍)] = 0 (42) which means 𝑓 has no component along 𝜓1. Its expansion uses only 𝑘 ≥ 2:

𝑓 = ∑︁

𝑎𝑘𝜑𝑘, 𝑎𝑘 = ⟨𝑓, 𝜑𝑘⟩𝜋 (43)

𝑘≥2

### E. Proof for Spectral Correlation Decay Lemma Proof. At stationarity (𝑧𝑠 ∼ 𝜋), the Tower Property tells:

𝔼[ 𝑓 (𝑧𝑡) 𝑔(𝑧𝑠)] = 𝔼 𝔼[ 𝑓 (𝑧𝑡) | 𝑧𝑠] · 𝑔(𝑧𝑠) (44)

Note that the inner expectation is the 𝜏-step prediction: for a fixed starting state 𝑧𝑠 = 𝑧, the expected value of 𝑓 (𝑧𝑡) after 𝜏 = 𝑡 − 𝑠 steps is exactly (𝑃𝜏 𝑓)(𝑧):

(𝑃𝜏 𝑓)(𝑧) = 𝔼[ 𝑓 (𝑧𝑡) | 𝑧𝑠 = 𝑧] (45) Therefore:

𝔼[ 𝑓 (𝑧𝑡) 𝑔(𝑧𝑠)] = 𝔼[(𝑃𝜏 𝑓)(𝑧𝑠) 𝑔(𝑧𝑠)] (46)

= ∑︁

𝜋(𝑧) (𝑃𝜏 𝑓)(𝑧) 𝑔(𝑧) (47)

𝑧

= ⟨𝑃𝜏 𝑓, 𝑔⟩𝜋 (48)

Then we expand 𝑓, 𝑔 by the orthonormal eigenbasis: 𝑓 = ∑︁

𝑎𝑘𝜑𝑘, 𝑔 = ∑︁ 𝑘≥2

𝑏𝑘𝜑𝑘, (49)

𝑘≥2

Using the orthonormality ⟨𝜑𝑘, 𝜑𝑗⟩𝜋 = 𝛿𝑘𝑗, we further simplify ⟨𝑃𝜏 𝑓, 𝑔⟩𝜋:

⟨𝑃𝜏 𝑓, 𝑔⟩𝜋 = ∑︁ 𝑘≥2

𝑎𝑘𝜆𝜏𝑘𝜑𝑘, ∑︁ 𝑗≥2

(50)

𝑏𝑗𝜑𝑗

= ∑︁

𝜋

𝑎𝑘 𝜆𝜏𝑘 𝑏𝑘 (51)

𝑘≥2

Then we bound it:

⟨𝑃𝜏 𝑓, 𝑔⟩𝜋 = ∑︁ 𝑘≥2

𝑎𝑘𝜆𝜏𝑘𝑏𝑘 (52) ≤ ∑︁

|𝑎𝑘| |𝜆𝑘|𝜏 |𝑏𝑘| (53)

≤ 𝜌𝜏 ∑︁ 𝑘≥2

𝑘≥2

|𝑎𝑘| |𝑏𝑘| (54)

Applying the Cauchy-Schwarz Inequality:

∑︁

|𝑎𝑘| |𝑏𝑘| ≤

𝑘≥2

###### √︄∑︁

√︄∑︁

|𝑏𝑘|2 (55)

|𝑎𝑘|2

𝑘≥2

𝑘≥2

= || 𝑓||𝐿2(𝜋)||𝑔||𝐿2(𝜋) (56)

Now combining all together:

𝔼[ 𝑓 (𝑧𝑡) 𝑔(𝑧𝑠)] ≤ 𝜌𝜏|| 𝑓||𝐿2(𝜋)||𝑔||𝐿2(𝜋) (57) = 𝜌𝑡−𝑠 ∥ 𝑓 ∥𝐿2(𝜋) ∥𝑔∥𝐿2(𝜋) (58) □

### F. Proof of the Gradient Locality Theorem

Proof. For clarity, we drop the layer/head superscripts in the following derivation. First, we bound the gradient:

𝜕L 𝜕𝑒𝑡𝑠 ≤ |𝑮𝑡 · 𝒗𝑠| + |𝑮𝑡 · 𝒗¯𝑡| (59)

Then we consider how to bound 𝔼[|𝑮𝑡 · 𝒗𝑠|]. Applying the Tower Property: 𝔼[𝑮𝑡 · 𝒗𝑠] = 𝔼[𝔼[𝑮𝑡|𝑧𝑡] · 𝔼[𝒗𝑠|𝑧𝑠]] (60)

= 𝔼[𝜙(𝑧𝑡) · 𝜓(𝑧𝑠)] (61) where we define:

𝜙(𝑧) = 𝔼[𝑮𝑡|𝑧𝑡 = 𝑧], 𝜓(𝑧) = 𝔼[𝒗𝑠|𝑧𝑠 = 𝑧] (62)

According to Assumption 4.4:

𝔼𝜋[𝜙(𝑧)] = ∑︁

𝜋(𝑧)𝜙(𝑧) (63)

= ∑︁

𝑧

𝑃(𝑧𝑡 = 𝑧)𝜙(𝑧) (64)

𝑧

= 𝔼[𝜙(𝑧𝑡)] = 𝔼[𝔼[𝑮𝑡|𝑧𝑡 = 𝑧]] (65) = 𝔼[𝑮𝑡] = 𝔼[[𝑮𝑡 𝑥1:𝑡]] = 0 (66)

Similarly, centering 𝜓 does not change the covariance (since 𝔼[𝜙(𝑧𝑡)] = 0). Both 𝜙 and 𝜓 are mean-zero functions on Z. Then applying Lemma 4.5:

𝔼[𝑮𝑡 · 𝒗𝑠] = 𝔼[𝜙(𝑧𝑡) · 𝜓(𝑧𝑠)] (67) ≤ 𝜌𝜏||𝜙||𝐿2(𝜋)||𝜓||𝐿2(𝜋) (68) ≤ 𝐵𝐺𝐵𝑉𝜌𝜏 (69)

Applying the same bound to the 𝒗¯𝑡 term, which satisfies ∥−𝒗¯𝑡∥ ≤ 𝐵𝑉 as a convex combination: 𝑔(𝜏) = 𝔼

𝜕L 𝜕𝑒𝑡𝑠 ≤ 2𝐵𝐺𝐵𝑉 𝜌𝜏 = 𝐶′ 𝑒−𝜏/𝑊, (70)

where 𝐶′ = 2𝐵𝐺𝐵𝑉 and 𝑊 = log−1𝜌. □

### G. Proof for Token Autocorrelation Decay

Define the centred indicator feature 𝑓𝑣(𝑥) := 1[𝑥 = 𝑣] − 𝜋(𝑣) and the 𝜋-weighted autocorrelation at lag 𝜏:

𝜌¯(𝜏) := ∑︁

Cov( 𝑓𝑣(𝑥𝑡), 𝑓𝑣(𝑥𝑡+𝜏)) Var( 𝑓𝑣(𝑥𝑡))

(71)

𝜋(𝑣)

𝑣

For the numerator Cov( 𝑓𝑣(𝑥𝑡), 𝑓𝑣(𝑥𝑡+𝜏)), we expand it as: 𝔼[ 𝑓𝑣(𝑥𝑡) 𝑓𝑣(𝑥𝑡+𝜏)] − 𝔼[ 𝑓𝑣(𝑥𝑡)]𝔼[ 𝑓𝑣(𝑥𝑡+𝜏)] (72)

Considering the expectation of 𝑓𝑣(𝑥): 𝔼[ 𝑓𝑣(𝑥𝑡)] = 𝔼[1[𝑥𝑡 = 𝑣] − 𝜋(𝑣)] (73)

= 𝜋(𝑣) − 𝜋(𝑣) = 0 (74) Therefore we can further simplify:

𝜌¯(𝜏) = ∑︁

𝔼[ 𝑓𝑣(𝑥𝑡) 𝑓𝑣(𝑥𝑡+𝜏)] 𝜋(𝑣)(1 − 𝜋(𝑣))

(75)

𝜋(𝑣)

𝑣

In the HMM, tokens are conditionally independent given their latent states: 𝑥𝑡 ⊥ 𝑥𝑡+𝜏 | 𝑧𝑡, 𝑧𝑡+𝜏. By the tower property:

𝔼[ 𝑓𝑣(𝑥𝑡) 𝑓𝑣(𝑥𝑡+𝜏)] (76) = 𝔼 𝔼[ 𝑓𝑣(𝑥) | 𝑧𝑡] · 𝔼[ 𝑓𝑣(𝑥) | 𝑧𝑡+𝜏] (77) = 𝔼[𝜙𝑣(𝑧𝑡) · 𝜙𝑣(𝑧𝑡+𝜏)] (78)

where 𝜙𝑣(𝑧) = 𝔼[ 𝑓𝑣(𝑥)|𝑧] = 𝑃(𝑥 = 𝑣|𝑧) − 𝜋(𝑣) is a mean-zero function of 𝑧: 𝔼𝜋[𝜙𝑣(𝑧)] = 𝔼𝜋[𝑃(𝑥 = 𝑣|𝑧) − 𝜋(𝑣)] (79)

= ∑︁

𝜋(𝑧)𝑃(𝑥 = 𝑣|𝑧) − 𝜋(𝑣) (80) = 𝜋(𝑣) − 𝜋(𝑣) = 0 (81)

𝑧

Since both 𝜙𝑣(𝑧𝑡) and 𝜙𝑣(𝑧𝑡+𝜏) are mean-zero, we can apply Lemma 4.5 directly: 𝔼[𝜙𝑣(𝑧𝑡) · 𝜙𝑣(𝑧𝑡+𝜏)] ≤ 𝜌𝜏 ∥𝜙𝑣∥2𝐿2(𝜋) (82)

where 𝜌 = 𝜆2 is the second-largest eigenvalue of the latent transition matrix 𝑃. Now we can bound the autocorrelation:

𝜌¯(𝜏) = ∑︁

𝔼[ 𝑓𝑣(𝑥𝑡) 𝑓𝑣(𝑥𝑡+𝜏)] 𝜋(𝑣)(1 − 𝜋(𝑣))

(83)

𝜋(𝑣)

𝑣

≤ 𝜌𝜏 ∑︁

∥𝜙𝑣∥2𝐿2(𝜋) 𝜋(𝑣)(1 − 𝜋(𝑣))

= 𝐶 · 𝜌𝜏 (84)

𝑣

Conclusion. Observing 𝜌¯(𝜏) ∼ 𝐶 𝜌ˆ𝜏 with 𝜌ˆ < 1 in the corpus confirms that the latent chain has spectral gap 1 − 𝜌ˆ > 0, which is exactly Lemma 4.5’s hypothesis.

### H. Discussion of Attention Gradient Decay in Pre-training Stage

Theorem 4.6 rests on Lemma 4.5, which holds for any ergodic Markov chain with a spectral gap: a property of the data-generating process, not specific to CoT structure, which is also observed in general pre-training data corpus.

Consequently, the gradient locality bound 𝑔(𝜏) ≤ 𝐶′𝑒−𝜏/𝑊 applies to any training corpus, including general pre-training data. The severity of routing collapse is governed by the mismatch Δ𝑊 = 𝑊corr − 𝑊grad: how far the data demands the model to attend vs. how far the gradient actually reinforces it. For general pre-training corpora, this mismatch is smaller than for CoT, so routing degrades more slowly but by the same mechanism.

For hybrid architectures, however, even the slower pre-training erosion meaningfully reduces available recall capacity before fine-tuning begins, leaving the model closer to the threshold. The hybrid architecture’s vulnerability therefore originates during pre-training and is further amplified by CoT fine-tuning.

### I. Proof for the Routing-Extraction Gradient Decoupling Theorem Proof. From Equation 7, we have:

∑︁

1 √𝑑ℎ

𝜕L 𝜕𝑒𝑡𝑠(ℎ)

′)

𝑠 𝒉𝑇𝑡 (85)

𝒌(ℎ

∇𝑾𝑄,(ℎ)L =

𝑡,𝑠

where ℎ′ is the corresponding KV head. Then considering its Frobenius norm: 𝔼

𝜕L 𝜕𝑒𝑡𝑠 · ∥𝒌𝑠∥ · ∥𝒉𝑡∥ (86) ≤ 𝐶′𝐵𝐾𝐵ℎ 𝜌𝜏 (87) = 𝐶𝑅 · 𝜌𝜏 (88)

𝜕L 𝜕𝑒𝑡𝑠

𝒌𝑠 𝒉𝑇𝑡

= 𝔼

𝐹

- Table 5 | Pre-Training Configs for HypeNet-2B

Stage Tokens LR LR Scheduler Context Len. Batch Training Steps

- 1 320M 1e-3 → 1e-5 Cosine 512 32 20,000

- 2 1B 1e-4 → 1e-5 Cosine 512 96 20,000

- 3 1B 1e-5 Constant 16,384 128 500

- Table 6 | Pre-Training Configs for HypeNet-5B

Stage Tokens LR LR Scheduler Context Len. Batch Training Steps

- 1 320M 1e-3 → 1e-5 Cosine 512 32 20,000

- 2 1B 5e-5 → 1e-5 Cosine 512 96 20,000

- 3 1B 1e-5 Constant 16,384 128 500

- Table 7 | Pre-Training Configs for HypeNet-9B

Stage Tokens LR LR Scheduler Context Len. Batch Training Steps

- 1 320M 1e-3 → 1e-5 Cosine 512 32 20,000

- 2 1B 3e-5 → 1e-5 Cosine 512 96 20,000

- 3 1B 1e-5 Constant 16,384 128 500

via Theorem 4.6 and bounded norms ∥𝒌𝑠∥ ≤ 𝐵𝐾, ∥𝒉𝑡∥ ≤ 𝐵ℎ. For the Exctraction Parameters, the value vector 𝒗𝑠 = 𝑾𝑉𝒉𝑠 enters the loss through 𝒐𝑡 = 𝑠 𝐴𝑡𝑠𝒗𝑠. Accumulating all downstream dependencies:

###### = ∑︁

𝜕L 𝜕𝒗𝑠

𝐴𝑡𝑠 𝑮𝑡 (89)

𝑡≥𝑠

Now we assume that 𝐴𝑠𝑠 ≥ 𝛿𝐴 > 0 for all 𝑠 and 𝔼[∥𝑮𝑠∥] ≥ 𝑐𝐺 > 0 for all 𝑠:

𝜕L 𝜕𝒗𝑠 ≥ 𝔼[𝐴𝑠𝑠∥𝑮𝑠∥] ≥ 𝛿𝐴𝑐𝐺 > 0 (90)

𝔼

□

### J. Training Data and Configuration Details

For HypeNet, we conduct both the pre-training and SFT.

Pre-training Data. Since we mainly focus on the long-context recall and math reasoning ability, during the pre-training stage, we mix the FineWeb-Edu (Lozhkov et al., 2024) and UltraData-Math (Zhou et al., 2026). Following observations from recent pretraining studies that maintain web corpora as the dominant source while increasing structured reasoning data (Allal et al., 2025), we construct a mixture with 80% general web data and 20% math-focused data. Prior evidence indicates that web-scale corpora preserve broad linguistic and factual competence, while specialized mathematical corpora improve reasoning and STEM performance.

Pre-training Config. We adopt the 3-stage training for HypeNet, keeping the same setting in Chen et al. (2026), shown in Table 5, Table 6, and Table 7.

- Table 8 | SFT Configs for 2B scale model

LR LR Scheduler Context Len. Batch Training Steps 1e-5 Constant 16,384 128 100

- Table 9 | SFT Configs for 5B, 9B scale model

LR LR Scheduler Context Len. Batch Training Steps 1e-5 → 1e-6 Cosine 16,384 128 100

- Table 10 | NIAH on pure softmax-attention models.

Model Method NIAH-Single-2 NIAH-Single-3

32K 64K 32K 64K Qwen2.5-7B Pre-train 100.0 95.8 99.8 98.6

- +SFT 100.0 92.6 99.8 99.4

Qwen2.5-3B Pre-train 100.0 94.6 100.0 99.6

- +SFT 100.0 93.8 100.0 90.8

+QK-Restore 100.0 93.8 100.0 93.4

Mistral-7B Pre-train 99.8 90.2 99.8 81.0 +SFT 99.6 59.6 65.8 4.80 +QK-Restore 99.8 62.0 70.0 5.40

SFT Data. To further improve the model’s math reasoning performance, we conduct CoT-SFT with MiroMind-M1 dataset (Li et al., 2025a), which is collected from OpenR1 (Hugging Face, 2025), Open-thoughts (Guha et al., 2025), Light-R1 (Wen et al., 2025), and Synthetic-1 (Mattern et al., 2025). SFT Config. The detailed configurations are shown in Table 8 and Table 9.

### K. Evaluation Details

For evaluating NIAH, we apply LM Evaluation Harness2 (Gao et al., 2024) for official test. For evaluating MATH500 and GSM8K, we report average pass@1 over 16 independent generations and Maj@16 (i.e., majority vote as the final prediction) per problem. In detail, for MATH500, the generation length is 8,192; for GSM8K, we set it as 2,048, since it is easier. For LiveCodeBench, we report the average pass@1 over 8 independent generations with 16,384 response length. To measure the model’s performance after fine-tuning on Tulu3, we report the Prompt-level strict accuracy on IFEval task (Zhou et al., 2023).

### L. Analysis on Pure Softmax-Attention Model

Our main experiments concentrate on the hybrid models, and in this section, we explore whether SFT can degrade the pure softmax-attention models as well. We include Qwen2.5-3B, Qwen2.5-7B (Qwen et al., 2025) and Misrtral-7B-Instruct-v0.3 (Jiang et al., 2023) for investigation. We include the details on how we identify the top layers for retrieval in Appendix M.

Analysis. Table 10 shows that for Mistral-7B, it exhibits strong sensitivity to SFT at longer contexts. SFT causes severe degradation on NIAH-Single-3 at 64K, dropping from 81.0 to 4.80, suggesting

2https://github.com/EleutherAI/lm-evaluation-harness

a breakdown of long-range retrieval. QK-Restore recovers this degradation limitedly, indicating weaker long-context representations in the pre-trained model.

[Figure 8]

Figure 7 | Attention gradient decay on Qwen2.5-3B. We observe a different pattern from the exponential decay of HypeNet.

For Qwen2.5-7B and Qwen2.5-3B, long-context performance is highly stable across NIAH-Single-2 and Single-3. SFT induces only minor fluctuations (e.g., Qwen2.5-7B drops from 95.8 to 92.6 at 64K), indicating that these models already learn robust long-context retrieval during pre-training. Accordingly, QK-Restore yields only marginal gains, consistent with a near-saturation regime where little capability is lost. To investigate the underlying mechanism, we measure the attention gradient decay on Qwen2.5-3B, shown in Figure 7. Unlike HypeNet-2B, whose full-attention layers exhibit a well-fitted exponential decay, it has a clear near-flat tail. We hypothesize that long-context recall in pure softmax-attention models is maintained structurally, via the abundance of full-attention layers.

### M. Method of Identifying Retrieval Layers in Pure Softmax-Attention Models

To identify which softmax-attention layers are primarily responsible for long-range retrieval, we conduct a leave-one-out ablation study over the set 𝐿attn. For each layer ℓ ∈ 𝐿attn, we construct an ablated model M(−ℓ) by setting 𝑾𝑄ℓ = 𝑾𝐾ℓ = 0. Each ablated model is evaluated on several retrieval benchmarks (Ruler-QA-SQuAD (Hsieh et al., 2024), FDA (Arora et al., 2023), SWDE (Lockard et al., 2019)). The importance score of layer ℓ is defined as:

𝛿Recall(ℓ) = S(M) − S(M(−ℓ)),

where S(·) denotes the average retrieval accuracy across benchmarks. A large 𝛿Recall(ℓ) indicates that layer ℓ contributes disproportionately to retrieval and that its routing weights are critical; a near-zero score indicates redundancy. Layers are ranked by 𝛿Recall in descending order, and the top-𝑘 layers are selected as the target set for weight restoration. In our experiments, we keep 𝑘 = 𝐿attn/4 as the same as in HypeNet.

### N. Details of deriving QK-Pro

The bilinear constraint 𝑾𝑄new𝑾new⊤

𝐾 = 𝑹pre has no closed-form joint solution, so we linearise by fixing one factor to its pre-SFT value, making the remaining problem a standard constrained least-squares

system solvable via Lagrange multipliers. We fix 𝑾𝐾new = 𝑾𝐾pre, find minimum-deviation 𝑾𝑄new:

𝑾𝑄new − 𝑾𝑄post

(91)

min

𝑾𝑄new

𝐹

s.t. 𝑾𝑄new𝑾𝐾new⊤ = 𝑹pre (92) Then applying the Lagrange multipliers:

L = 𝑿 − 𝑾𝑄post

+ tr(Λ⊤(𝑿𝑾𝐾new⊤ − 𝑹pre)) (93)

𝐹

Taking the derivative 𝜕L/𝜕𝑿 = 0, and we get:

𝑾𝑄new = 𝑾𝑄post + 𝑹pre − 𝑾𝑄post𝑾𝐾pre⊤ 𝑾𝐾pre𝑾𝐾pre⊤ + 𝜆𝑰 −1𝑾𝐾pre (94)

