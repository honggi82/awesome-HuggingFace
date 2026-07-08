## Demystifying When Pruning Works via Representation Hierarchies

Shwai He1 Guoheng Sun1 Haichao Zhang2 Yun Fu2 Ang Li1

# arXiv:2603.24652v3[cs.CL]11May2026

### Abstract

Network pruning, which removes less important parameters or architectures, is often expected to improve efficiency while preserving performance. However, this expectation does not consistently hold across language tasks: pruned models can perform well on non-generative tasks but frequently fail in generative settings. To demystify how such discrepancies arise under pruning, we analyze network pruning from a representationhierarchy perspective, decomposing the internal computation of language models into three sequential spaces: embedding (hidden representations), logit (pre-softmax outputs), and probability (post-softmax distributions). While representations in the embedding and logit spaces are largely robust to pruning-induced perturbations, the subsequent nonlinear transformation from logits to the probability space amplifies such deviations, whose persistence across time steps leads to substantial degradation during generation. By contrast, the stability of the categorical-token probability subspace, together with the robustness of the embedding space, supports the effectiveness of pruning for non-generative tasks such as retrieval and multiple-choice classification. Our representation-level analysis disentangles the effects of pruning across tasks and offers practical guidance for applying pruning effectively. The code is available in the project repository.

### 1. Introduction

Network pruning (Kusupati et al., 2020; Zhuang et al., 2020; Sun et al., 2023) is an effective approach for improving computational efficiency by removing less important parameters or architectures. As large language models continue to grow in scale (OpenAI, 2024; DeepSeek-AI, 2024; Team, 2025), compression via network pruning has become an increasingly attractive strategy for mitigating memory and computational costs.

1University of Maryland, College Park, USA 2Northeastern University, USA. Correspondence to: Shwai He <shwaihe@umd.edu>.

HumanEval

GSM8K

75

Attn Drop

Attn Drop

60

MLP Drop ShortGPT

MLP Drop ShortGPT

Performance(%)

50

40

25

20

0

0

0 2 4 6 8 10 12 Dropped Layers

0 2 4 6 8 10 12 Dropped Layers

(a) Generative tasks.

HellaSwag

MMLU

75

75

Performance(%)

Performance(%)

50

50

25

25

Attn Drop

Attn Drop

MLP Drop ShortGPT

MLP Drop ShortGPT

0

0

0 2 4 6 8 10 12 Dropped Layers

0 2 4 6 8 10 12 Dropped Layers

(b) Non-generative tasks.

Figure 1. Effect of inter-layer pruning on generative and nongenerative tasks. Inter-layer pruning is implemented by removing entire transformer blocks (ShortGPT (Men et al., 2025)) or attention/MLP layers (Attn/MLP Drop (He et al., 2026)).

However, as illustrated in Figure 1, the effectiveness of network pruning does not hold uniformly across language tasks (He et al., 2026). Empirically, pruned models often retain strong performance on non-generative tasks (Hendrycks et al., 2021; Zellers et al., 2019), which primarily depend on sequence-level representations or logits over a fixed set of categorical options, but frequently fail on generative tasks (Cobbe et al., 2021; Chen et al., 2021), where models generate output sequences by sampling from predicted probability distributions.

To investigate the root cause of this discrepancy, we analyze pruning from the perspective of internal representation transformations in language models. Specifically, we decompose model computation along the inference pipeline into three sequential spaces: embedding (hidden representations), logit

(pre-softmax outputs), and probability (post-softmax distributions). This decomposition naturally aligns with the distinct representational spaces involved in non-generative and generative tasks, while also providing a clear framework for tracing how pruning-induced perturbations propagate across different stages of the model and affect downstream performance.

Our empirical analyses reveal a clear representation hierarchy under pruning. The embedding space remains largely robust, exhibiting only minor deviations even when a substantial fraction of parameters is removed, consistent with prior findings (Gromov et al., 2024; He et al., 2026). Interestingly, the subsequent linear transformation from the embedding space to the logit space preserves comparable representational similarity.

In contrast, our empirical and theoretical analyses show that the nonlinear projection from logits to probabilities (Xuan et al., 2025) amplifies pruning-induced perturbations in the probability space, leading to disproportionately large deviations in the output distribution and ultimately destabilizing the generation process. These deviations persist across generation steps, further resulting in substantial degradation of generation quality. By contrast, non-generative tasks typically rely on the logits or probabilities of a small set of predefined option tokens at a single decision step, which remain comparatively stable under pruning. Together with the robustness of the embedding space, this property explains why network pruning remains effective for non-generative tasks such as retrieval and multiple-choice classification.

Through combined empirical and theoretical analyses, we develop a representation-level understanding of how pruning affects internal representations and why its impact differs across tasks. These findings explain why network pruning remains effective for non-generative tasks but poses substantial risks for generative ones, offering practical guidance for applying pruning. In summary, the contribution of this work is as follows:

- • This work reveals a clear discrepancy in the effectiveness of network pruning across non-generative and generative tasks.
- • For generative tasks, we identify the nonlinear mapping from logits to probabilities as a key mechanism that amplifies pruning-induced perturbations, leading to severe performance degradation.
- • By contrast, low pruning-induced perturbations in the embedding and logit spaces, as well as the stability of the categorical-token probability subspace, support the effectiveness of network pruning in non-generative tasks and provide practical guidance for its application.

### 2. Related Works

Efficiency Challenges in Large Language Models Scaling large language models (LLMs) has recently driven rapid progress across a wide range of tasks, demonstrating strong and increasingly general capabilities (OpenAI, 2024; DeepSeek-AI, 2024; Team, 2025). However, such improvements often come at a substantial efficiency cost: the massive model parameters and the intermediate representations maintained during inference incur significant memory and computational overhead, posing challenges for real-time and resource-constrained deployment. As a result, how to trade off model capability and efficiency has become a central problem in modern LLM systems (Hoffmann et al., 2022; Wan et al., 2024). Importantly, language models exhibit fundamentally different inference behaviors between single-pass settings (e.g., one-step prefilling) and multi-step generation settings, suggesting that the effects of efficient methods like network pruning are inherently regime-dependent.

Model Compression via Network Pruning Network pruning, motivated by the substantial redundancy inherent in large language models, aims to reduce memory footprint and inference cost by removing less important components (Liu et al., 2019; Tanaka et al., 2020; Cheng et al., 2024; Zhang & Fu, 2025). Existing approaches can be broadly categorized into two classes: (i) unstructured weight sparsification (e.g., Wanda (Sun et al., 2023) and SparseGPT (Frantar & Alistarh, 2023)), (ii) structured pruning of coupled structures like layers/blocks (Gromov et al., 2024; He et al., 2026; 2025; Zhang et al., 2025a). These pruning approaches primarily operate in the embedding space and have mainly been shown to succeed on non-generative tasks (Sun

- et al., 2024; Frantar & Alistarh, 2023; Lei et al., 2025; Zhang
- et al., 2025b; He et al., 2024), which typically depend on the model’s hidden representations or logits at a single inference step, without iterative feedback across decoding steps. In contrast, generative tasks pose additional challenges for network pruning. For instance, errors introduced at earlier time steps can propagate to subsequent steps. In this work, we analyze how network pruning affects non-generative and generative tasks differently and uncover the underlying principles for effective pruning.

### 3. Background on Language Modeling

Modern language models process text by mapping discrete tokens to continuous representations, transforming them through multiple continuous latent spaces, and finally producing probability distributions over discrete tokens. Formally, given an input text sequence T , the model first applies a tokenizer τ(·) to map text into discrete tokens, i.e., x = τ(T ) with xi ∈ {1,...,|V|}, where |V| denotes the

|Embeddingℎ| |
|---|---|
| | |
| | |
| | |

Non-linear

LLM Backbone

Amplification

Softmax(𝑻)

Embedding

Tokenizer

Embeddingℎ

LM

Sampled

Layer

Tokens

Logits 𝑧 Prob 𝑝

Input Prompt

[Figure 1]

Head

Token

Pruning

Pruned LLM

Autoregressive Loop (Error Propagation)

Linear/Latent Space (Robust)

Probabilistic Space (Sensitive)

[Figure 2]

|Large Δ𝑝|
|---|

|≈|
|---|

|Δ𝑧|
|---|

|Δℎ|
|---|

≈

[Figure 3]

###### Original Prediction

LLM

Direct use of representations

Non-Generative Tasks (e.g., Retrieve, Classification)

Pruned LLM

Robust

Generation Failure

- Figure 2. Propagation of pruning-induced perturbations across representation spaces in LLMs. Small embedding perturbations ∆h introduced by pruning remain stable in the logit space (i.e., small ∆z), but are amplified by the softmax nonlinearity in the highdimensional probability space, resulting in large probability shifts ∆p and degraded autoregressive generation.

size of the vocabulary. Each token xi is then mapped to a continuous embedding vector through an embedding lookup table E ∈ R|V|×d: e = E[x] ∈ Rd, where d ≪ |V|. The sequence of embeddings e is processed by a deep neural network composed of L layers, yielding a hierarchy of hidden representations:

###### h(l) = f(l) h(l−1) , l = 1,...,L, (1)

where h(0) = e, and f(l)(·) denotes the transformation induced by the l-th layer, which includes the residual connection (He et al., 2015) for simplicity. At the final layer, the hidden state h(L) ∈ Rd is projected onto the vocabulary space through a linear transformation (i.e., LM head projection), yielding the logits z:

z = Wh(L), W ∈ R|V|×d. (2)

The logits are then converted into a probability distribution over the vocabulary via the softmax function with a predefined temperature T:

pt+1 = softmax(zt/T). (3)

The output token at timestep t + 1, denoted as xˆt+1, is sampled according to the predictive distribution pt+1. Figure 2 illustrates the three distinct spaces involved in the LLM inference pipeline and provides an intuitive framework for understanding how pruning-induced perturbations may behave differently across these spaces, as detailed in the subsequent sections.

Generation Tasks The generated token is then mapped back to text via the inverse tokenizer, Tˆt+1 = τ−1(ˆxt+1). During autoregressive generation, the generated token index xˆt+1 is fed back into the language model together with previously generated tokens as historical context, forming a feedback loop that iteratively produces subsequent tokens.

As a result, at decoding step t, the model input consists of both the prompt tokens xprompt0:P and the sequence of modelgenerated tokens xgenP+1:t. While the prompt tokens remain fixed, the generated tokens depend on the model’s past outputs and may therefore differ between the baseline and pruned models, introducing additional sources of deviation during autoregressive decoding.

Non-generative Tasks In non-generative tasks, the model processes the input prompt only once, without subsequent iterative decoding. Under this formulation, retrieval and text classification are representative non-generative tasks, where the model is required to produce either an embedding representation or the probabilities over a small set of candidate tokens (or labels), rather than generating a sequence of output tokens. For instance, in retrieval tasks, the objective is defined directly in the embedding space:

###### S(q,d) = CosineSim(hq,hd), (4)

where hq and hd denote the embedding representations of the query and the document, respectively, typically obtained

- Table 1. Benchmark performance comparison of Mistral models under layer dropping. Results are reported for both non-generative tasks (embedding and multiple-choice benchmarks) and generative tasks. Drop-8A and Drop-8M denote models where 8 attention layers or 8 MLP layers are removed, respectively, while keeping the remaining architecture unchanged.

E5-Mistral Full-Model Drop-8A Drop-8M #Params 7.1B 6.8B 5.7B

###### Embedding Tasks

Arguana 60.9 54.7 58.6 Climate-FEVER 36.8 31.9 38.4 DBPedia 47.9 43.6 44.1 FEVER 87.6 82.9 88.7 FiQA 56.4 50.9 52.8 HotpotQA 74.9 66.8 74.2 NFCorpus 38.1 35.4 36.9 NQ 66.3 56.1 65.4 Quora 88.6 86.5 88.2 SCIDOCS 16.2 12.4 14.7 SciFact 75.8 71.4 73.6 TREC-COVID 85.9 84.3 79.6 Touche-2020 22.9 18.1 18.7

Average 58.9 53.4 56.8 (a) Retrieval performance of E5-Mistral (Wang et al., 2024).

Mistral-7B-Instruct Full-Model Drop-8A Drop-8M #Params 7.1B 6.8B 5.7B

###### Multi-choice Tasks

BoolQ 85.9 86.0 78.2 MMLU 62.1 62.0 59.1 OpenBookQA 47.0 46.8 41.2 RTE 72.9 74.0 72.1 Winogrande 78.8 80.0 71.1

Average 69.3 69.8 64.3 Generation Tasks

GSM8K 48.4 36.2 0.0 HumanEval 4.9 0.0 0.0 MBPP 13.8 0.4 0.0 NarrativeQA 16.3 9.6 2.0 NQ-Open 27.9 20.9 2.0

Average 22.3 13.2 0.8 (b) Benchmarks of Mistral-7B (Jiang et al., 2023).

from the final-layer hidden states of the model through a pooling or projection operation. Another representative nongenerative task is multiple-choice classification, where only the probabilities associated with a limited number of candidate tokens or options are considered (e.g., the A/B/C/D choices):

p(j | x), (5)

yˆ = arg max

j∈C

where C ⊂ {1,...,|V|} denotes the candidate token set. In practice, |C| ≪ |V|; for example, there may be only four candidate options compared to the full vocabulary. Therefore, non-generative tasks do not involve iterative autoregressive decoding, and the output space they operate on is significantly smaller than the model’s full vocabulary space.

### 4. Inconsistent Effects of Pruning

Overview of Pruning Strategies Network pruning is typically conducted at two levels: (1) fine-grained intra-layer pruning and (2) coarse-grained inter-layer pruning. The former removes less important parameters within individual layers, leading to sparse representations (Sun et al., 2023; Frantar & Alistarh, 2023), where the induced sparsity can be either structured or unstructured. The latter assesses the importance of each layer as a whole and removes less critical transformer blocks (Gromov et al., 2024; Men et al., 2025) or layers (He et al., 2026), motivated by the observation that layers at different depths contribute unequally to overall model performance. In this work, we adopt Wanda (Sun et al., 2023) and SparseGPT (Frantar & Alistarh, 2023) as representative intra-layer methods, and Attention/MLP Drop (He et al., 2026) and ShortGPT (Men et al., 2025) as representative inter-layer methods.

Divergent Effectiveness Across Tasks To examine how pruning affects performance across different task types, we evaluate the same model architecture across both generative and non-generative tasks. This comparison allows us to isolate whether pruning mainly preserves single-step decision quality or also maintains stable multi-step generation behavior. Table 1 compares the performance of the Mistral models (Jiang et al., 2023) under these two task settings. After dropping eight attention or MLP layers, Mistral exhibits markedly different behaviors: while its performance on multiple-choice and retrieval tasks remains largely comparable to that of the original model, its performance on generative tasks collapses significantly. E5-Mistral, evaluated on retrieval as another non-generative setting, also maintains competitive performance after substantial parameter removal. A comparable discrepancy is also observed for intra-layer pruning, as illustrated in Figure 3, where increasing sparsity likewise leads to a pronounced performance degradation in generative tasks. Table 2 further highlights that pruning can fundamentally compromise the model’s text generation behavior. Additional consistent results are provided in Appendix G.

The discrepancy between generative and non-generative tasks may stem from three key factors: (1) Representation Dimensionality: generative tasks operate in a substantially higher-dimensional output space, as the vocabulary size |V| far exceeds the embedding dimension d or the number of candidate labels k involved in non-generative tasks. (2) Nonlinear Projection: the nonlinear mapping from latent representations to token probabilities can further amplify pruning-induced perturbations. (3) Error Propagation: the autoregressive generation process causes errors introduced at early steps to propagate and accumulate over time.

Baseline

Unstructured

4:8

2:4

| |
|---|

| |
|---|

| |
|---|

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

80

Performance(%)

Performance(%)

60

60

40

40

20

20

0

0

Qwen2.5-7B Llama3-8B Mistral-7B

Qwen2.5-7B Llama3-8B Mistral-7B

GSM8K

HellaSwag

- Figure 3. Impact of intra-layer pruning on non-generative and generative tasks for the default Qwen-2.5-7B-Instruct model, namely HellaSwag (Zellers et al., 2019) and GSM8K (Cobbe et al., 2021). Results are reported using Wanda (Sun et al., 2023) under unstructured (50%), 4:8, and 2:4 (Zhou et al., 2021) sparsity patterns.

- 5. Hierarchical Effects of Pruning

Given that non-generative and generative tasks are conducted in different representation spaces, we next analyze how the representations shift after compression, using Qwen-2.5-7B-Instruct as the default model.

Specifically, at each decoding step, we run the baseline model on the current context. We then replace only the current layer with its pruned counterpart during the forward pass, while keeping all other layers unchanged, and measure the induced shift at that layer. Repeating this procedure across layers and decoding steps allows us to compare deviations under a shared dense-model context, without confounding effects from history differences caused by fully running the pruned model. Following Gromov et al. (2024); He et al. (2026), we quantify the impact of pruning using the deviation between the two outputs, measured by angular deviation, 1−CosineSim(hl,hl+∆hl), where hl denotes the output of the l-th layer and ∆hl represents the perturbation introduced by pruning. CosineSim measures the directional alignment between vectors and aligns well with the objectives of many language tasks, e.g., embedding similarity in retrieval and the relative ordering of logits underlying the arg max decision in multiple-choice classification.

To examine pruning-induced deviations across representation spaces, we further derive logits (z(l) = Wh(l)) and probabilities (p(l) = softmax(z(l)/T)) from the embedding representations, and measure the deviations in each space, thereby characterizing how the same pruning-induced perturbation evolves across representation spaces.

- Figure 4 reports the impact of layer dropping on the latent cosine similarity in three different spaces for each attention and MLP layer, measured over multiple prompts (detailed in Appendix F) and generation steps. The embedding space remains largely stable with consistently high similarity, except

at the first and last layers. However, the probability space exhibits substantial fluctuations under pruning despite comparable embeddings. Similar phenomena are observed when pruning a subset of parameters within individual layers, as shown in Appendix G. Notably, the logit space maintains similarity comparable to the embedding space, suggesting that the performance gap between non-generative and generative tasks cannot be simply explained by the increase in representational dimensionality from embeddings to logits.

### 6. Representation-level Analysis

Empirically, we observe distinct behaviors in the embedding, logit, and probability spaces, which cannot be explained solely by dimensionality differences. In this section, we analyze how pruning-induced perturbations propagate across representation spaces. Leveraging the localized nature of layer-wise deviations (Gromov et al., 2024; He et al., 2026), we adopt a Taylor-based local analysis to study how these perturbations are transformed and amplified.

##### 6.1. LM Head Preserves Similarity

Theorem 1 (Local Deviation Induced by Pruning) For cosine similarity in the embedding space, the deviation can be approximately characterized using a second-order Taylor expansion (detailed in Appendix D.1) as follows:

∥∆h⊥∥2 2∥h∥2

, (6)

1 − CosineSim(h,h + ∆h) ≈

where ∆h⊥ denotes the component of ∆h orthogonal to h (i.e., ∆h = ∆h∥ + ∆h⊥). This formulation holds under the assumption that ∆h⊥ is sufficiently small and confined to a local neighborhood, an assumption that holds for most layers, with the exception of the first and last layers.

By construction, ∥∆h⊥∥2 ≤ ∥∆h∥2, and in practice ∥∆h∥2 is typically much smaller than ∥h∥2 in a single layer. This

- Table 2. Generated output examples of Qwen-2.5-7B-Instruct under inter-layer pruning (Attention/MLP Drop (He et al., 2026)). Drop-kA and Drop-kM denote removing k attention layers or k MLP layers, respectively. While moderate pruning preserves correct generation, heavier pruning causes severe generation breakdown, including incoherent and repetitive outputs.

Question Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in

May. How many clips did Natalia sell altogether in April and May? Method Output Baseline Natalia sold 48 clips in April. In May, she sold half as many clips as in April, which

amounts to 24 clips. Therefore, the total number of clips sold across April and May is 48 + 24 = 72.

Drop-4A Natalia sold 48 clips in April and half as many in May, which is 24 clips. Therefore, the total number of clips sold over April and May is 48 + 24 = 72.

Drop-8A Let’s denote the number of clips Alicia, the ILUNNIE M ’<%=>t \[K1205359752000 1 61.72510155125556460445874923052e-08T.1000.0 01.10115651362430000.2023-01-02 23:21:07 202277447967445715556447

Drop-4M Natalia sold 48 clips in April and half that amount in May (24 clips), resulting in a total of 48 + 24 = 72 clips sold across the two months.

Drop-8M To calculate the total number of clips, we are adding the result of first you and your a year and then the second or your and your a year and your and your and your and your and your and your...

explains why the cosine similarity in the embedding space often remains high when perturbations are introduced at a single layer, and this phenomenon can further extend to the logit space, i.e.,

∥∆z⊥∥2 2∥z∥2

. (7)

1 − CosineSim(z,z + ∆z) ≈

Figures 16 and 17 compare the ground-truth and estimated cosine similarities, demonstrating the effectiveness of the proposed approximation in capturing local behavior. These formulations indicate that the relative magnitude of orthogonal components (i.e., relative orthogonal magnitude) plays a critical role in determining the similarity.

- Figure 5 and Figure 18 compare the relative orthogonal magnitude in the embedding and logit spaces, showing that the magnitude is significantly reduced after passing through the LM head. This suggests that pruning-induced perturbations remain limited in the logit space, consistent with comparable logit similarity before and after pruning.

##### 6.2. Nonlinear Softmax Amplifies Deviation

The softmax operation is the process that converts continuous logits into probability distributions. We further investigate how this nonlinear transformation amplifies differences, even when the underlying logits remain relatively similar.

Theorem 2 (Sensitivity of Probability Space to Logit Perturbations) To ensure comparability between deviations in the probability space and the logit space, we represent the deviation in terms of the logit variable z, instead of directly using Theorem 1. Similarly, using a second-order Taylor expansion (detailed in Appendix D.2), the cosine similarity in the probability space can be approximated as follows:

p2i ∥p∥2

Varr(∆z) 2T2

. (8)

1 − CosineSim(p, p + ∆p) ≈

, ri =

This indicates that the deviation is dominated by the temperature T and the weighted variance of ∆z, which incorporates contributions from both the orthogonal component ∆z⊥ and the parallel component ∆z∥. Notably, the variance of ∆z is substantial relative to the orthogonal magnitude ratio, especially in the last layers, which leads to pronounced deviations in the probability space. This effect is illustrated by the absolute values in Figure 19 and by the relative values normalized by the corresponding magnitude ratios in Figures 20 and 21. The temperature T is set to 1.0 by default, and the visualization exhibits consistent behavior for other temperature settings as detailed in Appendix H.

Figure 6a compares the ground-truth and estimated cosine similarity in the vocabulary space at the 14th attention layer; results across all depths are provided in Figure 15. Their close match suggests that our theorem captures the primary source of pruning-induced deviation.

1.0

0.7

CosineSim

0.4

Embedding

0.1

Logit

0.2

Probability

0.5

0 4 8 12 16 20 24

Layer Index

(a) Attention.

1.0

0.7

CosineSim

0.4

Embedding

Logit

0.1

Probability

0.2

0 5 10 15 20 25

Layer Index

(b) MLP.

Figure 4. Representation similarity across three spaces when each layer is individually dropped for the Qwen-2.5-7B-Instruct model, with layer dropping performed. Mean values are shown as curves and min–max ranges as shaded areas.

Theorem 3 (Distributional Shift under Pruning) In the probability space, KL divergence quantifies pruninginduced distributional shifts. From Appendix C,

Vari∼p(∆zi) 2T2

, (9)

KL(p∥q) ≈

where q = p + ∆p. Tokens with higher predicted probabilities contribute more substantially to the divergence.

- Figure 6b further compares the ground-truth and estimated KL divergence. The estimated trend closely aligns with the ground-truth values, providing strong empirical support for our analysis. Moreover, the large KL divergence highlights the pronounced discrepancy between the outputs of the original and pruned models, offering a clear explanation for the observed collapse in generative performance after pruning. Our proposed theorems naturally extend from pruning to quantization, as both generally stem from compressioninduced errors. A detailed comparison with quantization is presented in Appendix I.

### 7. Multi-Scale Effects of Pruning

We next analyze the multi-scale behavior of network pruning across generation time steps in generative tasks and

×10 2

z 2 2 z 2

h 2 2 h 2

RelativeMagnitude

6.0

4.0

2.0

0.0

0 50 100 150

Generation Step

Figure 5. Relative orthogonal magnitude in the embedding space (h) and the logit space (z) at the 14th attention layer under layer dropping.

across probability subspaces in non-generative multiplechoice tasks. We use Qwen-2.5-7B-Instruct with eight attention layers removed as the pruned model and compare it to the uncompressed baseline. In this setting, the pruned model performs comparably on non-generative tasks but fails on generative tasks. At the same time, this setting provides insight into the joint effects of pruning multiple layers.

##### 7.1. Persistent Divergence in Generation

We analyze how the similarity between the final outputs before and after pruning varies across different generation steps in Figure 7, using the same prompt as in Table 2. For all feature spaces, in Figure 7a, we observe that the cosine similarity at the first step remains significantly higher than at later steps. This supports the effectiveness of pruning on non-generative tasks, which typically rely on either the embedding or the logits at the first decoding step.

However, generative tasks involve iterative decoding, where deviations introduced at earlier steps persist and propagate to subsequent steps, potentially leading to generation collapse within only a few iterations. Based on Equations (8) and (9), the variance of ∆z emerges as the dominant factor governing this deviation. During generation, this variance can be attributed to two sources: (1) errors induced by network pruning through perturbed model parameters, which directly affect the processing of the current token, and (2) compounded errors propagated through historical states, e.g., the key–value cache (Pope et al., 2023), from previous decoding steps. As detailed in Appendix E, the latter is further amplified during generative tasks: beyond the prompt tokens (xprompt0:P ), which are identical for the baseline and pruned models, differences in sampled tokens (xgenP+1:t) lead the models to condition on diverging histories, thereby progressively enlarging the deviation. This is consistent with Figure 7b, where the first step shows low deviation because both models receive the same prompt tokens, whereas in subsequent steps, differences in previously generated tokens lead to sharp increases in deviation.

Under the combined effect of these factors, pruning induces

×10 1

Groundtruth Estimated

1-CosineSim

3.0

1.5

0.0

0 50 100 150

Generation Step

(a) Angular Deviation.

×10 1

Groundtruth Estimated

KLdivergence

- 2

- 3

0

0 40 80 120 160

Generation Step

(b) KL divergence.

- Figure 6. Comparison between the ground-truth values and the theoretical estimates across generation steps at the 14th attention layer under layer dropping, measured by (a) angular deviation and (b) KL divergence.

0 40 80 120 160 200

Generation Step

-0.2

0.0

0.2

0.5

0.8

CosineSim

Logit Embedding

(a) Embedding and Logit Spaces.

0.0

0.3

0.6

CosineSim

0 40 80 120 160 200

Generation Step

0

10

20

30

40

KLdivergence

(b) Probability Space.

- Figure 7. Representation similarities across different spaces between the outputs of the baseline and pruned models (Drop8A) across generation steps for the default Qwen-2.5-7B-Instruct model. Outliers in the probability space at later decoding steps primarily correspond to predictions involving special tokens.

\n \n\n To Let The First We If

TopTokens

Given ** In E

Baseline

Pruned

0.0 0.1 0.2 0.3 0.4 0.5

Probability

(a) Probability of top tokens sampled from distribution p.

Baseline

- A

- B

- C

- D

Pruned

CategoryTokens

14 12 10 8 6 4 2 0

Log Likelihood

(b) Log-likelihood of category tokens.

Figure 8. Comparison between the outputs of the full and pruned models in different subspaces under Drop-8A on multiple-choice prompts: (a) probabilities of top-probability tokens sampled from the full vocabulary, and (b) log-likelihoods restricted to the category-token subspace.

persistently high divergence across decoding steps, leading to a substantially more pronounced degradation in generative tasks than in non-generative ones.

##### 7.2. Robustness of Probability Subspaces

In contrast to generative tasks, which rely on predictions over the entire vocabulary, non-generative multiple-choice tasks depend on only a small subset of the vocabulary (e.g., categorical options such as A/B/C/D). Motivated by this distinction, we shift our analysis to the probability subspace for a more fine-grained examination.

For multiple-choice prompts, Figure 8 illustrates both the probabilities of the top-predicted tokens and the log-

likelihoods of the categorical candidate tokens. Notably, these candidate tokens do not appear among the topprobability tokens in most cases; instead, they lie in the tail of the distribution, where probability shifts are substantially milder than those observed for the top-ranked tokens. Therefore, despite the large discrepancies observed in the top-token probabilities, the log-likelihood over the relevant categorical subset exhibits a similar trend and often preserves the same argmax token, which is consistent with the robustness of non-generative tasks under pruning.

### 8. Discussion of Effective Pruning

Network pruning exhibits inconsistent effectiveness across tasks, making it crucial to understand when and why pruning succeeds. Our representation-level analysis shows how pruning-induced perturbations evolve across representation spaces and how this evolution shapes task robustness. We summarize several key factors that jointly shape postpruning performance.

Representation Space. Pruning-induced perturbations differ across representation spaces. Embedding and logit spaces are relatively robust, making tasks that operate directly on them more amenable to pruning.

Task-Relevant Subspace. Although the probability space spans the full vocabulary, many tasks depend only on lowdimensional or task-specific subspaces. Even when global probability distributions shift, these subspaces can remain stable, preserving predictions.

Temporal Dependence. In autoregressive generation, pruning errors compound over time due to temporal dependence. In contrast, tasks without temporal dependence (e.g., singlestep classification) avoid this amplification and are therefore more robust to pruning.

Beyond Training-Free Pruning. Our study focuses on training-free pruning. Post-training or fine-tuning after pruning offers a complementary approach to mitigate pruninginduced collapse, which we leave for future work.

### 9. Conclusion

In this work, we show that large language models exhibit task-dependent robustness to network pruning, performing well on non-generative tasks while often failing in generative settings. Through empirical and theoretical analyses from a representation-hierarchy perspective, we identify how pruning robustness varies across representation spaces, providing practical guidance for the effective application of network pruning.

### Impact Statement

This work examines the robustness of large language models under pruning and highlights a discrepancy between generative and non-generative tasks. By analyzing embeddings, logits, and probabilities, we show that pruning mainly affects the probability space, explaining why generation degrades while non-generative performance remains stable. These findings guide pruning and evaluation in appropriate task settings. Potential risks include treating non-generative performance as a proxy for generative robustness; taskaware evaluation and careful deployment across target use cases can reduce this risk.

### Acknowledgments

We sincerely thank Dr. Hong Cai and Dr. Mingu Lee for their valuable technical discussions that contributed to this work. We also gratefully acknowledge the Qualcomm Innovation Fellowship 2025 for supporting the authors during the course of this research.

### References

Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Cheng, H., Zhang, M., and Shi, J. Q. A survey on deep neural network pruning: Taxonomy, comparison, analysis, and recommendations. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):10558–10578, 2024. doi: 10.1109/TPAMI.2024.3447085.

Cobbe, K., Kosaraju, V., Bavarian, M., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

DeepSeek-AI. Deepseek-v3 technical report, 2024. Frantar, E. and Alistarh, D. Sparsegpt: Massive language

models can be accurately pruned in one-shot, 2023. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., et al. The llama 3 herd of models, 2024.

Gromov, A., Tirumala, K., Shapourian, H., Glorioso, P., and Roberts, D. A. The unreasonable ineffectiveness of the deeper layers, 2024.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385, 2015.

He, S., Dong, D., Ding, L., and Li, A. Towards efficient mixture of experts: A holistic study of compression techniques, 2024.

He, S., Deng, C., Li, A., and Yan, S. Understanding and harnessing sparsity in unified multimodal models, 2025.

He, S., Sun, G., Shen, Z., and Li, A. Uncovering the redundancy in transformers via a unified study of layer dropping. Transactions on Machine Learning Research, 2026. ISSN 2835-8856.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

Hoffmann, J., Borgeaud, S., Mensch, A., et al. Training compute-optimal large language models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA,

- 2022. Curran Associates Inc. ISBN 9781713871088.

Jiang, A. Q., Sablayrolles, A., Mensch, A., et al. Mistral 7b,

- 2023.

Kusupati, A., Ramanujan, V., Somani, R., et al. Soft threshold weight reparameterization for learnable sparsity. In Proceedings of the International Conference on Machine Learning, July 2020.

Lei, Y., He, S., Li, A., and Yates, A. Making large language models efficient dense retrievers, 2025.

Liu, Z., Sun, M., Zhou, T., Huang, G., and Darrell, T. Rethinking the value of network pruning. In International Conference on Learning Representations, 2019.

Men, X., Xu, M., Zhang, Q., et al. Shortgpt: Layers in large language models are more redundant than you expect, 2025.

OpenAI. Gpt-4 technical report, 2024. Pope, R., Douglas, S., Chowdhery, A., et al. Efficiently

scaling transformer inference. Proceedings of machine learning and systems, 5:606–624, 2023.

Raffel, C., Shazeer, N., Roberts, A., et al. Exploring the limits of transfer learning with a unified text-to-text transformer, 2023.

Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and effective pruning approach for large language models. arXiv preprint arXiv:2306.11695, 2023.

Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and effective pruning approach for large language models. In The Twelfth International Conference on Learning Representations, 2024.

Tanaka, H., Kunin, D., Yamins, D. L., and Ganguli, S. Pruning neural networks without any data by iteratively conserving synaptic flow. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 6377–6389. Curran Associates, Inc., 2020.

Team, K. Kimi k2: Open agentic intelligence, 2025.

Thakur, N., Reimers, N., R¨uckl´e, A., Srivastava, A., and Gurevych, I. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Wan, Z., Wang, X., Liu, C., et al. Efficient large language models: A survey. Transactions on Machine Learning Research, 2024. ISSN 2835-8856.

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., and Wei, F. Multilingual e5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672, 2024.

Xuan, H., Yang, B., and Li, X. Exploring the impact of temperature scaling in softmax for classification and adversarial robustness. arXiv preprint arXiv:2502.20604, 2025.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., et al. Qwen3 technical report, 2025.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.

Zhang, H. and Fu, Y. Vqtoken: Neural discrete token representation learning for extreme token reduction in video large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Zhang, H., Chai, W., He, S., Li, A., and Fu, Y. Dense video understanding with gated residual tokenization. arXiv preprint arXiv:2509.14199, 2025a.

Zhang, H., Lu, Y., Wang, L., Li, Y., Chen, D., Xu, Y., and Fu, Y. Linkedout: Linking world knowledge representation out of video llm for next-generation video recommendation. arXiv preprint arXiv:2512.16891, 2025b.

Zhou, A., Ma, Y., Zhu, J., Liu, J., Zhang, Z., Yuan, K., Sun, W., and Li, H. Learning n: M fine-grained structured sparse neural networks from scratch. In International Conference on Learning Representations, 2021.

Zhuang, T., Zhang, Z., Huang, Y., Zeng, X., Shuang, K., and Li, X. Neuron-level structured pruning using polarization regularizer. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 9865– 9877. Curran Associates, Inc., 2020.

### A. Implementation Details

Models. Our main experiments use Qwen-2.5-7B-Instruct as the primary model. Additional experiments are conducted on Mistral-7B-Instruct (Jiang et al., 2023), LLaMA-3-8B (Grattafiori et al., 2024), and Qwen3-4B (Yang et al., 2025) to verify cross-model generality.

Intra-layer Pruning. We adopt Wanda (Sun et al., 2023) and SparseGPT (Frantar & Alistarh, 2023) for intra-layer pruning. All intra-layer experiments use 50% sparsity with three sparsity patterns: unstructured (50%), semi-structured 4:8, and semi-structured 2:4 (Zhou et al., 2021). Pruning masks are computed using 128 randomly sampled C4 (Raffel et al., 2023) sequences as calibration data, following the standard setup of each method.

Inter-layer Pruning. For inter-layer pruning, we adopt layer dropping to remove individual attention or MLP layers, following Layer Drop (He et al., 2026), and ShortGPT (Men et al., 2025) to remove entire transformer blocks. The calibration data follows the same protocol as in intra-layer pruning, using 128 randomly sampled C4 (Raffel et al., 2023) sequences.

Evaluation. Generative tasks are evaluated on GSM8K (Cobbe et al., 2021), HumanEval (Chen et al., 2021), MBPP (3-shot), NarrativeQA, and NQ-Open. Non-generative tasks include multiple-choice benchmarks such as HellaSwag (Zellers et al., 2019), MMLU (Hendrycks et al., 2021), BoolQ, ARC-Challenge, OpenBookQA, WinoGrande, and RTE, all evaluated via log-likelihood over candidate options. Retrieval is another non-generative setting and is evaluated on the BEIR benchmark (Thakur et al., 2021) with E5-Mistral (Wang et al., 2024).

### B. Preliminaries for Theoretical Analysis

Model compression (such as pruning or structural modification) slightly perturbs the model parameters, which in turn induces a shift in the logits. To analyze how this perturbation affects the model’s prediction behavior, we compare the output probability distributions before and after compression.

- Let p = softmax(z/T) and q = softmax((z + ∆z)/T), where p denotes the original output distribution of the model, q denotes the output distribution after compression, z ∈ RV is the original logits, ∆z ∈ RV represents the logit perturbation introduced by compression, and T is the temperature. The resulting distributional change is

∆p = q − p, (10) which measures how compression shifts the vocabulary probability distribution. Explicitly, we have

ez

i/T

e(z

i+∆zi)/T

j e(zj+∆zj)/T . (11)

pi =

j ezj/T , qi =

### C. Approximation of KL Divergence in Probability Space We begin with the definition of qi:

ez

i/Te∆z

i/T

. (12)

qi =

V j=1 ezj/Te∆zj/T

Using the original distribution

pi =

we rewrite ez

i/T = piS, substituting yields

and cancelling S leads to

V

ez

i/T

, S =

V k=1 ezk/T

k=1

ez

k/T, (13)

(piS)e∆z

i/T

, (14)

qi =

S Vj=1 pje∆zj/T

|qi =<br><br>pie∆z<br><br>i/T<br><br>V j=1 pje∆zj/T|
|---|

. (15)

Finally, expressing the denominator as an expectation under p,

V

pje∆z

j/T = Ej∼p e∆z

j/T , (16)

j=1

we obtain the exact reweighted closed form:

|qi =<br><br>pie∆z<br><br>i/T<br><br>Ej∼p e∆zj/T|
|---|

. (17)

From the above, we immediately obtain

e∆z

i/T

qi pi

Ej∼p e∆zj/T . (18) Equivalently,

=

|log<br><br>qi pi<br><br>=<br><br>∆zi T − log Ej∼p e∆z<br><br>j/T|
|---|

. (19)

This formula is crucial because it compresses the nonlinearity of softmax into a single log–sum–exp (expectation) term. By definition,

V

pi qi

. (20)

KL(p∥q) =

pi log

i=1

From Equation (19),

pi qi

∆zi T

+ log Ej∼p e∆z

j/T . (21)

= −

log

Since i pi = 1, the closed-form expression is

|KL(p∥q) = −<br><br>1 T<br><br>Ei∼p[∆zi] + log Ei∼p e∆zi/T|
|---|

. (22)

Define Xi = ∆z

T , so that Equation (22) becomes KL(p∥q) = −Ep[X] + log Ep[eX].

i

We expand the log-moment term and apply expectation under p,

Ep[eX] = 1 + µ +

- 1

- 2

m2 + O(∥X∥3), (23)

µ = Ep[X], m2 = Ep[X2]. (24) Applying log(1 + u) = u − u

2

2 + O(u3) with u = µ + 12m2 + O(∥X∥3) gives

- 1

- 2

- 1

- 2

log Ep[eX] ≈ µ +

(m2 − µ2) = µ +

Varp(X). (25)

Substituting back,

KL(p∥q) ≈ −µ + µ +

- 1

- 2

Varp(X) =

- 1

- 2

Varp(X). (26)

##### Recalling Xi = ∆zi/T, we obtain Theorem 3 (Distributional Shift under Pruning):

|KL(p∥q) ≈<br><br>1<br><br>2T2<br><br><br>Vari∼p(∆zi)|
|---|

. (27)

- D. Approximation of Deviation via Angular Deviation We analyze the second–order sensitivity of cosine similarity between two probability vectors p and q. By definition,

CosineSim(p,q) =

- Let q = p + ∆p, and expand with respect to ∆p. The numerator and denominator are as follows:

p⊤q ∥p∥∥q∥

. (28)

p⊤q = p⊤(p + ∆p) = ∥p∥2 + p⊤∆p. (29)

∥q∥2 = ∥p∥2 + 2p⊤∆p + ∥∆p∥2. (30) Taking the square root and applying a second–order Taylor expansion gives

Hence

p⊤∆p ∥p∥2

+ ∥∆p∥2 2∥p∥2

∥q∥ = ∥p∥(1+

(p⊤∆p)2 2∥p∥4

+ O(∥∆p∥3)). (31)

−

1 + p∥⊤p∆∥2p 1 + p∥⊤p∆∥2p + ∥2∆∥pp∥∥22 − (p2⊤∥p∆∥p4)2 + O(∥∆p∥3)

. (32)

CosineSim(p, q) =

We now expand the reciprocal 1+1u = 1 − u + u2 + O(u3) and keep terms up to second order. After simplification, all first–order terms cancel out, giving

- 1

- 2

CosineSim(p, q) = 1 −

∥∆p∥2 ∥p∥2

−

(p⊤∆p)2 ∥p∥4

+ O(∥∆p∥3). (33)

Thus the second–order deviation is

|1 − CosineSim(p, q) =<br><br>1<br><br>2<br><br><br>∥∆p∥2 ∥p∥2<br><br>−<br><br>(p⊤∆p)2 ∥p∥4<br><br>+ O(∥∆p∥3)|
|---|

. (34)

Note the identity

pp⊤ ∥p∥2

(p⊤∆p)2 ∥p∥2

= ∆p⊤ I −

∥∆p∥2 −

∆p. (35)

Define the orthogonal projection matrix

pp⊤ ∥p∥2

. (36)

P⊥ = I −

Substituting P⊥ into the second-order deviation formula above gives the compact form

|1 − CosineSim(p, q) =<br><br>1<br><br>2∥p∥2<br><br><br>∆p⊤P⊥∆p + O(∥∆p∥3)|
|---|

. (37)

##### D.1. Angular Deviation Estimation via Perturbation Decomposition

We next consider the case where cosine similarity is computed directly on logits without passing through a softmax transformation, i.e., between z and z + ∆z. We show that its second–order behavior shares the same mathematical structure as the softmax case, but with uniform weighting instead of probability–dependent reweighting.

Similarly, Equation (37) can be interpreted as follows:

|1 − CosineSim(z, z + ∆z) ≈<br><br>∆z⊤Z⊥∆z 2∥z∥2<br><br>|
|---|

. (38)

We also define the orthogonal projection matrix

Z⊥ = I −

zz⊤ ∥z∥2

. (39)

To make the role of Z⊥ explicit, we decompose the perturbation ∆z into the component parallel to z and the component orthogonal to it:

∆z = ∆z∥ + ∆z⊥, z⊤∆z⊥ = 0. (40) The parallel component is obtained via standard projection,

and therefore the orthogonal component is

∆z∥ =

z⊤∆z ∥z∥2

z, (41)

∆z⊥ = ∆z − ∆z∥ = ∆z −

z⊤∆z ∥z∥2

z. (42)

Note that

zz⊤ ∥z∥2

Z⊥∆z = I −

∆z = ∆z⊥,

so ∆z⊥ is precisely the projection of ∆z onto the subspace orthogonal to z, and

∆z⊤Z⊥∆z = ∆z⊤∆z⊥ = ∥∆z⊥∥2, (43) Substituting into the cosine expansion yields Theorem 1 (Local Deviation Induced by Pruning):

|1 − CosineSim(z, z + ∆z) ≈<br><br>∥∆z⊥∥2 2∥z∥2<br><br>|
|---|

. (44)

This demonstrates that, without the softmax transformation, the cosine deviation is governed by the orthogonal magnitude of the perturbation in the orthogonal subspace under a uniform weighting.

- D.2. Angular Deviation Induced by Softmax Transformation When p and q arise from softmax with logits z and z + ∆z and temperature T, the first–order perturbation is

1 T

A∆z, A ≜ diag(p) − pp⊤. (45) Substituting into Equation (37) and ignoring the negligible term gives

∆p ≈

1 2T2∥p∥2

∆z⊤AP⊥A∆z. (46)

1 − CosineSim(p,q) ≈

Let µ ≜ Ep[∆z] = i pi∆zi, ∥p∥2 = i p2i, the i-th component of A∆z is

pj∆zj = pi(∆zi − µ). (47)

(A∆z)i = pi∆zi − pi

j

Since A is symmetric,

∆z⊤A2∆z = (A∆z)⊤(A∆z) =

i

p2i(∆zi − µ)2. (48)

We next compute ∆z⊤Ap and first evaluate

(Ap)i = p2i − pi

j

p2j = p2i − ∥p∥2pi, (49)

thus

∆z⊤Ap =

i

p2i∆zi − ∥p∥2µ. (50)

Then we obtain the fully explicit second–order form:

|1 − CosineSim(p, q) ≈<br><br>1 2T2∥p∥2 i<br><br>p2i(∆zi − µ)2 −<br><br>1 ∥p∥2 i<br><br>p2i∆zi − ∥p∥2µ<br><br>2|
|---|

. (51)

To obtain a more compact statistical form, we introduce a new distribution r that reweights tokens proportionally to p2:

p2i ∥p∥2

ri ≜

ri = 1. (52)

,

i

Let µr ≜ Er[∆z] = i ri∆zi = ∥p1∥2 i p2i∆zi, we rewrite the two terms in Equation (51) under r. For simplicity, we temporarily ignore the denominator. Then, the first term in Equation (51) can be written as

i

p2i(∆zi − µ)2 = ∥p∥2

i

ri(∆zi − µ)2 = ∥p∥2 Er (∆z − µ)2 . (53)

For the second term, using the definition of µr we obtain

i

p2i∆zi − ∥p∥2µ = ∥p∥2µr − ∥p∥2µ = ∥p∥2(µr − µ), (54)

and hence

1 ∥p∥2 i

p2i∆zi − ∥p∥2µ

Substituting Equations (53) and (55) into Equation (51) yields

2

###### = ∥p∥2(µr − µ)2. (55)

1 2T2∥p∥2

1 − CosineSim(p,q) ≈

1 2T2

∥p∥2 Er[(∆z − µ)2] − ∥p∥2(µr − µ)2 =

Er[(∆z − µ)2] − (µr − µ)2 . (56)

Note that

(∆z − µr)2 = (∆z − µ + µ − µr)2 = (∆z − µ)2 + 2(µ − µr)(∆z − µ) + (µ − µr)2, (57)

taking expectation under r gives

###### Er[(∆z − µr)2] = Er[(∆z − µ)2] + 2(µ − µr)Er[∆z − µ] + (µ − µr)2. (58)

Given that Er[∆z − µ] = µr − µ, the cross term becomes

2(µ − µr)Er[∆z − µ] = −2(µr − µ)2. (59) Substituting back yields the standard variance identity

|Er[(∆z − µr)2] = Er (∆z − µ)2 − (µr − µ)2|
|---|

. (60)

To simplify this expression, recall the variance decomposition identity

###### Varr(∆z) = Er[(∆z − µ)2] − (µr − µ)2, (61)

which follows from expanding Er[(∆z − µr)2]. Substituting the identity into Equation (51) yields Theorem 2 (Sensitivity of Probability Space to Logit Perturbations):

|1 − CosineSim(p, p + ∆p) ≈<br><br>Varr(∆z) 2T2<br><br>, ri =<br><br>p2i ∥p∥2<br><br>|
|---|

. (62)

### E. Error Decomposition and Propagation during Autoregressive Decoding

We present a theoretical analysis of error propagation in context-dependent operators, using self-attention as a canonical example since it explicitly depends on tokens from previous timesteps and reveals how errors propagate across timesteps.

##### E.1. Error Decomposition in Context-Dependent Operators

We begin by analyzing the output deviation of context-dependent operators, considering a single causal self-attention layer at decoding step t+1 as the representative case. Let αt+1,i denote the attention weight over token i ≤ t, and vi the corresponding value representation. The attention output is given by

αt+1,ivi. (63)

ot+1 =

i≤t

After pruning, both the attention weights and value representations are perturbed. Denoting the perturbed output as o˜t+1, a first-order Taylor expansion yields

∆ot+1 =

αt+1,i ∆vi +

i≤t

∆αt+1,i vi +

i≤t

∆αt+1,i ∆vi,≈

i≤t

αt+1,i ∆vi

i≤t

value path

+O(∥∆∥2), (64)

+

∆αt+1,i vi

i≤t

weight path

where ∆vi and ∆αt+1,i denote the perturbations in value representations and attention weights, respectively.

Eq. (64) reveals two dominant first-order error paths: (i) a value path, where deviations in representations directly propagate through attention aggregation, and (ii) a weight path, where perturbations in queries or keys alter the attention reweighting mechanism. Higher-order interaction terms are grouped into O(∥∆∥2).

##### E.2. Pruning-Induced Errors in Per-Token Operators

We next contrast self-attention with operators that do not depend on past tokens, such as linear layers or feed-forward networks. Let G(·) denote such an operator, whose output at step t depends only on the current input xt. Under parameter perturbation ∆W, the perturbed output satisfies

o˜t = G(W + ∆W, xt), (65)

∆ot ≜ o˜t − ot = F(∆W, xt), (66) where F(·) denotes an implicit function that captures the dependence of the output deviation on its arguments. In particular, for operators without historical dependency, ∆ot depends only on the parameter perturbation ∆W and the current input xt. In other words, in the absence of historical dependency, pruning-induced deviations depend solely on parameter perturbations and the current input. No accumulated representation errors from previous steps are involved.

##### E.3. Error Sources in Autoregressive Decoding

In autoregressive decoding, self-attention explicitly couples the current computation with representations from previous timesteps. As a result, the deviation at step t+1 admits a more general functional form:

∆ot+1 = F(∆W, xt+1) + F(∆x0:t) + O(∥∆∥2), (67)

where ∆x0:t denotes accumulated perturbations in historical representations, and ∆W represents the effective parameter perturbation induced by pruning (e.g., removing or zeroing a subset of model parameters).

The first term corresponds to deviations induced directly by parameter perturbations at the current step, analogous to Eq. (66). In contrast, the second term arises uniquely from self-attention, which converts perturbations in past activations into explicit contributors to the current output. This structural difference implies that, during decoding, pruning-induced errors are no longer localized but instead depend on accumulated historical deviations.

Together, these observations highlight a fundamental distinction between pruning behavior in self-attention and in nonhistorical operators: while the latter admits a closed-form dependence on (∆W,xt), self-attention introduces an additional error source driven by historical representations.

##### E.4. Prompt vs. Generated Context in Autoregressive Decoding

A key distinction between autoregressive and non-generative settings lies in the composition of the attention context. At decoding step t, the historical representations can be decomposed as

x0:t = xprompt0:P ∪ xgenP+1:t , (68)

where xprompt0:P denotes prompt tokens provided during the prefill stage, and xgenP+1:t denotes tokens generated by the model in previous decoding steps.

While both generative and non-generative tasks attend over the prompt tokens, only autoregressive decoding incorporates model-generated tokens into the attention context. This difference leads to a qualitative change in the source of historical perturbations. Specifically, perturbations associated with prompt tokens are data-dependent and fixed once prefill is completed, whereas perturbations in generated tokens are induced by prior decoding deviations and therefore depend on the model’s own outputs.

Formally, the accumulated historical perturbation term in Eq. (67) can be decomposed as

∆x0:t = ∆xprompt0:P + ∆xgenP+1:t, (69)

where ∆xprompt0:P is fixed after prefill, while ∆xgenP+1:t evolves recursively with the decoding process. Substituting Eq. (69) into Eq. (67) yields

###### ∆ot+1 = F(∆W,xt+1) + F(∆xprompt0:P ) + F(∆xgenP+1:t) + O(∥∆∥2). (70)

Crucially, the third term arises only in autoregressive decoding. Since generated tokens are produced based on model logits, perturbations in earlier steps can influence the representations or selections of subsequent tokens, leading to deviations in the generated context used at later decoding steps. Once such a deviation occurs, self-attention aggregates over a different historical context, causing pruning-induced errors to propagate and compound across decoding steps. This feedback mechanism does not exist in non-generative or prefill-only settings, where the attention context remains fixed and independent of the model outputs.

Table 3. Representative candidate prompts used in our analysis, grouped by task format. Multiple-Choice Classification

[MC1] Tom has 15 candies. He eats 4 and gives 3 to his friend. How many candies does Tom have left? Choose the correct answer: A) 6 B) 7 C) 8 D) 9 [MC2] Which planet is known as the Red Planet? Choose the correct answer: A) Venus B) Mars C) Jupiter D) Saturn [MC3] Which of the following animals is a mammal? Choose the correct answer: A) Snake B) Frog C) Dog D) Lizard [MC4] Mark is older than John, and John is older than Alex. Who is the youngest? Choose the correct answer: A) Mark B) John C) Alex D) None of them

###### Mathematical Reasoning

- [Math1] John has twice as many books as Mary. Together they have 18 books. How many books does John have?
- [Math2] Natalia sold clips to 48 friends in April and half as many in May. How many clips did she sell altogether?
- [Math3] Emma has three times as many apples as Liam. Together they have 24 apples. How many apples does Emma have?
- [Math4] Alice buys 5 packs of pencils, each containing 12 pencils. She gives 8 pencils to her brother and 15 to her friend. How many pencils does Alice have left?

###### Open-Ended Generation

- [Gen1] Tell a short, coherent story about a child who finds a mysterious key and discovers what it opens.
- [Gen2] Explain step by step how to solve a math word problem where a student calculates how many apples remain after giving some away.
- [Gen3] Write a short explanation suitable for a 10-year-old describing why the Earth has day and night.
- [Gen4] Describe, step by step, how to make a peanut butter and jelly sandwich.

### F. Representative Prompts

We summarize the representative prompt categories used throughout our analysis in Table 3, including multiple-choice classification, mathematical reasoning, and open-ended generation. These prompts are designed to cover a diverse range of task formats and difficulty levels commonly encountered in both evaluation benchmarks and real-world usage.

Notably, some of the prompts are relatively simple and require only basic reasoning or factual knowledge. However, despite their simplicity, we observe that compressed models may still exhibit severe performance degradation, particularly in generative settings. This highlights that the observed failures are not merely due to task difficulty, but rather stem from the intrinsic sensitivity of the generation process to model compression. These representative prompts therefore serve as controlled yet informative probes for analyzing the robustness and failure modes of compressed language models.

### G. Additional Empirical Results on Pruning

Baseline

Unstructured

4:8

2:4

| |
|---|

| |
|---|

| |
|---|

75

Performance(%)

75

50

50

25

25

0

0

Qwen2.5-7B Mistral-7B

Qwen2.5-7B Mistral-7B

(a) HellaSwag

(b) GSM8K

Figure 9. Performance comparison under different intra-layer sparsification strategies with SparseGPT.

To further examine the impact of different intra-layer sparsification strategies, we report additional results on HellaSwag and GSM8K in Figure 9, using SparseGPT (Frantar & Alistarh, 2023) as the pruning algorithm. On HellaSwag, all sparsification methods incur only mild performance degradation, indicating that short-context and classification-style benchmarks are relatively robust to parameter removal. In contrast, GSM8K exhibits a markedly different behavior: performance degrades

- sharply as sparsity becomes more structured or aggressive. This phenomenon consistently appears across other language models, e.g., LLaMA-3 (Grattafiori et al., 2024) and Qwen-3 (Yang et al., 2025) (see Figure 10), reinforcing our claim that generation-oriented tasks impose stricter robustness requirements under network pruning.

HumanEval

MMLU

HumanEval

MMLU

60

75

75

Attn Drop MLP Drop ShortGPT

Attn Drop MLP Drop ShortGPT

60

Performance(%)

Performance(%)

40

50

50

40

20

25

25

Attn Drop MLP Drop ShortGPT

Attn Drop MLP Drop ShortGPT

20

0

0

0

0

0 2 4 6 8 10 12 Dropped Layers

0 2 4 6 8 10 12 Dropped Layers

0 2 4 6 8 10 12 Dropped Layers

0 2 4 6 8 10 12 Dropped Layers

(b) Qwen-3-4B. Figure 10. Impact of inter-layer pruning on generative (HumanEval) and non-generative (MMLU) tasks.

(a) Llama-3-8B.

### H. Ablation Study on Temperature Factors

We conduct an ablation study on the temperature factor to examine the robustness of our analysis under different softmax scaling settings. Specifically, we vary the temperature while keeping all other configurations unchanged and compare the ground-truth measurements and theoretical estimates in terms of cosine similarity and KL divergence. As shown in

Figure 11, the estimated trends consistently align with the ground-truth measurements across different temperatures. These results indicate that our theoretical analysis is not sensitive to a particular temperature choice and generalizes well across commonly used temperature settings.

###### T = 0.3

###### T = 0.5

###### T = 0.7

###### T = 1.0

T = 1.3

0.6

0.9

0.9

0.9

0.9

1-CosineSim

0.4

0.6

0.6

0.6

0.6

0.2

0.3

0.3

0.3

0.3

0.0

0.0

0.0

0.0

0.0

0 60 120 180

0 60 120 180

0 50 100 150

0 60 120 180

0 25 50 75

Generation Step Groundtruth Estimated

(a) Angular deviation.

###### T = 0.3

###### T = 0.5

###### T = 0.7

###### T = 1.0

T = 1.3

0.8

3.0

KLdivergence

1.6

1.6

4.0

0.5

2.0

0.8

2.0

0.8

0.2

1.0

0.0

0.0

0.0

0.0

0.0

0 60 120 180

0 60 120 180

0 50 100 150

0 60 120 180

0 25 50 75

Generation Step Groundtruth Estimated

(b) KL divergence.

- Figure 11. Effect of temperature on pruning-induced deviation estimation. We compare the ground-truth measurements and theoretical estimates under different temperature settings in terms of (a) angular deviation and (b) KL divergence.

I. Complementary Discussion of Quantization

While this work mainly focuses on network pruning, our empirical and theoretical analysis also applies to quantization. As shown in panels (a)–(c) of Figure 12, we compare the resulting deviations induced by quantization and pruning. Quantization exhibits consistently higher similarity, i.e., lower deviations, because it approximates parameters with low-precision values, whereas pruning removes parameters entirely. As a result, the magnitude and variance of ∆z are much lower, and the KL divergence of p remains nearly stable in the early decoding steps. Although the KL divergence for quantization increases sharply at a certain point, this mainly occurs because the question has already been fully answered and redundant tokens are generated in the subsequent sequence.

0 60 120 180

0.0

0.4

0.8

hCosineSim()

(a) Embedding Deviation

0 60 120 180

-0.5

0.0

0.5

1.0

zCosineSim()

(b) Logit Deviation

0 60 120 180

0.0

0.3

0.6

0.9

pCosineSim()

(c) Probability Deviation

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 60 120 180

Generation Step

0

3

6

9

zRMS()

(d) Magnitude of Logit Perturbation

0 60 120 180

Generation Step

0.0

2.5

5.0

7.5

zVar()

(e) Variance of Logit Perturbation

0 60 120 180

Generation Step

0

15

30

KLdivergence

(f) Output Distribution Shift

Attn Drop Quantization 4:8

- Figure 12. Step-wise perturbation analysis under different compression methods. We compare inter-layer dropping via Attn Drop with 8 layers removed, intra-layer sparsification via Wanda with 4:8 (50%) sparsity, and weight-only quantization using AWQ. Panels (a–c) report cosine similarity between the original and perturbed representations in the embedding, logit, and probability spaces, respectively. Panels (d–f) further characterize the magnitude of logit perturbations and their distributional effects, including the resulting KL divergence in the output probability distribution.

### J. Supplementary Visualizations

Beyond representation similarity induced by removing entire layers, we also examine the effect of intra-layer pruning. For the i-th layer, we prune that layer to a target sparsity level and measure the representation similarity between the outputs of the baseline model and the pruned model. Figure 13 shows that Wanda pruning follows a trend similar to that observed with layer dropping. While the magnitude differs across pruning strategies, e.g., MLP layers exhibit greater representation similarity after intra-layer pruning, the same representation-hierarchy interpretation still applies.

1.0

0.7

CosineSim

0.4

Embedding

0.1

Logit

0.2

Probability

0.5

0 4 8 12 16 20 24

Layer Index

(a) Attention.

1.0

0.7

CosineSim

0.4

Embedding

Logit

0.1

Probability

0.2

0 5 10 15 20 25

Layer Index

(b) MLP.

- Figure 13. Layer-wise representation similarity between the outputs of the baseline model and its temporarily pruned counterpart. For the pruned model, we temporarily prune a single layer during the forward pass and compute the representation similarity at the same layer, while keeping all other layers identical to the baseline model.

We also present layer-wise comparisons between ground-truth measurements and theoretical estimates across different representation spaces to further validate the proposed approximation and trace how pruning-induced perturbations evolve during generation.

Figures 14 and 15 report the layer-wise evolution of distributional deviations in the probability space, measured by KL divergence and angular deviation, respectively. For each attention layer, we compare the ground-truth measurements with our theoretical estimates across decoding steps. The results show that the proposed estimator closely tracks the true deviation trends across layers and time steps. Notably, deeper layers consistently exhibit larger deviations, indicating stronger distributional shifts in later stages of the network.

Figures 16 and 17 further examine representation deviations in the embedding and logit spaces. In contrast to the probability space, both spaces show substantially smaller angular deviation, and the theoretical curves remain closely aligned with the ground-truth measurements. This observation supports our analysis that pruning-induced perturbations remain localized in these spaces and are less amplified before the softmax transformation. The only exceptions are the first and last layers, where the transformations are substantially larger and therefore violate the assumption of locality.

Finally, Figure 18 compares the relative magnitude ratios of representations in the embedding and logit spaces. The results indicate that the relative orthogonal energy is substantially reduced after the LM head projection, which is consistent with pruning-induced perturbations remaining limited in the logit space. Figure 19 shows the variance of ∆z under uniform and weighted sampling, while Figures 20 and 21 compare this variance against the corresponding relative magnitude ratios. The consistently large variance of ∆z explains the substantial deviation observed in the probability space.

Together, these visualizations corroborate the theoretical findings in the main text, illustrating how pruning-induced perturbations are progressively amplified across representation spaces, and highlighting the distinct roles played by linear and nonlinear transformations in this process.

×10 2 Layer 0

×10 3 Layer 1

×10 2 Layer 2

×10 2 Layer 3

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

1.6

1.6

3.0

3.0

0.8

0.8

1.5

1.5

0.0

×10 2 Layer 4

×10 2 Layer 5

×10 1 Layer 6

×10 2 Layer 7

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

4.0

6.0

1.2

3.0

2.0

3.0

1.5

0.6

0.0

0.0

0.0

×10 1 Layer 8

×10 1 Layer 9

×10 1 Layer 10

×10 1 Layer 11

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

8.0

4.0

2.0

1.0

4.0

2.0

1.0

0.5

0.0

0.0

0.0

0.0

KLdivergence

×10 1 Layer 12

×10 1 Layer 13

×10 1 Layer 14

×10 1 Layer 15

4.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

6.0

3.0

1.0

2.0

3.0

1.5

0.5

0.0

0.0

0.0

0.0

×10 1 Layer 16

×10 1 Layer 17

×10 1 Layer 18

Layer 19

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

4.0

8.0

2.0

0.8

2.0

4.0

1.0

0.4

0.0

0.0

0.0

0.0

×10 1 Layer 20

×10 1 Layer 21

×10 1 Layer 22

×10 1 Layer 23

4.0

5.0

3.0

3.0

2.0

2.5

1.5

1.5

0.0

0.0

0.0

0.0

×10 1 Layer 24

×10 1 Layer 25

×10 1 Layer 26

×101 Layer 27

8.0

3.0

5.0

1.2

4.0

1.5

2.5

0.6

0.0

0.0

0.0

0.0

0 80 160

0 80 160

0 80 160

0 80 160

Generation Step

Groundtruth Estimated

- Figure 14. Layer-wise evolution of KL divergence in attention layers across generation steps. For each transformer layer, we compare the ground-truth KL divergence (blue) and the theoretical estimates (orange). The results show that the proposed estimator closely tracks the true KL behavior across layers, while also revealing that deeper layers generally exhibit larger distributional shifts.

2.0

4.0

×10 2 Layer 0

2.0

4.0

×10 3 Layer 1

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

2.0

×10 2 Layer 2

1.0

2.0

×10 2 Layer 3

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

5.0

×10 2 Layer 4

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

5.0

×10 2 Layer 5

0.5

1.0

1.5

×10 1 Layer 6

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

5.0

×10 2 Layer 7

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

1.0

×10 1 Layer 8

0.0

1.0

Layer 9

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

2.0

×10 1 Layer 10

0.0

5.0

×10 1 Layer 11

0.0

2.5

×10 1 Layer 12

0.0

2.5

×10 1 Layer 13

0.0

5.0

×10 1 Layer 14

0.0

5.0

×10 2 Layer 15

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

1.0

×10 1 Layer 16

0.0

2.0

×10 2 Layer 17

0.0

2.0

×10 1 Layer 18

0.0

2.0

Layer 19

0.0

1.0

×10 1 Layer 20

0.0

2.0

×10 1 Layer 21

0.0

5.0

×10 1 Layer 22

0.0

2.0

×10 1 Layer 23

0 50 100 150

0.0

2.0

×10 1 Layer 24

0 50 100 150

0.0

5.0

×10 1 Layer 25

0 50 100 150

0.0

5.0

×10 1 Layer 26

0 50 100 150

0.0

2.0

Layer 27

Generation Step

1-CosineSim

Groundtruth Estimated

- Figure 15. Layer-wise evolution of angular deviation values of the probability space between the inputs and outputs of attention layers. We report the ground-truth measurements (blue) and the theoretical estimates (orange) across decoding steps. Consistent with the KL analysis, shallow layers remain relatively stable while deeper layers exhibit larger semantic deviations, and our estimator successfully captures these trends.

Layer 0

×10 1 Layer 1

×10 1 Layer 2

×10 1 Layer 3

1.5

1.5

2.0

1.0

1.0

1.0

0.5

0.5

1.0

0.5

×10 1 Layer 4

×10 1 Layer 5

×10 1 Layer 6

×10 2 Layer 7

1.0

1.0

1.0

5.0

0.5

0.5

2.5

0.5

×10 1 Layer 8

×10 1 Layer 9

×10 2 Layer 10

×10 2 Layer 11

1.0

7.5

2.0

5.0

5.0

0.5

1.0

2.5

2.5

1-CosineSim

×10 2 Layer 12

×10 2 Layer 13

×10 1 Layer 14

×10 2 Layer 15

7.5

4.0

1.0

5.0

5.0

0.5

2.0

2.5

2.5

×10 2 Layer 16

×10 2 Layer 17

×10 2 Layer 18

×10 1 Layer 19

5.0

3.0

4.0

1.5

1.0

2.0

2.5

2.0

0.5

1.0

×10 2 Layer 20

×10 2 Layer 21

×10 2 Layer 22

×10 2 Layer 23

7.5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

4.0

5.0

5.0

5.0

2.0

2.5

2.5

0.0

×10 2 Layer 24

×10 2 Layer 25

×10 2 Layer 26

×10 1 Layer 27

3.0

7.5

5.0

2.0

5.0

2.0

2.5

1.0

2.5

1.0

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

Generation Step

Groundtruth Estimated

- Figure 16. Layer-wise evolution of angular deviation values on the embedding space between the inputs and outputs of attention layers. We report the ground-truth measurements (blue) and the theoretical estimates (orange) across decoding steps.

1.0

2.0

Layer 0

2.5

5.0

7.5

×10 2 Layer 1

0.0

1.0

×10 1 Layer 2

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

2.0

×10 2 Layer 3

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

5.0

×10 2 Layer 4

0.0

5.0

×10 2 Layer 5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

2.5

×10 2 Layer 6

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

1.0

×10 2 Layer 7

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

2.0

×10 2 Layer 8

0.0

2.0

×10 1 Layer 9

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

1.0

×10 2 Layer 10

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

2.0

×10 2 Layer 11

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

2.0

×10 2 Layer 12

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

5.0

×10 3 Layer 13

0.0

2.5

×10 2 Layer 14

1.0

2.0

3.0

×10 3 Layer 15

2.5

5.0

×10 3 Layer 16

0.5

1.0

×10 3 Layer 17

2.0

4.0

×10 3 Layer 18

0.0

1.0

×10 1 Layer 19

1.0

2.0

×10 3 Layer 20

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

2.0

×10 3 Layer 21

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.0

1.0

×10 3 Layer 22

0.5

1.0

1.5

×10 3 Layer 23

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150

0.0

1.0

×10 3 Layer 24

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150

0.0

1.0

×10 3 Layer 25

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 50 100 150

0.0

1.0

×10 3 Layer 26

0 50 100 150

0.0

5.0

×10 2 Layer 27

Generation Step

1-CosineSim

Groundtruth Estimated

- Figure 17. Layer-wise evolution of angular deviation values on the logit space between the inputs and outputs of attention layers. We report the ground-truth measurements (blue) and the theoretical estimates (orange) across decoding steps.

×102 Layer 0

×10 1 Layer 1

×10 1 Layer 2

×10 1 Layer 3

1.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

1.0

1.0

5.0

0.0

0.0

0.0

0.0

×10 1 Layer 4

×10 1 Layer 5

×10 1 Layer 6

×10 2 Layer 7

1.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

1.0

1.0

5.0

0.0

0.0

0.0

0.0

×10 1 Layer 8

×10 1 Layer 9

×10 2 Layer 10

×10 2 Layer 11

1.0

5.0

5.0

2.0

RelativeMagnitude

0.0

0.0

0.0

0.0

×10 2 Layer 12

×10 2 Layer 13

×10 1 Layer 14

×10 2 Layer 15

1.0

5.0

5.0

2.5

0.0

0.0

0.0

0.0

×10 2 Layer 16

×10 2 Layer 17

×10 2 Layer 18

×10 1 Layer 19

5.0

2.0

2.5

1.0

0.0

0.0

0.0

0.0

×10 2 Layer 20

×10 2 Layer 21

×10 2 Layer 22

×10 2 Layer 23

5.0

5.0

5.0

2.5

0.0

0.0

0.0

0.0

×10 2 Layer 24

×10 2 Layer 25

×10 2 Layer 26

×10 1 Layer 27

5.0

2.0

2.0

5.0

0.0

0.0

0.0

0.0

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

Generation Step

z 2 2 z 2

h 2 2 h 2

Figure 18. Relative magnitude ratios of representations in the logit space (blue) and the embedding space (orange).

×10 2 Layer 0

×10 3 Layer 1

×10 2 Layer 2

×10 2 Layer 3

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

6.0

2.0

2.0

6.0

3.0

1.0

1.0

3.0

0.0

×10 2 Layer 4

×10 2 Layer 5

×10 1 Layer 6

×10 2 Layer 7

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

8.0

5.0

5.0

1.6

4.0

2.5

2.5

0.8

0.0

0.0

0.0

×10 1 Layer 8

Layer 9

×10 1 Layer 10

×10 1 Layer 11

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

1.0

2.0

1.2

4.0

0.5

1.0

0.6

2.0

0.0

0.0

0.0

0.0

×10 1 Layer 12

×10 1 Layer 13

×10 1 Layer 14

×10 1 Layer 15

8.0

Variance

4.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

1.6

4.0

4.0

2.0

0.8

2.0

0.0

0.0

0.0

0.0

×10 1 Layer 16

×10 2 Layer 17

×10 1 Layer 18

Layer 19

3.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

4.0

8.0

8.0

1.5

2.0

4.0

4.0

0.0

0.0

0.0

0.0

×10 1 Layer 20

Layer 21

×10 1 Layer 22

Layer 23

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

2.0

8.0

1.6

4.0

1.0

4.0

0.8

2.0

0.0

0.0

0.0

0.0

Layer 24

Layer 25

×101 Layer 26

×102 Layer 27

4.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

5.0

4.0

2.0

2.0

2.5

2.0

1.0

0.0

0.0

0.0

0.0

0 80 160

0 80 160

0 80 160

0 80 160

Generation Step

Uniform Variance Weighted Variance

Figure 19. Uniform and weighted variances of ∆z, where the uniform variance is computed under a uniform distribution (blue) and the weighted variance (orange) is computed under the vocabulary distribution ri = p

2 i

∥p∥2 .

×10 4 Layer 0

×10 2 Layer 1

×10 1 Layer 2

×10 1 Layer 3

3.0

5.0

3.0

7.5

2.0

2.0

5.0

2.5

1.0

1.0

2.5

×10 1 Layer 4

Layer 5

Layer 6

Layer 7

1.5

1.5

7.5

3.0

1.0

1.0

5.0

2.0

0.5

0.5

2.5

1.0

Layer 8

×101 Layer 9

Layer 10

Layer 11

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

5.0

2.0

5.0

1.0

1.0

0.0

0.0

0.0

Layer 12

Layer 13

Layer 14

Layer 15

#### Variance

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

5.0

2.0

5.0

5.0

0.0

0.0

0.0

0.0

Layer 16

Layer 17

Layer 18

×101 Layer 19

5.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

1.0

2.5

1.0

0.0

0.0

0.0

0.0

Layer 20

Layer 21

×101 Layer 22

Layer 23

1.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

5.0

5.0

5.0

0.0

0.0

0.0

0.0

×101 Layer 24

×101 Layer 25

×101 Layer 26

×101 Layer 27

2.0

2.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

2.0

2.5

0.0

0.0

0.0

0.0

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

Generation Step

Figure 20. Relative values of the weighted variance of ∆z, divided by the relative magnitude ratio of h.

×10 4 Layer 0

×10 1 Layer 1

Layer 2

Layer 3

1.5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

3.0

2.0

2.5

1.0

2.0

1.0

0.5

0.0

0.0

Layer 4

Layer 5

×101 Layer 6

×101 Layer 7

10.0

3.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

5.0

2.0

5.0

1.0

0.0

0.0

0.0

×101 Layer 8

×101 Layer 9

×101 Layer 10

×101 Layer 11

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

5.0

5.0

2.5

5.0

0.0

0.0

0.0

0.0

×101 Layer 12

×101 Layer 13

×101 Layer 14

×101 Layer 15

10.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

Variance

5.0

5.0

5.0

0.0

0.0

0.0

0.0

×101 Layer 16

×101 Layer 17

×101 Layer 18

×101 Layer 19

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

5.0

5.0

2.5

5.0

0.0

0.0

0.0

0.0

×102 Layer 20

×102 Layer 21

×102 Layer 22

×102 Layer 23

5.0

2.0

2.5

2.0

0.0

0.0

0.0

0.0

×103 Layer 24

×103 Layer 25

×103 Layer 26

×102 Layer 27

2.0

1.0

5.0

1.0

0.0

0.0

0.0

0.0

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

Generation Step

Figure 21. Relative values of the weighted variance of ∆z, divided by the relative magnitude ratio of z.

