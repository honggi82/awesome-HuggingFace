## Skrr: Skip and Re-use Text Encoder Layers for Memory Efficient Text-to-Image Generation

Hoigi Seo*1 Wongi Jeong*1 Jae-sun Seo2 Se Young Chun13

# arXiv:2502.08690v1[cs.LG]12Feb2025

#### Abstract

Large-scale text encoders in text-to-image (T2I) diffusion models have demonstrated exceptional performance in generating high-quality images from textual prompts. Unlike denoising modules that rely on multiple iterative steps, text encoders require only a single forward pass to produce text embeddings. However, despite their minimal contribution to total inference time and floating-point operations (FLOPs), text encoders demand significantly higher memory usage, up to eight times more than denoising modules. To address this inefficiency, we propose Skip and Re-use layers (Skrr), a simple yet effective pruning strategy specifically designed for text encoders in T2I diffusion models. Skrr exploits the inherent redundancy in transformer blocks by selectively skipping or reusing certain layers in a manner tailored for T2I tasks, thereby reducing memory consumption without compromising performance. Extensive experiments demonstrate that Skrr maintains image quality comparable to the original model even under high sparsity levels, outperforming existing blockwise pruning methods. Furthermore, Skrr achieves state-of-the-art memory efficiency while preserving performance across multiple evaluation metrics, including the FID, CLIP, DreamSim, and GenEval scores.

#### 1. Introduction

Diffusion generative models excel at tasks such as text-toimage (T2I) synthesis (Rombach et al., 2022; Podell et al., 2023; Esser et al., 2024; Chen et al., 2024), editing (Brooks et al., 2023; Cao et al., 2023; Kawar et al., 2023), video generation (Liu et al., 2024; Polyak et al., 2024), and 3D creation (Poole et al., 2022; Cao et al., 2023; Seo et al.,

*Equal contribution 1Dept. of Electrical and Computer Engineering, Seoul National University, Republic of Korea 2School of Electrical and Computer Engineering, Cornell Tech, USA 3INMC & IPAI, Seoul National University, Republic of Korea. Correspondence to: Se Young Chun <sychun@snu.ac.kr>.

|0.6|
|---|

|62.2<br><br>|Text Text Text|
|---|---|
|9.1| |
|26.5| |

2.8

Variational Autoencoder Encocder 3 (T5-XXL) Encocder 2 (CLIP-G) Encocder 1 (CLIP-L)

0.4

Percentage(%)

Percentage(%)

0.05

Variational Autoencoder

96.7

Text Encocder 3 (T5-XXL) Text Encocder 2 (CLIP-G) Text Encocder 1 (CLIP-L) Denoising Transformer

Denoising Transformer

|1.6|
|---|

(a) FLOPs ratio.

(b) Parameter ratio.

Figure 1. (a) FLOPs distribution during image generation in Stable Diffusion 3 (SD3) (Esser et al., 2024). (b) Parameter distribution across modules in SD3. The text encoders contributes less than 0.5% to the overall FLOPs but account for over 70% of the total model parameters. For VAE, only the decoder was considered.

2023; Wang et al., 2024c). With modern architecture and large-scale text encoders, they produce high-quality images that closely match text prompts. Despite these successes, they require significant computational resources, especially memory, making deployment and scalability challenging.

To address these issues, research suggests enhancing the efficiency of the T2I diffusion model through strategies such as knowledge distillation (KD) (Castells et al., 2024b; Li et al., 2024b; Song et al., 2024b; Kim et al., 2024; Zhao et al., 2024), which transfers knowledge from larger models to smaller ones; pruning (Castells et al., 2024a; Ganjdanesh et al., 2024; Lee et al., 2024; Wang et al., 2024b), which eliminates superfluous weights; and quantization (Li et al., 2023; He et al., 2024; Ryu et al., 2025), which reduces precision by utilizing fewer bits. While effective, these methods target the denoising module. As shown in Fig. 1, the text encoders account for over 70% of the total parameters, but only 0.5% of floating-point operations (FLOPs), causing disproportionate memory usage. Despite this imbalance, efforts to reduce the text encoder size have been limited.

Large language models (LLMs) also face similar challenges, where model sizes result in significant computational and memory overhead, hindering their practical use. To address this problem, studies such as KD (Hsieh et al., 2023; Huang et al., 2023; Ko et al., 2024), quantization (Xiao et al., 2023; Ashkboos et al., 2024b; Lin et al., 2024), and pruning (Sun et al., 2023; Ashkboos et al., 2024a; Men et al., 2024; Song

et al., 2024a; Yang et al., 2024; Zhang et al., 2024) have been proposed. While KD reduces the size of the model, it requires costly training. Quantization reduces memory usage by reducing precision, but it requires specific hardware support. In contrast, pruning offers parameter reduction with minimal performance loss, making it an efficient solution.

Among pruning techniques, structured pruning has been actively studied to remove rows and columns (van der Ouderaa et al., 2023; Ashkboos et al., 2024a) of the model weights, or entire layers or blocks (Gromov et al., 2024; Men et al., 2024; Yang et al., 2024; Zhang et al., 2024) of the model to reduce the size of the model and improve the inference speed. However, these methods are designed for autoregressive LLMs and face challenges when applied to T2I diffusion models, limiting their effectiveness in this context.

We propose Skip and reuse the layers (Skrr), a blockwise pruning technique for T2I diffusion models. Skrr effectively reduces text encoder size, alleviating memory overhead while preserving image quality and text alignment. Skrr involves two primary stages: the Skip to detect layers for pruning, followed by the Re-use to recycle the remaining layers to mitigate performance degradation. To the best of our knowledge, this is the first work to tackle the challenge of constructing a lightweight text encoder for T2I tasks.

In the Skip phase, sub-blocks of the text encoder transformer are pruned using a T2I diffusion-tailored discrepancy metric to align dense and pruned models. Prior methods greedily remove blocks to reduce computational costs, but often overlook block interactions, leading to suboptimal pruning. To mitigate this, we propose a beam search (Freitag & Al-Onaizan, 2017)-based approach that explores multiple pruning paths simultaneously, balancing the performance of exhaustive and efficiency of greedy strategies. In the Re-use phase, we assess the discrepancy from reusing adjacent unskipped blocks to identify those that can be effectively reutilized. Additionally, we provide theoretical support showing that Re-use can enhance performance beyond mere skipping. To improve discrepancy measurement in both phases, we employ a projection module identical to the one used for conditioning text embeddings in the denoising module.

We conducted comprehensive experiments across various metrics, sparsity levels, and diffusion models to thoroughly evaluate the T2I performance of compressed text encoders. The results indicate that Skrr surpasses current autoregressive LLM-targeted pruning techniques on image fidelity and text-image alignment in high sparsity (> 40%).

Our contributions can be summarized as follows.

- • We propose Skrr, an effective layer pruning method for the text encoder in T2I diffusion models.
- • We present Skip, a pruning approach for lightweight T2I diffusion models, and Re-use, a method to restore

T2I performance by leveraging the remaining layers, supported by theoretical analysis.

• Skrr achieves state-of-the-art blockwise pruning for T2I synthesis, improving GenEval scores by up to 20.4% at high sparsity over 40%.

#### 2. Related Works

##### 2.1. Efficient diffusion model

As diffusion generative models scale, T2I synthesis has achieved impressive results in generating high-fidelity, textaligned images. However, this advancement comes with significant computational and memory overhead. Previous research has primarily focused on optimizing the efficiency of the denoising module through methods such as knowledge distillation (Li et al., 2024b; Song et al., 2024b; Castells et al., 2024b; Zhao et al., 2024; Kim et al., 2024), pruning model weights (Fang et al., 2023; Ganjdanesh et al., 2024; Wang et al., 2024b; Castells et al., 2024a; Lee et al., 2024), and quantization of weights to lower precision bits (Li et al., 2023; He et al., 2024; Li et al., 2024a; Wang et al., 2024a; Ryu et al., 2025). While these approaches effectively reduce the computational and memory costs of the pipeline, they overlook the substantial memory burden imposed by the text encoder, which remains largely underexplored. To address this gap, we propose a targeted pruning strategy for the text encoder of a T2I diffusion model, which allows more memory efficient T2I synthesis with comparable performance.

##### 2.2. Blockwise pruning for LLMs

Although LLMs show promising performance in various tasks, their size often limits practical deployment. To address this problem, model compression techniques have been developed, with pruning emerging as a promising solution due to its ability to reduce the parameter count with minimal retraining. Notably, blockwise pruning of transformers (Men et al., 2024; Yang et al., 2024; Zhang et al., 2024) effectively reduces parameters while preserving performance.

ShortGPT (Men et al., 2024) proposed a method to prune blocks to the desired sparsity by removing them one by one and assigning a Block Influence (BI) score based on changes in cosine similarity and pruning those with lower BI first. LaCo (Yang et al., 2024) introduced a strategy to reduce the size of LLMs by merging the weights of adjacent transformer blocks, effectively compressing the model. FinerCut (Zhang et al., 2024) refined this approach by pruning sub-blocks, composed of Multi-Head Attention (MHA) and Feed-Forward Network (FFN) layers with normalization, in a fine-grained manner. It sequentially pruned sub-blocks while partially considering interactions between blocks, reducing the performance gap with the dense model.

Despite these advances, text encoder compression in diffu-

Re-use algorithm (Algorithm 2)

###### Skip algorithm (Algorithm 1)

Original Text Encoder

Original Text Encoder

MHA

MHA

MHA

MHA

FFN

FFN

FFN

FFN

MHA

MHA

MHA

MHA

FFN

FFN

FFN

FFN

Text embed

0 1

2 3

4 5

6 7

0 1

2 3

4 5

6 7

Calib. data Proj.

Skipped Text Encoder

𝑫𝒊𝒔𝒄.

𝑫𝒊𝒔𝒄.

MHA

MHA

MHA

MHA

MHA

MHA

MHA

FFN

FFN

FFN

FFN

FFN

0

1

2

4

7

0

1

2

4

6 7

Search for every skipped block

Search for every block

Re-used Text Encoder

Skipped Text Encoder

MHA

MHA

MHA

MHA

MHA

MHA

MHA

FFN

FFN

FFN

FFN

FFN

4

0

1

2

1

4

7

0

1

2

4

7

(a) Illustration of Skip phase.

(b) Illustration of Re-use phase.

- Figure 2. The visualization of overall framework of Skrr. (a) shows the Skip phase, which repeatedly assesses each sub-block by determining the output discrepancy (Disc.) between the dense and skipped models using a calibration dataset (Calib. data). To account for block interactions, it keeps the top k options with the smallest discrepancies and uses beam search for refined selection. (b) presents the Re-use phase, evaluating if recycling remaining block instead of skipped sub-blocks results in a smaller output discrepancy. If so, hidden states are fed back into the chosen layers. This two-phase approach efficiently reduces model size with minimal T2I performance loss.

sion models remains underexplored. To address this discrepancy, we propose Skrr, a blockwise text encoder pruning approach tailored for T2I diffusion models. We evaluated Skrr against existing LLM-based pruning techniques, demonstrating its effectiveness in reducing the size of the model while maintaining the T2I performance of its dense model.

LLMs. We extend this analysis to text encoders in diffusion models, especially T5-XXL (Raffel et al., 2020), widely used in T2I models, as shown in Fig. 3a. We observed a high degree of similarity between the hidden states of adjacent blocks. This strong similarity underscores the redundancy in the model and confirms the potential to prune blocks without significantly compromising performance.

#### 3. Method

Discrepancy metric. For effective block removal, it is crucial to evaluate impact of pruning each block on output. A strong metric must be defined to measure text embedding changes of post-removal. Given that text embeddings affect image quality in text-to-image models, choosing the right metric is vital. Current T2I diffusion models predominantly employ transformer frameworks to synthesize images from input noise and text embeddings. However, text embeddings from the output of the text encoder are not used as is. They undergo alignment through a single linear layer or multilayer perceptron (MLP), which is represented as:

Skrr is built around two main parts: Skip identifies layers to prune, and Re-use selects layers to reuse from the remained ones. During the Skip phase, each multi-head attention (MHA) and feed-forward network (FFN) sub-block is individually evaluated for its importance using a T2I diffusion-tailored metric. The sub-blocks are then ranked based on their significance. To optimize the pruning process, blocks with low importance are removed sequentially while exploring multiple possible combinations using a beam search-based algorithm. The Re-use phase evaluates each layer to reuse a layer based on the metric leveraged in Skip phase to the original output, ensuring that important information is conserved, thus reducing performance loss. The overall framework of Skrr is depicted in Fig. 2.

f = proj(E(c;θtext.);θdenoise.), (1)

where c is the input prompt, E(·;θtext.) is the text encoder parameterized with θtext., proj(·;θdenoise.) is the projection layer in the denoising module for condition vector from the text encoder. Using the features extracted in this manner, the importance of a block can be evaluated by comparing the similarity between the feature fdense of the dense model and the feature fskip of the model that skips (prunes) a block.

##### 3.1. Skip Algorithm

Feasibility of skipping blocks. Prior work (Men et al., 2024; Yang et al., 2024; Zhang et al., 2024) shows transformer blocks can be pruned based on output similarity in

[Figure 1]

[Figure 2]

TransformerBlockIndex

TransformerBlockIndex

Transformer Block Index

Transformer Block Index

(a) Hidden states similarity.

(b) Fixed input similarity.

- Figure 3. (a) The cosine similarity of hidden states in T5 transformer blocks demonstrates progressive variations. The result indicates specific layers could be omitted without serious performance degradation. (b) The cosine similarity of block outputs using fixed inputs. The similarity map reveals redundant role across blocks, suggesting that certain blocks could be replaced by adjacent blocks.

[Figure 3]

[Figure 4]

[Figure 5]

(a) Original image. (b) 7th, 22th removed. (c) 3rd, 5th removed.

Figure 4. (a) An image is created by the PixArt-Σ dense text encoder using the prompt “A car made out of vegetables.” with ||f∅||2 = 0.03. For image (b), the 7th and 22th sub-blocks are excluded, resulting in Metric1 = 0.85, Metric2 = 0.002, and ||f∅||2 = 0.19. Image (c) is generated by removing the 3rd and 5th sub-blocks, producing Metric1 = 0.89, Metric2 = 0.04, and ||f∅||2 = 3.34. Despite Metric1 being higher in (c), the large ||f∅||2 value compared to (b) leads to an abnormal image. Notably, Metric2 more accurately indicates differences in image quality.

A commonly used metric in prior studies (Men et al., 2024; Yang et al., 2024; Zhang et al., 2024) for measuring discrepancy is cosine similarity, which is formulated as:

fdense · fskip ||fdense||2||fskip||2

Metric1(fdense,fskip) = 1 −

(2)

Another metric worth considering is the mean-squared error (MSE) between two vectors, which differs from angular metrics by accounting for both the direction (angle) and the magnitude of the vectors. The MSE is formulated as:

1 d

Metric2(fdense,fskip) =

d

(fdensei − fskipi )2, (3)

i=1

where d is the dimension of the output feature vectors fdense and fskip, and fi represents the i-th component of vector f. Metric1 considers only the angle between vectors, while Metric2 integrates both the angle and magnitude, offering a more comprehensive evaluation of output discrepancies between dense and skipped models (Zhang et al., 2024). Therefore, we chose Metric2 to assess the discrepancies.

Null condition discrepancy. In diffusion generative models, guidance is essential to produce high-quality images. A prevalent approach is classifier-free guidance (CFG) (Ho & Salimans, 2021), which improves conditional synthesis by utilizing unconditional scores. This technique derives an unconditional score from a null condition, extrapolates it with the conditional score, and is formulated as follows:

###### ϵ˜(xt,z) = (1 + w)ϵ(xt,fc) − wϵ(xt,f∅), (4)

where ϵ(·,·) is the denoising score network, xt is a noisy sample at timestep t, fc denotes the condition vector from Eq. (1), f∅ represents the null condition vector, and w is the guidance scale. While guidance improves image quality,

excessive scaling may cause over-saturation or artifacts. We observed that pruning certain text encoder blocks amplifies its norm by over 100× compared to the dense version, leading to abnormal images. As illustrated in Fig. 4, pruning even two blocks significantly increases ||f∅||2, causing abnormalities shown in Fig. 4c. Thus, discrepancies in f∅ must be considered. Notably, in Fig. 4c, Metric1 failed to assess image quality, while Metric2 reported smaller values for Fig. 4b, confirming its reliability and effectiveness.

Beam search-based strategy. Most blockwise pruning methods perform (1) rank the importance of the block per layer and pruning in order (Men et al., 2024), or (2) perform sequential pruning while re-evaluating blocks (Yang et al., 2024; Zhang et al., 2024). The first method is efficient, but may ignore block interactions and lead to suboptimal results. The second method acknowledges interactions among blocks, yet its greedy approach can still lead to suboptimal results. To address these issues, we propose a novel method akin to beam search, which evaluates multiple pruning paths concurrently to better account for block interactions, achieving more effective pruning without degrading performance.

Algorithm. To synthesize the proposed approach, we introduce the Skip algorithm in Algorithm 1. Before delving into the specifics, we define two key discrepancy metrics: Df

, representing Metric2 with null inputs. The algorithm is inspired by a beam search (Freitag & Al-Onaizan, 2017), iterating over each unskipped sub-block while maintaining the k beams with the smallest sum of Df

, derived from Metric2, and Df

∅

c

. This process is repeated from the k beams, iteratively updating them to ensure smaller D. Upon traversing all blocks, the algorithm produces a list of Skip indices S∗, effectively capturing interblock interactions. Finally, the blocks are pruned sequentially according to the S∗ to achieve the desired sparsity.

and Df

∅

c

Algorithm 1 Skip Algorithm Require: Calibration dataset C, dense model M, null input

Algorithm 2 Re-use Algorithm Require: Calibration dataset C, dense model M, null input

c∅, number of layers L, beam size k Ensure: Skip index list S∗

c∅, skip indices list S, re-use indices dictionary R Ensure: Re-use indices dictionary R

- 1: R ← ∅, Mˆ ← Prune(M,S)
- 2: DM ← GetDiscrepancy(M,Mˆ ,C,c∅)
- 3: for each i ∈ S do
- 4: l ← max{j < i | j ∈/ S}, r ← min{j > i | j ∈/ S}
- 5: Mˆ l ← Update(Mˆ , R ∪ {i : l})
- 6: Mˆ r ← Update(Mˆ , R ∪ {i : r})
- 7: DM ← GetDiscrepancy(M, Mˆ , C, c∅)
- 8: Dl ← GetDiscrepancy(M, Mˆ l, C, c∅)
- 9: Dr ← GetDiscrepancy(M, Mˆ r, C, c∅)
- 10: if Dl < DM ∧ Dl < Dr then
- 11: R ← R ∪ {i : l}
- 12: else if Dr < DM ∧ Dr < Dl then
- 13: R ← R ∪ {i : r}
- 14: end if
- 15: Mˆ ← Update(Mˆ , R)
- 16: end for
- 17: return R

- 1: S ← []
- 2: B ← {(0,S)}, S∗ ← S
- 3: while ∃(D,S) ∈ B such that |S| < L do
- 4: Bnew ← ∅
- 5: for each (D,S) ∈ B do
- 6: for each layer index i ∈/ S do
- 7: S′ ← Append(S,i)
- 8: Mˆ ← Prune(M,S′)
- 9: D′ ← GetDiscrepancy(M,Mˆ ,C,c∅)
- 10: Bnew ← Bnew ∪ {(D′,S′)}
- 11: end for
- 12: end for
- 13: Update Bnew to smallest k candidates with D
- 14: (D∗,S∗) ← arg min(D′,S′) D′ in B
- 15: end while
- 16: return S∗ Function Definitions:

- 1: GetDiscrepancy(M,Mˆ ,C,c∅):
- 2: Df

c ← MSE(M(C),Mˆ (C))

- 3: Df

∅ ← MSE(M(c∅),Mˆ (c∅))

- 4: return Df

where Fi : (zi,θi)  → Rd is the i-th block with parameters θi, and zi ∈ Rd. Assume that Fi is Li-Lipschitz in zi and Mi-Lipschitz in θi. Then, for any two parameter sets θ = (θ1,...,θL) and θˆ = (θˆ1,...,θˆL), the following holds:

+ Df

∅

c

M(x;θ) − M(x;θˆ)

L

L

(6)

##### 3.2. Re-use Algorithm

(1 + Lk) Mi θi − θˆi := U

≤

Feasibility of reusing blocks. Despite extensive research on methods such as recurrent networks (Sherstinsky, 2020; Gu & Dao, 2023) which loop the output of the network back into the input for efficiency, the practice of reusing specific layers in neural networks by reintegrating hidden states into internal layers remains understudied. We conducted a feasibility study to investigate whether the internal components of the T2I diffusion text encoder serve analogous functions (see Fig. 3b). We randomly sampled tokens from the embedding, passed them through each transformer block as a fixed input, and measured the similarity of their output. The results indicate significant similarity between adjacent blocks, suggesting that performance can be restored by reintroducing non-omitted layers into adjacent skipped layers. We also verified the existence of condition for Re-use that achieves a tighter error bound compared to Skip alone. The existence is formalized in Theorem 3.2, which theoretically demonstrates the advantages of incorporating the Re-use phase. To establish this result, we first introduce the following lemma.

i=1

k=i+1

The proof for Lemma 3.1 is in the Appendix Sec. A.1. With the lemma, we can prove the following theorem.

Theorem 3.2 (Tighter error bound of Re-use). Under the assumptions of Lemma 3.1, let θi∗ be the parameters of the reused Fi. Define USkip as the error bound for the compressed model with Skip alone and USkip, Re-use as the error bound for the compressed model with Skip and Re-use. If ∥θi − θi∗∥ < ∥θi∥, then the following holds:

USkip, Re-use < USkip. (7)

This theorem establishes the theoretical feasibility of Reuse by showing a existence of condition under which the error bound becomes tight with its application. The proof for Theorem 3.2 is shown in the Appendix Sec. A.2.

Algorithm. The Re-use algorithm provided in Algorithm 2 enhances performance of pruned models by reintroducing adjacent layers for skipped layers. Starting with pruning based on skip indices S, it evaluates each skipped block for reuse with discrepancy score D between pruned and dense model outputs under three configurations: current

Lemma 3.1 (Error bound of two transformers). Let M : (x,θ)  → Rd be an L-block transformer with input x ∈ Rd and parameter set θ = (θ1,...,θL), defined as:

M = (FL + I) ◦ (FL−1 + I) ◦ ··· ◦ (F1 + I) (5)

Table 1. Quantitative comparisons of Skrr with baselines. We compared Skrr with the baselines of ShortGPT, LaCo, and FinerCut under three different sparsity scenarios on PixArt-Σ. The results show that Skrr reliably maintains image fidelity and performs comparable to the dense model across all given sparsity levels. Unlike ShortGPT and LaCo, FinerCut and Skrr use sub-block pruning, hindering direct sparsity level alignment. Sparsity levels were matched as closely as possible for fair evaluation, reflecting how much the compressed model’s parameters differ from the dense encoder. (↑ / ↓ denotes that a higher / lower metric is favorable.)

Sparsity

GenEval ↑

Method

FID ↓ CLIP ↑ DreamSim ↑

(%) Single Two Count. Colors Pos. Color attr. Overall Dense 0.0 22.89 0.314 1.0 0.988 0.616 0.475 0.795 0.108 0.255 0.539

24.3 24.96 0.309 0.753 0.944 0.381 0.431 0.715 0.033 0.083 0.431 32.4 27.28 0.294 0.651 0.834 0.197 0.291 0.537 0.048 0.038 0.324 40.5 55.26 0.215 0.357 0.306 0.025 0.090 0.100 0.0 0.0 0.087

ShortGPT

24.3 19.45 0.311 0.726 0.909 0.336 0.394 0.713 0.065 0.128 0.424 32.4 24.70 0.303 0.677 0.781 0.227 0.250 0.606 0.043 0.040 0.325

LaCo

- 40.5 21.60 0.291 0.620 0.784 0.162 0.150 0.489 0.030 0.033 0.275

FinerCut

26.3 20.66 0.313 0.798 0.947 0.465 0.394 0.737 0.103 0.105 0.458 32.2 20.49 0.313 0.771 0.903 0.409 0.344 0.697 0.078 0.128 0.426

- 41.7 20.36 0.308 0.731 0.841 0.306 0.306 0.628 0.050 0.073 0.367

27.0 20.15 0.315 0.800 0.956 0.434 0.425 0.763 0.095 0.145 0.471 32.4 20.19 0.313 0.775 0.928 0.397 0.413 0.774 0.100 0.118 0.455 41.9 19.93 0.312 0.741 0.913 0.410 0.450 0.755 0.055 0.068 0.442

Skrr (Ours)

state, reuse of the previous sub-block, and reuse of the subsequent sub-block. The configuration with the smallest D is selected, ensuring alignment with the dense model. MHA and FFN sub-blocks are reused as their respective types, maintaining consistency and functionality. The sub-block with the lowest discrepancy is reintroduced, iteratively updating the dictionary for Re-use R until all skipped layers are evaluated. This results in a refined dictionary R, allowing efficient compression while maintaining performance.

#### 4. Experiments

Baselines. To evaluate the performance of Skrr, we compared multiple LLM-based blockwise pruning techniques for T2I synthesis, including ShortGPT (Men et al., 2024), LaCo (Yang et al., 2024) and FinerCut (Zhang et al., 2024). Pruning in diffusion pipelines specifically targets the T5XXL (Raffel et al., 2020), which constitutes most of the parameter size. Detailed configurations and implementation specifics are available in Appendix Sec. B.1 and Sec. B.2.

Dataset. Most blockwise pruning methods rely on a calibration dataset to identify and remove less influential blocks. To this end, we constructed a calibration set by sampling 1k text prompts from the CC12M (Changpinyo et al., 2021), specifically tailored for T2I synthesis. The detailed configuration and example are provided in Appendix Sec. B.3.

##### 4.1. Quantitative results

Metrics. We evaluated the performance of the Skrr and baseline models with four metrics: Fr´echet Inception Dis-

tance (FID) (Heusel et al., 2017), CLIP (Radford et al., 2021) score and DreamSim (Fu et al., 2023) score with MS-COCO (Lin et al., 2014) 30k validation set, alongside GenEval (Ghosh et al., 2024). FID measures the real versus generated image similarity using Inception-v3 features (Szegedy et al., 2016). The CLIP score measures the semantic alignment of the image-text. DreamSim assesses the composition and color similarity in pruned text encoders versus the original. GenEval measures pruned text encoders’ effects on T2I synthesis across six dimensions: single object, two objects, counting, color accuracy, positional alignment, and color attribution. See Appendix Sec. B.4 for details.

T2I synthesis performance comparison. We evaluated the T2I synthesis performance of Skrr against various baselines and metrics, as shown in Table 1. ShortGPT maintains performance at low sparsity, but its performance deteriorates rapidly as sparsity increases. A similar pattern is observed for LaCo. FinerCut exhibits a more gradual decline in performance compared to other baselines, but the generated images still show a significant drop in GenEval scores. In contrast, Skrr demonstrates performance comparable to dense models in all the sparsity levels, achieving high fidelity while preserving strong text alignment metrics especially in high-sparsity settings. Interestingly, in some cases, the FID score for compressed models improves, while DreamSim, CLIP, and GenEval scores degrade. This indicates that compressing text encoders preserves or boosts image quality, while text alignment declines. We provide further detailed analysis in Sec. 5.

Dense ShortGPT LaCo

###### FinerCut Skrr (Ours)

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

(sparsitylevel1)(sparsitylevel2)(sparsitylevel3)

ΣPixArt-

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

“A vast galaxy with swirling nebulae in vivid reds, purples, and golds, and a radiant, glowing planet with rings reflecting light like a prism.”

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

ΣΣPixArt-PixArt-

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

“A lively fountain square where people play musical instruments, with rainbows refracting in the mist.”

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

“A child holding a lantern with a softly glowing golden light, standing in a field of tall grass under a deep indigo sky filled with sparkling stars.”

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

(sparsitylevel3)(sparsitylevel3)

SD3FLUX.1-dev

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

“A clock with intricate gears exposed, frozen in time.”

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

“A young artist with paint-splattered hands sitting cross-legged on the floor of a brightly lit studio. She’s wearing round glasses and a colorful apron, surrounded by unfinished canvases and jars of paintbrushes. Her focused expression hints at a masterpiece in progress.”

- Figure 5. Comparison of images generated with baseline and Skrr-compressed text encoders across PixArt-Σ, Stable Diffusion 3 (SD3), and FLUX.1-dev. At low sparsity (level 1–24.3% for ShortGPT and Laco, 26.3% for FinerCut, and 27.0% for Skrr), both methods perform comparably to dense models, but Skrr outperforms at higher sparsity(level 2–32.4% for ShortGPT and Laco, 32.2% for FinerCut, and 32.4% for Skrr, level 3–40.5% for ShortGPT and Laco, 41.7% for FinerCut, and 41.9% for Skrr), maintaining alignment to dense model and preserving details in the prompt such as “glasses”, “colorful apron”, and “paint-splattered hands”, where baseline methods fail.

Computational cost. To evaluate the efficiency of Skrr, we compared the computational cost of dense and pruned models. We measured the number of parameters, memory usage, and total FLOPs within the PixArt-Σ pipeline, with the model precision standardized to Bfloat16. As shown in Table 2, Skrr significantly reduces the parameters and memory usage similar to other baselines compared to dense model. While its FLOPs are slightly higher than the other baselines, this is negligible, since the text encoder accounts for only a small portion (0.6%) of the pipeline’s total FLOPs.

Table 2. Number of parameters (Param.), memory usage (Mem.), and FLOPs were evaluated on PixArt-Σ across pruning methods, considering the entire pipeline to analyze each strategy’s impact on computational cost. All metrics were measured at the maximum sparsity achieved by each pruning method.

Method Sparsity (%) Param. (B) Mem. (GB) TFLOPs Dense 0.0 5.42 10.18 91.94

ShortGPT 40.5 3.49 6.59 91.79

FinerCut 41.7 3.43 6.48 91.74 Skrr (Ours) 41.9 3.43 6.46 91.90

Skip only

###### Dense Skip + Re-use (Skrr)

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

(sparsitylevel3)(sparsitylevel3)

ΣPixArt-

[Figure 77]

[Figure 78]

[Figure 79]

“Two red balloons tied to a mailbox on a sunny suburban street.”

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

FLUX.1-dev

[Figure 86]

[Figure 87]

[Figure 88]

“A young artist with paint-splattered hands sitting cross-legged on the floor of a brightly lit studio. She’s wearing round glasses and a colorful apron, surrounded by unfinished canvases and jars of paintbrushes. Her focused expression hints at a masterpiece in progress.”

- Figure 6. Ablation study on Re-use. Without Re-use, Skip alone leads to images that often misalign with the prompt, while Re-use ensures more faithful adherence to the prompt.

##### 4.2. Qualitative results

We present qualitative results that demonstrate the performance of T2I synthesis using Skrr-compressed text encoders, extending the experiments to SD3 and FLUX, stateof-the-art diffusion models. As shown in Fig. 5, ShortGPT achieves satisfactory image quality at low sparsity, but diverges significantly at higher sparsity levels, failing to align with input text at sparsity over 40% (level 3). LaCo and FinerCut maintain image fidelity across sparsity levels but show reduced alignment with the dense model. In contrast, Skrr consistently preserves both image quality and alignment, closely resembling the outputs of dense models. For SD3 and FLUX, which use multiple text encoders, image fidelity remains intact at higher sparsity levels, but similarity to dense encoder output decreases. These results highlight Skrr’s robustness in preserving image quality and alignment to original image while exhibiting consistent behavior across models under varying sparsity conditions.

##### 4.3. Ablation study

We conducted an ablation study to evaluate the contribution of each component in the Skrr framework. Specifically, we analyzed the effectiveness of Re-use, highlighting its role in minimizing performance degradation from Skip. We also examined the size of the beam, demonstrating its effectiveness in block selection. The experiments were carried out on PixArt-Σ and a subset of the MS-COCO (Lin et al., 2014) validation dataset with the highest sparsity. We provide additional ablation study in Appendix Sec. C.5.

Influence of Re-use. The Re-use phase addresses performance degradation from the Skip phase, optimizing within memory constraints. Its impact was assessed by comparing model performance with and without Re-use. As shown in Fig. 6, Skip alone preserves high fidelity but may fail to accurately reflect the text prompt. In contrast, incorporating Re-use produces images closely resembling those from the

Table 3. T2I performance with various beam size k. While larger k values incur higher computational costs for the candidate search, they enhance T2I performance. (k = 1 is a greedy approach.)

Sparsity

GenEval (%) Single. Count. Colors.

k CLIP DreamSim

- 41.9 1 0.310 0.737 0.900 0.372 0.739
- 41.9 2 0.312 0.757 0.912 0.450 0.731
- 41.9 3 0.313 0.746 0.912 0.450 0.755 40.7 4 0.310 0.739 0.925 0.328 0.707

dense model, ensuring better alignment with the prompt.

Influence of beam search. We performed an ablation study that evaluated the effect of the beam search at different values of k, as shown in Table 3. As k increases, performance initially improves and then decreases. This trend is consistent with previous findings (Cohen & Beck, 2019; He et al., 2023), which observed similar behavior in LLM decoding in which performance increases and then deteriorates as the size of the beam increases. Based on this observation, we selected the optimal beam size k = 3.

#### 5. Discussion

Previously, we noted better FID results with T2I synthesis when using a text encoder with skipped layers. Beyond CFG, other guidance methods in diffusion models include perturbed attention guidance (PAG) (Ahn et al., 2024) and autoguidance (Karras et al., 2024), both of which approximate the unconditional score by modifications of denoising networks. We propose that layer skipping or merging affects the null text embedding similar to these methods. To verify this hypothesis, we perturbed f∅, as described below.

fˆ∅ = λz + f∅, z ∼ N(0,I), (8)

where z is a random vector sampled from normal distribution and λ is small scalar value. The FID and CLIP scores of the original model and the unconditional output with small perturbations applied to f∅ were measured on the MSCOCO 30k dataset. The results show that the FID decreases, indicating that the fidelity improved when perturbations are introduced to the unconditional feature f∅. Detailed results and configurations are provided in Appendix Sec. C.3.

#### 6. Conclusion

In this paper, we introduce Skip and Re-use Layers (Skrr), an effective compression method for the text encoder in text-to-image (T2I) diffusion models. Skrr integrates three key components: (1) a pruning metric based on the Skrr dot product to identify redundant sub-blocks, (2) a beam searchbased algorithm to account for interactions between transformer blocks during pruning, and (3) a re-use mechanism

that mitigates performance degradation by leveraging the remaining layers to recover lost capacity from skipped blocks with theoretical supports. Extensive experiments demonstrate that Skrr consistently outperforms existing blockwise pruning techniques for text encoder compression in image synthesis tasks, achieving qualitatively and quantitatively superior results. Additionally, our analysis reveals that pruning or merging layers not only reduces model complexity but can also enhance certain aspects of performance. We further analyze these improvements from the perspective of model guidance, offering insights into how structural adjustments contribute to more effective T2I generation.

#### Acknowledgements

This work was supported in part by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [NO.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)] and the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (No. NRF2022M3C1A309202211). Also, the authors acknowledged the financial support from the BK21 FOUR program of the Education and Research Program for Future ICT Pioneers, Seoul National University.

#### References

Ahn, D., Cho, H., Min, J., Jang, W., Kim, J., Kim, S., Park, H. H., Jin, K. H., and Kim, S. Self-rectifying diffusion sampling with perturbed-attention guidance. In ECCV, pp. 1–17. Springer, 2024.

Ashkboos, S., Croci, M. L., Nascimento, M. G. d., Hoefler, T., and Hensman, J. Slicegpt: Compress large language models by deleting rows and columns. arXiv preprint arXiv:2401.15024, 2024a.

Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B., Cameron, P., Jaggi, M., Alistarh, D., Hoefler, T., and Hensman, J. Quarot: Outlier-free 4-bit inference in rotated llms. arXiv preprint arXiv:2404.00456, 2024b.

Brooks, T., Holynski, A., and Efros, A. A. Instructpix2pix: Learning to follow image editing instructions. In CVPR, pp. 18392–18402, 2023.

Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., and Zheng, Y. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In CVPR, pp. 22560–22570, 2023.

Castells, T., Song, H.-K., Kim, B.-K., and Choi, S. Ldpruner: Efficient pruning of latent diffusion models using task-agnostic insights. In CVPR, pp. 821–830, 2024a.

Castells, T., Song, H.-K., Piao, T., Choi, S., Kim, B.-K., Yim, H., Lee, C., Kim, J. G., and Kim, T.-H. Edgefusion: On-device text-to-image generation. arXiv preprint arXiv:2404.11925, 2024b.

Changpinyo, S., Sharma, P., Ding, N., and Soricut, R. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, pp. 3558– 3568, 2021.

Chen, J., Ge, C., Xie, E., Wu, Y., Yao, L., Ren, X., Wang, Z., Luo, P., Lu, H., and Li, Z. Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In ECCV, pp. 74–91. Springer, 2024.

Cohen, E. and Beck, C. Empirical analysis of beam search performance degradation in neural sequence models. In ICML, pp. 1290–1299. PMLR, 2019.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML. PMLR, 2024.

Fang, G., Ma, X., and Wang, X. Structural pruning for diffusion models. In NeurIPS, 2023.

Freitag, M. and Al-Onaizan, Y. Beam search strategies for neural machine translation. ACL, pp. 56, 2017.

Fu, S., Tamir, N., Sundaram, S., Chai, L., Zhang, R., Dekel, T., and Isola, P. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344, 2023.

Ganjdanesh, A., Shirkavand, R., Gao, S., and Huang, H. Not all prompts are made equal: Prompt-based pruning of text-to-image diffusion models. arXiv preprint arXiv:2406.12042, 2024.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. NeurIPS, 36, 2024.

Gromov, A., Tirumala, K., Shapourian, H., Glorioso, P., and Roberts, D. A. The unreasonable ineffectiveness of the deeper layers. arXiv preprint arXiv:2403.17887, 2024.

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

He, J., Sun, S., Jia, X., and Li, W. Empirical analysis of beam search curse and search errors with model errors in neural machine translation. In EAMT, pp. 91–101, 2023.

He, Y., Liu, L., Liu, J., Wu, W., Zhou, H., and Zhuang, B. Ptqd: Accurate post-training quantization for diffusion models. NeurIPS, 36, 2024.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 30, 2017.

Ho, J. and Salimans, T. Classifier-free diffusion guidance. In NeurIPSW, 2021.

Hsieh, C.-Y., Li, C.-L., Yeh, C.-k., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.-Y., and Pfister, T. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In ACL, pp. 8003–8017, 2023.

Huang, K., Guo, X., and Wang, M. Towards efficient pretrained language model via feature correlation distillation. NeurIPS, 36:16114–16128, 2023.

Karras, T., Aittala, M., Kynk¨a¨anniemi, T., Lehtinen, J., Aila, T., and Laine, S. Guiding a diffusion model with a bad version of itself. arXiv preprint arXiv:2406.02507, 2024.

Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I., and Irani, M. Imagic: Text-based real image editing with diffusion models. In CVPR, pp. 6007–6017, 2023.

Kim, B.-K., Song, H.-K., Castells, T., and Choi, S. Bk-sdm: A lightweight, fast, and cheap version of stable diffusion. In ECCV, pp. 381–399. Springer, 2024.

Ko, J., Kim, S., Chen, T., and Yun, S.-Y. Distillm: Towards streamlined distillation for large language models. arXiv preprint arXiv:2402.03898, 2024.

Lee, Y., Lee, Y.-J., and Hwang, S. J. Dit-pruner: Pruning diffusion transformer models for text-to-image synthesis using human preference scores. In ECCVW, pp. 1–9, 2024.

Li, M., Lin, Y., Zhang, Z., Cai, T., Li, X., Guo, J., Xie, E., Meng, C., Zhu, J.-Y., and Han, S. Svdqunat: Absorbing outliers by low-rank components for 4-bit diffusion models. arXiv preprint arXiv:2411.05007, 2024a.

- Li, X., Liu, Y., Lian, L., Yang, H., Dong, Z., Kang, D., Zhang, S., and Keutzer, K. Q-diffusion: Quantizing diffusion models. In CVPR, pp. 17535–17545, 2023.
- Li, Y., Wang, H., Jin, Q., Hu, J., Chemerys, P., Fu, Y., Wang, Y., Tulyakov, S., and Ren, J. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. NeurIPS, 36, 2024b.

Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, X., Gan, C., and Han, S. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. MLSys, 6:87–100, 2024.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In ECCV, pp. 740–755. Springer, 2014.

Liu, Y., Zhang, K., Li, Y., Yan, Z., Gao, C., Chen, R., Yuan, Z., Huang, Y., Sun, H., Gao, J., et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. URL https://arxiv. org/abs/2402.17177, 2024.

Men, X., Xu, M., Zhang, Q., Wang, B., Lin, H., Lu, Y., Han, X., and Chen, W. Shortgpt: Layers in large language models are more redundant than you expect. arXiv preprint arXiv:2403.03853, 2024.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models, 2016.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In ICCV, pp. 4195–4205, 2023.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., Lee, A., Vyas, A., Shi, B., Ma, C.-Y., Chuang, C.-Y., et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Poole, B., Jain, A., Barron, J. T., and Mildenhall, B. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In ICML, pp. 8748–8763. PMLR, 2021.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 21(140):1–67, 2020.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In CVPR, pp. 10684–10695, 2022.

Ryu, H., Park, N., and Shim, H. Dgq: Distribution-aware group quantization for text-to-image diffusion models. arXiv preprint arXiv:2501.04304, 2025.

Seo, H., Kim, H., Kim, G., and Chun, S. Y. Ditto-nerf: Diffusion-based iterative text to omni-directional 3d model. arXiv preprint arXiv:2304.02827, 2023.

Sherstinsky, A. Fundamentals of recurrent neural network (rnn) and long short-term memory (lstm) network. Physica D: Nonlinear Phenomena, 404:132306, 2020.

Soboleva, D., Al-Khateeb, F., Myers, R., Steeves, J. R., Hestness, J., and Dey, N. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama, 2023.

Song, J., Oh, K., Kim, T., Kim, H., Kim, Y., and Kim, J.-J. Sleb: Streamlining llms through redundancy verification and elimination of transformer blocks. In ICML, 2024a.

Song, Y., Lorraine, J., Nie, W., Kreis, K., and Lucas, J. Multi-student diffusion distillation for better one-step generators. arXiv preprint arXiv:2410.23274, 2024b.

Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and effective pruning approach for large language models. arXiv preprint arXiv:2306.11695, 2023.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. Rethinking the inception architecture for computer vision. In CVPR, pp. 2818–2826, 2016.

van der Ouderaa, T. F., Nagel, M., Van Baalen, M., and Blankevoort, T. The llm surgeon. In ICLR, 2023.

Wang, H., Shang, Y., Yuan, Z., Wu, J., Yan, J., and Yan, Y. Quest: Low-bit diffusion model quantization via efficient selective finetuning. arXiv preprint arXiv:2402.03666, 2024a.

Wang, Z., Jiang, Y., Zheng, H., Wang, P., He, P., Wang, Z., Chen, W., Zhou, M., et al. Patch diffusion: Faster and more data-efficient training of diffusion models. NeurIPS,

- 36, 2024b.

Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., and Zhu, J. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. NeurIPS,

- 36, 2024c.

Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. Smoothquant: Accurate and efficient post-training quantization for large language models. In ICML, pp. 38087–38099. PMLR, 2023.

Yang, Y., Cao, Z., and Zhao, H. Laco: Large language model pruning via layer collapse. arXiv preprint arXiv:2402.11187, 2024.

Zhang, Y., Li, Y., Wang, X., Shen, Q., Plank, B., Bischl, B., Rezaei, M., and Kawaguchi, K. Finercut: Finer-grained interpretable layer pruning for large language models. arXiv preprint arXiv:2405.18218, 2024.

Zhao, Y., Xu, Y., Xiao, Z., Jia, H., and Hou, T. Mobilediffusion: Instant text-to-image generation on mobile devices. In ECCV, pp. 225–242. Springer, 2024.

#### A. Proofs

##### A.1. Proof of Lemma 3.1

Lemma 3.1 (Error bound of two transformers). Let M : (x,θ)  → Rd be an L-block transformer with input x ∈ Rd and parameters θ = (θ1,...,θL), defined as:

M = (FL + I) ◦ (FL−1 + I) ◦ ··· ◦ (F1 + I) , (A1) where Fi : (zi,θi)  → zi+1 is the i-th block with parameters θi, and zi+1 ∈ Rd. With each block Fi being Li-Lipschitz in the input that satisfies follows.:

||Fi(z;θi) − Fi(z′;θi)|| ≤ Li||z − z′|| (A2) And Mi-Lipschitz in the parameters which satisfies follows.:

||Fi(z;θi) − Fi(z;θi′)|| ≤ Mi||θi − θi′|| (A3) Then, for any two parameter sets θ = (θ1,...,θL) and θˆ = (θˆ1,...,θˆL), the following holds:

L

||M(x;θ) − M(x;θˆ)|| ≤

i=1

L

(1 + Lk) Mi||θi − θˆi|| (A4)

k=i+1

Proof. With Eq. A1, we can formulate the difference of two hidden states between dense model and modified model as follows.:

||zi+1 − zˆi+1|| = ||[zi + F(zi;θi)] − [ˆzi + F(ˆzi;θˆi)]|| (A5) By the triangle inequality:

||zi+1 − zˆi+1|| ≤ ||zi − zˆi|| + ||F(zi;θi) − F(ˆzi;θˆi)||

(A6)

(A)

We now split term (A) with the assumptions in Eq. A2 and Eq. A3:

(A) = ||F(zi;θi) − F(ˆzi;θˆi)|| (A7) ≤ ||Fi(zi;θi) − Fi(ˆz;θi)|| + ||Fi(ˆzi;θi) − Fi(ˆzi;θˆi)|| (A8) ≤ Li||zi − zˆi||

+ Mi||θi − θˆi|| parameter Lipschitz

(A9)

input Lipschitz

So combining,

||zi+1 − zˆi+1|| ≤ ||zi − zˆi|| + Li||zi − zˆi|| + Mi||θi − θˆi|| (A10) Hence,

||zi+1 − zˆi+1|| ≤ (1 + Li)||zi − zˆi|| + Mi||θi − θˆi|| (A11) Define the error at block i as

Ei = ||zi − zˆi|| (A12) Eq. A11 becomes

Ei+1 ≤ (1 + Li)Ei + Mi||θi − θˆi|| (A13) We start from E1 = ||z1 − zˆ1|| = ||x − x|| = 0 (since both netowrks see the same input x). Thus:

- E2 ≤ (1 + L1)E1 + M1||θ1 − θˆ1|| = M1||θ1 − θˆ1|| (A14)
- E3 ≤ (1 + L2)E2 + M2||θ1 − θˆ1|| ≤ (1 + L2)[M1||θ1 − θˆ1|| + M2||θ2 − θˆ2||] (A15)

If we do recursive telescoping over all blocks, we get

L

EL+1 = ||zL+1 − zˆL+1|| ≤

i=1

L

(1 + Lk) Mi||θi − θˆi|| (A16)

k=i+1

Since M(x;θ) = zL+1 and M(x;θˆ) = zˆL+1, we have shown:

L

||M(x;θ) − M(x;θˆ)|| = ||zL+1 − zˆL+1|| ≤

i=1

L

(1 + Lk) Mi||θi − θˆi|| (A17)

k=i+1

| |
|---|

##### A.2. Proof of Theorem 3.2

Theorem 3.2 (Tighter error bound of Re-use). Under the same assumptions in Lemma 3.1, let θi denote the i-th block of a transformer that is skipped, θi∗ represent the corresponding Re-used block, USkip is a error bound for compressed model with Skip alone, and USkip, Re-use is a error bound for a compressed model with skip and reuse. If following condition is satisfied,

||θi − θi∗|| < ||θi|| (A18) then, the following inequality holds:

USkip, Re-use < USkip (A19)

Proof. With Lemma 3.1, we can formulate the error bound of compressed model as follows:

||M(x;θ) − M(x;θˆ)|| ≤

L

Ci||θi − θˆi|| (A20)

i=1

where Ci is a constant for each i-th block. For the unskipped block, θi = θˆi, ||θi − θˆi|| = 0. So, we define a parameter set θˆSkip that exclude θi in the parameter set θ. And we can rewrite the Lemma 3.1 as follows:

||M(x;θ) − M(x;θˆSkip)|| ≤

Ci||θi − θˆi|| (A21)

i∈S

where S is a set of skipped (pruned) block indices. And skipped block can be represented as follows:

θˆi,Skip = 0 (A22) Then, we can manipulate the error bound Eq. A21 with following.

||M(x;θ) − M(x;θˆSkip)|| ≤

Ci||θi|| = USkip (A23)

i∈S

Re-use substitute the skipped weight to the weight of adjacent block:

θˆi,Re-use = θi∗ (A24) If we make set of i that satisfies Eq. A18 and denote as R then apply Re-use,

||M(x;θ) − M(x;θˆSkip, Re-use)|| <

i∈S∧i/∈R

Ci||θi|| +

Cj||θj − θj∗|| = USkip, Re-use (A25)

j∈R

Since the R consists of block indices that satisfies Eq. A18, we have shown:

USkip, Re-use =

Ci||θi|| +

i∈S∧i/∈R

Cj||θj − θj∗|| <

j∈R

Ck||θk|| = USkip (A26)

k∈S

| |
|---|

#### B. Detailed Experimental Setup

##### B.1. Baseline configurations

- Table A1. Block indices ordered by the Block Influence (BI) score obtained from ShortGPT with our calibration dataset. Blocks with lower BI scores are ranked earlier, while those with higher BI scores are ranked later. During block pruning, blocks with the lowest BI scores are pruned first, according to the specified pruning ratio.

Order of block index 10, 11, 8, 9, 12, 13, 5, 14, 6, 4, 3, 7, 15, 17, 16, 18, 2, 19, 20, 21, 22, 1, 23, 0

ShortGPT (Men et al., 2024) ShortGPT calculates the Block Influence (BI) score by measuring cosine similarity between intermediate features of each layer, extracted from the calibration data, and then taking its complement. We calculated the BI scores using the 1k calibration subset of the CC12M dataset that we constructed and sorted the blocks in ascending order based on their BI scores. The ordered block indices are presented in Table A1.

LaCo (Yang et al., 2024) LaCo presents an algorithm that merges adjacent transformer layers on the basis of the cosine similarity between their output features and those of the original model. If similarity exceeds a predefined threshold, the layers are merged to reduce the size of the model. However, our experiments revealed that LaCo’s performance is highly sensitive to its hyperparameter settings. In some cases, the algorithm failed to effectively reduce the number of parameters.

To ensure a fair comparison, we conducted thorough and extensive experiments to identify the optimal hyperparameters that maximize LaCo’s performance in the Text-to-Image (T2I) diffusion model. The selected hyperparameters are as follows: Layer collapse interval I = 2, Number of layers to merge M = 2, First layer to merge L = 1, Last layer to merge H = 24, cosine similarity threshold T = 0.7.

We applied LaCo by continuously merging layers until the compressed model achieved the desired target sparsity.

- Table A2. Sub-block indices ordered with MSE metrics from FinerCut with our calibration dataset. For the efficiency, we extracted top-24 sub-block indices for pruning. During the pruning process, blocks with the lowest MSE are pruned first, according to the specified pruning ratio.

Order of sub-block index 46, 5, 11, 6, 47, 44, 20, 19, 18, 45, 40, 41, 1, 32, 31, 17, 24, 23, 3, 42, 43, 30, 38, 37

FinerCut (Zhang et al., 2024) FinerCut employs a finer-grained blockwise pruning strategy based on the structural decomposition of transformer blocks into two distinct sub-blocks: Multi-Head Attention (MHA) and Feed-Forward Network (FFN) sub-blocks. To assess the importance of each sub-block, FinerCut originally proposed various evaluation metrics: 1) cosine similarity, 2) mean-squared error (MSE), and 3) Jensen-Shannon divergence (JSD). In its original implementation for auto-regressive LLMs, FinerCut adopted JSD after evaluating the effectiveness of these metrics. However, since the text encoder in the T2I diffusion model lacks a language modeling head, metrics based on perplexity and JSD could not be applied. For the fair comparison, we implemented FinerCut using MSE as the sole metric to evaluate sub-block importance, which is the closest metric with our discrepancy metric. We also conducted experiments of FinerCut with cosine similarity in Sec. C.6.

The sorted sub-block indices determined by FinerCut are presented in Table A2. The even indices represents the MHA blocks and the odd indices denote the FFN blocks. Due to FinerCut’s sub-block-level pruning granularity, it was not possible to achieve the exact sparsity ratios as ShortGPT and LaCo. Therefore, pruning was carried out up to the sub-block index that most closely matched the target sparsity for a fair comparison.

Skrr (Ours) This section details the order of indices determined during the Skip phase of Skrr. Because the projection module is incorporated into the denoising module of each diffusion model, the sub-block indices vary across different diffusion models. We executed the Skip phase for all models, and the resulting order of indices for each model is presented in the Table A3. This provides insight into how the indices are prioritized during the Skip phase for various diffusion architectures.

Additionally, the re-use indices of Skrr in PixArt-Σ are presented in Table A4, Table A5, and Table A6. The re-use indices were independently computed using the skip indices for each sparsity level, and overall, the later blocks exhibited a tendency not to be reused.

- Table A3. Sub-block indices ordered with discrepancy metric D from Skrr with our calibration dataset in PixArt-Σ, Stable Diffusion 3 (SD3), and FLUX.1-dev. For the efficiency, we extracted top-24 sub-block indices for pruning. During the pruning process, blocks with the lowest discrepancy are pruned first, according to the specified pruning ratio.

Model Order of sub-blocks PixArt-Σ 46, 5, 6, 47, 44, 2, 11, 24, 23, 17, 18, 45, 10, 30, 29, 40, 39, 38, 37, 42, 25, 22, 43, 36 SD3 46, 5, 45, 6, 11, 2, 22, 23, 17, 18, 30, 29, 43, 44, 20, 1, 41, 42, 40, 39, 38, 37, 21, 16 FLUX.1-dev 46, 45, 5, 6, 11, 20, 19, 22, 23, 30, 29, 47, 2, 44, 32, 31, 18, 17, 37, 36, 43, 40, 39, 42

|Skipped Block|Re-used Block<br><br>|
|---|---|
|2<br><br>5<br><br>6<br><br><br>10<br><br>11<br><br><br>17<br><br>18<br><br><br>23<br><br>24<br><br><br>29<br><br>30<br><br><br>39<br><br>40<br><br><br>44<br><br>45<br><br>46<br><br>47<br><br><br>|4 7 4 13<br><br>19<br>20<br>21<br><br><br>-|

|Skipped Block|Re-used Block<br><br>|
|---|---|
|2<br><br>5<br><br>6 11<br><br><br>17<br><br>18<br><br><br>23<br><br>24<br><br><br>44<br><br>45<br><br>46<br><br>47<br><br><br>|4 7 13 19 16 21 -|

|Skipped Block|Re-used Block<br><br>|
|---|---|
|2<br><br>5<br>6<br><br><br>10<br>11<br><br><br>17<br>18<br><br><br>23<br>24<br>25<br><br><br>29<br>30<br><br><br>37<br>38<br>39<br>40 42<br><br><br>44<br>45<br>46<br>47<br>|0 7 4 12 9 19 16 21 27 -<br><br>|

Table A4. Skrr Re-use indices of sparsity level 1 in PixArt-Σ.

Table A5. Skrr Re-use indices of sparsity level 2 in PixArt-Σ.

Table A6. Skrr Re-use indices of sparsity level 3 in PixArt-Σ.

##### B.2. Diffusion model and text encoder

For the quantitative comparison, we performed experiments on two diffusion transformer (Peebles & Xie, 2023) (DiT) text encoders: PixArt-Σ (Chen et al., 2024). PixArt-Σ employs the T5-XXL model (Raffel et al., 2020) as its text encoder. We have also conducted experiments on compressing text encoders that leverages several text encoders. For example, Stable Diffusion 3 (SD3) leverages CLIP-L, CLIP-G, and T5-XXL. The results of compressing multiple text encoders are presented in the Sec. C.5. For quantitative and qualitative evaluations, we measured discrepancy and generated images using Bfloat16 precision with the pretrained weights PixArt-alpha/PixArt-Sigma-XL-2-1024-MS obtained from the Hugging Face Diffusers library. For PixArt-Σ, all images were generated in the resolution of 512×512. Furthermore, the FLUX model and SD3, included in the qualitative results, was evaluated using the stabilityai/stable-diffusion-3-medium and black-forest-labs/FLUX.1-dev weights from the Hugging Face Diffusers library. The text encoder configuration and inference precision for both models were consistent with the aforementioned setup, using the Bfloat16 precision and applying the same pruning strategy to ensure a fair and consistent evaluation. Additionally, we fixed the number of function evaluations (NFE) to 20 across all diffusion models. All other hyperparameters, such as the classifier-free guidance (Ho &

Salimans, 2021) scale, were set to the default value. All experiments were performed on a single Nvidia A100 or RTX 4090 GPU.

##### B.3. Calibration dataset

We constructed a calibration set derived from the CC12M (Changpinyo et al., 2021) dataset to identify blocks for pruning. While existing LLM-based methods typically utilize calibration sets such as the Common Crawl’s web corpus (Raffel et al., 2020) (C4), SlimPajama (Soboleva et al., 2023) and WikiText (Merity et al., 2016), primarily focusing on perplexity-based performance metrics, these approaches are not directly applicable to T2I models. To address this gap, we curated a calibration set specifically tailored for the T2I text encoder by selecting only clean captions and semantically rich prompts ranging from 150 to 250 tokens from the CC12M image-text paired dataset. This selection ensures a more effective calibration for image synthesis tasks. Representative examples of prompts from the constructed dataset are provided in Table A7.

##### B.4. Metrics

Fr´echet Inception Distance (FID) (Heusel et al., 2017) score The Fr´echet Inception Distance (FID) is a widely used metric for evaluating the performance of image generative models by quantifying the similarity between the distributions of real and generated images. Specifically, FID measures the Fr´echet distance between feature representations extracted from a pre-trained image classification model, typically the Inception-V3 model. This approach leverages the model’s rich intermediate features to capture high-level image statistics. The FID score is formally defined as follows:

- 1

- 2 (A27)

dF(N(µ,Σ),N(µ′,Σ′)) = ||µ − µ′||22 + tr Σ + Σ′ + 2(ΣΣ′)

where µ and Σ are the mean and covariance of the feature representations of real images, µ′ and Σ′ are the mean and covariance of the feature representations of generated images. A lower FID score indicates that the generated images are more similar to the real images in both quality and diversity.

CLIP (Radford et al., 2021) score The CLIP score evaluates the semantic alignment between a given text prompt and the image generated from that prompt by measuring the cosine similarity between their CLIP embeddings. This metric leverages a CLIP model trained on extensive image-text pairs to capture cross-modal relationships. The CLIP score can be formally defined as:

Eimage(I) · Etext(T) ||Eimage(I)|||Etext(T)||

CLIP(I,T) = cos(Eimage(I),Etext(T)) =

(A28)

where I is the generated image, T is the text prompt, Eimage(·) represents the CLIP image encoder, and Etext(·) denotes the CLIP text encoder. The cosine similarity captures how well the generated image aligns with the semantic content of the text prompt. For our experiments, we leveraged the weights of the openai/clip-vit-base-patch32 model from the Hugging Face library to calculate the CLIP score.

DreamSim (Fu et al., 2023) score The DreamSim score quantifies the semantic similarity at the mid-level between two images by evaluating their compositional, stylistic, and color characteristics. It quantifies how closely the overall structure and visual attributes of the images align. Formally, the DreamSim score can be expressed as Eq. A29:

DreamSim(Iref,Ipru) = 1 − distDreamSim(Iref,Ipru), distDreamSim(·,·) ∈ [0,1] (A29)

where Iref represents the image generated using the original text encoder, and Ipru denotes the image generated using the pruned text encoder. The function distDreamSim(·,·) computes the normalized semantic distance between the two images, as output by the DreamSim model. By subtracting this distance from 1, the DreamSim score reflects higher similarity with higher values, effectively capturing the semantic consistency between the original and pruned models’ outputs.

GenEval (Ghosh et al., 2024) GenEval is a comprehensive evaluation metric designed to assess the degree to which a T2I generative model aligns generated images with input text prompts. In this study, GenEval was employed to evaluate whether the image synthesis results produced by the compressed text encoder accurately reflect the intended textual descriptions. The GenEval metric comprises six sub-metrics:

1. Single Object Generation – Assesses the model’s ability to generate images from prompts containing a single object (e.g.,

Table A7. Examples of prompts from the calibration set across various lengths are presented. These text prompts are sampled from the CC12M dataset, featuring rich and descriptive expressions with lengths optimized for calibration. This selection ensures the prompts are semantically meaningful and well-suited for effective text-to-image model calibration.

Example prompts in calibration dataset

- 1. A collection of photography equipment neatly arranged on a wooden surface. Items include a camera, smartphone, tablet, drone, portable power bank, tripod, cleaning kit, strap, case, and backpack. The warm wooden background contrasts with the modern gear.
- 2. A moment at a subway station with a vintage train numbered “7” at the platform. The platform has safety barriers and a yellow line, illuminated by fluorescent lights. Text reads “last stop for the 7 train” and credits the photographer.
- 3. A modern living room with a minimalist design. A pendant light provides a warm glow. A wooden table holds a glass of water, a book, a smartphone, and a notebook. A white cabinet and a cityscape view complete the cozy atmosphere.
- 4. A person wearing a white t-shirt with the text “A Day to Remember” in pink and black lettering. The shirt features a black collar and short sleeves, displayed plainly for product showcasing.
- 5. A vibrant digital artwork of a stylized cityscape. Buildings vary in color and pattern, resembling a patchwork quilt, creating a dense, lively urban environment.
- 6. A small modern bathroom with brick-patterned walls and tiled flooring. A white sink under a window, a glass shower enclosure, and a toilet create a rustic yet clean look.
- 7. A vintage light-colored train car with blue and white stripes is parked on a track under a metal canopy. Metal stairs lead to the entrance, possibly part of a museum exhibit.
- 8. A wall with a playful quote: “In this house we are real, we make mistakes, we say I’m sorry, we give hugs, we give second chances, we forgive, we laugh a lot, we love each other, we are a family.” A guitar leaning against the wall adds a cozy, homey touch.
- 9. A vibrant bouquet of flowers arranged in a clear glass vase. the bouquet consists of various types of flowers, including hydrangea, calla lilies, roses, and gerbera daisies, with burgundy berries interspersed among them. the flowers are in shades of pink and purple, creating a striking contrast. the arrangement is set against a plain, light-colored background, which accentuates the colors and textures of the flowers. the style of the image is a close-up photograph that captures the details of the floral arrangement.
- 10. The memorial church at stanford university, a large, ornate building with a prominent cross at the top, illuminated at night. the facade is adorned with intricate mosaics and sculptures, including a central figure that appears to be a religious figure, possibly a saint or deity. the church’s architecture is reminiscent of gothic and romanesque styles, with pointed arches and a large central archway that leads to the entrance. the surrounding area is dimly lit, with the church standing out as a beacon of light in the darkness.
- 11. A serene lakeside setting with a houseboat that resembles a private yacht. the boat is equipped with a dining area featuring a table set for four with blue tableware, and a bar area with a blender, wine glasses, and a bottle of wine. the deck is furnished with multiple lounge chairs and a dining table, all under a retractable awning. the houseboat is docked near a rocky shoreline with a clear blue sky and a majestic red rock formation in the distance, suggesting a location like lake powell. the overall atmosphere is one of relaxation and leisure, ideal for a vacation or getaway.
- 12. A graphic design with a stylized representation of a face, possibly a deity, with a serene expression. the face is framed by a green border with a white outline and a blue background. above the face, there is a crescent moon and a symbol that resembles a peace sign. below the face, the word “chill” is prominently displayed in bold, white capital letters. the overall style of the image is modern and graphic, with a clear emphasis on the word “chill” suggesting a theme of relaxation or tranquility.

“a photo of a giraffe”).

- 2. Two Objects Generation – Evaluates the model’s ability to correctly generate images from prompts with two distinct objects (e.g., “a photo of a knife and a stop sign”).
- 3. Counting – Measures whether the model can accurately represent the specified number of objects (e.g. “a photo of three apples”).
- 4. Colors - Verifies whether the generated image correctly reflects the color specified in the prompt (e.g., “a photo of a pink car”).
- 5. Position – Tests the model’s understanding of spatial relationships described in the prompt (e.g., “a photo of a sofa under

Table A8. Sparsity ratio of the text encoder, parameter count (Param.), memory usage (Mem.), FLOPs ratio of text encoder (T5-XXL) with respect to the total pipline, and total TFLOPs of the pipline were evaluated on SD3 across pruning methods, considering the entire pipeline to analyze each strategy’s impact on computational cost. All metrics were measured at the maximum sparsity achieved by each pruning method and NFE with 28.

Method Sparsity (%) Param. (B) Mem. (GB) FLOPs (%) TFLOPs Dense 0.0 7.66 14.49 0.41 174.2

ShortGPT 40.5 5.73 10.93 0.32 174.1 LaCo 40.5 5.73 10.82 0.32 174.1

FinerCut 41.7 5.67 10.82 0.30 174.0 Skrr (Ours) 41.9 5.73 10.93 0.35 174.1

a cup”).

- 6. Color Attribution – Assesses the correct assignment of specified colors to multiple objects (e.g., “a photo of a black car and a green parking meter”). For evaluation, we generated images using a fixed random seed, producing 553 distinct prompts with four images per prompt, resulting in a total of 2,212 images. This setup ensured consistent and reproducible evaluation across all sub-metrics.

- B.5. Ablation study setup

Due to constraints in our experimental environment, it was not feasible to evaluate the entire MS-COCO 30k image dataset for the ablation study. Instead, we performed the study using a 1k subset. Given that both the CLIP score and the DreamSim score effectively capture the impact of text encoder pruning, we focus our experiments on evaluating various configurations of these two metrics within the ablation study. Furthermore, for GenEval, experiments were conducted on the entire set. The experiments discussed in the Sec. C.5 were conducted using the same experimental setup as described above.

- C. Additional Experiments

- C.1. Interaction between blocks

In addition to the block pairs shown in Fig. 4, we identified additional pairs that exhibit interactions. These block pairs consistently maintain cosine similarity; however, their norms change significantly. We provide several qualitative results that highlight these interactions in Fig. A1 which is experimented with PixArt-Σ. For cases where k > 2, the number of combinations increases to k!(4848!−k)!, making exhaustive experimentation computationally infeasible. As a result, we limit our analysis to a subset of two-block interactions.

- C.2. Computational cost on different diffusion models

In the main paper, we report the computational cost only for PixArt-Σ. In this section, we extend the analysis to include the computational cost of dense, baselines, and Skrr-compressed models on Stable Diffusion 3 (SD3). Since SD3 contains more parameters and a more computationally intensive denoising module compared to PixArt-Σ, the text encoder consumes fewer FLOPs ratio in the total pipeline. However, the text encoder still accounts for significant memory usage. The detailed results are provided in Table A8. At the highest sparsity level, FinerCut achieves the lowest parameter count, memory usage, and FLOPs due to its slightly higher sparsity. Skrr exhibits the same parameter count and memory consumption as ShortGPT or LaCo at the same sparsity level (40.7%). While Skrr incurs a slightly higher FLOP count than other baselines due to the re-use mechanism, the additional computational cost is minimal, accounting for less than 0.05% of the entire pipeline or under 0.1 TFLOPS. Despite this, Skrr still demands less computation than the dense model. These results underscore ability of Skrr to perform T2I synthesis in a computationally efficient manner, even when applied to models with varying complexities.

- C.3. Perturbation to the null condition feature

We conducted an experiment to evaluate the impact of perturbations on f∅ and their effect on the FID and CLIP score. Using the PixArt-Σ model, we applied a small scalar parameter λ = 10−2 and generated images for the MS-COCO

3 skip 5 skip 3 ,5 skip 13 skip 15 skip 13 ,15 skip

Dense

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

“Color photo of a corgi made of transparent glass, standing on the riverside in Yosemite National Park.”

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

“Game-Art - An island with different geographical properties and multiple small cities floating in space.”

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

“Photorealistic closeup video of two pirate ships battling each other as they sail inside a cup of coffee.”

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

“Happy dreamy owl monster sitting on a tree bench, colorful glittering particles, forest background, detailed feathers.”

- Figure A1. Examples illustrate various block interactions. When individual blocks are skipped, the generated images remain highly similar to those produced by the dense model, showing minimal impact. However, when two blocks are skipped simultaneously, the image is severely degraded, demonstrating the presence of significant interactions between blocks.

30k validation set. The results demonstrated that the perturbed f∅ produced a lower FID score (22.89 → 20.65) and a comparable CLIP score (0.314 → 0.314) to the original model. Furthermore, we provide a qualitative comparison between the images generated with the original null condition f∅ and those created using the perturbed feature vector fˆ∅ in Fig. A2.

##### C.4. Applying Re-use to baselines

To further validate the effectiveness of Re-use, we conducted experiments to evaluate its compatibility with other block-wise pruning methods in a plug-and-play manner. Among the baselines considered for these experiments, ShortGPT and FinerCut employed block-pruning techniques, allowing us to apply Re-use and carry out the evaluations. ShortGPT prunes layer normalization, MHA, and FFN as a single unit, enabling us to perform Re-use on adjacent whole blocks. Similarly, FinerCut, which operates at the sub-block level like Skrr, also allowed the application of Re-use. The results of these experiments are presented in Fig. A3 and Fig. A4. The blocks re-used by shortGPT and Finercut are presented in Table A9 and Table A10, respectively. As shown in the results, Re-use demonstrated superior performance, further substantiating its effectiveness in alignment with both the experimental and theoretical findings presented earlier.

##### C.5. Additional ablation study

Effectiveness of Re-use. In this section, we provide additional experimental results to demonstrate the effectiveness of Re-use. This mechanism effectively mitigates the performance degradation caused by Skip by utilizing the remaining blocks in the model. To evaluate its impact, we performed extensive quantitative and qualitative experiments. Quantitatively, we

### Original Perturbed Original Perturbed Original Perturbed

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

“Young man wearing glasses lounging on a sofa with three white laptop computers on his lap.”

“A watery glass jar full of blooming flowers” “A woman carrying two skis and a couple of poles.”

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

“Apple in and over-flowing a bowl on a windowsill with a rose in a vase.”

“Dinosaur statue in a park with various kites flying through the air.”

“A family is sitting around a dinner table.”

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

“A narrow hotel room with two made up beds.” “A man holding a brown dog and a video camera.”

“A small plane flying through a cloudy blue sky.”

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

“There are many men preparing to cut a red ribbon”

“A golden clock rhino sculpture sitting on top of a fireplace.”

“A woman sitting on the back of a couch next to a man wearing glasses.”

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

“Small bird sitting on a skateboard posed in front of dark blue background cloth.”

“A guy stops himself from falling off of his skateboard.”

“A hotdog is sitting with fries in a paper car on a plate.”

- Figure A2. Images generated from the same seed under the different null condition (original vs. perturbed) reveal notable differences. While the original model occasionally exhibits poor fidelity or omits objects specified in the prompt, the images generated using the perturbed null condition feature demonstrate higher fidelity, more accurate representation of the given prompt, and improved conditional image synthesis performance. All image pairs were generated with the same seed.

Dense ShortGPT ShortGPT + Re-use Dense ShortGPT ShortGPT + Re-use

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

“Three giraffes standing amongst a copse of trees.” “Several people sitting around a table in a small room with a laptop, books and papers.”

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

“A knife and a pizza cut into slices on the plate” “a close up of a foam box of food”

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

“A crowd of young children sitting next to each other.” “A large green and yellow truck parked next to a similar green and yellow truck.”

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

ΣPixArt-

“A young boy wearing a colorful umbrella hat.” “The lunch plate includes both meat and vegetable choices.”

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

“A man standing next to another man wearing headphones.” “large mustard yellow commercial airplane parked in the airport.”

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

“A cat is curled up, asleep, on a chair.” “A red motorcycle with a back seat parked on the side of a road.”

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

“A double red bus parked on a street.” “A motorcycle parked outside in a parking lot near the beach.”

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

“A motorcycle is parked in a bushy field.” “A blue double decker bus that says Garage on it.”

- Figure A3. Qualitative results of applying Re-use to ShortGPT. At high sparsity levels, ShortGPT struggles to generate high-fidelity images. Incorporating Re-use restores fidelity, text adherence, and alignment with the dense model.

Dense FinerCut FinerCut + Re-use Dense FinerCut FinerCut + Re-use

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

“A child stops after taking a bite of a slice of pizza.” “Someone picks up a slice of fresh pizza with one hand.”

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

“A person on a motor bike on a road.” “a close up of a woman wearing jewelry and a boat in the background”

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

“The airplane is taking off the runway at the airport.” “A couple of older men sitting on top of a bench.”

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

ΣPixArt-

“A young woman riding skis down a ski slope while holding ski poles.” “Cauliflower, carrots, and broccoli are arranged in a vegetable bowl.”

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

“A close up of a vase with many flowers.” “A man riding a snowboard on top of a snow covered slope.”

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

“A view of a traffic street filled with cars and motorcycles.” “A room filled with clutter and a laptop computer.”

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

“A slice of pizza sitting on a plate on top of a box.” “A double red bus parked on a street.”

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

“A close up of a cat on a rug on the ground.” “A yellow duck boat floating on top of a lake.”

- Figure A4. Qualitative results of applying Re-use to FinerCut. While FinerCut performs well compares other baselines in maintaining fidelity and text alignment, its alignment with the dense model decreases. Incorporating Re-use effectively restores this alignment.

|Skipped Block<br><br>|Re-used Block|
|---|---|
|1 3<br><br>5<br><br>6 11<br><br><br>17<br><br>18<br><br>19<br><br>20<br><br><br>23<br><br>24<br><br><br>31<br><br>32<br><br><br>40<br><br>41<br><br>42<br><br><br>44<br><br>45<br><br>46<br><br>47<br><br><br>|7 4 9 21 16<br><br>15<br>16 21<br><br><br>29 -|

|Skipped Block|Re-used Block|
|---|---|
|4<br>5<br>6<br><br><br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>|7 7 7 7 7 7 7 7 7 7<br><br>|

Table A9. ShortGPT re-use indices

Table A10. Finercut re-use indices

measured the CLIP, DreamSim, and GenEval scores of the PixArt-Σ model with and without Re-use in the maximum sparsity. The results, presented in Table A11, show that Re-use enhances the CLIP, Dreamsim and GenEval scores. Although minor performance degradation was observed in other GenEval scores, the high performance achieved in challenging metrics like counting underscores the effectiveness of Re-use, especially in scenarios where the text encoder’s capability plays a critical role.

In addition to the quantitative findings and qualitative examples presented in the main paper, we further confirmed that Re-use outperforms Skip alone in a variety of settings. These results are illustrated in Fig. A5 for PixArt-Σ and Fig. A6 for FLUX.1-dev. These visualizations reinforce the effectiveness of Re-use in generating images that align closely with text prompts, while maintaining model-agnostic behavior. The results demonstrate that Re-use not only enhances adherence to textual descriptions but also improves the performance of dense models across diverse scenarios.

- Table A11. Quantitative ablation study for Re-use with PixArt-Σ. T2I synthesis performance with Re-use demonstrates that it effectively restores performance degraded by pruning, achieving excellent recovery without incurring additional memory overhead.

GenEval Single Two. Count. Colors Pos. Color attr. Overall

Re-use CLIP DreamSim

✗ 0.311 0.745 0.928 0.379 0.409 0.758 0.065 0.088 0.438 ✓ 0.312 0.746 0.913 0.409 0.450 0.755 0.055 0.068 0.442

Compressing multiple text encoders. Modern text-to-image (T2I) diffusion generative models frequently employ multiple text encoders to enhance their performance. A notable example is Stable Diffusion 3 (SD3), which incorporates CLIP-L, CLIP-G, and T5-XXL text encoders. In the experiments in the main article, we focused on compressing the T5-XXL text encoder, which is the largest and has the highest parameter count. Extending this approach, we evaluated the performance of the model when compressing all text encoders simultaneously. Specifically, for SD3, we applied compression to CLIP-L (30.6% sparsity), CLIP-G (30.3% sparsity) and T5-XXL (41.9% sparsity).

The quantitative results, summarized in Table A12, reveal that although the CLIP and GenEval scores experience slight reductions, they remain sufficiently comparable to those of the dense model. Furthermore, qualitative results, illustrated

Dense

Dense

###### Skip only Skip + Re-use (Skrr)

###### Skip only Skip + Re-use (Skrr)

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

“Three books stacked neatly on a desk, their spines displaying bold, colorful titles.”

“Two striped socks hanging on a clothesline in the wind.”

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

ΣPixart-

“Four vintage teacups stacked precariously on a matching saucer set.”

“Two paper cranes folded from patterned origami paper, sitting on a windowsill.”

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

“Two dolphins leaping out of the ocean under a bright blue sky.”

“Three books stacked neatly on a desk, their spines displaying bold, colorful titles.”

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

“Three oranges placed on a blue ceramic bowl on a rustic wooden table.”

“A serene countryside with rolling fields, sheep grazing, and a small stone bridge.”

- Figure A5. Qualitative results on Re-use. Images generated with PixArt-Σ and text encoder (T5-XXL) compressed by the full Skrr framework (dense, Skip only, and Skip with Re-use) are compared. With Skip only, performance noticeably degrades, particularly for detailed prompts or tasks requiring strong text encoder capabilities, such as counting, resulting in deviations from dense model outputs.

Dense

Dense

###### Skip only Skip + Re-use (Skrr)

###### Skip only Skip + Re-use (Skrr)

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

“A sculptor chiseling a marble statue in an open studio, dust and small fragments scattering with each precise strike. Sunlight reflects off the smooth, polished surfaces of finished pieces nearby.”

“Floating green cliffs with cascading waterfalls at sunset, surrounded by soft clouds and vibrant colors in the sky.”

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

FLUX.1-dev

“An ancient cliffside temple with ornate pillars, warm sunset lighting, and overgrown greenery.”

“A scientist in a laboratory surrounded by glowing vials and advanced equipment, carefully pipetting a blue liquid into a test tube. Her lab coat is slightly wrinkled, and safety goggles rest on her forehead.”

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

“A young woman with short, auburn hair sitting at a café table, reading a novel while sipping on a cappuccino. The sunlight streams through the window, casting a warm glow on her face as she pauses to reflect on the story.”

“A gardener planting seedlings in a lush community garden, surrounded by rows of vegetables and flowers. A watering can rests beside her as she gently pats the soil around the new plants.”

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

“A beekeeper in a white suit tending to a hive on a sunny farm. Bees buzz around as he carefully inspects a honeycomb, the golden liquid glistening in the light.”

“A young woman painting a mural on the wall of an orphanage, her hair tied back in a loose bun. Children gather around her, some helping with brushes, while others giggle and point at the growing artwork.”

- Figure A6. Qualitative results on Re-use. Images generated with FLUX.1-dev and text encoder (T5-XXL) compressed by the full Skrr framework (dense, Skip only, and Skip with Re-use) are compared. When only Skip is used, some prompts may not be accurately reflected, leading to images that deviate from those generated by the dense model or even shift to an animated style rather than a realistic one.

in Fig. A7, demonstrate that images generated by the compressed model using Skrr on multiple text encoders exhibit remarkable similarity to those generated by the dense model. These findings indicate that compressing multiple text encoders simultaneously does not significantly compromise the model’s ability to generate high-quality, text-aligned images, thus validating the robustness of the compression approach.

- Table A12. Quantitative ablation study for compressing multiple text encoders with Stable Diffusion 3. The second row presents the evaluation results of the model where only the T5 text encoder is compressed, while the third row corresponds to the model in which all three text encoders—T5, CLIP-L, and CLIP-G—are compressed.

Compressed CLIP DreamSim

GenEval Single Two. Count. Colors Pos. Color attr. Overall

Dense 0.318 1.0 0.994 0.869 0.600 0.856 0.280 0.538 0.689 T5 0.317 0.811 0.959 0.773 0.500 0.803 0.210 0.423 0.611 All 0.313 0.717 0.991 0.654 0.534 0.835 0.105 0.353 0.579

Skrr without projection module. To measure the discrepancy in Skrr, we tailored the evaluation for the T2I diffusion model by analyzing features extracted from the text encoder after projection through the projection module used in the denoising process. In this section, we present both quantitative and qualitative results to validate the effectiveness of incorporating the projection module. We evaluated the performance of the PixArt-Σ model compressed with Skrr, excluding the projection module. The results, summarized in Table A13, reveal that, while performance is largely preserved without the projection module, a slight degradation occurs due to loss of information. This degradation occurs because the omission of the projection module ignores crucial components that are influential in the image generation process.

- Table A13. Quantitative results of the ablation study on the projection module. Applying the projection improves CLIP score, DreamSim, and various GenEval tasks, demonstrating its significant impact on T2I generation performance when extracting features for discrepancy measurement.

GenEval Single Two. Count. Colors Pos. Color attr. Overall

Projection CLIP DreamSim

✗ 0.305 0.708 0.884 0.366 0.266 0.620 0.070 0.078 0.381 ✓ 0.312 0.746 0.913 0.409 0.450 0.755 0.055 0.068 0.442

This trend is also reflected in the qualitative results shown in Fig. A8. The images generated with the projection module exhibit a higher degree of similarity to those produced by the dense model compared to the images generated without it. These findings underscore the importance of the projection module in preserving key features necessary for accurate text-to-image generation, thereby proving its effectiveness in maintaining model performance and fidelity.

Qualitative results of beam size In the main manuscript, we quantitatively analyzed the performance variation with respect to the beam size k. Here, we provide qualitative results to further illustrate its impact on image generation. As shown in Fig. A9, increasing the beam size leads to better alignment between the generated images and the dense model, demonstrating the effectiveness of proper beam sizes in improving generation quality, but slight decline in performance as the beam size grows.

##### C.6. Additional quantitative results

FinerCut employs cosine similarity and Jensen-Shannon Divergence (JSD) in addition to the MSE used in our implementation. However, since the text encoder lacks a language head, JSD is not applicable. Moreover, FinerCut’s reported results indicate that MSE outperforms cosine similarity, leading us to adopt it in our implementation. To ensure a fair comparison, we reimplemented FinerCut using cosine similarity and evaluated its performance using FID, CLIP, DreamSim, and GenEval, as presented in Table A14. Consistent with FinerCut’s findings, cosine similarity yielded better performance compared to MSE. Consequently, we used the MSE-based FinerCut implementation as a baseline for an equitable comparison that shows higher performance.

Dense

T5 All Dense T5 All

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

“Two zebras nuzzle together in a wooded area.” “Two brown bears sitting on top of a black and white checkered bed.”

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

“A clock sitting on top of a wooden desk.” “The old, adult elephant stands near a wire fence.”

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

“An individual pizza with cheese, pineapple and sauce.” “Three brown cows grazing on a muddy grass field.”

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

StableDiffusion3

“A brown and white dog sitting in a black leather chair.” “The infamous Big Ben clock tower sitting under a cloudy blue sky.”

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

“A herd of sheep standing on a grass covered hillside.” “An adult polar bear is swimming in the water.”

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

“A couple of giraffe standing next to each other.” “Two brown bears swimming together in the water.”

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

“A shops table filled with apples oranges and other fruits.” “This pizza is square and has pepperoni and mushrooms.”

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

“A brown teddy bear sitting in the middle of a road.” “A man in a wetsuit on a surfboard in the ocean.”

- Figure A7. Qualitative results of Skrr-compressed text encoders in Stable Diffusion 3. We compare compressing only the Dense model’s output and T5-XXL (T5) versus compressing all encoders (T5, CLIP-L, and CLIP-G). Skrr maintains high image fidelity and text-image alignment across both cases.

Dense

No proj. Proj. Dense No proj. Proj.

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

“The airplane is taking off the runway at the airport.” “A man wearing goggles and standing in the snow.”

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

“A female in a green top and a laptop.” “A lit up counter with two fancy sinks on it.”

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

“Tree is a female tennis player that is playing on the court.” “A person standing in front of a plate of food.”

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

ΣPixArt-

“A little girl in a public bathroom for kids.” “A batter standing in the batter's box waiting to hit the baseball.”

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

“A man using a pair of scissors to cut a pizza” “A balding man with glasses, standing near a bridge.”

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

“A fancy dish on a white plate at a restaurant.” “A bunch of sheep are standing in a snowy field.”

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

“A man performing on a skateboard ramp jump.” “Two yellowish green vases sitting on a counter.”

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

“A clock on side of building with skyline in background.” “A man holding a smart phone in his right hand.”

- Figure A8. Qualitative comparison of using a projection layer versus no projection in PixArt-Σ. Without projection (No Proj.), text-image alignment remains similar, but the generated image deviates from the dense model’s output. With projection (Proj.), both text-image alignment and image fidelity closely match the dense model.

Dense

k=1

k=2 k=3 k=4

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

“A woman is helping a child put on skis.”

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

“A traffic sign has been vandalized to look like a scary face.”

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

ΣPixArt-

“An airport filled with planes sitting on tarmacs.”

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

“Tree is a female tennis player that is playing on the court.”

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

“A suitcase that is on the ground with some shoes.”

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

“People hold flying kites in the air by the water.”

- Figure A9. Qualitative analysis of the impact of beam size k on image generation quality. As shown, increasing the beam size enhances the alignment between generated images and the dense model, resulting in improved visual coherence and fidelity. Larger beam sizes allow for a more exhaustive exploration of the search space, leading to more refined and higher-quality outputs. These observations further support our quantitative findings presented in the main paper, demonstrating the effectiveness of using larger beam sizes.

- Table A14. Quantitative comparison of FinerCut similarity metrics, including cosine similarity and MSE, evaluated using FID, CLIP, DreamSim, and GenEval. The results confirm that MSE outperforms cosine similarity, which is consistent with the finding of ours.(↑ / ↓ denotes that a higher / lower metric is favorable).

Sparsity

GenEval ↑

Method

FID ↓ CLIP ↑ DreamSim ↑

(%) Single Two Count. Colors Pos. Color attr. Overall Dense 0.0 22.89 0.314 1.0 0.988 0.616 0.475 0.795 0.108 0.255 0.539

26.3 19.64 0.311 0.745 0.925 0.394 0.363 0.657 0.060 0.085 0.414 32.2 21.16 0.306 0.689 0.875 0.346 0.322 0.620 0.043 0.058 0.377 41.7 21.84 0.306 0.671 0.847 0.275 0.256 0.625 0.053 0.033 0.348

FinerCut (cos.)

26.3 20.15 0.315 0.800 0.947 0.465 0.394 0.737 0.103 0.105 0.458 32.2 20.19 0.313 0.775 0.903 0.409 0.344 0.697 0.078 0.128 0.426 41.7 19.93 0.312 0.741 0.841 0.306 0.306 0.628 0.050 0.073 0.367

FinerCut (MSE)

27.0 20.15 0.315 0.800 0.956 0.434 0.425 0.763 0.095 0.145 0.471 32.4 20.19 0.313 0.775 0.928 0.397 0.413 0.774 0.100 0.118 0.455 41.9 19.93 0.312 0.741 0.913 0.410 0.450 0.755 0.055 0.068 0.442

Skrr (Ours)

##### C.7. Additional qualitative results

In this section, we present additional qualitative results to further illustrate the performance of our approach. We define the model compressed with 20%-30% sparsity as sparsity level 1, 30%-40% as sparsity level 2, and 40%-50% as sparsity level 3. The qualitative comparison for sparsity level 1 is shown in Fig. A10, while comparisons for sparsity level 2 are presented in Fig. A11 and Fig. A12. Finally, Fig. A13 and Fig. A14 illustrate the comparisons for sparsity level 3. The results demonstrate that Skrr generates outputs that are more closely aligned with the prompts and more consistent with the outputs of the dense model compared to other baselines. All presented results were generated using PixArt-Σ and are part of a dataset comprising 30,000 images, which were produced for FID measurements.

Additionally, we provide a qualitative comparison of the text encoders compressed by each compression method across all sparsity levels, juxtaposed with the images generated by the dense model. By comparing images generated with the same seed at sparsity levels 1, 2, and 3, as defined above, we can assess the degree of deviation from the original image as the sparsity increases. These comparisons are illustrated in Fig. A15 and Fig. A16. The results reveal that, while the baseline methods occasionally generate images similar to those from the dense model at low sparsity, the differences become more pronounced as sparsity levels increase. In contrast, the model compressed using Skrr consistently maintains a high degree of similarity to the original images across all sparsity levels, demonstrating its robustness.

#### D. Limitations

In this section, we address the limitations of Skrr. While Skrr effectively preserves Text-to-Image (T2I) performance during pruning, its performance deteriorates noticeably at extreme sparsity levels (> 50%). This degradation is likely due to a significant reduction in the representational capacity of the text encoder as the number of parameters becomes excessively limited. However, such a sparsity could still provide additional memory efficiency through complementary techniques such as weight quantization. Another limitation is that Skrr does not achieve performance improvements beyond that of the original dense model. While pruning the text encoder can enhance certain image quality metrics, such as improving FID scores, we observed a consistent decline in performance on benchmarks like CLIP score and GenEval as sparsity increased. Addressing this performance drop and ensuring robustness across benchmarks would be an interesting future work.

“A bronze statue with a neck tie around its neck.”

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

“A stuffed animal tied to something with a chain on the ground.”

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

“A man running towards a kick ball on a field.”

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

“A brown and white cow lying in a green pasture.”

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

“A person with a tie and a suit”.

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

“A pizza sitting on top of a plate on a table.”

- Figure A10. Qualitative comparison of images generated by models compressed to sparsity range 20%-30% (sparsity level 1) using

”A couple of doughnuts are wrapped in paper.”

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

“A group of people sitting at tables next to each other.”

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

“A cat is lying on a white bedsheet.”

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

“Two brown bears embracing each other on top of a a dirt ground.”

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

“A brown dog laying on a pillow on the floor.”

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

“A jumbo jet airplane parked with a person checking it.”

“A man holding a tennis racquet into a tennis court.”

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

“Elephants stand in a large expanse of grass.”

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

“A small kitten is walking on a computer keyboard.”

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

“The two traffic lights are placed on a pole on the ground.”

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

“A bowl with some noodles inside of it.”

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

“A train that is on a large rail way.”

“A zebra standing on top of a dirt field.”

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

“A little girl smiling widely for the camera”

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

“A lot of blue and yellow umbrellas sitting under a clock.”

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

“A black cat is standing on a small desk.”

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

“A smiling man sitting in a fancy restaurant.”

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

“Two zebras nuzzle together in a wooded area.”

“a stuffed teddy bear in a pink dress holding a stuffed bouquet of roses”

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

“A bunch of sheep are standing in a snow field.”

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

“A box of apples sits in a marketplace.”

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

“A plate of food that is on a table.”

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

“There is a snow boarder doing a trick in the air.”

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

“A man standing in front of a flat screen TV.”

ShortGPT

LaCo FinerCut Skrr (Ours)

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

SparsityLev.1SparsityLev.2SparsityLev.3

Dense

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

“A cat is sitting on a desk next to a computer.”

ShortGPT LaCo FinerCut Skrr (Ours)

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

SparsityLev.1SparsityLev.2SparsityLev.3

Dense

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

“A brown teddy bear sitting in the middle of the road.”

- Figure A15. Qualitative comparison of images generated by models compressed to various sparsity level using ShortGPT, LaCo, FinerCut, and Skrr, alongside images generated by dense models. Skrr demonstrates a remarkable ability to produce images closely resembling those from the dense model in all the sparsity levels. Notably, the lighting and composition of objects in the images retains even after harsh pruning to the text encoder.

ShortGPT

LaCo FinerCut Skrr (Ours)

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

SparsityLev.1SparsityLev.2SparsityLev.3

Dense

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

“A domestic cat lounges on a quilt covering a bed.”

ShortGPT LaCo FinerCut Skrr (Ours)

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

SparsityLev.1SparsityLev.2SparsityLev.3

Dense

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

“A cat laying on a laptop and looking like it has glowing eyes.”

- Figure A16. Qualitative comparison of images generated by models compressed to various sparsity level using ShortGPT, LaCo, FinerCut, and Skrr, alongside images generated by dense models. Skrr demonstrates a remarkable ability to produce images closely resembling those from the dense model in all the sparsity levels. Notably, the lighting and composition of objects in the images retains even after harsh pruning to the text encoder.

