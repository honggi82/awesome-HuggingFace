# arXiv:2411.17686v4[cs.CV]15Nov2025

### Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration

#### Yuhang Han1*, Xuyang Liu2*, Zihan Zhang3, Pengxiang Ding1, Junjie Chen2, Donglin Wang1, Honggang Chen2, Qingsen Yang4,5, Siteng Huang6†

1Westlake University, 2Sichuan University, 3Johns Hopkins University, 4Northwestern Polytechnical University, 5Shenzhen Research Institute of Northwestern Polytechnical University, 6Zhejiang University yuhangh984@gmail.com, siteng.huang@gmail.com

Video-LLaVA-7B

###### Abstract

[Figure 1]

The quadratic complexity of Multimodal Large Language Models (MLLMs) with respect to context length poses significant computational and memory challenges, hindering their real-world deployment. In the paper, we devise a “filtercorrelate-compress” framework to accelerate the MLLM by systematically optimizing multimodal context length during prefilling. The framework first implements FiCoCo-V, a training-free method operating within the vision encoder. It employs a redundancy-based token discard mechanism that uses a novel integrated metric to accurately filter out redundant visual tokens. To mitigate information loss, the framework introduces a correlation-based information recycling mechanism that allows preserved tokens to selectively recycle information from correlated discarded tokens with a selfpreserving compression, thereby preventing the dilution of their own core content. The framework’s FiCoCo-L variant further leverages task-aware textual priors to perform token reduction directly within the LLM decoder. Extensive experiments demonstrate that the FiCoCo series effectively accelerates a range of MLLMs, achieves up to 14.7× FLOPs reduction with 93.6% performance retention. Our methods consistently outperform state-of-the-art training-free approaches, showcasing effectiveness and generalizability across model architectures, sizes, and tasks without requiring retraining.

LLaVA-1.5-13B Qwen2-VL-7B

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

LLaVA-1.5-7B LLaVA-NeXT-7B

Figure 1: The comparison to existing token reduction methods. Our FiCoCo series achieves state-of-the-art results with five popular MLLMs across benchmarks.

VL (Wang et al. 2024) in high-resolution visual question answering significantly outweighs its decoding time, constituting up to a remarkable 80% of the total latency. In this paper, we introduce “filter-correlate-compress”, a framework that systematically and progressively optimizes the length of multimodal context during prefilling. This enables MLLMs to minimize response latency while concurrently striving for the preservation of generation quality.

Code — https://github.com/kawhiiiileo/FiCoCo

#### 1 Introduction

Concurrently, we propose FiCoCo-V, a training-free method that represents the framework’s implementation within the vision encoder, primarily addressing data redundancy. Natural vision signals, such as images and videos, inherently possess a higher degree of information redundancy compared to human-generated languages (He et al. 2022; Feichtenhofer et al. 2022). However, in the constructed multimodal context, the number of visual tokens substantially exceeds that of textual tokens. Our framework thus initiates with a redundancy-based token discard, which reduces context length by measuring and filtering out redundant visual tokens at each layer. Specifically, unlike a potentially biased, single redundancy metric (Chen et al. 2024a; Zhang et al. 2024), FiCoCo-V integrates visionaware and semantic-aware redundancy to accurately discard those more redundant tokens.

Multimodal Large Language Models (MLLMs) (Liu et al. 2023, 2024a; Zhang, Li, and Bing 2023; Chen et al. 2024b; Lin et al. 2024b) have effectively extended the impressive emergent capabilities of Large Language Model (LLM) (Touvron et al. 2023; OpenAI 2023; Bai et al. 2023a) decoders by integrating visual features with textual inputs. However, the substantial increase in context length when processing images and videos imposes quadratically scaling computational and memory demands. This, in turn, renders prefilling a critical bottleneck for MLLM response generation. As an empirical instance, the prefilling time for Qwen2-

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

* Equal contribution. † Corresponding author.

One common oversight is that tokens considered redundant may contain noise or still hold information beneficial to the task (Liang et al. 2022). Developing effective mechanisms to flexibly recover such information facilitates performance maintenance of the MLLM. However, preserved tokens must carefully select which information to receive from discarded ones, preventing excessive dilution of their core information. We highlight that inter-token correlation provides a principled metric to guide where and how information should be recycled. Moving forward, our framework designs a correlation-based information recycling mechanism that allows each redundant token to correlate a variable number of preserved tokens to adaptively retain its information, while the FiCoCo-V method models such correlation with direct attention. Subsequently, a self-preserving compression operation ensures the prominence of the preserved tokens while allowing them to receive more information from highly correlated redundant tokens.

While FiCoCo-V strikes an appealing balance between efficiency and performance, executing token discard and information recycling in a task-agnostic manner can be shortsighted, as it fundamentally constrains the performance ceiling of the acceleration method. Consequently, we further propose FiCoCo-L, which performs token reduction directly within the LLM decoder. By leveraging task-aware textual priors, FiCoCo-L more precisely pinpoints redundant tokens and recovers crucial information, thereby minimizing loss during the compression process. In Figure 1, when applied to LLaVA-1.5-7B (Liu et al. 2024a), both methods consistently outperform existing token reduction baselines across different FLOPs. In the most extreme case, our method can obtain a maximum improvement of 5.7× in FLOPs while retaining 92.8% performance. When applied to the more powerful LLaVA-NeXT-7B (Liu et al. 2024b), our methods even show stronger superiority, achieving a 14.7× improvement in FLOPs while retaining at most 93.6% performance. We also evaluate our methods on video understanding tasks, where our methods retain at most 92.8% performance of vanilla Video-LLaVA (Lin et al. 2024a) with a 11.4× improvement in FLOPs. As a conclusion, our success in token budget reduction and model acceleration can generalize across various MLLM architectures, sizes, and tasks.

#### 2 Related Work

Multimodal Large Language Models (MLLMs). To acquire visual comprehension and reasoning capabilities, MLLMs (Dai et al. 2023; Bai et al. 2023b; Liu et al. 2023; Chen et al. 2024b) first use a pre-trained vision encoder (e.g., from CLIP (Radford et al. 2021)) to extract visual features, which are then projected into the input embedding space of a pre-trained Large Language Model (LLM) (Touvron et al. 2023; OpenAI 2023; Bai et al. 2023a) decoder. The LLM then processes these visual embeddings alongside user instructions to understand the images and craft suitable responses. A key trend in the development of MLLMs is to leverage longer multimodal contexts to capture finer-grained visual details, thereby enabling a more profound comprehension of the visual content. For example, LLaVA-1.5 (Liu et al. 2024a) improves the vision encoder for higher res-

olutions, while LLaVA-NeXT (Liu et al. 2024b) quadruples input resolution with flexible aspect ratios to enhance fine-grained understanding. And Video-LLaVA (Zhang, Li, and Bing 2023) employs extended context windows and dynamic frame aggregation to accommodate longer input sequences for video-text tasks. However, increased context length introduces significant inference latency and storage overheads, which become major deployment bottlenecks.

Token Reduction for MLLMs. Token reduction approaches can be broadly unified as token compression, which aims to eliminate redundancy and condense visual information into a more compact representation while minimizing information loss (Rao et al. 2021; Liang et al. 2022; Bolya et al. 2023; Liu et al. 2025d,c,b,a). Our proposed methods adaptively adjust the number of tokens each discarded token is compressed into. This functions as an automatic, per-token switching mechanism between the two techniques, designed to maximize benefits.

Token reduction for MLLMs has gradually shifted from training-based methods (Cha et al. 2024; Li et al. 2024) to training-free approaches (Chen et al. 2024a; Zhang et al. 2024), as the latter enables direct application to off-the-shelf models without costly retraining overheads. For instance, FastV (Chen et al. 2024a) prunes unnecessary visual tokens based on the ranking of attention scores derived from the self-attention mechanism in the LLM. SparseVLM (Zhang

- et al. 2024) adaptively prunes visual tokens in the LLM based on their attention scores with text tokens. PDrop (Xing
- et al. 2025) drops visual tokens according to the attention between all the visual tokens and the last token of the instruction. In this study, our FiCoCo shows that more precise identification of redundant tokens and controlled recovery of discarded information can achieve superior performance while maintaining high efficiency.

#### 3 Methodology of FiCoCo-V

##### 3.1 Preliminaries: Revisiting MLLMs

Prefilling. Given the textual instruction, a MLLM generates responses according to the input image, where the critical prefilling phase involves two key steps: (1) Input tokenization, where the visual encoder extracts the visual features and projects them into a shared latent space with discrete textual tokens. (2) Causal self-attention computation, wherein the LLM decoder performs causal self-attention over this entire concatenated sequence to establish contextual dependencies, providing intermediate key-value pairs for efficient autoregressive decoding. This phase establishes the decoding context for subsequent token generation, directly impacting inference latency and throughput.

Self-Attention. The self-attention mechanism (Vaswani et al. 2017) stands as both the most essential and the most resource-intensive operation in transformer-based visual encoder and LLM decoder. Given the input 1D sequence X of length N, the self-attention layer produces a selfattention map A ∈ RN×N to globally model the dependence relationships between tokens, formulated as A =

√

Attention(Q,K) = Softmax QK⊤/

D , where ⊤ denotes the transpose of the matrix, the query and key matrices

###### Redundancy-based Token Discard

-

Correlation-based Information Recycling

-

[Figure 6]

[Figure 7]

(1) filter stage

(1) filter stage

(2) correlate stage (3) compress stage

(2) correlate stage (3) compress stage

vision-aware redundancy

vision-aware redundancy

Vision EncoderLayer

semantic-aware redundancy

-aware redundancy

×

×

w/ token-adaptive

w/ token-adaptive

###### √

###### √

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

FiCoCo-V

FiCoCo-V

×

×

threshold

threshold

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

×

×

[Figure 20]

[Figure 21]

direct

## +

## +

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

correlation

correlation

[Figure 30]

[Figure 31]

[CLS]

[Figure 32]

[Figure 33]

×

×

×

×

[Figure 34]

[Figure 35]

| |√√| | |
|---|---|---|---|
| |√√| | |

×

×

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

+local penalty

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

token-adaptive correlation self-preserving compression

token-adaptive correlation self-preserving compression

(a) FiCoCo-V in Vision Encoder

FiCoCo-V

[Figure 50]

[Figure 51]

Text

Text

(1) filter stage (2) correlate stage (3) compress stage

(1) filter stage (2) correlate stage (3) compress stage

×

×

LLM DecoderLayer

vision-aware redundancy

vision-aware redundancy

task-aware

-aware

[Figure 52]

[Figure 53]

w/ token-adaptive threshold

w/ token-adaptive threshold

√

√

×

×

FiCoCo-L

FiCoCo-L

redundancy

redundancy

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

[Figure 69]

indirect correlation

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

## +

## +

[Figure 74]

[Figure 75]

correlation

Text

Text

[Figure 76]

[Figure 77]

× Text ×

× Text ×

|√√| |√√| |
|---|---|---|---|
| |√√| |√√|

[Figure 78]

[Figure 79]

√

√

×

×

[Figure 80]

[Figure 81]

×

×

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

×

×

token-adaptive correlation

token-adaptive correlation

self-preserving compression

self-preserving compression

(b) FiCoCo-L in LLM Decoder

FiCoCo-L

- Figure 2: Overview of FiCoCo-V and FiCoCo-L. Due to the two methods being applied to different modules (vision encoder and LLM decoder), they have different implementations for summarized redundancy and correlation matrix in the filter and correlate stages. Simultaneously, the compression modules of FiCoCo-V and FiCoCo-L are identical, both employing selfpreserving compression based on the correlation matrix.

Q,K ∈ RN×D are obtained by projecting X with learnable parameter matrices.

##### 3.2 Redundancy-based Token Discard

Filter: What token should be discarded? When evaluating the redundancy of tokens within the vision encoder, we draw inspiration from the two natural principles of humans when quickly and comprehensively summarizing the content of a given image. Firstly, to accelerate the recognition, we tend to ignore those similar pixels as they commonly provide the same information. Similarly, within a self-attention layer, if a visual token requires substantial information from other visual tokens, it indicates that its own information is not unique, and the token can be replaced by other visual tokens. Formally, given the self-attention weight matrix Av ∈ RN×N, where N is the number of the visual tokens, we can define the vision-aware redundancy of the i-th to-

ken by averaging its received attention, i.e., N1 Nj=1 Avi,j. We emphasize that this design is significantly different from previous methods (Chen et al. 2024a), as they regard attention between visual tokens as a measure of importance, and provide analysis in Appendix.

Secondly, if provided with the overall concept of the image, we rapidly identify the area of interest based on the global semantic clue and ignore other regions. As typical vision encoders (Dosovitskiy et al. 2021; Radford et al. 2021) employ a [CLS] token to capture the global image representation, its attention weights aCLS can quantify the global semantic content of patch tokens, which can be useful for multimodal understanding. And we can define the semanticaware redundancy by applying a negation operation. We regard this general solution as the default due to its efficiency, and provide an alternative solution for a limited number of MLLMs without a [CLS] token (e.g., SigLIP (Zhai

et al. 2023)). Specifically, we average the keys of all visual tokens as an alternative of the [CLS] token, and regard its cosine similarity with visual tokens as a substitute for attention. Details and experiments of this alternative can be found in Appendix Table 9. Therefore, we calculate the overall redundancy score for each visual token as

N

1 N

Avi,j − (1 − λ)aCLSi , (1)

svi = λ

j=1

where λ is a scalar hyperparameter that balances the factors. Since the visual tokens with higher redundancy scores are expected to be discarded, we filter out these tokens through a topK operation on the ranked scores, where the amount is related to the degree of reduction.

A concern is that tokens discarded in a layer might concentrate in a specific image area, potentially leading to spatial-centralized information loss. Therefore, we develop a local penalty strategy that encourages a more uniform spatial distribution of discarded tokens. Specifically, we represent the redundancy scoring vector s back to a 2D grid and partition it into non-overlapping windows of size W, using padding for previously discarded tokens to maintain the 2D information. Finally, we multiply the highest score within each window by a scaling coefficient, enhancing positive scores and diminishing negative ones. This effectively suppresses the global prominence of other large scores within the windows, reducing their likelihood of being discarded. As observed in the ablation study, this technique significantly enhances FiCoCo-V.

##### 3.3 Correlation-based Information Recycling

Correlate: Where should discarded information be recycled? We conduct a matrix that evaluates the correlation

1.5

###### Per-layer Pairwise Token Correlation Gap

Samples

Mean

1.0

0.5

0.0

1 4 7 10 13 16 19 22

- Figure 3: Layer-wise distribution of Pairwise Token Correlation Gap. We observe that the difference between the top-1 and top-2 correlation scores for each discarded token displays a high variance, highlighting the inadequacy of a fixed number of correlated preserved tokens.

between each discarded token and all the preserved visual tokens. Formally, given NS discarded tokens, the matrix can be defined as C ∈ RN

S×(N−NS). For FiCoCo-V in the vision encoder, attention weights inherently represent both the inter-token relationships and the flow of information, making them a measure of direct correlation. Therefore, the correlation matrix can be conducted as Cvi,j = Avi,j.

To select the preserved tokens that receive information from each discarded token, a topK operation can be applied on each row of the correlation matrix C. Here, K is the number of the selected preserved tokens, where K = 0 is equivalent to token pruning and K > 0 is token merging (e.g., K = 1 for ToMe (Bolya et al. 2023)). To find an appropriate K, we measure the Pairwise Token Correlation Gap—the difference between the top-1 and top-2 correlation scores—for each discarded token. Our FiCoCo-V experiment computes this gap by discarding 8 tokens per layer across a 24-layer ViT. In Figure 3, we observe that the distribution of correlation gaps varies significantly. A few layers show a concentration of large gaps, indicating that discarded tokens can easily identify their single most correlated preserved token. In contrast, some layers have gaps concentrated near zero, suggesting each discarded token has multiple candidate preserved tokens with similarly high correlation values. And more layers display distributions with high variance, highlighting the inadequacy of a fixed K value.

According to the above analysis, we devise a tokenadaptive K. Specifically, for the i-th discarded token, we compute the ε-th quantiles of the i-th row in the correlation matrix to determine a token-wise threshold τi. Then this threshold is re-applied to the row to identify the target tokens correlated to the i-th discarded token. In other word, for the j-th preserved token, if Ci,j ≥ τi, then this preserved token can be viewed as a correlated token for the i-th discarded token. And the number of correlated tokens for each discarded token is dynamic and adaptive. Therefore, we actually construct “dense” information pathways, where the correlation matrix facilitates the tracking of the information propagation from each discarded token to the candidate tokens. In contrast, a “convergent” correlation prompts all discarded tokens to merge into an additional token (Liang et al. 2022). Compared to that strategy, our “dense” correlations spread discarded information more widely among the remaining tokens and empirically demonstrate better performance.

Compress: How to effectively recycle information? After the correlate stage, each preserved token has a variable number of discarded tokens for updating itself. A straightforward update strategy involves averaging each preserved token with all discarded tokens that correlated to it (Bolya et al. 2023). However, as the number of discarded tokens increases, this strategy results in the preserved token having less information about itself after updates. And excessive integration of information from discarded tokens into preserved tokens leads to performance degradation through progressive dilution of their original semantic content. Therefore, our compression strategy must ensure the dominance of the preserved tokens. Moreover, naive averaging results in the amount of information received by a preserved token being independent of its correlation to the discarded tokens.

According to the above discussion, we update the preserved tokens with a self-preserving compression. Formally, we define the discarded tokens as a source set S, and the preserved visual tokens as a target set T. Therefore, given the correlation matrix C, we formulate the compression as

XTj +

αijXSi 1 +

i∈Ij

XTj ←

,where Ij = {i ∈ S and Ci,j ≥ τi},

αij

i∈Ij

Ci,j

,where Ji = {j ∈ T and Ci,j ≥ τi},

αij =

Ci,j

j∈Ji

(2) where the weight αij quantifies the proportion of information from the i-th discarded token that is allocated to the j-th correlated token. The strategy guarantees each preserved token preserves at least 50% of its original information. Moreover, the preserved token can receive more information from a discarded token with a strong correlation.

#### 4 Task-Aware Improvements for FiCoCo-L

Despite its promising performance, FiCoCo-V identifies and removes visual tokens based solely on the visual content. Applied within the vision encoder, such a task-agnostic method fails to preserve the essential visual information based on task context. Therefore, we provide a task-aware solution FiCoCo-L applied in the LLM decoder, leveraging textual priors to reduce visual token redundancy and recycle task-related visual information. Specifically, FiCoCo-L updates the redundancy calculation in the filter stage, and the correlation calculation in the correlate stage.

Task-aware redundancy calculation. In the LLM decoder, since textual tokens directly encode task instructions, the attention weights that visual tokens received from textual tokens indicate their task relevance. Therefore, we can cal-

culate a task-aware redundancy as - M1 Nk=+NM+1 Ali,k, where M denotes the number of textual tokens. As a result, the overall redundancy for FiCoCo-L is summarized as

N

N+M

1 N

1 M

sli = β

Ali,j − (1 − β)

Ali,k, (3)

j=1

k=N+1

where the scalar hyperparameter β balances the factors.

Method Source TFLOPs↓ SQA VQAT POPE GQA MMB VQAv2 Avg Avg(%) TFLOPs=8.5

LLaVA-1.5-7B NeurIPS23 8.5 69.5 58.2 86.4 62.5 66.1 79.1 70.3 100 TFLOPs=3.3(↓61.2%)

ToMe ICLR23 3.3 65.2 52.1 72.4 54.3 60.5 68.0 62.1 88.3 FastV ECCV24 3.3 67.3 52.5 64.8 52.7 61.2 67.1 60.9 86.6 SparseVLM ICML25 3.3 69.1 56.1 83.6 57.6 62.5 75.6 67.4 95.9 PDrop CVPR25 3.3 68.8 56.1 82.3 57.1 63.2 75.1 67.1 95.4 PruMerge ICCV25 3.3 67.9 54.3 71.3 54.3 59.6 70.6 63.0 89.6 FiCoCo-V Ours 3.3 67.8 55.7 82.5 58.5 62.3 74.4 66.9 95.2 FiCoCo-L Ours 3.3 69.6 56.6 84.6 61.1 64.6 76.8 68.9 98.0

TFLOPs=2.4(↓71.8%)

ToMe ICLR23 2.5 59.6 49.1 62.8 52.4 53.3 63.0 56.7 80.7 FastV ECCV24 2.5 60.2 50.6 59.6 49.6 56.1 61.8 56.3 80.1 SparseVLM ICML25 2.5 67.1 54.9 80.5 56.0 60.0 73.8 65.4 93.0 PDrop CVPR25 2.5 68.3 55.1 82.3 56.0 61.1 72.9 65.9 93.8 PruMerge ICCV25 2.5 67.1 54.3 67.2 53.3 58.1 68.8 61.5 87.5 FiCoCo-V Ours 2.4 68.3 55.6 82.2 57.6 61.1 73.1 66.3 94.3 FiCoCo-L Ours 2.4 69.4 56.3 84.4 60.6 61.9 73.4 67.7 96.3

TFLOPs=1.5(↓82.4%)

ToMe ICLR23 1.6 50.0 45.3 52.5 48.6 43.7 57.1 49.5 70.4 FastV ECCV24 1.6 51.1 47.8 48.0 46.1 48.0 61.8 50.5 71.8 SparseVLM ICML25 1.5 62.2 51.8 75.1 52.4 56.2 68.2 61.0 86.8 PDrop CVPR25 1.6 68.6 45.9 55.9 41.9 33.3 69.2 52.5 74.6 PruMerge ICCV25 1.5 68.1 54.0 65.3 51.9 55.3 67.4 60.3 85.8 FiCoCo-V Ours 1.5 68.4 55.5 79.8 54.9 60.2 72.1 65.2 92.7 FiCoCo-L Ours 1.5 69.5 55.7 82.1 53.2 61.5 69.7 65.3 92.8

TFLOPs=24.9 LLaVA-1.5-13B NeurIPS23 24.9 71.4 61.3 86.2 63.4 68.0 80.0 71.7 100

TFLOPs=15.4(↓47.6%)

FastV ECCV24 15.4 57.0 56.0 79.3 57.7 57.9 - 61.6 85.9 SparseVLM ICML25 15.4 69.9 49.9 81.1 57.9 65.8 - 64.9 90.5 FiCoCo-V Ours 15.4 72.1 57.2 82.3 59.2 63.1 76.8 68.5 95.5 FiCoCo-L Ours 15.4 72.4 58.3 83.1 60.1 65.2 77.6 69.5 96.9

- Table 1: Comparison results on LLaVA-1.5-7B/13B. We evaluate FiCoCo variants under various computational budgets, compared to baselines. Only shared datasets across both model sizes are included here.

Empirically, we observe that the “local penalty” strategy slightly degrades the performance of FiCoCo-L. We believe the reason is that this strategy weakens the task prior when encouraging spatial-uniform preservation of visual information. Consequently, we remove the strategy in FiCoCo-L.

Task-aware correlation calculation. When calculating the correlation matrix for FiCoCo-L, we explore an additional form of indirect semantic correlation, which leverages textual tokens as a bridge. Specifically, when measuring the association between the i-th token and the j-th token, we sum the products of the attention weights from the i-th token to all textual tokens and from all textual tokens to the j-th token. If the peak attention weights of the i-th token and the j-th token are concentrated on the same textual tokens, then the computed correlation between them is higher. In summary, we have

N+M

1 M

Ali,k · Alk,j, (4)

Cli,j = γAli,j + (1 − γ)

k=N+1

where γ is the scalar hyperparameter for factor balance.

To facilitate a clearer understanding of the proposed methods, we provide a comprehensive and detailed explanation of our FiCoCo-V and FiCoCo-L processes in Appendix. We also provide a theoretical estimation of the computing cost in Appendix. Note that for clarity, our formula calculations are designed to target individual elements within vectors or

matrices. However, these operations can be efficiently tensorized in the practical implementation to facilitate batched inference. And the implementation can be plug and play with less than 10 lines of additional code.

#### 5 Experiments

##### 5.1 Experimental Setups

We evaluate FiCoCo on multiple MLLMs: LLaVA-1.5 (Liu et al. 2024a), LLaVA-NeXT (Liu et al. 2024b), and Qwen2VL (Wang et al. 2024) for image understanding, and VideoLLaVA (Lin et al. 2024a) for video understanding. FiCoCo is benchmarked against mainstream token reduction methods: ToMe (Bolya et al. 2023), FastV (Chen et al.

- 2024a), SparseVLM (Zhang et al. 2024), PDrop (Xing et al.
- 2025), and PruMerge (Shang et al. 2025). Configurations and benchmark details are provided in Appendix.

##### 5.2 Main Comparisons

Results on LLaVA-1.5-7B/13B. Table 1 presents the performance of FiCoCo across benchmarks based on LLaVA1.5-7B/13B. The LLaVA-1.5-7B results yield two key findings: (1) Both FiCoCo-V and FiCoCo-L consistently outperform existing training-free methods across different computational budgets. Under extreme compression (TFLOPs = 1.5, 10% visual tokens), both variants achieve >92% average accuracy, surpassing the second-best SparseVLM

Method MMB SQA VQAT MMMU Avg Avg (%) TFLOPs=42.7

LLaVA-NeXT-7B 67.9 70.2 61.3 35.1 58.6 100.0 TFLOPs=5.0(↓88.3%) PDrop 63.4 67.5 54.4 29.8 53.8 91.7 TFLOPs=2.9(↓93.2%)

SparseVLM 63.1 67.5 46.3 32.8 52.4 89.4 FiCoCo-V 60.5 68.1 55.3 34.1 54.5 93.0 FiCoCo-L 63.6 67.9 53.1 34.8 54.9 93.6

###### Table 2: Comparison with LLaVA-NeXT-7B on crossimage understanding benchmarks.

Token Method MMB POPE VQAT Avg Avg (%) Base. ≈1300

Qwen2-VL-7B 80.5 86.4 84.3 83.7 100.00 600

SparseVLM 79.6 86.5 80.3 82.1 98.09 FiCoCo-V 79.9 86.5 81.2 82.5 98.57 FiCoCo-L 80.1 86.3 81.4 82.6 98.69

SparseVLM 78.8 86.3 79.0 81.4 97.25 FiCoCo-V 79.1 86.6 79.8 81.8 97.73 FiCoCo-L 78.9 86.1 79.7 81.6 97.49

500

SparseVLM 79.0 85.8 77.1 80.6 96.29 FiCoCo-V 79.1 86.0 78.3 81.1 96.89 FiCoCo-L 78.6 85.9 78.3 80.9 96.66

400

###### Table 3: Comparison with Qwen2-VL-7B under different token budgets. FiCoCo is compared against SparseVLM under 600/500/400 tokens on MMB, POPE, and VQAT.

by approximately 6%, demonstrating the effectiveness of our information recovery mechanism. (2) FiCoCo-L outperforms FiCoCo-V when computational budgets are generous, as it captures task-relevant visual tokens and maximally focuses on question-related regions. However, under constrained budgets, both variants achieve similar performance (with an average difference of only 0.1%). This convergence arises because, under severe token constraints, the distinction between preserving visual saliency (FiCoCo-V) and task-specific relevance (FiCoCo-L) diminishes—both are compelled to retain only the most essential visual elements for maintaining core model functionality. Moreover, under the LLaVA-1.5-13B setting, FiCoCo-V/L achieves superior performance with only ∼22% visual tokens (15.4 TFLOPs), outperforming the strongest baseline SparseVLM by 5.0% and 6.4% in average accuracy. Additional results in Tables 8 further validate this advantage.

Results on LLaVA-NeXT-7B. We impose two computational constraints: TFLOPs are set to 5.0 for PDrop to match reported results, and 2.9 for SparseVLM and FiCoCo. Table 2 shows that, under TFLOPs = 2.9, FiCoCoV and FiCoCo-L outperform SparseVLM by 3.6% and

- 4.2%, respectively. Moreover, despite operating under a lower TFLOPs budget than PDrop, our methods FiCoCoV and FiCoCo-L consistently outperform PDrop by 1.9% in terms of average accuracy, highlighting their superior efficiency and robustness in handling dense visual token scenarios within resource-constrained settings. Results on Qwen2-VL. Following SparseVLM’s settings on Qwen2-VL, we compress about 54.5% of visual tokens. Since Qwen2-VL lacks a [CLS] token, an equivalent token averaging scheme is used in FiCoCo-V. As shown in Table 3,

Method TGIF MSVD MSRVTT ActivityNet Avg Avg(%) TFLOPs=29.7

Video-LLaVA 47.1 69.8 56.7 43.1 54.2 100.0 TFLOPs=2.6(↓91.2%)

FastV 23.1 38.0 19.3 30.6 27.8 52.1 SparseVLM 44.7 68.2 31.0 42.6 46.9 86.5 FiCoCo-V 43.1 67.4 47.8 42.8 50.3 92.8 FiCoCo-L 44.3 64.5 49.2 40.1 49.5 91.4

Table 4: Comparison results on video understanding benchmarks with Video-LLaVA.

|Stage<br><br>|Method SQA TextVQA FiCoCo-V 68.37 55.46|
|---|---|
|Filter|w/o vision-aware redundancy 67.81 52.51 w/o semantic-aware redundancy 64.67 48.74 w/o local penalty 68.12 53.24<br><br>|
|Correlate<br><br>|fixed K=0 67.82 53.56<br>fixed K=1 67.43 46.97<br>fixed K=2 67.21 51.36 convergent correlation 67.60 54.38<br>|
|Compress<br><br>|average compression 67.92 53.34|

Table 5: Ablation results of FiCoCo-V.

under this ratio, FiCoCo maintains over 98% accuracy and surpasses SparseVLM. Further, compressing each additional 100 tokens yields only a 0.8% drop, highlighting FiCoCo’s robustness and the effectiveness of the averaging scheme.

Results on Video-LLaVA. For fair evaluation, the number of video tokens is limited to 136 (about 6.6% of all visual tokens). As shown in Table 4, both FiCoCo-V and FiCoCoL reach over 90% of Video-LLaVA’s performance, while FiCoCo-V surpasses FastV and SparseVLM by 40.7% and 6.3%, respectively, demonstrating its superiority in video understanding. The stronger performance of FiCoCo-V over FiCoCo-L likely stems from higher visual redundancy in videos, where compression aids attention to salient content.

##### 5.3 Ablation Study

To further validate the effectiveness of our design at each stage, we conduct extensive ablation studies on the SQA and TextVQA benchmarks under a fixed computational budget of 1.5 TFLOPs. In Table 5, we ablate all three stages of FiCoCo-V to analyze their individual contributions.

- • Filter. Both vision-aware and semantic-aware redundancy improve the identification of discarded tokens. Notably, semantic-aware redundancy has a more significant impact on the final performance. This indicates that token reduction within the vision encoder should prioritize the retention of tokens rich in global semantic information. Additionally, we observe that by promoting a spatially uniform distribution of discarded tokens, the local penalty strategy aids in preserving visual information.
- • Correlate. We evaluate fixed K values of 0 (pruning, i.e., no correlation-based recycling), 1 (single-token recycling), and 2 (multi-token recycling). While the token-adaptive K strategy performs best, an intriguing result is that K=0 surpasses the other two. This likely occurs because small fixed K values limit information sources for correlated tokens,

|Stage|Method SQA TextVQA<br><br>FiCoCo-L 69.46 55.72<br><br>|
|---|---|
|Filter|w/o vision-aware redundancy 69.16 55.43 w/o task-aware redundancy 68.22 55.64 w/ local penalty 68.79 55.38<br><br>|
|Correlate|w/o indirect correlation 68.89 54.78 w/o direct correlation 68.45 55.45<br><br>fixed K=0 68.96 50.33<br><br>fixed K=1 68.57 50.11<br><br>fixed K=2 68.32 50.18<br><br><br>convergent correlation 67.80 54.89<br><br>|
|Compress<br><br>|average compression 68.32 54.66|

###### Table 6: Ablation results of FiCoCo-L.

causing over-dilution and noise. Thus, pruning yields better performance. Moreover, our “dense” correlation outperforms the “convergent” variant, which compresses discarded tokens into one; retrieving information while preserving token integrity proves more effective.

- • Compress. Our self-preserving compression outperforms directly averaging the features, indicating that the calculated weights can effectively regulate the contribution of information sources in the updates of correlated tokens.

In Table 6, we ablate all three stages for FiCoCo-L:

- • Filter. Although both vision-aware and task-aware redundancies contribute to redundancy estimation, neither dominates. This may be because the attention mechanism in LLMs captures stable token dependencies, reducing the need for redundancy measurement to rely heavily on semantic cues. Moreover, applying the local penalty strategy in FiCoCo-L slightly degrades performance, likely because enforcing spatial uniformity of token retention disrupts the redundancy assessments already established by attention.
- • Correlate. It is observed that both correlations enhance the identification of relevant tokens, improving performance across datasets. Similar to FiCoCo-V, adopting a tokenadaptive K with “dense” correlations proves optimal.
- • Compress. Updating the preserved tokens with a selfpreserving compression still achieves better performance.

##### 5.4 Qualitative Analysis

We visualize the discarded tokens of FiCoCo-V (Figure 4a) and FiCoCo-L (Figure 4b) under varying compression levels across VQA scenarios. Tokens highly relevant to answers are highlighted with red boxes to assess information preservation. A token linked to “2” is traced in Figure 4a, and one linked to “GAMES” in Figure 4b. In both cases, as compression increases (TFLOPs from 4.2 to 1.5), more tokens—including key ones—are discarded, reducing critical information. We trace information recycling from discarded tokens (red arrows) and highlight correlated tokens (green boxes), where transparency reflects retained information. These correlated tokens aggregate essential cues and support question answering. Notably, discarded information is distributed across multiple correlated tokens, enhancing comprehension of salient regions (Figure 4b), qualitatively validating our method.

||[Figure 92]|[Figure 93]|[Figure 94]|
|---|---|---|
<br><br>|[Figure 95]<br><br>|[Figure 96]<br><br>|[Figure 97]|
|---|---|---|
<br><br>Q: “What event is this from?”FiCoCo-L: “millrose games.”<br><br>(a)<br>(b)<br><br><br>Q: “What number is on the player's jersey?”FiCoCo-V: “22.”<br><br>TFLOPs = 8.5 TFLOPs = 4.2 TFLOPs = 1.5|
|---|

###### Figure 4: Visualizations of token discard and information recycling by (a) FiCoCo-V and (b) FiCoCo-L. Red: traced patch token; Green: recycling destination.

LLaVA-NeXT-7B LLaVA-1.5-7B

Method

TFLOPs Throughput TFLOPs Throughput Vanilla 42.7 3.8 8.5 8.99

FiCoCo-V 2.9 (↓ 93.2%) 7.9 (↑ 107.9%) 1.5 (↓ 82.4%) 12.9 (↑ 43.5%) FiCoCo-L 2.9 (↓ 93.2%) 6.5 (↑ 71.1%) 1.5 (↓ 82.4%) 11.6 (↑ 29.1%)

Table 7: Efficiency analysis of FiCoCo. Lower TFLOPs and higher throughput (img/s) indicate better efficiency.

##### 5.5 Efficiency Analysis

As shown in Table 7, we present the trends of throughput and TFLOPs changes after applying FiCoCo in the LLaVANeXT and LLaVA-1.5 architectures. Introducing FiCoCo into LLaVA-NeXT reduces TFLOPs by 93.2%, increasing throughput by 2.08× (FiCoCo-V) and 1.71× (FiCoCo-L). In LLaVA, a TFLOPs reduction of 82.4% yields throughput gains of 1.43× and 1.29×, respectively. These results highlight FiCoCo’s ability to substantially lower computational overhead while enhancing throughput.

#### 6 Conclusion

In this paper, we propose a “filter-correlate-compress” acceleration framework to systematically eliminate visual redundancy in MLLMs through a principled three-stage pipeline. The filter stage performs dynamic redundancybased token discard using variation-aware thresholds, while the correlate stage identifies semantic relationships between tokens and the compress stage jointly enables correlationbased information recycling, thereby significantly reducing computational complexity while preserving critical multimodal information. The effectiveness of our framework is rigorously demonstrated through specialized variants for vision encoders (FiCoCo-V) and LLM decoders (FiCoCoL), achieving consistent acceleration benefits across diverse MLLM architectures for both image and video understanding tasks with minimal accuracy degradation.

#### 7 Acknowledgments

This work was supported in part by the National Natural Science Foundation of China (No. 62301432), the Natural Science Basic Research Program of Shaanxi Province (Grant No. QCYRCXM-2023-057), the Fundamental Research Funds for the Central Universities, and the Guangdong Basic and Applied Basic Research Foundation (Grant No. 2025A1515011119). It was also supported by the Chengdu Science and Technology Program (Grant No. 2025-YF1200006-RC), the Police Integration Computing Key Laboratory of Sichuan Province (Grant No. JWRH202502002), and the Open Fund of the Key Laboratory of the Ministry of Education on Artificial Intelligence in Equipment (Grant No. 2024-AAIE-KF04-03). This work was supported by the National Science and Technology Innovation 2030 - Major Project (Grant No. 2022ZD0208800), and NSFC General Program (Grant No. 62176215)

#### References

Bai, J.; Bai, S.; Chu, Y.; Cui, Z.; Dang, K.; Deng, X.; Fan, Y.; Ge, W.; Han, Y.; Huang, F.; Hui, B.; Ji, L.; Li, M.; Lin, J.; Lin, R.; Liu, D.; Liu, G.; Lu, C.; Lu, K.; Ma, J.; Men, R.; Ren, X.; Ren, X.; Tan, C.; Tan, S.; Tu, J.; Wang, P.; Wang, S.; Wang, W.; Wu, S.; Xu, B.; Xu, J.; Yang, A.; Yang, H.; Yang, J.; Yang, S.; Yao, Y.; Yu, B.; Yuan, H.; Yuan, Z.; Zhang, J.; Zhang, X.; Zhang, Y.; Zhang, Z.; Zhou, C.; Zhou, J.; Zhou, X.; and Zhu, T. 2023a. Qwen Technical Report. arXiv preprint arXiv:2309.16609.

Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023b. Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities. arXiv preprint arXiv:2308.12966.

Bolya, D.; Fu, C.; Dai, X.; Zhang, P.; Feichtenhofer, C.; and Hoffman, J. 2023. Token Merging: Your ViT But Faster. In Proceedings of the International Conference on Learning Representations.

Cha, J.; Kang, W.; Mun, J.; and Roh, B. 2024. Honeybee: Locality-Enhanced Projector for Multimodal LLM. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13817–13827.

Chen, L.; Zhao, H.; Liu, T.; Bai, S.; Lin, J.; Zhou, C.; and Chang, B. 2024a. An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large VisionLanguage Models. In Proceedings of the European Conference on Computer Vision, 19–35.

Chen, Z.; Wu, J.; Wang, W.; Su, W.; Chen, G.; Xing, S.; Zhong, M.; Zhang, Q.; Zhu, X.; Lu, L.; Li, B.; Luo, P.; Lu, T.; Qiao, Y.; and Dai, J. 2024b. InternVL: Scaling up Vision Foundation Models and Aligning for Generic VisualLinguistic Tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 24185– 24198.

Dai, W.; Li, J.; Li, D.; Tiong, A. M. H.; Zhao, J.; Wang,

- W.; Li, B.; Fung, P.; and Hoi, S. C. H. 2023. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. In Proceedings of the Advances in Neural Information Processing Systems, 49250–49267.

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; Uszkoreit, J.; and Houlsby, N. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In Proceedings of the International Conference on Learning Representations.

Feichtenhofer, C.; Fan, H.; Li, Y.; and He, K. 2022. Masked Autoencoders As Spatiotemporal Learners. In Proceedings of the Advances in Neural Information Processing Systems, 35946–35958.

Fu, C.; Chen, P.; Shen, Y.; Qin, Y.; Zhang, M.; Lin, X.; Qiu, Z.; Lin, W.; Yang, J.; Zheng, X.; Li, K.; Sun, X.; and Ji, R. 2023. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv preprint arXiv:2306.13394.

Gurari, D.; Li, Q.; Stangl, A. J.; Guo, A.; Lin, C.; Grauman,

- K.; Luo, J.; and Bigham, J. P. 2018. VizWiz Grand Challenge: Answering Visual Questions From Blind People. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3608–3617. He, K.; Chen, X.; Xie, S.; Li, Y.; Dollár, P.; and Girshick, R. B. 2022. Masked Autoencoders Are Scalable Vision Learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15979–15988. Hudson, D. A.; and Manning, C. D. 2019. GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6700–6709. Jang, Y.; Song, Y.; Yu, Y.; Kim, Y.; and Kim, G. 2017. TGIFQA: Toward Spatio-Temporal Reasoning in Visual Question Answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1359–1367. Li, W.; Yuan, Y.; Liu, J.; Tang, D.; Wang, S.; Zhu, J.; and Zhang, L. 2024. TokenPacker: Efficient Visual Projector for Multimodal LLM. arXiv preprint arXiv:2407.02392. Li, Y.; Du, Y.; Zhou, K.; Wang, J.; Zhao, W. X.; and Wen, J. 2023. Evaluating Object Hallucination in Large VisionLanguage Models. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 292– 305. Liang, Y.; Ge, C.; Tong, Z.; Song, Y.; Wang, J.; and Xie, P.

2022. Not All Patches are What You Need: Expediting Vision Transformers via Token Reorganizations. In Proceedings of the International Conference on Learning Representations.

Lin, B.; Ye, Y.; Zhu, B.; Cui, J.; Ning, M.; Jin, P.; and Yuan,

- L. 2024a. Video-LLaVA: Learning United Visual Representation by Alignment Before Projection. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 5971–5984. Lin, J.; Yin, H.; Ping, W.; Molchanov, P.; Shoeybi, M.; and Han, S. 2024b. VILA: On Pre-training for Visual Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26679–26689. Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2024a. Improved Baselines with Visual Instruction Tuning. In Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26286–26296.

Liu, H.; Li, C.; Li, Y.; Li, B.; Zhang, Y.; Shen, S.; and Lee, Y. J. 2024b. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual Instruction Tuning. In Proceedings of the Advances in Neural Information Processing Systems, 34892–34916.

Liu, X.; Gui, X.; Zhang, Y.; and Zhang, L. 2025a. Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models. arXiv preprint arXiv:2510.20707.

- Liu, X.; Wang, Y.; Ma, J.; and Zhang, L. 2025b. Video Compression Commander: Plug-and-Play Inference Acceleration for Video Large Language Models. arXiv preprint arXiv:2505.14454.
- Liu, X.; Wang, Z.; Han, Y.; Wang, Y.; Yuan, J.; Song, J.; Zheng, B.; Zhang, L.; Huang, S.; and Chen, H. 2025c. Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models. arXiv preprint arXiv:2501.05179.

- Liu, X.; Wen, Z.; Wang, S.; Chen, J.; Tao, Z.; Wang, Y.; Jin,

X.; Zou, C.; Wang, Y.; Liao, C.; et al. 2025d. Shifting ai efficiency from model-centric to data-centric compression. arXiv preprint arXiv:2505.19147.

- Liu, Y.; Duan, H.; Zhang, Y.; Li, B.; Zhang, S.; Zhao, W.; Yuan, Y.; Wang, J.; He, C.; Liu, Z.; Chen, K.; and Lin, D. 2024c. MMBench: Is Your Multi-modal Model an AllAround Player? In Proceedings of the European Conference on Computer Vision, 216–233.

Lu, P.; Mishra, S.; Xia, T.; Qiu, L.; Chang, K.; Zhu, S.; Tafjord, O.; Clark, P.; and Kalyan, A. 2022. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering. In Proceedings of the Advances in Neural Information Processing Systems, 2507–2521.

OpenAI. 2023. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the International Conference on Machine Learning, 8748–8763.

Rao, Y.; Zhao, W.; Liu, B.; Lu, J.; Zhou, J.; and Hsieh, C. 2021. DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification. In Ranzato, M.; Beygelzimer, A.; Dauphin, Y. N.; Liang, P.; and Vaughan, J. W., eds., Proceedings of the Advances in Neural Information Processing Systems, 13937–13949.

Shang, Y.; Cai, M.; Xu, B.; Lee, Y. J.; and Yan, Y. 2025. LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Singh, A.; Natarajan, V.; Shah, M.; Jiang, Y.; Chen, X.; Batra, D.; Parikh, D.; and Rohrbach, M. 2019. Towards VQA Models That Can Read. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, 8317–8326.

Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.; Lacroix, T.; Rozière, B.; Goyal, N.; Hambro, E.; Azhar, F.; Rodriguez, A.; Joulin, A.; Grave, E.; and Lample, G. 2023. LLaMA: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2302.13971.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L.; and Polosukhin, I. 2017. Attention is All you Need. In Proceedings of the Advances in Neural Information Processing Systems, 5998–6008.

Wang, P.; Bai, S.; Tan, S.; Wang, S.; Fan, Z.; Bai, J.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Fan, Y.; Dang, K.; Du, M.; Ren, X.; Men, R.; Liu, D.; Zhou, C.; Zhou, J.; and Lin,

- J. 2024. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. arXiv preprint arXiv:2409.12191.

Xing, L.; Huang, Q.; Dong, X.; Lu, J.; Zhang, P.; Zang, Y.; Cao, Y.; He, C.; Wang, J.; Wu, F.; and Lin, D. 2025. PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Xu, D.; Zhao, Z.; Xiao, J.; Wu, F.; Zhang, H.; He, X.; and Zhuang, Y. 2017. Video Question Answering via Gradually Refined Attention over Appearance and Motion. In Proceedings of the ACM International Conference on Multimedia, 1645–1653.

Yu, W.; Yang, Z.; Li, L.; Wang, J.; Lin, K.; Liu, Z.; Wang, X.; and Wang, L. 2024. MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities. In Proceedings of the International Conference on Machine Learning, 57730– 57754.

Yu, Z.; Xu, D.; Yu, J.; Yu, T.; Zhao, Z.; Zhuang, Y.; and Tao, D. 2019. ActivityNet-QA: A Dataset for Understanding Complex Web Videos via Question Answering. In Proceedings of the AAAI Conference on Artificial Intelligence, 9127–9134.

Yuan, Z.; Shang, Y.; Zhou, Y.; Dong, Z.; Zhou, Z.; Xue, C.; Wu, B.; Li, Z.; Gu, Q.; Lee, Y. J.; Yan, Y.; Chen, B.; Sun, G.; and Keutzer, K. 2024. LLM Inference Unveiled: Survey and Roofline Model Insights. arXiv preprint arXiv:2402.16363. Zhai, X.; Mustafa, B.; Kolesnikov, A.; and Beyer, L. 2023. Sigmoid Loss for Language Image Pre-Training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 11941–11952.

Zhang, H.; Li, X.; and Bing, L. 2023. Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 543–553.

Zhang, Y.; Fan, C.-K.; Ma, J.; Zheng, W.; Huang, T.; Cheng,

- K.; Gudovskiy, D.; Okuno, T.; Nakata, Y.; Keutzer, K.; and Zhang, S. 2024. SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference. arXiv preprint arXiv:2410.04417.

#### Appendix

In the appendix, we provide the theoretical calculation of FLOPs, detailed implementation information, extended experiments and analyses, and a comprehensive explanation of our proposed methods.

#### 8 Theoretical FLOPs Calculation

Here we consider a hypothetical scenario to analyze the changes in FLOPs before and after applying FiCoCo-V and FiCoCo-L. In this context, the hidden State dimension in a single transformer layer is denoted as D, while the feedforward layer dimension is represented by H. The total number of visual tokens is represented by N, with NS denoting the number of compressed visual tokens per layer.

Additionally, M represents the number of text tokens. To simplify the equations, we define:

###### N′ = N − NS, P = N + M, P′ = N′ + M.

Here, P represents the total number of visual and text tokens before compression, while P′ represents the total tokens after compression. Finally, for FiCoCo-V, we have:

FLOPsbefore = 4ND2 + 2N2D + 2NDH, FLOPsafter =4N′D2 + 2(N′)2D + 2N′DH, ∆ = 4NSD2 + 2 NNS − (NS)2 D + 2NSDH.

For FiCoCo-L, we have:

(5)

FLOPsbefore = 4PD2 + 2P2D + 2PDH, FLOPsafter =4P′D2 + 2(P′)2D + 2P′DH, ∆ = 4NSD2 + 2 2NNS − (NS)2 D + 2NSDH.

(6)

We now analyze the additional FLOPs introduced by the internal operations of FiCoCo-V and FiCoCo-L. As described in Sec. 3, the primary computational overhead for FiCoCo-V stems from the redundancy score calculation, the determination of token-adaptive K values, and the token updating process. In comparison, FiCoCo-L incorporates similar steps but introduces an additional interaction with the indirect text matrix during the correlate phase, resulting in a higher computational complexity. The variable NT represents the number of target tokens. However, since both FiCoCo-V and FiCoCo-L only operate on visual tokens, their FLOPs calculations are nearly identical. For FiCoCoV, we have:

FLOPs = N2 + 2N + NS(NT + 2D + 1) + D. (7) For FiCoCo-L, we have:

FLOPs = 2(N2 + 2N) + NS(NT + 2D + 1) + D. (8) Based on the above analysis, the additional FLOPs intro-

duced by FiCoCo-V and FiCoCo-L are negligible compared to the significant reduction in FLOPs ( ∆ ) achieved through token compression. Specifically, while ∆ grows quadratically with the hidden State dimension D, the additional FLOPs primarily grow linearly, making their impact inconsequential in practical scenarios.

#### 9 Implementation Details

Experimental Setup Details. For FiCoCo, we adopt the LLaVA-1.5-7B/13B models (Liu et al. 2024a) and employ the following settings: (1) λ = 0.35 in filter stage of FiCoCo-V, (2) β = 0.6 in filter stage of FiCoCo-L, (3) γ = 0.6 in correlate stage of FiCoCo-L, (4) scaling coefficient=2 in local penalty strategy, (5) ε = 0.998 to determine the token-wise threshold in compress stage. We provide sensitivity analyses of these hyperparameters in the latter half of this appendix. For the local penalty strategy, we fix a 2×2 window across all layers. Since the effectiveness of our FiCoCo is based on the reliability of attention mechanisms, we delay the token reduction until the attention converges to stability. Specifically, in FiCoCo-V, the token compression starts at the 12-th layer of the vision encoder, while in FiCoCo-L, it starts at the 4-th layer of the LLM. All experiments are conducted on a single A800 80GB GPU.

Backbone Details. We detail the involved backbones as follows.

- • LLaVA-1.5 (Liu et al. 2024a): An improved visionlanguage model built on the LLaVA framework that integrates high-resolution image processing and enhanced vision-language alignment. It employs CLIP-ViT-L/14 as the vision encoder and a Vicuna-based language backbone, with an optimized projection layer for visual token mapping. LLaVA-1.5 supports resolution up to 448×448 and achieves stronger accuracy across multiple benchmarks while maintaining efficient inference.
- • LLaVA-NeXT (Liu et al. 2024b): A strong visionlanguage baseline that adopts the AnyRes strategy to preserve fine-grained image details by quadrupling input resolution. This leads to a substantial increase in the number of visual tokens and significantly higher computational cost.
- • Qwen2-VL (Wang et al. 2024): A multimodal model that introduces a cross-attention-based vision-language adapter to compress image features into 256 tokens using learnable query embeddings. To preserve spatial details, 2D absolute positional encodings are incorporated into the attention mechanism. The compressed features are then fed into the Qwen language model for downstream tasks.
- • Video-LLaVA (Lin et al. 2024a): A video-based extension of LLaVA that utilizes Language-Bind as the visual encoder to process 8-frame clips, with 256 video tokens per frame. Following its original protocol, we adopt ChatGPT-based scoring as the main evaluation metric.

Benchmark Details. We assess the performance of FiCoCo across a comprehensive suite of multimodal understanding benchmarks, spanning both image-based and video-based reasoning tasks:

- • GQA (Hudson and Manning 2019): Comprises 113,018 real-world images annotated with scene graphs to support structured visual reasoning.
- • VizWiz (Gurari et al. 2018): Contains over 39,000 questions posed by blind users, accompanied by low-quality

- images and characterized by conversational query patterns.
- • SQA (Lu et al. 2022): Encompasses 21,000 scientific questions covering 26 distinct domains and a wide range of 379 reasoning skills.
- • VQA-T (Singh et al. 2019): Involves 28,000 highresolution images aimed at evaluating the understanding of textual content embedded within complex visual scenes.
- • POPE (Li et al. 2023): Offers 14,000 image-question pairs focusing on binary judgments for identifying object hallucinations.
- • MME (Fu et al. 2023): Provides a collection of 15,000 images encompassing 14 distinct subdomains of perceptual and cognitive reasoning.
- • MMB (Liu et al. 2024c): Presents 20,000 multiplechoice questions that assess robust visual reasoning across various categories.
- • MMB-CN (Liu et al. 2024c): Serves as the Chinese counterpart of MMB, consisting of 20,000 questions tailored for cross-lingual evaluation.
- • MM-Vet (Yu et al. 2024): Comprises 16,000 highresolution questions spanning factual knowledge, and sp

|visual recognition, OCR, spatial reasoning.<br><br>A collection of animated GIFs descriptions. It focuses on with fine-grained motion, video-to-text tasks such as cap-<br><br>The Microsoft Research consists of around 2,000 short<br><br>annotated with multiple humanmultilingual evaluation and captioning.<br><br>This dataset contains 10,000<br><br>categories, each accompaserves as a standard benchmark<br><br>understanding and text-video<br><br>Built on top of the Activuntrimmed videos annotated<br><br>natural language descriptions. video captioning and temporal<br><br>and Analysis<br><br>on LLaVA-1.5-7B<br><br>results further presents the on VizWiz (Gurari et al. 2018),<br><br>MMBCN (Liu et al. 2024c), and<br><br>|Quary<br><br>Key<br><br>Value<br><br>Hidden_states<br><br>[CLS]<br><br>Layer 1 Layer 8 Layer 16 Layer 24<br><br>[Figure 98]<br><br>| |
|---|
<br><br>Q: “What state is this car from?” A: “California.”<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]|
|---|
<br><br>Figure 5: Heatmap visualization using different inputs as the mean reference.<br><br>10.2 Disscussion about Evaluation without|
|---|

- • TGIF (Jang et al. 2017): A paired with natural language short, dynamic visual content and is widely used for vi tioning and retrieval.
- • MSVD (Xu et al. 2017): Video Description dataset YouTube videos, each an written captions. It supports is commonly used for video
- • MSRVTT (Xu et al. 2017): video clips from 20 diverse nied by 20 captions. It se for general-purpose video alignment tasks.
- • ActivityNet (Yu et al. 2019): ityNet dataset, it features with temporally localized It is designed for dense v event localization.

#### 10 More Experiments

##### 10.1 More Experiments

In Figure 8, more comparison performance of our method o MM-Vet (Yu et al. 2024), MM LLAVA-W(Liu et al. 2023). The results indicate that even with an 82.4% reduction in TFLOPs, both FiCoCo-V and FiCoCo-L maintain an average accuracy exceeding 90%, effectively preserving the capabilities of the MLLM.

Method TFLOPs↓ Vizwiz MM-VetMMBCNLLaVA-W TFLOPs = 8.5 LLaVA-1.5 8.5

50 31.6 59.3 63.7 100.0% 100.0% 100.0% 100.0% TFLOPs=3.3(↓61.2%)

51.5 29.7 55.3 60.4 103.0% 94.0% 93.3% 94.8% FiCoCo-L 3.3

FiCoCo-V 3.3

48.7 31.4 53.6 60.3 97.4% 99.4% 90.4% 94.7% TFLOPs=2.4(↓71.8%)

49.4 28.2 54.3 56.6 98.8% 89.2% 91.6% 88.9% FiCoCo-L 2.4

FiCoCo-V 2.4

48.4 30.1 53.5 59.4 96.8% 95.3% 90.2% 93.3% TFLOPs=1.5(↓82.4%)

52.4 26.8 53.0 58.6 104.8% 84.8% 89.4% 92.0% FiCoCo-L 1.5

FiCoCo-V 1.5

48.2 27.4 53.3 57.3 96.4% 86.7% 89.9% 90.0%

Table 8: Additional results of FiCoCo on LLaVA-1.5-7B.

##### [CLS] token

A key component of our FiCoCo-V’s semantic-aware redundancy metric relies on the [CLS] token, which is trained to aggregate global image representation. However, a growing

###### Method VQAT MMB POPE MM-Vet Vizwiz Avg (%)

TFLOPs=8.5

LLaVA-1.5 58.2 66.1 86.4 31.6 50.0 58.46

TFLOPs=1.5(↓82.4%)

aCLSi 55.5 60.2 79.8 26.8 52.4 54.94 aHi 54.2 59.6 81.4 25.9 49.8 54.18 aEqi (Quary) 52.0 57.8 79.6 25.1 49.9 52.89 aEqi (V alue) 54.3 61.4 81.0 25.4 50.8 54.59 aEqi (Key) 54.8 60.3 81.4 26.5 50.9 54.78

- Table 9: Comparison results across different benchmarks. The evaluation includes multiple datasets and varying FLOPs settings.

number of modern vision encoders, such as SigLIP (Zhai et al. 2023), have omitted this special token. This design choice poses a challenge for methods like ours that leverage it. To ensure the generalizability of FiCoCo-V, we explore the substitutability of the [CLS] token grounded in the mechanics of the self-attention mechanism.

Theoretical justification: Why the mean of Key vectors? To find the most suitable proxy for a global context vector, we revisit the fundamental roles of the Query (Q), Key (K), and Value (V) vectors within the self-attention mechanism (Vaswani et al. 2017).

- • Query (Q): A token’s Query vector acts as a “probe” or “request”, signifying the information it seeks from other tokens in the sequence. The average of all Query vectors would thus represent the average information need of the image patches, not their collective content.
- • Key (K): A token’s Key vector serves as its “identifier” or “advertisement”. It is the representation that the token “broadcasts” to the sequence, against which other tokens’ Queries are matched. Consequently, the average of all Key vectors logically synthesizes a representation of the average content or “gist” of the entire set of visual tokens. This is conceptually analogous to the function of a [CLS] token.
- • Value (V): A token’s Value vector contains the actual information or “payload” that is transmitted to other tokens once attention scores are computed. Averaging Value vectors would yield a representation of the average information payload, which may be less discriminative for identifying semantically distinct regions than the average Key.

Based on this analysis, we hypothesize that the mean of all patch tokens’ Key vectors is the most theoretically sound and effective proxy for a global context vector in the absence of a [CLS] token. It directly captures the aggregated semantic identity of the visual patches.

Empirical validation. We empirically validate this theoretical choice through both qualitative and quantitative analyses. We utilize attention mechanism vectors—Query (Q), Key (K), and Value (V)—along with feature representations (Hidden States) as baselines to compute global mean vectors. Equivalent token is then generated based on these vectors and subsequently used for experimental evaluation. The qualitative experimental results are shown in the Figure 5.

The findings indicate that as the layer increases, the saliency of instruction-related features becomes more pronounced. When using Q, V, and feature vectors as baselines, as the visual feature encoding is completed (i.e., at layer 24), although the features in answer-relevant regions become more prominent, answer-irrelevant regions still exhibit a certain degree of saliency. This suggests that the redundant tokens selected based on these methods are more likely to overlap with the answer-relevant regions, potentially affecting the final information selection process.

In contrast, when using the K vector as the baseline, the mean token exhibits distinct characteristics. Although its saliency in answer-relevant regions is less pronounced compared to the Q, K, and feature vector baselines, the scores of answer-irrelevant regions are better suppressed. This implies that the influence of answer-irrelevant regions on relevant regions is reduced, allowing for more effective filtering of redundant tokens. As a result, this setting proves to be more efficient in preserving information critical to the final task. Therefore, we compute the mean of the keys across the attention head dimension to mitigate local attention biases.

Second, the quantitative results in Table 9 further corroborate our choice. We evaluate different proxies under an extreme compression setting (82.4% TFLOPs reduction). The results indicate that replacing the aCLSi with the aEqi leads to only a 0.16% decrease in average accuracy, demonstrating the effectiveness of the selected alternative token. Moreover, compared to directly obtaining an equivalent token using the mean of feature vectors (aHi ), the equivalent token computed based on keys achieves a 0.60% improvement in average accuracy. Additionally, in comparison with equivalent tokens derived from Q and V, the key-based equivalent token improves average accuracy by 1.89% and 0.19%, respectively. These quantitative experimental results suggest that the equivalent token computed using key vectors can more comprehensively capture the global semantic information, thereby contributing to better performance.

Implementation of the Equivalent Token. The specific implementation process of equivalent tokens is as follows.First, to mitigate bias from any single attention head, we compute the mean of the Key States across all H attention heads:

H

1 H

Kh (9)

M =

h=1

where Kh ∈ RB×T×D represents the key States of the h-th attention head, H is the number of attention heads, and M ∈ RB×T×D represents the mean key. To compute the patch tokens, we extract all tokens except the first one ([CLS] token):

P = {Mi}Ti=2 = [M2,M3,...,MT] (10)

where P ∈ RB×(T−1)×D contains the patch tokens, Mi represents the i-th token in M, T − 1 represents the number of patch tokens after removing the [CLS] token. Then the mean patch token is computed as:

1 T − 1

µ =

T

Mi (11)

i=2

MMB

SQA

80

| |
|---|

68.4

67.3 66.2

70

64.9

60.2

58.0

60

###### Performance(%)

49.6

50

39.2

40

30

20

10

0

-VR +VR -VR-SR +VR-SR

- Figure 6: Comparative experiment on the sign of visionaware redundancy. VR denotes vision-aware redundancy, while SR denotes semantic-aware redundancy. Our proposed formulation (+VR) shows superior performance.

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Original Image [CLS] Attention Patch Attention Negative Patch Attention Quantitative results

[Figure 123]

- Figure 7: Visualization of attention scores. Brighter regions indicate higher attention weights, and red boxes highlight regions relevant to the answer. The “Negative Patch Attention” map (our redundancy metric) better aligns with the semantically focused “[CLS] Attention” map.

where µ ∈ RB×1×D represents the average patch token.

The cosine similarity between the mean patch token µ and each patch token Pi is given by:

µ · Pi ∥µ∥2∥Pi∥2

(12)

cos_sim(µ,Pi) =

where ∥µ∥2 is the L2 norm of the mean patch token, ∥Pi∥2 is the L2 norm of the i-th patch token.

The final computed is

aEqi = −cos_sim(µ,Pi) (13)

which results in a tensor of shape RB×(T−1), representing the negative cosine similarity between the mean patch token and each individual patch token. The core reason for taking the negative of cos_sim(µ,Pi) is that we aim to emphasize the difference from the global mean vector rather than its similarity. When cos_sim(µ,Pi) is negative, it indicates that µ and Pi are in opposite directions, suggesting that the token possesses a high degree of independence and contains crucial information. Conversely, when cos_sim(µ,Pi) is positive, it implies that the token’s features closely resemble the global mean, making it more likely to be redundant. By taking the negative, we prioritize preserving tokens with higher information density while suppressing the influence of redundant tokens, leading to a more precise selection of relevant information. Finally, by replacing aCLSi in Eq. 1 with aEqi , the FiCoCo-V can be used in a version that does not require the [CLS] token.

##### 10.3 Analysis of the Sign of Vision-aware Redundancy

When filtering out the discarded tokens, we propose a core hypothesis grounded in an information-theoretic perspective. Our hypothesis is that the total attention a visual token receives from other visual tokens (i.e., its “attention-in” score) serves as a proxy for its predictability. In information theory, a signal that is highly predictable by its context has low information entropy and thus contains less “new” or unique information. Such a token is, by definition, redundant.

However, this perspective stands in stark contrast to prior works like FastV (Chen et al. 2024a), which interpret attention scores as a direct measure of importance. We argue that such approaches may conflate a token’s importance with its predictability. A token might receive high attention because its content can be easily reconstructed from its neighbors (making it redundant), not necessarily because it is uniquely vital. Here, we provide empirical evidence to validate this hypothesis. First, we conduct a direct comparative experiment. We define our standard vision-aware redundancy term as +VR (as in Eq. 1, where it contributes positively to the total redundancy score). We then test an alternative formulation, -VR, where we flip the sign, aligning it with the “attention as importance” paradigm. This term is combined with our standard semantic-aware redundancy term (-TR). As shown in Figure 6, the +VR configuration, which treats high inter-patch attention as redundancy, consistently and significantly outperforms the -VR configuration across multiple benchmarks. This result provides strong quantitative support for our hypothesis, demonstrating that penalizing predictable tokens is more effective for performance preservation than preserving them.

Second, we offer qualitative insight through visualization in Figure 7. We compare the attention map of the [CLS] token (representing global semantic importance) with the averaged attention map of all patch tokens (representing interpatch dependency, i.e., our vision-aware redundancy). The [CLS] token clearly focuses on the semantically critical regions relevant to the answer (the text on the bottle). In contrast, the patch-to-patch attention is more diffuse and highlights regions of textural similarity and local context. When we treat this patch attention as redundancy (visualized as “Negative Patch Attention”, where brighter means less redundant), the resulting saliency map aligns much better with the [CLS] attention map. This visualization corroborates our information-theoretic view: tokens that are less predictable by their peers (darker in the “Patch Attention” map) are the ones that carry the most unique, globally relevant information.

Together, these quantitative and qualitative results provide strong validation for our core hypothesis, establishing a more principled foundation for identifying and filtering redundant visual tokens.

##### 10.4 Analysis of Local Penalty Strategy

We conduct qualitative analyses to validate the effectiveness of the proposed local penalty strategy. In the qualitative

w/ local penalty w/o local penalty

||[Figure 124]<br><br>[Figure 125]<br><br>|[Figure 126]|
|---|
|
|---|
<br><br>|[Figure 127]<br><br>[Figure 128]|
|---|
<br><br>|Q: “What is written at the top of the yellow sticker on the fridge?”|
|---|
<br><br>|Ground Truth:“Warning! Do not unplug!”|
|---|
<br><br>|LLaVA1.5: “Warning! Do not unplug!”√|
|---|
<br><br>|FiCoCo-V: “Ware.” ×|
|---|
<br><br>|Q: “What is the 3 letter word to the left of<br><br><casa> in the text?”|
|---|
<br><br>|Ground Truth:“tua.”|
|---|
<br><br>|LLaVA1.5: “tua.” √|
|---|
<br><br>|FiCoCo-L: “mal.” ×|
|---|
<br><br>|FiCoCo-V: “fica.” ×|
|---|
<br><br>(a)<br>(b)<br><br><br>|FiCoCo-L: “Warning.” ×|
|---|
|
|---|

| |
|---|

| |
|---|

[Figure 129]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 8: Visualization of local penalty. Yellow boxes denote regions irrelevant to the answer, whereas red boxes indicate answer-relevant areas.

|Method<br><br>|Quant TFLOPs↓ Memory (GB)↓ KV-Cache (MB)↓|
|---|---|
|LLaVA-1.5<br><br>FiCoCo-V FiCoCo-L|FP16 8.5 22.4 333<br><br>FP16 1.5 (↓82%) 14.4 (↓36%) 65.0 (↓80%) FP16 1.5 (↓82%) 14.3 (↓36%) 64.2 (↓81%)<br><br>|
|LLaVA-1.5<br><br>FiCoCo-V FiCoCo-L<br><br>|INT8 4.3 11.2 167 INT8 0.8 (↓81%) 7.8 (↓30%) 32.5 (↓81%) INT8 0.8 (↓81%) 7.2 (↓36%) 32.1 (↓81%)|
|LLaVA-1.5<br><br>FiCoCo-V FiCoCo-L|INT4 2.1 6.2 83.4<br><br>INT4 0.4 (↓81%) 4.4 (↓29%) 16.3 (↓81%) INT4 0.4 (↓81%) 3.3 (↓47%) 16.1 (↓81%)<br><br>|

Figure 9: Failure cases of FiCoCo. FiCoCo-L produces answers more closely aligned with the questions.

nificant improvements in both computational efficiency and GPU memory utilization. Specifically, our FiCoCo series reduces computational overhead by nearly 80%, GPU memory usage by approximately 40%, and KV-Cache storage by around 80%, all while achieving performance comparable to LLaVA-1.5-7B. Notably, this is accomplished without requiring any additional training, highlighting the efficiency and flexibility of our FiCoCo series.

- Table 10: Efficiency analysis of methods based on LLaVA1.5-7B.

study, we retain 200 visual tokens and evaluate the spatial uniformity of the discarded tokens. Figure 8 demonstrate that introducing the local penalty strategy significantly enhances the spatial uniformity of token distribution. Specifically, as highlighted in the red-boxed regions of the figure, without this strategy, token compression tends to concentrate spatially, leading to substantial information loss in critical regions and causing irreversible degradation in the final model output. In contrast, the local penalty strategy mitigates this global averaging tendency, promoting a more uniform spatial distribution of compressed tokens and better preservation of key information.

|Method|Quant TFLOPs↓ Memory (GB)↓ KV-Cache (MB)↓<br><br>|
|---|---|
|LLaVA-1.5<br><br>FiCoCo-V FiCoCo-L<br><br>|FP16 28.6 56.1 891 FP16 15.4 (↓46%) 38.6 (↓31%) 488 (↓43%) FP16 15.4 (↓46%) 38.4 (↓32%) 485 (↓46%)|
|LLaVA-1.5<br><br>FiCoCo-V FiCoCo-L<br><br>|INT8 14.3 28 446 INT8 7.7 (↓46%) 19.3 (↓31%) 244 (↓45%) INT8 7.7 (↓46%) 19.2 (↓31%) 242 (↓46%)|
|LLaVA-1.5<br><br>FiCoCo-V FiCoCo-L|INT4 7.6 14 223<br><br>INT4 3.9 (↓46%) 9.6 (↓32%) 122 (↓49%) INT4 3.9 (↓49%) 9.5 (↓32%) 121 (↓46%)<br><br>|

- Table 11: Efficiency analysis of methods based on LLaVA1.5-13B.

##### 10.6 Analysis of Failure Cases

FiCoCo maintains substantial performance even when compressing a significant number of visual tokens. However, the inevitable loss of visual information during the token reduction still causes failure cases. We show two cases in Figure 9 where the answers generated by LLaVA-1.5 are consistent with the ground truth, while FiCoCo-L and FiCoCo-V fail to answer correctly. By analyzing the erroneous responses generated by FiCoCo-L and FiCoCo-V, it can be observed that FiCoCo-L produces answers more closely aligned with the questions, guided by the token selection process involving textual information. For instance, in Figure 9(a), the prompts ‘top’ and ‘yellow sticker’ jointly indicate the yellow region at the top of the refrigerator, leading FiCoCo-L to search for the answer in this specific region. However, FiCoCo-V fails to attend to the crucial information regarding ‘top’. Moreover, in Figure 9(b), the cues ‘3 letter word’ and ‘left of casa’ jointly guide the answer towards ‘tua.’ Although the generated answer of FiCoCo-L is ‘mal’, it more effectively considers these two cues. In contrast, FiCoCo-V fails to adequately track the critical information pertaining to ‘3 letter word.’

##### 10.5 Detailed Efficiency Analysis

##### 10.7 Sensitivity Analysis of Hyperparameters

Utilizing the tools provided by (Yuan et al. 2024), we conduct a detailed analysis of the theoretical efficiency of our FiCoCo. In Table 10, we assume the number of textual tokens is 60 for LLaVA-1.5-7B. And in Table 11, we assume the number of textual tokens is 512 for LLaVA-1.5-13B. The results demonstrate that, compared to the baseline models of LLaVA-1.5-7B/13B, our FiCoCo series achieve sig-

We explore the hyperparameter configurations of FiCoCo, performing sensitivity analysis on individual parameters to assess their impact. The experiments are conducted on both TextVQA and SQA benchmarks, with TFLOPs at 1.5.

Trade-off hyperparameters. It is observed that: (1) The hyperparameter λ = 0.35 is the optimal setting. Under

70

70

70

68

68

68

66

66

66

64

Accuracy(%)

Accuracy(%)

Accuracy(%)

62

64

64

60

SQA FiCoCo-L

SQA FiCoCo-L

62

62

TextVQA FiCoCo-L

TextVQA FiCoCo-L

58

60

60

56

54

58

58

52

56

56

SQA FiCoCo-V

50

TextVQA FiCoCo-V

48

54

54

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

on SQA & TextVQA

on SQA & TextVQA

on SQA & TextVQA

Figure 10: Hyperparameter sensitivity analysis of λ, β and γ on TextVQA and SQA benchmarks.

|ε|FiCoCo-V FiCoCo-L SQA TextVQA SQA TextVQA<br><br>|
|---|---|
|0.998 0.996 0.994 0.992<br><br>|68.37 55.46 69.46 55.72 68.33 53.15 69.51 55.62 68.21 52.05 69.32 55.42 68.47 52.29 69.36 55.14|

- Table 12: Hyperparameter sensitivity analysis of ε on TextVQA and SQA benchmarks.

|scaling coefficient in local penalty strategy<br><br>|FiCoCo-V SQA TextVQA|
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br><br>|68.12 53.24 68.37 55.46 68.21 55.04 68.11 55.49|

- Table 13: Hyperparameter sensitivity analysis of scaling coefficient in local penalty strategy on TextVQA and SQA benchmarks.

fusion.

Scaling coefficient hyperparameter in local penalty strategy. Table 13 shows that when the scaling coefficient exceeds 2, the performance stably closes to optimal. Therefore, to balance design simplicity and performance stability, we opt to fix the punishment coefficient at 2.

#### 11 Algorithm Illustration

We provide a detailed explanation of our FiCoCo-V and FiCoCo-L processes in Algorithm 1 and Algorithm 2, respectively, to facilitate a clearer understanding of the methods we propose.

this configuration, both FiCoCo-V and FiCoCo-L variants achieve relatively optimal accuracy. This indicates that when λ = 0.35, FiCoCo effectively balances the local information conveyed by patch tokens with the global information carried by the [CLS] token, thereby enhancing the integration of visual features and the completeness of information. (2) The hyperparameter β = 0.6 is the optimal setting. For the SQA dataset, FiCoCo-L demonstrates a clear upward trend between β = 0.4 and β = 0.6, with a similar trend observed on the TextVQA dataset. This finding suggests that, under this parameter setting, an effective balance is achieved between textual information and the information conveyed by patch tokens. (3) The hyperparameter γ = 0.6 is the optimal setting. Figure 10 clearly shows that FiCoCo-V and FiCoCo-L both reach their performance peaks at γ = 0.6 across the two benchmarks. This result suggests that incorporating semantic similarity more effectively guides the selection of the target set during the compress stage, thereby optimizing overall performance.

ε hyperparameter. Table 12 compares the impact of different quantile thresholds ε-th. Experimental results demonstrate that setting ε to 0.998 yields optimal performance on both the TextVQA and SQA benchmarks. However, as ε-th decreases, the information of a single token gets distributed across more tokens, which leads to a noticeable performance drop in both benchmarks due to the excessive information

Algorithm 2: FiCoCo-L

Algorithm 1: FiCoCo-V

Require: Input tokens X ∈ R(N+M)×D, attention score tensor Al ∈ R(N+M)×(N+M), reduction factor NS ∈ R, number of visual tokens N ∈ R, number of textual tokens M ∈ R, hyperparameters β, γ, ε ∈ [0,1]

Require: Input tokens X ∈ RN×D, attention score tensor Av ∈ RN×N, [CLS] attention score vector aCLS ∈ RN, reduction factor NS ∈ R, number of visual tokens N ∈ R, hyperparameters λ, ε ∈ [0,1]

S)×D

Ensure: Output tokens X ∈ R(N+M−N

S)×D

Ensure: Output tokens X ∈ R(N−N

- 1: Stage 1: Filter
- 2: Compute redundancy scores for all visual tokens:

sli = β

1 N

N

j=1

Ali,j − (1 − β)

N+M

k=N+1

Ali,k

- 3: Identify source set S = topK(sv,NS) that contains the indices of NS discarded visual tokens
- 4: Identify target set T that contains the indices of (N − NS) preserved visual tokens
- 5: Stage 2: Correlate
- 6: Compute direct and indirect correlations:

Cli,j = γAli,j + (1 − γ)

N+M

k=N+1

Ali,k · Alk,j

- 7: Stage 3: Compress
- 8: Apply token-wise quantile-based thresholding: τi = quantile(Cli,:,ε)
- 9: Compute token-adaptive topK correlations: Ij = {i ∈ S and Cli,j ≥ τi}, Ji = {j ∈ T and Cli,j ≥ τi}
- 10: Compute compression weights:

αij =

Cli,j

j∈Ji Cli,j

- 11: Update preserved tokens with self-preserving compression:

XTj ←

XTj + i∈I

j

αijXSi 1 + i∈I

j

αij

- 12: Output tokens: X ← X \ XS
- 13: Return X

- 1: Stage 1: Filter
- 2: Compute redundancy scores for all visual tokens:

svi = λ

1 N

N

j=1

Avi,j − (1 − λ)aCLSi

- 3: Apply local penalty to sv
- 4: Identify source set S = topK(sv,NS) that contains the indices of NS discarded visual tokens
- 5: Identify target set T that contains the indices of (N − NS) preserved visual tokens
- 6: Stage 2: Correlate
- 7: Construct correlation matrix: Cvi,j = Avi,j, i ∈ S, j ∈ T
- 8: Stage 3: Compress
- 9: Apply token-wise quantile-based thresholding: τi = quantile(Cvi,:,ε)
- 10: Compute token-adaptive topK correlations: Ij = {i ∈ S and Cvi,j ≥ τi}, Ji = {j ∈ T and Cvi,j ≥ τi}
- 11: Compute compression weights:

αij =

Cvi,j

j∈Ji Cvi,j

- 12: Update preserved tokens with self-preserving compression:

XTj ←

XTj + i∈I

j

αijXSi 1 + i∈I

j

αij

- 13: Output tokens: X ← X \ XS
- 14: Return X

