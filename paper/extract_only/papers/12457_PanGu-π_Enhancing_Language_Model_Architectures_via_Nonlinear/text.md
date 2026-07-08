## PanGu-π: Enhancing Language Model Architectures via Nonlinearity Compensation

Yunhe Wang, Hanting Chen, Yehui Tang, Tianyu Guo, Kai Han, Ying Nie, Xutao Wang, Hailin Hu, Zheyuan Bai, Yun Wang, Fangcheng Liu, Zhicheng Liu, Jianyuan Guo, Sinan Zeng, Yinchen Zhang, Qinghua Xu, Qun Liu, Jun Yao, Chao Xu, and Dacheng Tao Fellow, IEEE

### arXiv:2312.17276v1[cs.CL]27Dec2023

Abstract—The recent trend of large language models (LLMs) is to increase the scale of both model size (a.k.a the number of parameters) and dataset to achieve better generative ability, which is definitely proved by a lot of work such as the famous GPT and Llama. However, large models often involve massive computational costs, and practical applications cannot afford such high prices. However, the method of constructing a strong model architecture for LLMs is rarely discussed. We first analyze the state-of-the-art language model architectures and observe the feature collapse problem. Based on the theoretical analysis, we propose that the nonlinearity is also very important for language models, which is usually studied in convolutional neural networks for vision tasks. The series informed activation function is then introduced with tiny calculations that can be ignored, and an augmented shortcut is further used to enhance the model nonlinearity. We then demonstrate that the proposed approach is significantly effective for enhancing the model nonlinearity through carefully designed ablations; thus, we present a new efficient model architecture for establishing modern, namely, PanGu-π. Experiments are then conducted using the same dataset and training strategy to compare PanGu-π with state-of-the-art LLMs. The results show that PanGu-π-7B can achieve a comparable performance to that of benchmarks with about 10% inference speed-up, and PanGu-π-1B can achieve state-of-the-art performance in terms of accuracy and efficiency. In addition, we have deployed PanGu-π-7B in the high-value domains of finance and law, developing an LLM named YunShan for practical application. The results show that YunShan can surpass other models with similar scales on benchmarks.

Index Terms—Transformer, Large Language Model, Nonlinearity, Network Architecture, Finance, Law.

✦

1 INTRODUCTION

# L

ARGE language models (LLMs) have significantly evolved and are capable of a wide range of NLP tasks such as ma-

chine translation, text summarization, and dialogue. Following the scaling law [1], a series of studies confirm significantly improved performances and emergent abilities [2] on downstream tasks by scaling up the model size and the data size, e.g., GPT-3 with 175B parameters [3] and PaLM with 540B parameters [4]. Recently, a remarkable innovation, ChatGPT, was introduced with the ability to interact with humans in a conversational way. The success of ChatGPT is attributed to pre-training on extensive textual data and fine-tuning with alignment to human preferences. This paradigm has a profound impact on subsequent research and real-world applications [5], [6]. The emergence of ChatGPT has inspired the community to develop more excellent large models, including LLaMA [7], ChatGLM [8], and Baichuan [9], which in turn drive the vigorous development of the LLM field.

In addition to general-purpose LLMs, high-value domainspecific large models are also being extensively researched to promote the implementation and practical application of LLMs. For example, LaWGPT [10] enhances the fundamental semantic understanding in the field of law with a specialized legal vocabulary and extensive pre-training on a large-scale Chinese legal corpus. FinGPT [11] develops an open-source LLM in a datacentric approach. Huatuo [12] builds a Chinese medical instruction

- • Corresponding to Yunhe Wang (Huawei Noah’s Ark Lab). E-mail: yunhe.wang@huawei.com.
- • Acknowledgments: This work is jointly funded by Huawei 2012 Labs and Huawei Group Finance. We also thank the work from both the data engineering team and IT architecture team.

fine-tuning dataset and enhances its performance in questionanswering within the medical field. As shown in Figure 1, our analysis of the industry distribution of domain-specialized LLMs reveals that those in the finance and law domains attract the most attention due to their widespread demand and commercial value.

The Transformer model introduced in 2017 [13] is the foundational architecture for many LLMs. The core components of the Transformer model include the multi-head self-attention (MSA) and the feed-forward network (FFN). MSA computes attention scores for each token in the input sequence with respect to all other tokens, capturing relationships and dependencies. FFN is performed on each token separately which provides more nonlinear transformation and model capacity. Transformer architectures are utilized to build encoder (e.g., BERT) and decoder (e.g., GPT-2) for NLP tasks. In LLM, decoder-only architecture is widely used by predicting the next token with the context information [3], [5]. Beyond the standard Transformer architecture, a series of studies have explored architecture modification (especially MSA [14], [15] or FFN [4], [7]) seeking better performance. PaLM [4] and LLaMA [7] use SwiGLU-based FFN [16] which consists of the component-wise product of two linear layers, showing significantly increased generation quality. RWKV (Receptance Weighted Key Value) [14] proposes an RNN-style attention mechanism to alleviate the quadratic complexity in standard MSA. Switch Transformer [17] allocates different parameters for each input example and results in a sparsely-activated model.

The development of an excellent LLM is a complex system engineering, which includes data preparation, data cleaning, model architecture, cluster commutation, and optimizer. The model architecture design is one of the most important components and

[Figure 1]

Domain Specialized LLMs Statistics @ Dec. 2023 from Github / Listed Company Report

[Figure 2]

41

Domain specialized

LLMs

[Figure 3]

###### 100+

26

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Finance

[Figure 8]

[Figure 9]

General LLMs

[Figure 10]

[Figure 11]

[Figure 12]

DevOpsModel

[Figure 13]

& Law

MathGPT

TransGPT NSFGPT

6

4

2 2 1 1 1

Domain specialized LLMs arise for real-world applications

- Fig. 1: Statistics of domain specialized LLMs. The general LLMs face challenges in supporting industry applications, leading to a growing emphasis on domain specialized LLMs. Among them, the fields of finance and law are particularly active.

determines the maximum performance potential of the deployed LLM. Among the recent projects in 2022-2023, the popular versions that are often used for secondary development are GPT3 [3] and LLaMA [7]. By inheriting the observations and analysis of our previous work [18], we find that the feature collapse problem also affects the expressive power of these well-designed Transformer architectures. Taking LLaMA as an example, we empirically analyze its feature collapse phenomenon using the rank metric [19]. The feature rank diminishes significantly in deeper layers, leading to a greater similarity among all tokens. This greatly degrades the generation quality and diversity of LLMs. We also theoretically analyze the feature collapse problem in Transformer architecture (details in Section 3). Through theoretical analysis, we have discovered that nonlinearity significantly impacts the capabilities of the Transformer model. Enhancing nonlinearity can effectively mitigate the issue of feature collapse and improve the expressive power of the Transformer model. We intend to construct stronger LLM architectures by approaching them from a nonlinear perspective.

In this paper, we introduce a new architecture for LLMs to address the feature collapse problem via nonlinearity compensation, named PanGu-π. We introduce more nonlinearity from two approaches in both FFN and MSA modules without significant increasing the model complexity. First, the series-based activation function with multiple learnable affine transformation is equipped in FFN, which can effectively enhance the nonlinearity of the entire network with negligible calculations. Then, the augmented shortcut is paralleled with the main branch of each MSA module to eschew the rank collapse. To maintain the model efficiency, we carefully refine the augmented shortcut operation with hardwarefriendly operations. The enhanced PanGu-π architectures (see Figure 2) are constructed with both the series activation-based FFN and shortcut-augmented MSA. We also prove that the superposition of these two operations can enhance nonlinear compensation. We build two versions of PanGu-π with different model sizes, i.e., PanGu-π-7B and PanGu-π-1B. By training on a largescale corpus, our PanGu-π models obtain general language ability on downstream tasks. Through carefully designed ablations, we demonstrate that the proposed approach can effectively enhance the model nonlinearity and alleviate feature collapse. Thus, with the same scale of parameters, we can achieve a substantial efficiency gain via the two new modules. Extensive experiments on various NLP tasks are evaluated to compare with state-of-

the-art LLMs. In a scenario with similar model size, PanGu-π models can achieve better performance in terms of both accuracy and efficiency. In addition to the foundational abilities, we have deployed PanGu-π-7B in the high-value domains of finance and law, developing a specialized LLM named YunShan for practical application. Extensive evaluations of finance and law benchmarks also show that YunShan surpasses other state-of-the-art models with similar scales.

This work introduces a new LLM network architecture (i.e., PanGu-π) with extensive experiments and theoretical analysis with some of the ideas borrowed from our preliminary works published on NeurIPS 2023 [20] and NeurIPS 2021 [18]. This present work makes the following significant contributions. First, the previous two papers, one involving the construction of a CNN backbone [20] and the other focusing on vision Transformers [18], have laid a foundation that this paper seeks to extend within LLMs. We achieved commendable results in this endeavor, and extensive experiments have validated the effectiveness of our methods. Second, the previous two papers analyzed the network design subject from distinct perspectives. In our current study, the two works are theoretically integrated and essentially address the same critical issue from a unified standpoint in the LLM domain. Third, we organically adapted series activation to FFN and integrated augmented shortcuts into MSA. These two components are complementary to each other and effectively introduce more nonlinearity into the Transformer architecture. Fourth, we developed the PanGu-π foundation models by large-scale training and fine-tuning (SFT), which achieved state-of-the-art results in general NLP tasks for a similar model size. In addition, we extend PanGu-π to finance and legal domains by transfer learning and obtain excellent performance on these downstream tasks.

The rest of this paper is organized as follows. Section 2 reviews related work in the field of Transformer architectures for building LLMs and the related hotspot applications. Section 3 provides a theoretical analysis of the feature collapse problem and the nonlinear capabilities of existing Transformer architectures. Section 4 introduces a nonlinear enhancement strategy based on the series activation function and augmented shortcut. Section 5 details the data, training strategies, and experimental results of the PanGu-π architecture with two models with important parameter scales, i.e., PanGu-π-7B and PanGu-π-1B. In Section 6, the PanGu-π architecture is deployed in the high-value domains of finance and law, developing the YunShan LLM for practical

|[Figure 14]<br><br>|
|---|

|[Figure 15]<br><br>|
|---|

|[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>|
|---|

|[Figure 19]<br><br>|
|---|

|[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>|
|---|

[Figure 23]

[Figure 24]

× N

|[Figure 25]<br><br>|
|---|

[Figure 26]

[Figure 27]

[Figure 28]

|[Figure 29]<br><br>|
|---|

|[Figure 30]<br><br>|
|---|

|[Figure 31]<br><br>|
|---|

|[Figure 32]<br><br>|
|---|

|[Figure 33]<br><br>|
|---|

- Fig. 2: The diagram of the proposed PanGu-π architecture. The series activation function is adapted to FFN, and the augmented shortcuts are integrated effectively introduces more nonlinearity into the Transformer architecture.

|into MSA, which| |
|---|---|
| | |

|evaluations of finance|
|---|

application. Extensive

and law benchmarks also show that YunShan surpasses other state-of-the-art models with similar scales. Section 7 concludes the entire paper and discusses future works.

counts. Skywork [32] presents 13B LLMs trained on a corpus drawn from both English and Chinese texts with a two-stage training methodology.

Inputs Embedding Norm Aug-MSA SIAF-MLP Causal Mask

Outputs

Norm

###### 2.2 Enhanced Transformer Architectures

× N

#### 2 RELATED WORKS

While Transformer architectures have gained significant prominence in LLMs recently, there continues to be a surge of interest in their effective utilization in diverse domains, including computer vision tasks. In light of this, we review classical works dedicated to enhancing Transformer structures, with a particular focus on augmenting model nonlinearity and efficiency.

In this section, we first summarize the recent representative works in the field of LLMs. We then review the classical works for enhancing Transformer architectures. Lastly, we investigate the domain-specific large language models, especially in finance and law.

MSA Feature Shortcut Linear Linear

Act Act

[Figure 34]

Augmented Shortcut Act

Natural language processing domain. The conventional selfattention mechanism, with quadratic computational complexity, poses challenges for handling long input sequences during training and inference. To mitigate this, various structural priors on attention, including sparsity [33], [34], [35], [36], [37] and linear attention [15], [38], have been proposed. Notably, Reformer [39] employs locality-sensitive hashing to approximate full attention. Longformer [40] integrates local windowed attention with taskmotivated global attention. Models such as GPT-3[41] incorporate locally banded sparse attention methods, such as Factorized Attention [34]. There are also works focusing on replacing the attention module by incorporating recurrent models [42], [43], [44]. Hyena [45] trained a recurrence of gating units and implicitly parametrized long convolutions, which serves as an attention-free drop-in replacement for the traditional Transformer architecture. RWKV [46] replaced the quadratic QK attention with a scalar formulation that has linear cost. RetNet [44] theoretically derived the connection between recurrence and attention and proposed the retention mechanism for sequence modeling. There are also efficient enhancements focused on the Feed-Forward Network (FFN). Mixture-of-Experts (MoE) [47], [17], [48], [49], [50] has demonstrated effectiveness in the pre-training of LLMs. In addition to MoE, PaLM [51] and LLaMA [23] leverage the SwiGLU activation for original FFN intermediate activations. This choice is grounded in the observation that SwiGLU activations, as demonstrated in compute-equivalent experiments [52], substantially enhance quality compared to standard activation functions like ReLU, GeLU, or Swish.

###### 2.1 LLMs

With the emergence of ChatGPT [21] from OpenAI, LLMs with billions of parameters achieved astounding performance on various natural language processing tasks. Subsequently, the latest GPT-4 [22] pushed the generalization capabilities of LLMs to a new level. However, the proliferation of GPT-series models is accompanied by a strict commercial orientation that is not conducive to a thriving open source community. The representative work of democratic LLMs is LLaMA [23], a collection of open-source foundation language models ranging from 7B to 65B parameters. Later, a more elaborate model LLaMA2 [24] is introduced, appearing to be on par with some closed-source models [21] based on human evaluations. Since its release, LLaMA has attracted extensive attention from the academia and industry. Subsequent efforts have been based on LLaMA by either instruction tuning or continual pre-training. Stanford Alpaca [25] is the first LLaMAbased chatbot fine-tuned with 52K instruction-following samples generated by the self-instruct method [26]. Vicuna [27] also fine-tunes LLaMA with user-shared conversations collected from ShareGPT [28]. In addition to the language models geared toward English-dominant applications, multilingual language models are also thriving. InternLM [29] presents a multilingual foundational language model pre-trained on a large corpus with 1.6T tokens with a multi-phase progressive process. Baichuan2 [9] introduces a series of large-scale multilingual language models containing 7 billion and 13 billion parameters. PanGu-Σ [30] extends the dense Transformer model to a sparse one with Random Routed Experts. Qwen [31] introduces a comprehensive language model series that encompasses distinct models with varying parameter

Computer vision domain. PVT [53] and Swin [54] utilize hierarchical structures across multiple stages, overcoming

challenges posed by the original isotropic ViT [55] for diverse computer vision tasks. Ongoing research focuses on refining local information processing [56], [57], [58], [59], [60], [61], [62], simplifying attention mechanisms [63], [64], [65], and exploring alternative modules [66], [67], [68], [69], [70], [71]. For example, T2T-ViT [72] reduces token length through iterative aggregation, whereas TNT [62] captures local information by dividing ViT’s patches. Swin [54] and Cswin [73] introduce local attention within a window and shifted window partitioning for cross-window connections. GFnet [74] employs Fast Fourier Transform for token mixing. Architectures like ResMLP [75] and MLP-Mixer [76], solely rely on multi-layer perceptrons (MLPs), excluding convolutions or self-attention mechanisms.

###### 2.3 LLMs for Finance and Law

In addition to the LLMs towards general purpose, the domainspecific models that are more capable of generating applied value are receiving increasing attention, with finance and law being the most representative.

Financial LLMs. Wu et al.propose the first proprietary LLM with 50 billion parameters specialized for the financial domain, i.e.BloombergGPT [77], which is a decoder-only causal language model based on BLOOM [78]. The proposed training strategy of mixing domain-specific and general-purpose data results in a balanced performance in both domains. Unlike the proprietary BloombergGPT, FinGPT [79], [80] takes a data-centric approach and presents an open-source LLM to researchers and practitioners. It exhibits promise in financial tasks such as sentiment classification, quantitative trading and financial fraud detection. PIXIU [81] has created a large-scale multi-task instruction dataset by manually reworking open-sourced datasets [82]. A financial LLM called FinMA is then introduced by fine-tuning LLaMA with the constructed instruction dataset. The comprehensive evaluation results including financial NLP and prediction tasks uncover the strengths and weaknesses of various LLMs when handling different financial tasks. To address the lack of open-sourced models specifically designed for Chinese finance, Zhang et al.introduce XUANYUAN 2.0 [83], built upon the BLOOM [78] architecture. To mitigate catastrophic forgetting, the hybrid-tuning strategy that combines the stages of pre-training and fine-tuning is proposed. By appropriately mixing the general and financial corpus in pretraining and fine-tuning, XUANYUAN 2.0 achieves impressive performance in both the general domain and financial domain. Chen et al.propose a financial LLM DISC-FinLLM [84] by multiple experts fine-tuning the framework based on Baichuan13B [85]. Experimental results on multiple evaluation benchmarks demonstrate its promising performance.

Legal LLMs. The legal sector is another area that is significantly benefitting from the advancement of LLMs. BaoLuo and Lychee [86], [87] are lawyer assistants developed by finetuning Chinese legal domain QA datasets. AI Lawyer [88] applies Active Learning to alleviate the problem of limited supervised data volume in the legal domain. FedJudge [89] focuses on the privacy of legal data and adopts Federated Learning [90] during instruction tuning, it also utilizes Continual Learning [91] to mitigate the issue of data distribution shifts. HanFei, LaWGPT, Lawyerllama, WisdomInterrogatory, and Fuzi.Mingcha [92], [10], [93], [94], [95] undergo a two-phase training process: further pretraining with unsupervised legal corpus to enhance the semantic understanding ability in the legal field and then supervised

training with corresponding datasets. HanFei [92] is the first legal LLM in China fully trained with 7B parameters and supports multi-turn dialogue, LaWGPT [10] expands the vocabulary by adding specific legal domain terms, Lawyer-llama [93] has experimented with different data permutations and training sequences during its instruction tuning phase. During inference time, LawGPT zh(XieZhi), LexiLaw, ChatLaw, Lawyer-llama, Fuzi.Mingcha and DISC-LawLLM [96], [97], [98], [93], [95], [99] introduce a retrieval module to ensure that a definite legal document in the knowledge base supports the response. Additionally, ChatLaw [98] also involves a Keyword model for key information extraction to reduce the ambiguity of user queries. Fuzi.Mingcha [95] enables the LLM to use syllogistic reasoning to arrive at verdict predictions by training the LLM on a selfconstructed dataset.

#### 3 PRELIMINARIES AND MOTIVATION

In this section, we commence by dissecting the foundational architecture of the Transformer model, introducing a metric of nonlinearity to articulate its capabilities. Subsequently, we delve into an analysis of the Transformer architecture’s components – the multi-head self-attention and the multi-layer perceptrons modules – scrutinizing their nonlinear expressive capability. This exploration also brings to light the limitations inherent in the current incarnations of Transformer architectures.

Recognized for the Transformer’s suitability for parallel computing and the inherent complexity of its model, the Transformer has demonstrated superior precision and performance compared to the widely adopted RNN recurrent neural network. The Transformer architecture consists of two parts: the multi-head selfattention and the multi-layer perceptrons modules.

The multi-head attention module is a fundamental component of the Transformer architecture. An MSA module with H heads is defined as

###### MSA(Zl) = Concat([AlhZlWlhv ]Hh=1)Wlo, l ∈ [1,2,··· ,L],

(1)

where Zl ∈ RN×d is the feature of the l-th MSA layer, Alh ∈ RN×N and Wlhv ∈ Rd×(d/H) are the corresponding attention map and value projection matrix in the h-th head, respectively. Concat(·) denotes the concatenating for features of the H heads and Wlo ∈ Rd×d is the output projection matrix. The attention matrix Alh is calculated by the self-attention mechanism, i.e.,

Alh = softmax

(ZlWlhq )(ZlWlhk )⊤

√

d

, (2)

where Wlhq ∈ Rd×(d/H) and Wlhk ∈ Rd×(d/H) are the query and value projection matrices, respectively. Attention Alh reflects the relation between different tokens, and a larger value Aijlh indicate that token i and token j have a stronger relationship.

An MLP module is defined as

MLP(Zl′) = σ(Zl′Wl′

##### )Wl′

, l ∈ [1,2,··· ,L], (3) where Zl′ ∈ RN×d is the features of the l-th MLP layer ,

2

1

##### Wl′

and Wl′

∈ Rd×d are the weight matrixs.

1

2

One of the paramount capabilities of neural networks is their nonlinear expressive capability. The higher the degree of nonlinearity, the more complex the function space the network can approximate, resulting in enhanced accuracy. In traditional

Convolutional Neural Networks (CNNs), nonlinearity is primarily imparted by activation functions. However, in the Transformer architectures, the sources of nonlinearity are twofold: the selfattention mechanisms and the activation functions within the Multi-Layer Perceptrons (MLP). Hence, our analysis separately scrutinizes the nonlinear expressive capabilities of both selfattention and the MLP within the Transformer framework.

Define Mm := {Y ∈ RN×m|Y = 1x⊤,x⊤ ∈ R1×m} as a subspace in RN×m, where 1 = [1,1,...,1]⊤ ∈ RN×1, n is the number of tokens and d is the dimension of token representation. We define the distance between matrix H ∈ RN×m and Mm as dM

m ∥H − Y ∥F, where ∥ · ∥F is the Frobenius norm. dM

(H) := minY ∈M

m

(Zl) [100] is a commonly used metric to measure the capability and nonlinearity of the Transformer architecture. Next, we investigate the distance between Zl the output of layer l and subspace Md.

m

We begin by examining the nonlinear expressive capabilities of the self-attention modules. The self-attention matrix within the Transformer can be intuitively likened to the normalized adjacency matrix of a corresponding graph. Viewed through a graph perspective, the self-attention layer can be seen as equivalent to a Graph Neural Network (GNN) operating on a fully connected graph with normalized edge weights. Excessive self-attention layers like GNN layers result in excessive smoothing, with node vector representations tending to be the same, resulting in convergence to a specific low-rank subspace for any given input.

The following theoretical analysis utilizes the formula of the self-attention layer to shed light on the intricacies of this phenomenon. We study how the self-attention layer converges with low rank based on matrix projection. Our analysis involves the definition of a subspace M characterized by a unique property in which each row vector of its elements is the same. By scrutinizing the behavior of the self-attention layer in relation to matrix projection, we aim to unravel the underlying mechanisms that drive the convergence of node vector representations toward a specific low-rank subspace. This exploration is essential for gaining deeper insights into the nuances of self-attention in Transformers, shedding light on the intricacies of their functioning and providing a theoretical foundation for potential enhancements and optimizations.

Lemma 1. For self-attention matrix A ∈ RN×N, any weight matrix W ∈ Rd×m, any H,B ∈ RN×d, α1,α2 ≥ 0 and σ is the nonlinear Lipschitz continuous activation function, we have:

(HW) ≤ sdM

dM

(H), dM

m

d

(σ(H)) ≤ LdM

(H), dM

d

d

(α1H + α2B) ≤ α1dM

(B), dM

(H) + α2dM

d

d

d

(AH) ≤ λmaxdM

###### (H),

d

d

where s is the largest singular value of W, λmax is the largest eigenvalue of A⊤(I − ee⊤)A and L is the Lipschitz constant of activation function σ(·).

Applying the lemma 1 to the single-head self-attention layer, we can obtain its low-rank attenuation.

Theorem 1. Given a model stacked by the MSA modules, the diversity dM(Zl) of feature in the l-th layer can be bounded by that of input data Z0, i.e.,

√

(AZlW) ≤

dM

λsυ1dM

(Zl).

m

d

where s > 0 is the largest element of all singular values of all W and λ is the largest eigenvalue of all A⊤(I − ee⊤)A for each self-attention matrix A.

For the low-rank matrix projection of concat matrices, we have the following lemma: Lemma 2. For block matrix Hh ∈ RN×m, we have:

H

(Concat([Hh]Hh=1))2 =

(Hh)2,

dM

dM

m

Hm

h=1

Applying the theorem 1 and the lemma 2 to the formula 1, we can see how the multi-headed self-attention layer decays layer by layer into the low-rank space.

- Theorem 2. Given a model stacked by the MSA modules, the diversity dM(Zl) of feature in the l-th layer can be bounded by that of input data Z0, i.e.,

dM

d

(MSA(Zl)) ≤

√

λHsυ1dM

d

(Zl). dM

d

(Zl) ≤ (

√

λHsυ1)ldM

d

(Z0).

where H is number of heads, s > 0 is the largest element of all singular values of all Wlhv and υ1 is the largest element of all singular values of all Wlo.

Assume further that A is doubly stochastic (so that A⊤e = e) with positive entries. Then by Perron–Frobenius theorem, A⊤A has a maximum eigenvalue 1 with associated eigenvector e as well. In this case, the matrix A⊤(I − ee⊤)A = A⊤A − ee⊤

has√a maximum eigenvalue λmax < 1.

λHsυ1 is usually smaller than 1, so the feature diversity dM

d

(Zl) decreases rapidly as the network depth increases. Recursively, Zl will converge toward subspace Md if √

λHsυ1 < 1

and all representations are the same, resulting in over-smoothing. In conclusion, the nonlinear expressive capabilities of the vanilla self-attention module are limited.

We then focus on the Multi-Layer Perceptrons (MLP) of the Transformer architecture.

- Theorem 3. Given a model stacked by the MLP modules, the

(Zl′) of feature in the l-th layer can be bounded by that of input data Z0′, i.e.,

diversity dM

d

(Zl′). dM

(MLP(Zl′)) ≤ Lsυ2dM

dM

d

d

(Zl′) ≤ (Lsυ2)ldM

###### (Z0′).

d

d

where s > 0 is the largest element of all singular values of all Wl′

, υ2 is the largest element of all singular values of all Wl′

1

2

and L is the Lipschitz constant of activation function σ(·).

The analysis from the prior proofs reveals that the diversity of MLP modules is constituted by two elements: the eigenvalues of parameters and the Lipschitz constant of the activation functions. In neural networks, parameters are typically normalized, which means the maximum eigenvalues of these parameters are bounded. Furthermore, the parameters in a neural network are learned through backpropagation. Given these factors, it becomes challenging to impose restrictions on the eigenvalues of parameters. Consequently, the activation functions in the MLP emerge as the most crucial aspect of their nonlinear expressive capabilities.

|…<br><br>| |
|---|
<br><br>|
|---|

|…<br><br>|
|---|

| |
|---|

| |
|---|

|| |
|---|
|
|---|

| |
|---|

Multi-head

| |Self-Attention<br><br>(MSA)<br><br>Shortcut|
|---|---|
| |…<br><br>Aug-S|

| |
|---|

| |
|---|

…

…

| |
|---|

| |
|---|

Input tokens

|| |
|---|
|
|---|

| |
|---|

|…<br><br>|
|---|

|…<br><br>|
|---|

| |
|---|

| |
|---|

…

…

…

| |
|---|

| |
|---|

Output tokens

| |
|---|

| |
|---|

- Fig. 3: The diagram of MSA module equipped with augmented shortcuts, where different patterns (rectangle, triangle, etc.) denote different features from various tokens. The original identity shortcut copies the input feature while the augmented shortcuts (Aug-S) project features of each input token to diverse representations.
- 4 PANGU-π MODULES AND ARCHITECTURES

In this section, we first propose the series informed activation function to enhance the nonlinearity of the MLP module. Then, we introduce the augmented shortcuts to improve MSA modules in the Transformer architecture. Finally, we prove that the combination of the two techniques results in a novel and stronger Transformer model.

###### 4.1 Augmented Shortcut

As discussed in Section 3, a pure attention suffer serve a feraure collapse problem. The typical LLM architecture only equips each MSA module with a single shortcut connection, which is an identity projection and directly copies the input features to the outputs. This simple formulation may not have enough representation capacity to improve the feature diversity maximally. We aim to refine the existing shortcut connections in vision Transformers and explore efficient but powerful augmented shortcuts to produce visual features with higher diversity.

We propose augmented shortcuts to alleviate the feature collapse problem by paralleling the original identity shortcut with more parameterized projections. The MSA module equipped with T augmented shortcuts can be formulated as:

T

Tli(Zl;Θli), l ∈ [1,2,··· ,L],

AugMSA(Zl) = MSA(Zl) + Zl +

(4)

i=1

where Tli(·) is the i-th augmented shortcut connection of the l-th layer and Θli denotes its parameters. In addition to the original shortcut, the augmented shortcuts provide more alternative paths to bypass the attention mechanism. Different from the identity projection that directly copies the input tokens to the corresponding outputs, the parameterized projection Tli(·) transforms input features into another feature space. Projections Tli(·) will apply different transformations to the input feature as long as their weight matrices Θli are different, and thus paralleling more augmented shortcuts has the potential to enrich the feature space.

A simple formulation for Tli(·) is the sequence of a linear projection and an activation function i.e.,

###### Tli(Zl;Θli) = σ(ZlΘli), l ∈ [1,··· ,L], i ∈ [1,2,··· ,T],

(5)

where Θli ∈ Rd×d is the weight matrix and σ is the nonlinear activation function (e.g., GELU). In Eq. 5, Tli(·) tackles each

token independently and preserves their specificity, which is complementary to the MSA modules aggregating different tokens. Note that the identity mapping is a special case of Eq. 5, i.e., σ(x) = x and Θli is the identity matrix.

This indicates that the upper bound of feature diversity

(Zl) decreases dramatically as the network depth increases without shortcut connections. We now analyze how the diversity

dM

d

(Zl) changes w.r.t. the layer l in the model stacked by the AugMSA modules. We have the following theorem. Theorem 4. Given a model stacked by the AugMSA modules, the diversity dM

dM

d

(Zl) of feature in the l-th layer can be bounded by that of input data Z0, i.e.,

d

T

√

dMd(AugMSA(Zl)) ≤ (

L∥Θli∥2)dMd(Zl),

λHsυ1 + 1 +

i=1

T

√

L∥Θli∥2)ldMd(Z0),

dMd(Zl) ≤ (

λHsυ1 + 1 +

i=1

where H is number of heads, s > 0 is the largest element of all singular values of all Wl, and ∥ · ∥2 is the ℓ2 norm of the matrix.

###### √

λHsυ1+1+ Ti=1 L∥Θli∥2) > 1, this allows us to prevent feature collapse.

Since αi = (

Compared with Theorem 2, the augmented shortcuts introduce an extra term (

###### √

λHsυ1+1+ Ti=1 L∥Θli∥2)l, which increases exponentially. This tends to suppress the diversity decay incurred by the attention mechanism. The term αi (0 ≤ i ≤ l) is determined by the norms of weight matrices Θli of the augmented shortcuts in the i-th layer, and the bound of diversity dM

(Zl) in the l-th layer can be affected by all the augmented shortcuts in the previous layers. For the ShortcutMSA module with only an identity shortcut, we have αi =

d

###### √

λHsυ1 + 1. Adding more augmented shortcuts can increase the magnitude of αi, which further improves the bound.

As discussed above, paralleling multiple augmented shortcuts with the MSA and MLP modules in a vision Transformer can improve the feature diversity to achieve higher performance. However, directly implementing Tli(·) (Eq. 5) involves a lot of matrix multiplications that are computationally expensive. For example, given feature Zl ∈ Rn×d and weight matrix Θli ∈ Rd×d, the matrix multiplication ZlΘli consumes nd2 FLOPs, where d is usually large in vision Transformers (e.g., 4096 in LLaMA-7B). In [104], the augmented shortcut is implemented with a blockcirculant matrix, which realizes fast inference with fast Fourier transformation (FFT). Even though it achieves high theoretical acceleration, we empirically find that its practical speed depends on the hardware optimization. Considering that LLM is a universal model, we propose to implement the augmented shortcut with a simpler bottleneck module. The module is constructed by stacking two FC layers with a nonlinear activation function (e.g., GeLU). The first FC layer reduces the d-dimension feature into a lowdimension space by a reduction ratio r and the second FC layer restores the original feature dimension. Then the computational cost is reduced to 2nd2/r FLOPs . A larger r implies a further decrease in the computational cost. For example, when the reduction ratio r is set to 32, the computational cost can be reduced by 16× compared to the original augmented shortcut (Eq. 5).

Obviously, shortcuts use a large weight identity branch, such as

AugMSA(Zl) = MSA(Zl) + αZl( Ti=1 Θli) (where α > 0), to prevent feature collapse, but this reduces network performance.

We theoretically analyze this because feature diversity dM

(Zl)

d

is adding excessive noise. The effect of noise on feature diversity is usually false positive. For example, if H = 0, then dM

(H) = 0. However, when the input matrix introduces a zeroaverage noise ϵ, then dM

d

(H) = 0. This requires us to improve the diversity features and minimize the impact of noise diversity on the network. This means reducing the value of |dM

(H + ϵ) = ∥ϵ∥F > dM

d

d

(AugMSA(Zl))| while ensuring that dM

(AugMSA(Zl + ϵ)) − dM

d

d

(Zl) is not attenuated.

d

The following describes the definition of feature diversity of noise. We consider the effect of noise on matrix projection

##### (H).

dM

d

(H)| ≤ ∥ϵ − 1(xHmin+ϵ − xHmin)⊤∥F ≤dM

|dM

(H + ϵ) − dM

d

d

(ϵ) = ∥(I − ee⊤)ϵ∥F = ∥ϵ − 1xϵmin⊤∥F ≤ ∥ϵ∥F.

d

(6)

For zero average noise, the following equation holds ee⊤ϵ = 0 ∈ RN×d, so that the above inequality is equal. Since |dM

(f(H +ϵ)−f(H)). We define dM

(f(H +ϵ))−dM

(f(H))| ≤ dM

d

d

d

(f(H + ϵ) − f(H)) to represent the diversity effect of noise ϵ on the f function whose input is H. The smaller the value, the higher the robustness of the function f. For the sake of simplicity and considering typical scenarios, the following discussion assumes that the input noise ϵ is zero average noise.

d

- Lemma 3. We consider the impact of noise on the MSA module, when H = 1. For a slight perturbation of the input ϵ, the selfattention matrix also produces a perturbation Aϵ = A + δ, i.e.,

dM

d

(MSA(Zl + ϵ) − MSA(Zl)) ≤ λA+δsυ1∥ϵ∥F + λδsυ1dM

d

(Zl),

where λA+δ is the largest eigenvalue of Aϵ⊤(I − ee⊤)Aϵ and λδ is the largest eigenvalue of δ⊤(I−ee⊤)δ, usually λA+δ < 1 and λδ < 1.

For the H heads MSA module, we can get the following formula:

dM

d

(MSA(Zl + ϵ) − MSA(Zl)) ≤ λA+δHsυ1∥ϵ∥F + λδHsυ1dM

d

(Zl),

- Lemma 4. We consider the noise diversity of linear parallel branch:

(L(Zl + ϵ)Θli − LZlΘli) ≤ L∥Θli∥2∥ϵ∥F,

dM

d

Theorem 5. If and only if ee⊤(σ(Zl + ϵ) − σ(Zl)) = 0, the following inequality is equal:

(Tli(Zl + ϵ;Θli) − Tli(Zl;Θli)) ≤ L∥Θli∥2∥ϵ∥F.

dM

d

For the nonlinear activation function, σ(Zl + ϵ) − σ(Zl) is no longer guaranteed to be zero-average. Therefore, the noise diversity of the nonlinear branch is weaker than that of the linear branch:

(Tli(Zl + ϵ;Θli) − Tli(Zl;Θli)) < L∥Θli∥2∥ϵ∥F.

dM

d

Theorem 6. Given a model stacked by the AugMSA modules, the noise diversity of feature in the l-th layer can be bounded by the following formula, i.e.,

dMd(AugMSA(Zl + ϵ) − AugMSA(Zl))

T

∥Θli∥2)∥ϵ∥F + λδHsυ1dMd(Zl)

< (1 + λA+δHsυ1 + L

i=1

This indicates that using a large number of nonlinear shortcuts

instead of LZl( Ti=1 Θli) prevents feature collapse, reduces the impact of input noise on feature diversity, and enhances

the robustness of the network. In addition, it also enhances the nonlinear expression ability.

###### 4.2 Series Informed Activation Function

A neural network Nd composed of d hidden layers can be regarded as a composite of d functions fi: Nd = f1 ◦ f2 ◦ ··· ◦ fd. In particular, each hidden layer function fi can be written as the composite of a function gi and an activation function σi: fi = σi ◦ gi. Actually, the learning procedure of fi amounts to an optimization problem over the layer hypothesis space Hi.

Usually, ϕi is taken as a non-learnable function; therefore, in the most common scenario, Hi = σi × Hg

i. gi is parameterized and learnable, and belongs to a hypothesis space Hg

i. This clearly limits the learnable space.

In this section, we introduce a technique to define learnable activation functions that could be plugged into all the hidden layers of MLP. We define the hypothesis space Hϕ

i, based on the following idea: (i) select a finite set of activation functions Σ := {σ1,··· ,σN}, whose elements will be used as base elements; (ii) define the learnable activation function ϕi as a linear combination of the elements of Σ; (iii) identify a suitable hypothesis space Hϕ

i; (iv) optimize the whole network, where the hypothesis space of each hidden layer is Hi = Hϕ

i. In this way, we expand the learnable space of each hidden layer, enhancing the model’s nonlinear expression capability.

i × Hg

Several different activation functions have been proposed for deep neural networks, including the most popular Rectified Linear Unit (ReLU) and its variants (PReLU [101], GeLU [102] and Swish [103]). They focus on enhancing the performance of deep and complex networks using different activation functions. However, as theoretically proven in the preceding section, the limited power of Transformer architecture is mainly due to poor nonlinearity, which has not been fully investigated by the existing activation functions.

There are two ways to improve the nonlinearity of a neural network: stacking the nonlinear activation layers or increasing the nonlinearity of each activation layer. The trend of existing networks is to choose the former, which results in high latency when there is excessive parallel computation ability.

One straightforward idea to improve the nonlinearity of the activation layer is stacking. The serial stacking of the activation function is the key idea of deep networks. However, stacking layers serially results in a large computation cost, which is not affordable for developing an efficient and effective LLM. Therefore, we choose concurrently stacking the activation function. Denote there are n activation functions for input x in a neural network as {σi(x)}ni=1, which can be the usual functions such ReLU and Tanh. The concurrent stacking of the activation functions can be formulated as: n

σi(aix + bi), (7)

i=1

where n denotes the number of stacked activation functions and ai,bi are the scale and bias (which are learned parameters) of each activation to prevent simple accumulation. The nonlinearity of the activation function can be largely enhanced by concurrent stacking. Equation 7 can be regarded as a series in mathematics, which is the operation of adding many quantities.

Since the nonlinearity of the Transformer is mainly derived from the feed-forward network (FFN), we apply the series informed activation function on the FFN block. Given an input feature x ∈ RN×D, where N and D are the number of tokens and its hidden dimension, the original FFN can be formulated as

##### )Wl′

MLP(Zl′) = σ(Zl′Wl′

, l ∈ [1,2,··· ,L], (8) where Wl′

2i

1

and Wl′

are two fully connected layers. Specifically, to further enrich the approximation ability of the series, we enable the series-based function to learn the global information by varying the inputs from their neighbors, which can be reformulated as:

2

1

n

SIAF−MLP(Zl′) = (

σi(Zl′Wl′

))Wl′

, l ∈ [1,2,··· ,L].

1i

2i

i=1

(9) It is easy to see that when n = 1, the series-based activation function σs(x) degenerates to the plain activation function σ(x), which means that the proposed method can be regarded as a general extension of existing activation functions.

- Theorem 7. Given a model stacked by the SIAF−MLP modules,

the diversity dM(Zl′) of feature in the l-th layer can be bounded by that of input data Z0′, i.e.

dM

d

(SIAF − MLP(Zl′)) ≤ (

n

i=1

Li)sυ2dM

d

(Zl′),

dM

d

(Zl′) ≤ (sυ2

n

i=1

Li)ldM

d

(Z0′),

where Li is the Lipschitz constant of the activation function σi.

As demonstrated in our proof, the Series Informed Activation Function (SIAF) we propose markedly amplifies the nonlinear expressive capacity of the MLP module compared to the original architecture. This enhancement in nonlinearity progressively intensifies with the increase of the parameter n.

- 4.3 Combination

Finally, we offer the upper bounds for the combination of multilayer AugMSA module and SIAF-MLP module to decay into subspace Md. We can obtain the upper bounds for the combination of multi-layer MSA module and MLP module to decay into subspace Md in vanilla Transformer architecture.

- Theorem 8. Provide a network consisting of p-layer MSA module

and q-layer MLP module, the diversity dM

(Zp+q) of feature in the l-th layer can be bounded by that of input data Z0, i.e.,

d

√

λHsυ1)p(Lsυ2)qdM

(Z0). (10)

(Zp+q) ≤ (

dM

d

d

It is evident that the original Transformer architecture possesses a relatively limited upper bound in terms of nonlinear expressive capability. Building upon this observation, we now proceed to analyze the enhanced expressive power of the Transformer when augmented with our proposed architectural modifications.

Theorem 9. Provide a network consisting of p-layer AugMSA module and q-layer SIAF-MLP module, the diversity dM

(Zp+q) of feature in the l-th layer can be bounded by that of input data Z0, i.e.,

d

dM

(Zl) ≤ (

d

T

n

√

###### (Z0′).

Li)qdM

L∥Θli∥2)p(sυ2

λHsυ1 + 1 +

d

i=1

i=1

The preceding theorem substantiates that our proposed augmented shortcut module, when used in conjunction with the series informed activation function module, enhances the model’s nonlinear expressive capabilities and diversity far beyond what is achievable by using either module independently. Consequently, our Transformer architecture amalgamates these two modules to form our PanGu-π architecture, resulting in a synergistic improvement in nonlinearity and diversity.

#### 5 EXPERIMENTS ON GENERAL FIELD

In this section, we compare the existing open-source 7B and 1B models. Furthermore, we conduct ablation studies of the proposed architecture.

Training data The pre-training data is gathered from diverse sources from the Internet, covering English and Chinese corpus in an equal 1 : 1 ratio. The tokenizer is built by byte-pair encoding (BPE) [105] from SentencePiece [106] upon our data. The final vocabulary size is about 0.1 million. After tokenization, the entire training dataset contains about 1.6 trillion tokens.

Training details Our models are trained using the AdamW optimizer [107] with β1 = 0.9,β2 = 0.95 for 1 epoch utilizing the cosine learning rate decay [108] with an initial learning rate 3 × 10−4. The total batch size for the training process is approximately 4M, and it includes a warm-up phase spanning 4000 steps.

Model details For fair comparison, we adopt the prenormalization [109], SwiGLU activation [52] and rotary embeddings [110] following the LLaMA architecture [23]. We then apply our series activation function and augmented shortcuts to build our models. The details of the models can be found in Table 1. We reduce the number of layers to make the number of parameters similar to the LLaMA model for fair comparison because the proposed modules introduce extra parameters.

Training Devices We use the Huawei Ascend 910A card to train and evaluate the proposed architecture. The HUAWEI Ascend 910A is a high-efficiency, flexible, and programmable artificial intelligence processor. For half-precision floating-point (FP16) operations, the Ascend 910 delivers 256 TeraFLOPS. For integer precision calculations (INT8), it delivers 512 TeraOPS. Despite its unparalleled performance, Ascend 910’s maximum power consumption is only 310W, which is significantly lower than its planned specifications of 350W. Developed using Huawei’s proprietary Da Vinci architecture, it integrates a rich array of computing units, enhancing the completeness and efficiency of AI computations, thereby extending its applicability. It significantly improves the performance of the entire AI system and effectively reduces deployment costs.

Benchmarks We use the OpenCompass platform [111] to evaluate on an extensive suite of downstream tasks. We selected 11 classical benchmarks from four different domains to conduct

TABLE 1: Model details of PanGu-π.

Models dimension n heads n layers

PanGu-π-1B 2048 16 12 PanGu-π-7B 4096 32 29

a comprehensive comparison. C-Eval [112] is a comprehensive Chinese evaluation benchmark to evaluate the knowledge and reasoning abilities of LLMs, which includes multiple-choice questions from 52 diverse disciplines across different difficulty levels. CMMLU [113] is also a comprehensive Chinese evaluation benchmark, which covers 67 topics including science, engineering, and humanities. MMLU [114] proposes an English evaluation benchmark for measuring LLM’s multitask accuracy by covering 57 tasks including mathematics, history, computer science, and law. AGI-Eval [115] is a human-centric benchmark specifically designed to evaluate the general abilities of foundation models in tasks pertinent to human cognition and problem-solving. BoolQ [116] is a reading comprehension dataset to evaluate the difficult entailment-like inference ability of LLMs. AX-b [117] is a broad-coverage diagnostic task and PIQA [118] is a physical interaction question-answering task. CSL [119] offers a Chinese Scientific Literature dataset to evaluate the performance of models across scientific domain tasks. EPRSTM [120] is a binary sentiment analysis dataset based on product reviews on an ecommerce platform. [121] is a single-document summarization task, and LCSTS [122] is a large corpus of Chinese short-text summarization datasets.

###### 5.1 Ablation Studies

To better understand the proposed Architecture, we conduct extensive experiments to investigate the impact of each component. All the ablation experiments are conducted based on the 1B model size.

TABLE 2: Ablation Study about series informed activation function.

Number of n C-Eval Inference Speed (ms)

- 1 35.0 7.56

- 2 36.9 7.76

- 3 37.0 8.02

- 4 37.0 8.35

Influence of series informed activation function. In the above section, we propose the SIAF to enhance the performance and enable global information exchange in feature maps. Table 2 shows the performance of the proposed SIAF using different numbers of n. When n = 1, the activation function degenerates into the plain activation function. We find that when n = 2, the performance and the inference speed strike an optimal balance. Therefore, we choose n = 2 for the following experiments.

Influence of augmented shortcuts. As mentioned above, the augment module can greatly improve the performance. As shown in Table 3, we trade off accuracy and speed by controlling the width of the bottleneck middle layer. By transforming the reduction rate, it is apparent that an increase in reduction rate results in a decrease in calculation speed and accuracy. After careful consideration, we determined a reduction rate of 32 to achieve the optimal balance between speed and accuracy.

TABLE 3: Influence of the width factor in augmented shortcuts.

Width factor of bottlenecks C-Eval Inference Speed (ms) 1 36.5 8.66

1/4 36.7 8.06 1/16 36.5 7.96 1/32 36.3 7.66 1/64 35.5 7.62

0 35.0 7.56

TABLE 4: Ablation Study of each component.

Method C-Eval Inference Speed (ms) Vanilla Transformer 35.0 7.56

WideNet [123] 36.5 8.06 SIAF 36.9 7.76

AS 36.3 7.66 SIAF+AS 37.9 7.86

Architecture We finally ablate the effectiveness of each component of the proposed method and report the language modeling results in Table 4. We ablate the series informed activation function (SIAF) and augmented shortcuts (AS) as described earlier. Furthermore, we compared the proposed method with WideNet [123], which is introduced to increase the nonlinearity of Transformer architectures. It can be seen through experiments that each component of the proposed method is effective for improving the performance of the Transformer architecture, which surpasses that of WideNet [123].

###### 5.2 Feature Analysis and Visualization

We also analyze the latent representation across different layers to demonstrate further the superiority of nonlinearity compensation introduced by the PanGu-π architecture. We are interested in the channel diversity of each architectural choice. Following the analysis method from [124], we characterize the effective dimensions of different decoder architectures. In particular, the effective dimension d(ϵ) is defined as the minimum principal component numbers that occupy the explained variance ratio of ϵ in a principal component analysis (PCA). In principle, a more powerful representation of individual tokens would result in a larger effective dimension. In comparison, a smaller effective dimension means the variance in token representation occurs mainly in smaller dimensions. As shown in Fig. 4, here we report each layer’s effective dimension d(0.8). Removing all augmented shortcuts limits the effective dimension to the greatest extent, and removing the series informed activation function significantly reduced the effective dimension consistently on each Transformer layer, indicating the significant role of these components in channel-wise feature diversity [125].

Furthermore, to offer a finer-grained characterization of linguistic features from different architectures, we also visualize the representation of tokens with different semantics, using the test set of Penn Tree Bank (PTB) [126] as a general domain corpus. In particular, we adopt a layer-wise visualization method to illustrate the concentration and diversity of features for each token and how these characteristics change along the Transformer layers Fig. 5. To assist the visualization, the top five frequent tokens are highlighted with different colors. PCA is used to reduce all feature maps to a 3D space, preventing nonlinear reduction as it may cause disruption. Additionally, the total variance accounted

2000

Pangu-π-7B w/o SIAF w/o AS

1800

1600

1400

1200

1000

800

600

400

200

0

1 6 11 16 21 26

- Fig. 4: The effective dimension d(0.8) across layers of different model architectures. A larger number of effective dimensions means more principal components are needed to account for 80% of variance, indicating more diversity in feature channels.

for by the first three principal components is labeled for each layer. From Fig. 5, we can see that PanGu-π architecture possesses the most diverse and isotropic feature space [127], with each token’s feature expanding to a high-dimensional cluster when moving to deeper layers. In comparison, removing the series informed activation function or the augmented shortcuts limit the feature on low-dimensional manifolds (i.e., aggregating along one axis in the middle panel), and the differentiation among tokens is also blurred, indicating less discriminative power in language modeling.

Layer 16 Layer 21 Layer 26

PanGu-π-7B

w/o SIAF

w/o AS

[Figure 35]

- Fig. 5: Visualization of hidden states from each layer. The most frequent five tokens are highlighted using different colors for visualization. The total variance accounted for by the first three principal components is labeled for each layer on the top. Note that the beginning tokens are removed from the analysis because they are considered outliers.

To verify the effectiveness of the language representation enhanced by PanGu-π architecture, we conduct case analyses where the saliency of each token’s feature dimension is calculated by deriving the absoluate value of the corresponding gradients with respect to the prediction target. As shown in Figure 6, the language model is required to echo the previous mentioned name “chester” as the next word. The PanGu-π model correctly identifies the key message “chester” in the context reflected by higher gradient values for most of the channels (Figure 6 (a)). In comparison, without the augmented shortcuts and series activation

0 125 250 375 500 625 750 875

0 125 250 375 500 625 750 875

0.8

0.8

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

0.7

0.7

1000 1125 1250 1375 1500 1625 1750 1875 2000 2125 2250 2375 2500 2625 2750 2875 3000 3125 3250 3375 3500 3625 3750 3875 4000

1000 1125 1250 1375 1500 1625 1750 1875 2000 2125 2250 2375 2500 2625 2750 2875 3000 3125 3250 3375 3500 3625 3750 3875 4000

0.6

0.6

0.5

0.5

0.4

0.4

0.3

0.3

0.2

0.2

0.1

0.1

0.0

0.0

to

to

to

to

``

``

in

in

?

?

s

s

''

''

head

head

shot

shot

happened

the

happened

happened

the

happened

che

che

what

what

what

what

ster

ster

(a) PanGu-π (b) Vanilla Transformer

Fig. 6: Saliency analysis for next word prediction task. Color brightness indicates the absoluate value of the corresponding gradients with respect to the prediction target. The gradient values are normalzied within each sample to ensure a fair comparison. Important messages are framed in red.

function, the model tend to leverage more information from the unmeaningful symbols after the key hints, leading to a wrong prediction that directly ends the sentence (Figure 6 (b)).

###### 5.3 Comparison with 7B Models

To showcase the competitive edge of PanGu-π, we conducted an extensive analysis of the PanGu-π model, comparing it against other state-of-the-art models with a similar size. The comprehensive results are detailed in Table 5. We segmented the comparative datasets into four tasks: examination, knowledge, reasoning, and understanding, to fully assess the capabilities of our model. Notably, in the examination category, our model almost reached the state-of-the-art (SOTA) benchmarks, surpassing LLaMA2, Baichuan2, and InternLM, and closely matching the current top performer, Qwen. The trend of superior performance in reasoning and understanding categories can also be observed, which is attributed to the PanGu-π model’s advanced nonlinear capability. This allows it to fit more complex function spaces, giving it a competitive advantage in these challenging SOTA benchmarks.

In the knowledge dataset, our model still lags behind in performance when compared to the existing SOTA benchmarks. This could be due to the lower concentration of knowledge in our collected dataset and the presence of many unseen data points in the BoolQ dataset. Nevertheless, our model achieved excellent results, demonstrating strong generalization abilities.

Overall, our model exhibits consistently better average performance indices compared to the current state-of-the-art models. When measured against other open-sourced models of a 7B size, PanGu-π achieves significantly higher average performance. In the future, we plan to train our model with better data, aiming to enhance its performance metrics in the knowledge domain.

Moreover, we evaluated the latency (milliseconds per token) of these models. As the compared models utilize a similar architecture to LLaMA, their latencies are comparable. Our findings indicate that PanGu-π achieves a much faster inference speed compared to the LLaMA architecture, further establishing its superiority.

The PanGu-π model can serve as a swift and effective foundational model, leveraging strategies such as Supervised Fine-

- TABLE 5: Comparison with SOTA open-source 7B models. The best model is listed in bold. Models

Task Dataset LLaMA2-7B Baichuan2-7B InternLM-7B Qwen-7B PanGu-π-7B

Examination

C-Eval 32.50 56.90 53.21 63.40 59.86 CMMLU 31.80 57.02 51.86 62.50 59.92

MMLU 46.80 54.72 51.39 59.70 61.91

AGI-Eval 21.80 34.77 37.77 45.30 54.16 Knowledge BoolQ 74.90 63.21 64.10 76.10 64.77 Reasoning

AX-b 53.50 51.72 42.57 57.00 57.52 PIQA 78.30 76.22 78.02 77.90 77.15

Understanding

CSL 55.60 66.25 65.62 56.20 63.75 EPRSTMT 46.20 69.38 88.12 88.80 90.62

XSum 19.70 20.89 8.12 1.30 19.58 LCSTS 9.10 15.57 18.19 12.00 16.62

Average 42.75 51.19 50.82 54.56 56.90 Latency on 910A (ms) 47.60 43.00

- TABLE 6: Comparison with SOTA open-source 1B models. The best model is listed in bold.

Models Task Dataset Sheared-LLaMA-1.3B Chinese-LLaMA2-1.3B TinyLLaMA-1.1B PanGu-π-1B

C-Eval 24.28 28.70 27.85 36.85 CMMLU 25.10 24.78 24.64 35.90

Examination

MMLU 25.77 24.55 25.75 35.96

AGI-Eval 18.01 19.40 18.54 30.77 Knowledge BoolQ 62.39 56.79 56.06 58.44 Reasoning

AX-b 43.57 47.46 45.47 43.48 PIQA 72.91 56.91 70.62 61.92

CSL 51.88 55.60 53.12 55.62 EPRSTMT 46.25 72.50 46.25 55.62

Understanding

XSum 16.44 8.90 20.15 15.92 LCSTS 15.37 13.16 13.97 14.61

Average 36.54 37.16 36.58 40.46

Tuning (SFT), AI Agent, or retrieval augmentation to become a competitive AI solution. It has the potential to harness the power of LLMs in edge devices like smartphones, offering a compelling choice for various applications.

###### 5.4 Comparison with 1B Models

For comparison, we have meticulously selected three SOTA models of similar size, all based on the LLaMA architecture. These include Chinese-LLaMA2-1.3B [128], TinyLlama-1.1B [129], and Sheared-LLaMA-1.3B [130]. Notably, Sheared-LLaMA-1.3B was initially pruned from a larger LLaMA2-7B model and subsequently trained with a condensed dataset of 50B tokens. Our extensive experiments, as presented in Table 6, demonstrate that PanGu-π-1B significantly surpasses existing LLMs of similar size, and in some cases, even larger sizes.

Similar to our findings with the 7B model, our 1B model demonstrates a remarkable superiority in the examination category over existing models, affirming the efficacy of our modifications even at the 1B scale. This outcome is a testament to our model’s enhanced nonlinear capabilities, enabling it to fit more complex function spaces. In the realms of reasoning and understanding, we observe similar trends, further validating our model’s robust performance.

However, as with our larger model, the knowledge density in our collected dataset for training the 1B model was relatively low.

Consequently, in datasets like BoolQ, which contain data points previously unseen by our model, we acknowledge a noticeable gap in performance when compared to current state-of-the-art benchmarks. This indicates room for improvement in our model’s ability to handle unfamiliar data, a challenge we aim to address in future iterations.

Furthermore, when assessing latency, a critical factor in realworld applications, PanGu-π-1B shows a marked advantage. It records lower latency times of 13.8 ms on the 910A, compared to its deeper counterpart, LLaMA2-1B, which clocks 15.4 ms on the 910A, despite having a roughly equivalent number of parameters. This not only underscores the efficiency of PanGuπ-1B but also highlights its suitability for deployment in timesensitive applications.

#### 6 YUNSHAN: DOMAIN SPECIALIZED MODEL FOR FINANCE AND LEGAL FIELDS

In the current landscape of LLMs, there is a growing trend towards using domain-specific models for specialized applications, which differ from the capabilities of general-purpose large models. These domain-specific models are developed with a deeper level of expertise, and are particularly adept at addressing tasks in specific fields. For instance, in the finance industry, LLMs are used to provide in-depth analysis of financial documents and offer interactive information services for market insights and queries.

TABLE 7: The vocab size and text compression rate of YunShan’s tokenizer compared with other tokenizers of LLaMA2, InternLM, Baichuan2 and PanGu-π.

Tokenizer Vocab Size Compression Rate

LLaMA2 32,000 1.479 InternLM 103,168 0.657 Baichuan2 125,696 0.605

PanGu 100,883 0.649 YunShan 110,428 0.606

Similarly, in the legal field, LLMs are used to facilitate expedited case research and information retrieval. Exemplary models such as Xuanyuan [83], FinGPT [79] and FinMA [81] in financial services, and Lawyer LLaMA [93] and LawGPT [96] for legal application. This progression not only exemplifies the robust development within the sphere of domain-specific LLMs but also signals the future role these technologies are poised to play in furthering innovation and growth across various domains.

###### 6.1 Datasets

To enhance the model’s ability to solve domain-specific tasks, the model is continually pretrained and instruction finetuned on the domain datasets. We collect a variety of domain datasets both in the financial domain and the legal domain. When we construct the datasets, we consider the professionalism, diversity, data size and accessibility of the sources. We process and clean the collected datasets to construct high-quality domain corpora.

Financial Domain Our financial pretraining data consists of company announcements, financial news, articles, and examinations. Some of the data is from FinCorpus1, which is a high-quality Chinese financial dataset. The rest of the data is crawled through TuShare2, which is a Python package that provides free access to Chinese financial data from various sources. After cleaning, the total size of the data is 36.5B tokens.

Legal Domain Our legal pretraining data comprises legal regulations, legal case texts, academic papers, and legal examinations. We use Pile of Law [131] and LeXFiles [132] to construct some of the data, and selectively collect data from two legal-related public platforms34. After cleaning, the legal pretraining data contains 111.7B tokens.

Instruction-following Data For supervised fine-tuning, we collect 995k domain instruction-following data. The data consists of JECQA [133] and the instructions open sourced by ChatLaw [98], Lawyer LLaMA [93], LawGPT [96], and FinCorpus1, covering Judicial Examination examples, legal consultations, and financial examination examples.

###### 6.2 Tokenizer

Corpora from financial and legal fields often contain specialized words that are not included in the general domain vocabulary. During tokenization, these unknown words are deconstructed into sub-words or even UTF-8 bytes, which reduces model training efficiency. To alleviate the inefficiency, we expanded the original vocabulary size from 100883 to 110428 by adding a specialized

- 1. https://huggingface.co/datasets/Duxiaoman-DI/FinCorpus
- 2. https://tushare.pro/
- 3. https://pkulaw.com/
- 4. https://wenshu.court.gov.cn/

financial-legal word list. We used byte-pair encoding (BPE) [134] from Tokenizer5 to train our tokenizer. The specialized vocabulary is trained from a subset of the financial and legal corpora and then merged into the original vocabulary. Table 7 shows the compression rate we calculate on the JEC-QA dataset [135], a legal domain QA dataset. Our new tokenizer has a much lower compression ratio than the original one and ties for the best with Baichuan 2 while having a smaller vocabulary.

###### 6.3 Training Process

Further Pretraining To transfer financial and legal knowledge to the PanGu-π model, we further pre-train it with a specialized vertical corpus. To alleviate catastrophic forgetting of formerly learned knowledge, we partially introduce the during the pretraining process. Specifically, we resample a portion of the previous high-quality data and mix it with the vertical domain data at a 1 : 1 ratio.

Instruction Tuning During the instruction tuning phase, we utilize supervised fine-tuning samples of both general-purpose tasks and domain-specific tasks. We shuffle and combine them into one supervised dataset and execute the instruction tuning process in one stage. On the basis of the YunShan-base model, which has already acquired general and specialized knowledge during previous phase, this approach aims to simultaneously teach the model how to follow human instructions in different domains.

###### 6.4 Benchmarks

The proposed YunShan model is evaluated across the two specialized law and finance domains. This approach aims to comprehensively assess YunShan model’s capabilities in knowledge retention, comprehension, and application within domain-specific tasks, covering Chinese and English languages.

- 6.4.1 Financial Domain For the financial domain, we use two distinct datasets: FinanceIQ [136] and FinEval [137], each serving a unique purpose in our evaluation framework. FinanceIQ is a Chinese evaluation dataset focused on the financial domain, designed to assess the knowledge and reasoning capabilities of LLMs in financial scenarios. It spans 10 broad financial categories and 36 subcategories, offering a total of 7,173 multiple-choice questions. This dataset is tailored to evaluate a wide range of financial knowledge, from basic concepts to complex problem-solving in areas such as accounting, banking, and financial analysis. FinEval is an expansive benchmark tailored for measuring the adeptness of LLMs in the finance domain. Encompassing 4,661 multiple-choice questions, it spans 34 academic subjects across the finance, economics, accounting, and professional certification domains. This dataset is structured to facilitate a layered evaluation through diverse assessment methods, including zero-shot, fewshot, answer-only, and chain-of-thought prompts.
- 6.4.2 Legal Domain LawBench [138]: For the legal domain, we implement the test over LawBench dataset. LawBench is a sophisticated benchmark designed to evaluate the legal knowledge capabilities of LLMs within the Chinese civil-law system. This benchmark tests LLMs

###### 5. https://github.com/huggingface/tokenizers

TABLE 8: The experimental results of general-domain and domain specialized LLMs on financial FinancelQ [83] benchmark.

Model CPA BQ SPQ CFPQ IPQ Economist Tax Accountant FQ Financial Planner Actuary AVG

LLaMA2-7B 28.38 26.53 26.11 24.54 26.44 25.77 28.41 26.10 25.00 25.41 26.27 InternLM-7B 31.81 43.57 34.69 40.02 36.49 42.50 43.19 37.29 32.95 31.56 37.41 Baichuan2-7B 34.25 49.04 40.14 42.32 41.52 52.69 41.11 41.02 30.68 35.86 40.86

Qwen-7B 30.36 37.38 35.97 37.61 33.19 43.65 38.11 38.31 27.27 27.66 34.95

FinGPT-7B 25.32 25.88 24.40 24.89 25.57 25.58 27.48 20.68 22.73 29.51 25.20 FinMA-7B 24.94 29.66 33.25 29.93 35.34 32.31 26.56 26.78 22.73 25.00 28.65

LawGPT-7B 26.47 24.92 23.98 24.77 25.14 24.62 27.02 27.46 20.45 26.43 25.13

HanFei-7B 27.00 34.08 33.93 34.52 37.36 37.88 32.79 30.51 22.73 30.33 32.11 YunShan (Ours) 52.78 63.42 52.89 57.57 52.30 72.50 52.66 55.93 34.09 51.43 54.56

- TABLE 9: The experimental results of general-domain and domain specialized LLMs on the financial FinEval [137] benchmark.

Model Accounting Certificate Economy Finance AVG

LLaMA2-7B 37.56 34.13 34.30 28.85 33.71 InternLM-7B 49.51 52.99 42.03 53.11 50.13 Baichuan2-7B 53.11 56.89 50.24 59.67 55.43

Qwen-7B 62.30 61.38 56.04 61.00 60.56

FinGPT-7B 29.18 25.45 30.90 25.90 27.54 FinMA-7B 35.08 32.34 34.78 35.41 34.32

LawGPT-7B 25.25 31.74 28.50 27.54 28.32

HanFei-7B 29.84 29.64 27.54 31.48 29.80 YunShan (Ours) 65.57 69.46 51.21 55.08 61.34

across three cognitive levels: Memorization, Understanding, and Applying. It includes 20 tasks across five types: single-label classification, multi-label classification, regression, extraction, and generation. With over 500 cases, LawBench employs metrics including accuracy, F1, and ROUGE-L to ensure a comprehensive assessment of LLMs in diverse legal applications.

Through this diverse and challenging array of datasets, our model is subjected to an extensive evaluation, covering a range of tasks and scenarios in law and finance. This setup not only tests the model’s domain-specific expertise but also its adaptability and versatility across different language settings.

###### 6.5 Comparison with other domain-specific models

For financial task, FinGPT [79] and FinMA [81] are adopted. Specifically, FinGPT v3.2, which uses LLaMA2-7B as a base model, and FinMA v0.1 NLP 7B version are adopted for a fair comparison. For legal tasks, we conduct experiments compared with Hanfei [92] and LaWGPT [10]. Specifically, we chose the Legal-Base-7B version for LaWGPT, which is trained based on Chinese-LLaMA-7B [128].

6.5.1 Results on the Financial Benchmark

Results on FinancelQ. In Table 8, we first compare the general domain and domain- specialized LLMs on the FinancelQ benchmark. These models were assessed across multiple financial professional sub-domains, including Certified Public Accountant (CPA), Banking Qualification (BQ), Stock Portfolio Query (SPQ), Certified Financial Planner Qualification (CFPQ), Investment Portfolio Query (IPQ), Economist, Tax Accountant, Finance Qualification (FQ), Financial Planner, and Actuary. The ”AVG” column represents the average score across all these subdomains. The YunShan model demonstrated exceptional performance across nearly all sub-domains, particularly excelling in the Economist, Banking Qualification, and Certified Financial Planner Qualification categories, with scores of 72.50, 63.42, and

57.57, respectively. These scores significantly surpassed those of other models, highlighting its efficiency in handling financerelated tasks. Its average score of 54.56 was markedly higher than its counterparts, indicating its broad adaptability and proficiency in the financial domain. While models like Baichuan2-7B showed good performance in certain sub-domains such as Banking Qualification, Investment Portfolio Query, and Certified Financial Planner Qualification, they fell short in others. This suggests that while they are adaptable to specific financial sub-domains, they do not match the overall proficiency of the YunShan model. Conversely, other models such as FinGPT-7B and FinMA-7B exhibited generally weaker performance across all sub-domains, with average scores not exceeding 35. This may indicate a lesser degree of specialization in handling financial domain knowledge compared to other models. Overall, the exceptional performance of the YunShan model in this financial domain-specific evaluation reflects its outstanding ability in understanding and processing financial professional knowledge and showcases its potential as an LLM for the financial industry.

Results on FinEval. We also conducted an experiment on the FinEval benchmark in Table 9. The significant advantages of the YunShan model still exist for the FinEval benchmark. As shown in

- Table 9, models were assessed across four different sub-domains: Accounting, Certificate, Economy, and Finance, with the ”AVG” column indicating the average score across all sub-domains. Specifically, we computed the average score by AVG = (305 × Accounting + 334 × Certificate + 207 × Economy + 305 × Finance)/1151 according to the FinEval [137]. It’s noteworthy that the YunShan model excelled in Certificate and Accounting, scoring 69.46 and 65.57 respectively, significantly outperforming other models. Its leading average score of 61.34 highlights its proficiency in handling finance-related tasks. Comparatively, Qwen-7B and Baichuan2-7B demonstrated robust performance, especially in the Finance sub-domain, with scores of 61.00 and 59.67 respectively. Their overall average scores, 60.56 and 55.43, indicate their strong adaptability to the financial sector. In contrast, models like FinGPT-7B, FinMA-7B, LawGPT-7B, and HanFei-7B showed relatively weaker performances across all sub-domains, with average scores below 35. This suggests that these models may be less specialized or inadequately trained for tasks specific to the financial domain. Overall, the YunShan model stands out as the most suitable model for financial domain tasks.

6.5.2 Results on Legal Benchmark

- Table 10 shows the overall zero-shot results of each model in the legal field on the different categories of LawBench [139], for three key aspects: Memorization, Understanding, and Applying. The average score is computed by AVG = (2×Memorization+

- TABLE 10: The experimental results of general-domain and domain specialized LLMs on the legal benchmark.

Model Memorization Understanding Applying AVG

LLaMA2-7B 1.52 9.09 16.19 11.16 InternLM-7B 3.10 8.29 29.76 16.36 Baichuan2-7B 2.78 8.13 19.79 12.26

Qwen-7B 0.82 17.43 31.54 21.41

FinGPT-7B 0.00 0.28 0.00 0.11 FinMA-7B 0.88 6.43 14.10 8.94

LawGPT-7B 1.69 3.41 19.64 9.73

HanFei-7B 13.71 6.91 25.22 14.92 YunShan (Ours) 44.11 19.72 43.70 31.75

10×Understanding+8×Applying)/20. Notably, the YunShan model yet again outperforms with its impressive scores, with the highest average score being 31.75. This performance indicates a superior capability in recalling legal information but also in understanding and applying it effectively, a crucial factor in legal contexts. Other models like Qwen-7B also demonstrate strong performances, with average scores of 21.35 and 21.41, respectively. In contrast, models such as FinGPT-7B exhibit significantly lower scores across all dimensions, leading to an overall average score of 0.11. Other models like InternLM-7B, Baichuan2-7B, and HanFei-7B demonstrate varied performances, with HanFei-7B scoring high in Memorization (13.71) but lower in Understanding (6.91), reflecting a potential imbalance in its skillset. Overall, the YunShan model demonstrates exceptional capability in the legal domain, outperforming others across all tested skills. This suggests that the YunShan model is adept at recalling legal information and excels in comprehending and applying it, making it a highly effective tool for legal applications.

#### 7 CONCLUSIONS AND DISCUSSIONS

In this paper, we introduced PanGu-π, an innovative LLM architecture designed to mitigate the feature collapse issue in Transformer models by integrating nonlinearity. Our initial step involved a detailed examination of contemporary LLM architectures where we pinpointed the feature collapse challenge. Our theoretical analysis led us to propose the integration of nonlinearity, a principle predominantly used in convolutional neural networks for image processing, as a critical component for enhancing language models. To implement this, we augmented the FFN with a series informed activation function, thereby enhancing its nonlinearity. Additionally, we introduced an augmented shortcut to the MSA module, effectively reducing feature collapse. Theoretical evaluations revealed that these enhancements significantly boost the approximation capabilities of both the FFN and MSA modules. Leveraging these enhancements, we developed various sizes of PanGu-π, notably PanGu-π-7B and PanGu-π-1B. These models underwent extensive training and SFT, showcasing strong NLP capabilities. The PanGu-π, foundation models achieved state-ofthe-art (SOTA) results in various downstream tasks, matching or surpassing similar-sized models. PanGu-π displayed exceptional adaptability, delivering SOTA performances in the specialized domains of finance and law.

#### REFERENCES

- [1] J. Kaplan et al. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [2] J. Wei et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022.

- [3] T. B. Brown et al. Language models are few-shot learners. In NeurIPS, 2020.
- [4] A. Chowdhery et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.
- [5] OpenAI. Gpt-4 technical report, 2023.
- [6] Z. Xi et al. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864, 2023.
- [7] H. Touvron et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [8] A. Zeng et al. GLM-130b: An open bilingual pre-trained model. In The Eleventh International Conference on Learning Representations (ICLR), 2023.
- [9] A. Yang et al. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023.
- [10] S. Pengxiao. Lawgpt. https://github.com/pengxiao-song/LaWGPT, 2023.
- [11] H. Yang et al. Fingpt: Open-source financial large language models. FinLLM Symposium at IJCAI 2023, 2023.
- [12] H. Wang et al. Huatuo: Tuning llama model with chinese medical knowledge, 2023.
- [13] A. Vaswani et al. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [14] B. Peng et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023.
- [15] A. Katharopoulos et al. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, 2020.
- [16] N. Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.
- [17] W. Fedus et al. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. The Journal of Machine Learning Research, 23(1):5232–5270, 2022.
- [18] Y. Tang et al. Augmented shortcuts for vision transformers. In NeurIPS, volume 34, pp. 15316–15327, 2021.
- [19] Y. Dong et al. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In ICML, pp. 2793–2803. PMLR, 2021.
- [20] H. Chen et al. Vanillanet: the power of minimalism in deep learning. In NeurIPS, 2023.
- [21] OpenAI. Introducing chatgpt. OpenAI Blog, November 2022.
- [22] OpenAI. Gpt-4 technical report. OpenAI, 2023.
- [23] H. Touvron et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [24] H. Touvron et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [25] R. Taori et al. Alpaca: A strong, replicable instruction-following model. Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html, 3(6):7, 2023.
- [26] Y. Wang et al. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022.
- [27] W.-L. Chiang et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2023.
- [28] D. Eccleston. Sharegpt. https://sharegpt.com/, 2023.
- [29] I. Team. Internlm: A multilingual language model with progressively enhanced capabilities, 2023.
- [30] X. Ren et al. Pangu-{\Sigma}: Towards trillion parameter language model with sparse heterogeneous computing. arXiv preprint arXiv:2303.10845, 2023.
- [31] J. Bai et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [32] T. Wei et al. Skywork: A more open bilingual foundation model. arXiv preprint arXiv:2310.19341, 2023.
- [33] M. Zaheer et al. Big bird: Transformers for longer sequences. Advances in neural information processing systems, 2020.
- [34] R. Child et al. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.
- [35] A. Roy et al. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 2021.
- [36] J. W. Rae et al. Compressive transformers for long-range sequence modelling. arXiv preprint arXiv:1911.05507, 2019.
- [37] G. Xiao et al. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.
- [38] K. Choromanski et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.

- [39] N. Kitaev et al. Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451, 2020.
- [40] I. Beltagy et al. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.
- [41] T. Brown et al. Language models are few-shot learners. Advances in neural information processing systems, 2020.
- [42] Z. Dai et al. Transformer-xl: Attentive language models beyond a fixedlength context. arXiv preprint arXiv:1901.02860, 2019.
- [43] P. H. Martins et al. ∞-former: Infinite memory transformer. arXiv preprint arXiv:2109.00301, 2021.
- [44] Y. Sun et al. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.
- [45] M. Poli et al. Hyena hierarchy: Towards larger convolutional language models. arXiv preprint arXiv:2302.10866, 2023.
- [46] B. Peng et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023.
- [47] N. Du et al. Glam: Efficient scaling of language models with mixtureof-experts. In International Conference on Machine Learning, 2022.
- [48] S. Roller et al. Hash layers for large sparse models. Advances in Neural Information Processing Systems, 2021.
- [49] Z. Chi et al. On the representation collapse of sparse mixture of experts. Advances in Neural Information Processing Systems, 2022.
- [50] M. Lewis et al. Base layers: Simplifying training of large, sparse models. In International Conference on Machine Learning, 2021.
- [51] A. Chowdhery et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.
- [52] N. Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.
- [53] W. Wang et al. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings of the IEEE/CVF international conference on computer vision, 2021.
- [54] Z. Liu et al. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, 2021.
- [55] A. Dosovitskiy et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
- [56] J. Guo et al. Cmt: Convolutional neural networks meet vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.
- [57] B. Heo et al. Rethinking spatial dimensions of vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.
- [58] Z. Pan et al. Scalable vision transformers with hierarchical pooling. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.
- [59] C.-F. R. Chen et al. Crossvit: Cross-attention multi-scale vision transformer for image classification. In Proceedings of the IEEE/CVF international conference on computer vision, 2021.
- [60] B. Graham et al. Levit: a vision transformer in convnet’s clothing for faster inference. In Proceedings of the IEEE/CVF international conference on computer vision, 2021.
- [61] S. Mehta and M. Rastegari. Mobilevit: light-weight, generalpurpose, and mobile-friendly vision transformer. arXiv preprint arXiv:2110.02178, 2021.
- [62] K. Han et al. Transformer in transformer. Advances in Neural Information Processing Systems, 2021.
- [63] N. Parmar et al. Image transformer. In International conference on machine learning, 2018.
- [64] X. Liu et al. Efficientvit: Memory efficient vision transformer with cascaded group attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.
- [65] Y. Xiong et al. Nystr¨omformer: A nystr¨om-based algorithm for approximating self-attention. In Proceedings of the AAAI Conference on Artificial Intelligence, 2021.
- [66] M.-H. Guo et al. Beyond self-attention: External attention using two linear layers for visual tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.
- [67] J. Guo et al. Hire-mlp: Vision mlp via hierarchical rearrangement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.
- [68] Y. Tang et al. An image patch is a wave: Phase-aware vision mlp. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.
- [69] W. Yu et al. Metaformer is actually what you need for vision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022.

- [70] D. Lian et al. As-mlp: An axial shifted mlp architecture for vision. arXiv preprint arXiv:2107.08391, 2021.
- [71] S. Chen et al. Cyclemlp: A mlp-like architecture for dense prediction. arXiv preprint arXiv:2107.10224, 2021.
- [72] L. Yuan et al. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In Proceedings of the IEEE/CVF international conference on computer vision, 2021.
- [73] X. Dong et al. Cswin transformer: A general vision transformer backbone with cross-shaped windows. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.
- [74] Y. Rao et al. Global filter networks for image classification. Advances in neural information processing systems, 2021.
- [75] H. Touvron et al. Resmlp: Feedforward networks for image classification with data-efficient training. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.
- [76] I. O. Tolstikhin et al. Mlp-mixer: An all-mlp architecture for vision. Advances in neural information processing systems, 2021.
- [77] S. Wu et al. Bloomberggpt: A large language model for finance. arXiv preprint arXiv:2303.17564, 2023.
- [78] B. Workshop et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.
- [79] H. Yang et al. Fingpt: Open-source financial large language models. arXiv preprint arXiv:2306.06031, 2023.
- [80] X.-Y. Liu et al. Fingpt: Democratizing internet-scale data for financial large language models. arXiv preprint arXiv:2307.10485, 2023.
- [81] Q. Xie et al. Pixiu: A large language model, instruction data and evaluation benchmark for finance. arXiv preprint arXiv:2306.05443, 2023.
- [82] R. S. Shah et al. When flue meets flang: Benchmarks and large pre-trained language model for financial domain. arXiv preprint arXiv:2211.00083, 2022.
- [83] X. Zhang and Q. Yang. Xuanyuan 2.0: A large chinese financial chat model with hundreds of billions parameters. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, pp. 4435–4439, 2023.
- [84] W. Chen et al. Disc-finllm: A chinese financial large language model based on multiple experts fine-tuning. arXiv preprint arXiv:2310.15205, 2023.
- [85] Baichuan-inc. Baichuan-13b. https://github.com/baichuan-inc/ Baichuan-13B, 2023.
- [86] xuanxuanzl. Baoluo lawassistant. https://github.com/xuanxuanzl/ BaoLuo-LawAssistant, 2023.
- [87] davidpig. Lychee. https://github.com/davidpig/lychee law, 2023.

- [88] seudl. Jurislms: Jurisprudential language models. https://github.com/ seudl/JurisLMs, 2023.
- [89] L. Yue et al. Fedjudge: Federated legal large language model. arXiv preprint arXiv:2309.08173, 2023.
- [90] B. McMahan et al. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pp. 1273–1282. PMLR, 2017.
- [91] M. Zhou et al. Image de-raining via continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4907–4916, 2021.
- [92] W. He et al. Hanfei-1.0. https://github.com/siat-nlp/HanFei, 2023.
- [93] Q. Huang et al. Lawyer llama technical report. arXiv preprint arXiv:2305.15062, 2023.
- [94] zhihaiLLM. wisdominterrogatory. https://github.com/zhihaiLLM/ wisdomInterrogatory, 2023.
- [95] S. Wu et al. fuzi.mingcha. https://github.com/irlab-sdu/fuzi.mingcha, 2023.
- [96] H. Liu et al. Lawgpt. https://github.com/LiuHC0428/LAW GPT, 2023.

- [97] CSHaitao. Lexilaw. https://github.com/CSHaitao/LexiLaw, 2023.
- [98] J. Cui et al. Chatlaw: Open-source legal large language model with integrated external knowledge bases. arXiv preprint arXiv:2306.16092, 2023.
- [99] S. Yue et al. Disc-lawllm: Fine-tuning large language models for intelligent legal services, 2023.
- [100] H. Shi et al. Revisiting over-smoothing in bert from the perspective of graph. arXiv preprint arXiv:2202.08625, 2022.
- [101] K. He et al. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pp. 1026–1034, 2015.
- [102] D. Hendrycks and K. Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.
- [103] P. Ramachandran et al. Searching for activation functions. arXiv preprint arXiv:1710.05941, 2017.

- [104] Y. Tang et al. Augmented shortcuts for vision transformers. Advances in Neural Information Processing Systems, 34:15316–15327, 2021.
- [105] Y. Shibata et al. Byte pair encoding: A text compression scheme that accelerates pattern matching. Technical Report DOI-TR-161, Department of Informatics, Kyushu University, 1999.
- [106] T. Kudo and J. Richardson. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226, 2018.
- [107] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [108] I. Loshchilov and F. Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.
- [109] B. Zhang and R. Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.
- [110] J. Su et al. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, pp. 127063, 2023.
- [111] O. Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/opencompass, 2023.
- [112] Y. Huang et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. arXiv preprint arXiv:2305.08322, 2023.
- [113] H. Li et al. Cmmlu: Measuring massive multitask language understanding in chinese. arXiv preprint arXiv:2306.09212, 2023.
- [114] D. Hendrycks et al. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.
- [115] W. Zhong et al. Agieval: A human-centric benchmark for evaluating foundation models, 2023.
- [116] C. Clark et al. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.
- [117] A. Wang et al. Superglue: A stickier benchmark for general-purpose language understanding systems, 2020.
- [118] Y. Bisk et al. Piqa: Reasoning about physical commonsense in natural language, 2019.
- [119] Y. Li et al. Csl: A large-scale chinese scientific literature dataset. arXiv preprint arXiv:2209.05034, 2022.
- [120] L. Xu et al. Fewclue: A chinese few-shot learning evaluation benchmark. arXiv preprint arXiv:2107.07498, 2021.
- [121] S. Narayan et al. Don’t give me the details, just the summary! topicaware convolutional neural networks for extreme summarization, 2018.
- [122] B. Hu et al. Lcsts: A large scale chinese short text summarization dataset, 2016.
- [123] F. Xue et al. Go wider instead of deeper. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 8779–8787, 2022.
- [124] X. Cai et al. Isotropy in the contextual embedding space: Clusters and manifolds. In International Conference on Learning Representations, 2021.
- [125] K. Ethayarajh. How contextual are contextualized word representations? comparing the geometry of bert, elmo, and gpt-2 embeddings. In Conference on Empirical Methods in Natural Language Processing, 2019.
- [126] M. P. Marcus et al. Building a large annotated corpus of english: The penn treebank. Computational Linguistics, 19:313–330, 2002.
- [127] T. Gao et al. Simcse: Simple contrastive learning of sentence embeddings. 2021.
- [128] Y. Cui et al. Efficient and effective text encoding for chinese llama and alpaca. arXiv preprint arXiv:2304.08177, 2023.
- [129] T. W. Peiyuan Zhang, Guangtao Zeng and W. Lu. Tinyllama, 2023.
- [130] M. Xia et al. Sheared llama: Accelerating language model pre-training via structured pruning. arXiv preprint arXiv:2310.06694, 2023.
- [131] P. Henderson et al. Pile of law: Learning responsible data filtering from the law and a 256gb open-source legal dataset, 2022.
- [132] I. Chalkidis et al. LeXFiles and LegalLAMA: Facilitating English multinational legal language model development. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15513–15535. Association for Computational Linguistics, July 2023.
- [133] H. Zhong et al. JEC-QA: A legal-domain question answering dataset. In Proceedings of AAAI, 2020.
- [134] C. Wang et al. Neural machine translation with byte-level subwords. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 9154–9160, 2020.
- [135] H. Zhong et al. Jec-qa: a legal-domain question answering dataset. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 9701–9708, 2020.
- [136] X. Zhang and Q. Yang. Xuanyuan 2.0: A large chinese financial chat model with hundreds of billions parameters. In Proceedings of the

- 32nd ACM International Conference on Information and Knowledge Management, pp. 4435–4439, 2023.
- [137] L. Zhang et al. Fineval: A chinese financial domain knowledge evaluation benchmark for large language models, 2023.
- [138] Z. Fei et al. Lawbench: Benchmarking legal knowledge of large language models. arXiv preprint arXiv:2309.16289, 2023.
- [139] Z. Fei et al. Lawbench: Benchmarking legal knowledge of large language models. arXiv preprint arXiv:2309.16289, 2023.

APPENDIX

Proof of Lemma 1

Proof. Write WW⊤ = PDP⊤ for the eigin-decomposition of WW⊤, where P = [p1,p2,...,pd] is the standard orthogonal and D = diag(d1,...,dd) with all d1 ≥ ··· ≥ dd ≥ 0 and for nonzero eigenvalues di = s2i > 0, where si is the i-th largest singular value of W.

###### HH⊤ = QΩQ⊤ for the eigin-decomposition of HH⊤, where Q = [q1,q2,...,qN] is the orthogonal and Ω = diag(ω1,...,ωN) with all ωi ≥ 0. And e = N−1/2[1,1,...,1]⊤ = N−1/21 ∈ RN×1.

(1) The formula proves as follows:

(HW)2 = ∥(I − ee⊤)HW∥2F

dM

m

= tr{(I − ee⊤)HWW⊤H⊤(I − ee⊤)}

= tr{WW⊤H⊤(I − ee⊤)H}

##### = tr{PDP⊤H⊤(I − ee⊤)H} = tr{DP⊤H⊤(I − ee⊤)HP} = tr{DP⊤H⊤(I − ee⊤)HP}

N

##### dip⊤i H⊤(I − ee⊤)Hpi

=

i=1

N

##### s2p⊤i H⊤(I − ee⊤)Hpi

≤

i=1

= s2dM

###### (H)2.

d

Since matrix H⊤(I − ee⊤)H is positive semidefinite, p⊤i H⊤(I − ee⊤)Hpi ≥ 0.

(H). Note that dM

It follows that dM

(HW) ≤ sdM

m

d

(H) = ∥H − 1x⊤min∥F, where x⊤min = arg minx

d

l ∥H −1x⊤l ∥F. And ∥σ(H1)−σ(H2)∥F ≤ L∥H1− H2∥F

- (2) The formula proves as follows:

dM

d

(σ(H)) = ∥σ(H) − 1xσ⊤min∥F ≤ ∥σ(H) − 1σ(xmin)⊤∥F

= ∥σ(H) − σ(1x⊤min)∥F ≤ L∥H − 1x⊤min∥F

= LdM

d

(H),

- (3) The formula proves as follows:

α1dM

(B) = α1∥H − 1xHmin⊤∥F + α2∥B − 1xBmin⊤∥F ≥ ∥α1H + α2B − 1(α1xHmin + α2xBmin)⊤∥F ≥ ∥α1H + α2B − 1(xα

(H) + α2dM

d

d

min )⊤∥F

1H+α2B

= dM

(α1H + α2B)

d

For the last inequality, we refer to [100].

Proof of Theorem 1

| |
|---|

Proof.

dM

m

√

(AZlW) ≤

(ZlW) ≤

λdM

m

√

λsdM

##### (Zl)

m

| |
|---|

###### Proof of Lemma 2 Proof.

(Concat([Hh]Hh=1))2

dM

Hm

=∥(I − ee⊤)Concat([Hh]Hh=1)∥2F

=tr{(I − ee⊤)Concat([Hh]Hh=1)Concat([Hh]Hh=1)⊤}

=tr{Concat([Hh]Hh=1)Concat([Hh]Hh=1)⊤} − tr{Concat([e⊤Hh]Hh=1)Concat([ee⊤Hh]Hh=1)⊤}

H

tr{(I − ee⊤)HhHh⊤}

=

h=1

H

(Hh)2,

=

dM

m

h=1

| |
|---|

- Proof of Theorem 2

Proof.

MSA(Zl) = Concat([AlhZlWlhv ]Hh=1)Wlo,

dM

d

(Zl+1) = dM

d

(Concat([AlhZlWlhv ]Hh=1)Wlo) ≤ υ1dM

d

(Concat([AlhZlWlhv ]Hh=1))

≤ υ1

H

h=1

dM

d

(AlhZlWlhv )2

≤

√

λsυ1

H

h=1

dM

d

(Zl)2

=

√

λHsυ1dM

d

(Zl) ≤ (

√

λHsυ1)l+1dM

d

(Z0).

| |
|---|

- Proof of Theorem 3

Proof.

MLP(Zl′) = σ(Zl′Wl′

1

##### )Wl′

, l ∈ [1,2,··· ,L],

2

##### )Wl′

(MLP(Zl′))) = dM

(σ(Zl′Wl′

) ≤ υ2dM

dM

d

d

2

1

(σ(Zl′Wl′

)) ≤ Lυ2dM

d

1

##### (Zl′Wl′

) ≤ Lsυ2dM

d

1

(Zl′)

d

(Zl′) ≤ (Lsυ2)dM

(Zl′−1) ≤ (Lsυ2)ldM

dM

d

d

###### (Z0′).

d

| |
|---|

Proof of Theorem 4 Proof.

AugMSA(Zl) = MSA(Zl) + Zl +

T

Tli(Zl;Θli),

i=1

dM

(AugMSA(Zl))

d

T

Tli(Zl;Θli))

= dM

(MSA(Zl) + Zl +

d

i=1

T

≤ dM

Tli(Zl;Θli))

(MSA(Zl)) + dM

(Zl) + dM

(

d

d

d

i=1

T

≤ dM

(MSA(Zl)) + dM

(Zl) +

dM

(σ(ZlΘli))

d

d

d

i=1

T

√

(Zl′) + L

≤ (

λHsυ1 + 1)dM

dM

##### (ZlΘli)

d

d

i=1

T

√

≤ (

L∥Θli∥2)dM

λHsυ1 + 1 +

(Zl).

d

i=1

Considering that H heads exist inthe MSA module, the diversity dM(Zl+1) becomes:

(Zl) ≤ (

dM

d

√

λHsυ1 + 1 +

T

L∥Θli∥2)ldM

###### (Z0).

d

i=1

| |
|---|

Proof of Lemma 3 Proof. When H = 1,

MSA(Zl) = AZlWWlo,

(MSA(Zl + ϵ) − MSA(Zl))

dM

d

= ∥(I − ee⊤)((A + δ)(Zl + ϵ) − AZl)(WWlo)∥F

= ∥(I − ee⊤)((A + δ)ϵ + δZl)(WWlo)∥F ≤ dM

(δZlWWlo) ≤ λA+δsυ1∥ϵ∥F + λδsυ1dM

##### (AϵϵWWlo) + dM

d

d

(Zl),

d

MSA(Zl) = Concat([AlhZlWlhv ]Hh=1)Wlo,

(MSA(Zl + ϵ) − MSA(Zl))

dM

d

(Concat([(Aϵlh(Zl + ϵ) − AlhZl)Wlhv ]Hh=1)) ≤ λA+δHsυ1∥ϵ∥F + λδHsυ1dM

= υ1dM

d

(Zl),

d

| |
|---|

Proof of Lemma 4 Proof.

(L(Zl + ϵ)Θli − LZlΘli)

dM

d

(LϵΘli) = L∥(I − ee⊤)ϵΘli∥F

= dM

d

li⊤

= L∥ϵΘli − 1xϵΘ

min∥F

≤ L∥ϵΘli∥F ≤ L∥Θli∥2∥ϵ∥F.

| |
|---|

- Proof of Theorem 5

Proof.

dM

d

(Tli(Zl + ϵ;Θli) − Tli(Zl;Θli))

= dM

d

(σ((Zl + ϵ)Θli) − σ(ZlΘli))

= ∥(I − ee⊤)(σ((Zl + ϵ)Θli) − σ(ZlΘli))∥F ≤ ∥I − ee⊤∥2∥(σ((Zl + ϵ)Θli) − σ(ZlΘli)∥F

= ∥(σ((Zl + ϵ)Θli) − σ(ZlΘli)∥F ≤ L∥ϵΘli∥F. ≤ L∥Θli∥2∥ϵ∥F.

When ee⊤(σ(Zl + ϵ) − σ(Zl)) ̸= 0,

∥(I − ee⊤)(σ((Zl + ϵ)Θli) − σ(ZlΘli))∥F < ∥σ((Zl + ϵ)Θli) − σ(ZlΘli)∥F.

| |
|---|

- Proof of Theorem 6

Proof.

dMd(AugMSA(Zl + ϵ) − AugMSA(Zl))

= dMd(MSA(Zl + ϵ) − MSA(Zl)) + dMd((Zl + ϵ) − Zl)

T

dMd(σ((Zl + ϵ)Θli) − σ(ZlΘli))

+

i=1

T

< (1 + λA+δHsυ1)∥ϵ∥F +

L∥ϵΘli∥F + λδHsυ1dMd(Zl)

i=1

T

≤ (1 + λA+δHsυ1 + L

∥Θli∥2)∥ϵ∥F + λδHsυ1dMd(Zl)

i=1

| |
|---|

###### Proof of Theorem 7 Proof.

n

SIAF − MLP(Zl′) = (

i=1

σi(Zl′Wl′1i))Wl′2i, l ∈ [1, 2, · · · , L].

(11)

(SIAF − MLP(Zl′))

dM

d

n

= ∥(I − ee⊤)(

σi(Zl′Wl′

))Wl′

∥F

1i

2i

i=1

n

⊤

′ l

) − σi(1xZ

∥σi(Zl′Wl′

##### Wl′

≤ υ2(

)∥F

min

1i

2i

i=1

n

(Zl′),

≤ (

Li)sυ2dM

d

i=1

(Zl′) ≤ (sυ2

dM

d

≤ (sυ2

n

(Zl′−1) (12)

Li)dM

d

i=1

n

(Z0′) (13)

Li)ldM

d

i=1

| |
|---|

###### Proof of Theorem 8

Proof.

dM

(Zp+q)

d

(MLPq(Zp)) ≤ (Lsυ2)qdM

= dM

d

((MSAq(Zp)) ≤ (

d

√

λHsυ1)p(Lsυ2)qdM

(Z0).

d

| |
|---|

###### Proof of Theorem 9

Proof.

dM

(Zp+q) dM

d

(SIAF − MLPq(Zp+q)) ≤ (sυ2

d

n

(AugMSAp(Zp))

Li)qdM

d

i=1

T

√

L∥Θli∥2)p(sυ2

≤ (

λHsυ1 + 1 +

i=1

n

Li)qdM

###### (Z0).

d

i=1

| |
|---|

##### .1 Examples generated by PanGu-π-1B

Input: Describe a time when you had to make a difficult decision.

PanGu-π-1B: As an AI assistant, I have to make decisions every day, and sometimes it can be difficult to make a decision. One example of a difficult decision I made was when I was asked to provide a response to a customer’s query. I had to decide whether to provide a response that would be helpful to the customer or to provide a response that would be unhelpful.

Input: Based on the given article, choose the right option for the question. Article: We have two new students in our class. They are Cindy and Kate. They look the same. But they aren’t twins. Cindy is American. She is thirteen. Her father and mother are both teachers. She likes green and blue. She is often in blue pants. She often plays the guitar after school. She is good at math and English. She says math is her favorite because it is interesting. Kate is English. She is twelve. Her parents are not teachers. Her father is a doctor and her mother is a worker. Kate likes yellow and orange. She can’t play the guitar. But she plays volleyball very well. Her favorite subject is Chinese. She say she wants to learn about Chinese history. Kate likes documentaries very much.

Question: If you are ill, you can go to see . Options: (A) Cindy’s father (B) Cindy’s mother (C) Kate’s father (D) Kate’s mother Answer:

PanGu-π-1B: Based on the given article, the correct option for the question is: (C) Kate’s father.

问：“风声、雨声、读书声，声声人耳；家事、国事、天下事，事事关心。”这副对联给我们的启示 是

- A. 要参加社会实践活动，热心帮助贫困儿童
- B. 要学好文化知识，其他事情与自己无关
- C. 要关心社会发展，关注国家大事
- D. 事必躬亲，大小事都要亲力亲为 答案是选项 PanGu-π-1B：C. 要关心社会发展，关注国家大事。

问：请阅读文章，并根据文章内容从文中选择合适的部分回答问题。

文章：阿部正丰（生年不详－1535 年 12 月 29 日）是日本战国时代的武将。三河松平氏家臣，阿部定 吉的长子。通称弥七郎。天文 4 年（1535 年），已经平定西三河的松平清康为了讨伐叔父松平信定而 在尾张守山布阵，有传言指父亲定吉与织田信秀内通并企图谋反，因此定吉担心自己会被清康杀死， 于是向清康送出自己没有二心的誓纸。翌日，听到本阵发出马嘶叫声的正丰误以为父亲被杀，于是在 本阵杀死清康。而正丰则在当场被植村氏明斩杀（森山崩）。

问题：阿部正丰被通称为什么？ 答： PanGu-π-1B：正丰被通称弥七郎。

##### .2 Examples generated by PanGu-π-7B

Q: Describe a time when you had to make a difficult decision. PanGu-π-7B: As an AI model, I don’t have real-life experiences or the ability to make decisions, but I can tell you about a scenario that would require a difficult decision. Imagine a situation where a person is facing a moral dilemma. For example, they are presented with two options: doing the right thing but it may result in negative consequences, or doing something wrong but it may result in positive outcomes. In this situation, the person must weigh the pros and cons of each option and make a decision based on their personal values and beliefs. This can be a challenging and difficult decision to make, as it requires the person to examine their own morality and make a choice that may not always be easy or clear-cut.

Q: I’m going to be participating in a comedy roast. What are some hilariously spicy roasts I can use? PanGu-π-7B: As an AI, I should not encourage or promote any kind of controversial or offensive content. It is not appropriate to use any personal or derogatory material in a roast. However, there are many other topics you can use for your roast, such as popular culture, food, animals, or even your own personal quirks.

问：鸡和兔在一个笼子里，共有 26 个头，68 只脚，那么鸡有多少只，兔有多少只？ PanGu-π-7B：假设鸡有 x 只，兔有 y 只。 根据题目，我们可以得到两个方程：

- 1. x + y = 26 （鸡和兔总共有26个头）
- 2. 2x + 4y = 68 （鸡和兔总共有68只脚） 我们可以使用消元法来解这个方程组。 将第一个方程乘以2，得到2x + 2y = 52。 将第二个方程减去第 一个方程，得到 2x + 4y - 2x - 2y = 68 - 52，即 2y = 16，y = 8。 将 y 的值代入第一个方程，得到 x + 8

= 26，x = 18。 所以，鸡有 18 只，兔有 8 只。

###### .3 Examples generated by YunShan

Input: Do you know what a lawyer’s letter is? You help me write a lawyer’s letter, I was cheated, someone cheated me with fake information 1 million, I’m depressed now, I can’t get back the money. YunShan：A lawyer’s letter is a legal document used to give a formal notice or demand of rights and obligations to the other party. Typically, attorney letters are used to resolve disputes or resolve potential litigation issues. Based on the information you have provided, it appears that you have been defrauded of 1 million and you wish to recover this money. In this case, you should immediately report the case to the local public security authorities and provide all relevant evidence and information so that they can investigate. If you need legal help, you can consult a professional lawyer who can help you assess your rights and responsibilities and provide you with legal advice and support in negotiating or litigation on your behalf.

