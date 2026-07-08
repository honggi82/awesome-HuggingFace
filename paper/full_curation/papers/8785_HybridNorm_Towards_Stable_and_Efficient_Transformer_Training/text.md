# arXiv:2503.04598v4[cs.CL]8Dec2025

[Figure 1]

## HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization

### Zhijian Zhuo1,2, Yutao Zeng2,†, Ya Wang2,†, Sijun Zhang2, Jian Yang3, Xiaoqing Li4, Xun Zhou2, Jinwen Ma1

1School of Mathematical Sciences, Peking University, 2ByteDance Seed, 3Beihang University, 4Capital University of Economics and Business

†Corresponding authors

### Abstract

Transformers have become the de facto architecture for a wide range of machine learning tasks, particularly in large language models (LLMs). Despite their remarkable performance, many challenges remain in training deep transformer networks, especially regarding the position of the layer normalization. While Pre-Norm structures facilitate more stable training owing to their stronger identity path, they often lead to suboptimal performance compared to Post-Norm. In this paper, we propose HybridNorm, a simple yet effective hybrid normalization strategy that integrates the advantages of both Pre-Norm and Post-Norm. Specifically, HybridNorm employs QKV normalization within the attention mechanism and Post-Norm in the feed-forward network (FFN) of each transformer block. We provide both theoretical insights and empirical evidence to demonstrate that HybridNorm improves the gradient flow and the model robustness. Extensive experiments on large-scale transformer models, including both dense and sparse variants, show that HybridNorm consistently outperforms both Pre-Norm and Post-Norm approaches across multiple benchmarks. These findings highlight the potential of HybridNorm as a more stable and effective technique for improving the training and performance of deep transformer models.

Date: December 9, 2025 Correspondence: Yutao Zeng at yutao.zeng@bytedance.com, Ya Wang at wangyazg@gmail.com Project Page: https://github.com/BryceZhuo/HybridNorm

### 1 Introduction

Transformers have become the backbone of large language models (LLMs) and a wide range of machine learning applications. These architectures are capable of modeling long-range dependencies through self-attention mechanisms, which have made them the preferred choice for a variety of tasks, including language modeling, machine translation, and image processing [3, 8, 10]. However, as transformer models become deeper and more complex, ensuring stable training remains a significant challenge. One critical factor that influences training stability is the choice of normalization methods, which is crucial for mitigating issues such as internal covariate shift and gradient instability [44]. Effectively addressing these challenges is crucial for fully harnessing the potential of deep transformer models in large-scale applications.

In transformers, Layer Normalization (LayerNorm) [1] plays a central role in stabilizing training by normalizing

the activations within each layer. The two predominant strategies for applying LayerNorm are Pre-Layer Normalization (Pre-Norm) and Post-Layer Normalization (Post-Norm), each with its respective benefits and trade-offs. In the Pre-Norm architecture, normalization is applied before the residual addition, resulting in a more prominent identity path that facilitates faster convergence and more stable gradients [44]. This design is particularly advantageous when training deep models, as it helps mitigate gradient-related issues that can arise during backpropagation. However, while Pre-Norm can stabilize training, it often leads to inferior final performance compared to Post-Norm [43]. In contrast, Post-Norm applies normalization after the residual connection, resulting in stronger regularization effects, which contribute to improved model performance. This approach has been shown to improve the generalization ability of transformers, particularly in very deep networks [39]. Further discussion of related work is provided in Appendix A.

Despite the benefits of each approach, there is an inherent trade-off between training stability and final model performance. Pre-Norm structures typically stabilize training but may underperform in terms of generalization, while Post-Norm architectures provide better performance but can be more difficult to train, especially in deep models. To reconcile these trade-offs, we propose a hybrid normalization method that applies QKV normalization in the attention mechanism and Post-Norm in the feed-forward network (FFN), which is named as HybridNorm. The QKV normalization in the attention mechanism stabilizes the flow of information between layers by normalizing the query, key, and value components, while Post-Norm in the FFN ensures the effective depth of the transformer.

Through extensive experiments on large-scale models, we validate the effectiveness of our approach. Our results show that the hybrid normalization method significantly outperforms both Pre-Norm and Post-Norm across multiple benchmarks, providing a stable training process and improved model performance. We believe that this hybrid approach offers a promising solution for enhancing the training stability and performance of deep transformer architectures, particularly in the rapidly evolving field of LLMs. The main contributions of this paper can be summarized as follows:

- • We propose HybridNorm, a novel hybrid normalization structure that combines the advantages of Pre-Norm and Post-Norm, offering a simple yet effective solution to enhancing performance in large transformer models. Our method is designed to exploit the strengths of both normalization approaches, ensuring robust convergence during training and superior final performance.
- • We present both theoretical and empirical analyses of HybridNorm, demonstrating its advantages in enhancing gradient flow stability and improving model robustness. Our findings underscore the method’s effectiveness in mitigating core challenges inherent to deep transformer architectures.
- • Through extensive experiments on large-scale models, we empirically validate the effectiveness of our approach. Our results show that hybrid normalization significantly outperforms both Pre-Norm and Post-Norm across a variety of tasks, leading to more stable training and improved model performance, particularly in the context of LLMs.

### 2 Preliminaries

Scaled Dot-Product Attention The scaled dot-product attention computes the attention scores between the Query (Q) and Key (K) matrices, scaled by the square root of the key dimension dk, and applies these scores to the Value (V) matrix. The formulation is expressed as

attn(Q,K,V ) = softmax

QK⊤ √dk

V, (1)

where Q,K,V ∈ Rn×d

k represent the query, key, and value matrices respectively, and n is the sequence length.

Multi-Head Attention Multi-head attention (MHA) extends the scaled dot-product attention mechanism by splitting the query, key, and value matrices into h heads, each of size dk = d/h. Each head independently computes attention scores, and the outputs are concatenated and linearly projected to the original dimension,

MHA(X) = Concat(head1,...,headh)WO, (2)

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Norm

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

###### FFN

###### FFN

###### FFN

FFN

Norm

Norm

Norm

Norm

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Linear

Linear

Linear

Linear

Attention

Attention

Attention

Attention

Norm Norm Norm

Norm Norm

Q K V Q K V Q K V Q K V

Linear

Linear Linear

Linear

Linear Linear

Linear

Linear Linear

Linear

Linear Linear

Norm

Norm

(a) (b) (c) (d)

Figure 1 Illustrations of different transformer layer structures: (a) Post-Norm architecture; (b) Pre-Norm architecture; (c) Pre-Norm with QK-Norm architecture; (d) HybridNorm architecture.

where headi = attn(Qi,Ki,Vi) for i = 1,2,...,h, {•i}hi=1 = Split(XW•) for • ∈ {Q,K,V }, and learnable parameters WQ,WK,WV ,WO ∈ Rd×d . By enabling the model to focus on different subspaces of the input representation, MHA enhances the transformer’s capacity to capture diverse patterns in the input sequence.

#### 2.1 Post-Norm and Pre-Norm

The transformer architecture is composed of a stack of L blocks, each consisting of two key components: MHA and FFN. Residual connections and normalization layers are applied around both the MHA and FFN in each block to facilitate effective training and improve model stability. Figure 1 (a)&(b) illustrate Post-Norm and Pre-Norm, respectively.

Post-Norm Post-Norm applies the normalization layer after the residual connection in each transformer sub-layer. Formally, the output of Post-Norm can be expressed as

Y l = Norm(MHA(Xl) + Xl), Xl+1 = Norm(FFN(Y l) + Y l), (3) where Norm denotes RMSNorm [48] or LayerNorm [1].

Pre-Norm In contrast, Pre-Norm normalizes the input to the sub-layer, which allows for a more prominent identity path. The output of Pre-Norm is given by

Y l = MHA(Norm(Xl)) + Xl, Xl+1 = FFN(Norm(Y l)) + Y l. (4)

This structure facilitates better gradient flow and stable convergence, particularly for deep models. However, its reliance on normalization before the residual connection can lead to suboptimal performance compared to Post-Norm, as the normalization does not account for the interaction between the residual connection and the sub-layers output. An analysis of the fundamental differences between the two approaches is provided in the Appendix F.

### 3 HybridNorm

To address the trade-offs between Post-Norm and Pre-Norm, we propose HybridNorm, a hybrid normalization strategy that integrates their strengths. Specifically, HybridNorm combines QKV-Norm [22, 30] in MHA and Post-Norm in FFN.

QKV Normalization in Attention In the attention mechanism, the query, key, and value matrices are normalized individually before computing the attention output. The normalized QKV matrices are then used in the scaled dot-product attention. QKV-Norm enhances the stability of model training and leads to improved downstream performance. Formally, attention with QKV-Norm is defined as

attnQKV (Q,K,V ) = softmax

Norm(Q)Norm(K)⊤ √dk

Norm(V ). (5)

And we denote the multi-head attention with attnQKV as MHAQKV . HybridNorm Architecture Combining the above, the overall output of a transformer block with HybridNorm can be expressed as

Y l = MHAQKV (Xl) + Xl, Xl+1 = FFN(Norm(Y l)) + Norm(Y l). (6)

The architecture illustration can be found in Figure 1(d) and the pseudocode is shown in Algorithm 1. By integrating QKV normalization in the attention mechanism and Post-Norm in the FFN, HybridNorm achieves stable training dynamics and enhanced final performance. The theoretical gradient analysis can be found in Appendix B.

Remark 1. The method most closely related to ours is Mix-LN [20], which applies Post-Norm to the earlier layers and Pre-Norm to the deeper layers, resulting in enhanced training stability and performance. In contrast, our proposed HybridNorm integrates Pre-Norm and Post-Norm within each transformer layer, thereby providing a more uniform approach across different layers to leverage the benefits of both normalization strategies. Moreover, experiments demonstrate that HybridNorm achieves superior downstream performance compared to Mix-LN (see Table 6 & Table 12).

Special Treatment of First Block Inspired by prior work [6], which employs the Mixture of Experts (MoE) architecture with specialized handling of the first layer, we explore the impact of introducing specialized normalization to the first transformer block. In our approach, the first layer of the transformer is treated differently by applying Pre-Norm on MHA and FFN, while maintaining QKV-Norm. Specifically, the structure of our first layer is defined as

Y 0 = MHAQKV (Norm(X0)) + X0, X1 = FFN(Norm(Y 0)) + Y 0. (7)

We refer to this variation of HybridNorm, which incorporates the specialized first block treatment, as HybridNorm∗. This design aims to stabilize the training of the first transformer block and boost overall performance by improving the flow of gradients in the early stages of training.

### 4 Theoretical Analysis

#### 4.1 Benefits of Hybrid Method

To gain deeper insights into the stability introduced by HybridNorm, we follow the approach of [20, 35] and analyze the evolution of gradient norms throughout training iterations. Suppose x and F are the input and the sublayer of Transformer, respectively. The output of Post-Norm is yPost = Norm(x + F(x)) and the output of Pre-Norm is yPre = x + F(Norm(x)). Then we have

∂yPost ∂x

∂Norm(x + F(x)) ∂(x + F(x))

=

∂F(x) ∂x

I +

, (8)

Pre-LN Post-LN HybridNorm

| |
|---|

- 0.8

| |
|---|

GradientNorm

0.6

0.4

0.2

0.0

0 4 8 12

Layer Index

Pre-LN Post-LN HybridNorm

0.12

| |
|---|

| |
|---|

0.10

GradientNorm

0.08

0.06

0.04

0.02

0.00

0 4 8 12

Layer Index

(b) Layer gradient norm at step 100. Figure 2 Gradient norm of Pre-Norm, Post-Norm, and HybridNorm at different training steps.

(a) Layer gradient norm at step 1.

∂yPre ∂x

∂F(Norm(x)) ∂Norm(x)

∂Norm(x) ∂x

, (9)

= I +

√

⊤

where the gradient of normalization is ∂Norm(∂x x) = α ⊙

. The gradient of Post-Norm is the product of two gradients, one of which is the normalization. If the spectral radius of the normalization gradient is less than 1, it causes gradient vanishing. In Pre-Norm, the residual connection is isolated from the normalization, ensuring that gradients retain a lower bound, thereby preventing gradient vanishing. To ensure relatively stable gradients, a natural idea is to use both types of normalization within a Transformer block, leading to Pre-Post (Pre-Norm in MHA and Post-Norm in FFN) and Post-Pre. From Table 6, we find that Pre-Post achieves the best performance, which is why we adopt this. And placing Pre-Norm in MHA with QKV-Norm further enhances performance.

||x||2 I − xx

d

||x||22

In Figure 2, we compare the gradient norms of Pre-Norm, Post-Norm, and HybridNorm at steps 1 and 100. The results indicate that Pre-Norm tends to exhibit gradient explosion in deeper models, while Post-Norm suffers from vanishing gradients, both of which hinder effective optimization. In contrast, HybridNorm maintains a well-balanced gradient flow throughout training, effectively mitigating these issues. An intuitive understanding is that Pre-Norm tends to amplify gradients, while Post-Norm diminishes them. HybridNorm alternates between these two normalization strategies, leading to more stable gradient propagation during backpropagation and effectively preventing gradient explosion or vanishing. This balanced gradient propagation contributes to smoother optimization dynamics and faster convergence, further reinforcing the effectiveness of HybridNorm in stabilizing training.

#### 4.2 Benefits of QKV-Norm

Theoretically, we study how QKV-Norm affects the gradient flow during backpropagation, which is crucial for training stability in deep transformer models. Our analysis reveals that QKV-Norm helps decouple gradients between different weight matrices, thereby stabilizing training.

Theorem 1 (Informal version of Theorem 2). Suppose the the output of the attention is S, the input X ∈ Rs×d, parameters WQ,WK,WV ,WO⊤ ∈ Rd×d

k. For the attention with Pre-Norm, we have ∂S

∂S ∂WV F

= O (∥WO∥F), ∂S

= O (∥WV ∥2),

∂WO F

∂S ∂WK F

= O (∥WQ∥2∥WV ∥2∥WO∥2).

= O (∥WK∥2∥WV ∥2∥WO∥2),

∂WQ F

For the attention with Pre-Norm and QK-Norm, we have ∂S ∂WO F

∂S ∂WV F

= O (∥WV ∥2),

= O (∥WO∥F),

∂S ∂WQ F

=

∂S ∂WK F

= O (∥WV ∥2∥WO∥2).

For the attention with QKV-Norm, we have ∂S ∂WO F

∂S ∂WV F

= O (1),

= O (∥WO∥2),

∂S ∂WQ F

=

∂S ∂WK F

= O (∥WO∥2).

The above theorem is an informal version of Theorem 2, with a more precise statement and proof provided in Appendix B. In attention with Pre-Norm, gradients of weights exhibit strong dependencies on other weights; for instance, WQ and WK are influenced by all three other weights but not by themselves. In contrast, with QKV-Norm, the gradient of each weight depends at most on itself and WO. This indicates that the gradient in Pre-Norm is more tightly coupled with other weights compared to QKV-Norm, while Pre-Norm with QK-Norm lies between the two. Consequently, during training, if the norm of a certain weight becomes excessively large, it is harder to control in Pre-Norm, leading to increased gradient magnitude. This creates a vicious cycle that may cause model collapse. In contrast, QKV-Norm alleviates this issue and significantly improves training stability. In summary, the degree of gradient coupling follows: Pre-Norm > Pre-Norm with QK-Norm > QKV-Norm; whereas training stability follows the reverse: Pre-Norm < Pre-Norm with QK-Norm < QKV-Norm.

### 5 Experiments

#### 5.1 Experiment Settings

Baseline We evaluate HybridNorm across two series of models: dense models and Mixture of Experts (MoE) models. The dense models include two scales: 550M and 1B, with the latter containing approximately 1.27 billion parameters and utilizing an architecture similar to LLaMA 3.2 [11]. All analytical experiments are conducted on the 550M dense models. For the MoE model, we use the OLMoE framework [24], which activates

- 1.3B parameters out of a total of 6.9B parameters (MoE-1B-7B). Both models are trained from scratch on the OLMoE Mix dataset [24].

Model Configuration The 550M dense model has a model dimension of 1536, an FFN dimension of 4096, and utilizes 16 attention heads with 4 key/value heads per attention head. The 1.2B model features a larger model dimension of 2048 and an FFN dimension of 9192, with 32 attention heads and 8 key/value heads per attention head. The MoE-1B-7B model employs 16 attention heads, a model dimension of 2048, and an FFN dimension of 1024. Notably, it features 8 experts out of 64, providing a more fine-grained distribution of computational resources. All models consist of 16 layers and are trained with a consistent context length of 4096. More details can be found in Appendix C.

Hyperparameters Model weights are initialized using Megatron initialization [33] (See Section 5.4 for more details). For the optimization, we apply the AdamW optimizer with β1 = 0.9 and β2 = 0.95. All models are trained on sequences of 4096 tokens. For the dense model, we set the learning rate to 3e-4, decaying to 3e-5 using a cosine scheduler. The MoE model starts with a learning rate of 4e-4, decaying with a cosine schedule. We summarize the hyperparameters in Table 10.

Evaluation Metrics To evaluate the performance of LLMs with HybridNorm, we employ a diverse set of open benchmarks, including ARC-Easy (ARC-E) [5], ARC-Challenge (ARC-C) [5], HellaSwag [47], PIQA [2], SciQ [42], CoQA [29], Winogrande [31], MMLU [15], BoolQ [4], COPA [13], CSQA [36], OBQA [23], and SocialIQA [32]. We leverage the LM Eval Harness [12] for standardized performance evaluation.

- Figure 3 Training dynamics for 1.2B dense models with Pre-Norm, HybridNorm and HybridNorm∗ under 1T training tokens. We present the training loss, validation loss, and downstream performance on HellaSwag and ARC-Easy, demonstrating that HybridNorm∗ achieves superior performance.

- Table 1 Downstream evaluation results of 1.2B dense models with Pre-Norm, HybridNorm, and HybridNorm∗ under 1T training tokens. OQA refers to OpenbookQA.

Methods BasicArithmetic HellaSwag SciQ ARC-C ARC-E PIQA OQA COPA Avg.↑ Pre-Norm 44.10 63.41 91.80 39.20 69.82 75.19 38.40 82.00 62.99 HybridNorm 44.12 64.22 91.88 39.13 71.05 74.72 38.88 82.00 63.25 HybridNorm∗ 47.21 65.12 91.38 37.06 71.79 75.72 39.16 85.78 64.15

#### 5.2 Main Results

Dense Models We evaluate the performance of HybridNorm and HybridNorm∗ on 1.2B dense transformer models. Figure 3 compares the training dynamics of dense models with different normalization methods. As shown in the figure, models with HybridNorm and HybridNorm∗ exhibit consistently lower training loss and validation perplexity throughout training compared to Pre-Norm, highlighting their effectiveness in enhancing training stability and convergence. Table 1 presents the downstream evaluation results. HybridNorm∗ consistently outperforms Pre-Norm in most tasks, achieving the highest average score. Notably, it demonstrates substantial improvements in tasks such as BasicArithmetic (+3.11), HellaSwag (+1.71), and COPA (+3.78), indicating enhanced generalization and robustness. These results underscore the scalability of HybridNorm∗ in larger transformer models, further validating its effectiveness in improving both training stability and downstream performance. More results can be found in Figure 9. In Appendix E.1, we present additional comparisons with other approaches, such as Post-Norm and Mix-LN. We further provide analyses of signal propagation and entropy dynamics across different methods, as detailed in Appendix E.2 & E.3.

MoE Models For MoE models, we conduct experiments on MoE-1B-7B with 8 experts selected from a pool of 64. Figure 4 presents the training dynamics of MoE models under different normalization strategies. Throughout the training, HybridNorm∗ consistently achieves lower training loss and validation perplexity compared to Pre-Norm. These findings indicate that HybridNorm∗ effectively alleviates optimization difficulties in large-scale MoE models, resulting in more stable training and enhanced downstream performance. Further, as shown in Table 2, HybridNorm∗ consistently outperforms Pre-Norm across various downstream tasks, achieving the highest average score. Notably, it demonstrates significant improvements in ARC-C (+2.35), ARC-E (+2.40), and OpenbookQA (+0.81), highlighting its ability to enhance generalization across diverse benchmarks.

#### 5.3 More Experiment in 7B Dense Model

The experimental setup primarily follows the 7B model configuration outlined in [27] and the number of training tokens is 150B. To further enhance model stability, we introduce an additional normalization layer to the output of the attention module in 7B model, with its weight initialized to 1/

√

2L [41], where L denotes

- Figure 4 Training dynamics for MoE-1B-7B models with Pre-Norm and HybridNorm∗ under 500B training tokens. We present the training loss, validation loss, and downstream performance on HellaSwag and MMLU Var, demonstrating that HybridNorm∗ achieves superior performance.

- Table 2 Downstream evaluation results of MoE-1B-7B with Pre-Norm and HybridNorm∗ under 500B training tokens. OQA refers to OpenbookQA.

Methods HellaSwag ARC-C ARC-E PIQA WinoGrande OQA BoolQ COPA Avg.↑ Pre-Norm 69.94 39.92 73.37 77.82 63.34 42.37 67.47 85.40 64.95 HybridNorm∗ 70.71 42.27 75.77 78.06 64.58 43.18 68.41 86.00 66.12

- Table 3 Training loss and perplexity (PPL) comparison of Pre-Norm and HybridNorm∗ for 7B dense mdoel across multiple datasets. CC means Common Crawl.

Methods Loss C4 Books CC peS2o Reddit Stack Wiki Pile Wikitext

Pre-Norm 2.469 15.32 13.37 17.10 8.07 20.31 3.57 9.96 7.85 10.09 HybridNorm∗ 2.430 14.83 12.77 16.77 7.67 19.65 3.40 9.34 7.81 9.16

the number of model layers. Table 3 presents the training loss and validation perplexity (PPL), while Table 4 reports the downstream evaluation metrics.

The experimental results demonstrate the clear superiority of HybridNorm∗ over the traditional Pre-Norm approach in the 7B model. Firstly, HybridNorm∗ achieves a lower training loss (2.430 vs. 2.469), indicating more efficient optimization during training. This improvement in training dynamics translates into consistently better performance across a range of language modeling benchmarks. Specifically, HybridNorm∗ yields lower perplexity scores on all evaluated datasets, including C4, Books, Common Crawl, Wiki, and Wikitext 103. For instance, perplexity on the C4 dataset drops from 15.32 to 14.83, suggesting stronger generalization across both structured and unstructured corpora.

Moreover, HybridNorm∗ shows consistent improvements on all downstream tasks, covering various domains such as arithmetic reasoning, commonsense QA, and natural language inference. Notably, performance on Basic Arithmetic improves significantly from 43.50% to 50.67%. Across the full suite of tasks, including ARC, PIQA, COPA, BoolQ, and WinoGrande, HybridNorm∗ outperforms Pre-Norm in every case, leading to an overall increase in average accuracy from 60.61% to 63.06%.

#### 5.4 Ablation Studies

Initialization To evaluate the sensitivity of Pre-Norm and HybridNorm to initialization schemes, we conduct ablation studies comparing three widely used initialization strategies: Normal initialization [25], Depth-Scaled initialization [14, 49], and Megatron initialization [33]. Normal initialization initializes all weights of linear layers using a truncated normal distribution with mean zero and standard deviation 1/

√

2.5d, where d is the hidden dimension. Depth-Scaled initialization and Megatron initialization introduce scaling factors to

- Table 4 Downstream evaluation results of 7B dense models trained with Pre-Norm and HybridNorm∗. All numbers denote task accuracies (%). BA and OAQ mean Basic Arithmetic and Openbook QA, respectively.

Methods BA

Hella Swag

SciQ ARC-C ARC-E PIQA OQA COPA

Wino Grande

BoolQ Avg.↑

Pre-Norm 43.50 69.03 46.57 41.47 74.95 76.71 39.40 84.00 63.00 67.43 60.61 HybridNorm∗ 50.67 70.77 47.44 43.82 75.82 78.93 43.77 86.01 63.32 70.06 63.06

- Table 5 Training loss and validation perplexity of 550M dense models with Pre-Norm and HybridNorm under various initialization methods and 400B training tokens.

Method Initialization Loss↓ PPL on C4↓ Method Initialization Loss↓ PPL on C4↓

Pre-Norm

Normal 2.75 20.29

HybridNorm

Normal 2.76 20.44 Depth-Scaled 2.76 20.49 Depth-Scaled 2.76 20.40 Megatron 2.76 20.44 Megatron 2.74 20.00

- Table 6 Abalation study of the position of normalization layers on 550M dense models with 400B tokens. We report the training loss, the perplexity on C4 and Pile, and the accuracy on HS (HellaSwag).

Methods Loss↓ C4↓ Pile↓ HS ↑ QKVC-Post 2.74 20.05 10.34 52.68 QKC-Post 2.73 20.00 10.31 52.26 QK-Post - diverge KV-Post 2.74 20.11 10.38 52.10 KC-Post 2.75 20.34 10.47 51.15

Methods Loss↓ C4↓ Pile↓ HS ↑ Post-Norm 2.76 20.43 10.57 51.20 Pre-Norm 2.75 20.30 10.48 51.97 QK-norm 2.75 20.22 10.43 52.29 Mix-LN 2.76 20.43 10.56 51.29 Post-Pre 2.75 20.26 10.46 51.19 Pre-Post 2.74 20.15 10.40 52.42 HybridNorm 2.74 20.00 10.29 53.35 HybridNorm∗ 2.73 19.85 10.25 53.36

Pre-QKV-Post 2.74 20.13 10.37 52.65 Pre-QKV-Pre 2.74 19.97 10.33 53.05 Pre-QK-Pre 2.75 20.22 10.43 52.29 QKV-Pre 2.74 19.96 10.33 52.57

stabilize training in deep architectures. Specifically, Depth-Scaled initialization scales down the output projections of the attention and FFN by a factor of √

2l, where l is the layer index. In contrast, Megatron initialization scales down these projections by √

2L, where L is the total number of layers, mitigating gradient variance accumulation in very deep transformers. As shown in Table 5, Pre-Norm and HybridNorm exhibit sensitivity across different initialization methods, achieving the lowest training loss and perplexity under Normal initialization and Megatron initialization, respectively. Therefore, we set the default initialization method for Pre-Norm to Normal initialization and for HybridNorm to Megatron initialization in all experiments, respectively, which ensures that even under settings that may be more favorable to baseline models, the superiority of our approach is effectively demonstrated.

Normalization Position We investigate the impact of the position of normalization layers within the transformer block. First, we examine the effect of varying the placement of QKV normalization (e.g., normalization setting in attention). We extend the normalization setting by considering not only the Query (Q), Key (K), and Value (V) components but also the Context (C), which refers to the output of the attention mechanism. For instance, QKVC-Norm applies normalization to all four components: Query, Key, Value, and Context, while KV-Norm and KC-Norm focus on the normalization of the Key-Value and Key-Context pairs, respectively. QKVC-Post refers to transformer blocks that employ QKVC-Norm in the MHA while using Post-Norm in the FFN. Second, we explore the effect of integrating QKV-Norm into different transformer architectures. For instance, Pre-QKV-Post refers to a configuration where QKV-Norm is applied with Pre-Norm in the MHA layer, while the FFN layer utilizes Post-Norm. Other configurations follow similar definitions. Finally, we

- Figure 5 Training loss and accuracy on HellaSwag of 550M dense models with different normalization methods for the first block.

3.2

Pre-Norm

HybridNorm*

3.1

3.0

TrainingLoss

2.9

2.8

2.7

2.6

2.5

151M 285M 550M 1.2B

Model Size

Figure 6 Scaling law curves of Pre-Norm and HybridNorm∗.

- Table 7 Performance of dense models with different depths under 400B training tokens. We report the training loss, the perplexity on C4 and Pile, and the accuracy on HellaSwag and PIQA.

550M, 16 Layers 543M, 29 Layers Loss↓ C4↓ Pile↓ HellaSwag↑ PIQA↑ Loss↓ C4↓ Pile↓ HellaSwag↑ PIQA↑

Models

Post-Norm 2.76 20.43 10.57 51.20 71.80 diverge Pre-Norm 2.75 20.30 10.48 51.97 71.14 2.73 19.88 10.31 53.86 71.63 HybridNorm 2.74 20.00 10.29 53.35 71.96 2.72 19.67 10.18 54.89 72.62 HybridNorm∗ 2.73 19.85 10.25 53.36 71.15 2.71 19.52 10.10 54.54 72.01

compare various hybrid combinations of Pre-Norm and Post-Norm. Pre-Post refers to transformer blocks that apply Pre-Norm in the MHA and Post-Norm in the FFN, whereas Post-Pre adopts the opposite configuration. Mathematical formulas for methods mentioned above can be found in Appendix G.

As shown in Table 6, HybridNorm (a.k.a. QKV-Post) and its variant HybridNorm∗ consistently surpass other methods. Notably, HybridNorm∗ achieves the lowest training loss and perplexity while attaining the highest accuracy on HellaSwag. Specifically, by comparing HybridNorm with the left first block in Table 6, we find that QKV-Norm is the most effective normalization. Similarly, comparing HybridNorm with the left second block, we observe that combining QKV-Norm with Post-Norm in the FFN yields superior performance. From the right table, one can see that the Pre-Post configuration indeed leads to improved performance, while replacing Pre-Norm in the MHA with QKV-Norm to form HybridNorm further enhances performance, achieving the best results.

Special Treatment of First Block For the special treatment of the first block, we test different architectures, such as adding a normalization layer after embedding (call EmbedNorm) and armed the first block with QKV-norm and Pre-Norm in FFN (call First-QKV-Pre), which formulations are:

Y 0 = MHAQKV (X0) + X0, X1 = FFN(Norm(Y 0)) + Y 0. (10)

As shown in Figure 5, we can see that, except for EmbedNorm, the special treatment of the first block effectively reduces training loss and improves downstream performance.

#### 5.5 Scaling Laws Experiments

We compare the loss scaling curves between Pre-Norm and HybridNorm∗ across a range of dense model sizes, from 151M to 1.2B parameters. The model sizes used for the scaling law experiments are detailed in Table 9, and all models are trained using the same setting and hyperparameters for fair comparison, as specified in Table 10. Models with 151M, 285M, 550M, and 1.2B parameters are trained on 200B, 200B, 300B, and 1T tokens, respectively. As shown in Figure 6, HybridNorm∗ exhibits superior scaling properties,

demonstrating lower training loss as the model size increases. This highlights its capacity to maintain both training stability and performance, even for extremely large models, thereby making it highly suitable for scaling to billion-parameter regimes.

#### 5.6 Deeper Models

To further evaluate the robustness of HybridNorm and HybridNorm∗ in deeper architectures, we conduct experiments on transformers with depths ranging from 16 to 29 layers while maintaining a comparable parameter budget. This setup allows for a fair comparison of different normalization strategies in deep transformer architectures. As shown in Table 7, both HybridNorm and HybridNorm∗ consistently outperform Pre-Norm and Post-Norm across various depths, demonstrating their effectiveness in stabilizing deep model training. A particularly striking observation is that Post-Norm fails to converge at 29 layers, reinforcing its well-documented instability in deeper architectures. In contrast, HybridNorm and HybridNorm∗ not only ensure stable training across all depths but also achieve significantly lower training loss and perplexity on both the C4 and Pile datasets. These improvements indicate that HybridNorm-based normalization strategies mitigate optimization difficulties that commonly arise in deep transformers. Furthermore, HybridNorm∗ achieves the highest accuracy on challenging downstream benchmarks such as HellaSwag and PIQA, suggesting that its benefits extend beyond mere training stability to enhanced generalization on real-world tasks. These results provide strong empirical evidence that HybridNorm-based normalization schemes enable deeper transformer training while preserving superior optimization efficiency and downstream task performance.

### 6 Conclusion

In this paper, we have introduced HybridNorm, a novel hybrid normalization strategy that has seamlessly integrated the advantages of both Pre-Norm and Post-Norm, thereby addressing the longstanding tradeoffs in transformer training. We have provided both comprehensive theoretical and empirical analyses to demonstrate how HybridNorm has stabilized gradient propagation while preserving strong regularization effects, ultimately improving both convergence speed and final model performance. Extensive experiments across diverse benchmarks have substantiated the effectiveness of our approach, consistently showing that HybridNorm has outperformed conventional normalization schemes in terms of stability and accuracy. These findings have highlighted the importance of re-examining the role and placement of normalization within transformer architectures, paving the way for further exploration of hybrid normalization paradigms. We believe that HybridNorm has marked a significant step forward in the development of more robust and efficient transformer models, offering practical advantages for training next-generation large-scale neural networks.

### References

- [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

- [2] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.

- [3] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, 2020.

- [4] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, 2019.

- [5] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

- [6] DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024.
- [7] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In ICML, 2023.

- [8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019.

- [9] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. In NeurIPS, 2021.

- [10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

- [11] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

- [12] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023. URL https://zenodo.org/records/10256836.
- [13] Andrew Gordon, Zornitsa Kozareva, and Melissa Roemmele. Semeval-2012 task 7: Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In * SEM 2012: The First Joint Conference on Lexical and Computational Semantics–Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012), pages 394–398, 2012.

- [14] Suchin Gururangan, Mitchell Wortsman, Samir Yitzhak Gadre, Achal Dave, Maciej Kilian, Weijia Shi, Jean Mercat, Georgios Smyrnis, Gabriel Ilharco, Matt Jordan, Reinhard Heckel, Alex Dimakis, Ali Farhadi, Vaishaal Shankar, and Ludwig Schmidt. Openlm: a minimal but performative language modeling (lm) repository, 2023. URL https://github.com/mlfoundations/open_lm/. GitHub repository.

- [15] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021.

- [16] Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4246–4253, 2020.

- [17] Roger A Horn and Charles R Johnson. Matrix analysis. Cambridge university press, 2012.

- [18] Nandan Kumar Jha and Brandon Reagen. Relu’s revival: On the entropic overload in normalization-free large language models. arXiv preprint arXiv:2410.09637, 2024.

- [19] Guillaume Klein, Yoon Kim, Yuntian Deng, Jean Senellart, and Alexander M Rush. Opennmt: Open-source toolkit for neural machine translation. In Proceedings of ACL 2017, System Demonstrations, pages 67–72, 2017.

- [20] Pengxiang Li, Lu Yin, and Shiwei Liu. Mix-LN: Unleashing the power of deeper layers by combining pre-LN and post-LN. In The Thirteenth International Conference on Learning Representations, 2025.

- [21] Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, and Jiawei Han. Understanding the difficulty of training transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5747–5763, 2020.

- [22] Stephen Menary, Samuel Kaski, and Andre Freitas. Transformer normalisation layers and the independence of semantic subspaces. arXiv preprint arXiv:2406.17837, 2024.

- [23] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, 2018.

- [24] Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Evan Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh Hajishirzi. OLMoe: Open mixture-of-experts language models. In The Thirteenth International Conference on Learning Representations, 2025.

- [25] Toan Q Nguyen and Julian Salazar. Transformers without tears: Improving the normalization of self-attention. In Proceedings of the 16th International Conference on Spoken Language Translation, 2019.

- [26] Lorenzo Noci, Sotiris Anagnostidis, Luca Biggio, Antonio Orvieto, Sidak Pal Singh, and Aurelien Lucchi. Signal propagation in transformers: Theoretical perspectives and the role of rank collapse. In NeurIPS, 2022.

- [27] Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Michal Guerquin, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya Malik, William Merrill, Lester James V. Miranda, Jacob Morrison, Tyler Murray, Crystal Nam, Valentina Pyatkin, Aman Rangapur, Michael Schmitz, Sam Skjonsberg, David Wadden, Christopher Wilhelm, Michael Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. 2 olmo 2 furious. arXiv preprint arXiv:2501.00656, 2024.

- [28] Weronika Ormaniec, Felix Dangel, and Sidak Pal Singh. What does it mean to be a transformer? insights from a theoretical hessian analysis. In ICLR, 2025.

- [29] Siva Reddy, Danqi Chen, and Christopher D Manning. Coqa: A conversational question answering challenge. Transactions of the Association for Computational Linguistics, 7:249–266, 2019.

- [30] Oleg Rybakov, Mike Chrzanowski, Peter Dykas, Jinze Xue, and Ben Lanir. Methods of improving llm training stability. arXiv preprint arXiv:2410.16682, 2024.

- [31] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

- [32] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.

- [33] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

- [34] Sidak Pal Singh, Gregor Bachmann, and Thomas Hofmann. Analytic insights into structure and rank of neural network hessian maps. In NeurIPS, 2021.

- [35] Sho Takase, Shun Kiyono, Sosuke Kobayashi, and Jun Suzuki. B2T connection: Serving stability and performance in deep transformers. In Findings of the Association for Computational Linguistics: ACL, 2023.

- [36] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, 2019.

- [37] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

- [38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 2017.

- [39] Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, and Furu Wei. Deepnet: Scaling transformers to 1,000 layers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

- [40] Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F Wong, and Lidia S Chao. Learning deep transformer models for machine translation. In ACL, 2019.

- [41] Ya Wang, Zhijian Zhuo, Yutao Zeng, Xun Zhou, Jian Yang, and Xiaoqing Li. Scale-distribution decoupling: Enabling stable and effective training of large language models. arXiv preprint arXiv:2502.15499, 2025.

- [42] Johannes Welbl, Nelson F Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Proceedings of the 3rd Workshop on Noisy User-generated Text, pages 94–106, 2017.

- [43] Shufang Xie, Huishuai Zhang, Junliang Guo, Xu Tan, Jiang Bian, Hany Hassan Awadalla, Arul Menezes, Tao Qin, and Rui Yan. Residual: Transformer with dual residual connections. arXiv preprint arXiv:2304.14802, 2023.

- [44] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In ICML, 2020.

- [45] Zhuliang Yao, Yue Cao, Yutong Lin, Ze Liu, Zheng Zhang, and Han Hu. Leveraging batch normalization for vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, 2021.

- [46] Seniha Esen Yuksel, Joseph N Wilson, and Paul D Gader. Twenty years of mixture of experts. IEEE transactions on neural networks and learning systems, 23(8):1177–1193, 2012.

- [47] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, 2019.

- [48] Biao Zhang and Rico Sennrich. Root mean square layer normalization. In NeurIPS, 2019.

- [49] Biao Zhang, Ivan Titov, and Rico Sennrich. Improving deep transformer with depth-scaled initialization and merged attention. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 898–909, 2019.

## Appendix

### Contents

- 1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- 2 Preliminaries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 2.1 Post-Norm and Pre-Norm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 3 HybridNorm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 4 Theoretical Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 4.1 Benefits of Hybrid Method . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 4.2 Benefits of QKV-Norm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 5 Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 5.1 Experiment Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 5.2 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 5.3 More Experiment in 7B Dense Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 5.4 Ablation Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 5.5 Scaling Laws Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 5.6 Deeper Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- A Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- B Theoretical Gradient Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- B.1 Theorem 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B.2 Proof of Lemma 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B.3 Proof of Lemma 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- B.4 Proof of Theorem 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- B.5 Expand Theoretical Clarifications to LayerNorm . . . . . . . . . . . . . . . . . . . . . . . . . 25

- C Details of Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- C.1 Architectures of Different Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- C.2 Hyperparameters for Pretraining . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- C.3 PyTorch Style Implementation of HybridNorm . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- D Computational Overhead . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- E Additional Experimental Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- E.1 Comparison with Other Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- E.2 Signal Propagation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- E.3 Entropy Dynamics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- E.4 Overall Results for Dense Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- E.5 Overall Results for MoE Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

- F Essential Differences Between Pre-Norm and Post-Norm . . . . . . . . . . . . . . . . . . . 31
- G Formulas for Different Positions of Normalization Layers . . . . . . . . . . . . . . . . . . . . 32
- H Broader Impacts and Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

- H.1 Broader Impacts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- H.2 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

### A Related Work

Architecture Modifications in Transformers Recent efforts in transformer architecture modifications have sought to optimize both the computational efficiency and the expressiveness of the model. These efforts include changes to the attention mechanism and feed-forward networks all aimed at improving performance on a variety of tasks, ranging from language modeling to vision tasks [10, 38]. For example, Multi-head Latent Attention (MLA) [6], Mixture of Experts (MoE) [46]. While these modifications contribute to more efficient training, they also require careful integration with other components, such as normalization layers, to maintain model stability and performance.

Normalization Types in Transformers Normalization layers are integral to the success of deep learning models, and transformers are no exception. The most commonly used normalization technique in transformers is LayerNorm [1], which normalizes the activations of each layer independently. However, alternative methods such as RMSNorm [48], which normalizes using root mean square statistics, have been proposed as more effective alternatives in certain settings. These methods are designed to mitigate the challenges of internal covariate shift and gradient instability, which are critical for the success of large-scale transformer models.

Normalization Settings in Attention For training stability, QK-Norm [7, 16] modifies the standard attention mechanism by applying normalization directly to the query (Q) and key (K) components during attention computation. Building upon this, QKV-Norm [22, 30] extends the approach by normalizing the Query (Q), Key (K), and Value (V) components. This comprehensive normalization ensures that all critical components of the attention mechanism are normalized, resulting in enhanced stability and improved performance.

Location of Normalization Layers Recent research has also explored the impact of normalization location in both Vision Transformers [21, 45] and language models [21, 40]. For example, the choice between Pre-Norm and Post-Norm architectures has been widely studied in the transformer literature [19, 38, 40]. Pre-Norm, where normalization is applied before the residual connection, has been shown to be more stable in deep networks and accelerates convergence [44]. Although Post-Norm is more challenging to train, it tends to deliver better final performance by normalizing after the residual connection [21]. DeepNorm [39] was proposed as a strategy to address training instability in deep transformers, which scales the residual connections by a carefully chosen factor to improve gradient flow and mitigate exploding or vanishing gradients. Ding et al. [9] introduced Sandwich-LN in multimodal settings to improve training stability, a strategy that has also been adopted by the Gemma team in their recent models [37]. Similarly, OLMo-2 [27] applies the normalization layer after the sublayer but before the residual connection, differing from both traditional Pre-LN and Post-LN schemes. The method most similar to ours is Mix-LN [20], which applies Post-Norm to the earlier layers and Pre-Norm to the deeper layers, achieving improved training stability and better performance. In contrast, our HybridNorm integrates Pre-Norm and Post-Norm within each transformer block. This intra-layer hybridization offers several key advantages: (1) consistently improved model performance, (2) intra-layer hybridization ensures uniformity across all layers, facilitating other post-training such as pruning and quantization.

### B Theoretical Gradient Analysis

For simplicity, we consider a single-head attention layer. The input is X ∈ Rs×d, representing a sequence of s tokens with dimension d. Throughout this section, we denote RMSNorm1 as Norm(·), i.e., Norm(x) = α ⊙ RMS(x x) for x ∈ Rd, where RMS(x) = (x21 + ··· + x2d)/d. For further simplicity, we set α = 1d. The learnable parameters WQ,WK,WV ∈ Rd×d

XNWQWK⊤XN⊤, and A = softmax(M). The output of the attention block with Pre-Norm is then given by

k and WO ∈ Rd

k×d. Let XN = Norm(X), M = √1d

k

S = AXNWV WO. (11)

1Given that the vast majority of popular LLMs are based on RMSNorm, our experiments and conclusions are broadly applicable to standard LLM architectures. Futher, in Appendix B.5, we demonstrate that RMSNorm and LayerNorm exhibit no fundamental differences, both in theoretical analysis and empirical observations.

Defining Q = XWQ, K = XWK, and V = XWV , with their normalized counterparts QN = Norm(Q), KN = Norm(K), and VN = Norm(V ), the output of the attention block with QKV-Norm is formulated as

SN = ANVNWO, (12) where AN = softmax(MN) and MN = √1d

QNKN⊤.

k

Defining Qˆ = XNWQ and KˆN = XNWK, with their normalized counterparts QˆN = Norm(Qˆ) and KˆN = Norm(Kˆ), the output of the attention block with Pre-Norm and QK-Norm is formulated as

Sˆ = AˆNXNWV WO, (13) where AˆN = softmax(MˆN) and MˆN = √1d

QˆNKˆN⊤.

k

Following the prior work of Noci et al. [26], Ormaniec et al. [28], we analyze the gradients by computing derivatives using row-wise vectorization and arranging the Jacobian in the numerator layout, i.e.,

∂Y ∂X

∂vecr(Y ) ∂vecr(X)⊤.

=

The following derivation primarily relies on the chain rule and the following rule

∂AWB ∂W

= A ⊗ B⊤, (14)

where A ∈ Rm×n,W ∈ Rn×p,B ∈ Rp×q, and ⊗ is the Kronecker product. The proof of Eq. 14 can be found in [34].

- B.1 Theorem 2 We first present the following extension of Lemma 2 in Noci et al. [26].

- Lemma 1 (Extention of Lemma 2 in Noci et al. [26]). The gradients of the attention with Pre-Norm defined in Eq. (11) are given by

XNWQWK⊤XN⊤ √dk

∂S ∂WO

XNWV ⊗ Id, (15) ∂S ∂WV

= softmax

XNWQWK⊤XN⊤ √dk

XN ⊗ WO⊤, (16) ∂S ∂WQ

= softmax

XN ⊗ XNWK √dk

∂A ∂M

= Is ⊗ WO⊤WV⊤XN⊤

, (17) ∂S ∂WK

XNWQ ⊗ XN √dk

∂A ∂M

= Is ⊗ WO⊤WV⊤XN⊤

k,d, (18) where the gradients of the softmax with respect to its inputs is

Kd

∂A ∂M

= blockdiag

∂Ai,: ∂Mi,:

= blockdiag diag(Ai,:) − Ai,:A⊤i,: , (19)

k,d is a permutation matrix that transforms the row-wise vectorization of WK into the column-wise vectorization of WK, i.e.,

Ai,: is the i-th row of A in column vector format, and the commutation matrix Kd

k,dvecr(WK) = vecr(WK⊤).

Kd

The gradient of XN with respect to X is

∂XN ∂X

=

∂Norm(X) ∂X

= blockdiag

∂Norm(Xi,:) ∂Xi,:

= blockdiag

√

d ∥Xi,:∥2

Xi,:Xi,⊤: ∥Xi,:∥22

Id −

, (20)

∂V , ∂Qˆ

where Xi,: is the i-th row of X represented as a column vector. The definitions of ∂Q

∂Qˆ , and ∂KˆN

∂Q , ∂K

∂K , ∂V

N

N

N

N

∂Kˆ follow similarly, i.e., for • ∈ {Q,K,V,Q,ˆ Kˆ},

√dk ∥ •i,: ∥2

•i,:•⊤i,: ∥ •i,: ∥22

∂•N ∂•

. (21)

k −

= blockdiag

Id

- Lemma 2. The gradients of the attention with QKV-Norm defined in Eq. (12) are ∂SN

∂WO

= softmax

Norm(XWQ)Norm(XWK)⊤ √dk

Norm(XWV ) ⊗ Id, (22)

∂SN ∂WV

= softmax

Norm(XWQ)Norm(XWK)⊤ √dk ⊗ WO⊤

∂VN ∂V

(X ⊗ Id

k

), (23)

∂SN ∂WQ

= Is ⊗ WO⊤Norm(XWV )⊤

∂AN ∂MN

Is ⊗ Norm(XWK) √dk

∂QN ∂Q

(X ⊗ Id

k

), (24)

∂SN ∂WK

= Is ⊗ WO⊤Norm(XWV )⊤

∂AN ∂MN

Norm(XWQ) ⊗ Is √dk

Kd

k,s

∂KN ∂K

(X ⊗ Id

k

), (25)

where the definition of ∂A

N

∂MN is similar to ∂M∂A and Kd

k,s is the commutation matrix s.t., Kd

k,svecr(KN) = vecr(KN⊤).

- The proof of Lemma 2 is provided in Appendix B.2. Similarly, for the attention with Pre-Norm and QK-Norm, we derive the following lemma.

Lemma 3. The gradients of the attention with Pre-Norm and QK-Norm defined in Eq. (13) are

∂Sˆ ∂WO

= softmax

Norm(XNWQ)Norm(XNWK)⊤ √dk

XNWV ⊗ Id, (26) ∂Sˆ ∂WV

= softmax

Norm(XNWQ)Norm(XNWK)⊤ √dk

XN ⊗ WO⊤, (27) ∂Sˆ ∂WQ

= Is ⊗ WO⊤WV⊤XN⊤

∂AˆN ∂MˆN

Is ⊗ Norm(XNWK) dk

∂QˆN ∂Qˆ

(XN ⊗ Id

k

), (28) ∂Sˆ ∂WK

= Is ⊗ WO⊤WV⊤XN⊤

∂AˆN ∂MˆN

Norm(XNWQ) ⊗ Is √dk

Kd

k,s

∂KˆN ∂Kˆ

(XN ⊗ Id

k

), (29)

where the definition of ∂Aˆ

N

∂MˆN is similar to ∂M∂A and Kˆd

k,s is the commutation matrix s.t., Kˆd

k,svecr(KˆN) = vecr(KˆN⊤).

- The proof of Lemma 3 can be found in Appendix B.3.

Armed with the above lemmas, we arrive at the following theorem, which characterizes the gradient norms of Pre-Norm, Pre-Norm with QK-Norm, and QKV-Norm.

Theorem 2. For the attention with Pre-Norm, we have

√

∂S ∂WO F

d∥WV ∥2 , (30) ∂S

= O s

= O (s∥WO∥F), (31) ∂S

∂WV F

(s)3/2 2√dk ∥WK∥2∥WV ∥2∥WO∥2 , (32)

= O

∂WQ F

(s)3/2 √dk ∥WQ∥2∥WV ∥2∥WO∥2 . (33)

∂S ∂WK F

= O

For the attention with Pre-Norm and QK-Norm, we have

∂Sˆ ∂WO

√

d∥WV ∥2 , (34)

= O s

F

∂Sˆ ∂WV

= O (s∥WO∥F), (35)

F

s√sdk σminQ

∂Sˆ ∂WQ

∥WV ∥2∥WO∥2 , (36)

= O

F

s√sdk

∂Sˆ ∂WK

σminK ∥WV ∥2∥WO∥2 . (37) For the attention with QKV-Norm, we have

= O

F

√

∂SN ∂WO F

d , (38)

= O s

sdk σminV ∥WO∥2 , (39)

∂SN ∂WV F

= O

s√sdk σminQ

∂SN ∂WQ F

∥WO∥2 , (40)

= O

s√sdk

∂SN ∂WK F

σminK ∥WO∥2 , (41) where σminQ ,σminK ,σminV are minimal singular value of WQ,WK,WV , respectively.

= O

Theorem 2 presents the gradient norms of various methods, and its proof is provided in Appendix B.4. In the attention with Pre-Norm, the gradient of the weight matrix exhibits strong dependencies on other weights; for instance, WQ and WK are influenced by all three other weight matrices but not by themselves. In contrast, in the attention with QKV-Norm, the gradient of each weight matrix depends at most on itself and WO. This suggests that the gradient of the attention with Pre-Norm is more tightly coupled with other weight matrices compared to the gradient of the attention with QKV-Norm. Whereas the attention with Pre-Norm and QK-Norm lies between the two methods. Therefore, during the gradient optimization process, if the norm of a certain weight becomes excessively large, it is more challenging to control in the attention with Pre-Norm, leading to an increase in gradient magnitude. This, in turn, creates a vicious cycle that may result in model collapse. In contrast, the attention with QKV-Norm alleviates this issue to some extent, which significantly benefits the stability of model training. Regarding the degree of coupling, the relationship follows Pre-Norm > Pre-Norm with QK-Norm > QKV-Norm, whereas for training stability, the hierarchy is reversed: Pre-Norm < Pre-Norm with QK-Norm < QKV-Norm.

#### B.2 Proof of Lemma 2

- Proof of Lemma 2. For ∂S

∂WO , according to Eq. (14), we obtain

N

∂SN ∂WO

= ANVN ⊗ Id = softmax

Norm(XWQ)Norm(XWK)⊤ √dk

Norm(XWV ) ⊗ Id.

∂WV , using the chain rule and Eq. (14), we have

For ∂S

N

∂SN ∂WV

∂SN ∂VN

∂VN ∂V

∂V ∂WV

=

∂VN ∂V

= (A ⊗ WO⊤)

(X ⊗ Id

)

k

= softmax

Norm(XWQ)Norm(XWK)⊤ √dk ⊗ WO⊤

∂VN ∂V

(X ⊗ Id

).

k

∂WQ , using the chain rule and Eq. (14), we obtain

For ∂S

N

∂SN ∂WQ

∂SN ∂AN

∂AN ∂MN

∂MN ∂QN

∂QN ∂Q

∂Q WQ

=

Is ⊗ KN √dk

∂AN ∂MN

= Is ⊗ WO⊤VN⊤

= Is ⊗ WO⊤Norm(XWV )⊤

∂WK , we have

Similarly, for ∂S

N

∂SN ∂WK

∂SN ∂AN

∂AN ∂MN

∂MN ∂KN

∂KN ∂K

∂K WK

=

QN ⊗ Is dk

∂AN ∂MN

= Is ⊗ WO⊤VN⊤

∂AN ∂MN

= Is ⊗ WO⊤Norm(XWV )⊤

∂AN ∂MN

∂QN ∂Q

(X ⊗ Id

)

k

Is ⊗ Norm(XWK) dk

∂QN ∂Q

(X ⊗ Id

).

k

∂vecr(KN⊤) ∂vecr(KN)⊤

∂KN ∂K

(X ⊗ Id

)

k

Norm(XWQ) ⊗ Is √dk

∂KN ∂K

(X ⊗ Id

).

Kd

k,s

k

#### B.3 Proof of Lemma 3

- Proof of Lemma 3. For ∂W∂Sˆ

, according to Eq. (14), we obtain ∂Sˆ ∂WO

O

Norm(XNWQ)Norm(XNWK)⊤ √dk

= AˆNXNWV ⊗ Id = softmax

XNWV ⊗ Id.

For ∂W∂Sˆ

, using Eq. (14), we have

V

∂Sˆ ∂WV

= AˆNXN ⊗ WO⊤ = softmax

Norm(XNWQ)Norm(XNWK)⊤ √dk

XN ⊗ WO⊤.

For ∂W∂Sˆ

Q

, using the chain rule and Eq. (14), we obtain

∂Sˆ ∂AˆN

∂AˆN ∂MˆN

∂MˆN ∂QˆN

∂QˆN ∂Qˆ

∂Sˆ ∂WQ

=

∂AˆN ∂MˆN

= Is ⊗ WO⊤WV⊤XN⊤

∂AˆN ∂MˆN

= Is ⊗ WO⊤WV⊤XN⊤

Similarly, for ∂W∂Sˆ

, we have ∂Sˆ ∂WK

K

∂Sˆ ∂AˆN

∂AˆN ∂MˆN

∂MˆN ∂KˆN

∂KˆN ∂Kˆ

∂Kˆ WK

=

∂AˆN ∂MˆN

= Is ⊗ WO⊤WV⊤XN⊤

∂Qˆ WQ

Is ⊗ KˆN √dk

∂QˆN ∂Qˆ

(XN ⊗ Id

)

k

Is ⊗ Norm(XNWK) dk

∂QˆN ∂Qˆ

(XN ⊗ Id

).

k

Q ˆN ⊗ Is dk

∂vecr(KˆN⊤) ∂vecr(KˆN)⊤

∂KˆN ∂Kˆ

(XN ⊗ Id

)

k

| |
|---|

= Is ⊗ WO⊤WV⊤XN⊤

∂AˆN ∂MˆN

Norm(XNWQ) ⊗ Is √dk

∂KˆN ∂Kˆ

(XN ⊗ Id

Kd

).

k,s

k

| |
|---|

- B.4 Proof of Theorem 2 The proof is primarily based on the following facts

- • tr(B ⊗ C) = tr(B)tr(C)
- • (B ⊗ C)(D ⊗ E) = (BD) ⊗ (CE)
- • ∥B ⊗ C∥F = ∥B∥F∥C∥F
- • ∥B ⊗ C∥2 = ∥B∥2∥C∥2
- • ∥BC∥2 ≤ ∥B∥2∥C∥2
- • ∥BC∥F ≤ ∥B∥2∥C∥F ≤ ∥B∥F∥C∥F
- • ∥XN∥F = √s

- • If p ∈ Rs, pi ≥ 0 and si=1 pi = 1, then ∥diag(p) − pp⊤∥2 ≤ 12. Proof. According to Gershgorin Circle Theorem [17], every eigenvalue of diag(p) − pp⊤ lies within

s

i=1

[pi − p2i −

j̸=i

pipj,pi − p2i +

j̸=i

pipj]

=

s

i=1

[pi(1 − pi) − pi

j̸=i

pj,pi(1 − pi) + pi

j̸=i

pj]

=

s

i=1

[pi(1 − pi) − pi(1 − pi),pi(1 − pi) + pi(1 − pi)]

=

s

i=1

[0,2pi(1 − pi)]

⊆ [0,

- 1

- 2

].

Therefore, ∥diag(p) − pp⊤∥2 ≤ 12. When p1 = p2 = 12, the equality holds, indicating that this bound is tight.

| |
|---|

- • If A ∈ Rs×s is a stochastic matrix, i.e., A1s = 1s and each entry is nonnegative, then ∥A∥2 ≤ ∥A∥F ≤

√s. Proof. Note that

s

s

s

s

s

1 = √s.

a2ij ≤

∥A∥F =

aij =

i=1

j=1

i=1

j=1

i=1

If A = 1se1, then

1 ) = √s. Hence, the bound is tight.

∥A∥2 = λmax(A⊤A) = λmax(se1e⊤

| |
|---|

Proof of Theorem 2. According to fundamental algebraic operations and Lemma 1, we obtain ∂S ∂WO F

√

√

= ∥AXNWV ⊗ Id∥F = ∥AXNWV ∥F∥Id∥F ≤

d∥A∥2∥XN∥F∥WV ∥2 ≤ s

d∥WV ∥2,

∂S ∂WV F

= AXN ⊗ WO⊤ F = ∥AXN∥F∥WO∥F ≤ ∥A∥2∥XN∥F∥WO∥F ≤ s∥WO∥F,

XN ⊗ XNWK √dk F

∂S ∂WQ F

∂A ∂M

= Is ⊗ WO⊤WV⊤XN⊤

1 √dk

Is ⊗ WO⊤WV⊤XN⊤ blockdiag(diag(Ai,:) − Ai,:A⊤i,:)(XN ⊗ XNWK) F

=

1 √dk

blockdiag((WO⊤WV⊤XN⊤)(diag(Ai,:) − Ai,:A⊤i,:))(XN ⊗ XNWK) F

=

  

  

XN⊤1,: ⊗ WO⊤WV⊤XN⊤(diag(A1,:) − A1,:A⊤1,:))XNWK . XN⊤s,: ⊗ WO⊤WV⊤XN⊤(diag(As,:) − As,:A⊤s,:))XNWK

1 √dk

=

F

1

- 1

- 2∥WK∥2∥XN∥F

√dk ∥XN∥F∥WO∥2∥WV ∥2∥XN∥2 ≤

≤

- 1

- 2√dk ∥WK∥2∥WV ∥2∥WO∥2∥XN∥3F

(s)3/2 2√dk ∥WK∥2∥WV ∥2∥WO∥2,

=

Therefore,

XNWQ ⊗ XN √dk

∂A ∂M

∂S ∂WK F

= Is ⊗ WO⊤WV⊤XN⊤

Kd

k,d

F

XNWQ ⊗ XN √dk F

∂A ∂M

= Is ⊗ WO⊤WV⊤XN⊤

1 √dk

Is ⊗ WO⊤WV⊤XN⊤ blockdiag(diag(Ai,:) − Ai,:A⊤i,:)(XNWQ ⊗ XN) F

=

1 √dk

blockdiag((WO⊤WV⊤XN⊤)(diag(Ai,:) − Ai,:A⊤i,:))(XNWQ ⊗ XN) F

=

  

  

(XNWQ)⊤1,: ⊗ WO⊤WV⊤XN⊤(diag(A1,:) − A1,:A⊤1,:))XN . (XNWQ)⊤s,: ⊗ WO⊤WV⊤XN⊤(diag(As,:) − As,:A⊤s,:))XN

1 √dk

=

F

1

- 1

- 2∥XN∥F

√dk ∥WQ∥2∥XN∥F∥WO∥2∥WV ∥2∥XN∥2 ≤

≤

- 1

- 2√dk ∥WQ∥2∥WV ∥2∥WO∥2∥XN∥3F

(s)3/2 2√dk ∥WQ∥2∥WV ∥2∥WO∥2.

=

√

∂S ∂WO F

= O s

d∥WV ∥2 ,

∂S ∂WV F

= O (s∥WO∥F), ∂S

(s)3/2 2√dk ∥WK∥2∥WV ∥2∥WO∥2 ,

= O

∂WQ F

(s)3/2 √dk ∥WQ∥2∥WV ∥2∥WO∥2 .

∂S ∂WK F

= O

Since the attention with Pre-Norm and QK-Norm lies between Pre-Norm and QKV-Norm, its proof can be directly derived from those of the other two. Therefore, we defer its proof to the end. As for the attention with QKV-Norm, we have

∂SN ∂WO F

= ∥ANVN ⊗ Id∥F = ∥ANVN∥F∥Id∥F ≤

√

√

d∥AN∥2∥VN∥F ≤ s

d.

For ∂S

∂WV , we have

N

According to Eq. (21), we get

∂VN ∂V

(X ⊗ Id

)

k

Similarly, we can get

It follows that

∂QN ∂Q

∂VN ∂V

∂SN ∂WV F

= (A ⊗ WO⊤)

(X ⊗ Id

)

k

F ≤ A ⊗ WO⊤ 2

∂VN ∂V

(X ⊗ Id

)

k

F

√s∥WO∥2

∂VN ∂V

≤

(X ⊗ Id

.

)

k

F

√dk ∥Vi,:∥2

Vi,:Vi,⊤: ∥Vi,:∥22

k −

= blockdiag

Id

F



√dk

⊤

k − V1,:V

X1⊤,: ⊗

1,: ∥V1,:∥22

∥V1,:∥2 Id

=

. Xs,⊤: ⊗

 

√dk

⊤

k − Vs,:V

s,: ∥Vs,:∥22

∥Vs,:∥2 Id

s

∥Xi,:∥22 ∥Vi,:∥22

= dk(dk − 1)

i=1

s

∥Xi,:∥22 ∥W⊤

= dk(dk − 1)

V Xi,:∥22 ≤

i=1

√sdk σminV

.

###### (X ⊗ Id

)

k



 

F

F

(X ⊗ Id

)

k

√sdk σminQ

≤

,

F

∂KN ∂K

(X ⊗ Id

)

k

√sdk σminK

≤

.

F

∂SN ∂WV F ≤

sdk σminV ∥WO∥2. (42)

For ∂S

∂WQ , we have

N

Is ⊗ KN √dk

∂AN ∂MN

∂SN ∂WQ F

∂QN ∂Q

= Is ⊗ WO⊤VN⊤

(X ⊗ Id

)

k

F

1 √dk

∂QN ∂Q

∂AN ∂MN

Is ⊗ WO⊤VN⊤

≤

(Is ⊗ KN)

(X ⊗ Id

)

k

2

F

√sdk σminQ

1 √dk

Is ⊗ WO⊤VN⊤ blockdiag(diag((AN)i,:) − (AN)i,:(AN)⊤i,:)(Is ⊗ KN) 2

≤

√sdk σminQ

blockdiag(WO⊤VN⊤(diag((AN)i,:) − (AN)i,:(AN)⊤i,:)KN) 2

=

√sdk σminQ

1 2∥KN∥2

∥WO∥2∥VN∥2

≤

s√sdk 2σminQ

∥WO∥2.

≤

Similarly, for ∂S

∂WK , we have

N

Therefore,

s√sdk 2σminK ∥WO∥2.

∂SN ∂WK F ≤

√

∂SN ∂WO F

= O s

d ,

sdk σminV ∥WO∥2 ,

∂SN ∂WV F

= O

s√sdk σminQ

∂SN ∂WQ F

= O

∥WO∥2 ,

s√sdk σminK ∥WO∥2 .

∂SN ∂WK F

= O

Finally, we present the proof for the attention mechanism with Pre-Norm and QK-Norm. For ∂W∂Sˆ

and ∂W∂Sˆ

, whose proofs are essentially identical to that of Pre-Norm, we have

O

O

∂Sˆ ∂WO

= A ˆNXNWV ⊗ Id

F

= ∥AˆNXNWV ∥F∥Id∥F ≤

F

√

√

d∥AˆN∥2∥XN∥F∥WV ∥2 ≤ s

d∥WV ∥2,

∂Sˆ ∂WV

= A ˆNXN ⊗ WO⊤

F

= ∥AˆNXN∥F∥WO∥F ≤ ∥AˆN∥2∥XN∥F∥WO∥F ≤ s∥WO∥F.

F

and ∂W∂Sˆ

For ∂W∂Sˆ

Q

K

, whose proofs are similar to that of QKV-Norm, we have

∂Sˆ ∂WQ

= Is ⊗ WO⊤WV⊤XN⊤

F

∂AˆN ∂MˆN

1 √dk

≤

Is ⊗ WO⊤WV⊤XN⊤

∂QˆN ∂Qˆ

Is ⊗ KˆN) dk

(XN ⊗ Id

)

k

F

∂AˆN ∂MˆN

∂QˆN ∂Qˆ

(Is ⊗ KˆN)

(XN ⊗ Id

)

k

2

F

Similarly,

Hence,

1 √dk

Is ⊗ WO⊤WV⊤XN⊤ blockdiag(diag((AˆN)i,:) − (AˆN)i,:(AˆN)⊤i,:) Is ⊗ KˆN

≤

√sdk σminQ

blockdiag(WO⊤WV⊤XN⊤(diag((AˆN)i,:) − (AˆN)i,:(AˆN)⊤i,:)KˆN)

=

2

√sdk σminQ

- 1

- 2∥KˆN∥2

≤

∥WO∥2∥WV ∥2∥XN∥2

√sdk σminQ

√s

√s

- 1

- 2

∥WO∥2∥WV ∥2

≤

s√sdk 2σminQ

≤

∥WV ∥2∥WO∥2.

√sdk σminQ

2

∂Sˆ ∂WK

s√sdk 2σminQ

≤

∥WV ∥2∥WO∥2.

F

∂Sˆ ∂WO

∂Sˆ ∂WV

∂Sˆ ∂WQ

∂Sˆ ∂WK

√

= O s

F

d∥WV ∥2 ,

= O (s∥WO∥F),

F

s√sdk σminQ

∥WV ∥2∥WO∥2 ,

= O

F

s√sdk σminK ∥WV ∥2∥WO∥2 .

= O

F

| |
|---|

#### B.5 Expand Theoretical Clarifications to LayerNorm

In the following, we demonstrate that RMSNorm and LayerNorm exhibit no fundamental differences, both in theoretical analysis and empirical observations. It is worth noting that, given the vast majority of popular LLMs are based on RMSNorm, our experiments and conclusions are broadly applicable to standard LLM architectures.

Suppose the input is given by X ∈ Rs×d. Let P = I − d11d1⊤d . Then EX = X

1 d

1 d

1d1⊤d = XP. For simplicity, we omit the affine transformation in the normalization layers. It follows that

1d1⊤d ,and X − EX = X − X

X − EX V ar(X)

X − EX RMS(X − EX)

XP RMS(XP)

LayerNorm(X) =

=

=

= RMSNorm(XP).

Hence, analogous to the gradient of RMSNorm (Eq. 20), the gradient of LayerNorm is given by

∂LayerNorm(X) ∂X

∂RMSNorm(XP) ∂X

=

= blockdiag

= blockdiag

∂RMSNorm((XP)i) ∂Xi

√

(XP)i(XP)⊤i ∥(XP)i∥22

d ∥(XP)i∥2

Id −

P ,

where Xi means that the i-th column of X. Note that P is a positive semidefinite matrix with eigenvalues bounded between 0 and 1, hence ||P||2 = 1. As a result, the gradient norms of LayerNorm and RMSNorm differ only by a constant factor. This implies that the main result, Theorem 2, also holds for LayerNorm.

On the experimental side, we conducted controlled comparison using models of identical size (550M parameters), each trained on 400B tokens. Both models adopt the Pre-Norm architecture and employ the Megatron initialization scheme; the only difference lies in the normalization method—one uses RMSNorm, while the other uses LayerNorm. As shown in Table 8, the training losses of RMSNorm and LayerNorm are nearly identical, with a marginal difference of only 0.0008. In the context of large-scale models, a loss difference smaller than 0.001 is typically considered negligible.

Table 8 Training loss comparison between RMSNorm and LayerNorm under identical training settings.

Methods Training Loss

RMSNorm 2.7631 LayerNorm 2.7639

### C Details of Experiments

#### C.1 Architectures of Different Models

For dense models, we adopt a decoder-only transformer architecture akin to LLaMA 3.2 [11], with model sizes ranging from 151M to 1.2B parameters. For the MoE model, we follow the structure of OLMoE [24]. The specific architecture of models is summarized in Table 9. All experiments are conducted on NVIDIA A100-80G GPUs, utilizing 32 GPUs for dense models with fewer than 1B parameters, and 64 GPUs for the 1.2B dense model and MoE-1B-7B. Pretraining durations vary from one to ten days, depending on the model size and the number of training tokens.

Table 9 Model architecture for dense models and MoE models.

Dense-151M Dense-285M Dense-550M Dense-543M Dense-1.2B MoE-1B-7B

Model Dimension 768 1024 1536 1024 2048 2048 FFN Dimension 2048 4096 4096 4096 9192 1024 Attention heads 16 16 16 16 32 16 Key/Value Heads 4 4 4 4 8 16 Layers 12 12 16 29 16 16 Vocabulary Size 100278 100278 100278 100278 100278 50280 Weight Tying True True True True True False Context Length 4096 4096 4096 4096 4096 4096 Expert Granularity - - - - - 8 in 64

#### C.2 Hyperparameters for Pretraining

For the training hyperparameters, we primarily adopt the configuration outlined in OLMo 2 [27] and OLMoE [24]. The training hyperparameters for our models across different sizes are presented in Table 10. The model is trained using the AdamW optimizer with a learning rate (LR) of 3e-4 (4e-4 for MoE), which is scheduled

to decay following a cosine function. The minimum LR is set to 3e-5 (5e-5 for MoE) to prevent excessively small updates in the later stages of training. A weight decay of 0.1 is applied to regularize the model and prevent overfitting. The AdamW optimizer employs β1 = 0.9 and β2 = 0.95 to control the first and second momentum estimates, respectively. Gradient clipping is utilized with a threshold of 1 to mitigate the impact of large gradients during optimization. The model’s training also incorporates a warmup phase with a total of 8,388,608,000 tokens (10,485,760,000 for MoE). The initialization of the model’s parameters follows a normal distribution with a standard deviation defined as 1/

√

2.5d, where d is the model dimension. Furthermore, the initialization is truncated at 3 standard deviations to ensure a more stable starting point for training. The RoPE (Rotary Position Embedding) parameter θ is set to 500,000 (10000 for MoE), controlling the scale of position encodings. Finally, the activation function used in the model is SwiGLU, which has been shown to outperform traditional activation functions in various tasks.

Table 10 Hyperparameters for Pretraining.

Dense Model MoE-1B-7B

Optimizer AdamW AdamW Learning Rate (LR) 3e-4 4e-4 Minimum LR 3e-5 5e-5 LR Schedule cosine cosine Weight Decay 0.1 0.1

- β1 0.9 0.9
- β2 0.95 0.95 Gradient Clipping 1 1 Batch Size 1024 1024 Warmup Tokens 8,388,608,000 10,485,760,000 Init Distribution Megatron Megatron

√

√

Init std 1/

2.5d Init Truncation 3 std 3 std RoPE θ 500000 10000 Activation SwiGLU SwiGLU Load Balancing Loss Weight - 0.01 Router z-loss Weight - 0.001

2.5d 1/

#### C.3 PyTorch Style Implementation of HybridNorm

We provide a PyTorch-style implementation of HybridNorm below, and a more detailed implementation can be found at https://github.com/BryceZhuo/HybridNorm.

###### Algorithm 1 PyTorch style pseudocode for a Transformer block with HybridNorm

# q_norm , k_norm , v_norm , ffn_norm are normalization layers # attn_proj and attn_out are linear layers # attn is the attention # ffn is the feedforward network

def forward(x): # Attention block

res = x # shape (b, s, d) q, k, v = attn_proj(x).split((d, d, d), dim=-1) # shape (b, s, d) # dk = d / h, h is the number of attention heads

q, k, v = q.view(b, s, h, dk) q, k, v = q_norm(q), k_norm(k), v_norm(v) x = attn(q,k,v) # shape (b, s, d) x = attn_out(x) + res

# FFN block x = ffn_norm(x) x = ffn(x) + x

return x

### D Computational Overhead

The direct contribution of RMSNorm to the overall parameter count and computational cost of a large Transformer is, in fact, negligible. Below, we provide a detailed analysis to support this claim.

For simplicity, we consider only the MHA and the SwiGLU activation function, excluding the Embedding and Output layers. Suppose the hidden dimension is d, the intermediate FFN size is 83d, the number of layers is L, and sequence length is s. As shown in the table below, RMSNorm constitutes only a negligible fraction of the actual runtime and memory consumption in Transformer under both Pre-Norm and HybridNorm configurations. And the ratio decreases as the model size increases. Moreover, when employing GQA, the relative overhead of RMSNorm in HybridNorm is further diminished.

Parameters Each RMSNorm layer introduces d parameters. In a Pre-Norm architecture, there are two RMSNorm layers per Transformer layer, resulting in a total of 2dL parameters. In HybridNorm, there are four RMSNorm layers per Transformer layer, yielding 4dL parameters in total. Each attention block has four weight matrices (for Q, K, V, and O projections), thereby contributing 4d2L parameters in total for the MHA component. Similarly, the FFN component introduces an additional 8d2L parameters.

Computation The FLOPs for RMSNorm to process a single token vector of dimension d is (4d + 4). Since the constant term becomes negligible for any reasonably large d, this can be simplified to 4d. Accordingly, for Pre-Norm, the 2L RMSNorm operations incur a total cost of approximately 8sdL, whereas HybridNorm incurs 16sdL. In practice, however, the dominant computational overhead in a Transformer stems from matrix-vector multiplications, primarily within MHA and FFN modules. Specifically, the FLOPs for MHA amount to (8sd2 + 4s2d)L, while the FLOPs for FFN total 16sd2L.

### E Additional Experimental Results

#### E.1 Comparison with Other Methods

We compare the downstream evaluation results of 1.2B dense models trained on 200B tokens using five normalization strategies: Post-Norm, Pre-Norm, Mix-LN, HybridNorm, and HybridNorm∗. As shown in Table 12, HybridNorm∗ consistently delivers the highest average performance across eight downstream tasks, outperforming all other methods both on average and in the majority of individual cases.

In particular, HybridNorm∗ achieves the top scores on HellaSwag (59.56), SciQ (90.70), ARC-C (36.15), PIQA (73.83, and COPA (80.40), while remaining competitive on the remaining tasks. In contrast to Post-Norm and

- Table 11 Parameter and computation cost comparison between Pre-Norm and HybridNorm architectures. L denotes the number of layers, d the hidden dimension, and s the sequence length.

Type Architecture RMSNorm

Main Transformer Components (MHA & FFN)

Ratio (RMSNorm / Total)

Parameter Pre-Norm 2dL 12d2L

2dL 12d2L + 2dL ≈

1 6d HybridNorm 4dL 12d2L

4dL 12d2L + 4dL ≈

1 3d

Computation (FLOPs)

Pre-Norm 8sdL (24sd2 + 4s2d)L

8sdL (24sd2 + 4s2d)L

=

2 6d + s HybridNorm 16sdL (24sd2 + 4s2d)L

16sdL (24sd2 + 4s2d)L

=

4 6d + s

- Table 12 Downstream evaluation results of 1.2B dense models with Post-Norm, Pre-Norm, Mix-LN, HybridNorm, and HybridNorm∗ under 200B training tokens. OQA refers to OpenbookQA.

Methods BasicArithmetic HellaSwag SciQ ARC-C ARC-E PIQA OQA COPA Avg.↑ Post-Norm 37.07 57.44 88.86 35.85 66.32 73.23 37.62 75.80 59.02 Pre-Norm 40.06 57.38 89.78 32.14 67.12 73.04 36.58 80.40 59.56 Mix-LN 33.23 56.91 89.20 33.78 69.30 72.25 37.00 79.00 58.83 HybridNorm 35.29 58.24 88.36 35.69 68.63 73.45 37.02 76.80 59.18 HybridNorm∗ 39.46 59.56 90.70 36.15 68.25 73.83 37.00 80.40 60.67

Pre-Norm, which show strong results on select benchmarks but suffer from variability elsewhere, HybridNorm∗ exhibits robust and consistently balanced performance. Notably, it attains an average score of 60.67, surpassing the second-best method (Pre-Norm, 59.56) by over one point, underscoring its effectiveness in enhancing generalization across a diverse set of evaluation tasks.

We also observe that Post-Norm underperforms Pre-Norm in both Table 12 and Table 6. This can be attributed to the fact that, although Post-Norm generally offers a higher performance upper bound, it suffers from reduced training stability and requires extensive hyperparameter tuning [21, 43, 44]. In our experiments, we adopt the default hyperparameters from OLMo 2 without performing extensive hyperparameter search; hence, it is unsurprising that Post-Norm does not outperform Pre-Norm under these settings. This observation motivates us to combine the higher performance ceiling of Post-Norm with the superior training stability and hyperparameter robustness of Pre-Norm. Our experimental results show that HybridNorm achieves stronger robustness to hyperparameters, enabling a balanced trade-off between stability and performance without the need for extensive tuning.

We also have extended Table 6 to include DeepNorm [39], Sandwich-LN [9], and OutputNorm (OLMo 2 [27]), as shown in Table 13. We have also clarified in the revised manuscript that OutputNorm refers to the normalization strategy employed in OLMo 2, where RMSNorm is applied to the outputs of attention and MLP sublayers.

It shows that HybridNorm consistently outperforms all other normalization strategies, including the aforementioned advanced methods, across all metrics. This highlights the strength of our approach in both generative perplexity tasks and HellaSwag.

#### E.2 Signal Propagation

Following [26], we plotted the evolution of the cosine similarity between tokens during pretraining. As shown in Figure 7, both Pre-Norm and Mix-LN exhibit a notable increase in token similarity in certain layers, indicating a tendency toward representation degeneration. In contrast, HybridNorm maintains consistently

| | |
|---|---|
| |[Figure 14]|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
|[Figure 15]| |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
|[Figure 16]| |
| | |
| | |
| | |
| | |

[Figure 17]

| | | |
|---|---|---|
| |[Figure 18]| |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | |
|---|---|
| | |
| |[Figure 19]|
| | |
| | |
| | |
| | |

| | | |
|---|---|---|
| | |[Figure 20]|
| | | |
| | | |
| | | |
| | | |
| | | |

[Figure 21]

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

| | | |
|---|---|---|
| | | |
| |[Figure 22]| |
| | | |
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | |[Figure 23]|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | |[Figure 24]| |
| | | | |
| | | | |
| | | | |

[Figure 25]

| | | | |
|---|---|---|---|
| | |[Figure 26]| |
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | |[Figure 27]|
| | | |
| | | |
| | | |

[Figure 28]

[Figure 29]

- Figure 7 Evolution of the cosine similarity between tokens for Post-Norm, Pre-Norm, Mix-LN, and HybridNorm at different layers.

lower similarity across most layers, suggesting better capabilities in information representation [26]. This highlights the benefit of employing QKV-Norm in the attention module and Post-Norm in FFN to improve information flow.

#### E.3 Entropy Dynamics

Following [18], we also examine the layerwise entropy evolution during pre-training. From Figure 8, HybridNorm maintains entropy within a relatively stable range, avoiding the sharp fluctuations observed in other methods. A stable entropy distribution is known to correlate with smoother training dynamics and improved optimization stability [18], providing further empirical support for the robustness of HybridNorm.

#### E.4 Overall Results for Dense Models

Overall results for the dense model are presented in Figure 9, depicting validation losses and downstream evaluations over 1T training tokens. The comparison includes models with Pre-Norm, HybridNorm, and HybridNorm∗. One can see that both HybridNorm and HybridNorm∗ outperform Pre-Norm, with HybridNorm∗ achieving the lowest training and validation losses while delivering the best downstream performance on average.

Table 13 Comparison of normalization methods on 550M models.

Methods Training Loss↓ C4 PPL↓ Pile PPL↓ HellaSwag↑

Post-Norm 2.760 20.43 10.57 51.20 Pre-Norm 2.751 20.30 10.48 51.97 Mix-LN 2.760 20.43 10.56 51.29 DeepNorm 2.746 20.34 10.48 52.11 Sandwich-LN 2.751 20.45 10.58 52.07 OutputNorm (OLMo 2) 2.750 20.34 10.44 52.82

HybridNorm 2.737 20.00 10.29 53.35 HybridNorm∗ 2.731 19.85 10.25 53.36

#### E.5 Overall Results for MoE Models

Overall results for the MoE model are presented in Figure 10, illustrating validation losses and downstream evaluations over 500 billion training tokens. The comparison focuses on models employing Pre-Norm and HybridNorm∗. From the figures, we can see that HybridNorm∗ achieves lower training loss and validation loss compared to Pre-Norm on all datasets, such as C4, Books, and Pile. Additionally, HybridNorm∗ outperforms Pre-Norm on most downstream tasks, though there are some cases where it underperforms. On average, however, HybridNorm∗ demonstrates superior downstream performance.

### F Essential Differences Between Pre-Norm and Post-Norm

To facilitate a unified analysis of the various normalization variants, we propose a categorization of Pre-Norm and Post-Norm based on their distinct approaches to residual connections (refer to Figure 11, particularly the sections highlighted by the dashed boxes). From this perspective, the FFN sublayer in HybridNorm can be considered as adopting a connection scheme similar to the Post-Norm.

[Figure 30]

Figure 11 A unified view of Pre-Norm and Post-Norm

| | | | | |
|---|---|---|---|---|
| | |[Figure 31]| | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | |[Figure 32]| | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

[Figure 33]

[Figure 34]

| | | | | |
|---|---|---|---|---|
| | |[Figure 35]| | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | |[Figure 36]| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | |[Figure 37]|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | |[Figure 38]| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | |[Figure 43]|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | |[Figure 44]| | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

[Figure 45]

[Figure 46]

- Figure 8 Evolution of the layerwise entropy for Post-Norm, Pre-Norm, Mix-LN, and HybridNorm at different layers.

### G Formulas for Different Positions of Normalization Layers

In this section, we present the mathematical formulations for various normalization techniques. We begin by introducing the normalization layer within the attention mechanism.

Vanilla scaled dot-product attention are show in Eq. 1, and attention with QKV-Norm is defined in Eq. 5. Similarly, attention with QK-Norm is defined as

attnQK(Q,K,V ) = softmax

Attention with KV-Norm is defined as

attnKV (Q,K,V ) = softmax

Norm(Q)Norm(K)⊤ √dk

V. (43)

QNorm(K)⊤ √dk

Norm(V ). (44)

As mentioned in Section 5.4, we extend traditional normalization approaches by considering not only the Query (Q), Key (K), and Value (V) components but also the Context (C), which refers to the output of the

###### Figure 9 Overall loss and downstream evaluations for the 1.2B dense models with 1T training tokens.

###### Figure 10 Overall loss and downstream evaluations for the MoE models with 500B training tokens.

attention mechanism. And attention with QKVC-Norm is defined as

attnQKV C(Q,K,V ) = Norm softmax

Norm(Q)Norm(K)⊤ √dk

Norm(V ) . (45)

##### Attention with QKC-Norm is defined as

attnQKC(Q,K,V ) = Norm softmax

Norm(Q)Norm(K)⊤ √dk

V . (46)

Attention with KC-Norm is defined as

attnKC(Q,K,V ) = Norm softmax

QNorm(K)⊤ √dk

V . (47)

Then we denote MHA with attn# as MHA# for # ∈ {QKV C,QKV,QKC,QK,KV,KC}, MHA(X)# = Concat(head1,...,headh)WO, (48)

where headi = attn#(Qi,Ki,Vi) for i = 1,2,...,h, {•i}hi=1 = Split(XW•) for • ∈ {Q,K,V }, and WQ,WK,WV ,WO ∈ Rd×d are learnable parameters.

With the aforementioned definitions in hand, we present the mathematical formulations for the methods discussed in the Ablation Study below (# ∈ {QKV C,QKV,QKC,QK,KV,KC}).

#-Post:

Y l = MHA#(Xl) + Xl, (49)

Xl+1 = FFN(Norm(Y l)) + Norm(Y l). (50) #-Pre:

Y l = MHA#(Xl) + Xl, (51) Xl+1 = FFN(Norm(Y l)) + Y l. (52)

##### Pre-#-Post:

Y l = MHA#(Norm(Xl)) + Xl, (53) Xl+1 = FFN(Norm(Y l)) + Norm(Y l). (54)

##### Pre-#-Pre:

Y l = MHA#(Norm(Xl)) + Xl, (55) Xl+1 = FFN(Norm(Y l)) + Y l. (56)

##### Pre-Post:

##### Post-Pre:

Y l = MHA(Norm(Xl)) + Xl, (57) Xl+1 = FFN(Norm(Y l)) + Norm(Y l). (58)

Y l = MHA(Norm(Xl)) + Norm(Xl), (59) Xl+1 = FFN(Norm(Y l)) + Y l. (60)

### H Broader Impacts and Limitations

#### H.1 Broader Impacts

This paper proposes HybridNorm, a simple yet effective hybrid normalization technique that improves the training stability and performance of transformers. It has the potential to assist the LLM community in advancing transformer architectures and enhancing their overall effectiveness. While there may be societal implications of our work, none of which we feel must be specifically highlighted here.

#### H.2 Limitations

First, due to limited computational resources, our experiments are conducted on models ranging from 151M to 7B parameters. While our method shows strong effectiveness on smaller models, its performance on largerscale models has not yet been empirically validated. Second, although our theoretical analysis demonstrates improved gradient stability, this does not directly guarantee better overall model performance.

