## Scale-Distribution Decoupling: Enabling Stable and Effective Training of Large Language Models

Ya Wang † *1 Zhijian Zhuo*12 Yutao Zeng*1 Xun Zhou1 Jian Yang Xiaoqing Li3

[Figure 1]

### Abstract

Training stability is a persistent challenge in the pre-training of large language models (LLMs), particularly for architectures such as Post-Norm Transformers, which are prone to gradient explosion and dissipation. In this paper, we propose Scale-Distribution Decoupling (SDD), a novel approach that stabilizes training by explicitly decoupling the scale and distribution of the weight matrix in fully-connected layers. SDD applies a normalization mechanism to regulate activations and a learnable scaling vector to maintain well-conditioned gradients, effectively preventing gradient explosion and dissipation. This separation improves optimization efficiency, particularly in deep networks, by ensuring stable gradient propagation. Experimental results demonstrate that our method stabilizes training across various LLM architectures and outperforms existing techniques in different normalization configurations. Furthermore, the proposed method is lightweight and compatible with existing frameworks, making it a practical solution for stabilizing LLM training. Code is available at https: //github.com/kaihemo/SDD.

# arXiv:2502.15499v2[cs.CL]25Feb2025

###### 1.5X 1.5X

Figure 1. Training/validation loss with downstream performance on HellaSwag and PIQA for 1B dense models trained with 2T tokens: SDD-1B (Post-Norm) achieves superior convergence (1.5×) and generalization over OLMo2-1B (Pre-Norm).

hindering the efficient and effective training of these models. Although Pre-Norm Transformer (Xiong et al., 2020; Zhuo et al., 2025) architectures exhibit greater stability during training, they suffer from feature collapse (Wang et al., 2024a; Xie et al., 2023), where representations across different layers become increasingly similar as depth increases. This phenomenon may contribute to the scaling bottleneck in large models. On the other hand, Post-Norm configurations remain significantly more difficult to train, exhibiting severe gradient explosion or vanishing issues, making stability in such settings a challenge in LLM research.

### 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable success in various natural language processing tasks (Li et al., 2024; Zhu et al., 2024; Huang et al., 2025), fueled by advances in model architectures, large-scale datasets, and computational resources. However, the training stability of LLMs remains a critical challenge, especially as model size and complexity continue to grow. Instabilities during pre-training often lead to issues such as gradient explosion, vanishing gradients, or optimization stagnation,

A fundamental source of these instabilities lies in the complexity of optimizing weight matrices in high-dimensional spaces. Specifically, the scale of weight parameters becomes challenging to regulate as the matrix grows in size, making convergence increasingly delicate. While existing strategies, such as sophisticated initialization schemes (Zhang et al., 2019) and normalization techniques (Ding et al., 2021; Xiong et al., 2020), offer partial mitigation, they fail to resolve the core issue: the entanglement between the weight matrix’s scale and distribution. This coupling induces suboptimal optimization dynamics, amplifying training insta-

*Equal contribution †Project lead 1Seed-Foundation-Model, ByteDance 2School of Mathematical Sciences, Peking University 3Capital University of Economics and Business. Correspondence to: Xiaoqing Li <xqli@cueb.edu.cn>.

bilities, particularly in large-scale models where gradient propagation is susceptible to divergence or attenuation.

Self-attention Self-attention with SDD

Output

Output

FC

SDD

To tackle these challenges, we introduce Scale-Distribution Decoupling (SDD), a novel approach that restructures fullyconnected layers to explicitly separate the scale and distribution of weight matrices. In contrast to conventional formulations, SDD applies a normalization step to standardize activations, ensuring optimization focuses on learning the distribution rather than jointly optimizing both scale and distribution. Additionally, a learnable scaling vector is introduced to control the overall magnitude of activations, preventing gradient explosion and dissipation. This decoupling leads to more stable gradient propagation, enhancing both convergence efficiency and model robustness.

[Figure 2]

[Figure 3]

scale & softmax

scale & softmax

Query Key Value

Query Key Value

FC FC

FC

SDD SDD

SDD

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Input

Input

FFN

silu

FC

gate

[Figure 10]

FC output

Input

SDD is lightweight, requires minimal architectural modifications, and seamlessly integrates with a wide range of model configurations. Empirical evaluations demonstrate that SDD significantly improves training stability across various LLM architectures, including notoriously unstable Post-Norm Transformers. Furthermore, SDD accelerates convergence, improves generalization, and enables efficient large-scale pre-training, making it a practical and effective solution for stabilizing LLM training.

[Figure 11]

FC

[Figure 12]

FFN with SDD

silu

SDD

gate

[Figure 13]

SDD output

Input

[Figure 14]

SDD

[Figure 15]

Figure 2. Comparison of vanilla and SDD-based Self-Attention /FFN Architectures. The top-left figure shows the standard selfattention module, while the top-right presents the self-attention module with SDD. Similarly, the middle figure depicts the standard feed-forward network (FFN), and the bottom shows the SDD-based FFN. In these figures, “FC” represents a fully-connected layer, and “SDD” denotes the SDD-based fully-connected layer, formulated as Eqn. 1. Labels beneath “FC” and “SDD” indicate their learnable parameters. Notably, the additional parameter α in “SDD” is a one-dimensional vector, contributing negligible overhead.

This work makes the following key contributions:

- 1. We introduce a novel design that explicitly decouples the scale and distribution of weight matrices, addressing a fundamental limitation in LLM optimization.
- 2. We empirically demonstrate that SDD stabilizes training across diverse LLM architectures, including both Pre-Norm and Post-Norm configurations, mitigating issues such as gradient explosion and dissipation.
- 3. We provide empirical evidence showing that our method improves both convergence stability and training efficiency, making it highly applicable to largescale pre-training tasks.

high-dimensional weight matrices. Specifically, the scale of weight parameters has a profound impact on model outputs and gradient magnitudes but is inherently difficult to learn effectively. Existing techniques, such as advanced initialization schemes and normalization strategies, provide partial mitigation but fail to address a fundamental issue: the entanglement of the weight matrix’s scale and distribution. This entanglement introduces unnecessary complexity to the optimization process, especially in Post-Norm Transformers, which are more susceptible to instability.

The structure of this paper is organized as follows: Section

- 2 elaborates on the proposed methodology in detail. Section
- 3 provides a theoretical analysis of the principles underpinning SDD. Experimental results and an in-depth analysis are presented in Section 4. Section 5 discusses related work on training stability and normalization techniques for large language models. Finally, Section 6 concludes the paper with key insights and potential directions for future research.

To address this issue, we propose Scale-Distribution Decoupling (SDD), which disentangles the scale and distribution of weights in fully-connected layers. By isolating these two components, SDD not only simplifies the learning dynamics but also notably improves the training stability.

### 2. Scale-Distribution Decoupling

#### 2.2. Method

#### 2.1. Motivation

In conventional fully-connected layers, the output is computed as y = Wx, where W ∈ Rn×n represents the learnable weight matrix and x ∈ Rn is the input vector. The

The training stability of large language models (LLMs) is frequently undermined by the challenges of optimizing

SDD formulation modifies this operation as follows:

y = α ⊙ norm(V x), (1)

where V ∈ Rn×n is a learnable weight matrix, ⊙ denotes the element-wise multiplication. norm(·) is a normalization function that removes scale information while preserving the distribution of V x and norm(x) = ∥xx∥ with ∥x∥ = (x21 + x22 + ··· + x2n)/n, following the normalization commonly used in Layer Normalization (LN) (Ba, 2016; Wang et al., 2022). α is a learnable scaling vector to stabilize training during early stages (Figure 2).

This reformulation separates the roles of the weight matrix: norm(V x) captures the distributional characteristics, while α independently governs the scale. Such a decoupling has two key advantages. First, it simplifies optimization by disentangling scale and distribution, reducing complex parameter interactions that hinder learning. Second, normalization ensures bounded outputs, which inherently prevents gradient-related issues such as explosion or vanishing. These properties make SDD particularly effective for training deep and wide models, improving convergence and stability in challenging architectures.

SDD introduces minimal computational and memory overhead compared to standard fully-connected layers. The additional FLOPs for SDD are 6BSH, where B is the batch size, S is the sequence length, and H is the hidden size, accounting for only 3/H of the total model FLOPs. The parameter overhead is similarly negligible, adding just m parameters from the scaling vector α, contributing 1/H to the total parameter count. Given that H > 1024 in typical settings, both FLOPs and parameter overheads are negligible. Furthermore, SDD’s additional memory cost can be effectively eliminated through gradient checkpointing, making it a lightweight yet effective modification.

Its proof, demonstrating the approximate expressiveness between standard and SDD-based layers, is provided in Appendix A. The expectation symbol E is omitted for brevity.

This equivalence encapsulates the fundamental principle of Scale-Distribution Decoupling (SDD): disentangling the scale and distribution of the weight matrix W. SDD achieves this by introducing a learnable scaling vector α to regulate magnitude, while norm(V x) preserves the distributional structure of the transformed input. By explicitly decoupling these components, SDD streamlines optimization, obviating the need to simultaneously learn both scale and distribution. This separation enhances numerical stability, as α facilitates precise control over output magnitudes, while normalization ensures a well-conditioned distribution. Furthermore, SDD exhibits strong adaptability, seamlessly accommodating both orthogonal and general weight matrices V , making it a versatile and robust solution across diverse neural architectures.

#### 3.2. Gradient Analysis: Standard vs. SDD Layers

The gradients with respect to α, V , and x in the SDD-based formulation y = α ⊙ norm(V x) differ significantly from those in the standard fully-connected layer y = Wx:

- 1. The gradient with respect to α is well-conditioned and bounded, enabling faster and more stable optimization of the scale parameter.
- 2. The gradient with respect to V is constrained by the normalization operation, ensuring bounded updates and avoiding gradient explosion or vanishing.
- 3. The gradient norm with respect to x is moderated by the normalization operation, preventing gradient explosion or vanishing.

### 3. Theoretical Analysis

The SDD method is supported by a theoretical foundation that demonstrates its validity and advantages under common assumptions. To begin, we show that the proposed decoupling is equivalent to the standard fully-connected operation under Gaussian assumptions.

#### 3.1. Expressiveness of Standard and SDD-Based Layers

Let x ∈ Rn be sampled from a standard Gaussian distribution N(0,I), and each element of W ∈ Rn×n be i.i.d. Gaussian random variables with mean 0 and variance σ2/n. For any fully-connected layer y = Wx, there exists an approximate representation y = α ⊙ norm(V x), where α ∈ Rn is a vector and V ∈ Rn×n is an matrix derived from W. Conversely, any output of the form y = α ⊙ norm(V x) can be approximately represented in the form y = Wx.

Proof. For the standard fully-connected layer y = Wx, the gradient with respect to W, which encodes both scale and distributional properties, is:

∂L ∂W

=

∂L ∂y · x⊤, (2)

where ∂L

∂y is the backpropagated gradient. The magnitude of ∂L

∂W is highly sensitive to the initialization of both W and x. Poorly scaled W or x can lead to gradient explosion or vanishing, complicating optimization. In contrast, the SDDbased formulation y = α ⊙ norm(V x) decouples these components, leading to the following gradient properties:

Gradient with Respect to α: The scale parameter α, is explicitly learned in the SDD formulation, with its gradient given by:

∂L ∂α

∂L ∂y ⊙ norm(V x). (3)

=

Since norm(V x) is bounded due to the normalization operation, ∂L

∂α remains stable and well-conditioned throughout training. Unlike the standard formulation, where scale and distribution are entangled in W, the decoupling in SDD allows α to be optimized independently. This results in consistently larger and more stable gradient updates for α, enabling faster convergence of the scale parameter.

Gradient with Respect to V : The distributional characteristics of the input are controlled by V in the SDD formulation. Given z = V x, the gradient of the loss function L with respect to V is expressed as:

∂L ∂V

∂L ∂y ·

=

∂y ∂V

. (4)

Since y = α ⊙ norm(z), we have:

∂y ∂V

∂norm(z) ∂V

. (5)

= α ⊙

The chain rule gives:

∂norm(z) ∂V

=

∂norm(z) ∂z ·

∂z ∂V

. (6)

Using the formula for the gradient of the normalized vector:

zz⊤ n∥z∥2

∂norm(z) ∂z

1 ∥z∥

, (7)

I −

=

and ∂V∂z = x⊤. Substituting this into the gradient of L with respect to V :

zz⊤ n∥z∥2

∂L ∂V

∂L ∂y · I −

α ∥z∥

· x⊤. (8)

⊙

=

Next, assuming that V and x are i.i.d. with elements following a standard normal distribution N(0,σ2), we further simplify the expression. Let ∥V ∥F denote the Frobenius norm of V , which is defined as:

##### ∥V ∥F =

Vi,j2 . (9)

i,j

Incorporating this definition, the gradient becomes:

zz⊤ n∥z∥2

x⊤ ∥x∥

∂L ∂V ≈

∂L ∂y · I −

α ∥V ∥F

. (10)

⊙

·

A key observation is that ∂L

∂y remains stable across layers, with its magnitude exhibiting minimal fluctuations as it propagates through the network. This stability will be formally demonstrated in the subsequent gradient analysis with respect to x. Consequently, the gradient norm of ∂L

∂V is primarily determined by ∥V ∥F, ensuring robustness during training. Furthermore, this stability enables precise control

over ∂L

∂V via adjusting the standard deviation (std) of V . By simply initializing V with small values, we can enhance convergence speed and improve overall training efficiency.

Gradient with Respect to x: In the standard fully-connected layer, the gradient of the loss L toward the input x is:

∂L ∂x

∂L ∂y

= W⊤ ·

. (11)

The gradient depends entirely on the transpose of the weight matrix W and the backpropagated gradient ∂L

∂y . In this formulation, the gradient magnitude is sensitive to the scale and condition of W, meaning poorly scaled or ill-conditioned weight matrices can lead to gradient explosion or dissipation. Large singular values in W amplify the gradient norm, resulting in unstable optimization due to gradient explosion, while small singular values reduce the gradient norm, leading to gradient dissipation and slowed convergence.

The SDD formulation y = α ⊙ norm(V x) incorporates a normalization step for V x, fundamentally altering the gradient behavior. For the gradient with respect to x :

zz⊤ n∥z∥2

∂L ∂x ≈

∂L ∂y · I −

α ∥x∥

V ∥V ∥F

, (12)

⊙

Due to the SDD network design, hidden embedding x typically follows a standard normal distribution N(0,1). According to Theorem 3.1.1 (Vershynin, 2018), ∥x∥ lies within a small neighborhood of 1, i.e., ∥x∥ ≈ 1. For simplicity, we set ∥x∥ = 1 by default. Hence, the gradient norm becomes:

∂L ∂x ∥ ≈ ∥

∂L ∂y ∥. (13)

∥

This equality implies that the gradient magnitude is preserved during backpropagation, neither exploding nor vanishing. The combination of normalization and initialization ensures that the network maintains stable gradients, regardless of the depth or dimensionality of the layers.

| |
|---|

SDD enhances training stability by disentangling the scale and distributional components of the weight matrix. By introducing normalization into all fully-connected layers, SDD ensures gradients remain bounded, mitigating gradient explosion and dissipation. The learnable scaling vector α independently controls the scale, while the normalized transformation norm(V x) isolates the distribution, improving the conditioning of V . These properties simplify optimization, enabling more robust and efficient training, especially in architectures prone to instability such as Post-Norm Transformers or high-dimensional layers. By addressing core challenges in large-scale neural network training, SDD provides a versatile and effective framework for stability and scalability.

### 4. Experiment

We evaluate SDD on both dense and MoE models, measuring training stability, convergence speed, and downstream performance. Our experiments include large-scale benchmarks, ablation studies, and robustness tests. Results show that SDD consistently improves training efficiency, mitigates instability, and outperforms existing normalization techniques across various architectures and tasks.

#### 4.1. Experimental Setup

Backbones. We evaluate SDD on two Transformer architectures: a 1B dense model and an MoE model with 588M active parameters (3.4B in total), both using Pre-Norm as the baseline. The dense model follows OLMo2 (OLMo et al., 2024) with 16 layers, dmodel = 2048, 32 heads, and GQA (8 groups). The MoE model follows OLMoE (Muennighoff et al., 2024) with 32 layers, dmodel = 1024, 16 heads, and 64 experts (8 active per token). Both models are trained from scratch for fair evaluation, with architectural details summarized in Table 3 and full configurations provided in Appendix B. All models are trained on the OLMoE Mix dataset (Muennighoff et al., 2024). We compare SDD against Pre-Norm (baseline), Post-Norm (Vaswani et al., 2017), and DeepNorm (Wang et al., 2024a).

Training Setup. We train all models using the AdamW optimizer (β1 = 0.9,β2 = 0.95) on 4096-token sequences. Baseline models follow OLMo2 (Groeneveld et al., 2024) and OLMoE (Muennighoff et al., 2024) initialization, combining truncated normal (Groeneveld et al., 2024) and Megatron-Init (Shoeybi et al., 2019). In SDD, the parameter α is initialized as 1/√layers for the output mappings of the attention and feed-forward networks (FFNs), and as 1 for other projections. The remaining parameters are initialized using a normal distribution N(0,1/√2.5 · dmodel), ensuring that the initial outputs are aligned with the baselines. The dense model uses a learning rate of 3e−4 (decaying to 1.5e−5), while the MoE model starts at 4e−4, both following a cosine schedule. Training is conducted on 64 NVIDIA H800 80GB GPUs with a global batch size of 1024 and a micro-batch size of 4 per device, using next-token prediction loss (NLL). We also use gradient clipping (max norm 1.0) and BF16 mixed precision for stable and efficient training.

Evaluation. We evaluate SDD across benchmarks covering reasoning, commonsense understanding, and question answering. Reasoning tasks include ARC-Easy, ARCChallenge (Clark et al., 2018), PIQA (Bisk et al., 2020), and MMLU (Hendrycks et al., 2021). Commonsense understanding is assessed via HellaSwag (Zellers et al., 2019), Winogrande (Sakaguchi et al., 2021), SocialIQA (Sap et al., 2019), and CSQA (Talmor et al., 2019). For question answering, we use SciQ (Welbl et al., 2017), CoQA (Reddy et al., 2019), BoolQ (Clark et al., 2019), COPA (Gordon

- Figure 3. Training and validation loss on C4 for dense models trained with 200 billion tokens. A comparison of OLMo2-1B (PreNorm), DeepNorm-1B (Post-Norm), PostNorm-1B (Post-Norm), and SDD-1B (Post-Norm) highlights the superior convergence and stability of SDD-1B.

- Figure 4. Downstream performance on MMLU, HellaSwag, ARCChallenge, and OpenbookQA for dense models trained on 200B tokens. SDD-1B consistently outperforms others, showcasing superior generalization.

et al., 2012), and OBQA (Mihaylov et al., 2018). Performance is measured via accuracy and loss using the LM Eval Harness framework (Gao et al., 2023).

#### 4.2. Results on Dense Model

We evaluate SDD on OLMo2-1B, a 1B-parameter dense model using Pre-Norm, comparing it to PostNorm-1B, DeepNorm-1B, and SDD-1B. PostNorm-1B and DeepNorm1B are trained on 200B tokens, while OLMo2-1B and SDD1B are trained on 2T tokens. For consistency, we report 200B token results here, with all evaluation metrics provided in Appendix C, while full training dynamics for the 2T token runs are shown in Figure 1. All models share identical hyperparameters, except DeepNorm-1B, which follows its official initialization scheme, ensuring a fair comparison.

Training Dynamics of 1B Dense Model. Figure 3 shows the training and validation loss on C4 for 1B dense mod-

- Table 1. Performance comparison of the 1B dense models. This table compares training loss and downstream accuracy (%). “ARC-E” and “ARC-C” denote ARC-Easy and ARC-Challenge. The best results are in bold, and “Avg.” represents average accuracy across tasks. SDD-1B achieves the best performance, demonstrating superior efficiency and generalization.

#### Model Loss ↓ MMLU HellaSwag ARC-C ARC-E Winogrande Openbook QA COPA Avg. ↑

OLMo2-1B 2.70 34.06 56.98 34.11 66.90 58.25 35.80 78.00 52.01 PostNorm-1B 2.69 32.94 57.78 32.66 65.96 58.22 37.33 79.33 52.03 DeepNorm-1B 2.72 33.06 55.73 31.77 65.09 55.99 35.67 79.67 51.00 SDD-1B 2.65 34.71 59.65 37.57 69.65 59.06 37.33 80.33 54.04

els trained with 200B tokens. Among OLMo2-1B (PreNorm), PostNorm-1B, DeepNorm-1B (both Post-Norm), and SDD-1B (Post-Norm), SDD-1B converges faster and reaches the lowest loss. It achieves 2.65, outperforming OLMo2-1B (2.70), PostNorm-1B (2.69), and DeepNorm1B (2.72), demonstrating superior stability and efficiency. These results highlight SDD’s ability to improve optimization by decoupling scale and distribution.

Figure 5. Training and Validation Loss on C4 for MoE Models with 250 Billion Tokens: Comparison of OLMoE-588M-3B (PreNorm) and SDD-588M-3B (Post-Norm).

Dowmstream Evaluation. Table 1 and Figure 4 summarize downstream results across MMLU, HellaSwag, ARCChallenge, ARC-Easy, Winogrande, Openbook QA, and COPA. SDD-1B consistently outperforms its counterparts, achieving the highest average accuracy of 54.04%, surpassing OLMo2-1B (52.01%), PostNorm-1B (52.03%), and DeepNorm-1B (51.00%). Notable gains include a 3.46% and 2.67% improvement over the second-best model on ARC-Challenge (37.57%) and HellaSwag (59.65%), respectively. These results reinforce SDD-1B’s effectiveness in capturing complex linguistic patterns and improving generalization across diverse benchmarks.

#### 4.3. Results on MoE Model

We evaluate SDD on OLMoE-588M-3B, an MoE model with 588M active parameters out of 3.4B total (Muennighoff et al., 2024). Due to computational constraints, we compare it to the baseline OLMoE-588M-3B with identical hyperparameters. SDD introduces only a 0.1% increase in parameters due to the learnable scaling vector α, ensuring a fair comparison without modifying training settings.

Figure 6. Downstream performance on MMLU, HellaSwag, ARCChallenge, and Commonsense for MoE models with 250 billion training tokens.

Training dynamics of MoE model. Figure 5 presents the training and validation loss curves for MoE models trained on 250B tokens. SDD-588M-3B consistently achieves lower losses than OLMoE-588M-3B, demonstrating improved convergence and stability. This suggests that SDD not only accelerates training but also mitigates optimization challenges common in large-scale MoE models.

Dowmstream Evaluation. Figure 6 shows that SDD-588M3B outperforms OLMoE-588M-3B across all benchmarks, particularly in MMLU, which evaluates multi-domain reasoning. More metrics are available in Appendix D. These improvements underscore SDD’s capacity to enhance generalization and capture intricate linguistic patterns. Overall, SDD boosts both training efficiency and downstream performance in MoE architectures, providing a robust and scalable solution for large-scale model optimization.

#### 4.4. Ablation Study

Gradient Visualization. Figure 7 compares gradient norms across layers for OLMo2-1B (Pre-Norm), PostNorm1B, DeepNorm-1B (both Post-Norm), and SDD-1B (PostNorm). SDD-1B maintains significantly more stable gra-

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Figure 9. Layer-Wise Feature Similarity Across Normalization Methods. This figure compares feature similarity across layers in OLMo2-1B, PostNorm-1B, DeepNorm-1B, and SDD-1B. SDD1B achieves the highest inter-layer similarity, indicating more stable feature propagation.

- Figure 7. Comparison of Gradient Norms Across Layers. We compare four methods: OLMo2-1B (Pre-Norm), PostNorm-1B, DeepNorm-1B, and SDD-1B (all Post-Norm). “att proj” refers to the query/key/value projection, “attn out” to the attention output projection, “ff proj” to the gating and first FC layer in the feedforward network (FFN), and “ff out” to the second FC layer in the FFN. SDD-1B demonstrates notably stable gradient norms, effectively addressing gradient explosion and vanishing.

dient norms, mitigating gradient explosion and vanishing, which commonly affect Post-Norm variants. This stability improves optimization and training robustness, especially in deep networks, making SDD particularly effective for large-scale models.

- Figure 8. Training and downstream performance of SDD-588M3B with Pre-Norm and Post-Norm compared to OLMoE-588M-3B (Pre-Norm). Models trained on 250 billion tokens show that SDD improves convergence speed and downstream accuracy in the PreNorm setting. Switching to Post-Norm with SDD yields even greater performance gains.

SDD on Pre-Norm. Figure 8 evaluates SDD-588M-3B under both Pre-Norm and Post-Norm settings. When applied to Pre-Norm, SDD accelerates convergence and enhances downstream accuracy. Further gains are observed when transitioning from Pre-Norm to Post-Norm, highlighting SDD’s adaptability and effectiveness in improving training stability and generalization.

Layer-wise Similarity. Figure 9 illustrates inter-layer feature similarity across normalization methods. SDD-1B exhibits the lowest similarity, indicating reduced feature redundancy and effectively mitigating feature collapse. This suggests that SDD promotes more diverse representations across layers, contributing to better optimization and enhanced generalization.

Robustness on Hyperparameter Perturbations. Table 2 assesses model robustness under hyperparameter variations, including increased learning rates, reduced initialization scale, and removal of warmup. While PostNorm-581M and DeepNorm-581M fail to converge under certain conditions, SDD-581M consistently stabilizes training and achieves lower loss, demonstrating resilience to hyperparameter changes.

Scaling law for model depth. Figure 10 compares OLMo21B (Pre-Norm) and SDD-1B (Post-Norm) across varying depths. SDD enables deeper models to scale effectively, overcoming training instability that typically limits PostNorm architectures. This is particularly evident as the depth increases, where SDD maintains stability and ensures smooth optimization. These results further validate SDD’s ability to improve convergence and performance in largescale Transformer models, making it a promising solution for very deep architectures.

- Table 2. Impact of Hyperparameter Perturbations on Model Performance. “−” indicates non-convergence. All models are trained on 200B tokens. “lr∗5” refers to a 5x increase in learning rate, “Initstd∗0.1” scales the initialization standard deviation by 0.1, and “wo Warmup” denotes the removal of the warmup phase.

#### Model lr∗5 Initstd∗0.1 wo WarmUp Loss ↓

OLMo2-581M ✗ ✗ ✗ 2.85 OLMo2-581M ✓ ✗ ✗ 2.84 OLMo2-581M ✗ ✓ ✗ 2.86 OLMo2-581M ✗ ✗ ✓ 2.85

PostNorm-581M ✗ ✗ ✗ 2.84 PostNorm-581M ✓ ✗ ✗ − PostNorm-581M ✗ ✓ ✗ − PostNorm-581M ✗ ✗ ✓ −

DeepNorm-581M ✗ ✗ ✗ 2.84 DeepNorm-581M ✓ ✗ ✗ − DeepNorm-581M ✗ ✓ ✗ − DeepNorm-581M ✗ ✗ ✓ 2.87 SDD-581M ✗ ✗ ✗ 2.83 SDD-581M ✓ ✗ ✗ 2.81 SDD-581M ✗ ✓ ✗ 2.82 SDD-581M ✗ ✗ ✓ 2.83

- Figure 10. Scaling with model depth: OLMo2-1B (Pre-Norm) vs. SDD-1B (Post-Norm). All models are trained on 200 billion tokens, with only the number of layers varied. SDD shows superior scaling behavior as model depth increases, highlighting its robustness in deeper networks.

### 5. Related Work

Normalization Techniques in Transformers. Normalization is essential for stabilizing deep Transformer training (Wang et al., 2024b; 2022), with Layer Normalization (LN) (Ba, 2016; Wang et al., 2022) being the standard. Pre-Norm (Xiong et al., 2020) improves stability but often reduces expressivity, while Post-Norm (Vaswani et al., 2017) enhances generative performance but is prone to gradient explosion in deep networks. Approaches like DeepNorm (Wang et al., 2024a) and Sandwich-LN (Ding et al., 2021) aim to address these challenges by balancing stability and expressivity. Our method, Scale-Distribution Decoupling (SDD), builds on these efforts by explicitly disentangling the scale and distribution of the weight matrix, preserving stability while

enhancing expressivity and optimizing training.

Mixture of Experts and Large-Scale Model Training. The adoption of Mixture of Experts (MoE) architectures (Shazeer et al., 2017; Fedus et al., 2022) has allowed for more efficient computation by activating subsets of parameters per forward pass. However, MoE introduces instability in expert selection and training divergence. OLMoE (Muennighoff et al., 2024) and architectures like Switch Transformers (Fedus et al., 2022) mitigate these issues with improved routing and load balancing. SDD complements these approaches by enhancing convergence and robustness, ensuring MoE models remain stable even under varying hyperparameter settings.

Scaling and Stability in Large Language Models. Training stability becomes more difficult as Transformer depth increases, with gradient-related issues like vanishing or exploding gradients. Techniques such as T-Fixup (Huang et al., 2020) and GradNorm (Chen et al., 2018) focus on balancing gradient magnitudes, while Megatron-Init (Shoeybi et al., 2019) improves initialization. However, these methods primarily address stability from a weight-scaling perspective, rather than tackling optimization dynamics directly. SDD addresses these challenges by improving depth scalability and maintaining stable feature representations across layers, reducing redundancy, and mitigating feature collapse. These advantages make SDD a robust solution for training large-scale Transformers.

By addressing both stability and expressivity, SDD offers a scalable and efficient solution that enhances training stability while preserving the model’s capacity to capture complex patterns. This decoupling of scale and distribution ensures robust optimization, enabling effective training of modern Transformer architectures, even in deep or high-dimensional networks, while maintaining model performance.

### 6. Conclusion

We propose Scale-Distribution Decoupling (SDD), a method that stabilizes Transformer training by explicitly separating the scale and distribution of fully connected layer parameters. Our theoretical analysis establishes its expressivity and training benefits, while gradient analysis confirms improved stability, reducing the risk of gradient explosion or vanishing. Extensive experiments on both dense and Mixture of Experts (MoE) models demonstrate that SDD accelerates convergence, improves generalization, and enhances robustness to hyperparameter perturbations. Additionally, SDD exhibits superior scalability with depth and fosters more consistent inter-layer representations. By addressing key training challenges, SDD provides a principled approach for improving the efficiency and stability of large-scale language models.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Ba, J. L. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020.

Chen, Z., Badrinarayanan, V., Lee, C.-Y., and Rabinovich, A. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International conference on machine learning, pp. 794–803. PMLR, 2018.

Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2924–2936, 2019.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Ding, M., Yang, Z., Hong, W., Zheng, W., Zhou, C., Yin, D., Lin, J., Zou, X., Shao, Z., Yang, H., et al. Cogview: Mastering text-to-image generation via transformers. Advances in neural information processing systems, 34: 19822–19835, 2021.

Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. A framework for few-shot language model evaluation, 12 2023. URL https://zenodo.org/records/ 10256836.

Gordon, A., Kozareva, Z., and Roemmele, M. Semeval2012 task 7: Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In * SEM 2012: The First Joint Conference on Lexical and Computational Semantics–Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012), pp. 394–398, 2012.

Groeneveld, D., Beltagy, I., Walsh, P., Bhagia, A., Kinney, R., Tafjord, O., Jha, A., Ivison, H., Magnusson, I., Wang, Y., Arora, S., Atkinson, D., Authur, R., Chandu, K. R., Cohan, A., Dumas, J., Elazar, Y., Gu, Y., Hessel, J., Khot, T., Merrill, W., Morrison, J. D., Muennighoff, N., Naik, A., Nam, C., Peters, M. E., Pyatkin, V., Ravichander, A., Schwenk, D., Shah, S., Smith, W., Strubell, E., Subramani, N., Wortsman, M., Dasigi, P., Lambert, N., Richardson, K., Zettlemoyer, L., Dodge, J., Lo, K., Soldaini, L., Smith, N. A., and Hajishirzi, H. Olmo: Accelerating the science of language models. arXiv preprint, 2024. URL https://api.semanticscholar.

org/CorpusID:267365485.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021.

Huang, H., Zhu, D., Wu, B., Zeng, Y., Wang, Y., Min, Q., and Zhou, X. Over-tokenized transformer: Vocabulary is generally worth scaling. arXiv preprint arXiv:2501.16975, 2025.

Huang, X. S., Perez, F., Ba, J., and Volkovs, M. Improving transformer optimization through better initialization. In International Conference on Machine Learning, pp. 4475–

4483. PMLR, 2020.

Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.

Li, Z., Zeng, Y., Zuo, Y., Ren, W., Liu, W., Su, M., Guo, Y., Liu, Y., Lixiang, L., Hu, Z., Bai, L., Li, W., Liu, Y., Yang, P., Jin, X., Guo, J., and Cheng, X. KnowCoder: Coding structured knowledge into LLMs for universal information extraction. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8758–8779, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.475. URL https: //aclanthology.org/2024.acl-long.475/.

Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a suit of armor conduct electricity? a new dataset for open

book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2381–2391, 2018.

Muennighoff, N., Soldaini, L., Groeneveld, D., Lo, K., Morrison, J., Min, S., Shi, W., Walsh, P., Tafjord, O., Lambert, N., Gu, Y., Arora, S., Bhagia, A., Schwenk, D., Wadden, D., Wettig, A., Hui, B., Dettmers, T., Kiela, D., Farhadi, A., Smith, N. A., Koh, P. W., Singh, A., and Hajishirzi, H. Olmoe: Open mixture-of-experts language models, 2024. URL https://arxiv.org/abs/2409.02060.

OLMo, T., Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., Bhagia, A., Gu, Y., Huang, S., Jordan, M., et al. 2 olmo 2 furious. arXiv preprint arXiv:2501.00656, 2024.

Reddy, S., Chen, D., and Manning, C. D. Coqa: A conversational question answering challenge. Transactions of the Association for Computational Linguistics, 7:249–266, 2019.

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

Sap, M., Rashkin, H., Chen, D., LeBras, R., and Choi, Y. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., and Catanzaro, B. Megatron-lm: Training multibillion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Talmor, A., Herzig, J., Lourie, N., and Berant, J. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4149–4158, 2019.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in Neural Information Processing Systems, 2017.

Vershynin, R. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018.

Wang, H., Ma, S., Dong, L., Huang, S., Zhang, D., and Wei, F. Deepnet: Scaling transformers to 1,000 layers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024a.

Wang, J., Wu, B., Jiang, H., Xun, Z., Xiao, X., Guo, H., and Xiao, J. World to code: Multi-modal data generation via self-instructed compositional captioning and filtering. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 4608–4623, 2024b.

Wang, Y., Sun, X., Fengzong, L., Kang, Z., and Xu, C. X. An anchor-based relative position embedding method for cross-modal tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 5401–5413, 2022.

Welbl, J., Liu, N. F., and Gardner, M. Crowdsourcing multiple choice science questions. In Proceedings of the 3rd Workshop on Noisy User-generated Text, pp. 94–106, 2017.

Xie, S., Zhang, H., Guo, J., Tan, X., Bian, J., Awadalla, H. H., Menezes, A., Qin, T., and Yan, R. Residual: Transformer with dual residual connections. arXiv preprint arXiv:2304.14802, 2023.

Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T. On layer normalization in the transformer architecture. In International Conference on Machine Learning, pp. 10524– 10533. PMLR, 2020.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, 2019.

Zhang, B., Titov, I., and Sennrich, R. Improving deep transformer with depth-scaled initialization and merged attention. arXiv preprint arXiv:1908.11365, 2019.

Zhu, D., Huang, H., Huang, Z., Zeng, Y., Mao, Y., Wu, B., Min, Q., and Zhou, X. Hyper-connections. arXiv preprint arXiv:2409.19606, 2024.

Zhuo, Z., Wang, Y., Zeng, Y., Li, X., Zhou, X., and Ma, J. Polynomial composition activations: Unleashing the dynamics of large language models. In The Thirteenth International Conference on Learning Representations, 2025.

- A. Omitted Proof Proof. (1) y = Wx =⇒ y = α ⊙ norm(V x).

Let W ∈ Rn×n be the weight matrix of a fully-connected layer, where each element of W is sampled from an independent Gaussian distribution N(0,σ2/n). Using singular value decomposition (SVD), W can be written as:

##### W = UΣV ′⊤, (14)

where U ∈ Rn×n and V ′ ∈ Rn×n are orthogonal matrices, and Σ ∈ Rn×n is a diagonal matrix containing the singular values σ1,σ2,...,σn of W. Substituting W into y = Wx, we can rewrite the output as:

##### y = Wx = UΣV ′⊤x. (15)

Let z = V ′⊤x. Since x ∼ N(0,I), the orthogonal transformation V ′⊤x preserves the Gaussian distribution of x, meaning z ∼ N(0,I). According to Theorem 3.1.1 (Vershynin, 2018), ∥x∥ is approximately equal to 1. So for simplicity, we set ∥z∥ = 1. The term Σz scales the components of z along the singular directions, where:

Σz = [σ1z1,σ2z2,...,σnzn]⊤, (16) The orthogonal matrix U then rotates the scaled vector Σz:

y = UΣz. (17) Next, we normalize y, effectively removing the rotational effect of U:

UΣz ∥UΣz∥

UΣz ∥Σz∥

= U · norm(Σz). (18)

norm(y) = norm(UΣz) =

=

where ∥Σz∥ denotes the norm of the diagonal matrix Σz. Thus, U can always be absorbed into subsequent layers’ mappings in a neural network without affecting the overall output, regardless of whether normalization is applied between U and the subsequent layers. For example, in Transformer models, U can propagate through the value, output projection of attention, and feed-forward network (FFN) mappings, making its explicit presence inconsequential. In other words, y′ = ΣV ′⊤x is equivalent in expressiveness to y = Wx in Transformer models. Therefore, we make no distinction between y′ and y.

After absorbing U, and noting that ∥z∥ = 1 implies norm(z) = z, the output can be reformulated as:

##### y = α ⊙ V ′⊤x = α ⊙ norm(V ′⊤x), (19)

where α = diag(Σ) = [σ1,σ2,...,σn]⊤ captures the scale information of W. Let V = V ′⊤, then the output y = Wx can be equivalently expressed in the form y = α ⊙ norm(V x).

##### (2) y = α ⊙ norm(V x) =⇒ y = Wx.

Consider the representation y = α ⊙ norm(V x), where α ∈ Rn is a vector, and V ∈ Rn×n is a general matrix that may not necessarily be orthogonal. To demonstrate that this output can be equivalently expressed as y = Wx, the matrix V is decomposed using singular value decomposition (SVD). Specifically, let:

V = PΛQ⊤, (20) where P ∈ Rn×n and Q ∈ Rn×n are orthogonal matrices, and Λ ∈ Rn×n is a diagonal matrix containing the singular values of V , denoted as γ1,γ2,··· ,γn. Substituting the decomposition of V into the given equation, the output becomes:

##### y = α ⊙ norm(V x) = α ⊙ norm(PΛQ⊤x). (21)

Define z = Q⊤x. Since x ∼ N(0,I), and by Theorem 3.1.1 (Vershynin, 2018), ∥x∥ is approximately 1. For brevity, we assume ∥x∥ = 1. The orthogonality of Q guarantees that ∥z∥ = 1. Therefore, the expression for y can now be written as:

##### y = α ⊙ norm(PΛz). (22)

To simplify further, note that the normalization operation satisfies:

norm(PΛz) = P · norm(Λz). (23) For a diagonal matrix Λ, the normalization of Λz can be approximately expressed as:

Λz ∥Λz∥

norm(Λz) =

Λz ∥Λ∥

. (24)

≈

Here, ∥Λ∥ = (γ12 + γ22 + ··· + γn2)/n. Substituting this result, the output becomes:

y = α ⊙ P

By substituting back z = P⊤x, we have:

y = α ⊙ P

The equivalence to y = Wx is now established by defining:

Λz ∥Λ∥

. (25)

Λ ∥Λ∥

Q⊤x. (26)

Λ ∥Λ∥

Q⊤. (27)

W = α ⊙ P

Thus, W ∈ Rn×n is a valid weight matrix that satisfies y = Wx for any x, completing the proof of reverse equivalence.

| |
|---|

### B. Architectural Configuration

Table 3 presents the architectural specifications of the evaluated models, including the OLMo2-581M, OLMo2-1B OLMo21.5B, and OLMo2-2B dense models, as well as the OLMoE-588M-3B Mixture of Experts (MoE) model. Key attributes such as parameter counts, hidden dimensions, attention configurations, and expert routing details are provided for comparison.

Table 3. Architectural Configurations of the Dense Model and MoE Model.

Property OLMo2-581M OLMo2-1B OLMo2-1.5B OLMo2-2B OLMoE-588M-3B Activate Params 581M 1B 1.5B 2B 588M Total Params 581M 1B 1.5B 2B 3.4B Hidden Size 2048 2048 2048 2048 1024 Intermediate Size 8192 8192 8192 8192 512 GQA Groups 8 8 8 8 1 Attention Heads 32 32 32 32 16 Hidden Layers 8 16 24 32 32 Experts − − − − 64 Topk Experts − − − − 8 Context Length 4096 4096 4096 4096 4096 Vocabulary Size 100278 100278 100278 100278 50280

### C. Additional Results on Dense models

- Figure 11 presents validation loss and downstream evaluation results for dense models under different training regimes. It compares SDD-1B and OLMo2-1B trained on 2T tokens with PostNorm-1B and DeepNorm-1B trained on 200B tokens. SDD-1B consistently achieves lower validation loss and outperforms all baselines across multiple downstream tasks, highlighting its superior convergence and generalization capabilities. These results further demonstrate the advantages of Scale-Distribution Decoupling (SDD) in stabilizing optimization and improving performance in large-scale language model training.

###### Figure 11. Training and Downstream Performance of Dense Models. This figure compares validation loss and downstream task performance for SDD-1B and OLMo2-1B trained on 2T tokens, alongside PostNorm-1B and DeepNorm-1B trained on 200B tokens. SDD-1B exhibits lower loss and superior generalization, demonstrating its effectiveness in large-scale training.

D. Additional Results on MoE models

###### Figure 12 presents the validation loss and downstream task performance of MoE models trained with 250B tokens, comparing SDD-588M-3B and OLMoE-588M-3B. SDD-588M-3B consistently achieves lower validation loss, indicating improved training stability and efficiency. Additionally, it outperforms OLMoE-588M-3B across multiple benchmarks, demonstrating superior generalization. These results highlight the benefits of Scale-Distribution Decoupling (SDD) in enhancing MoE model optimization, leading to more stable convergence and improved downstream task performance.

###### Figure 12. Training and Downstream Performance of MoE Models with 250B Tokens. This figure compares the validation loss and downstream task performance of SDD-588M-3B and OLMoE-588M-3B. SDD-588M-3B demonstrates lower loss and superior generalization across benchmarks, highlighting its effectiveness in MoE training.

