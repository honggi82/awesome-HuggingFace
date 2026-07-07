# arXiv:2507.05566v1[cs.AI]8Jul2025

## SingLoRA: Low Rank Adaptation Using a Single Matrix

David Bensaïd Technion - IIT Haifa, Israel bensaiddavid@gmail.com

Noam Rotstein Technion - IIT Haifa, Israel

Roy Velich Technion - IIT Haifa, Israel

Daniel Bensaïd University Paris Dauphine Paris, France

Ron Kimmel Technion - IIT Haifa, Israel

### Abstract

Low-Rank Adaptation (LoRA) has significantly advanced parameter-efficient finetuning of large pretrained models. LoRA augments the pre-trained weights of a model by adding the product of two smaller matrices that together form a lowrank matrix update. Recent research has shown that scale disparities between these two matrices often cause unstable training dynamics, leading to suboptimal performance. In this paper, we propose SINGLORA , which reformulates lowrank adaptation by learning the weights update as a decomposition of a single low-rank matrix multiplied by its transpose. This simple design inherently removes inter-matrix scale conflicts, ensuring stable optimization, and roughly halves the parameter count. We analyze SINGLORA within the infinite-width neural network framework, showing that it guarantees stable feature learning by construction. Extensive experiments on multiple tasks validate these benefits. In common sense reasoning, fine-tuning LLama 7B on MNLI with SINGLORA achieves 91.3% accuracy — surpassing LoRA (89.1%) and LoRA+ (90.2%) — while using only 60% of their parameter budget. In image generation, fine-tuning Stable Diffusion with SINGLORA significantly improves image fidelity on DreamBooth, achieving a DINO similarity score of 0.151, compared to scores of 0.148 and 0.143 for DoRA and LoRA, respectively.

### 1 Introduction

Adapting large pretrained models for specialized tasks has emerged as a central focus in machine learning research. These efforts aim to leverage the strong generalization capabilities of such models while meeting domain-specific requirements. Rapid scaling of model sizes and datasets has made full fine-tuning computationally prohibitive, driving the development of parameter-efficient fine-tuning (PEFT) methods to reduce computational costs. Among PEFT approaches, Low-Rank Adaptation (LoRA) [6] has gained particular popularity due to its simplicity and effectiveness in various domains. LoRA augments pretrained weights matrices W0 ∈ Rd×k with low-rank updates,

W0 + BA, where B ∈ Rd×r, A ∈ Rr×k and r ≪ min(d,k). (1)

Recent research [4, 19, 20] has identified scale disparities between the matrices A and B as a fundamental challenge in LoRA. Such scale imbalances lead to unstable training dynamics, causing vanishing or exploding gradients and ultimately resulting in suboptimal performance. To address this

Preprint.

challenge, we introduce SINGLORA , which reformulates the low rank adaptation paradigm using a single low-rank matrix A ∈ Rn×r yielding the symmetric update,

##### W0 + AA⊤, (2)

This formulation provides two key benefits over traditional LoRA. First, it ensures stable optimization by design, eliminating inter-matrix scale conflicts. Second, it achieves a parameter reduction by roughly halving the number of learned parameters. Our empirical results show that these architectural improvements enable SINGLORA to consistently exceed LoRA’s performance while using significantly fewer parameters.

We analyze SINGLORA in the infinite-width neural network setting [3, 4], showing that, unlike LoRA, it guarantees stable feature learning by construction. Building on this theoretical foundation, we extend the approach to non-square matrices, ensuring its applicability to any neural network, and validate its results through comprehensive experiments across multiple modalities. For example, in comprehension reasoning, fine-tuning LLaMA [14] on MNLI [16] with SINGLORA outperforms LoRA and LoRA+ (91.3% against 89.1% and 90.2% respectively) while using only 60% of their parameter budget. In image generation, SINGLORA increases the image fidelity of Stable Diffusion V1.5 finetuned on Dreambooth [12] by 5.4% compared to LoRA [9].

### 2 Related Efforts

##### 2.1 LoRA

As large multimodal models continue to scale, an increasing number of parameter-efficient finetuning (PEFT) techniques have been developed to facilitate their adaptation to downstream tasks. LoRA has emerged as one of the most popular adapters, offering efficient model adaptation across various domains. It has been studied extensively, resulting in numerous variations. DyLoRA [15] and AdaLoRA [21] focus on rank adjustment during training, with adaptive strategies to optimize rank allocation. LoHA [7] and LoKR [2], propose architectural extensions and, respectively, leverage Hadamard and Kronecker products of two rank r approximations to obtain more expressive adapters. Delta-LoRA [23] modifies LoRA by updating pre-trained weights using differences in successive low-rank updates, and LoRA-Drop [22] selects the most impactful LoRA adapters to reduce computational cost. More recently, Weight-Decomposed Low-Rank Adaptation (DoRA) [9] decomposes the pretrained weight matrix into magnitude and direction, before employing LoRA to update the direction. The proposed approach is complementary to these architectural extensions and can be seamlessly integrated with them, further enhancing their effectiveness.

##### 2.2 Stable extensions of LoRA

A recent line of research has identified fundamental limitations in standard optimizers, such as Stochastic Gradient Descent (SGD) and Adam, to finetune LoRA modules. LoRA+ [4] shows that the matrices A and B should be optimized with different learning rates to ensure stable learning dynamics. Building on this theoretical foundation, Zhang et al. [20] have proposed to use Riemannian gradient descent and conditioning methods to stabilize the optimization process. Recently, LoRADoneRite [19] identified the multiplicity of possible optimizer updates for a single low-rank adapter as the source of instability in LoRA’s training. Building on these theoretical insights into optimization stability, we analyze the convergence properties of SINGLORA . We demonstrate that its streamlined low-rank adaptation paradigm naturally promotes stable and robust optimization, without requiring careful learning rate tuning or modifications to classical optimizers.

#### 3 LoRA’s Stability Issue For a pretrained weight matrix W0 ∈ Rd×k, LoRA introduces a low-rank update,

α r

BA (3)

W = W0 +

where B ∈ Rd×r, A ∈ Rr×k are trainable matrices with rank r ≪ min(d,k), and α ∈ R is a scaling factor. During fine-tuning, only A and B are trained while W0 remains frozen. To preserve the

pretrained model’s behavior at the start of training, B is initialized to zero while A uses a random Gaussian initialization.

A recent line of work [19, 4, 20] has highlighted that LoRA’s training dynamics often suffers from instability issues, particularly as the model width, denoted n, increases. Based on the analysis of LoRA+ [4], we examine these stability challenges in an infinite-width environment [13, 17, 18, 3]. Specifically, we investigate how the learning rate should scale so that the changes in the model output between iterations, denoted ∆f, remain both bounded and non-vanishing as the network width grows. We refer to this property as stable features learning, expressed mathematically as ∆f = Θn(1).

##### 3.1 Analysis of a toy model

In this subsection we extend the analysis of the toy model proposed in [4]. For a more complete discussion of the theoretical foundations of the stable feature learning theory, we refer the reader to [4, 3]. Consider a linear model f : Rn → Rn with a rank-1 LoRA update where we have a pretrained weight matrix W0 ∈ Rn×n and trainable LoRA vectors b,a ∈ Rn. The model, defined as f(x) = (W0+ba⊤)x, is trained on input-output pairs (x,y), x,y ∈ Rn with loss L = 21∥f(x)−y∥2. L is minimized using a gradient descent method with learning rate η. Without loss of generality, we assume W0 = 0 by defining y˜ = y − W0x. It holds,

∇aL = (ft−1(x) − y)(x⊤bt−1) and ∇bL = (ft−1(x) − y)(a⊤t−1x), (4) and at iteration t, the gradient updates are thus,

at = at−1 − η(ft−1(x) − y)(x⊤bt−1), and bt = bt−1 − η(ft−1(x) − y)(a⊤t−1x). (5) To analyze stability, we examine how the network output changes between iterations, ∆ft = ft(x) − ft−1(x) = (bta⊤t − bt−1a⊤t−1)x

−η(a⊤t−1x)2[ft−1(x) − y]

+η2∥ft−1(x) − y∥2(a⊤t−1x)(b⊤t−1x)x

= −η∥bt−1∥2∥x∥2[ft−1(x) − y]

δt1

δt2

δt3

Having ∆ft = Θ(1) implies that at least one of the terms (δti)i∈{1,2,3} is Θ(1). To ensure that updates to a and b significantly impact ft(x), both δt1 and δt2 must be Θ(1). This is because δt1 = o(1) implies that the model keeps a fixed, effectively training only b, and similarly if only δt2 = o(1), only a is effectively trained. When both δt1 and δt2 are Θ(1), it follows that δt3 is guaranteed to be Θ(1), as shown in [4].

##### 3.2 Efficiency Analysis

We analyze the scaling behavior with width using the γ−notation. For a parameter v, we define,

v = Θ(nγ[v]), (6)

where γ[v] represents how v scales with n. When applied to vectors, γ[v] indicates that all the entries of v respect Eq. 6. For the product of vectors v,w ∈ Rn, we follow a key scaling rule: γ[v⊤w] = γ[v] + γ[w] + 1, where the extra +1 comes from summing n terms.

As theorized in [17, 18], the initialization scheme of the weights and the learning rate should be adapted as a function of the width of the network, n, to ensure efficient learning. We therefore assume that the model is trained with a gradient descent procedure with a learning rate that respects η = Θ(nc) for some c ∈ R. a is initialized with a random Gaussian distribution scaled as Θ(n−1/2) (known as Kaiming initialization [5]) and b is initialized as 0. Although each component of a0 is Θ(n−1/2), by the Central Limit Theorem (CLT), the term a⊤0 x = Θ(1), since it represents the sum of n independent terms with variance Θ(n−1).

Starting from initialization where f0(x) = 0, efficient LoRA fine-tuning requires δt1 = Θ(1), δt2 = Θ(1) and δt3 = Θ(1) for all t > 1, with ft(x) = Θ(1) for t > 1. Using the γ notation, this translates to the system,

c + 2γ[bt−1] + 2 = 0 (δt1 = Θ(1)) c + 2γ[a⊤t−1x] = 0 (δt2 = Θ(1))

γ[bt−1] + γ[a⊤t−1x] = 0 (ft−1(x) = Θ(1)) (7)

Simple calculations with this system yield c = −1. Let us analyze the updates at each step t. At initialization, γ[a⊤0 x] = 0 by the CLT. Using an inductive argument, for t > 0, the update equation for b is bt = bt−1 − η(ft−1(x) − y)(a⊤t−1x). Analyzing each term yields,

- • (ft−1(x) − y) = Θ(1), as we assume that the error of the model is bounded.
- • a⊤t−1x = Θ(1) by CLT.

Multiplying these factors, the overall update to b is thus of order of η, i.e. Θ(n−1), implying that for t > 0, bt = Θ(n−1).

Similarly for a, at = at−1 − η(ft−1(x) − y)(x⊤bt−1). Analyzing each term yields,

- • (ft−1(x) − y) = Θ(1).
- • x⊤bt−1 = Θ(1) by the vector product rule, γ[x⊤bt−1] = γ[x] + γ[bt−1] + 1 = 0.

Multiplying these factors, the overall update to a is of order Θ(n−1), thus maintaining γ[at] = −1/2. Applying the CLT, we finally obtain a⊤t x = Θ(1).

However, plugging bt = Θ(n−1) and a⊤t x = Θ(1) in the definition of f yields ft(x) = Θ(n−1), in contradiction with the assumption that ft(x) = Θ(1) for efficient learning. Therefore, LoRA is not stable as updates inherently scale differently with width. This inefficiency motivates the need for alternative approaches, such as using different learning rates for a and b, as in LoRA+ [4], or reformulating the low-rank update, as in our research.

### 4 SINGLORA : Low-Rank Adaptation with a Single Matrix

In this section, we first present SINGLORA ’s core formulation (4.1), showing how it achieves parameter-efficient adaptation through a symmetric low-rank update. We then analyze its training dynamics through a simplified model (4.2), proving that, unlike LoRA, our approach guarantees stable feature learning by construction. We establish formal convergence guarantees under standard optimizers like SGD and AdamW (4.3), demonstrating that SINGLORA eliminates the need for specialized optimization techniques. Finally, we extend SINGLORA to non-square weight matrices (4.4), making it applicable across diverse neural architectures.

##### 4.1 SINGLORA ’s formulation

SINGLORA reformulates low-rank adaptation by replacing the traditional two-matrix decomposition with a single learnable matrix. Given a pretrained model with frozen weight matrix W0 ∈ Rn×n, SINGLORA computes the adapted weights as,

α r

u(t)AA⊤, (8)

W0 +

where A ∈ Rn×r is a low-rank trainable matrix with rank r ≪ n, α is a scaling factor, and u(t) is a scalar function controlling the adaptation rate at optimization step t. This formulation provides two key advantages: (1) it eliminates inter-matrix scale conflicts by construction, ensuring stable optimization, and (2) it reduces the parameter count by roughly half compared to standard LoRA.

Initialization scheme. To enable effective gradient flow during training, we initialize A with a Kaiming distribution. To preserve the behavior of the pretrained model at initialization, we require u(0) = 0, ensuring that the model starts from the pretrained weights. In practice, we adopt a simple

ramp function for u(t), namely u(t) = min T t ,1 , where T ∈ R controls the adaptation rate. u provides a smooth transition from the pretrained weights to the adapted model, allowing for gradual

incorporation of task-specific features while maintaining stability during the early training stages.

##### 4.2 Efficiency Analysis of a Toy model

Similarly to Section 3, we analyze a toy example with the proposed single-matrix formulation (using only a instead of both a and b). Formally, f(x) = (W0 + u(t)aaT)x, where W0 ∈ Rn×n represents the pretrained weights and a ∈ Rn is the trainable vector. Like in Section 3, we consider a single

training sample (x,y) with the loss L = 12∥f(x) − y∥2 which is minimized using a gradient descent procedure with a learning rate η = Θ(nc). The gradient ∇aL and its update at step t are respectively

given by, ∇aL = u(t)[(a⊤t−1x)(ft−1(x) − y) + a⊤t−1(ft−1(x) − y)x] and by at = at−1 − η∇aL.

Scaling Analysis. For stable feature learning, we require ft(x) = Θ(1). As W0 remains frozen, this means u(t)at (a⊤t x) = Θ(1). If at has entries of order Θ(np), then at(a⊤t x) scales as Θ(n2p+1). For stability, this implies 2p + 1 = 0, yielding p = −1/2. To ensure the stability of the optimization, at should thus maintain entries of order Θ(n−1/2).

Let us analyze the factor of the gradient update:

- • η = Θ(nc).
- • ∇aL = Θn(1) since (f(x) − y) = Θn(1), as we assume the error of the model to be bounded, and aTt−1x = Θ(1), aTt−1(f(x) − y) = Θ(1) by CLT.

The gradient update thus scales as Θ(nc) and we require c = −1/2 to maintain at with entries Θ(n−1/2). Therefore, setting η = Θ(n−1/2) ensures stable feature learning.

Unlike LoRA where balancing two matrices requires careful learning rate tuning, SINGLORA achieves stable feature learning by design, as we can always set an appropriate learning rate scale to ensure ft = Θ(1). This characteristic guarantees efficient feature learning in the infinite-width limit.

##### 4.3 Transformation Invariance of Low Rank Adapters

Here, we extend the analysis of the previous subsection to a more general setting, and show that a model adapted with SINGLORA and finetuned using standard optimizers, such as SGD, achieves stable feature learning.

Observing that pairs of matrices (A1,B1) and (A2,B2) representing the same adapter, i.e. A1B1 =

- A2B2, can produce different optimizer updates, Yen et al. [19] recently introduced the notion of transformation-invariance for low rank adapters. Definition 1. Transformation-Invariance Let (A1,B1) and (A2,B2) be LoRA adapters satisfying

- A1B1 = A2B2. An optimizer is transformation-invariant if its updates (δA1,δB1) and (δA2,δB2) satisfy,

(A1 + δA1)(B1 + δB1) = (A2 + δA2)(B2 + δB2) (9)

To illustrate how the multiplicity in LoRA’s parametrization can yield different gradient descent updates, we consider two parameterizations (A1,B1) and (A2,B2) of a low-rank adapter related by a scaling factor s ∈ R. Namely,

1 s

B1. (10)

A2 = sA1, B2 =

Defining Z = A1B1 = A2B2 and applying the chain rule yields ∇A1 = ∇ZB1, ∇A2 = ∇ZB2, where ∇P stands for the gradient of the loss L of the model with respect to the parameter P. We can

thus rewrite, ∇A2 = 1s∇ZB1 = 1s∇A1. Therefore,

δA1B1 = −η ∇A1B1 = −η s∇A2B1 = s2 δA2B2,

making Eq. 9 unsatisfied in the general case. This example reveals why LoRA exhibits unstable training dynamics when using optimizers that are not transformation-invariant: when the scaling factor s is much larger than 1, the matrices A and B (and their corresponding updates) operate at vastly different scales. This scale mismatch creates a fundamental problem: first-order optimizers using a single learning rate struggle to achieve stable feature learning, as they cannot simultaneously accommodate both large and small-scale updates effectively. This issue frequently arises during training since LoRA’s matrices A and B are typically initialized with different scales.

- Theorem 1. Any transformation invariant optimizer applying the same update rule for A and B achieves efficient feature learning. A proof is presented in the appendix.

Recent efforts [19, 20] attempt to address the stability issues of LoRA by building a dedicated scale-invariant optimizer. In contrast, SINGLORA formulation inherently solves those challenges

###### Dataset Method RoBERTa GPT-2

LR Acc. (%) Params LR Acc. (%) Params

QQP LoRA 2e−4 88.5 0.15M 4e−4 87.9 1.78M LoRA+ (2e−4,4e−3) 89.1 0.15M (2e−4,4e−3) 89.1 1.78M DoRA 5e−4 89.2 0.16M 5e−4 89.2 1.78M Ours 1e−3 88.9 0.075M 1e−3 88.8 0.89M

QNLI LoRA 4e−4 90.9 0.15M – – – LoRA+ (2e−4,4e−3) 92.1 0.15M – – – DoRA 5e−4 92.1 0.16M – – – Ours 1e−3 92.2 0.075M – – –

MNLI LoRA 4e−4 85.6 0.15M 2e−4 81.3 1.78M LoRA+ (5e−5,4e−3) 86.5 0.15M (2e−4,4e−3) 82.0 1.78M DoRA 5e−4 86.4 0.16M 5e−4 82.2 1.78M Ours 1e−3 86.5 0.075M 1e−3 82.5 0.89M

Mean LoRA – 88.3 0.15M – 84.6 1.78M LoRA+ – 89.2 0.15M – 85.6 1.78M DoRA – 89.2 0.16M – 85.7 1.78M Ours – 89.2 0.075M – 85.7 0.89M

Table 1: Accuracy of RoBERTa and GPT-2 fine-tuned on GLUE datasets with rank 8 updates. LoRA+ uses learning rates (µA,µB). GPT-2 results on QNLI were not reported in [4].

and SINGLORA can be efficiently tuned with standard deep learning optimizers, such as SGD or Adam [8], without requiring special modifications or careful hyper-parameters tuning.

- Theorem 2. A gradient descent optimizer is transformation-invariant for SINGLORA . A proof is presented in the appendix.

Theorem 1 and Theorem 2 guarantee the existence of a learning rate yielding stable dynamics when training SINGLORA with first-order optimization methods.

4.4 Extension to Non-Square Matrices

Our discussion has so far focused on square weight matrices W0 ∈ Rn×n which are commonly used in the attention layers of transformer architecture. We now extend our approach to rectangular weight

matrices W0 ∈ Rd

in×dout. Without loss of generality, we assume din < dout. Considering a low rank matrix A ∈ Rd

out×r we define A∗ ∈ Rd

in×r as a truncation of A consisting of

its first din rows. The adapted layer is then computed by W0 + A∗A⊤. Training this adapter with standard optimizers preserves the stability and transformation-invariance properties demonstrated for the square case.

- Theorem 3. The generalization of SINGLORA to non-square matrix preserves the stability and transformation-invariance properties demonstrated for the square case. A proof is presented in the appendix.

### 5 On the expressiveness of SINGLORA in Transformer Architectures

In the previous section, we demonstrated that SINGLORA improves the optimization process and yields more stable training dynamics. We now turn to investigating its expressive capacity within the Transformer architecture, which serves as the foundation of most current NLP and vision models. We analyze how SINGLORA ’s symmetric updates affect the key-query interaction in self-attention layers. For input X ∈ RL×d, the attention mechanism is computed as

QKT √dk

Attention(Q,K,V) = softmax

##### V, where Q = XWq, K = XWk andV = XWv.

(11)

[Figure 1]

Figure 1: Synthetic experiment: convergence plot for LoRA and SINGLORA . When applying SINGLORA , the weight matrices become,

Wq = Wq0 + AqATq and Wk = Wk0 + AkATk (12) Examining the key-query interaction QKT, we get

##### QKT = X[Wq0Wk0T + Wq0AkATk + AqATq Wk0T + AqATq AkATk ]XT. (13)

The crucial insight that emerges is that although SINGLORA uses symmetric updates (AqATq and AkATk ), their interaction in the computation of attention is more general. The product of two symmetric matrices (AqATq )(AkATk ) is not necessarily symmetric unless they commute. Since there is no constraint forcing AqATq and AkATk to commute, SINGLORA can learn general (nonsymmetric) transformations of attention patterns despite its symmetric parameterization and does not fundamentally limit the model’s ability to learn diverse attention patterns.

Synthetic experiment. To empirically verify this property, we implement LoRA and SINGLORA in an attention mechanism, where they learn to approximate a target attention pattern Z given input X. Attention scores are computed according to 11. For SINGLORA , we define Wq = Wq0 + AqA⊤q and Wk = Wk0 +AkA⊤k , where both Wq,Wk ∈ R128×128. For LoRA, we define Wˆq = Wˆq0 +BˆqAˆq and Wˆk = Wˆk0 + BˆkAˆk, also with Wˆq,Wˆk ∈ R128×128. Both approaches are configured to use the exact same number of parameters for a fair comparison. We optimize both models using an identical AdamW configuration with a learning rate of 10−4 for 15,000 iterations, minimizing the loss ∥XWqWkTXT − Z∥22. As shown in Figure 1, SINGLORA outperforms LoRA in both convergence speed and final approximation accuracy which drops to approximately 10−5 while LoRA remains around 10−2, in fewer iterations. We verified these results with 1K different random seeds to sample Z and X. This experiment shows how SINGLORA maintains expressiveness in attention mechanisms while enhancing optimization dynamics and performance.

### 6 Experiments

We conduct extensive experiments to evaluate SINGLORA in low-rank adaptation for linguistic and visual tasks. Additional experiments and ablative studies on the ramp-up function, and influence of ranks are presented in the appendix. Code will be released upon publication.

##### 6.1 Language Models

To evaluate SINGLORA in Natural Language Processing, we consider selected comprehension and reasoning tasks from the General Language Understanding Evaluation (GLUE) benchmark[16]. We

##### Method LoRA LoRA+ DoRA SINGLORA

Accuracy 89.1 90.2 90.6 91.3 # Params 20M 20M 21M 12M

Table 2: Accuracy of Llama tuned on MNLI with SINGLORA and baselines with ranks 8.

[Figure 2]

Figure 2: Accuracy of Llama-7B fine-tuned on MNLI across different learning rates. The plot compares the stability of LoRA and SINGLORA under varying learning rates, demonstrating that SINGLORA ’s accuracy fluctuates by approximately 1%, while LoRA’s performance varies by 4.8%. These results highlight the robustness of SINGLORA ’s optimization dynamics.

strictly follow the NLP’s experimental protocol of LoRA+, which similarly to SINGLORA addresses LoRA’s training stability issues and thus serves as a key benchmark. To fairly quantify algorithmic differences rather than hyper-parameter tuning advantages, we adopt their training/evaluation codebase, model architectures, modified layers, optimization settings, and training duration. We also compare to DoRA [9], a recent state-of-the-art variation of LoRA.

RoBERTa-base and GPT-2. We first evaluate each approach based on its ability to fine-tune smaller language models - RoBERTa-base and GPT-2 - on the MNLI, QQP, and QNLI tasks from

the GLUE benchmark. Following the LoRA+ setup, we set r = α = 8. We use u(t) = min(10t3 ,1) where t is the training step. Table 1 summarizes the accuracy in these tasks. Accuracies reported

for LoRA and LoRA+ are directly taken from the original paper of LoRA+ [4]. Results show that SINGLORA outperforms LoRA, achieving a 0.9% mean accuracy improvement for RoBERTa and 1.1% for GPT-2. It also achieves slightly better performance than LoRA+ and DoRA, while using only half the number of trainable parameters of both baselines. Note that while LoRA and LoRA+ explore multiple learning rates (5 for LoRA and 25 for LoRA+) and report results using carefully selected learning rates for each dataset and model, SINGLORA employs a single learning rate across all experiments. This indicates that our method’s stability reduces the need for extensive hyperparameter tuning, including the exhaustive grid search required for LoRA+. We further analyze the robustness to learning rate variations in Subsection 6.3.

Llama 7B. To further validate our approach, we fine-tune a large language model (LLM), LLaMA7B, on the MNLI task. Table 2 shows that SINGLORA outperforms LoRA (by more than 2%), LoRA + (by more than 1%) and DoRA, while reducing the number of training parameters by 40%. Since fine-tuning LLMs such as LLaMA is one of the most common applications of low-rank adapters, this result underlines the practical advantages of our approach.

##### 6.2 Image Generation

To showcase the versatility of SINGLORA in architectures and modalities, we evaluated its effectiveness in diffusion-based image generation, where LoRA is commonly used to fine-tune models in small-scale datasets of specific subjects. We consider Stable Diffusion V1.5 [11], a popular image diffusion model.

Dreambooth. We benchmark the approaches on DreamBooth [12], a known dataset with 30 classes of objects and animals, each containing 4–5 training images and 25 evaluation prompts. Following

[Figure 3]

Figure 3: Qualitative comparison of LoRA, DoRA and SINGLORA on Dreambooth.

##### Method CLIP Image CLIP Text DINO Similarity Rank # Params

LoRA 0.677 0.319 0.143 8 0.9M LoRA+ 0.688 0.315 0.150 8 0.9M DoRA 0.687 0.317 0.148 8 0.9M SINGLORA 0.677 0.318 0.141 8 0.45M SINGLORA 0.690 0.317 0.151 16 0.9M

- Table 3: Performance of Stable diffusion V5 finetuned in Dreambooth with same number of parameters. Similarity of the generated image with the originals was measured using CLIP Image and DINO Similarity. Prompt fidelity is evaluated with CLIP Text.

standard practice, we train each model for 400 iterations using the template prompt "a photo of a sks <class>", where "sks" is a rare English token. This setup allows the model to learn new object representations while retaining its general capacities. We finetune the query and key projections of the attention matrices in the U-Net component of Stable Diffusion. For all methods, training is conducted for 400 epochs using a learning rate of 10−3. Figure 3 presents a qualitative comparison between the methods, illustrating how our adaptation approach enhances subject learning. For instance, note that the shoe in the second row retains its iridescent color, whereas other methods fail to do so. Quantitatively, Table 3 demonstrates that SINGLORA achieves a 5.4% improvement in the DINO [1] similarity score compared to DoRA, indicating better object resemblance, while maintaining prompt fidelity measured by the CLIP [10] Text score. Additional image generation experiments are presented in the appendix.

##### 6.3 Stability of SINGLORA

To validate the optimization stability analysis of SINGLORA presented in Section 4.3, we compare its performance against LoRA in a range of learning rates. Specifically, we fine-tune Llama-7B in MNLI using learning rates ranging from 5 · 10−5 to 10−4. As shown in Figure 2, the precision of SINGLORA fluctuates by approximately 1%, whereas LoRA exhibits a larger variation of up to 4.8%. These empirical results validate our theoretical findings, demonstrating that the design of SINGLORA inherently improves learning stability. In practice, this stability translates to simpler hyperparameter tuning, as our method maintains strong performance without requiring extensive searches for an optimal learning rate. Together with previous experiments, these findings highlight that SINGLORA not only offers a more efficient and accurate parameterization, but also ensures robust convergence in practical settings.

### 7 Conclusion

We introduced SINGLORA , a novel formulation of low-rank adaptation that learns and employs a single matrix instead of two. Through theoretical analysis, we demonstrated that the proposed design inherently eliminates the inter-matrix scale disparities present in LoRA and guarantees stable feature learning without requiring special optimizers or careful hyperparameter tuning. Extensive experiments on language and vision tasks validated these benefits, consistently demonstrating improved performance with fewer trainable parameters than LoRA and its variants. Since our approach is complementary to various LoRA’s variants, suchs as DoRA [9], a promising direction is to explore their integration, harnessing their independent strengths to further enhance efficiency and performance.

### References

- [1] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [2] Ali Edalati, Marzieh Tahaei, Ivan Kobyzev, Vahid Partovi Nia, James J Clark, and Mehdi Rezagholizadeh. Krona: Parameter efficient tuning with kronecker adapter. arXiv preprint arXiv:2212.10650, 2022.
- [3] Soufiane Hayou, Arnaud Doucet, and Judith Rousseau. On the impact of the activation function on deep neural networks training. In International conference on machine learning, pages 2672–2680. PMLR, 2019.
- [4] Soufiane Hayou, Nikhil Ghosh, and Bin Yu. Lora+: Efficient low rank adaptation of large models. International Conference on Machine Learning, 2024.
- [5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
- [6] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.
- [7] Nam Hyeon-Woo, Moon Ye-Bin, and Tae-Hyun Oh. Fedpara: Low-rank hadamard product for communication-efficient federated learning. International Conference on Learning Representations, 2021.
- [8] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [9] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation. International Conference on Machine Learning, 2024.
- [10] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [11] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [12] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.
- [13] Samuel S Schoenholz, Justin Gilmer, Surya Ganguli, and Jascha Sohl-Dickstein. Deep information propagation. arXiv preprint arXiv:1611.01232, 2016.
- [14] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [15] Mojtaba Valipour, Mehdi Rezagholizadeh, Ivan Kobyzev, and Ali Ghodsi. Dylora: Parameter efficient tuning of pre-trained models using dynamic search-free low-rank adaptation. Findings of the Association for Computational Linguistics, 2022.
- [16] Alex Wang. Glue: A multi-task benchmark and analysis platform for natural language understanding. International Conference on Learning Representations, 2018.

- [17] Greg Yang, Edward J Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466, 2022.
- [18] Greg Yang, Dingli Yu, Chen Zhu, and Soufiane Hayou. Tensor programs vi: Feature learning in infinite-depth neural networks. arXiv preprint arXiv:2310.02244, 2023.
- [19] Jui-Nan Yen, Si Si, Zhao Meng, Felix Yu, Sai Surya Duvvuri, Inderjit S Dhillon, Cho-Jui Hsieh, and Sanjiv Kumar. Lora done rite: Robust invariant transformation equilibration for lora optimization. International Conference on Learning Representations, 2025.
- [20] Fangzhao Zhang and Mert Pilanci. Riemannian preconditioned lora for fine-tuning foundation models. arXiv preprint arXiv:2402.02347, 2024.
- [21] Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adalora: Adaptive budget allocation for parameterefficient fine-tuning. International Conference on Learning Representations, 2023.
- [22] Hongyun Zhou, Xiangyu Lu, Wang Xu, Conghui Zhu, Tiejun Zhao, and Muyun Yang. Lora-drop: Efficient lora parameter pruning based on output evaluation. arXiv preprint arXiv:2402.07721, 2024.
- [23] Bojia Zi, Xianbiao Qi, Lingzhi Wang, Jianan Wang, Kam-Fai Wong, and Lei Zhang. Deltalora: Fine-tuning high-rank parameters with the delta of low-rank matrices. arXiv preprint arXiv:2309.02411, 2023.

Supplementary

- A Additional experiments

- A.1 Initialization and choice of T

In LoRA, A is initialized using a Kaiming distribution while B is set to zero. This initialization scheme ensures that the model is equivalent to the pretrained one at the beginning of finetuning (since AB = 0), while maintaining effective gradient flow. For SingLoRA, we adopt a similar approach by initializing A with a Kaiming distribution. To achieve equivalence to the original model at the beginning of finetuning, we rely on a simple parametric ramp-up function defined as t/T, where t represents the current training step and T is a hyperparameter, and consider the adapter,

u(t)AA⊤ (14)

To analyze the robustness of this initialization scheme, we conduct an ablation study focusing on the ramp-up function and the hyperparameter T. Figure 4 demonstrates that SINGLORA is robust to the choice of T, showing comparable of RoBERTa’s performance in GLUE across a wide range of T values (from 0.5% to 8% of the total number of training steps). In our main experiments, we set T equal to 1% of the total number of training steps.

[Figure 4]

- Figure 4: Ablation study on the choice of the hyperparameter T. The experiments shows that SINGLORA is robust to a wide range of T which thus does not require extensive hyperparameter search.

- A.2 Additional visual experiment: human faces

To further analyze the ability of SINGLORA in visual tasks, we follow [20] and benchmark the adapters on a dataset that includes 40 human faces. This experiment quantifies the expressive power of adapters and their ability to learn complex details present in human faces. Figure 5 shows a subset of the dataset. Following standard practice, we train each adapter for 1500 iterations using the template prompt “a photo of a sks human”, where sks is a rare English token. Each adapter has the same number of trainable parameters. We finetune the query and key projections of the attention matrices in the U-Net component of Stable Diffusion. Table 4 shows an improvement in the DINO similarity score compared to DoRA, indicating a better similarity to the reference image.

- A.3 Hardware All experiments were performed on a single NVIDIA A40 GPU with a memory of 48GB.

[Figure 5]

- Figure 5: Samples from the dataset used in our experiment. The dataset was automatically generated with a state-of-the-art image generation model trained on faces available in https: //this-person-does-not-exist.com/en.

##### Method LoRA DoRA SINGLORA

DINO Similarity 0.463 0.471 0.501 Rank 8 8 16

- Table 4: DINO Similarity of Stable Diffusion v1.5 tuned on a specific face with the reference face. SINGLORA and baselines uses the same number of trainable parameters.

### B Proofs

In this section we demonstrate the theorem introduced in Sections 4 and 5. We rely on the following sufficient conditions for transformation invariance (see [20]),

- (i) δA1B1 = δA2B2
- (ii) A1δB1 = A2δB2
- (iii) δA1δB1 = δA2δB2 (15)

##### B.1 Theorem 1

Any transformation invariant optimizer applying the same update rule for A and B achieves efficient feature learning.

Here for completeness we present the proof proposed in [20]. Proof:

Let ∥A1∥ = Θ(na), ∥B1∥ = Θ(nb), ∥∇Z∥ = Θ(nc), and η = Θ(nd), where η is the learning rate and n denotes the network width. Since Z = A1B⊤1 , by the chain rule, we have ∇Z = ∇Z⊤ = ∇ZB1A⊤1 . Given the symmetry of the update rule, the updates satisfy:

∥δA1∥ = Θ(nx+a+(y+1)b+c+d), ∥δB1∥ = Θ(nx+b+(y+1)a+c+d).

Assuming the update rule is invariant under scalar reparameterization, we compare two equivalent decompositions A2 = nsA1 and B2 = n−sB1, giving:

∥δA1∥∥B1∥ = ∥δA2∥∥B2∥. From this, it follows:

xa + (y + 1)b + zc + d = x(a + s) + (y + 1)(b − s) + zc + d, which simplifies to:

xs − (y + 1)s = 0 ∀s ⇒ x = −1. Hence, we deduce:

∥δA1∥∥B1∥ = Θ(na+(y+1)b+c+d) = Θ(na+b+yc+d). Similarly, we find:

∥A1∥∥δB1∥ = Θ(na+(y+1)b+c+d) = Θ(na+b+yc+d).

Given that these expressions are equal, the update process enables efficient feature learning:

∥δA1∥∥B1∥ = ∥A1∥∥δB1∥ = Θ(1), by selecting a proper learning rate η = Θ(nd), where x = −1 is fixed and d is chosen accordingly.

##### B.2 Theorem 2

A gradient descent optimizer is transformation-invariant for SINGLORA . Proof:

We prove that the three sufficient conditions for transformation invariance hold. Proof of (i): Assume two parametrizations of a LoRA adapter, A1, A2 ∈ Rn×r with identical ranks such that A1A⊤1 = A2A⊤2 . From the Polar Decomposition Theorem, there exists an orthogonal matrix Q ∈ Rr×r such that A1 = A2Q. Therefore, defining X = A1A⊤1 and using the chain rule lead to,

δA1A⊤1 = −η∇A1A⊤1 = −2η∇ZA1A⊤1 = −2η∇ZA2QQ⊤A⊤2

= −2η∇ZA2A⊤2 = −η∇A2A⊤2 = δA2A⊤2 . (16) The first sufficient condition for transformation invariance 15 is therefore satisfied. Condition (ii) holds by symmetry with the condition (i). Proof of (iii): δA1 δA1 = δA2 δA2 For the gradients with respect to A, the chain rule gives

###### ∇A1L = ∇ZL · A1 ∇A2L = ∇ZL · A2 (17)

where Z = AiA⊤i is the output of the adapted layer. Using the former relation and the Polar Theorem, there exists a matrix Q ∈ R∖×∖ such that,

∇A1L = ∇ZLA1 = (∇ZLA2)Q = ∇A2LQ (18) Hence, the update for A becomes

δA1 = −η ∇A1L = −η ∇A2LQ = δA2 Q. (19) Now, consider the product of the updates:

δA1 δA⊤1 = (δA2 Q)(δA2 Q)⊤ = δA2 (QQ⊤)δA⊤2

= δA2 δA⊤2 , (20) since QQ⊤ = I. This completes the proof of (ii). B.3 Theorem 3

The generalization of SINGLORA to non-square matrix preserves the stability and transformationinvariance properties demonstrated for the square case.

Proof: Recall that for a rectangular weight matrix,

in×dout (din < dout), a low-rank adapter is defined via a matrix A ∈ Rd

W0 ∈ Rd

out×r, with its truncation A∗ ∈ Rd

in×r formed by

the first din rows of A. Suppose that there exist two matrices A1,A2 ∈ Rd

out×r such that their truncations satisfy A∗1A⊤1 = A∗2A⊤2 .

For clarity, divide each Ai as

Xi Yi

Ai =

, i = 1,2,

where Xi ∈ Rd

out−din)×r. By definition, the equality A∗1A⊤1 = A∗2A⊤2 implies,

in×r and Yi ∈ R(d

- X1X1⊤ = X2X2⊤
- X1Y1⊤ = X2Y2⊤.

Using the polar theorem, X2 and Y2 admit a polar decomposition:

###### X2 = X1Q Y2 = Y1Q.

Where Q is a symmetric orthogonal matrix. Denote the gradient descent updates for Ai as,

δAi =

δXi δYi

, i = 1,2.

By equation (22) we have,

(21)

(22)

δA2 =

- δX1 Q
- δY1 Q

. (23)

We now demonstrate that the three sufficient conditions for transformation invariance hold. Proof of (i) and (ii): A1 δA⊤1 = A2 δA⊤2 . Using equations (22) and (23), we get

⊤

- δX1Q
- δY1Q

A∗2 δA⊤2 = X1Q

= [X1QQ⊤δX1⊤|X1QQ⊤δY1⊤]

⊤

- δX1
- δY1

= [X1δX1⊤|X1δY1⊤] = X1

= A∗1δA⊤1 (24) The proof for A∗⊤1 δA1 = A∗⊤2 δA2 is symmetric. Proof of (iii): δA1 δA⊤1 = δA2 δA⊤2 . Using equations (22) and (23), we get

⊤

- δX1Q
- δY1Q

δA∗2 δA⊤2 = δX1Q

= [δX1QQ⊤δX1⊤|δX1QQ⊤δY1⊤]

= [δX1δX1⊤|δX1δY1⊤] =

= δA∗1δA⊤1 (25) This completes the proof of iii.

