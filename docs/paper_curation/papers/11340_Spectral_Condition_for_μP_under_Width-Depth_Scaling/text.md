# arXiv:2603.00541v2[cs.LG]11May2026

[Figure 1]

[Figure 2]

## Spectral Condition for µP under Width–Depth Scaling

Chenyu Zheng1‡, Rongzhen Wang1, Xinyu Zhang2, Chongxuan Li1† 1Gaoling School of AI, Renmin University of China, 2ByteDance Seed ‡Work done during an internship at ByteDance Seed, †Correspondence to Chongxuan Li.

### Abstract

Generative foundation models are increasingly scaled in both width and depth, posing significant challenges for stable feature learning and reliable hyperparameter (HP) transfer across model sizes. While maximal update parameterization (µP) has provided a principled solution to both problems for width scaling, existing extensions to the joint width–depth scaling regime remain fragmented, architecture- and optimizer-specific, and often rely on technically involved theories. In this work, we develop a simple and unified spectral framework for µP under joint width–depth scaling. For deep residual networks whose residual blocks contain k transformations, the framework specifies how the norms of weights and their per-step updates should scale with width and depth. It reveals a fundamental transition from k = 1 to k ≥ 2, unifying previously disparate µP formulations and identifying the k ≥ 2 case as more appropriate for practical architectures with multi-transformation branches such as Transformers. Building on this framework, we derive a general recipe for implementing µP across a broad class of optimizers by mapping spectral constraints to concrete HP parameterizations, recovering existing results and extending them to additional optimizers. Finally, experiments on GPT-2 style language models show that the µP formulation derived from the k ≥ 2 case achieves stable feature learning and robust HP transfer under width–depth scaling, whereas standard parameterization and µP in the k = 1 case often fail to do so. These results support the practical effectiveness of the proposed spectral framework.

Date: Feb 28, 2026 (v1), May 11, 2026 (v2) Project Page: https://github.com/ML-GSAI/Width-Depth-muP

### 1 Introduction

Generative foundation models have been rapidly scaling in both width and depth [16, 22, 24, 33, 40], and this trend is expected to continue in the foreseeable future as datasets grow and task complexity increases. However, when model sizes become sufficiently large (e.g., billions of parameters), feature learning dynamics often become unstable or degenerate [19, 32], and the hyperparameter (HP) tuning becomes prohibitively expensive [44]. These issues pose fundamental obstacles to efficient scaling, underscoring the need for principled methods enabling stable feature learning and reliable HP transfer from small models to larger ones.

Maximal update parameterization (µP) [42, 44] was originally proposed to address both challenges for width scaling, and has recently been preliminarily extended to settings that jointly scale width and depth [5, 6, 10, 30, 46]. By appropriately reparameterizing HPs with model size, µP aims to preserve scale-invariant feature learning, while maximizing the feature change induced by parameter updates, leading to stable and efficient training dynamics [10, 42]. Moreover, µP empirically stabilizes optimal HPs across different scales, enabling

direct transfer of HPs tuned on small models to much larger ones [44, 48].

However, in the joint width–depth scaling regime, existing µP formulations remain preliminary. They are often tightly coupled to specific architectural choices, such as the internal depth of residual blocks [5, 6, 10, 46] and particular optimization algorithms [6, 10, 30, 46]. Moreover, their derivations typically rely on technically involved tools such as Tensor Programs [43, 46] or dynamical mean-field theory [5, 6]. Consequently, it remains difficult for the community to both systematically understand existing results and extend the µP principle to new architectures and optimizers, highlighting the need for a simple and unified theoretical framework.

To address the challenges outlined above, we draw inspiration from the unified spectral perspective developed for width-scaling µP [45]. We extend this spectral perspective to the joint width–depth scaling regime, yielding a simple, unified framework for realizing the µP principle in deep residual networks and systematically deriving µP formulations across a broad class of optimizers. Our main contributions are summarized as follows.

First, we introduce a unified spectral scaling framework for realizing the µP principle in deep residual networks with fixed k-layer residual blocks under width-depth scaling. It specifies how the RMS operator norms of weights and per-step updates should scale with model size. Across architectures with different fixed residual-block depth ks, the spectral condition changes fundamentally from k = 1 (Condition B.1) to k = 2 (Condition 3.1) due to the emergence of higher-order update terms, but yields no essential further change in the resulting µP formulation for k ≥ 2. Besides, Condition B.1 recovers Depth-µP-style results [6, 46], while Condition 3.1 recovers CompleteP-style results [5, 10, 30], which our theory suggests are more appropriate for practical architectures with multi-layer residual branches such as Transformers. Notably, our analysis uses only elementary linear algebra and probability, making it easier to follow than previous works.

Second, building on the proposed spectral condition, we present a unified recipe for implementing µP across a broad class of optimizers by mapping the spectral constraints to concrete HP parameterizations. Concretely, we systematically derive µP parameterizations for Muon-Kimi [26], Muon [21], Shampoo [12], SOAP [38], AdamW [27], Sophia [25], Lion [7], SGD, and Spectral Sphere Optimizer (SSO) [39]. These parameterizations are derived from the optimizers’ update rules rather than ad hoc tuning heuristics; they also recover important existing width-depth µP formulations [5, 6, 10, 30, 46] as special cases.

Finally, through controlled experiments on GPT-2 style Transformer language models [23, 31], we empirically demonstrate that the µP formulation derived from Condition 3.1 achieves stable feature learning and robust HP transfer under joint width-depth scaling. In contrast, standard parameterization (SP) and the µP formulation derived from Condition B.1 are often unstable or fail to transfer HPs reliably. Together, these results support the practical effectiveness of the proposed spectral framework in realistic pretraining settings.

### 2 Preliminaries

We begin by establishing the necessary background for mathematical techniques and µP. Additional related work is discussed in Appendix A.

#### 2.1 Mathematical Notations and Properties

Scaling notation. Let f and g be positive functions of the scaling variables (e.g., width and depth). We use f = O(g), f = Ω(g), and f = Θ(g) in the standard asymptotic sense. In width-scaling settings, the asymptotics are taken with respect to width; in width-depth scaling settings, they are taken with respect to jointly increasing width and depth. Fixed quantities, such as data dimensions, are hidden in the constants.

Vector and matrix norms. We define [n] = {1,2,...,n}. For a vector a ∈ Rn, we use ∥a∥2 and ∥a∥R to denote its ℓ2 norm and Root Mean Square (RMS) norm, respectively. By definition, we have ∥a∥R = ∥a∥2/√n. For a matrix A ∈ Rm×n, we use ∥A∥2, and ∥A∥R to denote its spectral norm and RMS operator norm, respectively. The RMS operator norm is defined as ∥A∥R := maxv̸=0 ∥Av∥

∥v∥R = mn ∥A∥2. Since spectral norm conditions can be equivalently expressed using the RMS operator norm, we adopt the latter to write spectral conditions for notational simplicity throughout this paper. Finally, in the main text, we primarily rely on the following elementary properties of vector and matrix norms.

R

- • Subadditivity: ∥A + B∥R ≤ ∥A∥R + ∥B∥R and ∥a + b∥R ≤ ∥a∥R + ∥b∥R.
- • Submultiplicativity: ∥AB∥R ≤ ∥A∥R∥B∥R and ∥Av∥R ≤ ∥A∥R∥v∥R.
- • Spectral norm of random matrices [37]: for a matrix A ∈ Rm×n with i.i.d. entries sampled from Gaussian distribution N(0,σ2), its spectral norm satisfies ∥A∥2 = Θ σ(√m+√n) with high probability.

#### 2.2 Spectral Condition for µP under Width Scaling

We briefly review µP and its spectral condition under width scaling [45], which serves as the conceptual foundation of our extension to joint width–depth scaling.

Theoretical setup. A canonical setting [45] for analyzing µP under width scaling is the deep linear multi-layer perceptron (MLP) trained with one step on a single data point (x,y). Specifically, we set h0(x) = W0x and denote by Wl the matrix weight at layer l. The network is then defined as

hl(x) = Wlhl−1(x), l ∈ [L + 1],

where the depth L = Θ(1) is fixed, while the model widths scale to infinity. Although highly simplified, this setup captures the core width-scaling behavior of feature learning [45]. Moreover, µP formulations motivated by this setup have been successfully used in practical pretraining [17, 44, 48], including Transformers trained with AdamW, enabling stable feature learning and reliable HP transfer.

µP principle and its spectral condition under width scaling. As network size increases, standard parameterization (SP) typically leads to either exploding or vanishing feature updates. µP resolves this issue by reparameterizing HPs with size to realize the following stable and efficient principle [42].

Principle 2.1 (µP principle). µP aims to realize scale-invariant feature learning while maximizing the feature change induced by parameter updates. Formally, it requires

∥hl(x)∥R = Θ(1), ∥∆hl(x)∥R = Θ(1), l ∈ [L]. (P1) maximize ∆Wl’s contribution to ∆hL(x), l ∈ [L]. (P2)

Under the width-scaling regime, Yang et al. [45] showed that Principle 2.1 is ensured by the following simple spectral scaling condition on the weights and their per-step updates:

∥Wl∥R = Θ(1), ∥∆Wl∥R = Θ(1), l ∈ [L + 1]. (1)

This spectral condition provides a concise and unified perspective on µP under width scaling, from which the HP parameterization of a broad class of optimization algorithms can be derived in a unified and transparent manner [13, 29, 45].

Limitation of the width-scaling condition. The spectral condition (1), however, applies only when depth is fixed. In contrast, modern foundation models scale both width and depth, and existing µP results in this regime rely on complex analyses, with conclusions that depend on specific architectures and optimizers [5, 6, 10, 30, 46]. This motivates our central question: Can we establish a simple and unified spectral perspective in the joint width–depth scaling regime?

### 3 Spectral Condition for µP under Width-Depth Scaling

In this section, we establish the spectral condition for µP under width-depth scaling. We first introduce our problem setup, then derive the corresponding spectral µP condition and discuss its implications.

#### 3.1 Problem Setup

Our setup extends the width-scaling setting in Section 2.2 by introducing residual connections, which are essential for stabilizing training of deep networks [14]. Motivated by practical residual branches that often contain multiple transformations (e.g., attention or FFN modules in Transformers), we consider residual networks whose residual branches contain k = Θ(1) linear transformations. For clarity, the main text focuses on the two-layer residual block (k = 2), which is the minimal setting that captures the essential scaling behavior of any fixed-depth branches with k ≥ 2 while keeping the analysis minimal. The k = 1 case and the general k ≥ 2 case are deferred to Appendices B.1 and B.2, respectively. Formally, the k = 2 network studied in the main text is defined as:

h0(x) = α0W0x, hl(x) = hl−1(x) + αlWl(2)Wl(1)hl−1(x), l ∈ [L] (2) hL+1(x) = αL+1WL+1hL(x),

,Wl(1) ∈ Rn

l×n,Wl(2) ∈ Rn×n

where the weights W0 ∈ Rn×d

L+1×n are all initialized with Gaussian distribution (Wl)ij i.∼i.d. N(0,σl2)1 and trained with layerwise learning rates ηl. Furthermore, {αl}Ll=0+1 are block multipliers that control the effective strength of each transformation.

,WL+1 ∈ Rd

0

l

Following existing µP literature [5, 6, 10, 46], we fix the input and output data dimensions and scale the width and depth to infinity, that is

d0,dL+1 = Θ(1), nl = Θ(n), n,L → ∞. (3)

This setting is standard in Transformer-based large models [33, 36], where n denotes the model width and is typically of the same order as nl (e.g., the feed-forward width). Moreover, we assume ∥x∥R = Θ(1), which holds for common data modalities such as natural images (∥x∥R = Θ(1)) and one-hot encoded language inputs (∥x∥R = 1/d0 = Θ(1)).

In the following subsections, we derive a sufficient and unified spectral condition under this k = 2 setup for realizing the µP Principle 2.1 under joint width-depth scaling.

#### 3.2 Spectral Scaling Condition

Analogous to the width-scaling condition in Equation (1), the width-depth condition has two components. The initial condition on Wl controls forward feature propagation, yielding ∥hl(x)∥R = Θ(1) across depth. The update condition on ∆Wl controls the one-step feature change, ensuring ∥∆hl(x)∥R = Θ(1) while maximizing the direct contribution of weight updates as required by Principle (P2). We now state the resulting sufficient spectral condition.

Condition 3.1 (Spectral condition for µP under joint width-depth scaling, two-layer residual block). To realize µP Principle 2.1, the initial weights and their per-step updates should satisfy:

- • Initial condition. Input and output weights:

α0∥W0∥R = Θ(1), αL+1∥WL+1∥R = Θ(1). (C1.1) Hidden weights:

αl∥Wl(2)∥R ∥Wl(1)∥R = Θ(1/L), l ∈ [L]. (C1.2)

- • Update condition.

1For notation simplicity, when quantities associated with Wl(1) and Wl(2) take the same form, we omit the superscript.

Input and output weights:

α0∥∆W0∥R = Θ(1), αL+1∥∆WL+1∥R = Θ(1). (C2.1) Hidden weights (first-order weight update):

αl∥∆Wl(2)∥R ∥Wl(1)∥R = Θ(1/L), l ∈ [L] αl∥Wl(2)∥R ∥∆Wl(1)∥R = Θ(1/L), l ∈ [L].

Hidden weights (second-order weight update):

(C2.2)

αl∥∆Wl(2)∥R ∥∆Wl(1)∥R = Θ(1/L), l ∈ [L]. (C2.3)

Compared with the width-only condition in Equation (1), Condition 3.1 introduces explicit depth factors for hidden residual blocks. In particular, the products of the residual multiplier and the relevant weight or update norms must scale as Θ(1/L), reflecting the accumulation of residual contributions over L blocks.

The hidden-layer update constraints in Condition 3.1 arise from expanding the one-step feature update ∆hl(x). For a two-layer residual block, this expansion contains first-order terms like ∆Wl(2)Wl(1), where exactly one branch weight is updated, and a second-order term including ∆Wl(2)∆Wl(1), where both branch weights are updated; these give rise to (C2.2) and (C2.3), respectively. By contrast, a one-layer residual block produces only first-order direct update terms, so the second-order constraint is absent.

This update-order viewpoint helps unify prior disparate µP results under joint width-depth scaling [5, 6, 10, 30, 46] by varying the residual block depth k. When k = 1, the update expansion contains no secondorder term, and the same analysis gives Condition B.1 in Appendix B.1. The resulting looser constraints naturally lead to residual multipliers αl = Θ(1/

√

L) under standard width-scaling µP initialization, which recovers Depth-µP-style results [6, 46]. We defer the details to Appendix B.1 and D. In contrast, the k = 2 case introduces the second-order constraint (C2.3), which tightens the hidden residual scaling and leads to residual multipliers αl = Θ(1/L) under the same initialization convention. This recovers CompleteP-style results [5, 10, 30], which are more appropriate for practical architectures with multi-transformation residual branches (e.g., Transformers). Detailed HP parameterizations are given in Section 4 and Appendix C.

Moreover, our analysis naturally extends to residual blocks with any fixed depth k (Condition B.2 in Appendix B.2) and to architectures with biases (Condition B.3 in Appendix B.3). For residual blocks of depth k, the update condition constrains all first- through k-th order update terms to scale as Θ(1/L). Analogous conditions arise in the presence of biases, accounting for their interactions with weight updates. As shown in Appendix B.2 and B.3, these additional constraints do not lead to a different µP formulation compared to the k = 2 case. Therefore, the residual network with two-layer blocks in Equation (2) is the minimal setting that captures the core scaling behavior of practical architectures with multi-layer residual blocks.

Although the spectral results are derived for a linear residual MLP with a one-step update, they generalize to more general and practical training regimes. Theoretically, we introduce and verify some natural assumptions from Yang et al. [45] in Appendix G, under which the spectral results generalize to finite multiple gradient steps, nonlinearities, and finite multiple training examples. Empirically, experiments in Section 5 demonstrate that the resulting µP formulations from Condition 3.1 achieve stable feature learning and reliable HP transfer on GPT-2 style models. Together with prior empirical µP studies [10, 30], these results suggest that the simplified setup captures the core scaling behavior relevant in practice.

Takeaway 1. Residual block depth k determines the depth µP scaling rule: k = 1 gives the Depth-µP-style scaling, whereas k ≥ 2 introduces high-order update terms and requires the stricter CompleteP-style scaling. The latter is better suited to practical architectures such as Transformers.

#### 3.3 Theoretical Derivation

In this section, we derive Condition 3.1 for the residual network in Equation (2). We first obtain a preliminary initialization condition from forward feature stability, which controls the accumulated residual branch at initialization. We then expand the one-step feature update into zero-, first-, and second-order terms. The first- and second-order terms should satisfy the maximal-update requirement in Principle (P2) while keeping the total feature update stable. In particular, the second-order term, absent in one-layer residual blocks, yields the additional constraint (C2.3). Finally, combining these update constraints refines the preliminary initialization condition to Condition 3.1.

Throughout the derivation, we use norm estimates based on subadditivity and submultiplicativity to track the typical scales of ∥hl(x)∥R and ∥∆hl(x)∥R. For example, under standard random initialization, we use estimates of the form ∥h0(x)∥R = α0∥W0x∥R = Θ(α0∥W0∥R∥x∥R). These estimates should be understood as scale estimates whose tightness relies on standard non-cancellation and alignment behavior in neural network initialization and training. We discuss the tightness justification under width-depth scaling in Appendix F, following the width-scaling treatment of Yang et al. [45].

###### 3.3.1 Preliminary Initial Condition

We first derive a preliminary initialization condition that ensures stability of feature magnitudes during forward propagation. We consider each layer sequentially.

Input layer. By submultiplicativity of the RMS operator norm, we can estimate the norm of h0(x) = α0W0x as ∥h0(x)∥R = Θ(α0∥W0∥R∥x∥R) = Θ(α0∥W0∥R), where we have assumed ∥x∥R = Θ(1). Thus, requiring α0∥W0∥R = Θ(1) ensures ∥h0(x)∥R = Θ(1).

Hidden layers. To estimate the scale of hidden features, we expand the residual recursion in Equation (2), which yields

hs(x) = h0(x) +

s

αlWl(2)Wl(1)hl−1(x), s ∈ [L]. (4)

l=1

Applying subadditivity together with the scale-estimation convention described above, we estimate

∥hs(x)∥R = Θ ∥h0(x)∥R +

s

αlWl(2)Wl(1)hl−1(x) R .

l=1

Since we have ∥h0(x)∥R = Θ(1), it suffices to ensure that ∥ sl=1 αlWl(2)Wl(1)hl−1(x)∥R = O(1) for any s ∈ [L] to preserve ∥hs(x)∥R = Θ(1). Under i.i.d. zero-mean Gaussian initialization, the summands are approximately independent zero-mean random vectors [10, 42, 46], so the typical squared RMS norm of their sum scales as the sum of the squared RMS norms (see Theorem 3.3.1 in Vershynin [37]), yielding

that ∥ sl=1 αlWl(2)Wl(1)hl−1(x)∥R = Θ( sl=1 ∥αlWl(2)Wl(1)hl−1(x)∥2R). By submultiplicativity, we can further estimate ∥αlWl(2)Wl(1)hl−1(x)∥R = Θ(αl∥Wl(2)∥R∥Wl(1)∥R∥hl−1(x)∥R). Therefore, starting from ∥h0(x)∥R = Θ(1), imposing

√

αl∥Wl(2)∥R∥Wl(1)∥R = O(1/

L), l ∈ [L],

recursively ensures ∥ sl=1 αlWl(2)Wl(1)hl−1(x)∥R = O(1) for any s ∈ [L]. This provides a preliminary initial condition on the hidden weights, which will be further refined by incorporating update constraints established

in the next subsection.

Output layer. Submultiplicativity gives ∥hL+1(x)∥R = Θ(αL+1∥WL+1∥R∥hL(x)∥R) = Θ(αL+1∥WL+1∥R), where ∥hL(x)∥R = Θ(1) follows from the hidden-layer argument above. Thus choosing αL+1∥WL+1∥R = Θ(1) keeps the output stable.

- 3.3.2 Update Condition We next derive the update condition required to ensure stable feature evolution ∥∆hl(x)∥R = Θ(1) in

- Principle (P1), while maximally updating parameters as prescribed by Principle (P2).

Input layer. Since ∆h0(x) = α0∆W0x, submultiplicativity of matrix norms yields

∥∆h0(x)∥R = Θ(α0∥∆W0∥R∥x∥R) = Θ(α0∥∆W0∥R), and hence we set α0∥∆W0∥R = Θ(1) to ensure ∥∆h0(x)∥R = Θ(1).

Hidden layers. To analyze the hidden feature updates ∆hs(x), we expand the residual representation in Equation (4) after a single gradient step: hs(x)+∆hs(x) = h0(x)+∆h0(x)+ sl=1 αl(Wl(2)+∆Wl(2))(Wl(1)+ ∆Wl(1))(hl−1(x) + ∆hl−1(x)), leading to

∆hs(x) = ∆h0(x) +

s

l=1

αlWl(2)Wl(1)∆hl−1(x)

ϵ0(s)

+

s

l=1

αlWl(2)∆Wl(1)(hl−1(x) + ∆hl−1(x))

ϵ(1)1 (s)

+

s

l=1

αl∆Wl(2)Wl(1)(hl−1(x) + ∆hl−1(x))

ϵ(2)1 (s)

+

s

l=1

αl∆Wl(2)∆Wl(1)(hl−1(x) + ∆hl−1(x))

ϵ2(s)

.

According to the degree of weight updates in the current residual block, we refer to these contributions as the zero-, first-, and second-order update terms, denoted respectively by ϵ0(s), ϵ(1)1 (s),ϵ(2)1 (s), and ϵ2(s). Using subadditivity together with the scale-estimation convention above, we have

∥∆hs(x)∥R = Θ ∥∆h0(x)∥R + ∥ϵ0(s)∥R + ∥ϵ(1)1 (s)∥R + ∥ϵ(2)1 (s)∥R + ∥ϵ2(s)∥R . (5)

Since ∥∆h0(x)∥R = Θ(1) by the input-layer update, we have ∥∆hs(x)∥R = Ω(1) for all s ∈ [L]. Moreover, by subadditivity estimation, the order of remaining terms do not decay with depth, implying ∥∆hs(x)∥R = O(∥∆hL(x)∥R) for any s ∈ [L]. Therefore, to enforce Principle 2.1, it suffices to require ∥∆hL(x)∥R = Θ(1) while satisfying Principle (P2). We next control terms on the right-hand side of Equation (5).

Zero-order term. The term ϵ0(L) propagates feature updates from earlier layers and does not depend on the weight update ∆Wl at the current layer, so it does not need to be maximized from Principle (P2). Therefore, it suffices to verify that ϵ0(L) remains O(1) under the preliminary initial condition. In fact, the same argument used for deriving ∥hL(x)∥R in Section 3.3.1 directly implies ∥ϵ0(L)∥R =

Θ( Ll=1 αl2∥Wl(2)∥2R∥Wl(1)∥2R∥∆hl−1(x)∥2R) = O(1), where we use the self-consistent fact that ∥∆hl−1(x)∥R = Θ(1) for l ∈ [L] if we finally ensure ∥∆hL(x)∥R = Θ(1).

First-order terms. Using subadditivity and submultiplicativity, we estimate the order of ∥ϵ(1)1 (L)∥R as

∥ϵ(1)1 (L)∥R = Θ

L

l=1

αl∥Wl(2)∥R∥∆Wl(1)∥R∥hl−1(x)∥R + Θ

L

l=1

αl∥Wl(2)∥R∥∆Wl(1)∥R∥∆hl−1(x)∥R .

For l ∈ [L], using ∥hl−1(x)∥R = Θ(1) by the preliminary initial condition and ∥∆hl−1(x)∥R = Θ(1) if we finally set ∥∆hL(x)∥R = Θ(1), we can obtain ∥ϵ(1)1 (L)∥R = Θ Ll=1 αl∥Wl(2)∥R∥∆Wl(1)∥R . To satisfy

- Principle (P2), we need to maximize the contribution from each ∆Wl(1) and ensure ∥ϵ(1)1 (L)∥R = Θ(1) at the same time, which naturally requires

αl∥Wl(2)∥R∥∆Wl(1)∥R = Θ(1/L), ∀l ∈ [L].

To control the scale of ϵ(2)1 (L), an identical argument gives αl∥∆Wl(2)∥R∥Wl(1)∥R = Θ(1/L) for every l ∈ [L], which completes the first-order update condition (C2.2).

Second-order term. This is the key term that distinguishes two-layer residual branches from one-layer residual branches because it is absent when k = 1. As a direct update term, it is also subject to Principle (P2). Using

subadditivity and submultiplicativity inequalities as for deriving ∥ϵ(1)1 (L)∥R, we can estimate its scale as

L

αl∥∆Wl(2)∥R∥∆Wl(1)∥R .

∥ϵ2(L)∥R = Θ

l=1

Principle (P2) requires maximizing each summand and ensuring ∥ϵ2(L)∥R = Θ(1) in the meanwhile, leading to αl∥∆Wl(2)∥R∥∆Wl(1)∥R = Θ(1/L) for all l ∈ [L], which completes the derivation for the second-order update condition on hidden weights (C2.3).

Output layer. For the output layer hL+1(x) = αL+1WL+1hL(x), its one-step feature update is

∆hL+1(x) = αL+1WL+1∆hL(x) + αL+1∆WL+1 hL(x) + ∆hL(x) . By subadditivity and submultiplicativity, we estimate

∥∆hL+1(x)∥R = Θ(αL+1∥WL+1∥R∥∆hL(x)∥R) + Θ(αL+1∥∆WL+1∥R∥hL(x) + ∆hL(x)∥R)

= Θ(1) + Θ αL+1∥∆WL+1∥R ,

where we used αL+1∥WL+1∥R, ∥hL(x)∥R = Θ(1) by the preliminary initial condition, and ∥∆hL(x)∥R = Θ(1) by the update condition on the hidden weights. Therefore, requiring Principle 2.1 yields the update condition αL+1∥∆WL+1∥R = Θ(1).

###### 3.3.3 Final Initial Condition

We now derive the final initialization condition for the hidden weights (C1.2) by incorporating the constraints imposed by the update conditions. Multiplying the two first-order update conditions on hidden weights yields

αl2∥Wl(1)∥R∥Wl(2)∥R∥∆Wl(1)∥R∥∆Wl(2)∥R = Θ(1/L2), ∀l ∈ [L].

On the other hand, the second-order update condition is αl∥∆Wl(1)∥R∥∆Wl(2)∥R = Θ(1/L) for all l ∈ [L]. Dividing the product of the first-order conditions by the second-order condition immediately gives

αl∥Wl(1)∥R∥Wl(2)∥R = Θ(1/L) for any l ∈ [L], which completes the derivation of Condition 3.1.

Given the first-order update condition (C2.2), the refined initialization condition (C1.2) and the second-order update condition (C2.3) are algebraically equivalent up to constant factors. Thus, one of them could be derived from the other under (C2.2). We nevertheless keep both in Condition 3.1 because (C1.2) states the final initial forward-scale requirement, while (C2.3) makes explicit the second-order maximal-update constraint responsible for the k = 1 versus k ≥ 2 distinction.

### 4 Implementation of Spectral Condition

In this section, we map Condition 3.1 to concrete HP parameterizations. The initialization parameters σl and αl are optimizer-agnostic, while the learning-rate scaling of ηl depends on the optimizer. In the main text, we instantiate this recipe for Muon-Kimi [26]. Table 2 in Appendix C extends the similar recipe to additional optimizers and optimizer-dependent HPs, including weight decay and stability term ε. The corresponding results for Condition B.1 can be found in Table 8 of Appendix D.

Table 1 µP implementation of Condition 3.1 (k = 2) for Muon-Kimi [26] under width-depth scaling. Entries in purple indicate differences between µP and SP, while gray shows the corresponding SP choices. Here, rn and rL denote the width and depth scaling ratios relative to the base model. The variance of input weights is σbase2 for language and σbase2 /d0 for image.

Input weights Hidden weights Output weights Block Multiplier αbase αbase/rL (αbase) αbase/rn (αbase)

Initial Variance σbase2 /d0 or σbase2 σbase2 /rn (σbase2 ) σbase2 Learning Rate ηbase ηbase/√rn (ηbase) ηbase

#### 4.1 Initial Condition

Since these HPs interact to satisfy the spectral condition, multiple equivalent parameterization solutions exist [42, 43]. To facilitate practical adoption, we choose to align the initial variance to the standard width-scaling µP implementation [44]. Specifically, for any weight matrix Wl ∈ Rn

out×nin, we set:

 

min{1, n

Θ √n 1

nin } , 0 ≤ l ≤ L, Θ(1), l = L + 1.

out

σl =

in



Under this variance parameterization, the RMS operator norms of weight matrices at initialization satisfy:

∥Wl∥R =

nin nout ∥Wl∥2 =

nin nout · Θ σl(√nin + √nout) =

Θ(1), 0 ≤ l ≤ L, Θ(nin), l = L + 1,

(6)

where we used the spectral norm property of random matrices reviewed in Section 2. Based on Equation (6), we are ready to determine the parameterization of αl to satisfy initial conditions in Condition 3.1. For the input and output layers, given

αl∥Wl∥R =

Θ(α0), l = 0, Θ(αL+1nin), l = L + 1,

- to satisfy (C1.1), we need to set α0 = Θ(1), αL+1 = Θ(1/nin). (7)

For the hidden layers, given

αl∥Wl(1)∥R∥Wl(2)∥R = Θ(αl), l ∈ [L],

- to satisfy (C1.2), we need to set αl = Θ(1/L), l ∈ [L]. (8)

#### 4.2 Update Condition for Muon-Kimi

Since different optimizers take different scales of ∥∆Wl∥R, the implementation of the update condition depends on the choice of optimizer. In the main text, we focus on Muon-Kimi [26]. The derivations for Muon [21], Shampoo [12], SOAP [38], AdamW [27], Sophia [25], Lion [7], SGD, and SSO [39] are deferred to Appendix C, where we recover several existing µP results (e.g., for SGD, AdamW, and some matrix-preconditioned optimizers) in the width–depth scaling setting [5, 6, 10, 30, 46].

Muon-Kimi [26] is a widely used variant of Muon [21] designed to align its update scales of matrix parameters with those of AdamW-optimized vector parameters by applying RMS normalization, which facilitates the

reuse of HPs well-tuned for AdamW. It has been successfully applied to pretraining models with up to 1T parameters [34]. Specifically, for a weight matrix Wl ∈ Rn

out×nin, the update rule (without weight decay) is: ∆Wl = −ηl · 0.2 max{nin,nout} · UlVl⊤,

where Ul,Vl are the left and right singular vector matrices of the gradient that ∇WlL = UlΣlVl⊤. The resulting update norm satisfies

 

  =

 ηl√nin max 1,

Θ(ηl), l = 0, Θ(ηl√nin), l ∈ [L], Θ(ηlnin), l = L + 1.

nin nout∥∆Wl∥2 = Θ

nin nout

(9)

∥∆Wl∥R =



Based on Equation (9), we are now ready to determine the parameterization of ηl to satisfy the update condition.

For the input and output layers, given the dimension magnitude assumption in Equation (3) and the αl parameterization in Equation (7), we have:

Θ(1)Θ(ηl), l = 0, Θ(1/nin)Θ(ηlnin), l = L + 1,

αl∥∆Wl∥R =

= Θ(ηl).

- Thus, to satisfy (C2.1), we need to set: η0 = Θ(1), ηL+1 = Θ(1).

For the hidden layers, let us first consider the first-order update condition. Given the dimension magnitude in Equation (3), the weight norm at initialization in Equation (6), and the αl parameterization in Equation (8), we have:

αl∥∆Wl(2)∥R∥Wl(1)∥R = Θ(1/L) · ∥∆Wl(2)∥R = Θ

1 L

ηl(2)√nin .

- Thus, to satisfy (C2.2), we need to set:

1 √nin

ηl(2) = Θ(

), l ∈ [L]. (10)

Symmetrically, we have the same choice for Wl(1) that ηl(1) = Θ(√n1

) to ensure the first-order condition.

in

By the algebraic equivalence discussed in Section 3.3.3, the second-order condition (C2.3) is then automatically satisfied by the initial condition (C1.2) and the first-order condition (C2.2), so no further constraint is needed for implementing the second-order condition (C2.3).

This completes the µP parameterization of Muon-Kimi, which is summarized in Table 1.

#### 4.3 Simplified Implementation Rule for Modern Optimizers

The Muon-Kimi derivation above, together with derivations for other optimizers in Appendix C, reveals a useful implementation-level simplification. For the modern optimizers considered in this paper, except SGD, implementing Condition 3.1 amounts to taking the corresponding width-scaling µP implementation and adding the hidden residual multiplier αl = Θ(1/L). The reason is that normalized and preconditioned updates remove the depth factor αl inherited by the raw hidden-layer gradients. As a result, their update norms do not depend on αl, and the optimizer-specific HP rule remains the same as in width-scaling µP. SGD does not share this simplification because its update is proportional to the raw gradient. After setting αl = Θ(1/L), the spectral update condition therefore still requires an additional αl-dependent learning rate rescaling.

Takeaway 2. For most modern optimizers, normalization or preconditioning removes the depth factor αl of hidden gradients, so the µP formulation from Condition 3.1 is just width-scaling µP plus a hidden residual multiplier αl = Θ(1/L).

#### 4.4 Practical HP Parameterization and Transfer

In practice, µP is often implemented using a ratio-based approach [10, 44, 48]. We define width and depth scaling ratios as rn = n/nbase and rL = L/Lbase, where nbase and Lbase are some fixed base model constants. The target model’s HPs are then set by scaling the corresponding base HPs, denoted as αbase, σbase2 , and ηbase, according to these ratios.

For instance, for Muon-Kimi, the hidden layer learning rate is set to ηl = ηbase/√rn, which satisfies the theoretical requirement ηl = ηbase/ n/nbase = Θ(ηbase/√n) in Equation (10). Table 1 summarizes the complete HP parameterization for Muon-Kimi under width-depth scaling derived above.

As illustrated in Section 1, a critical utility of µP is enabling HP transfer, which effectively reduces the cost of HP search for training large models. In practice, the transfer follows this procedure: optimal base HPs (e.g., ηbase) are first identified on a small model; these optimal values are then transferred to a larger target model to obtain true HPs (e.g., ηbase/√rn). Consequently, we only need to search the base HPs on the computationally inexpensive small model.

Note that although the µP parameterization is derived from a simplified setup, we apply it to standard language models pretraining and verify its practical utility in the next section.

### 5 Experiments

In this section, we empirically evaluate the µP formulations derived from our spectral conditions on GPT-2 style language models. We first show that Condition 3.1 (k ≥ 2)2 enables stable feature learning and robust HP transfer under width-depth scaling. We then compare it with Condition B.1 (k = 1) to validate the role of residual block depth k on depth scaling. The complete details and results are deferred to Appendix E.

#### 5.1 Experimental Settings

Following standard empirical µP studies [10, 30], we train GPT-2 style Transformer language models [23, 31] on the OpenWebText dataset [11], using the GPT-2 tokenizer with a maximum sequence length of 1024. All models fix the attention head dimension to 64 and use a feedforward (FFN) dimension of 4n. We vary model width and depth around a base model (nbase,Lbase) = (256,4), with widths up to 4096 and depths up to 256. Since SGD and AdamW have been extensively studied in prior µP studies under width-depth scaling [5, 6, 10, 46], we focus on more recent optimizers: Muon-Kimi-AdamW, Muon-AdamW, Shampoo-AdamW, and Sophia. Following common practice [26], Muon-Kimi-AdamW uses Muon-Kimi for hidden matrix parameters and AdamW for all other parameters, such as embeddings, the LM head, and biases. Muon-AdamW and Shampoo-AdamW are defined analogously. In the main text, we present Muon-Kimi-AdamW as the primary example; additional optimizer results are reported in Appendix E.

We implement µP for both Condition 3.1 (k ≥ 2) and Condition B.1 (k = 1); the corresponding HP parameterization overviews are given in Table 2 at Appendix C and Table 8 at Appendix D, respectively. For example, for the hybrid Muon-Kimi-AdamW optimizer under Condition 3.1, the Muon-Kimi part follows Table 3, while the AdamW part follows Table 6. Muon-AdamW and Shampoo-AdamW are implemented analogously, with details deferred to Appendix E. The main text focuses on learning rate transfer without weight decay, and weight decay transfer results are deferred to Figure 4 in Appendix E.3.

#### 5.2 Feature Learning and HP Transfer

In this section, we compare the feature learning stability and HP transferability of SP and the µP formulation derived from Condition 3.1 (k ≥ 2). The main results are shown in Figure 1, with complete numerical results deferred to Appendix E.3.2.

2In fact, Condition 3.1 is derived when k = 2. Since Condition B.2 (k ≥ 2) leads to the same µP formulation, we use k ≥ 2 to refer to this formulation throughout the experiments.

###### (a) P

###### SP

|Step| | | |
|---|---|---|---|
|5<br><br>6<br><br>7<br>| || |
|---|
<br><br>| |
|8<br><br>9<br>|| |
|---|
<br><br>| |
|---|
<br><br>| | |
|1|0| | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

24

22

hLR

20

2 2

28 210 212

28 210 212

Width n

Width n

###### (b) P

SP

26

|Step<br><br>|5| | |
|---|---|---|---|
| |6<br>7<br>|| |
|---|
| |
<br><br>| |
| || |
|---|
<br><br>8<br>9<br>| | |
| |10| | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

24

hLR

22

20

2 2

22 24 26

22 24 26

Depth L

Depth L

(c) P

SP

| |
|---|

5.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

Width n

ValidationLoss

128 256 512

4.5

1024 2048 4096

4.0

3.5

2 10 2 9 2 8 2 7 2 6 2 5

2 10 2 9 2 8 2 7 2 6 2 5

Base Learning Rate base

Base Learning Rate base

###### (d) P

SP

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

Depth L

4.2

ValidationLoss

4 8

4.0

16 32 64

3.8

128 256

2 9 2 8 2 7 2 6 2 5

2 9 2 8 2 7 2 6 2 5

Base Learning Rate base

Base Learning Rate base

- Figure 1 Feature learning and HP transfer of Muon-Kimi-AdamW under SP and µP. We train GPT-2 style models with Muon-Kimi-AdamW using SP and µP derived from Condition 3.1 (see Tables 1 and 6). µP maintains stable feature norms and enables robust HP transfer across both width and depth scaling, while generally achieving lower loss than SP as the model size increases. The detailed numerical values are provided in Appendix E.3.2.

Feature learning. We first examine feature-scale stability using standard coordinate-check tests [10, 29, 44]. Models are trained for 10 steps while scaling either width n or depth L, and we measure the RMS norm at the output of the final Transformer block ∥hL∥R. As shown in Figure 1(a,b), under SP, the feature scale grows rapidly with both width and depth. In contrast, µP maintains stable and scale-invariant feature scales, consistent with Principle 2.1. This supports that the µP formulation derived from Condition 3.1 preserves stable feature learning under width-depth scaling.

HP transfer. We next evaluate HP transferability by training all models for 300M tokens with a batch size of 240, using a learning rate schedule with linear warmup followed by cosine decay. As shown in Figure 1(c), SP exhibits substantial shifts in the optimal learning rate when width is scaled, whereas µP keeps the optimal base learning rate nearly invariant. Under depth scaling, µP also preserves HP transferability and achieves lower loss than SP as depth increases (Figure 1(d)). These results support that Condition 3.1 provides a practical HP-transfer rule, which can significantly reduce tuning cost when scaling model size, particularly for pretraining large models [33, 34].

Figures 5, 6, and 8 in Appendix E report analogous feature learning and HP transfer results for Muon-AdamW, Shampoo-AdamW, and Sophia, respectively. They show the similar advantage of the µP formulation from Condition 3.1 over SP.

#### 5.3 Role of Residual Block Depth

We further examine the central prediction of our theory: residual-branch depth k determines the appropriate depth µP formulation. Condition B.1 gives the Depth-µP-style scaling for k = 1, whereas Condition 3.1 gives the stricter CompleteP-style scaling for k ≥ 2. Since the GPT-2 style Transformer contains multiple transformations in each residual branch, our theory predicts that Condition 3.1 should provide a better parameterization.

To test this prediction, we compare the two µP implementations under depth scaling. As shown in Figure 2, Condition 3.1 yields stable learning-rate transfer for both Muon-Kimi-AdamW and Muon-AdamW. In contrast,

- Condition B.1 shifts the optimal learning rate as depth increases, indicating a failure of HP transfer in practical multi-transformation architectures. These results support the claim in Section 3: the high-order update term appearing at k ≥ 2 imposes the stricter scaling needed for Transformer-like residual blocks. The same comparison for Shampoo-AdamW and Sophia in Figures 7 and 9 at Appendix E.3 shows similar

(a) Muon-Kimi-AdamW, P (k 2)

Muon-Kimi-AdamW, P (k = 1)

| |
|---|

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

4.2

Depth L

ValidationLoss

4 8

4.0

16 32 64

3.8

128 256

2 9 2 8 2 7 2 6 2 5 2 4

2 9 2 8 2 7 2 6 2 5

Base Learning Rate base

Base Learning Rate base

(b) Muon-AdamW, P (k 2)

Muon-AdamW, P (k = 1)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

Depth L

4.4

ValidationLoss

4 8

4.2

16 32 64

4.0

128 256

3.8

2 7 2 6 2 5 2 4 2 3 2 2

2 7 2 6 2 5 2 4 2 3

Base Learning Rate base

Base Learning Rate base

- Figure 2 Validating the role of residual block depth k. We compare two µP implementations for GPT-2-style models trained with Muon-Kimi-AdamW and Muon-AdamW: Depth-µP-style formulation from Condition B.1 (k = 1) and CompleteP-style formulation from Condition 3.1 (k ≥ 2). Condition 3.1 yields more stable HP transfer, empirically supporting it as the appropriate µP condition for architectures with multi-transformation residual branches. The numerical values are provided in Appendix E.3.

trends, further supporting the effectiveness of Condition 3.1 (k ≥ 2) beyond Muon-style optimizers.

#### 5.4 Additional Diagnostics

For Muon-Kimi-AdamW, SP can appear to transfer the optimal learning rate reasonably well across depths in Figure 1(d). We attribute this to two factors. First, the tested depths are still moderate; as depth increases further, Figure 1(b) suggests that hidden features eventually diverge, making stable depth scaling under SP infeasible. Second, modern architectural components such as LayerNorm [1] and QKNorm [15] substantially enhance training stability, partially masking the underlying scaling pathology of SP at practical depths. To isolate this effect, we remove LayerNorm layers and repeat the depth-scaling experiments in Appendix E.3.2. The results in Figure 3 show that SP training becomes unstable and depth-wise HP transfer breaks down, while µP remains stable even at large depths (L = 256) and continues to exhibit robust HP transfer.

We emphasize that this apparent SP depth transfer is not universal. Figure 6 and 8 in Appendix E show that for Shampoo-AdamW and Sophia, the µP parameterization gives substantially more reliable depth-wise HP transfer than SP. Thus, the Muon-Kimi-AdamW behavior in Figure 1(d) appears to be a partially masked case, likely aided by normalization and the tested depth range.

Takeaway 3. CompleteP-style scaling from Condition 3.1 (k ≥ 2) achieves stable feature learning and robust HP transfer under width-depth scaling, while SP and Depth-µP-style scaling from Condition B.1

- (k = 1) often fail to do so.

### 6 Conclusion

In this paper, we present a simple and unified spectral framework for µP under joint width-depth scaling. The framework gives an operational condition on the norm of weights and their per-step updates, and explains why residual blocks with one transformation and those with multiple transformations lead to different depth-scaling rules. By mapping this condition to concrete HP choices, we obtain a general recipe for implementing µP across a broad class of optimizers. Experiments on GPT-2 style language models show that the resulting k ≥ 2 formulation preserves scale-invariant feature learning and supports robust HP transfer, while the k = 1 formulation and SP often fail to do so. These results suggest that the proposed spectral perspective provides a practical and interpretable route to width-depth µP for modern architectures.

### References

- [1] Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. CoRR, abs/1607.06450, 2016.

- [2] Laura Balzano, Tianjiao Ding, Benjamin D. Haeffele, Soo Min Kwon, Qing Qu, Peng Wang, Zhangyang Wang, and Can Yaras. An overview of low-rank structures in the training and adaptation of large models. CoRR, abs/2503.19859, 2025.

- [3] Charlie Blake, Constantin Eichenberg, Josef Dean, Lukas Balles, Luke Yuri Prince, Björn Deiseroth, Andrés Felipe Cruz-Salinas, Carlo Luschi, Samuel Weinbach, and Douglas Orr. u-µp: The unit-scaled maximal update parametrization. In ICLR, 2025.

- [4] Blake Bordelon and Cengiz Pehlevan. Self-consistent dynamical field theory of kernel evolution in wide neural networks. In NeurIPS, 2022.

- [5] Blake Bordelon, Hamza Tahir Chaudhry, and Cengiz Pehlevan. Infinite limits of multi-head transformer dynamics. In NeurIPS, 2024.

- [6] Blake Bordelon, Lorenzo Noci, Mufan Bill Li, Boris Hanin, and Cengiz Pehlevan. Depthwise hyperparameter transfer in residual networks: Dynamics and scaling limit. In ICLR, 2024.

- [7] Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V. Le. Symbolic discovery of optimization algorithms. In NeurIPS, 2023.

- [8] Nolan Dey, Gurpreet Gosal, Zhiming Chen, Hemant Khachane, William Marshall, Ribhu Pathria, Marvin Tom, and Joel Hestness. Cerebras-gpt: Open compute-optimal language models trained on the cerebras wafer-scale cluster. CoRR, abs/2304.03208, 2023.

- [9] Nolan Dey, Shane Bergsma, and Joel Hestness. Sparse maximal update parameterization: A holistic approach to sparse training dynamics. In NeurIPS, 2024.

- [10] Nolan Dey, Bin Claire Zhang, Lorenzo Noci, Mufan Bill Li, Blake Bordelon, Shane Bergsma, Cengiz Pehlevan, Boris Hanin, and Joel Hestness. Don’t be lazy: Completep enables compute-efficient deep transformers. CoRR, abs/2505.01618, 2025.

- [11] Aaron Gokaslan and Vanya Cohen. Openwebtext corpus. http://Skylion007.github.io/OpenWebTextCorpus, 2019.
- [12] Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization. In International Conference on Machine Learning, pages 1842–1850. PMLR, 2018.

- [13] Moritz Haas, Jin Xu, Volkan Cevher, and Leena Chennuru Vankadara. µp2: Effective sharpness aware minimization requires layerwise perturbation scaling. In NeurIPS, 2024.

- [14] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016.

- [15] Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Trevor Cohn, Yulan He, and Yang Liu, editors, Findings of the Association for Computational Linguistics: EMNLP 2020, volume EMNLP 2020, pages 4246–4253, 2020.

- [16] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. CoRR, abs/2203.15556, 2022.

- [17] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

- [18] Satoki Ishikawa and Ryo Karakida. On the parameterization of second-order optimization effective towards the infinite width. In ICLR, 2024.

- [19] Arthur Jacot, Clément Hongler, and Franck Gabriel. Neural tangent kernel: Convergence and generalization in neural networks. In NeurIPS, pages 8580–8589, 2018.

- [20] Keller Jordan, Jeremy Bernstein, Brendan Rappazzo, @fernbear.bsky.social, Boza Vlado, You Jiacheng, Franz Cesista, Braden Koszarsky, and @Grad62304977. modded-nanogpt: Speedrunning the nanogpt baseline, 2024. URL https://github.com/KellerJordan/modded-nanogpt.
- [21] Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cecista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks. URL https://kellerjordan. github. io/posts/muon, 6, 2024.

- [22] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. CoRR, abs/2001.08361, 2020.

- [23] Andrej Karpathy. nanogpt. https://github.com/karpathy/nanoGPT, 2022.
- [24] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

- [25] Hong Liu, Zhiyuan Li, David Leo Wright Hall, Percy Liang, and Tengyu Ma. Sophia: A scalable stochastic second-order optimizer for language model pre-training. In ICLR, 2024.

- [26] Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982, 2025.

- [27] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.

- [28] Yurii Nesterov. A method for solving the convex programming problem with convergence rate o(1/k2). In Dokl akad nauk Sssr, volume 269, page 543, 1983.

- [29] Marieme Ngom, Sam Foreman, Venkatram Vishwanath, et al. Extending µp: Spectral conditions for feature learning across optimizers. In OPT 2025: Optimization for Machine Learning, 2025.

- [30] Shikai Qiu, Zixi Chen, Hoang Phan, Qi Lei, and Andrew Gordon Wilson. Hyperparameter transfer enables consistent gains of matrix-preconditioned optimizers across scales. arXiv preprint arXiv:2512.05620, 2025.

- [31] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

- [32] Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli, and Jascha Sohl-Dickstein. Deep information propagation. In ICLR, 2017.

- [33] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

- [34] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

- [35] Leena Chennuru Vankadara, Jin Xu, Moritz Haas, and Volkan Cevher. On feature learning in structured state space models. In NeurIPS, 2024.

- [36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, pages 5998–6008, 2017.

- [37] Roman Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018.

- [38] Nikhil Vyas, Depen Morwani, Rosie Zhao, Mujin Kwun, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham Kakade. Soap: Improving and stabilizing shampoo using adam. arXiv preprint arXiv:2409.11321, 2024.

- [39] Tian Xie, Haoming Luo, Haoyu Tang, Yiwen Hu, Jason Klein Liu, Qingnan Ren, Yang Wang, Wayne Xin Zhao, Rui Yan, Bing Su, et al. Controlled llm training on spectral sphere. arXiv preprint arXiv:2601.08393, 2026.

- [40] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

- [41] Greg Yang. Tensor programs III: neural matrix laws. CoRR, abs/2009.10685, 2020.

- [42] Greg Yang and Edward J. Hu. Tensor programs IV: feature learning in infinite-width neural networks. In ICML, volume 139, pages 11727–11737. PMLR, 2021.

- [43] Greg Yang and Etai Littwin. Tensor programs ivb: Adaptive optimization in the infinite-width limit. CoRR, abs/2308.01814, 2023.

- [44] Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs V: tuning large neural networks via zero-shot hyperparameter transfer. CoRR, abs/2203.03466, 2022.

- [45] Greg Yang, James B. Simon, and Jeremy Bernstein. A spectral condition for feature learning. CoRR, abs/2310.17813, 2023.

- [46] Greg Yang, Dingli Yu, Chen Zhu, and Soufiane Hayou. Tensor programs VI: feature learning in infinite depth neural networks. In ICLR, 2024.

- [47] Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian. Galore: Memory-efficient LLM training by gradient low-rank projection. In ICML, 2024.

- [48] Chenyu Zheng, Xinyu Zhang, Rongzhen Wang, Wei Huang, Zhi Tian, Weilin Huang, Jun Zhu, and Chongxuan Li. Scaling diffusion transformers efficiently via µp. CoRR, abs/2505.15270, 2025.

### Contents of Appendix

###### A Additional Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- A.1 µP under Width Scaling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.2 µP under Width-Depth Scaling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

###### B Spectral Condition for General Residual Networks . . . . . . . . . . . . . . . . . . . . . . . . 18

- B.1 One-layer Residual Block . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B.2 Multi-layer Residual Block . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- B.3 Bias Parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

###### C Implementing Condition 3.1 for Optimizers with Weight Decay . . . . . . . . . . . . . . . . 31

- C.1 Preparation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- C.2 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- C.3 Muon-Kimi . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- C.4 Muon . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- C.5 SGD . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- C.6 AdamW . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- C.7 Lion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- C.8 Sophia . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- C.9 Shampoo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- C.10 SOAP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- C.11 Spectral Sphere Optimizer (SSO) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45

###### D Implementing Condition B.1 for Optimizers with Weight Decay . . . . . . . . . . . . . . . . 46

- D.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
- D.2 Muon-Kimi . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
- D.3 Muon, Shampoo and SOAP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
- D.4 SGD . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
- D.5 AdamW, Sophia and Lion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
- D.6 Spectral Sphere Optimizer (SSO) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53

###### E Additional Details and Results of GPT-2 Experiments . . . . . . . . . . . . . . . . . . . . . . 54

- E.1 Assets and Licenses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54
- E.2 Additional Details of Feature Learning Experiments . . . . . . . . . . . . . . . . . . . . . . . 55
- E.3 Additional Details of HP Transfer Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . 55

###### F Justification of Upper Bound Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71

- F.1 Subadditivity Inequalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71
- F.2 Submultiplicativity Inequalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71

###### G Extension to General Training Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73

- G.1 Assumptions for Extensions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
- G.2 Experimental Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 75

### Appendix A Additional Related Work

- A.1 µP under Width Scaling

µP was originally introduced to characterize and control training dynamics in the infinite-width limit of neural networks, to enable stable feature learning through appropriate HPs adjustment [42]. Early theoretical work formalized µP for MLP trained with SGD using Tensor Programs [41, 42] and dynamical mean-field theory [4]. Empirically, Yang et al. [44] showed that µP stabilizes optimal HPs across model widths, thereby substantially reducing the tuning cost when scaling up model size.

Motivated by these advantages, the µP principle has been successfully extended to a wide range of modern architectures, including convolutional neural networks [43], Transformers [43], diffusion Transformers [48], and state-space models [35]. In parallel, µP has been developed for a broad class of optimization algorithms, such as AdamW [27], Muon [29], sharpness-aware optimizer [13], second-order optimizers [18], low-precision training [3], and sparse training [9]. These µP-based methods have also been successfully applied to the pretraining of large-scale foundation models in industrial settings [8, 17, 44, 48].

Despite substantial progress, µP formulations are often tightly coupled to specific architectures [35, 43, 48] or particular optimization algorithms [13, 18, 29, 43], and their derivations typically rely on technically involved tools such as Tensor Programs or dynamical mean-field theory [4, 41–43]. As a result, it remains difficult to systematically analyze new architectures or optimizers and derive the corresponding µP formulations. To alleviate this limitation, Yang et al. [45] proposed a simple and general spectral condition for realizing µP in the width-scaling regime, enabling transparent derivations for a broad class of optimization algorithms [13, 29, 45]. However, this spectral perspective focuses solely on width scaling and does not account for depth scaling, which is crucial for modern deep architectures.

- A.2 µP under Width-Depth Scaling

Recent work has begun to extend the µP principle beyond pure width scaling to regimes where network depth grows jointly with model size. Early theoretical analyses [6, 46] of residual networks with one-layer residual blocks trained by SGD or Adam showed that a hidden residual multiplier αl of order Θ(1/

√

L) suffices to preserve stable feature learning, but observed that this scaling fails to maintain HP transferability in practical architectures such as Transformers [10, 46].

Subsequent studies [5] of Transformers with two-layer residual blocks trained by SGD using dynamical meanfield theory argued that a stronger hidden residual scaling of Θ(1/L) is preferable, as it enables both nontrivial feature learning and non-negligible updates in attention layers. More recently, Dey et al. [10] shows that for residual networks with two-layer blocks trained by AdamW, the residual multiplier αl of Θ(1/L) is in fact necessary to simultaneously maintain stable feature learning and maximize parameter updates. Moreover, Dey et al. [10] empirically find that this parameterization enables HP transfer in GPT-2-style Transformer. This hidden residual multiplier [5, 10] is further applied to some matrix-preconditioned optimizers [30].

Overall, existing µP extensions to the joint width-depth scaling regime remain fragmented, architecture- and optimizer-specific, and often rely on technically involved analyses, motivating the need for a simple and unified framework.

### Appendix B Spectral Condition for General Residual Networks

In this section, we provide derivations that complement the main-text analysis of the two-layer residual block

- (k = 2) in Section 3. The goal is to clarify how the spectral condition changes with the internal depth k of the residual branch and to justify why the two-layer case is the minimal representative of fixed-depth branches with k ≥ 2. We first analyze one-layer residual blocks (k = 1), where the absence of second-order update terms leads to the looser Depth-µP-style scaling in Condition B.1. We then extend the analysis to residual blocks with an arbitrary fixed number k ≥ 2 of transformations, showing that the additional higher-order update terms do not change the resulting µP implementation beyond the two-layer case. Finally, we discuss bias parameters and show that they can be incorporated as a lightweight extension without changing the main HP parameterization for the matrix weights.

As in the main text, we assume ∥x∥R = Θ(1) for simplicity, which holds for natural image data and one-hot language data (Θ(1/√d0) = Θ(1)). We also assume the network dimensions satisfy Equation (3). Furthermore, we also use norm estimates based on subadditivity and submultiplicativity to track the typical scales of ∥hl(x)∥R and ∥∆hl(x)∥R. The tightness justification under width-depth scaling is in Appendix F, following the width-scaling treatment of Yang et al. [45].

#### B.1 One-layer Residual Block

- B.1.1 Problem Setup We consider a residual network with one-layer residual blocks (k = 1), defined as

h0(x) = α0W0x, hl(x) = hl−1(x) + αlWlhl−1(x), ∀l ∈ [L],

hL+1(x) = αL+1WL+1hL(x), where W0 ∈ Rn×d

0, Wl ∈ Rn×n for l ∈ [L], and WL+1 ∈ Rd

L+1×n. The network output hL+1(x) ∈ Rd

L+1 is used to compute the loss L(hL+1(x),y). This case serves as a reference point for understanding the transition from k = 1 to k ≥ 2: because each residual branch contains only one transformation, its update expansion has no second-order direct update term.

- B.1.2 Spectral Scaling Condition

We now state the spectral scaling condition for the above residual network with one-layer blocks for realizing the µP Principle 2.1 under joint width–depth scaling.

- Condition B.1 (Spectral condition for µP under joint width-depth scaling, one-layer residual block). To realize µP Principle 2.1, the initial weights and their per-step updates should satisfy:

###### • Initial condition.

- – Input and output weights: α0∥W0∥R, αL+1∥WL+1∥R = Θ(1).
- – Hidden weights: αl∥Wl∥R = O(1/

√

L), ∀l ∈ [L].

###### • Update condition.

- – Input and output weights: α0∥∆W0∥R, αL+1∥∆WL+1∥R = Θ(1).
- – Hidden weights (first-order): αl∥∆Wl∥R = Θ(1/L), ∀l ∈ [L].

The essential distinction between one-layer and two-layer residual blocks lies in the order of the weight-update terms that directly affect feature evolution. For a one-layer residual block, the feature update expansion contains only zero-order (ϵ0(L)) and first-order (ϵ1(L)) terms in the weight updates (see details in Appendix B.1.4). As a result, only the first-order direct update term needs to be maximized under Principle P2, leading to the condition αl∥∆Wl∥R = Θ(1/L), while leaving the initialization scale unconstrained beyond the preliminary condition αl∥Wl∥R = O(1/

√

L).

From an algorithmic (HP parameterization) perspective, when the initialization variance σl2 is aligned with the standard width-scaling µP framework [44] as in Section 4, the condition αl∥Wl∥R = O(1/

√

L) naturally induces an O(1/

√

√

L) residual multiplier. Moreover, Depth-µP-style formulation [6, 46] adopts the Θ(1/

L) residual multiplier, which they interpret as further promoting feature diversity. Within our spectral framework, this choice is unified as a natural case corresponding to further maximizing the magnitude of the zero-order feature update ∥ϵ0(L)∥R (see derivations in Appendix B.1.4). The parameterization of other HPs (e.g., learning rate) in Depth-µP-style formulation [6, 46] can also be recovered from Condition B.1, with the details deferred to Appendix D.

In contrast, two-layer residual blocks introduce second-order update terms arising from products of weight updates across the two sublayers. To satisfy the µP principle (P2), these second-order contributions should be

maximized to Θ(1) as in (C2.3). This requirement imposes an additional constraint on the scaling of weight updates, which in turn tightens the initialization condition to αl∥Wl(1)∥R ∥Wl(2)∥R = Θ(1/L) in (C1.2) and thus the residual multiplier to αl = Θ(1/L). This important difference explains why the Depth-µP-style scaling (k = 1) does not directly capture residual branches with two or more transformations, and helps account for its poor depth-wise HP transfer behavior in Transformer experiments in Section 5 and Dey et al. [10], Yang et al. [46].

###### B.1.3 Derivation for Initial Condition

We first derive the initialization condition that ensures stability of feature magnitudes during forward propagation for single-layer residual blocks. We consider each layer sequentially.

Input layer. The argument is identical to the two-layer case. By the submultiplicativity of the RMS operator norm, we have

∥h0(x)∥R = α0∥W0x∥R = Θ(α0∥W0∥R∥x∥R) = Θ(α0∥W0∥R), where we assume ∥x∥R = Θ(1). Thus, choosing α0∥W0∥R = Θ(1) ensures ∥h0(x)∥R = Θ(1). Hidden layers. For a single-layer residual block, the forward recursion is

hl(x) = hl−1(x) + αlWlhl−1(x). Expanding the recursion yields

hs(x) = h0(x) +

s

αlWlhl−1(x). (11)

l=1

Applying subadditivity, we can estimate their order as

 ∥h0(x)∥R + ∥

 .

s

∥hs(x)∥R = Θ

αlWlhl−1(x)∥R

l=1

Since we have ∥h0(x)∥R = Θ(1), it suffices to ensure that ∥ sl=1 αlWlhl−1(x)∥R = O(1) for any s ∈ [L] to preserve ∥hs(x)∥R = Θ(1). Under i.i.d. zero-mean Gaussian initialization, the summands are approximately independent zero-mean random vectors [10, 42, 46], so the typical squared RMS norm of their sum scales as the sum of the squared RMS norms (see Theorem 3.3.1 in Vershynin [37]), yielding that

  

  .

s

s

∥αlWlhl−1(x)∥2R

∥

αlWlhl−1(x)∥R = Θ

l=1

l=1

By submultiplicativity, we can further estimate ∥αlWlhl−1(x)∥R = Θ(αl∥Wl∥R∥hl−1(x)∥R). Therefore, starting from ∥h0(x)∥R = Θ(1), imposing

√

αl∥Wl∥R = O(1/

L), l ∈ [L]

recursively ensures ∥ sl=1 αlWlhl−1(x)∥R = O(1) for any s ∈ [L]. This provides the initial condition on the hidden weights.

Output layer. The same argument as for the two-layer block case gives

∥hL+1(x)∥R = ∥αL+1WL+1hL(x)∥R = Θ(αL+1∥WL+1∥R∥hL(x)∥R) = Θ(αL+1∥WL+1∥R), so choosing αL+1∥WL+1∥R = Θ(1) keeps the output scale stable. This completes the initialization analysis.

###### B.1.4 Derivation for Update Condition

We next derive the update condition required to ensure stable feature evolution, i.e., ∥∆hl(x)∥R = Θ(1), while maximally updating parameters as prescribed by µP Principle (P2).

Input layer. Since ∆h0(x) = α0∆W0x, submultiplicativity yields

∥∆h0(x)∥R = Θ(α0∥∆W0∥R∥x∥R) = Θ(α0∥∆W0∥R), and thus we set α0∥∆W0∥R = Θ(1). Hidden layers. Expanding Equation (11) after a single gradient step gives

∆hs(x) = ∆h0(x) +

s

+

αlWl∆hl−1(x)

l=1

ϵ0(s)

s

.

αl∆Wl(hl−1(x) + ∆hl−1(x))

l=1

ϵ1(s)

Unlike the two-layer case, there is no second-order update term, since each residual block contains only a single weight matrix. By the subadditivity of vector norms, we have

∥∆hs(x)∥R = Θ(∥∆h0(x)∥R + ∥ϵ0(s)∥R + ∥ϵ1(s)∥R).

Since ∥∆h0(x)∥R = Θ(1) by the input-layer update, we have ∥∆hs(x)∥R = Ω(1) for all s ∈ [L]. Moreover, by subadditivity, the remaining terms do not decay with depth, implying ∥∆hs(x)∥R = O(∥∆hL(x)∥R) for any s ∈ [L]. Therefore, to enforce Principle 2.1, it suffices to require ∥∆hL(x)∥R = Θ(1) while satisfying Principle (P2).

Zero-order term. The term ϵ0(L) propagates feature updates from earlier layers and does not depend on the weight update ∆Wl at the current layer, so it does not need to be maximized from Principle (P2). Therefore, it suffices to verify that ϵ0(L) remains O(1) under the initial condition. In fact, the same argument used for deriving ∥hL(x)∥R directly implies

   = O(1),

  

L

αl2∥Wl∥2R∥∆hl−1(x)∥2R

∥ϵ0(L)∥R = Θ

l=1

where we use the self-consistent fact that ∥∆hl−1(x)∥R = Θ(1) for l ∈ [L] if we finally set ∥∆hL(x)∥R = Θ(1). We note that if we further set αl∥Wl∥R = Θ(1/

√

L) for all l ∈ [L] in the initial condition, then ∥ϵ0(L)∥R = Θ(1) and is maximized, which leads to the Depth-µP formulations [6, 46].

First-order terms. The first-order update terms reflect the direct effect of weight updates ∆Wl on features and must be maximized (Θ(1)) to satisfy the µP Principle (P2). Using subadditivity and submultiplicativity, we estimate the order of ∥ϵ1(L)∥R as

 

  + Θ

 

 .

L

L

∥ϵ1(L)∥R = Θ

αl∥∆Wl∥R∥hl−1(x)∥R

αl∥∆Wl∥R∥∆hl−1(x)∥R

l=1

l=1

For l ∈ [L], using ∥hl−1(x)∥R = Θ(1) by the preliminary initial condition and the self-consistent fact that ∥∆hl−1(x)∥R = Θ(1) if we finally set ∥∆hL(x)∥R = Θ(1), we can obtain ∥ϵ1(L)∥R = Θ Ll=1 αl∥∆Wl∥R . To satisfy Principle (P2), we need to maximize the contribution from each ∆Wl and ensure ∥ϵ1(L)∥R = Θ(1) at the same time, which naturally requires

αl∥∆Wl∥R = Θ(1/L), ∀l ∈ [L], which completes the first-order update condition on hidden weights.

Output layer. The output layer has the same form as in the two-layer block case (k = 2), since it depends only on hL(x) and not on the internal structure of the residual blocks. Given ∥hL(x)∥R = Θ(1) and ∥∆hL(x)∥R = Θ(1) from the hidden-layer analysis, the same argument as in Section 3.3 yields

αL+1∥∆WL+1∥R = Θ(1).

#### B.2 Multi-layer Residual Block

This subsection justifies the main-text choice of analyzing the two-layer block (k = 2) as the representative case for all fixed-depth residual branches with k ≥ 2.

###### B.2.1 Problem Setup

We now extend the spectral analysis from one- and two-layer residual blocks to the general case of k-layer residual blocks, where k ≥ 2 is a fixed Θ(1) constant. The fixed-k assumption is important: we do not address regimes where the internal block depth itself scales with width or network depth. Specifically, we consider a residual network of depth L whose forward propagation is given by

h0(x) = α0W0x,

k

hl(x) = hl−1(x) + αlWl(k)Wl(k−1) ···Wl(1)hl−1(x) = hl−1(x) + αl

Wl(i)hl−1(x), ∀l ∈ [L], hL+1(x) = αL+1WL+1hL(x).

i=1

Here, each residual block consists of a depth-k linear transformation, with {Wl(i)}ki=1 denoting the weight matrices within the l-th block. As in the previous sections, hL+1(x) denotes the network output used to compute the loss. As in the two-layer block case, our goal is to derive a spectral condition for realizing µP Principle 2.1 in this setting.

In the following, we show that although increasing the internal block depth k introduces higher-order interactions between weight updates, the resulting spectral conditions admit a simple and systematic form, and do not fundamentally alter the algorithmic implementation of µP.

###### B.2.2 Spectral Scaling Condition

We now state the spectral scaling condition for the above residual network with k-layer residual blocks that is sufficient for the µP principle under joint width–depth scaling.

- Condition B.2 (Spectral condition for µP under joint width-depth scaling, k-layer residual block). To realize µP Principle 2.1, the initial weights and their per-step updates should satisfy:

###### • Initial condition.

- – Input and output weights: α0∥W0∥R, αL+1∥WL+1∥R = Θ(1).
- – Hidden weights: αl ki=1 ∥Wl(i)∥R = Θ(1/L), ∀l ∈ [L].

###### • Update condition.

- – Input and output weights: α0∥∆W0∥R, αL+1∥∆WL+1∥R = Θ(1).
- – Hidden weights (first-order): αl∥∆Wl(i)∥R m̸=i ∥Wl(m)∥R = Θ(1/L), ∀l ∈ [L],i ∈ [k].
- – Hidden weights (j-order, j ≥ 2), automatically satisfied by combining the initial condition and the

first-order update condition: αl i∈S ∥∆Wl(i)∥R i/∈S ∥Wl(i)∥R = Θ(1/L), ∀S ⊆ [k], |S| = j, j ∈ [k], l ∈ [L].

- Condition B.2 reveals that extending residual blocks from two layers to a general fixed depth k ≥ 2 does not change the algorithmic realization of the µP principle. Compared to the two-layer case, the new elements introduced by a deeper block are higher-order interaction terms among weight updates within the same block. However, we show these higher-order terms do not impose additional constraints beyond those already enforced by the initial condition and the first-order update condition.

Concretely, once the product of spectral norms at initialization satisfies αl ki=1 ∥Wl(i)∥R = Θ(1/L) and each update obeys the first-order scaling αl∥∆Wl(i)∥R m̸=i ∥Wl(m)∥R = Θ(1/L), all higher-order update contributions of order j ≥ 2 are automatically controlled as Θ(1/L). As a result, increasing the internal block depth k only increases the number of such higher-order contributions, but does not alter their scaling behavior.

Following the same steps as derivations for implementations in Section 4 and Appendix C, we can find that implementing µP for a k-layer residual block requires no additional parameterization beyond those already needed for the two-layer case. In particular, when the initialization variance is aligned with the standard width-scaling µP formulation [42, 44] as in Section 4 (∥Wl∥R = Θ(1), ∀l ∈ [L]), the initial condition still induces the residual multiplier αl = Θ(1/L) for l ∈ [L], which is the same as the two-layer case. Built upon the initial condition, the first-order update condition is reduced to

αl∥∆Wl(i)∥R

∥Wl(m)∥R = Θ

m̸=i

1 L∥∆Wl(i)∥R .

Therefore, requiring the first-order update condition yields ∥∆Wl(i)∥R = Θ(1) for ∀l ∈ [L],i ∈ [k]. This is also in the same way as the two-layer case (∥∆Wl(i)∥R = Θ(1) for ∀l ∈ [L],i ∈ [2]), thus leading to the same optimizer-related HPs adjustment. The multi-layer analysis, therefore, serves to justify the robustness and generality of the two-layer µP prescription, rather than to introduce a distinct algorithm dependent on block depth.

###### B.2.3 Derivation for Preliminary Initial Condition

We first derive a preliminary initialization condition that guarantees stability of feature magnitudes during forward propagation for k-layer (k ≥ 2) residual blocks. As in the two-layer case, we analyze each layer sequentially.

Input layer. By the submultiplicativity of the RMS operator norm, we have

∥h0(x)∥R = α0∥W0x∥R = Θ(α0∥W0∥R ∥x∥R) = Θ(α0∥W0∥R), where we have assumed ∥x∥R = Θ(1). Thus, choosing α0∥W0∥R = Θ(1) ensures ∥h0(x)∥R = Θ(1). Hidden layers. Expanding the residual recursion yields

hs(x) = hs−1(x) + αs

k

s

Ws(i)hs−1(x) = ··· = h0(x) +

i=1

l=1

k

Wl(i)hl−1(x). (12)

αl

i=1

Applying subadditivity, we can estimate their order as

 ∥h0(x)∥R +

 .

s

k

Wl(i)hl−1(x)

∥hs(x)∥R = Θ

αl

R

i=1

l=1

Since we have ∥h0(x)∥R = Θ(1), it suffices to ensure that ∥ sl=1 αl ki=1 Wl(i)hl−1(x)∥R = O(1) for any s ∈ [L] to preserve ∥hs(x)∥R = Θ(1). Under i.i.d. zero-mean Gaussian initialization, the summands are

approximately independent zero-mean random vectors [10, 42, 46], so the typical squared RMS norm of their sum scales as the sum of the squared RMS norms (see Theorem 3.3.1 in Vershynin [37]), yielding that

  

  .

s

k

s

k

Wl(i)hl−1(x)

Wl(i)hl−1(x)∥2R

∥αl

= Θ

αl

i=1

i=1

l=1

l=1

R

By submultiplicativity, we can further estimate ∥αl ki=1 Wl(i)hl−1(x)∥R = Θ(αl ki=1 ∥Wl(i)∥R∥hl−1(x)∥R). Therefore, starting from ∥h0(x)∥R = Θ(1), imposing

k

√

∥Wl(i)∥R = O(1/

L), l ∈ [L]

αl

i=1

recursively ensures ∥ sl=1 αl ki=1 Wl(i)hl−1(x)∥R = O(1) for any s ∈ [L]. This provides a preliminary initial condition on the hidden weights, which will be further refined once update constraints are incorporated.

Output layer. The output layer has the same form as in the two-layer case (k = 2), yielding αL+1∥WL+1∥R = Θ(1), which keeps the output stable. This completes the preliminary initialization analysis.

###### B.2.4 Derivation for Update Condition

We next derive the update conditions required to ensure stable feature evolution by Principle (P1), i.e., ∥∆hl(x)∥R = Θ(1), while maximally updating parameters as prescribed by Principle (P2).

Input layer. Since ∆h0(x) = α0∆W0x, the submultiplicativity yields

∥∆h0(x)∥R = Θ(α0∥∆W0∥R∥x∥R) = Θ(α0∥∆W0∥R), and thus we set α0∥∆W0∥R = Θ(1). Hidden layers. Expanding the residual recursion in Equation (12) after one update step gives

∆hs(x) = ∆h0(x) +

s

k

Wl(i)∆hl−1(x)

αl

+

i=1

l=1

ϵ0(s)

k

ϵj(s),

j=1

where ϵj(s) collects all terms that are j-th order in {∆Wl(i)}ki=1. By the subadditivity of vector norms, we have

 ∥∆h0(x)∥R + ∥ϵ0(s)∥R +

 .

k

∥∆hs(x)∥R = Θ

∥ϵj(s)∥R

j=1

Since ∥∆h0(x)∥R = Θ(1) by the input-layer update, we have ∥∆hs(x)∥R = Ω(1) for all s ∈ [L]. Moreover, by subadditivity, the remaining terms do not decay with depth, implying ∥∆hs(x)∥R = O(∥∆hL(x)∥R) for any s ∈ [L]. Therefore, to enforce Principle 2.1, it suffices to require ∥∆hL(x)∥R = Θ(1) while satisfying Principle (P2).

Zero-order term. The term ϵ0(L) propagates feature updates from earlier layers and does not depend on the weight update ∆Wl at the current layer, so it does not need to be maximized from Principle (P2). Therefore, it suffices to verify that ϵ0(L) remains O(1) under the preliminary initial condition. In fact, the same argument used for deriving ∥hL(x)∥R directly implies

   = O(1),

  

L

k

∥Wl(i)∥2R∥∆hl−1(x)∥2R

αl2

∥ϵ0(L)∥R = Θ

i=1

l=1

where we use the self-consistent fact that ∥∆hl−1(x)∥R = Θ(1) for l ∈ [L] if we finally enforce ∥∆hL(x)∥R = Θ(1).

First-order terms. The first-order contributions take the form

L

k

Wl(k) ···∆Wl(i) ···Wl(1) (hl−1(x) + ∆hl−1(x)).

ϵ1(L) =

αl

i=1

l=1

Using subadditivity and submultiplicativity,

 

  =

 

 ,

L

k

k

L

∥∆Wl(i)∥R

∥Wl(m)∥R

αl∥∆Wl(i)∥R

∥Wl(m)∥R

∥ϵ1(L)∥R = Θ

αl

Θ

i=1

i=1

m̸=i

m̸=i

l=1

l=1

where we used ∥hl−1(x)∥R = Θ(1) for l ∈ [L] by the preliminary initial condition and the self-consistent fact that ∥∆hl−1(x)∥R = Θ(1) for l ∈ [L] if we finally enforce ∥∆hL(x)∥R = Θ(1). To satisfy Principle (P2), we need to maximize the contribution from each ∆Wl and ensure ∥ϵ1(L)∥R = Θ(1) at the same time, which naturally requires

αl∥∆Wl(i)∥R

∥Wl(m)∥R = Θ(1/L), ∀l ∈ [L], i ∈ [k].

m̸=i

Any j-order terms. Similar to the first-order term, for j ∈ [k], the j-th order feature update term ϵj(L) admits the explicit form

 

 

 

 (hl−1(x) + ∆hl−1(x)),

L

Wl(i)

∆Wl(i)

αl

ϵj(L) =

i∈S

l=1

i/∈S

S⊆[k] |S|=j

where S indexes the subset of sublayers whose weights are replaced by their per-step updates, and the products are ordered consistently with the forward computation within each residual block. Therefore, by the same subadditivity and submultiplicativity arguments for ∥ϵ1(L)∥R, the j-th order update terms satisfy

 

 .

L

∥∆Wl(i)∥R

∥Wl(i)∥R

∥ϵj(L)∥R =

Θ

αl

i∈S

l=1

S⊆[k] |S|=j

i/∈S

Principle (P2) requires maximizing each summand while ensuring ∥ϵj(L)∥R = Θ(1). It is therefore sufficient to impose

∥∆Wl(i)∥R

αl

i∈S

∥Wl(i)∥R = Θ(1/L), ∀S ⊆ [k], |S| = j, j ∈ [k], l ∈ [L].

i/∈S

Output layer. The same argument as in the two-layer case in Section 3 yields αL+1∥∆WL+1∥R = Θ(1).

- B.2.5 Derivation for Final Initial Condition Multiplying the first-order update conditions for each hidden weight yields

k

∥Wl(i)∥kR−1

αlk

i=1

k

∥∆Wl(i)∥R = Θ(1/Lk), ∀l ∈ [L].

i=1

On the other hand, the highest k-order update condition is αl ki=1 ∥∆Wl(i)∥R = Θ(1/L) for all l ∈ [L]. Combining the two relations immediately gives

k

∥Wl(i)∥R = Θ(1/L), ∀l ∈ [L],

αl

i=1

which refines the preliminary initialization condition.

Finally, as in the two-layer case, we prove that the refined initial condition and the first-order update condition can derive any j-order (j ≥ 2) update condition on hidden weights. Thus, retaining the refined initial condition and the first-order update condition in Condition B.2 is sufficient.

Formally, for any S ⊆ [k], |S| = j, j ∈ [k], l ∈ [L], we need to prove that αl

∥∆Wl(i)∥R

∥Wl(i)∥R = Θ(1/L)

i∈S

i/∈S

based on the refined initial condition and the first-order update condition. By multiplying the first-order update conditions, we have

 αl∥∆Wl(i)∥R

 

1 Lj

∥Wl(m)∥R

=

i∈S

m̸=i

 

 

 

 

∥∆Wl(i)∥R

∥Wl(m)∥R

= αlj

i∈S

i∈S m̸=i

 

 

 

 

 

 

j−1

k

∥∆Wl(i)∥R

∥Wl(i)∥R

∥Wl(i)∥R

= αlj

i∈S

i=1

i/∈S

 αl

 

 αl

 

j−1

k

∥∆Wl(i)∥R

∥Wl(i)∥R

∥Wl(i)∥R

=

i∈S

i=1

i/∈S

 αl

  ·

1 Lj−1,

∥∆Wl(i)∥R

∥Wl(i)∥R

=

i∈S

i/∈S

which implies that αl i∈S ∥∆Wl(i)∥R i/∈S ∥Wl(i)∥R = Θ(1/L), which finishes the derivation. Therefore, for any fixed k ≥ 2, the spectral constraints reduce to the same implementation as in the k = 2 case: hidden

residual multiplier αl = Θ(1/L) and hidden update norms ∥∆Wl(i)∥R = Θ(1) under standard width-scaling µP initialization.

#### B.3 Bias Parameters

The purpose of this subsection is not to introduce a new depth-scaling rule, but to show that bias parameters can be assigned order-one initialization and update scales once the matrix-weight parameterization from Condition 3.1 is in place.

###### B.3.1 Problem Setup

As shown in Appendix B.2, residual blocks with an arbitrary fixed internal depth k ≥ 2 admit spectral scaling conditions that are algorithmically equivalent to the k = 2 case. Therefore, to illustrate that bias parameters do not change the main µP prescription, we analyze the representative two-layer residual block with additive biases. Specifically, we consider a residual network whose forward propagation is given by

h0(x) = α0 W0x + b0 , hl(x) = hl−1(x) + αl Wl(2) Wl(1)hl−1(x) + b(1)l + b(2)l , ∀l ∈ [L],

hL+1(x) = αL+1WL+1hL(x).

Here, each residual block consists of a two-layer linear transformation with additive biases, where Wl(1),Wl(2) denote the weight matrices and b(1)l ,b(2)l denote the corresponding bias vectors within the l-th block. The

scalars {αl}Ll=0+1 represent block multipliers that control the effective strength of each transformation. As in the previous sections, hL+1(x) denotes the network output used to compute the loss. Our goal is to derive a spectral condition that realizes µP Principle 2.1 in this setting.

###### B.3.2 Spectral Scaling Condition

We now state the spectral scaling condition for the residual network with biases for realizing the µP principle under joint width–depth scaling.

- Condition B.3 (Spectral condition for µP under joint width-depth scaling, two-layer residual block with biases). To realize µP Principle 2.1, the initial parameters and their per-step updates should satisfy:

###### • Initial condition.

- – Input parameters: α0∥W0∥R = Θ(1), α0∥b0∥R = Θ(1).
- – Hidden parameters: ∗ αl∥Wl(2)∥R ∥Wl(1)∥R = Θ(1/L), ∀l ∈ [L]. ∗ αl∥Wl(2)∥R∥b(1)l ∥R = Θ(1/L), ∀l ∈ [L].

∗ αl∥b(2)l ∥R = O(1/

√

L), ∀l ∈ [L].

- – Output parameters: αL+1∥WL+1∥R = Θ(1).

###### • Update condition.

- – Input parameters: α0∥∆W0∥R = Θ(1), α0∥∆b0∥R = Θ(1).
- – Hidden parameters (first-order): ∗ αl∥∆Wl(2)∥R ∥Wl(1)∥R = Θ(1/L), ∀l ∈ [L]. ∗ αl∥Wl(2)∥R ∥∆Wl(1)∥R = Θ(1/L), ∀l ∈ [L]. ∗ αl∥∆Wl(2)∥R∥b(1)l ∥R = Θ(1/L), ∀l ∈ [L]. ∗ αl∥Wl(2)∥R∥∆b(1)l ∥R = Θ(1/L), ∀l ∈ [L]. ∗ αl∥∆b(2)l ∥R = Θ(1/L), ∀l ∈ [L].
- – Hidden parameters (second-order), automatically satisfied given initial condition and first-order update conditions:

∗ αl∥∆Wl(2)∥R ∥∆Wl(1)∥R = Θ(1/L), ∀l ∈ [L]. ∗ αl∥∆Wl(2)∥R∥∆b(1)l ∥R = Θ(1/L), ∀l ∈ [L].

- – Output parameters: αL+1∥∆WL+1∥R = Θ(1).

###### • Efficient implementation. Under the µP parameterization (k = 2) of block multipliers {αl} and matrix weights {Wl} described in Section 4 and Appendix C, all bias-related spectral conditions can be satisfied simultaneously by initializing and training with biases of order Θ(1). Concretely, it is sufficient to enforce

∥bl∥R = Θ(1), ∥∆bl∥R = Θ(1), ∀0 ≤ l ≤ L. (13) The initial condition ∥bl∥R = Θ(1) can be satisfied by setting σb

= Θ(1), and the implementation for update condition ∥∆bl∥R = Θ(1) will be derived in Appendix C.

l

- Condition B.3 shows that, under joint width-depth scaling, bias parameters can be incorporated without modifying the existing HP parameterization of the weight matrices. Specifically, once the block multipliers

{αl} and weights {Wl} are implemented as in Section 4, biases admit additional, simple order-one spectral conditions that guarantee their initialization and updates scale properly. Thus, biases can be handled by lightweight extensions of our framework, while the µP formulation for bias-free residual blocks remains unchanged.

###### B.3.3 Derivation for Preliminary Initialization Condition Input layer. By subadditivity and submultiplicativity,

∥h0(x)∥R = Θ(α0(∥W0∥R∥x∥R + ∥b0∥R)) = Θ α0∥W0∥R + Θ α0∥b0∥R , where we assumed ∥x∥R = Θ(1). Thus, choosing α0∥W0∥R,α0∥b0∥R = Θ(1) ensures ∥h0(x)∥R = Θ(1). Hidden layers. Expanding the residual recursion yields

hs(x) = h0(x) +

s

αl Wl(2)Wl(1)hl−1(x) + Wl(2)b(1)l + b(2)l . (14)

l=1

Applying subadditivity, we can estimate their order as

 ∥h0(x)∥R +

 .

s

s

s

αlb(2)l

αlWl(2)b(1)l +

αlWl(2)Wl(1)hl−1(x) +

∥hs(x)∥R = Θ

R

l=1

l=1

l=1

Since we have ∥h0(x)∥R = Θ(1), it suffices to ensure other terms are O(1) for any s ∈ [L] to preserve ∥hs(x)∥R = Θ(1). Under i.i.d. zero-mean Gaussian initialization, the summands are approximately independent zero-mean random vectors [10, 42, 46], so the typical squared RMS norm of their sum scales as the sum of the squared RMS norms (see Theorem 3.3.1 in Vershynin [37]). Therefore, we can obtain

s

s

s

αlWl(2)Wl(1)hl−1(x) +

αlWl(2)b(1)l +

αlb(2)l

R

l=1

l=1

l=1

s

s

s

2 R

2 R

2 R

αlb(2)l

αlWl(2)b(1)l

αlWl(2)Wl(1)hl−1(x)

+

=

+

l=1

l=1

l=1

Furthermore, using the same probability argument and submultiplicativity inequality as in the derivation without biases (e.g., see Section 3), we have

 

 ,

s

s

2 R

αlWl(2)Wl(1)hl−1(x)

αl2∥Wl(2)∥2R∥Wl(1)∥2R∥hl−1(x)∥2R

= Θ

l=1

l=1

 

 ,

s

s

2 R

αlWl(2)b(1)l

αl2∥Wl(2)∥2R∥b(1)l ∥2R

= Θ

l=1

l=1

 

 .

s

s

2 R

αlb(2)l

αl2∥b(2)l ∥2R

= Θ

l=1

l=1

Therefore, starting from ∥h0(x)∥R = Θ(1), imposing αl∥Wl(2)∥R∥Wl(1)∥R = O(1/

√

√

√

L), αl∥Wl(2)∥R∥b(1)l ∥R = O(1/

L), αl∥b(2)l ∥R = O(1/

L)

recursively ensures ∥hs(x)∥R = Θ(1) for s ∈ [L]. This yields a preliminary initialization condition, which will be refined after incorporating update constraints.

Output layer. The same argument as in the two-layer residual block case yields αL+1∥WL+1∥R = Θ(1).

###### B.3.4 Derivation for Update Condition

Input layer. Recall that

h0(x) = α0 W0x + b0 . After one gradient step, the feature update satisfies

∆h0(x) = α0 ∆W0 x + ∆b0 . By subadditivity, submultiplicativity, and using ∥x∥R = Θ(1) by data assumption, we obtain

∥∆h0(x)∥R = Θ(α0∥∆W0∥R + α0∥∆b0∥R). Therefore, we choose

α0∥∆W0∥R = Θ(1), α0∥∆b0∥R = Θ(1) to realize ∥∆h0(x)∥R = Θ(1).

Hidden layers. We next analyze the feature updates ∆hs(x) after one gradient step. Expanding Equation (14) yields

∆hs(x) = ∆h0(x) + ∆

= ∆h0(x) + ∆

s

αl Wl(2)Wl(1)hl−1(x) + Wl(2)b(1)l + b(2)l

l=1

s

s

αlWl(2)Wl(1)hl−1(x) + ∆

αlWl(2)b(1)l + ∆

l=1

l=1

s

αlb(2)l .

l=1

By the subadditivity of vector norms, we have

 .

 ∥∆h0(x)∥R + ∆

s

s

s

αlb(2)l

αlWl(2)b(1)l

αlWl(2)Wl(1)hl−1(x)

∥∆hs(x)∥R = Θ

+ ∆

+ ∆

R

R

R

l=1

l=1

l=1

Since ∥∆h0(x)∥R = Θ(1) by the input-layer update, we have ∥∆hs(x)∥R = Ω(1) for all s ∈ [L]. Moreover, by subadditivity, the remaining terms do not decay with depth, implying ∥∆hs(x)∥R = O(∥∆hL(x)∥R) for any s ∈ [L]. Therefore, to enforce Principle 2.1, it suffices to require ∥∆hL(x)∥R = Θ(1) while satisfying Principle (P2). We discuss the components of ∆hL(x) in sequence.

Matrix-weight terms. The contributions from

∆h0(x) + ∆

L

αlWl(2)Wl(1)hl−1(x)

l=1

have been fully analyzed in the bias-free two-layer case (see Section 3). Applying the same reasoning yields the first- and second-order update conditions on hidden matrix weights:

αl∥∆Wl(2)∥R ∥Wl(1)∥R = Θ(1/L), αl∥Wl(2)∥R ∥∆Wl(1)∥R = Θ(1/L),

αl∥∆Wl(2)∥R ∥∆Wl(1)∥R = Θ(1/L), ∀l ∈ [L]. We then need to control the newly introduced bias-related terms.

First-layer bias-related term. Consider ∆ Ll=1 αlWl(2)b(1)l . Expanding the update yields

L

αlWl(2)b(1)l =

∆

l=1

L

αl ∆Wl(2)b(1)l + Wl(2)∆b(1)l + ∆Wl(2)∆b(1)l .

l=1

By subadditivity and submultiplicativity of the RMS norm, we have

L

αlWl(2)b(1)l

∆

l=1

= Θ

R

L

αl∥∆Wl(2)∥R∥b(1)l ∥R +

l=1

L

αl∥Wl(2)∥R∥∆b(1)l ∥R +

l=1

L

αl∥∆Wl(2)∥R∥∆b(1)l ∥R .

l=1

According to Principle (P2), we require ∆ Ll=1 αlWl(2)b(1)l

= Θ(1) and maximize the contribution from each summand, leading to

R

αl∥∆Wl(2)∥R∥b(1)l ∥R = Θ(1/L), αl∥Wl(2)∥R∥∆b(1)l ∥R = Θ(1/L),

αl∥∆Wl(2)∥R∥∆b(1)l ∥R = Θ(1/L), ∀l ∈ [L].

Second-layer bias-related term. Finally, for ∆ Ll=1 αlb(2)l , we have

L

αlb(2)l

∆

l=1

=

R

L

αl∆b(2)l

l=1

= Θ

R

L

αl∥∆b(2)l ∥R .

l=1

To maximally update parameters according to Principle (P2), we require this term to remain Θ(1) and maximize each summand, which yields

αl∥∆b(2)l ∥R = Θ(1/L), ∀l ∈ [L].

Output layer. The same argument as in the two-layer case in Section 3 yields αL+1∥∆WL+1∥R = Θ(1).

###### B.3.5 Derivation for Final Initial Condition

We now derive the final initialization conditions by incorporating the update constraints obtained in the previous subsection.

Hidden matrix weights. As already shown in the bias-free setting (Section 3), combining the first-order and second-order update conditions immediately yields the initialization constraint

αl∥Wl(1)∥R ∥Wl(2)∥R = Θ(1/L), ∀l ∈ [L]. Therefore, the presence of biases does not alter the initialization scaling of hidden matrix weights.

Bias parameters. We now derive the initialization conditions for bias terms by combining the first- and second-order update constraints. For the first-layer bias b(1)l , the first-order update conditions give

αl∥∆Wl(2)∥R∥b(1)l ∥R = Θ(1/L), αl∥Wl(2)∥R∥∆b(1)l ∥R = Θ(1/L),

while the second-order update condition yields

αl∥∆Wl(2)∥R∥∆b(1)l ∥R = Θ(1/L).

Multiplying the two first-order conditions and dividing by the second-order one, we obtain

αl∥Wl(2)∥R∥b(1)l ∥R = Θ(1/L), ∀l ∈ [L].

Similar to the hidden matrix weights, the second-order bias-related condition is automatically satisfied by combining the refined initial condition and the corresponding first-order update condition.

We note that the second-layer bias b(2)l has no multiplicative interaction with another parameter in the forward block, so its initialization condition remains the preliminary upper bound αl∥b(2)l ∥R = O(1/

√

L).

###### B.3.6 Derivation for Efficient Implementation

Recall that, based on the µP parameterization (k = 2) introduced for matrix weights in Section 4 and Appendix C, we have α0 = Θ(1) in Equation (7), αl = Θ(1/L) for l ∈ [L] in Equation (8), ∥Wl∥R = Θ(1) for 0 ≤ l ≤ L in Equation (6) and ∥∆Wl∥R = Θ(1) for 0 ≤ l ≤ L. Based on these conditions, Condition B.3 reduces to

###### • Initial condition.

- – Input parameters: ∥b0∥R = Θ(1).
- – Hidden parameters:

- ∗ ∥b(1)l ∥R = Θ(1), ∀l ∈ [L].
- ∗ ∥b(2)l ∥R = O(

√

L), ∀l ∈ [L].

###### • Update condition.

- – Input parameters: ∥∆b0∥R = Θ(1).
- – Hidden parameters (first-order): ∗ ∥∆b(1)l ∥R = Θ(1), ∀l ∈ [L].

∗ ∥∆b(2)l ∥R = Θ(1), ∀l ∈ [L]. Therefore, it is sufficient to enforce the order-one spectral condition for biases:

∥bl∥R = Θ(1), ∥∆bl∥R = Θ(1), ∀0 ≤ l ≤ L.

√

Although the preliminary condition permits ∥b(2)l ∥R as large as O(

L), choosing it to be Θ(1) is a simpler sufficient choice and keeps all biases on the same scale.

### Appendix C Implementing Condition 3.1 for Optimizers with Weight Decay

Recall that in Section 4.1 of the main text, we implemented Condition 3.1 for initialization and specified the parameterization of the block multipliers αl and the initialization variances σl, which is optimizer-agnostic. In Section 4.2, we further implemented Condition 3.1 for updates and derived the parameterization of the learning rates ηl for the Muon-Kimi [26]. We now extend this update-condition analysis to a broader class of optimizers with weight decay. For bias parameters, we also derive the corresponding bias HPs when the optimizer is commonly applied to biases.

#### C.1 Preparation

Unified update form with weight decay. To provide a unified derivation across different optimizers, we begin by expressing their one-step update rules in a common form. When weight decay is included, a single update step of the weight matrix can be written as

∆Wl = −ηl Al + λlWl ,

where Al denotes the optimizer-specific update direction before applying the learning rate and excluding weight decay. For example, Al = ∇WlL for SGD, while for Muon-Kimi Al = 0.2 max{nin,nout}UlVl⊤. The scalar λl is the weight-decay coefficient.

The update magnitude ∥∆Wl∥R = ηl∥Al+λlWl∥R is required to satisfy the update conditions in Condition 3.1. We analyze this requirement under two complementary regimes.

Without weight decay. When weight decay is disabled (λl = 0), the update reduces to ∥∆Wl∥R = ηl∥Al∥R. In this case, the learning rate is chosen so that

∥∆Wl∥R = ηl∥Al∥R satisfies Condition 3.1. (∆1)

With weight decay. When weight decay is enabled (λl ̸= 0), we choose the weight decay term to be comparable in scale to the optimizer-driven term.

∥λlWl∥R = Θ ∥Al∥R . (∆2) This is a non-degenerate implementation convention: it keeps weight decay active in the update dynamics without letting it dominate the optimizer-driven term. Under the usual scale estimate, ∥Al + λlWl∥R = Θ(∥Al∥R), so the learning-rate scaling rule derived from Equation (∆1) is preserved.

Weights and biases. In the following, we derive parameterizations of the learning rate and weight decay coefficient for a range of optimizers. Since matrix-based optimizers such as Muon and Shampoo are typically not used for bias parameters, we restrict our analysis of bias parameterization to vector-based optimizers such as SGD and AdamW. The same two rules, Equations (∆1) and (∆2), are then applied to the biases with corresponding spectral condition in Equation 13 on vector RMS norms.

Momentum. We note that momentum is typically omitted in standard µP analyses [42–44] (e.g., by setting

- β1 = β2 = 0 in AdamW), while in practical µP implementations the momentum coefficients are taken to be Θ(1). The main rationale is that the norm of the momentum term and the norm of the current update are expected to be of the same order, and since the spectral condition aims to control the update scale, omitting the momentum is regarded as an acceptable simplification. Moreover, analyzing the update without momentum can be interpreted as studying the first update step after initialization, which has been empirically observed to be reliable for understanding neural network training [4, 5, 29, 45]. In the subsequent derivations, we adopt this simplification as well.

Low-rank structure of updates. Following the common practice used in the µP spectral analysis [29, 45], we also introduce a useful preliminary result for controlling the norm of the update term Al. The key idea is to exploit the effective low-rank structure of neural-network updates, which has been widely observed in neural network training [2, 45, 47]: only a small number of dominant singular directions carry most of the update.

Assumption C.1 (Low-rank update structure). For vector-based optimizers such as SGD and AdamW, we assume that the effective update term Al has constant rank with respect to width and depth, i.e., r(Al) = Θ(1). Under this assumption, the spectral norm and Frobenius norm of the update term are of the same order:

∥Al∥2 = Θ(∥Al∥F), (15) where the hidden constants are independent of width and depth. Indeed, this is because

∥Al∥2 ≤ ∥Al∥F ≤ r(Al)∥Al∥2 = Θ(∥Al∥2). We will use this low-rank update property to estimate the scale of ∥Al∥R for vector-based optimizers.

#### C.2 Overview

Table 2 summarizes the optimizer families covered in this section and points to the corresponding detailed µP parameterization tables. All rows use the optimizer-agnostic initialization and block-multiplier rules from Section 4.1; the table only organizes the optimizer-dependent update rules derived below.

- Table 2 Overview of µP implementation from Condition 3.1 (k = 2) for optimizer families with weight decay. Optimizers in the same row share the same µP scaling rules.

Optimizer family Detailed parameterization

Muon-Kimi Table 3 Muon / Shampoo / SOAP Table 4

SGD Table 5 AdamW / Lion / Sophia Table 6

SSO Table 7

- Table 3 µP implementation of Condition 3.1 (k = 2) for Muon-Kimi [26] with weight decay under width-depth scaling. Entries in purple indicate differences between µP and SP, while gray shows the corresponding SP choices. Here, rn and rL denote the width and depth scaling ratios relative to the base model. The variance of input weights is σbase2 for language and σbase2 /d0 for image.

Input weights Hidden weights Output weights Block Multiplier αbase αbase/rL (αbase) αbase/rn (αbase)

Initial Variance σbase2 /d0 or σbase2 σbase2 /rn (σbase2 ) σbase2 Learning Rate ηbase ηbase/√rn (ηbase) ηbase Weight Decay λbase λbase√rn (λbase) λbase

#### C.3 Muon-Kimi

For a weight matrix Wl ∈ Rn

out×nin, the update rule of Muon-Kimi [26] with weight decay is ∆Wl = −ηl 0.2 max{nin,nout}UlVl⊤

+λlWl ,

Al

where Ul,Vl arise from the compact SVD of the gradient ∇WlL = UlΣlVl⊤. Recalling that in Section 4.2 in the main text, we have derived the learning rate parameterizations to achieve (∆1) that

η0 = Θ(1), ηl(1) = Θ

1 √nin

, ηl(2) = Θ

1 √nin

, ηL+1 = Θ(1).

According to the update norm in Equation (9), we have

 

 √nin max 1,

  =

Θ(1), l = 0, Θ(√nin), l ∈ [L], Θ(nin), l = L + 1.

nin nout

∥Al∥R = Θ



Given the magnitude of ∥Wl∥R in Equation (6), as desired ∥λlWl∥R = Θ ∥Al∥R by (∆2), the parameterizations of λl need to be set as follows:

 

Θ(1), l = 0, Θ(√nin), l ∈ [L], Θ(1), l = L + 1.

λl =



This completes the implementation of the update condition for Muon-Kimi with weight decay, as summarized

- in Table 3.

- Table 4 µP implementation of Condition 3.1 (k = 2) for Muon [21], Shampoo [12], and SOAP [38] with weight decay under width-depth scaling. Entries in purple indicate differences between µP and SP, while gray shows the corresponding SP choices. Here, rn and rL denote the width and depth scaling ratios relative to the base model. The variance of input weights is σbase2 for language and σbase2 /d0 for image.

Input weights Hidden weights Output weights Block Multiplier αbase αbase/rL (αbase) αbase/rn (αbase)

Initial Variance σbase2 /d0 or σbase2 σbase2 /rn (σbase2 ) σbase2 Learning Rate ηbase√rn (ηbase) ηbase ηbase√rn (ηbase) Weight Decay λbase/√rn (λbase) λbase λbase/√rn (λbase) Shampoo ε εbase/rn (εbase) εbase/rL2 (εbase) εbase/rn (εbase)

#### C.4 Muon

In this section, we derive the µP implementation from Condition 3.1 for Muon, which recovers and extends the µP scaling rules studied in Qiu et al. [30].

###### C.4.1 Update Rule

For a weight matrix Wl ∈ Rn

out×nin, the update rule of Muon [21] is ∆Wl = −ηl UlVl⊤

+λlWl , (16)

Al

where Ul,Vl arise from the compact SVD of the gradient ∇WlL = UlΣlVl⊤. Compared with Muon-Kimi [26], the only difference lies in the absence of the 0.2 max{nin,nout} prefactor.

Considering the dimension assumption in Equation (3), the resulting norm of Al satisfies

 

Θ(1/√nout), l = 0, Θ(1), l ∈ [L], Θ(√nin), l = L + 1.

nin nout∥UlVl⊤∥2 =

nin nout

∥Al∥R = ∥UlVl⊤∥R =

(17)

=



###### C.4.2 Derivation of Parameterization

Input and output layers. When weight decay is disabled (λ0 = 0), given the dimension assumption d0,dL+1 = Θ(1),nl = Θ(n) in Equation (3), the multiplier parameterizations α0 = Θ(1),αL+1 = Θ(1/nin) in Equation (7), and the scale of ∥Al∥R in Equation (17), we have

Θ(η0/√nout), l = 0, Θ(ηL+1/√nin), l = L + 1.

αl∥∆Wl∥R = αlηl∥Al∥R =

As desired in (∆1), to satisfy (C2.1) that α0∥∆W0∥R,αL+1∥∆WL+1∥R = Θ(1), we need to set η0 = Θ(√nout), ηL+1 = Θ(√nin). When λl ̸= 0, given Equation (6) that ∥W0∥R = Θ(1) and ∥WL+1∥R = Θ(nin), we have ∥λlWl∥R =

Θ(λ0), l = 0, Θ(λL+1nin), l = L + 1.

To satisfy (∆2) that ∥λlWl∥R = Θ ∥Al∥R , we need to set λ0 = Θ(1/√nout), λL+1 = Θ(1/√nin),

- Table 5 µP implementation of Condition 3.1 (k = 2) for SGD with weight decay under width–depth scaling. Entries in purple indicate differences between µP and SP, while gray shows the corresponding SP choices. Here, rn and rL denote the width and depth scaling ratios relative to the base model. The variance of input weights is σbase2 for language and σbase2 /d0 for image. The initial variance of input bias is σbase2 .

Input weights & biases Hidden weights Output weights Hidden biases Block Multiplier αbase αbase/rL (αbase) αbase/rn (αbase) αbase/rL (αbase)

Initial Variance σbase2 /d0 or σbase2 σbase2 /rn (σbase2 ) σbase2 σbase2 Learning Rate ηbasern (ηbase) ηbaserL (ηbase) ηbasern (ηbase) ηbaserLrn (ηbase) Weight Decay λbase/rn (λbase) λbase/rL (λbase) λbase/rn (λbase) λbase/(rLrn) (λbase)

Hidden layers (first-order). When weight decay is disabled (λl = 0), given the dimension assumption d0,dL+1 = Θ(1),nl = Θ(n) in Equation (3), the weight norm ∥Wl∥R = Θ(1) in Equation (6), the multiplier parameterization αl = Θ(1/L) in Equation (8), and the scale of ∥Al∥R in Equation (17), we have

αl∥∆Wl(2)∥R∥Wl(1)∥R = Θ(1/L) · ηl(2)∥A(2)l ∥R∥Wl(1)∥R = Θ(ηl(2)/L).

As desired in (∆1), to satisfy the first-order update condition on hidden weights (C2.2) that αl∥∆Wl(2)∥R ∥Wl(1)∥R = Θ(1/L), we need to set

ηl(2) = Θ(1). When weight decay is enabled (λl ̸= 0), given the weight norm ∥Wl∥R = Θ(1) in Equation (6), we have

∥λ(2)l Wl(2)∥R = Θ(λ(2)l ). To satisfy (∆2) that ∥λlWl∥R = Θ ∥Al∥R we need to set

λ(2)l = Θ(1).

Symmetrically, we have the same choice for Wl(1): ηl(1) = Θ(1), λ(1)l = Θ(1).

Hidden layers (second-order). As discussed in Section 3.3.3, the second-order update condition is satisfied automatically once the initial condition and the first-order update condition are met. We explain here again

for clarity: Multiplying two equations in (C2.2) gives αl2∥Wl(2)∥R∥Wl(1)∥R ∥∆Wl(2)∥R∥∆Wl(1)∥R = Θ(1/L2). Combining this with (C1.2) that αl∥Wl(2)∥R ∥Wl(1)∥R = Θ(1/L) directly implies the second-order condition (C2.3).

This completes the implementation of the update condition for Muon with weight decay, which is summarized

- in Table 4.

#### C.5 SGD

In this section, we derive the µP implementation from Condition 3.1 for SGD, which recovers and extends the µP scaling rules studied in Bordelon et al. [5].

###### C.5.1 Update Rule

For a weight matrix Wl ∈ Rn

out×nin, the SGD update rule with weight decay can be written as: ∆Wl = −ηl ∇WlL

+λlWl .

Al

Here, we estimate the scale of raw gradient ∇WlL following the spectral argument of Yang et al. [45]. From the derivation of the update condition in Section 3, we can observe that gradient updates ∆W0,∆WL+1 induce a change ∥∆hL+1∥R = Θ(1) in the output, which induces a change ∆L = Θ(1) for common loss functions L. In contrast, each hidden gradient update ∆Wl (l ∈ [L]) induces a change ∥∆hL+1∥R = Θ(1/L) in the output, which induces a change ∆L = Θ(1/L). We use these properties to derive the scale of ∇WlL as follows.

For the input weights, we have Θ(1) = ∆W

0L = Θ(⟨∆W0,∇W0L⟩) = Θ(∥∆W0∥F∥∇W0L∥F) = Θ(∥∆W0∥2∥∇W0L∥2),

where ⟨·,·⟩ denotes the trace inner product, and we use the facts that the two arguments of the inner product are proportional to each other and the low-rank property of Al in Equation (15). Since we finally realize the spectral condition (C2.1) that α0∥∆W0∥R = Θ(1) and use α0 = Θ(1) by initial implementation in Equation (7), we have ∥∆W0∥R = Θ(1) so ∥∆W0∥2 = Θ( nout/nin). Therefore, we obtain ∥∇W0L∥2 = Θ( nin/nout), which leads to

nin nout ∥∇W0L∥2 = Θ(

nin nout

1 nout

∥A0∥R = ∥∇W0L∥R =

) = Θ(

).

Similarly, for the hidden weight Wl, we have Θ(

1 L

lL = Θ(⟨∆Wl,∇WlL⟩) = Θ(∥∆Wl∥F∥∇WlL∥F) = Θ(∥∆Wl∥2∥∇WlL∥2),

) = ∆W

Since we finally set αl∥∆Wl∥R∥Wl∥R = Θ(1/L) to satisfy the update condition (C2.2), and use αl = Θ(1/L) in Equation (8), ∥Wl∥R = Θ(1) in Equation (6) by initial implementation, we have ∥∆Wl∥R = Θ(1) so ∥∆Wl∥2 = Θ( nout/nin). Therefore, we obtain ∥∇WlL∥2 = Θ(L−1 nin/nout), which leads to

nin nout∥∇WlL∥2 = Θ(

nin Lnout

1 L

∥Al∥R = ∥∇WlL∥R =

) = Θ(

).

Finally, for the output weight WL+1, we have Θ(1) = ∆W

L+1L = Θ(⟨∆WL+1,∇WL+1L⟩) = Θ(∥∆WL+1∥F∥∇WL+1L∥F) = Θ(∥∆WL+1∥2∥∇WL+1L∥2).

Since we will set αL+1∥∆WL+1∥R = Θ(1) to realize the update condition (C2.1) and use αL+1 = Θ(1/nin) in initial implementation in Equation (7), we have ∥∆WL+1∥R = Θ(nin) so ∥∆WL+1∥2 = Θ(√noutnin). Therefore, we obtain ∥∇WL+1L∥2 = Θ(1/√ninnout), which leads to

nin nout ∥∇WL+1L∥2 = Θ(

1 nout

∥AL+1∥R = ∥∇WL+1L∥R =

) = Θ(1).

To sum up, we have

∥Al∥R = ∥∇WlL∥R =

###### C.5.2 Derivation of Parameterization

 

Θ(1/nout), l = 0, Θ(1/L), l ∈ [L], Θ(1), l = L + 1.



(18)

Input and output layers. When weight decay is disabled (λ0 = 0), using the dimension assumptions d0,dL+1 = Θ(1) and nl = Θ(n) in Equation (3), together with the initialization parameterization α0 = Θ(1) and αL+1 = Θ(1/nin) in Equation (7), we obtain

αl∥∆Wl∥R = αlηl∥Al∥R =

Θ η0/nout , l = 0, Θ ηL+1/nin , l = L + 1.

To satisfy the input/output update requirement α0∥∆W0∥R, αL+1∥∆WL+1∥R = Θ(1) in Condition (C2.1), we therefore choose

η0 = Θ(nout), ηL+1 = Θ(nin).

When weight decay is active (λl ≠ 0), using ∥W0∥R = Θ(1) and ∥WL+1∥R = Θ(nin) from the initialization implementation in Equation (6), we have

∥λlWl∥R =

Θ(λ0), l = 0, Θ(λL+1nin), l = L + 1.

Matching this to ∥Al∥R as desired by condition (∆2) yields λ0 = Θ(1/nout), λL+1 = Θ(1/nin).

Hidden layers (first-order). For a hidden block we have implemented αl = Θ(1/L) in Equation (8) and ∥Wl(i)∥R = Θ(1) in Equation (6). When λl = 0 we obtain

αl∥∆Wl(2)∥R ∥Wl(1)∥R = Θ(1/L) · ηl(2)∥A(2)l ∥R = Θ(ηl(2)/L2). Enforcing the first-order hidden update condition (C2.2) that αl∥∆Wl(2)∥R ∥Wl(1)∥R = Θ(1/L) gives

ηl(2) = Θ(L). By the same reasoning, the same choice applies to the other learning rate, ηl(1) = Θ(L). If weight decay is enabled on hidden matrices, using ∥Wl(i)∥R = Θ(1) by Equation (6), we obtain ∥λ(li)Wl(i)∥R = Θ(λ(li)), so condition (∆2) that ∥λlWl∥R = Θ ∥Al∥R = Θ(1/L) implies the natural choice

λ(li) = Θ(1/L), i = 1,2.

Hidden layers (second-order). As illustrated in Section 3.3.3 and Appendix C.4, the second-order update condition is satisfied automatically once the initial condition and the first-order update condition are met.

Biases. For biases, we use the spectral bias condition in Equation (13), which sets ∥∆bl∥R = Θ(1) for the input and hidden biases under Condition 3.1. We estimate the corresponding raw-gradient scales in the same way as for matrix weights. For the input biases b0 ∈ Rn

out×1, we have Θ(1) = ∆b

0L = Θ(⟨∆b0,∇b0L⟩) = Θ(∥∆b0∥2∥∇b0L∥2),

Since we finally set ∥∆b0∥R = Θ(1) to satisfy the update condition in Equation (13), we have ∥∆b0∥2 = Θ(√nout). Therefore, we obtain ∥∇b0L∥2 = Θ(1/√nout), which leads to

1 nout

1 nout∥∇b0L∥2 = Θ(

∥∇b0L∥R =

).

Similarly, for the hidden biases bl ∈ Rn

out×1, we have Θ(1/L) = ∆b

lL = Θ(⟨∆bl,∇blL⟩) = Θ(∥∆bl∥2∥∇blL∥2),

Since we finally set ∥∆bl∥R = Θ(1) to satisfy the update condition in Equation (13), we have ∥∆bl∥2 = Θ(√nout). Therefore, we obtain ∥∇blL∥2 = Θ(1/(L√nout)), which leads to

1 nout∥∇blL∥2 = Θ(

1 Lnout

∥∇blL∥R =

).

- Table 6 µP implementation of Condition 3.1 (k = 2) for AdamW [27], Lion [7], and Sophia [25] with weight decay under width–depth scaling. Entries in purple indicate differences between µP and SP, while gray shows the corresponding SP choices. Here, rn and rL denote the width and depth scaling ratios relative to the base model. The variance of input weights is σbase2 for language and σbase2 /d0 for image. The initial variance of input bias is σbase2 .

Input weights & biases Hidden weights Output weights Hidden biases Block Multiplier αbase αbase/rL (αbase) αbase/rn (αbase) αbase/rL (αbase)

Initial Variance σbase2 /d0 or σbase2 σbase2 /rn (σbase2 ) σbase2 σbase2 Learning Rate ηbase ηbase/rn (ηbase) ηbase ηbase Weight Decay λbase λbasern (λbase) λbase λbase

AdamW ε εbase/rn (εbase) εbase/(rLrn) (εbase) εbase/rn (εbase) εbase/(rLrn) (εbase)

To sum up, we can estimate the scale of ∥∇blL∥R as

Θ(1/nout), l = 0, Θ(1/(Lnout)), l ∈ [L].

∥∇blL∥R =

Requiring ∥∆bl∥R = ηb

l∥∇blL∥R = Θ(1) according to update condition in Equation (13) leads to the learning rate as

Θ(nout), l = 0, Θ(Lnout), l ∈ [L].

ηb

=

l

For the weight decays, we need to satisfy λb

l∥bl∥R = ∥∇blL∥R and given ∥bl∥R = Θ(1) by initialization implementation in Condition B.3, we have

λb

=

l

Θ(1/nout), l = 0, Θ 1/(Lnout) , l ∈ [L].

This completes the implementation of the update condition for SGD with weight decay, which is summarized

- in Table 5.

#### C.6 AdamW

In this section, we derive the µP implementation from Condition 3.1 for AdamW, which recovers and extends the µP scaling rules studied in Dey et al. [10].

###### C.6.1 Update Rule

First, we present the full update rule of AdamW [27]. To distinguish iteration steps, we append a superscript t ∈ [T], which might be omitted later when no confusion arises.

Wl(t) = Wl(t−1) − ηl(t) AdamW ∇W(t)

L + λlWl(t) , where

l

mˆ (lt) vˆl(t) + εl

AdamW ∇W(t)

L =

,

l

 

(t) l

mˆ (lt) = m

1−β1t , m(lt) = β1m(lt−1) + (1 − β1)∇W(t)

L, vˆl(t) = v

where

l

2

(t) l



1−β2t , vl(t) = β2vl(t−1) + (1 − β2) ∇W(t)

L

.

l

(19)

We simplify the full update rule by omitting the momentum and the stabilization term, i.e., setting β1 = 0,

- β2 = 0, and εl = 0. As discussed at the beginning of the Appendix C, omitting momentum does not affect the

scaling analysis. The stabilization term εl must, in fact, be scaled consistently with vˆl(t), and since it does not alter the resulting parameterization of learning rate, we defer its discussion to the end of this section.

Now, the AdamW is reduced to sign gradient descent as:

Wl(t) = Wl(t−1) − ηl(t) sign ∇W(t)

L + λlWl(t) . Here, the superscript of the iteration step can be left out, and we write this simplified update rule as: ∆Wl = −ηl sign ∇WlL

l

+λlWl . (20)

Al

Given the dimension assumption d0,dL+1 = Θ(1),nl = Θ(n) in Equation (3), the norm of Al satisfies

nin nout

∥Al∥R = sign ∇WlL

=

R

nin nout · Θ sign ∇WlL

=

sign ∇WlL

F

√ninnout

nin nout

= Θ

 

Θ(1), l = 0, Θ(nin), l ∈ [L], Θ(nin), l = L + 1.

= Θ(nin) =



###### C.6.2 Derivation of Parameterization

2

(low-rank property of Al in Equation (15))

(21)

Input and output layers. When weight decay is disabled (λ0 = 0), given the dimension assumption d0,dL+1 = Θ(1),nl = Θ(n) in Equation (3), the multiplier parameterizations α0 = Θ(1),αL+1 = Θ(1/nin) in Equation (7), and the scale of ∥Al∥R in Equation (21), we have

Θ(η0), l = 0, Θ(ηL+1), l = L + 1.

αl∥∆Wl∥R = αlηl∥Al∥R =

As desired in (∆1), to satisfy (C2.1) that α0∥∆W0∥R,αL+1∥∆WL+1∥R = Θ(1), we need to set η0 = Θ(1), ηL+1 = Θ(1). When λl ̸= 0, given Equation (6) that ∥W0∥R = Θ(1) and ∥WL+1∥R = Θ(nin), we have ∥λlWl∥R =

Θ(λ0), l = 0, Θ(λL+1nin), l = L + 1.

To satisfy (∆2) that ∥λlWl∥R = Θ ∥Al∥R , we need to set λ0 = Θ(1), λL+1 = Θ(1).

Hidden layers (first-order). When weight decay is disabled (λ0 = 0), given the dimension assumption d0,dL+1 = Θ(1),nl = Θ(n) in Equation (3), the weight norm ∥Wl∥R = Θ(1) in Equation (6), the multiplier parameterization αl = Θ(1/L) in Equation (8), and the scale of ∥Al∥R in Equation (21), we have

αl∥∆Wl(2)∥R∥Wl(1)∥R = Θ(1/L) · ηl(2)∥A(2)l ∥R∥Wl(1)∥R = Θ(ηl(2)nin/L).

As desired in (∆1), to satisfy the first-order update condition on hidden weights (C2.2) that αl∥∆Wl(2)∥R ∥Wl(1)∥R = Θ(1/L), we need to set

ηl(2) = Θ(1/nin). When λl ̸= 0, given the weight norm ∥Wl∥R = Θ(1) in Equation (6), we have

∥λ(2)l Wl(2)∥R = Θ(λ(2)l ). To satisfy (∆2) that ∥λlWl∥R = Θ ∥Al∥R we need to set

λ(2)l = Θ(nin).

Symmetrically, we have the same choice for Wl(1): ηl(1) = Θ(1/nin), λ(1)l = Θ(nin).

Hidden layers (second-order). As illustrated in Section 3.3.3 or in Appendix C.4 for Muon, the second-order update condition is satisfied automatically once the initial condition and the first-order update condition are met.

out×1, by the definition we have ∥sign ∇blL ∥R = Θ(1), 0 ≤ l ≤ L. To satisfy the condition ∥∆bl∥R = Θ(1) for ∀ 0 ≤ l ≤ L in Equation (13), we set ηb

Biases. For bias parameters bl ∈ Rn

= Θ(1), 0 ≤ l ≤ L, and the corresponding weight decay

l

= Θ(1), 0 ≤ l ≤ L.

λb

l

Parameterization of εl. To make the stabilization term εl effective and not dominate the gradient, we desire it to be of the same scale as vˆl(t). When omitting the momentum, we have √vˆl = ∇WlL. Therefore, we need to ensure εl = Θ(∥∇Vec(Wl)L∥R), the latter can be estimated based on the derivation for ∥∇WlL∥R in Equation (18) of Appendix C.5.

For the input weights W0, we have

1 √ninnout∥∇W0L∥F = Θ

∥∇Vec(W0)L∥R =

1 √ninnout ∥∇W0L∥2 = Θ

1 √ninnout

nin nout

1 nout

= Θ(

).

Therefore, we set

1 nout

ε0 = Θ(

).

For the hidden weights Wl,l ∈ [L], we have

∥∇Vec(Wl)L∥R = Θ

1 √ninnout ∥∇WlL∥2 = Θ

1 √ninnout

1 L

Therefore, we set

1 Lnout

), l ∈ [L].

εl = Θ(

nin nout

1 Lnout

= Θ(

).

For the output weights WL+1, we have

∥∇Vec(WL+1)L∥R = Θ

1 √ninnout∥∇WL+1L∥2 = Θ

Therefore, we set

1 nin

εL+1 = Θ(

).

Similarly, for the biases we have derived in Appendix C.5 that

1 √ninnout

1 ninnout

1 nin

= Θ(

).

∥∇blL∥R =

Therefore, we set the stabilization term as

Θ(1/nout), l = 0, Θ(1/(Lnout)), l ∈ [L].

εb

=

l

Θ(1/nout), l = 0, Θ(1/(Lnout)), l ∈ [L].

This completes the implementation of the update condition for AdamW, which is summarized in Table 6.

- C.7 Lion In this section, we derive the µP implementation from Condition 3.1 for Lion. The full update rule of Lion [7] is

Wl(t) = Wl(t−1) − η(t) u(lt) + λlWl(t) , where

u(lt) = sign β1m(lt−1) + (1 − β1)∇W(t)

l

L ,

m(lt) = β2m(lt−1) + (1 − β2)∇W(t)

l

L.

If the momentum terms are omitted, i.e., setting β1 = 0 and β2 = 0, Lion is reduced to sign gradient descent as in the simplified AdamW update rule in Equation (20). Therefore, we reuse AdamW’s parameterizations in

- Table 6 for Lion.

- C.8 Sophia In this section, we derive the µP implementation from Condition 3.1 for Sophia. The full update rule of Sophia [25] is

 clip

 ,

m(lt) max{ρh(lt),ε}

Wl(t) = Wl(t−1) − η(t)

,1 + λlWl(t)

where

m(lt) = β1m(lt−1) + (1 − β1)G(lt), and h(lt) is updated every k iterations as

h(lt) =

β2h(lt−1) + (1 − β2)hˆ(lt), t mod k = 1, h(lt−1), t mod k ̸= 1.

where the elements of hˆ(lt) are the diagonal second-order derivatives with respect to Wl(t), i.e., h ˆ(lt)

∂2L ∂(Wl(t))2ij

.

=

ij

(t) l

Letting A(lt) = clip m

,1 , we use the following upper bound to estimate its norm, following Ngom et al. [29], which derives a parameterization for Sophia under width scaling:

max{ρh(lt),ε}

∥A(lt)∥R ≤ ∥1n

out×nin∥R =

nin nout ∥1n

out×nin∥2 =

√ninnout = nin.

nin nout

The resulting width-scaling parameterization is empirically validated to be effective in Ngom et al. [29]. Since this bound has the same order as the AdamW estimate in Equation (21), Sophia shares AdamW’s parameterizations in Table 6.

#### C.9 Shampoo

In this section, we derive the µP implementation from Condition 3.1 for Shampoo, which recovers and extends the µP scaling rules studied in Qiu et al. [30].

###### C.9.1 Update Rule

Denote G(lt) = ∇W(t)

L. Following Shampoo [12], the gradient is preconditioned by Kronecker-factored left

l

and right preconditioners: Ml(t) = β1Ml(t−1) + (1 − β1)G(lt), L(lt) = β2L(lt−1) + (1 − β2)G(lt)G(lt)

⊤

, Rl(t) = β2Rl(t−1) + (1 − β2)G(lt)

⊤

G(lt),

L(lt) 1 − β2t

Rl(t) 1 − β2t

L(lt) =

, Rl(t) =

,

− 41

− 41

Wl(t) = Wl(t−1) − η(t) L(lt) + εlI

Ml(t) Rl(t) + εlI

+ λlWl(t) , (22)

We simplify the full update rule by omitting the momentum and the stabilization term, i.e., setting β1 = 0, β2 = 0, and εl = 0. As discussed at the beginning of the Appendix C, omitting momentum does not affect the scaling analysis. The stabilization term εl must, in fact, be scaled consistently with L(lt) and Rl(t), and since it does not alter the resulting parameterization of learning rate, we defer its discussion to the end of this section.

⊤

⊤

Now, we have L(lt) = G(lt)G(lt)

, and Rl(t) = G(lt)

G(lt). Therefore, we obtain

  G(lt)G(lt)

− 14

− 14

⊤

⊤

Wl(t) = Wl(t−1) − η(t)

G(lt) G(lt)

G(lt)

###### C.9.2 Derivation of Parameterization

 .

+ λlWl(t)

Learning rate and weight decay. Applying compact SVD to G(lt) as in Muon, and interpreting the inverse powers on the support of the gradient, we have

⊤

G(lt) = Ul(t)Σ(lt)Vl(t)

,

and thus

⊤

⊤

⊤

⊤

2

2

G(lt)G(lt)

= Ul(t)Σ(lt)

Ul(t)

, G(lt)

G(lt) = Vl(t)Σ(lt)

Vl(t)

. The preconditioned direction is therefore

⊤

G(lt)G(lt)

− 41

G(lt) G(lt)

⊤

G(lt)

− 41

− 12

− 12

⊤

⊤

⊤

⊤

= Ul(t)Σ(lt)

Ul(t)

Ul(t)Σ(lt)Vl(t)

Vl(t)Σ(lt)

Vl(t)

= Ul(t)Vl(t)

. Consequently, under this simplification, Shampoo reduces to

⊤

Wl(t) = Wl(t−1) − η(t) Ul(t)Vl(t)

+ λlWl(t) ,

which matches the update rule of Muon in Equation (16). Therefore, the learning rate and weight decay of Shampoo share Muon’s parameterizations in Table 4.

out/nin)1−(eL+eR) L2(eL+eR)−1neblkL+eR

Note that the hidden layer learning rate parameterization derived in Qiu et al. [30, Table 1] is (n

,

where the eL and eR are the exponents of L(lt) and Rl(t) which equal 14 in the standard Shampoo as Equation (22), and nblk is the number of blocks which equals 1 when blocking is not used. In this case,

(nout/nin)1−(eL+eR) L2(eL+eR)−1neblkL+eR

= nout/nin = Θ(1), consistent with our result for Shampoo in Table 4.

Parameterization of εl. To make the stabilization term εl effective and not dominate the update, we desire it to be of the same scale as the single value of L(lt) and Rl(t). When omitting the momentum, we have

 

Θ( n

nout) = Θ(n1

), l = 0, Θ(L12 n

in

out

∥ L(lt)∥2 = ∥ Rl(t)∥2 = ∥G(lt)∥22 = ∥∇W(t)

nout) = Θ(L12 ), l ∈ [L], Θ(n 1

L∥22 =

in



l

innout) = Θ(n1

), l = L + 1,

in

L∥2 can be found in the derivation for ∥∇WlL∥R in Equation (18) at Appendix C.5. Therefore, we need to set the stabilization term as

where the estimation of ∥∇W(t)

l

 

Θ(n1

), l = 0,

out

Θ(L12 ), l ∈ [L], Θ(n1

εl =



), l = L + 1.

in

Note that the hidden layer εl parameterization derived in Qiu et al. [30, Table 1] is n

L2noutnblk, where the nblk is the number of blocks which equals 1 when blocking is not used. In this case, n

in

L2nout = Θ(L12), consistent with our result for Shampoo in Table 4.

L2noutnblk = n

in

in

#### C.10 SOAP

In this section, we derive the µP implementation from Condition 3.1 for SOAP, which recovers and extends the µP scaling rules studied in Qiu et al. [30]. Denote the weight gradient as G(lt) = ∇W(t)

out×nin and its rank r = rank(G(lt)). SOAP adopts a similar precondition of Shampoo’s for L(lt) and Rl(t):

L ∈ Rn

l

⊤

L(lt) = β3L(lt−1) + (1 − β3)G(lt)G(lt)

⊤

, Rl(t) = β3Rl(t−1) + (1 − β3)G(lt)

G(lt).

By applying eigendecomposition to matrices L(lt) ∈ Rn

out×nout and Rl(t) ∈ Rn

in×nin, we get two orthogonal matrices Q(Lt)

out×nout and Q(Rt)

in×nin as:

∈ Rn

∈ Rn

l

l

⊤

⊤

L(lt) = Q(Lt)

###### Λ(Lt)

###### Q(Lt)

, Rl(t) = Q(Rt)

###### Λ(Rt)

###### Q(Rt)

. (23) This induces a rotated gradient:

l

l

l

l

l

l

⊤

G′l(t) = Q(Lt)

l

###### G(lt)Q(Rt)

.

l

The full update rule of SOAP is

⊤

AdamW G′l(t) Q(Rt)

Wl(t) = Wl(t−1) − η(t) Q(Lt)

l

l

+ λlWl(t) ,

where AdamW (·) is defined as in Equation (19). First, omit the momentum and the stabilization term in the AdamW operator. Then, AdamW (·) is reduced to sign(·) as discussed in Appendix C.6.1. Then, omit the momentum term in L(lt) and Rl(t), i.e., set β3 = 0. Applying compact SVD to the weight matrix G(lt) = Ul(t)Σ(lt)Vl(t)

⊤

, where Ul(t) ∈ Rn

out×r, Σ(lt) ∈ Rr×r, and Vl(t) ∈ Rn

in×r, we have

⊤

⊤

⊤

⊤

2

2

L(lt) = G(lt)G(lt)

= Ul(t)Σ(lt)

Ul(t)

, Rl(t) = G(lt)

G(lt) = Vl(t)Σ(lt)

Vl(t)

.

According to Equation (23), the eigenvectors corresponding to the non-zero eigenvalues match the singular vectors, i.e.,

###### Q(Lt)

l[:,:r] = Ul(t), Q(Rt)

l[:,:r] = Vl(t).

We can partition the orthogonal matrices as Q(Lt)

= [Vl(t) V⊥(t)]. Substituting the eigendecomposition of G(lt), the rotated gradient becomes:

= [Ul(t) U⊥(t)] and Q(Rt)

l

l

 Ul(t)

 Ul(t)Σ(lt)Vl(t)

⊤

Σ(lt) 0 0 0

⊤

⊤

G′l(t) = Q(Lt)

###### G(lt)Q(Rt)

Vl(t) V⊥(t) =

.

=

⊤

U⊥(t)

l

l

Then, we find that SOAP is reduced to:

  Q(Lt)

  

  Σ(lt) 0

 Q(Rt)

⊤

Wl(t) = Wl(t−1) − η(t)

+ λlWl(t)

sign

0 0

l

l

Ir×r 0 0 0

⊤

= Wl(t−1) − η(t) Q(Lt)

+ λlWl(t)

Q(Rt)

l

l

⊤

= Wl(t−1) − η(t) Ul(t)Vl(t)

+ λlWl(t) ,

which, again, matches exactly the update rule of Muon in Equation (16). Therefore, SOAP shares Muon’s parameterizations in Table 4.

eL/2 out neinR/2

Note that the hidden layer learning rate parameterization derived in Qiu et al. [30, Table 1] is n

nin , where the eL and eR are the indicators for left- and right-side preconditioners, which equals 1 for standard SOAP. In this case, n

√noutnin

eL/2 out neinR/2

nin = Θ(1), consistent with our result for SOAP in Table 4.

nin =

- Table 7 µP implementation of Condition 3.1 (k = 2) for SSO [39] with weight decay under width-depth scaling. Entries in purple indicate differences between µP and SP, while gray shows the corresponding SP choices. Here, rn and rL denote the width and depth scaling ratios relative to the base model. The variance of input weights is σbase2 for language and σbase2 /d0 for image.

Input weights Hidden weights Output weights Block Multiplier αbase αbase/rL (αbase) αbase/rn (αbase)

Initial Variance σbase2 /d0 or σbase2 σbase2 /rn (σbase2 ) σbase2 Learning Rate ηbase ηbase ηbasern (ηbase) Weight Decay λbase λbase λbase/rn (λbase)

- C.11 Spectral Sphere Optimizer (SSO) In this section, we derive the µP implementation from Condition 3.1 for SSO.

###### C.11.1 Update Rule

SSO [39] aims to perform steepest descent on the spectral sphere (see Section 3.1 in the original paper), where the update follows:

∆Wl = −ηl RΦl

+λlWl ,

Al

with

R = Θ

nout nin

, and Φl = arg max

⟨∇WlL,Φ⟩ s.t. ∥Φ∥2 = 1, ∥Wl − ηlΦ∥2 = ∥Wl∥2 = R.

Φ

Thus we have

∥Al∥R =

nin nout∥Al∥2 =

nin nout

R∥Φl∥2 =

nin nout

Θ

nout nin

= 1. (24)

###### C.11.2 Derivation of Parameterization

Input and output layers. When weight decay is disabled (λ0 = 0), given the dimension assumption d0,dL+1 = Θ(1),nl = Θ(n) in Equation (3), the multiplier parameterizations α0 = Θ(1),αL+1 = Θ(1/nin) in Equation (7), and the scale of ∥Al∥R in Equation (24), we have

Θ(η0), l = 0, Θ(ηL+1/nin), l = L + 1.

αl∥∆Wl∥R = αlηl∥Al∥R =

As desired in (∆1), to satisfy (C2.1) that α0∥∆W0∥R,αL+1∥∆WL+1∥R = Θ(1), we need to set η0 = Θ(1), ηL+1 = Θ(nin). When λl ̸= 0, given Equation (6) that ∥W0∥R = Θ(1) and ∥WL+1∥R = Θ(nin), we have ∥λlWl∥R =

Θ(λ0), l = 0, Θ(λL+1nin), l = L + 1.

To satisfy (∆2) that ∥λlWl∥R = Θ ∥Al∥R , we need to set λ0 = Θ(1), λL+1 = Θ(1/nin),

Hidden layers (first-order). When weight decay is disabled (λl = 0), given the dimension assumption d0,dL+1 = Θ(1),nl = Θ(n) in Equation (3), the weight norm ∥Wl∥R = Θ(1) in Equation (6), the multiplier parameterization αl = Θ(1/L) in Equation (8), and the scale of ∥Al∥R in Equation (24), we have

αl∥∆Wl(2)∥R∥Wl(1)∥R = Θ(1/L) · ηl(2)∥A(2)l ∥R∥Wl(1)∥R = Θ(ηl(2)/L).

As desired in (∆1), to satisfy the first-order update condition on hidden weights (C2.2) that αl∥∆Wl(2)∥R ∥Wl(1)∥R = Θ(1/L), we need to set

ηl(2) = Θ(1). When weight decay is enabled (λl ̸= 0), given the weight norm ∥Wl∥R = Θ(1) in Equation (6), we have

∥λ(2)l Wl(2)∥R = Θ(λ(2)l ). To satisfy (∆2) that ∥λlWl∥R = Θ ∥Al∥R we need to set

λ(2)l = Θ(1).

Symmetrically, we have the same choice for Wl(1): ηl(1) = Θ(1), λ(1)l = Θ(1).

Hidden layers (second-order). As illustrated in Section 3.3.3 or in Appendix C.4 for Muon, the second-order update condition is satisfied automatically once the initial condition and the first-order update condition are met.

This completes the implementation of the update condition for SSO with weight decay, which is summarized in Table 7.

### Appendix D Implementing Condition B.1 for Optimizers with Weight Decay

We implement Condition B.1 using the same width-scaling µP initialization convention as in Section 4.1. Under this convention, hidden matrix weights satisfy ∥Wl∥R = Θ(1) at initialization, so the hidden initial condition in Condition B.1 permits αl = O(1/

√

L). In this section, we focus on the Depth-µP-style choice αl = Θ(1/

√

L), which corresponds to maximizing the zero-order feature-update contribution discussed in Appendix B.1. The input and output layer multipliers remain the same as in Section 4.1, that α0 = Θ(1) and αL+1 = Θ(1/nin). We now start update-condition analysis for optimizers with weight decay.

To provide a unified derivation across different optimizers, we begin by expressing their update rules in a general form. When weight decay is included, a single update step of the weight matrix can be written as

∆Wl = −ηl Al + λlWl , where Al denotes an optimizer-specific update for Wl, and λl is the weight decay coefficient. The update magnitude ∥∆Wl∥R = ηl∥Al+λlWl∥R is required to satisfy the update conditions in Condition B.1. We analyze this requirement under two complementary regimes. Without weight decay. When weight decay is disabled (λl = 0), the update reduces to ∥∆Wl∥R = ηl∥Al∥R. In this case, we require:

∥∆Wl∥R = ηl∥Al∥R satisfies Condition B.1. (Υ1)

With weight decay. When weight decay is enabled (λl ̸= 0), we choose the weight decay term to be comparable in scale to the optimizer-driven term:

∥λlWl∥R = Θ ∥Al∥R . (Υ2)

As discussed in Appendix C, this is a non-degenerate implementation convention: it keeps weight decay active in the update dynamics without letting it dominate the optimizer-driven term. Under the usual scale estimate, ∥Al + λlWl∥R = Θ(∥Al∥R), so the learning-rate scaling rule derived from Equation (Υ1) is preserved.

- Table 8 Overview of µP implementation from Condition B.1 (k = 1) for optimizer families with weight decay. Optimizers in the same row share the same µP scaling rules.

Optimizer family Detailed parameterization Muon-Kimi Table 9 Muon / Shampoo / SOAP Table 10

SGD Table 11 AdamW / Lion / Sophia Table 12

SSO Table 13

- Table 9 µP implementation of Condition B.1 (k = 1) for Muon-Kimi [26] with weight decay under width-depth scaling. Entries in purple indicate differences between µP and SP, while gray shows the corresponding SP choices. Here, rn and rL denote the width and depth scaling ratios relative to the base model. The variance of input weights is σbase2 for language and σbase2 /d0 for image.

Input weights Hidden weights Output weights Block Multiplier αbase αbase/√rL (αbase) αbase/rn (αbase)

Initial Variance σbase2 /d0 or σbase2 σbase2 /rn (σbase2 ) σbase2 Learning Rate ηbase ηbase/√rnrL (ηbase) ηbase Weight Decay λbase λbase√rn (λbase) λbase

Weights and biases. In the following, we derive parameterizations of the learning rate and weight decay coefficient for a range of optimizers. As in Appendix C, matrix-based optimizers such as Muon and Shampoo are typically not applied to bias parameters, so we restrict the analysis of bias parameterization to vector-based optimizers such as SGD and AdamW. The same two rules, Equations (Υ1) and (Υ2), are then applied to the biases with the corresponding one-layer specialization of spectral Condition B.3.

Momentum. We use the same momentum simplification as in Appendix C: the derivations omit momentum, while practical implementations take momentum coefficients to be Θ(1). This can also be interpreted as analyzing the first update step after initialization.

#### D.1 Overview

Table 8 summarizes the optimizer families covered in this section and points to the corresponding detailed Depth-µP-style parameterization tables. All rows use the initialization convention above, with hidden residual multiplier αl = Θ(1/

√

L); the table only organizes the optimizer-dependent update rules derived below.

##### D.2 Muon-Kimi In this section, we derive the µP implementation from Condition B.1 for Muon-Kimi.

For a weight matrix Wl ∈ Rn

out×nin, the update rule of Muon-Kimi [26] with weight decay is ∆Wl = −ηl 0.2 max{nin,nout}UlVl⊤

+λlWl ,

Al

where Ul,Vl arise from the compact SVD of the gradient ∇WlL = UlΣlVl⊤. Using the norm computation in Equation (9), we have

 

 √nin max 1,

  =

Θ(1), l = 0, Θ(√nin), l ∈ [L], Θ(nin), l = L + 1.

nin nout

∥Al∥R = Θ



- Table 10 µP implementation of Condition B.1 (k = 1) for Muon [21], Shampoo [12], and SOAP [38] with weight decay under width-depth scaling. Entries in purple indicate differences between µP and SP, while gray shows the corresponding SP choices. Here, rn and rL denote the width and depth scaling ratios relative to the base model. The variance of input weights is σbase2 for language and σbase2 /d0 for image.

Input weights Hidden weights Output weights Block Multiplier αbase αbase/√rL (αbase) αbase/rn (αbase)

Initial Variance σbase2 /d0 or σbase2 σbase2 /rn (σbase2 ) σbase2 Learning Rate ηbase√rn (ηbase) ηbase/√rL (ηbase) ηbase√rn (ηbase) Weight Decay λbase/√rn (λbase) λbase λbase/√rn (λbase) Shampoo ε εbase/rn (εbase) εbase/rL (εbase) εbase/rn (εbase)

###### D.2.1 Derivation of Parameterization

Input and output layers. The input and output layer conditions in Condition B.1 are identical to those in Condition 3.1. Therefore, as in Appendix C.3, their Muon-Kimi learning-rate and weight-decay parameterizations are

η0 = Θ(1), ηL+1 = Θ(1), λ0 = Θ(1), λL+1 = Θ(1).

Hidden layers (first-order). The only change from the k = 2 Muon-Kimi implementation is the hidden multiplier: Condition B.1 uses the Depth-µP-style choice αl = Θ(1/

√

L) rather than Θ(1/L). When λl = 0, given the dimension assumption d0,dL+1 = Θ(1),nl = Θ(n) in Equation (3) and the scale of ∥Al∥R above, we have

√

√

L) · ηl∥Al∥R = Θ(ηl√nin/

αl∥∆Wl∥R = Θ(1/

L).

As desired in Equation (Υ1), to satisfy the first-order update condition on hidden weights, αl∥∆Wl∥R = Θ(1/L), we need to set

1 √ninL

ηl = Θ(

).

When λl ̸= 0, the weight-decay parameterization is unchanged from the k = 2 Muon-Kimi implementation because it matches the scale of Al and does not depend on the residual multiplier. Given the weight norm ∥Wl∥R = Θ(1) in Equation (6), we have

∥λlWl∥R = Θ(λl). To satisfy Equation (Υ2) that ∥λlWl∥R = Θ ∥Al∥R , we need to set λl = Θ(√nin).

This completes the implementation of Condition B.1 for Muon-Kimi with weight decay, as summarized in Table 9.

- D.3 Muon, Shampoo and SOAP In this section, we derive the µP implementation from Condition B.1 for Muon, Shampoo, and SOAP.

For a weight matrix Wl ∈ Rn

out×nin, the update rule of Muon [21] with weight decay is ∆Wl = −ηl UlVl⊤

+λlWl ,

Al

where Ul,Vl arise from the compact SVD of the gradient ∇WlL = UlΣlVl⊤. Using the norm computation in Equation (17), we have

 

Θ(1/√nout), l = 0, Θ(1), l ∈ [L], Θ(√nin), l = L + 1.

nin nout

∥Al∥R =

=



As discussed in Appendices C.9 and C.10, Shampoo and SOAP reduce to the same Muon update direction under the simplifications used in this section, so they share the same parameterization.

- D.3.1 Derivation of Parameterization Input and output layers. The input and output layer conditions in Condition B.1 are identical to those in

- Condition 3.1. Therefore, as in Appendix C.4, their Muon learning-rate and weight-decay parameterizations are

η0 = Θ(√nout), ηL+1 = Θ(√nin), λ0 = Θ(1/√nout), λL+1 = Θ(1/√nin).

Hidden layers (first-order). The only change from the k = 2 Muon implementation is the hidden multiplier: Condition B.1 uses the Depth-µP-style choice αl = Θ(1/

√

L) rather than Θ(1/L). When λl = 0, given the dimension assumption d0,dL+1 = Θ(1),nl = Θ(n) in Equation (3) and the scale of ∥Al∥R above, we have

√

√

αl∥∆Wl∥R = Θ(1/

L) · ηl∥Al∥R = Θ(ηl/

L).

As desired in Equation (Υ1), to satisfy the first-order update condition on hidden weights, αl∥∆Wl∥R = Θ(1/L), we need to set

√

ηl = Θ(1/

L).

When λl ̸= 0, the weight-decay parameterization is unchanged from the k = 2 Muon implementation because it matches the scale of Al and does not depend on the residual multiplier. Given the weight norm ∥Wl∥R = Θ(1) in Equation (6), we have

∥λlWl∥R = Θ(λl). To satisfy Equation (Υ2) that ∥λlWl∥R = Θ ∥Al∥R , we need to set λl = Θ(1).

Parameterization of Shampoo εl. As in Appendix C.9, Shampoo’s stabilization term is added to the left and right preconditioners L(lt) and Rl(t), so it should match the scale of these preconditioners rather than the elementwise gradient scale. When omitting the momentum, this gives

∥ L(lt)∥2 = ∥ Rl(t)∥2 = ∥G(lt)∥22 = ∥∇W(t)

L∥22.

l

The input and output gradient estimates are unchanged from Appendix C.9. For hidden weights, the one-layer condition uses αl = Θ(1/

√

L), and the raw-gradient estimate in Equation (25) gives ∥∇WlL∥2 = Θ( nin/(noutL)). Therefore,

 

Θ( n

###### nout) = Θ(n1

###### ), l = 0, Θ(L1 n

in

out

∥ L(lt)∥2 = ∥ Rl(t)∥2 =

nout) = Θ(L1 ), l ∈ [L], Θ(n 1

in



innout) = Θ(n1

), l = L + 1. Thus, the stabilization term should be parameterized as

in

 

Θ(n1

), l = 0,

out

Θ(L1 ), l ∈ [L], Θ(n1

εl =



###### ), l = L + 1.

in

- Table 11 µP implementation of Condition B.1 (k = 1) for SGD with weight decay under width–depth scaling. Entries in purple indicate differences between µP and SP, while gray shows the corresponding SP choices. Here, rn and rL denote the width and depth scaling ratios relative to the base model. The variance of input weights is σbase2 for language and σbase2 /d0 for image. The initial variance of input bias is σbase2 .

Input weights & biases Hidden weights Output weights Hidden biases Block Multiplier αbase αbase/√rL (αbase) αbase/rn (αbase) αbase/√rL (αbase)

Initial Variance σbase2 /d0 or σbase2 σbase2 /rn (σbase2 ) σbase2 σbase2 Learning Rate ηbasern (ηbase) ηbase ηbasern (ηbase) ηbasern (ηbase) Weight Decay λbase/rn (λbase) λbase/√rL (λbase) λbase/rn (λbase) λbase/(rn√rL) (λbase)

This corresponds to the Shampoo ε row in Table 10. This completes the implementation of Condition B.1 for Muon, Shampoo, and SOAP with weight decay, as summarized in Table 10.

#### D.4 SGD

In this section, we derive the µP implementation from Condition B.1 for SGD, which recovers and extends the µP scaling rules studied in Bordelon et al. [6], Yang et al. [46].

For a weight matrix Wl ∈ Rn

out×nin, the SGD update rule with weight decay can be written as: ∆Wl = −ηl ∇WlL

+λlWl .

Al

For SGD, Al is the raw gradient, so the scale of Al depends on the residual multiplier used in the spectral update condition. The input and output layer estimates are the same as in Appendix C.5: ∥A0∥R = Θ(nin/nout) = Θ(1/nout) and ∥AL+1∥R = Θ(1/nout) = Θ(1). The hidden-layer estimate changes because Condition B.1 uses αl = Θ(1/

√

L). For a hidden weight Wl, we have

1 L

lL = Θ(⟨∆Wl,∇WlL⟩) = Θ(∥∆Wl∥F∥∇WlL∥F) = Θ(∥∆Wl∥2∥∇WlL∥2).

Θ(

) = ∆W

Since we set αl∥∆Wl∥R = Θ(1/L) to satisfy the hidden update condition in Condition B.1, and use αl = Θ(1/

√

√

L) and thus ∥∆Wl∥2 = Θ( nout/(ninL)). Therefore, ∥∇WlL∥2 = Θ( nin/(noutL)), which leads to

L), we have ∥∆Wl∥R = Θ(1/

∥Al∥R = ∥∇WlL∥R =

nin nout ∥∇WlL∥2 = Θ(

nin nout

1 √

√

) = Θ(

).

L

L

To sum up, we have

 

Θ(1/nout), l = 0, Θ(1/

√

(25)

∥Al∥R = ∥∇WlL∥R =

L), l ∈ [L], Θ(1), l = L + 1.



- D.4.1 Derivation of Parameterization Input and output layers. The input and output layer conditions in Condition B.1 are identical to those in

- Condition 3.1. Therefore, as in Appendix C.5, their SGD learning-rate and weight-decay parameterizations are

η0 = Θ(nout), ηL+1 = Θ(nin), λ0 = Θ(1/nout), λL+1 = Θ(1/nin).

Hidden layers (first-order). Unlike normalized or preconditioned optimizers, SGD uses the raw gradient, so changing the hidden multiplier also changes the hidden raw-gradient scale. When λl = 0, using αl = Θ(1/

√

L) and ∥Al∥R = Θ(1/

√

L), we obtain αl∥∆Wl∥R = Θ(1/

√

L) · ηl∥Al∥R = Θ(ηl/L). Enforcing the hidden update condition in Condition B.1 that αl∥∆Wl∥R = Θ(1/L) gives ηl = Θ(1).

If weight decay is enabled on hidden matrices, using ∥Wl∥R = Θ(1) by Equation (6), we obtain ∥λlWl∥R = Θ(λl), so Equation (Υ2) implies ∥λlWl∥R = Θ(∥Al∥R) = Θ(1/

√

L) and therefore λl = Θ(1/

√

L).

Biases. As in Appendix C.5, the raw-gradient scale of input biases is ∥∇b0L∥R = Θ(1/nout). For hidden biases bl ∈ Rn

out×1, we similarly have Θ(1/L) = ∆b

lL = Θ(⟨∆bl,∇blL⟩) = Θ(∥∆bl∥2∥∇blL∥2), Since we set αl∥∆bl∥R = Θ(1/

√

L) · ∥∆bl∥R = Θ(1/L) to satisfy the one-layer bias update condition in Condition B.3, we have ∥∆bl∥R = Θ(1/

√

L) and thus ∥∆bl∥2 = Θ( nout/L). Therefore, ∥∇blL∥2 = Θ(1/√noutL), which leads to

∥∇blL∥R =

1 nout∥∇blL∥2 = Θ(

1 nout

√

).

L

To sum up, we can estimate the scale of ∥∇blL∥R as

 

Θ(1/nout), l = 0, Θ 1/(nout

√

∥∇blL∥R =



L) , l ∈ [L]. Requiring α0∥∆b0∥R = α0ηb

0∥∇b0L∥R = Θ(1) and αl∥∆bl∥R = αlηb

l∥∇blL∥R = Θ(1/L) for l ∈ [L] gives ηb

= Θ(nout), l ≤ L. For the weight decays, we match λb

l

l∥bl∥R to ∥∇blL∥R. Given ∥bl∥R = Θ(1) by the initialization implementation in Condition B.3, we have

 

Θ(1/nout), l = 0, Θ 1/(nout

√

λb

=



l

L) , l ∈ [L].

This completes the implementation of Condition B.1 for SGD with weight decay, as summarized in Table 11.

#### D.5 AdamW, Sophia and Lion

In this section, we derive the µP implementation from Condition B.1 for AdamW (same for Sophia and Lion), which recovers and extends the µP scaling rules studied in Yang et al. [46].

As in Appendix C.6, we reduce AdamW to sign gradient descent by setting β1 = 0, β2 = 0, and εl = 0: ∆Wl = −ηl sign ∇WlL

+λlWl .

Al

Using the norm estimate in Equation (21), we have

 

Θ(1), l = 0, Θ(nin), l ∈ [L], Θ(nin), l = L + 1.

∥Al∥R = sign ∇WlL

= Θ(nin) =



R

As discussed in Appendices C.8 and C.7, Sophia and Lion share the same parameterization as AdamW under the simplifications used in this section.

- Table 12 µP implementation of Condition B.1 (k = 1) for AdamW [27], Lion [7], and Sophia [25] with weight decay under width–depth scaling. Entries in purple indicate differences between µP and SP, while gray shows the corresponding SP choices. Here, rn and rL denote the width and depth scaling ratios relative to the base model. The variance of input weights is σbase2 for language and σbase2 /d0 for image. The initial variance of input bias is σbase2 .

Input weights & biases Hidden weights Output weights Hidden biases Block Multiplier αbase αbase/√rL (αbase) αbase/rn (αbase) αbase/√rL (αbase)

Initial Variance σbase2 /d0 or σbase2 σbase2 /rn (σbase2 ) σbase2 σbase2 Learning Rate ηbase ηbase/(rn√rL) (ηbase) ηbase ηbase/√rL (ηbase) Weight Decay λbase λbasern (λbase) λbase λbase

AdamW ε εbase/rn (εbase) εbase/(rn√rL) (εbase) εbase/rn (εbase) εbase/(rn√rL) (εbase)

- D.5.1 Derivation of Parameterization Input and output layers. The input and output layer conditions in Condition B.1 are identical to those in

- Condition 3.1. Therefore, as in Appendix C.6, their AdamW learning-rate and weight-decay parameterizations are

η0 = Θ(1), ηL+1 = Θ(1), λ0 = Θ(1), λL+1 = Θ(1).

Hidden layers (first-order). Unlike SGD, the sign-style update direction has norm determined by dimension rather than by the raw-gradient magnitude. Thus, the only change from the k = 2 AdamW implementation is the hidden multiplier: Condition B.1 uses the Depth-µP-style choice αl = Θ(1/

√

L) rather than Θ(1/L). When λl = 0, given the scale of ∥Al∥R above, we have

√

√

αl∥∆Wl∥R = Θ(1/

L) · ηl∥Al∥R = Θ(ηlnin/

L).

As desired in Equation (Υ1), to satisfy the first-order update condition on hidden weights in Condition B.1, αl∥∆Wl∥R = Θ(1/L), we need to set

1 nin

√

ηl = Θ(

).

L

When λl ≠ 0, the weight-decay parameterization is unchanged from the k = 2 AdamW implementation because it matches the scale of Al and does not depend on the residual multiplier. Given the weight norm ∥Wl∥R = Θ(1) in Equation (6), we have

∥λlWl∥R = Θ(λl). To satisfy Equation (Υ2) that ∥λlWl∥R = Θ ∥Al∥R , we need to set λl = Θ(nin).

Biases. For bias parameters bl ∈ Rn

out×1, by the definition we have

∥sign ∇blL ∥R = Θ(1), 0 ≤ l ≤ L. To satisfy the one-layer bias update condition, α0∥∆b0∥R = Θ(ηb

√

) = Θ(1) and αl∥∆bl∥R = Θ(1/

L · ηb

) = Θ(1/L) for l ∈ [L], we need to set

0

l

√

L), l ∈ [L]. For the weight decays, we match λb

ηb

###### = Θ(1), ηb

= Θ(1/

0

l

l∥bl∥R to ∥sign ∇blL ∥R = Θ(1). Given ∥bl∥R = Θ(1) by the initialization implementation in Condition B.3, we have

###### = Θ(1), 0 ≤ l ≤ L.

λb

l

Parameterization of εl. To make the stabilization term εl effective and not dominate the gradient, we desire it to be of the same scale as vˆl(t). When omitting the momentum, we have √vˆl = ∇WlL. Therefore, we need to ensure εl = Θ(∥∇Vec(Wl)L∥R). The latter can be estimated from the raw-gradient derivation for ∥∇WlL∥R in Equation (25) of Appendix D.4.

For the input weights W0, we have

1 √ninnout∥∇W0L∥F = Θ

∥∇Vec(W0)L∥R =

Therefore, we set

1 √ninnout ∥∇W0L∥2 = Θ

1 nout

ε0 = Θ(

).

1 √ninnout

nin nout

1 nout

= Θ(

).

For the hidden weights Wl where l ∈ [L], we have

1 √ninnout∥∇WlL∥2 = Θ

∥∇Vec(Wl)L∥R = Θ

1 √ninnout

Therefore, we set

For the output weights WL+1, we have

1 nout

√

), l ∈ [L].

εl = Θ(

L

nin noutL

1 nout

√

= Θ(

).

L

∥∇Vec(WL+1)L∥R = Θ

1 √ninnout∥∇WL+1L∥2 = Θ

Therefore, we set

1 nin

εL+1 = Θ(

).

Similarly, for the biases we have derived in Appendix D.4 that

1 √ninnout

1 ninnout

1 nin

= Θ(

).

Θ(1/nout), l = 0, Θ(1/(nout

√

∥∇blL∥R =

L)), l ∈ [L]. Therefore, we set the stabilization term as

εb

=

l

Θ(1/nout), l = 0, Θ(1/(nout

√

L)), l ∈ [L].

This completes the implementation of Condition B.1 for AdamW, Lion, and Sophia with weight decay, as summarized in Table 12.

- D.6 Spectral Sphere Optimizer (SSO) In this section, we derive the µP implementation from Condition B.1 for SSO. As in Appendix C.11, SSO [39] uses the update

where

∆Wl = −ηl RΦl

+λlWl ,

Al

nout nin

, and Φl = arg max

⟨∇WlL,Φ⟩ s.t. ∥Φ∥2 = 1, ∥Wl − ηlΦ∥2 = ∥Wl∥2 = R. Using the norm computation in Equation (24), we have

R = Θ

Φ

∥Al∥R = Θ(1).

- Table 13 µP implementation of Condition B.1 (k = 1) for SSO [39] with weight decay under width-depth scaling. Entries in purple indicate differences between µP and SP, while gray shows the corresponding SP choices. Here, rn and rL denote the width and depth scaling ratios relative to the base model. The variance of input weights is σbase2 for language and σbase2 /d0 for image.

Input weights Hidden weights Output weights Block Multiplier αbase αbase/√rL (αbase) αbase/rn (αbase)

Initial Variance σbase2 /d0 or σbase2 σbase2 /rn (σbase2 ) σbase2 Learning Rate ηbase ηbase/√rL (ηbase) ηbasern (ηbase) Weight Decay λbase λbase λbase/rn (λbase)

###### D.6.1 Derivation of Parameterization

Input and output layers. The input and output layer conditions in Condition B.1 are identical to those in Condition 3.1. Therefore, as in Appendix C.11, their SSO learning-rate and weight-decay parameterizations are

η0 = Θ(1), ηL+1 = Θ(nin), λ0 = Θ(1), λL+1 = Θ(1/nin).

Hidden layers (first-order). The only change from the k = 2 SSO implementation is the hidden multiplier: Condition B.1 uses the Depth-µP-style choice αl = Θ(1/

√

L) rather than Θ(1/L). When λl = 0, given the scale of ∥Al∥R above, we have

√

√

αl∥∆Wl∥R = Θ(1/

L) · ηl∥Al∥R = Θ(ηl/

L).

As desired in Equation (Υ1), to satisfy the first-order update condition on hidden weights in Condition B.1, αl∥∆Wl∥R = Θ(1/L), we need to set

√

ηl = Θ(1/

L).

When λl ̸= 0, the weight-decay parameterization is unchanged from the k = 2 SSO implementation because it matches the scale of Al and does not depend on the residual multiplier. Given the weight norm ∥Wl∥R = Θ(1) in Equation (6), we have

∥λlWl∥R = Θ(λl). To satisfy Equation (Υ2) that ∥λlWl∥R = Θ ∥Al∥R , we need to set λl = Θ(1). This completes the implementation of Condition B.1 for SSO with weight decay, as summarized in Table 13.

### Appendix E Additional Details and Results of GPT-2 Experiments

This section provides the experimental details and complete numerical results for the GPT-2 style languagemodel experiments in Section 5. We organize the results around three comparisons. First, we compare SP with the µP formulation derived from Condition 3.1 (k ≥ 2), which corresponds to the CompleteP-style scaling predicted for residual branches with multiple transformations. Second, we compare this formulation with the µP formulation derived from Condition B.1 (k = 1), which corresponds to the Depth-µP-style scaling for one-layer residual branches. Third, we report additional weight-decay transfer results.

- E.1 Assets and Licenses All used assets (datasets and codes) and their licenses are listed in Table 14.

Table 14 Used assets and their licenses.

URL Citation License https://github.com/EleutherAI/nanoGPT-mup/tree/completep [10] MIT

https://github.com/karpathy/nanoGPT [23] MIT https://skylion007.github.io/OpenWebTextCorpus/ [11] Creative Commons CC0

#### E.2 Additional Details of Feature Learning Experiments

This subsection reports the coordinate-check experiments used to evaluate feature-scale stability under width and depth scaling. We measure the RMS norm at the output of the final Transformer block, ∥hL∥R, after short training runs. Results are averaged over three independent runs with random seeds 1, 2, and 3. The base initialization variance for matrix weights and biases is set to 0.022 and 0, respectively. All models are trained with a constant learning-rate schedule, batch size 8, gradient clipping 1.0, and no weight decay for 10 training steps. The HP scaling rule can be found in Tables 2 and 8 for µP (k ≥ 2) and µP (k = 1), respectively. Optimizer-specific HPs are listed below.

Muon-Kimi-AdamW. Following common practice [26], hidden matrix parameters are optimized by MuonKimi with a Nesterov-style momentum [28] of 0.95. All other parameters (e.g., embedding layer, LM head, all biases) are updated by AdamW with β1 = 0.9, β2 = 0.95, ϵbase = 10−16. The learning rate is 2−7 for both Muon-Kimi and AdamW. The results are presented in Figure 1(a,b).

Muon-AdamW. Following common practice [21], hidden matrix parameters are optimized by Muon with a learning rate of 0.02 and a Nesterov-style momentum of 0.95. All other parameters (e.g., embedding layer, LM head, all biases) are updated by AdamW with a learning rate of 0.001, β1 = 0.9, β2 = 0.95, and ϵbase = 10−16. The results are presented in Figure 5(a,b).

Shampoo-AdamW. Following common practice [20], hidden matrix parameters are optimized by Shampoo with a learning rate of 0.001, β1 = 0.95, β2 = 0.95, ϵbase = 10−16, a shampoo precondition frequency of 1, and a maximal precondition dimension of 20000. All other parameters (e.g., embedding layer, LM head, all biases) are updated by AdamW with a learning rate of 0.002, β1 = 0.9, β2 = 0.95, and ϵbase = 10−16. The results are presented in Figure 6(a,b).

Sophia. Following common practice [25], parameters are updated by Sophia with a learning rate of 0.001,

- β1 = 0.965, β2 = 0.99, ρ = 0.05, and a Hessian update frequency of 1. The results are presented in Figure 8(a,b).

Overview of results. Across these optimizer settings, the coordinate-check results show the same qualitative pattern as in the main text: SP exhibits feature-scale growth under width or depth scaling, whereas the µP formulation from Condition 3.1 (k ≥ 2) keeps the final-block feature norm approximately scale-invariant.

#### E.3 Additional Details of HP Transfer Experiments

This subsection provides the detailed setup and complete numerical results for the HP-transfer experiments in Section 5. We organize the results by optimizer. For each optimizer, we first compare SP with the µP formulation from Condition 3.1 (k ≥ 2) under width and depth scaling. When applicable, we then compare this formulation with the µP formulation from Condition B.1 (k = 1) to test the role of residual-block depth. For Muon-Kimi-AdamW, we additionally report weight-decay transfer and the no-LayerNorm depth-scaling diagnostic discussed in the main text.

| | |P (w/o LN)<br><br>| | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

SP (w/o LN) Depth L

| |
|---|

| |
|---|

5.5

ValidationLoss

4 8

5.0

16 32 64

4.5

4.0

128 256

2 11 2 10 2 9 2 8 2 7

2 13 2 12 2 11 2 10 2 9

Base Learning Rate base

Base Learning Rate base

- Figure 3 Feature learning and HP transfer of Muon-Kimi-AdamW under SP and µP from Condition 3.1 (k ≥ 2) without LayerNorm. First, in terms of training stability, SP becomes increasingly prone to loss divergence as depth increases in the absence of LayerNorm, whereas µP enables stable training. Second, unlike SP, µP preserves HP transferability at large depths without LayerNorm.

2 4 2 3 2 2 2 1 20

Base Weight Decay base

3.5

4.0

4.5

5.0

ValidationLoss

SP

Width n

128 256 512

1024 2048 4096

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

2 4 2 3 2 2 2 1 20

Base Weight Decay base

(a) P

2 4 2 3 2 2 2 1 20

Base Weight Decay base

3.6

3.8

4.0

4.2

ValidationLoss

SP

Depth L

4 8

16 32 64

128 256

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

2 4 2 3 2 2 2 1 20

Base Weight Decay base

(b) P

- Figure 4 Weight decay transfer of Muon-Kimi-AdamW under SP and µP from Condition 3.1 (k ≥ 2). We train GPT-2 style models with Muon-Kimi and AdamW using SP and µP derived from Condition 3.1 (see Tables 3 and 6). µP enables robust HP transfer across both width and depth scaling, while consistently achieving lower loss than SP as the model depth increases. The corresponding numerical values are reported in Appendix E.3.2.

###### E.3.1 Basic Experimental Setup

Unless otherwise stated, all HP-transfer experiments use GPT-2 style models trained on OpenWebText with sequence length 1024 and the GPT-2 tokenizer, following the setup in Section 5. The base model has width nbase = 256 and depth Lbase = 4. For width-scaling experiments, we vary n while keeping L = 4; for depth-scaling experiments, we vary L while keeping n = 256. The base initialization variance for matrix weights is set to 0.022, and biases are initialized to zero.

All models are trained with batch size 240 for 1221 iterations, corresponding to about 300M tokens, using 120 warmup iterations followed by cosine learning-rate decay to a minimum learning rate of 3 × 10−5, gradient clipping of 1.0, and the optimizer-specific settings described below.

The HPs shown in the tables are base HPs, such as ηbase or λbase; the actual model HPs are obtained by applying the corresponding SP or µP scaling rules. The µP implementation can be found in Tables 2 and 8 for Condition 3.1 (k ≥ 2) and Condition B.1 (k = 1), respectively. We report the final validation loss.

###### E.3.2 Additional Details of Muon-Kimi-AdamW

This subsection provides the complete Muon-Kimi-AdamW results underlying Figures 1, 2, 3, and 4. We first report learning-rate transfer under width and depth scaling, including the comparison between Condition 3.1 (k ≥ 2) and Condition B.1 (k = 1). We then report the no-LayerNorm diagnostic and the weight-decay transfer experiments.

Experimental setup. Following common practice [26], hidden matrix parameters are optimized by Muon-Kimi with Nesterov-style momentum 0.95, while all other parameters (e.g., embeddings, LM head, and biases) are

- Table 15 For Muon-Kimi-AdamW, SP fails to transfer the optimal base learning rate across widths. This table reports the numerical results corresponding to Figure 1, where the best validation loss for each width is highlighted in bold.

n/log2(ηbase) -10 -9 -8 -7 -6 -5

128 5.127 4.685 4.45 4.373 4.364 4.372 256 4.552 4.219 4.081 4.053 4.062 4.091 512 4.093 3.886 3.819 3.833 3.837 4.062

1024 3.817 3.699 3.672 3.68 3.952 5.603 2048 3.654 3.571 3.555 3.798 5.472 6.438 4096 3.56 3.516 3.747 5.557 6.159 6.552

- Table 16 For Muon-Kimi-AdamW, µP from Condition 3.1 (k ≥ 2) approximately transfers the optimal base learning rate across widths, and achieves lower loss than SP as the width increases. This table reports the numerical results corresponding to Figure 1, where the best validation loss for each width is highlighted in bold.

n/log2(ηbase) -10 -9 -8 -7 -6 -5

128 4.875 4.53 4.42 4.374 4.383 4.397 256 4.561 4.227 4.081 4.059 4.079 4.104 512 4.305 3.974 3.83 3.811 3.828 3.873

1024 4.125 3.798 3.654 3.646 3.676 3.726 2048 3.957 3.636 3.516 3.515 3.552 3.689 4096 3.882 3.531 3.446 3.461 3.523 3.752

optimized by AdamW with β1 = 0.9, β2 = 0.95, and ϵbase = 10−16. For learning-rate transfer experiments, both Muon-Kimi and AdamW use the same base learning rate ηbase, and weight decay is disabled. For weight-decay transfer experiments, we fix the base learning rate to 2−7 and sweep the base weight decay λbase.

Additional results of width-wise learning rate transfer. Complete numerical results of base learning rate transferability across different widths are presented in Table 15 and Table 16 for SP and µP from Condition 3.1 (k ≥ 2), respectively.

Additional results of depth-wise learning rate transfer with LayerNorm. With LayerNorm, complete numerical results of base learning rate transferability across different depths are presented in Table 17, Table 18, and Table 19 for SP, µP from Condition 3.1 (k ≥ 2), and µP from Condition B.1 (k = 1), respectively. As discussed in Section 5, this apparent depth-wise transfer under SP should be interpreted cautiously, because LayerNorm and the tested depth range can partially mask feature-scale instability.

Additional results of depth-wise learning rate transfer without LayerNorm. Without LayerNorm, depth-wise base learning-rate transfer results for SP and µP from Condition 3.1 (k ≥ 2) are shown in Figure 3, with complete numerical results in Tables 20 and 21.

Additional results of width-wise weight decay transfer. Width-wise base weight-decay transfer results are shown in Figure 4(a). Complete numerical results are presented in Table 22 and Table 23 for SP and µP from Condition 3.1 (k ≥ 2), respectively.

- Table 17 For Muon-Kimi-AdamW, with LayerNorm, SP transfers the optimal base learning rate across the tested depths. This table reports the numerical results corresponding to Figure 1, where the best validation loss for each depth is highlighted in bold.

L/log2(ηbase) -9 -8 -7 -6 -5 4 4.219 4.081 4.056 4.067 4.09 8 4.109 3.985 3.952 3.973 4.013 16 4.016 3.893 3.864 3.889 3.929 32 3.949 3.824 3.799 3.82 3.885 64 3.916 3.777 3.747 3.777 3.91 128 3.898 3.75 3.723 3.772 4.031 256 3.883 3.719 3.688 3.753 4.174

- Table 18 For Muon-Kimi-AdamW, with LayerNorm, µP from Condition 3.1 (k ≥ 2) transfers the optimal base learning rate across depths, and achieves lower loss than SP as the depth increases. This table reports the numerical results corresponding to Figure 1, where the best validation loss for each depth is highlighted in bold.

L/log2(ηbase) -9 -8 -7 -6 -5

4 4.228 4.081 4.06 4.075 4.098 8 4.089 3.972 3.938 3.957 3.988

16 4.01 3.886 3.85 3.874 3.907 32 3.96 3.826 3.8 3.828 3.879 64 3.917 3.771 3.747 3.796 3.942

128 3.878 3.715 3.694 3.754 4.002 256 3.878 3.697 3.678 3.761 3.964

Additional results of depth-wise weight decay transfer with LayerNorm. With LayerNorm, depth-wise base weight-decay transfer results are shown in Figure 4(b). Complete numerical results are presented in Table 24 and Table 25 for SP and µP from Condition 3.1 (k ≥ 2), respectively.

- Table 19 For Muon-Kimi-AdamW, with LayerNorm, µP from Condition B.1 (k = 1) fails to transfer the optimal base learning rate across depths. This table reports the numerical results corresponding to Figure 2, where the best validation loss for each depth is highlighted in bold. The implementation can be found in Table 9 and 12.

L/log2(ηbase) -9 -8 -7 -6 -5 -4 4 4.225 4.081 4.06 4.067 4.095 4.971 8 4.149 3.988 3.923 3.934 3.951 4.03 16 4.198 3.963 3.871 3.857 3.88 3.922 32 4.325 3.987 3.843 3.796 3.814 3.855 64 4.517 4.052 3.825 3.728 3.725 3.768

128 4.666 4.242 3.912 3.745 3.7 3.746 256 4.774 4.454 4.025 3.788 3.667 3.681

- Table 20 For Muon-Kimi-AdamW, without LayerNorm, SP fails to preserve stable training. NaN data points indicate training instability, where the loss explodes.

L/log2(ηbase) -13 -12 -11 -10 -9

4 7.318 6.394 5.784 5.169 13.77 8 6.775 5.974 5.426 4.811 NaN 16 6.115 5.631 5.052 4.409 10.814 32 5.809 5.328 4.706 4.233 4.282 64 5.519 5.038 4.516 4.189 7.251

128 5.316 4.896 4.484 4.313 NaN 256 5.179 4.867 4.678 5.752 NaN

- Table 21 For Muon-Kimi-AdamW, without LayerNorm, µP from Condition 3.1 (k ≥ 2) has stable runs and approximately transfers the optimal base learning rate at large depth L ≥ 32. NaN data points indicate training instability, where the loss explodes. The best validation loss for each depth is highlighted in bold.

L/log2(ηbase) -11 -10 -9 -8 -7 4 5.791 5.169 11.6 NaN 345.85 8 5.741 5.084 131.43 NaN NaN 16 5.73 5.059 8.732 246.45 122.99 32 5.734 5.069 4.275 3.964 3.894 64 5.73 5.051 4.253 3.912 3.815

128 5.728 5.052 4.214 3.862 3.742 256 5.733 5.045 4.217 3.859 3.724

- Table 22 For Muon-Kimi-AdamW, SP fails to transfer the optimal base weight decay across widths. This table

- reports the numerical results corresponding to Figure 4, where the best validation loss for each width is highlighted in bold.

n/log2(λbase) -4 -3 -2 -1 0

128 4.361 4.359 4.375 4.478 4.975 256 4.026 4.019 4.013 4.061 4.327 512 3.809 3.79 3.77 3.789 3.971

1024 3.642 3.614 3.585 3.582 3.724 2048 3.574 3.514 3.467 3.45 3.545 4096 4.711 4.032 3.478 3.406 3.444

- Table 23 For Muon-Kimi-AdamW, µP from Condition 3.1 (k ≥ 2) approximately transfers the optimal base weight decay across widths. This table reports the numerical results corresponding to Figure 4, where the best validation loss for each width is highlighted in bold.

n/log2(λbase) -4 -3 -2 -1 0 128 4.359 4.35 4.352 4.406 4.74 256 4.039 4.029 4.033 4.073 4.357 512 3.785 3.767 3.766 3.815 4.109

1024 3.622 3.617 3.602 3.659 3.966 2048 3.493 3.48 3.475 3.539 3.847 4096 3.412 3.398 3.392 3.463 3.782

- Table 24 This table reports the numerical results corresponding to depth scaling of Muon-Kimi-AdamW under SP in Figure 4, where the best validation loss for each depth is highlighted in bold.

L/log2(λbase) -4 -3 -2 -1 0

4 4.03 4.021 4.017 4.062 4.328 8 3.919 3.911 3.905 3.965 4.314

16 3.842 3.832 3.832 3.905 4.259 32 3.784 3.773 3.781 3.876 4.246 64 3.734 3.726 3.735 3.862 4.244

128 3.675 3.661 3.68 3.831 4.242 256 3.665 3.651 3.68 3.847 4.289

- Table 25 For Muon-Kimi-AdamW, µP from Condition 3.1 (k ≥ 2) approximately transfers the optimal base weight decay across depths, and achieves lower loss than SP as the depth increases. This table reports the

- numerical results corresponding to Figure 4, where the best validation loss for each depth is highlighted in bold.

L/log2(λbase) -4 -3 -2 -1 0

4 4.041 4.037 4.033 4.071 4.343 8 3.905 3.896 3.89 3.954 4.267

16 3.834 3.824 3.825 3.906 4.232 32 3.776 3.765 3.77 3.863 4.192 64 3.699 3.686 3.693 3.807 4.143

128 3.669 3.657 3.669 3.794 4.137 256 3.626 3.61 3.634 3.769 4.109

###### (a) P

###### SP

23

| | | | |
|---|---|---|---|
| | | | |
| | | | |

Step

- 5

- 6

- 7

- 8

- 9

- 10

21

hLR

2 1

2 3

28 210 212

28 210 212

Width n

Width n

###### (b) P

SP

25

|Step| | | | | | |
|---|---|---|---|---|---|---|
| |5| | | | | |
| |6<br>7<br>| | | | | |
| |8<br>9<br>10<br>| | | | | |
| | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

23

hLR

21

2 1

2 3

22 24 26 28

22 24 26 28

Depth L

Depth L

(c) P

SP

| |
|---|

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Width n

ValidationLoss

4.5

128 256 512

1024 2048 4096

4.0

3.5

2 7 2 6 2 5 2 4 2 3

2 7 2 6 2 5 2 4 2 3

Base Learning Rate base

Base Learning Rate base

###### (d) P

SP

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Depth L

4.4

ValidationLoss

4 8

4.2

16 32 64

4.0

128 256

3.8

2 7 2 6 2 5 2 4 2 3

2 7 2 6 2 5 2 4 2 3

Base Learning Rate base

Base Learning Rate base

- Figure 5 Feature learning and HP transfer of Muon-AdamW under SP and µP. We train GPT-2 style models with Muon-AdamW using SP and µP derived from Condition 3.1 (see Tables 4 and 6). µP maintains stable feature norms and enables robust HP transfer across both width and depth scaling, while generally achieving lower loss than

- SP as the model size increases. The detailed numerical values are provided in Appendix E.3.3.

- Table 26 For Muon-AdamW, SP fails to transfer the optimal base learning rate across widths. This table

- reports the numerical results corresponding to Figure 5, where the best validation loss for each width is highlighted in bold.

n/log2(ηbase) -7 -6 -5 -4 -3

128 4.88 4.544 4.416 4.377 4.375 256 4.38 4.168 4.061 4.067 4.055 512 4.054 3.888 3.831 3.819 3.818

1024 3.848 3.711 3.655 3.648 3.923 2048 3.742 3.595 3.532 3.553 5.137 4096 3.735 3.586 3.543 3.795 5.992

###### E.3.3 Additional Details of Muon-AdamW

This subsection provides the complete Muon-AdamW results underlying Figures 5 and 2. We report learningrate transfer under width and depth scaling, including the comparison between SP, Condition 3.1 (k ≥ 2) and Condition B.1 (k = 1).

Experimental setup. Following common practice [20, 21], hidden matrix parameters are optimized by Muon with a base learning rate of ηbase, and a Nesterov-style momentum of 0.95, while all other parameters (e.g., embedding layer, LM head, all biases) are updated by AdamW with a base learning rate of ηbase/10, β1 = 0.9,

- β2 = 0.95, ϵbase = 10−16. We do not use weight decay in all learning rate transfer experiments.

Additional results of width-wise learning rate transfer. Complete numerical results of base learning rate transferability across different widths are presented in Table 26 and Table 27 for SP and µP from Condition 3.1 (k ≥ 2), respectively.

Additional results of depth-wise learning rate transfer. Complete numerical results of base learning rate transferability across different depths are presented in Table 28, Table 29, and Table 30 for SP, µP from

###### Table 27 For Muon-AdamW, µP from Condition 3.1 (k ≥ 2) approximately transfers the optimal base learning rate across widths, and achieves lower loss than SP as the width increases. This table reports the numerical results corresponding to Figure 5, where the best validation loss for each width is highlighted in bold.

n/log2(ηbase) -7 -6 -5 -4 -3

128 4.671 4.473 4.385 4.385 4.387 256 4.403 4.187 4.076 4.073 4.077 512 4.169 3.947 3.834 3.823 3.825

1024 4.02 3.75 3.677 3.668 3.682 2048 3.868 3.624 3.548 3.571 3.669 4096 3.771 3.537 3.486 3.532 4.672

###### Table 28 For Muon-AdamW, SP shifts the optimal base learning rate across depths. This table reports the

- numerical results corresponding to Figure 5, where the best validation loss for each depth is highlighted in bold.

L/log2(ηbase) -7 -6 -5 -4 -3

4 4.381 4.169 4.064 4.062 4.055 8 4.267 4.059 3.968 3.956 3.961

16 4.201 3.973 3.893 3.883 3.925 32 4.153 3.917 3.841 3.857 4.047 64 4.124 3.884 3.804 3.868 4.772

128 4.084 3.835 3.766 3.886 5.798 256 4.099 3.841 3.78 3.92 6.705

Condition 3.1 (k ≥ 2), and µP from Condition B.1 (k = 1), respectively.

- Table 29 For Muon-AdamW, µP from Condition 3.1 (k ≥ 2) approximately transfers the optimal base learning rate across depths, and achieves lower loss than SP as the depth increases. This table reports the numerical results corresponding to Figure 5, where the best validation loss for each depth is highlighted in bold.

L/log2(ηbase) -7 -6 -5 -4 -3

4 4.402 4.187 4.082 4.072 4.075 8 4.259 4.044 3.961 3.946 3.981

16 4.198 3.972 3.888 3.888 3.95 32 4.152 3.91 3.832 3.854 3.995 64 4.09 3.842 3.767 3.849 4.223

128 4.076 3.819 3.753 3.869 4.415 256 4.06 3.789 3.733 3.858 5.201

- Table 30 For Muon-AdamW, µP from Condition B.1 (k = 1) fails to transfer the optimal base learning rate across depths. This table reports the numerical results corresponding to Figure 2, where the best validation loss for each depth is highlighted in bold. The implementation can be found in Table 10 and 12.

L/log2(ηbase) -7 -6 -5 -4 -3 -2

4 4.402 4.189 4.078 4.07 4.073 4.068 8 4.323 4.067 3.97 3.937 3.96 3.983

16 4.368 4.052 3.909 3.862 3.881 3.947 32 4.477 4.073 3.884 3.809 3.823 3.902 64 4.649 4.12 3.874 3.754 3.735 3.834

128 4.793 4.262 3.946 3.78 3.722 3.795 256 4.887 4.447 4.031 3.815 3.707 3.72

###### (a) P

###### SP

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

Step

21

- 5

- 6

- 7

- 8

- 9

- 10

hLR

2 1

2 3

27 28 29 210 211

27 28 29 210 211

Width n

Width n

###### (b) P

SP

|Step<br><br>5| | | |
|---|---|---|---|
|6<br><br>7<br><br>8<br>| | | |
|9 1<br><br>|0| | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

22

20

hLR

2 2

2 4

2 6

22 24 26

22 24 26

Depth L

Depth L

(c) P

SP

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Width n

5.0

ValidationLoss

128 256 512

4.5

1024 2048 4096

4.0

3.5

2 8 2 7 2 6 2 5 2 4

2 8 2 7 2 6 2 5 2 4

Base Learning Rate base

Base Learning Rate base

###### (d) P

| |
|---|

SP

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

Depth L

ValidationLoss

4 8

5.0

16 32 64

4.5

128 256

4.0

2 7 2 6 2 5 2 4 2 3

2 7 2 6 2 5 2 4 2 3

Base Learning Rate base

Base Learning Rate base

- Figure 6 Feature learning and HP transfer of Shampoo-AdamW under SP and µP. We train GPT-2 style models with Shampoo-AdamW using SP and µP derived from Condition 3.1 (see Tables 4 and 6). µP maintains stable feature norms and enables robust HP transfer across both width and depth scaling, while generally achieving lower loss than

- SP as the model size increases. The detailed numerical values are provided in Appendix E.3.4.

2 7 2 6 2 5 2 4 2 3

Base Learning Rate base

4.00

4.25

4.50

4.75

5.00

ValidationLoss

Shampoo-AdamW, P (k = 1)

Depth L

4 8

16 32 64

128 256

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

2 7 2 6 2 5 2 4 2 3

Base Learning Rate base

Shampoo-AdamW, P (k 2)

- Figure 7 Validating the role of residual block depth k. We compare two µP implementations for GPT-2-style models trained with Shampoo-AdamW: Depth-µP-style formulation from Condition B.1 (k = 1) and CompleteP-style formulation from Condition 3.1 (k ≥ 2). Condition 3.1 yields more stable HP transfer and lower validation loss, empirically supporting it as the appropriate µP condition for architectures with multi-transformation residual branches. The numerical values are provided in Appendix E.3.4.

###### E.3.4 Additional Details of Shampoo-AdamW

This subsection provides the complete Shampoo-AdamW results underlying Figures 6 and 7. We report learning-rate transfer under width and depth scaling, including the comparison between SP, Condition 3.1 (k ≥ 2) and Condition B.1 (k = 1).

Experimental setup. Following common practice [20], hidden matrix parameters are optimized by Shampoo with a base learning rate of ηbase, β1 = 0.95, β2 = 0.95, a shampoo precondition frequency of 1, a maximal precondition dimension of 20000, ϵbase = 10−8 for width scaling, and ϵbase = 10−5 for depth scaling, while all other parameters (e.g., embedding layer, LM head, all biases) are updated by AdamW with a base learning rate of 2ηbase, β1 = 0.9, β2 = 0.95, and ϵbase = 10−16. We do not use weight decay in all learning rate transfer experiments.

Additional results of width-wise learning rate transfer. Complete numerical results of base learning rate transferability across different widths are presented in Table 31 and Table 32 for SP and µP from Condition 3.1 (k ≥ 2), respectively.

- Table 31 For Shampoo-AdamW, SP fails to transfer the optimal base learning rate across widths. This table

- reports the numerical results corresponding to Figure 6, where the best validation loss for each width is highlighted in bold.

n/log2(ηbase) -8 -7 -6 -5 -4

128 4.925 4.749 4.614 4.516 4.521 256 4.688 4.407 4.294 4.239 4.236 512 4.481 4.242 4.147 4.084 4.097

1024 4.385 4.229 4.059 3.992 4.067 2048 4.338 4.205 4.042 3.981 6.002 4096 4.366 4.247 4.148 4.604 7.476

- Table 32 For Shampoo-AdamW, µP from Condition 3.1 (k ≥ 2) transfers the optimal base learning rate across widths, and achieves lower loss than SP as the width increases. This table reports the numerical results corresponding to Figure 6, where the best validation loss for each width is highlighted in bold.

n/log2(ηbase) -8 -7 -6 -5 -4

128 4.969 4.796 4.61 4.548 4.552 256 4.689 4.404 4.291 4.246 4.249 512 4.44 4.159 4.057 4.012 4.012

1024 4.272 3.995 3.919 3.874 3.887 2048 4.131 3.873 3.787 3.754 3.771 4096 3.995 3.761 3.699 3.658 3.694

Additional results of depth-wise learning rate transfer. Complete numerical results of base learning rate transferability across different depths are presented in Table 33, Table 34, and Table 35 for SP, µP from Condition 3.1 (k ≥ 2), and µP from Condition B.1 (k = 1), respectively.

- Table 33 For Shampoo-AdamW, SP yields increasing validation loss under depth scaling. This table reports the numerical results corresponding to Figure 6.

L/log2(ηbase) -7 -6 -5 -4 -3 4 4.696 4.617 4.503 4.425 4.62 8 4.721 4.638 4.511 4.421 4.634 16 4.728 4.662 4.61 4.491 4.811 32 4.783 4.754 4.72 4.704 4.966 64 4.886 4.855 4.851 4.802 5.131

128 5.033 5.023 4.994 4.983 5.276 256 5.281 5.27 5.207 5.183 5.945

- Table 34 For Shampoo-AdamW, µP from Condition 3.1 (k ≥ 2) approximately transfers the optimal base learning rate across depths, and achieves lower loss than SP as the depth increases. This table reports the

- numerical results corresponding to Figure 6, where the best validation loss for each depth is highlighted in bold.

L/log2(ηbase) -7 -6 -5 -4 -3

4 4.696 4.605 4.511 4.487 4.623 8 4.567 4.391 4.335 4.339 4.535

16 4.544 4.293 4.265 4.273 4.415 32 4.451 4.267 4.172 4.166 4.373 64 4.407 4.198 4.134 4.109 4.625

128 4.381 4.176 4.112 4.163 5.603 256 4.371 4.159 4.1 4.379 5.601

- Table 35 For Shampoo-AdamW, µP from Condition B.1 (k = 1) gives weaker depth scaling and higher validation loss than µP from Condition 3.1 (k ≥ 2). This table reports the numerical results corresponding to Figure 7.

L/log2(ηbase) -7 -6 -5 -4 -3

4 4.704 4.613 4.534 4.471 4.616 8 4.693 4.472 4.383 4.349 4.548

16 4.749 4.56 4.337 4.314 4.425 32 4.82 4.597 4.439 4.277 4.377 64 4.913 4.651 4.457 4.314 4.433

128 5.045 4.787 4.575 4.428 4.493 256 5.128 4.918 4.693 4.503 4.57

###### (a) P

###### SP

| | | | |
|---|---|---|---|
|Step<br><br>5| | | |
|6<br><br>7<br>| || |
|---|
<br><br>| |
|8<br><br>9<br><br>10<br>|| |
|---|
| |
<br><br>| | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

211

28

hLR

25

22

2 1

28 210 212

28 210 212

Width n

Width n

###### (b) P

SP

| | | | |
|---|---|---|---|
| | | | |
| | | | |
|| |
|---|
| |
| |
| |
| |
<br><br>| | | |
| | || |
|---|
<br><br>|| |
|---|
<br><br>|

Step

28

- 5

- 6

- 7

- 8

- 9

- 10

26

hLR

24

| |
|---|

| |
|---|

| |
|---|

22

| |
|---|

20

22 24 26

22 24 26

Depth L

Depth L

(c) P

SP

7.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

Width n

ValidationLoss

128 256 512

6.0

1024 2048 4096

5.0

4.0

2 16 2 15 2 14 2 13 2 12 2 11

2 15 2 14 2 13 2 12 2 11 2 10

Base Learning Rate base

Base Learning Rate base

###### (d) P

SP

7.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

Depth L

ValidationLoss

6.5

4 8

6.0

16 32 64

5.5

128 256

5.0

2 15 2 14 2 13 2 12 2 11

2 15 2 14 2 13 2 12 2 11

Base Learning Rate base

Base Learning Rate base

- Figure 8 Feature learning and HP transfer of Sophia under SP and µP. We train GPT-2 style models with Sophia using SP and µP derived from Condition 3.1 (see Table 6). µP maintains stable feature norms and enables robust HP transfer across both width and depth scaling. The detailed numerical values are provided in Appendix E.3.5.

2 15 2 14 2 13 2 12 2 11 2 10 2 9

Base Learning Rate base

5.0

5.5

6.0

6.5

7.0

ValidationLoss

Sophia, P (k = 1)

Depth L

4 8

16 32 64

128 256

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

2 15 2 14 2 13 2 12 2 11

Base Learning Rate base

Sophia, P (k 2)

- Figure 9 Validating the role of residual block depth k. We compare two µP implementations for GPT-2-style models trained with Sophia: Depth-µP-style formulation from Condition B.1 (k = 1) and CompleteP-style formulation from Condition 3.1 (k ≥ 2). Condition 3.1 yields more stable HP transfer, empirically supporting it as the appropriate µP condition for architectures with multi-transformation residual branches. The numerical values are provided in Appendix E.3.5.

###### E.3.5 Additional Details of Sophia

This subsection provides the complete Sophia results underlying Figures 8 and 9. We report learning-rate transfer under width and depth scaling, including the comparison between SP, Condition 3.1 (k ≥ 2) and Condition B.1 (k = 1).

Experimental setup. Following common practice [25], parameters are updated by Sophia with a base learning rate of ηbase, β1 = 0.965, β2 = 0.99, ρ = 0.05, and a Hessian update frequency of 10.

Additional results of width-wise learning rate transfer. Completed numerical results of base learning rate transferability across different widths are presented in Table 36 and Table 37 for SP and µP from Condition 3.1 (k ≥ 2), respectively.

Additional results of depth-wise learning rate transfer. Complete numerical results of base learning rate transferability across different depths are presented in Table 38, Table 39, and Table 40 for SP, µP from Condition 3.1 (k ≥ 2), and µP from Condition B.1 (k = 1), respectively.

- Table 36 For Sophia, SP fails to transfer the optimal base learning rate across widths. This table reports the numerical results corresponding to Figure 8, where the best validation loss for each width is highlighted in bold.

n/log2(ηbase) -16 -15 -14 -13 -12 -11

128 7.485 6.982 6.43 5.935 5.54 5.648 256 6.293 5.97 5.603 5.232 5.143 5.361 512 5.506 5.257 4.946 5.005 4.986 5.28 1024 4.924 4.682 4.536 4.48 4.708 5.988 2048 4.457 4.289 4.157 4.249 5.651 6.169 4096 4.321 4.209 4.413 5.677 5.964 6.101

- Table 37 For Sophia, µP from Condition 3.1 (k ≥ 2) approximately transfers the optimal base learning rate across widths larger than 256. This table reports the numerical results corresponding to Figure 8, where the best validation loss for each width is highlighted in bold.

n/log2(ηbase) -15 -14 -13 -12 -11 -10

128 6.133 5.788 5.446 5.474 5.571 5.781 256 5.969 5.613 5.235 5.143 5.362 5.756 512 5.903 5.521 5.124 4.955 5.113 5.707

1024 5.863 5.455 5.055 4.98 4.976 5.598 2048 5.827 5.419 5.01 4.932 4.919 5.555 4096 5.833 5.406 4.991 4.831 4.88 5.419

- Table 38 For Sophia, SP yields increasing validation loss under depth scaling. This table reports the numerical results corresponding to Figure 8.

L/log2(ηbase) -15 -14 -13 -12 -11

4 5.967 5.61 5.235 5.249 5.314 8 5.902 5.547 5.189 4.999 5.315

16 5.881 5.541 5.226 5.263 5.76 32 5.894 5.58 5.267 5.525 5.737 64 5.922 5.65 5.402 6.047 6.29 128 6.041 5.731 5.492 5.95 6.389 256 6.087 5.822 6.892 6.498 6.51

- Table 39 For Sophia, µP from Condition 3.1 (k ≥ 2) approximately transfers the optimal base learning rate across depths, and achieves lower loss than SP as the depth increases. This table reports the numerical results corresponding to Figure 8, where the best validation loss for each depth is highlighted in bold.

L/log2(ηbase) -15 -14 -13 -12 -11

4 5.966 5.603 5.242 5.319 5.375 8 5.895 5.544 5.178 4.935 5.225

16 5.861 5.532 5.19 4.994 5.362 32 5.812 5.497 5.162 4.906 5.237 64 5.837 5.527 5.238 5.154 5.402

128 5.861 5.599 5.329 5.384 5.915 256 5.841 5.537 5.271 5.702 5.709

- Table 40 For Sophia, µP from Condition B.1 (k = 1) fails to transfer the optimal base learning rate across depths. This table reports the numerical results corresponding to Figure 9, where the best validation loss for each depth is highlighted in bold. The implementation can be found in Table 12.

L/log2(ηbase) -15 -14 -13 -12 -11 -10 -9

4 5.969 5.609 5.234 5.329 5.356 5.938 6.264 8 5.966 5.612 5.231 4.936 5.212 5.515 6.252

16 5.99 5.67 5.302 5.052 4.939 5.438 6.061 32 6.028 5.73 5.402 5.095 5.026 5.014 5.97 64 6.051 5.785 5.515 5.231 5.072 5.077 5.218 128 6.066 5.83 5.609 5.39 5.252 5.118 5.359 256 6.078 5.845 5.663 5.5 5.434 5.403 6.009

### Appendix F Justification of Upper Bound Estimation

In the derivation in Section 3, we rely on the assumption that the subadditivity and submultiplicativity inequalities used throughout the analysis are tight under standard neural network initialization and training dynamics. Under this assumption, controlling the upper bounds of ∥hl(x)∥R and ∥∆hl(x)∥R is sufficient to characterize the typical scaling behavior of ∥hl(x)∥R and ∥∆hl(x)∥R themselves, up to constant factors. In this section, we provide a more concrete justification for the validity of this assumption.

#### F.1 Subadditivity Inequalities

Subadditivity inequalities are used in the derivation of the update conditions to control the norm of the accumulated feature update. For instance, by decomposing ∆hs(x) into several layerwise contributions, we obtain

s

s

αlWl(2)∆Wl(1)(hl−1(x) + ∆hl−1(x))

αlWl(2)Wl(1)∆hl−1(x)

+

∆hs(x) = ∆h0(x) +

l=1

l=1

ϵ0(s)

ϵ(1)1 (s)

s

s

αl∆Wl(2)Wl(1)(hl−1(x) + ∆hl−1(x))

αl∆Wl(2)∆Wl(1)(hl−1(x) + ∆hl−1(x))

+

+

l=1

l=1

ϵ2(s)

ϵ(2)1 (s)

(26)

which leads to the upper bound in Equation (5):

∥∆hL(x)∥R ≤ ∥∆h0(x)∥R + ∥ϵ0(L)∥R + ∥ϵ(1)1 (L)∥R + ∥ϵ(2)1 (L)∥R + ∥ϵ2(L)∥R. A similar subadditivity argument is further applied to each term, e.g.,

∥ϵ(1)1 (L)∥R ≤

L

αl Wl(2)∆Wl(1) hl−1(x) + ∆hl−1(x) R.

l=1

In principle, such subadditivity bounds may be loose when the summands point in largely different or canceling directions. However, due to the chain rule in backpropagation, the parameter updates {∆Wl}Ll=1 across different layers are strongly correlated (e.g., see discussion in Dey et al. [10], Yang et al. [46]). More precisely, each ∆Wl is proportional to the product of a forward feature hl−1(x) and a backpropagated error signal, which itself is obtained by repeatedly multiplying upstream Jacobians. As a result, the layerwise update contributions to ∆hL(x) share similar directions in feature space rather than behaving as independent or adversarial vectors.

Consequently, the terms appearing in the sums defining ϵ0(L), ϵ(1)1 (L), and ϵ(2)1 (L) tend to be positively aligned, and cancellations between different layers are atypical [10, 46]. In this regime, the norm of the sum scales proportionally to the sum of the norms, implying that the subadditivity inequality provides an accurate characterization of the magnitude of ∆hL(x) up to constant factors. Therefore, under standard training dynamics, controlling the subadditive upper bounds suffices to capture the typical scaling behavior of the feature updates.

#### F.2 Submultiplicativity Inequalities

Submultiplicativity inequalities are extensively used in the analysis of both the initial condition and the update condition. In this section, we discuss these two scenarios separately and clarify why the resulting upper bounds are typically tight under standard neural network initialization and training dynamics. Our reasoning is closely aligned with that of Yang et al. [45], which employs a similar perspective in deriving spectral conditions for width scaling.

###### F.2.1 Initalization Condition

In the derivation of the initialization conditions, submultiplicativity inequalities are applied to the input, hidden, and output layers. For the input and output layers, the analysis is the same as for the width-scaling setting, since each involves a single linear transformation (e.g., we used ∥h0(x)∥R ≤ α0∥W0∥R∥x∥R). Accordingly, the tightness of the corresponding bounds directly follows from Claim 1 in Yang et al. [45]. In contrast, the hidden layers in our setting require additional justification, since each residual block consists of two or more stacked linear transformations rather than a single mapping (e.g., we used ∥αlWl(2)Wl(1)hl−1(x)∥R ≤ αl∥Wl(2)∥R∥Wl(1)∥R∥hl−1(x)∥R). In what follows, we therefore focus on establishing the tightness of the submultiplicativity bounds for these multi-layer residual blocks.

- Claim F.1 (Alignment of initial weight matrices). Fix a feature vector hl−1(x) ∈ Rn. Recall that Wl(1) ∈

Rn

l×n,Wl(2) ∈ Rn×n

l are initialized with Wl(1)

ij

, Wl(2)

ij

i.∼i.d. N(0,σl2). Provided that nl = Θ(n), then with high probability:

∥Wl(2)Wl(1)hl−1(x)∥R = Θ ∥Wl(2)∥R∥Wl(1)∥R∥hl−1(x)∥R , which means that the submultiplicativity inequalities used in the initialization regime are tight.

Proof. We first consider the intermediate feature zl := Wl(1)hl−1(x) ∈ Rn

l. Since Wl(1) has i.i.d. Gaussian entries with zero mean and variance σl2, by the law of large numbers, we have

∥zl∥R = ∥Wl(1)hl−1(x)∥R ≈ σl√n∥hl−1(x)∥R.

In the meanwhile, by the standard concentration inequalities for random matrices [37] we have, with high probability that ∥Wl(1)∥R = n/nl · σl(√n + √nl) = Θ(σl√n). Therefore, we obtain

∥zl∥R = ∥Wl(1)hl−1(x)∥R = Θ(∥Wl(1)∥R∥hl−1(x)∥R). (27)

Next, we apply Wl(2) to zl. Again, Wl(2) is an i.i.d. Gaussian matrix with variance σl2, so by the law of large numbers, we have

∥Wl(2)zl∥R ≈ σl√nl∥zl∥R. As well, by the standard concentration inequalities for random matrices [37] we have, with high probability that ∥Wl(2)∥R = nl/n = σl(√n + √nl) = Θ(σl√nl). Therefore, we obtain

∥Wl(2)Wl(1)hl−1(x)∥R = ∥Wl(2)zl∥R = Θ(∥Wl(2)∥R∥zl∥R) = Θ(∥Wl(1)∥R∥Wl(1)∥R∥hl−1(x)∥R),

which shows that the submultiplicativity ∥αlWl(2)Wl(1)hl−1(x)∥R ≤ αl∥Wl(2)∥R∥Wl(1)∥R∥hl−1(x)∥R used to derive the initial condition is tight.

| |
|---|

F.2.2 Update Condition

We now justify the use of submultiplicativity inequalities in the update regime and argue that the resulting upper bounds on ∥∆hL(x)∥ are tight in terms of scaling. For the input and output layers, the analysis is identical to that in the width-scaling regime. As a consequence, the tightness of the corresponding bounds follows directly from Claim 2 in Yang et al. [45]. In contrast, the hidden layers in our setting require additional justification, as each residual block consists of multiple stacked linear transformations and gives rise to a more involved update structure due to the presence of residual connections. Analogous to Claim 2 in Yang et al. [45], we therefore begin by establishing the following observation.

- Claim F.2 (Alignment of updates). For any l ∈ [L], an update ∆Wl(2) given by gradient descent with batch size 1, we have

∥∆Wl(2)Wl(1)hl−1(x)∥R = Θ ∥∆Wl(2)∥R∥Wl(1)∥R∥hl−1(x)∥R .

Proof. By the chain rule, we can write ∆Wl(2) as

∆Wl(2) = −ηl(2)∇hl(x)L · (Wl(1)hl−1(x))⊤, which is rank-one and aligns with the incoming feature. Therefore, we have

∥∆Wl(2)Wl(1)hl−1(x)∥R = ∥ηl(2)∇hl(x)L · (Wl(1)hl−1(x))⊤Wl(1)hl−1(x)∥R

= ηl(2)∥∇hl(x)L∥R∥Wl(1)hl−1(x)∥22

nl n · ηl(2)√n∥∇hl(x)L∥R∥Wl(1)hl−1(x)∥2 · ∥Wl(1)hl−1(x)∥R

=

nl n · ηl(2)∥∇hl(x)L∥2∥Wl(1)hl−1(x)∥2 · ∥Wl(1)hl−1(x)∥R

=

nl n · ∥∆Wl(2)∥2 · ∥Wl(1)hl−1(x)∥R

=

= ∥∆Wl(2)∥R · ∥Wl(1)hl−1(x)∥R.

Furthermore, by the initial alignment ∥Wl(1)hl−1(x)∥R = Θ(∥Wl(1)∥R∥hl−1(x)∥R) in Equation (27), we obtain

∥∆Wl(2)Wl(1)hl−1(x)∥R = Θ ∥∆Wl(2)∥R∥Wl(1)∥R∥hl−1(x)∥R , which completes the proof.

| |
|---|

Based on Claim F.2, we demonstrate how the tightness of such submultiplicativity inequality directly leads to a tight upper bound on ∥∆hL(x)∥R in terms of scaling. In particular, the claim ensures that the norms of the layerwise update contributions are accurately captured by their submultiplicative estimates, so that summing these bounds yields an upper bound that faithfully reflects the true magnitude of the accumulated feature update. We can rewrite the expression of the hidden layer update in Equation (26) as

∆hL(x) =

L

αl∆Wl(2)Wl(1)hl−1(x) + ···

l=1

Therefore, as long as the term sl=1 αl∆Wl(2)Wl(1)hl−1(x) does not perfectly cancel with other terms, we have

 ∥

  = Ω

 

 

L

L

αl∆Wl(2)Wl(1)hl−1(x)∥R

αl∥∆Wl(2)Wl(1)hl−1(x)∥R

∥∆hL(x)∥R = Ω

l=1

l=1

 

 

L

αl∥∆Wl(2)∥R∥Wl(1)∥R∥hl−1(x)∥R

= Ω

l=1

= Ω(1), where the second equality uses the tightness of subadditivity inequalities under the principle in Appendix F.1. Therefore, the estimation of ∥∆hL(x)∥R by using submultiplicativity inequalities in Section 3 is tight.

### Appendix G Extension to General Training Settings

As derived in the main text, our theoretical framework primarily investigates a simplified scenario: a one-step update of a linear residual MLP on a single datapoint. In this section, we discuss its extension to the general practical setting: finite multi-step updates of a non-linear residual MLP on a finite batch of datapoints.

This extension relies on three key assumptions, as those justified in the width-scaling literature [45]. Below, we formally restate these assumptions and empirically verify their validity in the width-depth scaling context using the experimental setup detailed in Appendix G.2.

Input Layer (l = 0)

Residual Block Layer 1 (l = L(1))

Residual Block Layer 2 (l = L(2))

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1.00

W+WlRlR

W+WllR

0.75

0.50

Initial (t = 0)

Intermediate (t = T2) Final (t = T)

0.25

0.00

4 8 16 32 64 128 256

4 8 16 32 64 128 256

4 8 16 32 64 128 256

Depth L (log scale)

Depth L (log scale)

Depth L (log scale)

###### Figure 10 Validation of Assumption G.1 (weight update). During the training, the ratio ∥W∥Wl+∆Wl∥R

l∥R+∥∆Wl∥R remains constant near 1 across depth for the input layer and residual block layers, showing non-vanishing updates throughout multiple-step training.

Input Layer (l = 0)

Residual Block Layer 1 (l = L(1))

Residual Block Layer 2 (l = L(2))

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1.00

h+hlRlR

0.75

h+hllR

0.50

Initial (t = 0)

Intermediate (t = T2) Final (t = T)

0.25

0.00

4 8 16 32 64 128 256

4 8 16 32 64 128 256

4 8 16 32 64 128 256

Depth L (log scale)

Depth L (log scale)

Depth L (log scale)

###### Figure 11 Validation of Assumption G.1 (feature update). During the training, the ratio ∥h∥hl+∆hl∥R

l∥R+∥∆hl∥R remains around constant near 1 across varying depths, showing non-vanishing updates throughout multiple-step training.

#### G.1 Assumptions for Extensions

Multi-step Training. In the main text, we derived a parameterization that ensures the weight matrices Wl and their first-step updates ∆Wl (also, ∥hl∥R and ∥∆hl∥R) scale correctly with the model size to achieve feature learning. To ensure these properties hold throughout finite multi-step training, the updated parameters must maintain the same scaling order as the first step. This is formalized in Assumption G.1.

- Assumption G.1 (Non-vanishing update). During finite-step training, we assume the updated weights and feature vectors for any layer l satisfy:

∥Wl + ∆Wl∥R = Θ ∥Wl∥R + ∥∆Wl∥R , ∥hl(x) + ∆hl(x)∥R = Θ ∥hl(x)∥R + ∥∆hl(x)∥R .

In Assumption G.1, the upper bound of the orders of the left-hand side by the right-hand side (or say, O(·)) is guaranteed by the subadditivity. The core constraint is the lower bound of the orders (or say Ω(·)), which implies that the update ∆Wl does not destructively cancel out the existing weight Wl (i.e., the update does not cause the norm to vanish). As discussed in Yang et al. [45], such exact cancellation is extremely rare in practical neural network training. We empirically verify this assumption in Figures 10 and 11, where the norm ratios remain constant across varying depths.

Non-linearity. To extend the analysis to non-linear activations, we substitute the linear transformation Wlhl−1(x) with ϕ Wlhl−1(x) , where ϕ(·) is an activation function (e.g., ReLU). The resulting architecture is as in Equation (28) versus Equation (2) in the linear case. We assume that the activation function preserves the asymptotic order of the feature norms, ensuring that the scaling properties derived for the pre-activations remain valid for the post-activations.

- Assumption G.2 (Stable activation). During the training, we assume that the features before and after the nonlinear activation are of the same scale:

∥ϕ(Wlhl−1(x))∥R = Θ(∥Wlhl−1(x)∥R).

Input Layer (l = 0)

Residual Block Layer 1 (l = L(1))

Residual Block Layer 2 (l = L(2))

1.00

(Wh)ll12

0.75

Whll12

0.50

Initial (t = 0)

Intermediate (t = T2) Final (t = T)

0.25

0.00

4 8 16 32 64 128 256

4 8 16 32 64 128 256

4 8 16 32 64 128 256

Depth L (log scale)

Depth L (log scale)

Depth L (log scale)

###### Figure 12 Validation of Assumption G.2 (stable activation). During the training, the ratio of post-activation to pre-activation norms ∥ϕ(Wlhl−1)∥R

∥Wlhl−1∥R remains stable across varying depths, confirming that the ReLU activation does not collapse the norm in non-linear networks.

Input Layer (l = 0) Initial (t = 0)

Residual Block Layer 1 (l = L(1))

Residual Block Layer 2 (l = L(2))

1.0

0.2

0.2

Intermediate (t = T2) Final (t = T)

(i)(i)Wh2ll1B

(i)Whl2l1

0.5

0.1

0.1

1

0.0

0.0

0.0

4 8 16 32 64 128 256

4 8 16 32 64 128 256

4 8 16 32 64 128 256

Depth L (log scale)

Depth L (log scale)

Depth L (log scale)

###### Figure 13 Validation of Assumption G.3 (per-sample update alignment). We report the averaged ratio of ∥ B1 ∆Wl(i)h(l−i)1∥R

across the batch of data. The values remain stable across varying depths, suggesting that the batch update does not alter the depth-wise scaling of the single-sample update.

∥∆Wlh(l−i)1∥R

Figure 12 empirically verifies this assumption for the ReLU activation, showing that the ratio of post-activation to pre-activation norms is stable across depth.

Training with Mini-batch. Finally, to extend beyond the single-sample setting, we consider updates computed on a batch of data {x(i),y(i)}Bi=1. Let ∆Wl(i) denote the update contribution from the i-th datapoint (e.g., ∆Wl(i) = −ηl∇WlL(x(i),y(i)) for SGD), such that the total batch update is ∆Wl = B1 Bi=1 ∆Wl(i). We expect that the gradient contributions from different samples do not destructively cancel out. This is formalized in Assumption G.3.

- Assumption G.3 (Per-sample update alignment). We assume that the batch size satisfies B = Θ(1), and the batch update norm scales consistently with the per-sample update norm during the training:

∥∆Wlhl−1(x(i))∥R = Θ

1 B ∥∆Wl(i)hl−1(x(i))∥R .

We verify this in Figure 13, where the alignment ratio remains Θ(1), indicating that batch-averaged updates preserve the scaling properties of single-sample updates.

#### G.2 Experimental Details

We conduct simulations to empirically Assumptions G.1–G.3. Our experimental setup largely follows Yang et al. [45], with a different emphasis on depth scaling instead of width scaling. Details are provided below.

Dataset. We construct a binary classification dataset using a subset of CIFAR-10, selecting 100 samples each from the “airplane” and “automobile” classes. The inputs are flattened image vectors in R3072 associated with binary labels in {0,1}.

Architecture and Training. The architecture is a deep residual MLP with ReLU activations, consisting of an input layer, L residual blocks, and a final linear output layer:

h0(x) = α0ϕ(W0x), hl(x) = hl−1(x) + αlϕ Wl(2)ϕ Wl(1)hl−1(x) , l ∈ [L], hL+1(x) = αL+1WL+1hL(x),

(28)

where ϕ denotes the ReLU activation. The dimensions are set as: input dimension d0 = 3072, model width n = 256, residual block width nl = n, and output dimension dL+1 = 1. This aligns well with the simplified setup discussed in the main text (Section 3.1). Models are trained to minimize the binary cross-entropy loss using full-batch Gradient Descent (GD) for T = 200 steps.

Parameterization. We implement the width-depth µP parametrization for SGD derived in Table 5 of Appendix C.5 as follows:

αbase L

αbase n

α0 = αbase, αl =

, αL+1 =

,

σbase2 n

σbase2 d0

, σl2 =

, σL2+1 = σbase2 ,

σ02 =

η0 = ηbasen, ηl = ηbaseL, ηL+1 = ηbasen, with base constants set to:

αbase = 1, σbase2 = 2, ηbase = 0.001.

Verification of Assumptions. We perform a depth scaling analysis by training networks with depths L ∈ {4,8,16,32,64,128,256}. We track the metrics corresponding to the assumptions above at three distinct training phases: initialization (t = 0), intermediate training (t = T/2), and the end of training (t = T), and different layers: the input layer (l = 0) and the internal layers of the final residual block (here, we denote them by l = L(1) and l = L(2)), as representatives.

