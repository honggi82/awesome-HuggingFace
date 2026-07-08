# arXiv:2601.19895v2[cs.LG]30Jan2026

[Figure 1]

## Post-LayerNorm Is Back: Stable, ExpressivE, and Deep

Chen Chen∗, Lai Wei∗,† ByteDance Seed ∗Equal Contribution, †Corresponding authors

### Abstract

Large language model (LLM) scaling is hitting a wall. Widening models yields diminishing returns, and extending context length does not improve fundamental expressivity. In contrast, depth scaling offers theoretically superior expressivity, yet current Transformer architectures struggle to train reliably at extreme depths. We revisit the Post-LayerNorm (Post-LN) formulation, whose instability at scale caused its replacement by Pre-LN in modern LLMs. We show that the central failure mode of Post-LN arises from the ResNet-style residual pathway, which introduces gradient vanishing in deep networks. We present Keel, a Post-LN Transformer that replaces this residual path with a Highway-style connection. This modification preserves the gradient flow through the residual branch, preventing signal vanishing from the top layers to the bottom. Unlike prior methods, Keel enables stable training at extreme depths without requiring specialized initialization or complex optimization tricks. Keel trains robustly at depths exceeding 1000 layers and consistently improves perplexity and depth-scaling characteristics over Pre-LN. These findings indicate that Post-LN, when paired with a Highway-style connection, provides a simple and effective foundation for building deeply scalable LLMs, opening the possibility for future infinite-depth architectures.

Date: February 2, 2026 Correspondence: Lai Wei at laiwei.future@bytedance.com

###### Training Stability (LR = 4.5×10 ³)

###### Performance Across Capabilities

###### Depth Scaling of KEEL

60.9

Pre-LN

Pre-LN KEEL

Pre-LN

80

12

+6.6%

KEEL

60

KEEL

| |
|---|

58.1

###### AverageBenchmarkScore(%)

70.8

+4.4%

70

+6.5%

66.4

66.4

57.9

63.6

62.5

10

55

58.7

60

AverageScore(%)

54.3

TrainingLoss

+16.5%

50

45.0

50

8

38.6

40

46.5

45

30

45.3

6

20

39.6

40

4

10

37.9

35

0

64 128 512 1024

0 200 400 600 800 1000

Multilingual Understanding

General Knowledge & Commonsense

Math & Code Overall Average

Number of Layers

Training Step

(a) Training Stability

(b) Expressiveness

(c) Depth Scaling

Figure 1 KEEL enables stable, expressive, and deep LLM training. (a) Training Stability: Keel maintains smooth convergence at aggressive learning rates, while Pre-LN exhibits severe instability under the same configuration. (b) Expressiveness: Keel demonstrates superior performance across all capability domains, particularly in Math & Code (+16.5%). (c) Depth Scaling: Keel consistently outperforms Pre-LN across all depths (64-1024 layers). Together, these results demonstrate that Keel’s architectural improvements enable stable optimization of ultra-deep networks with enhanced learning efficiency and model expressiveness.

### 1 Introduction

Large language model (LLM) progress has been driven primarily by scaling: bigger models, longer context windows, and larger training corpora. Yet these conventional scaling axes are beginning to show diminishing returns. Width scaling saturates quickly, context scaling grows increasingly expensive, and parameter growth alone does not unlock qualitatively new behaviors. As a result, LLM scaling is hitting a wall, and there is increasing interest in architectural directions that can deliver more expressivity per parameter.

Depth scaling offers a promising path forward. In principle, deeper networks can represent exponentially richer functions and support more hierarchical reasoning. However, current LLM architectures struggle to capitalize on depth. Training becomes increasingly unstable at extreme depths, and even when optimization succeeds, depth scaling delivers substantially worse returns than width scaling under current architectures.

The placement of Layer Normalization (LN), which is a seemingly simple architectural choice, has an enormous effect on depth scaling. The original Transformer architecture used Post-LayerNorm (Post-LN), but modern LLMs overwhelmingly adopt Pre-LayerNorm (Pre-LN) [4, 25]. Pre-LN stabilizes early training by normalizing each sublayer’s input, preventing the divergence commonly seen in deep Post-LN networks [29]. However, Pre-LN introduces its own structural limitations: it weakens gradient propagation and reduces the effective contribution of deeper layers [17]. As models grow deeper, this results in poor depth scaling and representational expressivity, limiting the potential of depth as a new scaling axis.

In contrast, Post-LN maintains large gradient signals in deeper layers, which can support superior depth scaling. Yet its training instability has made it unsuitable for LLM-scale models. When the residual output and transformed features are summed and then normalized, gradients in LayerNorm can exhibit extreme variability, especially in deep regimes [29]. Previous attempts to revive Post-LN, such as DeepNorm [27], Admin [15], and more recent hybrid normalization strategies [14, 35], mitigate some failure modes but do not fundamentally resolve the gradient pathologies of Post-LN LLMs, nor do they demonstrate reliable behavior at the depths needed to break the scaling limits of today.

To understand the root cause of these instabilities, we formally analyze the gradient dynamics of Post-LN. We derive bounds on the backward signal and show that the ResNet-style residual path is the primary source of gradient vanishing. These issues arise not from normalization itself, but from the way that residual and transformed activations are mixed before normalization.

Motivated by this analysis, we consider a small yet impactful architectural change: replace the ResNet-style residual branch with a simplified Highway-style connection, and re-express its gradient dynamics under the same theoretical framework. Our results show that this Highway-style pathway provides provable control of gradient magnitudes, allowing signals to propagate through depth without vanishing. Crucially, it maintains the inter-layer coupling that makes Post-LN expressive, while suppressing the unstable mixing that previously made Post-LN difficult to train.

Guided by these findings, we introduce Keel, a Post-LN architecture that incorporates a lightweight Highwaystyle gated connection [21]. The gate dynamically balances carry and transform signals, regulating both forward and backward information flow. This simple modification stabilizes Post-LN at scale, enabling it to realize its expressivity advantages without special initialization or customized residual scaling.

Empirically, Keel delivers substantial gains in depth scalability and model performance, effectively addressing the training stability issues often associated with traditional deep architectures. By integrating a Highwaystyle pathway with a revived Post-LN configuration, Keel enables robust training at depths exceeding 1000 layers.1 While standard Post-LN or Pre-LN architectures often exhibit severe instability when subjected to aggressive learning rates, Keel maintains smooth convergence, suggesting a more well-conditioned optimization landscape.

This newfound stability does not come at the cost of representational power. Instead, Keel demonstrates superior expressiveness across the entire depth spectrum ranging from 64 to 1024 layers. These gains are

1In this paper, unless otherwise specified, “layer” refers to the total count of residual connections, which includes both the Attention and Feed-Forward Network (FFN) layers.

particularly pronounced in specialized capability domains such as Math and Code, where the model achieves a +16.5% performance increase over Pre-LN baselines. Collectively, these results suggest that the architectural improvements in Keel facilitate a more efficient learning process, allowing for the stable optimization of ultra-deep networks. By breaking the conventional depth-scaling barrier, this approach establishes a practical and powerful framework for the next generation of large language model scaling.

### 2 Preliminary

In this section, we review the architectural components relevant to depth scaling and gradient propagation in deep learning. We revisit Highway Networks and Residual Networks, and then summarize the Post-LN and Pre-LN normalization schemes that define the operational behavior of modern Transformer blocks.

#### 2.1 Highway Networks

Highway Networks [21] were introduced as an early mechanism for training very deep feed-forward architectures. Their key idea is to allow hidden layers to adaptively regulate how much of their input is transformed versus directly carried forward. Given an input x and transformation F(x), a Highway layer computes:

xl+1 = T(xl) ⊙ F(xl) + C(xl) ⊙ xl, (1)

where T(xl) is a typically trainable gate, C(xl) is typically set to 1 − T(xl) and ⊙ denotes elementwise multiplication. This gating mechanism ensures that gradients can bypass transformations when necessary, preventing gradient attenuation.

##### 2.2 Residual Networks Residual Networks [10] replace the Highway gate with a fixed identity path, computing:

xl+1 = xl + F(xl). (2)

This unconditional skip connection was later adopted in Transformers for both MHA and FFN sublayers. However, the interaction between this residual path and normalization plays a decisive role in stability. Specifically, whether LayerNorm is applied before or after the residual addition determines how gradients propagate and how activations accumulate across depth.

##### 2.3 Post-LN: The Original Transformer Formulation The original Transformer architecture [26] applies LayerNorm after the residual addition:

xl+1 = LN(xl + F(xl)). (3)

Because the residual and transformed signals are jointly normalized, this design induces strong forward and backward coupling across layers. Prior work [29] has shown, however, that this coupling can lead to optimization difficulty at scale. During backpropagation, gradients must pass through the Jacobian of LayerNorm applied to a sum of two potentially misaligned signals:

∂xl+1 ∂xl

= JLN xl + F(xl) , (4)

making the gradient sensitive to activation statistics and susceptible to attenuation in deep networks. To mitigate this issue, DeepNorm [27] introduces a depth-dependent scaling factor on the residual branch:

xl+1 = LN(α xl + F(xl)), (5)

where α is chosen according to theoretical prescriptions. Besides, it down-scales the weight initialization by a factor β. For the decoder-only architectures, α and β are set to L0.25 and L−0.25, respectively.

- 2.4 Pre-LN: The Modern Standard in LLMs The Pre-LN formulation [29] inverts the normalization placement:

xl+1 = xl + F(LN(xl)). (6)

Here, the residual path bypasses normalization entirely, yielding a clean identity gradient path:

∂xl+1 ∂xl

= I +

∂F ∂xl

. (7)

This enables stable optimization without special initialization and has become the default in modern LLMs such as GPT-3, PaLM, and LLaMA. However, because gradients flow predominantly through the identity connection, deeper layers often contribute diminishingly to the update signal, reducing depth utilization and harming scaling behavior [22].

- 2.5 Hybrid Variants: Interpolating Between Post-LN and Pre-LN

Several recent approaches [14, 35] explore hybrid normalization placements that attempt to combine the advantages of Post-LN and Pre-LN. Two representative strategies are:

- • HybridNorm [35], which interleaves Post-LN and Pre-LN blocks throughout the network to blend their respective optimization and expressivity properties,
- • Mix-LN [14], which applies Post-LN in lower layers and transitions to Pre-LN in upper layers, aiming to leverage stronger representational coupling at the bottom while retaining stability in deeper regions.

These hybrid designs provide improved robustness over pure Post-LN and can outperform pure Pre-LN in certain regimes. However, they do not fundamentally resolve the gradient degeneration inherent to Post-LN at very large depths, nor do they provide principled guarantees for stable scaling in the extreme-depth setting required for future LLM architecture development.

### 3 KEEL

We introduce Keel, a novel architecture designed to stabilize training in deep LLMs. The forward propagation for the l-th layer is defined as:

xl+1 = LN(αxl + Fl(LN(xl))) (8) Compared to the vanilla Post-LN architecture, our method incorporates two critical structural modifications. Highway-style Residual Scaling: We introduce a scalar α to weight the skip connection, creating a highway-like structure. Based on our gradient flow analysis (detailed in Section 3.3), we set α = L, where L represents the total number of sub-layers (including both Attention and FFN layers)2. Unlike standard gating mechanisms that require coefficients to sum to 1 (e.g., (1 − λ)x + λF(x)), we rely on the final Post-LN to normalize the output magnitude, rendering explicit variance constraints on the summation unnecessary.

Residual Branch Normalization: We inject an additional Layer Normalization step into the input of the residual function Fl. While this conceptually aligns the residual branch with a Pre-LN formulation (where the input is normalized), the global architecture retains the Post-LN topology. As discussed in Section 4, this additional normalization is crucial for stabilizing the gradient flow through the residual branch, preventing the attenuation often seen in deep networks.

2Setting α = L is critical for maintaining training stability in very large-scale or deep models. For smaller architectures where vanishing or exploding gradients are less pronounced, α can be treated as a tunable hyperparameter (α > 1) to potentially accelerate convergence.

Norm

x L-1

Norm

FFN

Attn/FFN

Norm

α

Norm

Attn

1st layer

Norm

Figure 2 Illustration of our Keel architecture.

- 3.1 Implementation Details To ensure optimal performance and stability, we adopt the following implementation strategies for Keel.

Input Layer Initialization: As shown in Figure 2, we remove the Post-LN for the very first attention layer as well as the scaling factor α for both the very first attention and FFN layers. Consequently, they effectively degrade to standard Pre-LN or Post-LN blocks, ensuring stable signal initialization from the embedding layer.

Hyperparameters: Empirically, Keel allows for and benefits from a larger learning rate compared to standard Pre-LN baselines, accelerating convergence.

Normalization Configuration: All Layer Normalization operations utilize learnable affine weights (γ) but omit the additive bias term (β = 0) to improve parameter efficiency and stability.

- 3.2 Instability of Post-LN LLMs: A Gradient Perspective

We first define the operation of the RMS-based Layer Normalization (LN) used in the l-th layer. Given an input vector x, the normalization is formulated as:

x ∥x∥2

⊙ γ, (9)

LN(x) =

where ⊙ denotes element-wise multiplication, and γ ∈ Rd is a learnable affine transformation parameter. We define the scalar magnitude of this parameter as γ = ∥γ∥∞. In a standard Post-LN architecture, the forward propagation of the l-th and (l + 1)-th sub-layers is formulated as:

xl+1 = LNl (xl + Fl(xl)), (10) xl = LNl−1 (xl−1 + Fl−1(xl−1)). (11)

Here, the residual connection is a simple summation. Let zl = xl + Fl(xl) denote the pre-normalization state. During backpropagation, the gradient flows from the loss L through the layers via the chain rule:

∂L ∂xl

∂L ∂xl+1

∂xl+1 ∂xl

. (12)

=

Expanding the Jacobian ∂x

∂xl yields:

l+1

∂xl+1 ∂xl

∂ LNl(zl) ∂zl

∂zl ∂xl

=

∂ LNl(zl) ∂zl

=

∂Fl ∂xl

I +

. (13)

Focusing on the gradient flow through the residual connection, the Jacobian magnitude is determined by the normalization step. Consequently, the gradient magnitude satisfies:

∂ LNl(zl) ∂zl 2

γl √2γl−1

J∗LN

. (14)

= O

(zl) =

l

Typically, γ is initialized to 1. The cumulative gradient magnitude across L layers thus scales as:

L

J∗LN

(zl) = O

l,1

l=1

- 1

- 2L2

. (15)

This indicates that in standard Post-LN, the gradient signal exponentially decays as it propagates to lower layers, hindering the training of deep models.

#### 3.3 KEEL: Stabilizing Deep LLMs

To mitigate this instability, Keel introduces a scaling factor α and an additional normalization step. The forward pass is reformulated as:

xl+1 = LNl,1 (αxl + Fl(LNl,2(xl))), (16) xl = LNl−1,1 (αxl−1 + Fl−1(LNl−1,2(xl−1))). (17)

With the reintroduction of α to scale the shortcut branch, the denominator in the gradient derivation changes. The gradient magnitude through the l-th sub-layer’s residual connection satisfies:

 . (18)

  γl,1α

J∗LN

(zl) = O

l,1

γl2−1,1α2 + γl,22

We propose setting α = L. Analyzing the asymptotic behavior as L → ∞, we observe:

− L2

L

L

α √α2 + 1

1 L2

J∗LN

= 1. (19)

lim

(zl) = lim

= lim

1 +

l,1

L→∞

L→∞

L→∞

l=1

This limit confirms that our formulation prevents gradient vanishing.

#### 3.4 KEEL is a Post-LN Architecture

We classify Keel as a Post-LN architecture. Although Keel includes a normalization step inside the transformation branch F (similar to Pre-LN), its structure is fundamentally Post-LN. This is because the distinction between Pre-LN and Post-LN depends on the shortcut branch, not the input to the transformation function.

First, notice that in deep networks, the input to F is effectively normalized in both architectures. In Pre-LN: We explicitly normalize the input to the function: F(LN(xl)). In Post-LN: The input xl comes directly from the previous layer’s LayerNorm. Therefore, the function F(xl) already receives normalized input. More specifically, If we view the xl as the input of the LayerNorm from the previous layer, we can write the standard Post-LN forward pass as:

xPostl+1 = LNl−1(xl) + Fl(LNl−1(xl))) (20)

Comparing this to Pre-LN:

xPrel+1 = xl + Fl(LNl(xl)) (21)

The input to the transformation branch F is normalized in both cases. The real difference is the shortcut. In Pre-LN, the shortcut preserves the unnormalized scale of the previous layers, whereas in Post-LN, the shortcut itself is normalized.

Similarly, Keel can be written as:

xKeell+1 = α LNl−1(xl) + Fl(LNl(LNl−1(xl)))) (22)

Because the shortcut in Keel carries a normalized signal, it follows the structural definition of a Post-LN architecture.

Non-Redundancy of Successive LayerNorms: A natural question arises with Equation 22 - if the transformation branch Fl receives LNl(LNl−1(xl)), can these two normalization operations be merged into one? Mathematically, they cannot. Recall that a LayerNorm operation is defined as:

x ∥x∥2

⊙ γ, (23)

LN(x) =

where γ is learnable element-wise affine parameters. In Keel, the first normalization LNl−1 applies its own scaling γl−1 to the input. Because the subsequent LNl re-calculates the normalization term after this affine transformation has been applied, the two operations do not collapse into a single linear step. The motivation of designing two successive LayerNorms is discussed in Section 4.2.

### 4 Discussions

#### 4.1 KEEL vs. DeepNorm

The design of Keel draws inspiration from DeepNorm [27], a normalization approach originally proposed to stabilize deep Transformers. However, our empirical evidence suggests that while DeepNorm is effective for standard encoder-decoder architectures, its stability and performance degrade when applied to large-scale decoder-only LLMs, often underperforming standard Pre-LN baselines. Keel addresses these shortcomings through two fundamental methodological divergences.

Gradient Flow vs. Output Magnitude: The core theoretical premise of DeepNorm relies on bounding the forward output magnitude to prevent explosion. It derives scaling factors α specifically to ensure the variance of the output remains bounded as depth increases. However, bounding the forward pass does not inherently guarantee a healthy backward gradient flow. In contrast, Keel is derived explicitly from a gradient perspective (as detailed in Section 3.3). By analyzing the Jacobian properties of the residual blocks, we identify that the primary instability in deep Post-LN models stems from vanishing gradients in the lower layers, not just exploding variance in the forward pass. Our formulation ensures that the gradient norm remains consistent (≈ 1) regardless of depth L, providing a more robust optimization signal for LLMs.

Structural Topology vs. Initialization Dependence: DeepNorm relies heavily on a specialized initialization scheme (scaling weights by L−0.25) to complement its architecture. This creates a strong dependency on the initial state of the network. In the context of LLMs, which undergo massive pre-training on trillions of tokens, the model weights drift significantly from their initialization. As the training dynamics evolve, the benefits of a specific initialization strategy often diminish, leading to instability in later stages of training. Keel eliminates this dependency by baking the stabilization mechanism into the architecture itself (via the scaling factor α and the auxiliary LN). This structural approach ensures stability throughout the entire training trajectory, rather than just the initial phase.

#### 4.2 Design Evolution: From Naive Scaling to KEEL

To arrive at the optimal architecture, we conducted a step-by-step ablation study, incrementally refining the formulation based on gradient flow analysis.

- Attempt 1: Naive Residual Scaling. Our initial experiments adapted a modified DeepNorm [27] approach to the Post-LN architecture. We scaled the shortcut connections by α = L without adopting DeepNorm’s specific initialization constraints:

xl+1 = LN(αxl + Fl(xl)). (24)

Observation: While this formulation exhibited better stability than the vanilla DeepNorm on LLMs, both its training stability and final convergence performance lagged behind the standard Pre-LN baseline. We hypothesized that the unnormalized input to Fl resulted in high variance within the residual branch, complicating the optimization landscape.

- Attempt 2: Learnable Input Scaling. To control the variance entering the residual block, we introduced a learnable vector β (initialized to 1) to dynamically scale the input to Fl:

xl+1 = LN(αxl + Fl(β ⊙ xl)). (25)

Observation: While this improved forward pass stability (as β learned to scale down inputs), it introduced a side effect: gradient attenuation. As β decreases to control variance, it proportionally scales down the gradients flowing back through Fl, effectively choking the learning signal for the attention and FFN blocks. Consequently, stability remained inferior to Pre-LN.

- Attempt 3: Decoupling Scale and Variance. To resolve the gradient attenuation, we sought to normalize the input distribution while maintaining learnable scaling. We introduced an explicit Layer Normalization step before the scaling factor β:

xl+1 = LN(αxl + Fl(β ⊙ LN(xl))). (26) This addition is pivotal. As established in our theoretical analysis (Equation 27), the normalization ensures that the gradient magnitude through F is preserved:

 . (27)

  γl,1

∂F ∂ LNl,2(xl)

∂F ∂ LNl,2(xl) 2

∂ LNl,1(zl) ∂zl

∂ LNl,2(xl) ∂xl

γl,2 γl−1,1

= O

γl2−1,1α2 + γl,22

Crucially, the scaling factor γl,2 (associated with the inner LN) is effectively canceled by the affine weights γl−1,1 of the previous layer, preventing the gradient vanishing observed in Attempt 2.

Final Formulation (KEEL). We observe that Equation (14) contains redundant parameters. The Layer Normalization operation LN(x) already includes a learnable affine weight vector γ. Therefore, the external learnable vector β can be conceptually absorbed into the internal affine weights of the LN. By merging these terms, we arrive at the concise and efficient formulation of Keel:

xl+1 = LN(αxl + Fl(LN(xl))). (28)

#### 4.3 Depth-wise Test-Time Training

A prominent trend in modern architecture design involves replacing quadratic-complexity attention mechanisms with linear recurrent models. These linear attention paradigms can be formulated as a recurrent state update. For instance, standard linear attention updates its hidden state St as:

St+1 = St + vtk⊤t . (29) Recent works [2, 31, 33] interpret this recurrence through the lens of Test-Time Training (TTT). In this view, the state update is equivalent to a gradient descent step minimizing a transient objective function Lseq at each time step t:

L(seqt) = −⟨Stkt,vt⟩. (30)

By maximizing the alignment between the retrieved state Stkt and the value vt, the model "trains" itself on the context history.

We identify a structural isomorphism between this sequence-wise recurrence and the depth-wise propagation in Residual Networks. Consider the update rule for a standard Pre-LN Transformer layer:

xl+1 = xl + G(xl;W)Wo⊤, (31) where G(·) represents the transformation block (Attention or FFN):

Gattn(xl) = Attention(xl;Wq,Wk,Wv), (32) GFFN(xl) = σ(Winxl). (33)

This residual update can analogously be viewed as Test-Time Training along the depth dimension. The layer output xl+1 is the result of a gradient step optimizing a depth-wise objective Ldepth:

L(depthl) = −⟨xlWo,G(xl)⟩. (34)

From this perspective, the architectural modifications in Keel, particularly the scaling factor α and the additional normalization, function as optimization prerequisites that stabilize the “depth-wise training” process. Furthermore, we observed that deep modeling exhibits phenomena similar to those in sequence modeling. For instance, in deep modeling, the first few layers are crucial, akin to the attention sink phenomenon [9, 28]. Moreover, both of them exhibit local patterns: in sequence modeling, recent tokens typically receive more attention, whereas in deep modeling, deeper layers become increasingly significant. Further details can be found in Appendix A.

This creates a compelling parallel with recent advancements in sequence modeling, such as Titans or LaCT. Just as these models introduce gating mechanisms (analogous to our α) and state normalization (analogous to our residual LN) to improve the recurrent TTT objective in Equation 30, Keel applies similar principles to improve the depth-wise TTT objective in Equation 34.

We believe this duality suggests a promising avenue for future research: techniques developed to improve long-context sequence recurrence can likely be adapted to develop infinite-depth model propagation, and vice versa.

### 5 Experiments

#### 5.1 Stability Analysis

We evaluate the training stability of Keel against a comprehensive suite of established normalization strategies. Our baselines include the standard Post-LN and Pre-LN architectures, as well as recent variants designed for stability: DeepNorm [27], and hybrid approaches such as HybridNorm [35] and Mix-LN [14].

Benchmark Protocol: Maximum Tolerable Learning Rate. To quantify stability, we measure the Maximum Tolerable Learning Rate (Max LR). It is defined as the highest learning rate a model can sustain during warm-up stage without diverging.

We adopt a stress-test protocol where the learning rate schedule is set with an aggressively high peak of ηpeak = 5 × 10−2. The learning rate increases linearly from 0 to ηpeak over the warm-up period. We monitor the training dynamics at every step; the learning rate recorded at the exact step immediately preceding divergence is reported as the Max LR. A higher Max LR indicates a more robust optimization landscape and superior training stability.

Criteria for Divergence. Identifying divergence in LLMs goes beyond simple numerical overflow (NaN). Based on our observations, we categorize divergence into three distinct pathological behaviors:

1. Loss Stagnation: The loss curve enters a plateau significantly earlier than expected, ceasing to decrease despite continued training. This often indicates vanishing gradients or saturation in the normalization layers.

converged

converged

converged

12

12

12

diverged

diverged

diverged

10

10

10

8

8

8

Loss

Loss

Loss

6

6

6

4

4

4

2

2

0 5000 10000 15000 20000 Step

0 500 1000 1500 2000 Step

0 1000 2000 3000 4000 5000 Step

(c) Optimization degradation Figure 3 The illustration of three distinct pathological behaviors of model divergence in the training of LLMs.

(a) Loss stagnation

(b) Irrecoverable instability

Post-LN DeepNorm HybridNorm Mix-LN Pre-LN KEEL

- Configuration A: 64 Layers, 5,000 Warm-ups, ηpeak = 5 × 10−2

Max LR 3.0 × 10−4 3.5 × 10−4 4.9 × 10−4 8.6 × 10−4 7.65 × 10−3 1.01 × 10−2

- Configuration B: 512 Layers, 5,000 Warm-ups, ηpeak = 5 × 10−2

Max LR 2.8 × 10−4 3.5 × 10−4 3.5 × 10−4 3.5 × 10−4 4.67 × 10−3 6.31 × 10−3

- Table 1 Comparison of Maximum Tolerable Learning Rates across different architectures. Keel consistently supports higher learning rates, indicating superior training stability, particularly as model depth increases to 256 layers.

- 2. Irrecoverable Instability: The loss exhibits high-magnitude spikes from which the model fails to recover. Unlike transient spikes common in LLM training, these result in a permanent degradation of the loss value (i.e., the loss never returns to the pre-spike baseline).
- 3. Optimization Degradation: The model does not exhibit explicit spikes, and the loss continues to decrease. However, the convergence rate is anomalously slow compared to a healthy baseline run. This “silent” failure mode typically suggests that the effective update step size has been excessively dampened.

If a training run satisfies any of the above criteria, it is marked as diverged. Results. Table 1 summarizes the stability limits across varying depths (64 and 512 layers). Keel demonstrates a significant improvement in stability compared to all baselines.

Notably, in the 64-layer configuration, Keel tolerates a learning rate of 1.01 × 10−2, surpassing the standard Pre-LN (7.65 × 10−3) and exceeding the vanilla Post-LN baseline by nearly two order of magnitude. This trend holds for extremely deep networks: at 512 layers, Keel maintains a Max LR of 6.31 × 10−3, validating our theoretical claims regarding training stability in deep architectures.

#### 5.2 Optimal Learning Rate

Having established the theoretical stability of Keel, we investigate its impact on downstream performance. We hypothesize that the ability to tolerate larger learning rates is not merely a stability metric, but an optimization advantage that allows the model to traverse the loss landscape more effectively and escape local minima.

To verify this, we conducted a controlled sweep of peak learning rates (ηpeak ∈ {1.5,3.0,6.0} × 10−3) for both the Pre-LN baseline and Keel. All models have 512 layers and a hidden dimension of 1024. We pre-train these models with the same 250B tokens from the internal data. The learning rate linearly increases from 0 to ηpeak over the first 2500 steps, then concludes with a cosine decay to 1.0 × 10−7.

As summarized in Table 2, the results reveal two critical optimization behaviors. The Pre-LN baseline exhibits

12

Peak LR = 4.5e-3

12

Peak LR = 4.5e-3

Peak LR = 3e-3

Peak LR = 3e-3

10

10

8

8

Loss

Loss

6

6

4

4

2

0 250 500 750 1000 1250 1500 Step

0 500 1000 1500 2000 2500 3000 Step

(a) 256 layers, batch size 8M

(b) 512 layers, batch size 4M

- Figure 4 Training loss curves of Pre-LN during the early stage of training. Pre-LN exhibits a pronounced loss spike when trained with a higher learning rate.

performance saturation and instability at higher learning rates. While some tasks (e.g., MMLU) improve at η = 6.0 × 10−3, others suffer from degradation (e.g., ARC-Easy drops from 75.8 to 74.0; MBPP drops from 25.0 to 22.8), suggesting that the model is oscillating in specific subspaces. In contrast, Keel demonstrates a consistent, monotonic performance gain across all benchmarks as the learning rate increases. At η = 6.0×10−3, Keel achieves a global average of 55.5, significantly outperforming the best Pre-LN configuration (52.3).

The benefits of stable, high-learning-rate training are most pronounced in reasoning-intensive tasks. On GSM-8K, increasing the learning rate to 6.0 × 10−3 boosts Keel to 43.8, a massive improvement over the Pre-LN baseline (38.1). This confirms that Keel effectively unlocks the optimization potential of large step sizes without suffering from the gradient vanishing or instability typical of deep LLMs.

#### 5.3 Scalability Analysis: Performance at Depth Scaling

To validate the advantages of Keel in mitigating training instability, we conducted a comprehensive scaling study. We compared Keel against the robust Pre-LN baseline across three distinct depth configurations: 64, 128, 512 and 1024 layers. All models underwent a rigorous two-stage training protocol consisting of general pre-training followed by continued pre-training (CPT).

Setup. We use a batch size of 1024 and a sequence length of 4096. All models are first pre-trained on 190B tokens and then further pre-trained on an additional 60B tokens. We adopt the AdamW optimizer with β1 = 0.9, β2 = 0.95, and a weight decay of 0.01. Gradient clipping is applied with a maximum norm of 1.0. The learning rate schedule consists of a linear warmup over the first 2500 steps, followed by cosine decay to 1.0 × 10−7. We set the peak learning rate to 9.0 × 10−3 for the 64-layer and 128-layer models, and to 6.0×10−3 for the 512-layer models. For the 1024-layer models, we adopt a peak learning rate of 4.5×10−3 for Keel and 3.0 × 10−3 for Pre-LN, as we observe that training 1024-layer Pre-LN models with a peak learning rate of 4.5 × 10−3 leads to substantial instability (see Figure 4b).

Results. Table 3 details the zero-shot and few-shot performance across a diverse suite of benchmarks. The results reveal a clear trend: while Keel provides consistent gains at shallower depths (64L), its advantage becomes significantly more pronounced as the model depth increases, maintaining strong scalability even at 1024 layers.

• Reasoning Capabilities: On complex reasoning tasks like GSM-8K and HumanEval, the performance gap widens dramatically at extreme depths. For instance, on GSM-8K, Keel achieves a score of 58.6 at 1024 layers, surpassing the 1024-layer Pre-LN baseline (49.8) by 8.8 points. Notably, the Pre-LN baseline shows signs of stagnation between 512 and 1024 layers on several reasoning tasks, whereas

Benchmark Pre-LN KEEL

Peak LR (×10−3) 1.5 3.0 6.0 1.5 3.0 6.0 Common Sense & Knowledge

MMLU (5-shot) 47.5 48.1 52.9 49.8 53.5 56.3 ARC-Easy (25-shot) 75.8 75.3 74.0 74.5 76.4 77.1 ARC-Challenge (25-shot) 45.9 44.4 43.6 45.1 44.5 48.9 HellaSwag (0-shot) 64.2 64.2 64.9 64.3 65.5 67.4 LAMBADA (0-shot) 66.7 66.3 67.0 66.5 67.9 68.8 PIQA (0-shot) 75.6 75.5 75.1 76.3 75.8 76.7 AGI-Eval (0-shot) 29.3 29.7 34.7 30.5 34.6 39.6 Winogrande (0-shot) 64.1 64.2 63.5 63.8 64.5 65.7 CommonsenseQA (0-shot) 48.6 46.6 55.7 53.2 52.3 61.3

Reasoning & Coding

GSM-8K (5-shot) 30.3 31.1 38.1 30.5 36.6 43.8 HumanEval (0-shot) 14.0 17.7 17.7 14.6 18.3 19.5 MBPP (0-shot) 25.0 24.0 22.8 22.6 26.0 26.0

Multilingual Understanding

CMMLU (5-shot) 52.8 52.7 60.3 57.0 61.5 64.4 C-Eval (5-shot) 52.7 52.1 61.4 55.9 60.8 62.0

Average Score 49.5 49.4 52.3 50.5 52.7 55.5

- Table 2 Performance scaling with Learning Rate. While Pre-LN results are inconsistent at high learning rates (degrading on tasks like ARC-Easy and MBPP at η = 6.0 × 10−3), Keel exhibits robust, monotonic improvement, effectively leveraging larger step sizes to achieve superior convergence.

Keel continues to yield significant gains.

• Global Average: The average performance improvement scales significantly with depth. Keel improves over the baseline by approximately +1.7 points at 64 layers, +1.2 points at 128 layers, +3.8 points at 512 layers, and +3.0 points at 1024 layers.

This empirical evidence confirms that Keel effectively stabilizes optimization in ultra-deep LLMs, unlocking capabilities that are otherwise hindered by optimization difficulties or diminishing returns in standard architectures.

#### 5.4 Scalability Analysis: Performance at Data Scaling

To assess the scalability of Keel with increasing training data, we compare it against Pre-LN across different token budgets. We train Pre-LN and Keel on 10B and 40B tokens from the FineWeb-EDU dataset [16]. Both models have 256 layers and a hidden size of 1024, resulting up to 3B parameters. On the 10B-token run, we tune the peak learning rate for both Keel and the Pre-LN baseline over {1.0,1.5,3.0,4.5} × 10−3. The batch size is set as 256. We use a maximum sequence length of 2048 for the 10B-token training and 4096 for the 40B-token training. The learning rate schedule uses a 2000-step linear warmup followed by cosine decay.

- Figure 5 presents the training loss curves of Keel and the baseline. As shown in Figure 5a, although Keel exhibits higher loss early in training, it overtakes Pre-LN as training proceeds and achieves lower loss by the end. More importantly, when scaling the training budget from 10B to 40B tokens, the performance gap

64 Layers 128 Layers 512 Layers 1024 Layers

Benchmark

Pre-LN KEEL Pre-LN KEEL Pre-LN KEEL Pre-LN KEEL Common Sense & Knowledge

MMLU (5-shot) 36.8 38.8 45.4 46.9 53.9 57.0 57.5 59.2 ARC-Easy (25-shot) 65.8 65.0 69.7 68.7 76.6 78.6 80.4 80.3 ARC-Challenge (25-shot) 33.7 33.6 38.7 37.8 45.5 50.4 51.1 51.7 HellaSwag (0-shot) 46.7 48.0 53.7 55.4 64.1 66.3 68.2 69.1 LAMBADA (0-shot) 51.5 52.9 57.7 59.0 65.1 67.5 68.8 70.0 PIQA (0-shot) 69.3 69.0 71.7 72.1 75.0 76.3 77.0 78.3 AGI-Eval (0-shot) 26.0 26.1 28.8 31.0 34.3 39.8 37.3 44.9 Winogrande (0-shot) 57.4 58.2 59.5 59.5 64.2 65.4 66.1 67.0 CommonsenseQA (0-shot) 31.4 37.8 48.2 49.1 60.0 65.8 60.6 67.1

Reasoning & Coding

GSM-8K (5-shot) 9.6 12.9 22.4 28.0 45.6 49.8 49.8 58.6 HumanEval (0-shot) 9.8 11.0 17.1 15.9 22.6 32.3 29.9 32.9 MBPP (0-shot) 13.0 13.2 20.8 22.2 32.0 34.2 36.2 38.6

Multilingual Understanding

CMMLU (5-shot) 39.3 43.4 50.1 52.7 61.0 65.0 64.9 67.0 C-Eval (5-shot) 40.3 44.6 50.0 53.1 60.6 64.9 63.2 67.3

Average Score 37.9 39.6 45.3 46.5 54.3 58.1 57.9 60.9

- Table 3 Scalability Benchmark. Performance comparison between Pre-LN and Keel across increasing model depths (64L, 128L, 512L, 1024L). The gain provided by Keel increases significantly with depth, particularly in reasoning-heavy tasks (GSM-8K, HumanEval), highlighting the method’s effectiveness in stabilizing ultra-deep networks.

further widens in favor of Keel. As shown in Table 4, the accuracy on various end tasks also exhibits similar trends: the improvement on HellaSwag increases from 0.9% to 2.6%. We attribute this to the lower effective depth of Pre-LN relative to Keel, which limits its representational capacity. As the amount of training data increases, the performance gain gradually plateaus. Overall, these results suggest that Keel is particularly well suited for large-scale training, while its advantages are less pronounced in low-data regimes.

#### 5.5 Deeper vs. Wider

A fundamental question in neural scaling laws is the optimal allocation of a fixed parameter budget: is it more effective to build deeper, narrower networks or shallower, wider ones? Theoretically, deeper networks possess greater expressivity and reasoning depth. However, in practice, Wide topologies often outperform Deep ones because deep networks are notoriously difficult to train.

To investigate whether Keel overcomes this optimization barrier, we conducted a controlled experiment with a fixed parameter budget of 3B. We compare three configurations:

- 1. Baseline Deep (Pre-LN): A standard deep topology (512 layers) which typically suffers from gradient degradation.
- 2. Baseline Wide (Pre-LN): A standard wide topology (128 layers, 2048 hidden size), representing the industry standard for stability.
- 3. Ours (Deep): The same deep topology (512 layers) augmented with Keel.

3.0

2.8

Keel

Keel

Pre-LN

Pre-LN

2.9

2.7

Loss

Loss

2.8

2.6

2.7

2.5

2.6

2.4

0 5k 10k 15k 20k Step

0 10k 20k 30k 40k Step

(a) 10B tokens, 512 layers

(b) 40B tokens, 512 layers

Figure 5 Training loss curves of Pre-LN and Keel on FineWeb-EDU dataset [16] with varying training tokens. As the training token scales from 10B to 40B tokens, Keel achieves larger gain on training loss compared to Pre-LN baseline.

Best LR

ARC-Easy

###### PIQA

HellaSwag

###### LAMBADA

Winogrande

SciQ

Models

Average

25-shot

0-shot

0-shot

0-shot

0-shot

0-shot

(×10−3)

10B tokens, 256 layers, 3B parameters

Pre-LN 1.5 56.9 72.7 52.9 50.0 52.6 76.1 60.3 Keel 3.0 58.8 73.4 53.8 49.6 56.0 78.1 61.5

40B tokens, 256 layers, 3B parameters

Pre-LN 1.5 64.4 75.4 61.8 56.3 59.6 82.9 66.7 Keel 3.0 64.1 76.3 64.4 58.5 62.4 83.6 68.2

- Table 4 Performance comparison between Pre-LN and Keel across increasing training data (10B, 40B) on FineWebEDU dataset [16].

We pre-train these models on 250B tokens of private data. We use a peak learning rate of 6.0 × 10−3 for the 512-layer models and 3.0×10−3 for the 128-layer models. We set the AdamW β coefficients to (0.9, 0.95). The batch size and sequence length are set to 1024 and 4096, respectively. We use a warmup phase of 2500 steps.

Results. Table 5 reports the results under a fixed budget of 3B parameters. We first compare the two Pre-LN variants. Although the shallow-and-wide model achieves a lower training loss than the deep-and-narrow model (see Appendix B), their average downstream scores are close. This discrepancy suggests that training loss is not necessarily positively correlated with end-task performance, particularly when optimizing very deep LLMs. Meanwhile, the deep-and-narrow model exhibits a consistent advantage on more complex reasoning-oriented benchmarks (e.g., GSM-8K and MMLU), indicating that increasing depth has the potential to improve complex reasoning when the model can be trained effectively.

Motivated by these observations, Keel targets the stability and degradation issues of deep modeling. By stabilizing gradient flow and making deep training more reliable, Keel not only improves reasoning performance beyond the deep Pre-LN baseline (e.g., 43.8 on GSM-8K vs. 38.1), but also achieves the best overall performance: our 512-layer Keel model reaches an average score of 55.5, outperforming both the deep Pre-LN baseline (+3.2) and the wide Pre-LN baseline (+3.3).

Configuration Deep (Pre-LN) Wide (Pre-LN) Deep (KEEL)

Parameters 3B 3B 3B Layers 512 128 512 Hidden Dim 1024 2048 1024

Multilingual Understanding

CMMLU (5-shot) 60.3 59.3 64.4 C-Eval (5-shot) 61.4 57.8 62.0

General Knowledge & Commonsense

MMLU (5-shot) 52.9 51.5 56.3 ARC-Easy (25-shot) 74.0 75.2 77.1 ARC-Challenge (25-shot) 43.6 45.2 48.9 HellaSwag (0-shot) 64.9 65.9 67.4 LAMBADA (0-shot) 67.0 68.3 68.8 PIQA (0-shot) 75.1 76.3 76.7 AGI-Eval (0-shot) 34.7 36.1 39.6 Winogrande (0-shot) 63.5 65.5 65.7 CommonsenseQA (0-shot) 55.7 52.9 61.3

Math & Code

GSM-8K (5-shot) 38.1 35.3 43.8 HumanEval (0-shot) 17.7 16.5 19.5 MBPP (0-shot) 22.8 24.4 26.0

###### Average Score 52.3 52.2 55.5

- Table 5 Deeper vs. Wider. Comparison of models with a fixed 3B parameter budget. While standard Deep models underperform Wide ones due to optimization difficulties, Keel effectively stabilizes the deep network, allowing it to outperform the Wide baseline significantly, particularly on reasoning-intensive tasks.

#### 5.6 Experimental Setup

To rigorously evaluate the scalability of our approach, we train a deep 512-layer LLM using the Keel formulation and compare it against a standard Pre-LN baseline. Both models utilize a hidden dimension of dmodel = 1024, resulting in an extreme depth-to-width ratio of 0.5 (512 layers vs. 1024 width). This topology is specifically chosen to stress-test the optimization stability of deep networks where gradient signal preservation is critical. Both models contain approximately 3 Billion parameters.

Models are trained on a massive corpus of 1T tokens from our internal dataset. The training pipeline proceeds in two distinct phases: a general pre-training stage on the first 750B tokens, followed by continued pre-training (CPT) on the remaining 250B tokens to enhance reasoning and coding capabilities. The optimization is performed using AdamW with β1 = 0.9,β2 = 0.95, a weight decay of 0.01, a global batch size of 2048, and a sequence length of 4096.

A critical differentiator in our setup is the learning rate schedule. We employ a linear warm-up over 2500 steps followed by cosine decay. Keel remains robust at higher learning rates, while the Pre-LN baseline becomes unstable, exhibiting pronounced loss spikes that force us to cap its peak rate at 3.0 × 10−3 (see Figure 4a). This stability allows us to train Keel at a significantly higher peak learning rate of 4.5 × 10−3, unlocking superior convergence properties.

Following pre-training, both models undergo supervised fine-tuning (SFT) on a high-quality instruction mix. We perform a comprehensive grid search over learning rates ({1.0,2.0,3.0} × 10−6) and training durations (1 to 3 epochs) to ensure that the reported results reflect the optimal capability of each architecture. We utilize the lm-evaluation-harness [8] for standardized assessment across a broad suite of benchmarks spanning general

Configuration Pre-LN KEEL

Architecture 512 Layers / 3B Params 512 Layers / 3B Params Peak Learning Rate 3.0 × 10−3 4.5 × 10−3

Multilingual Understanding

CMMLU (5-shot) 66.6 72.0 C-Eval (5-shot) 66.2 69.5

General Knowledge & Commonsense

MMLU (5-shot) 59.5 62.7 ARC-Easy (25-shot) 79.7 81.6 ARC-Challenge (25-shot) 51.6 53.6 HellaSwag (0-shot) 68.2 69.8 LAMBADA (0-shot) 68.0 69.7 PIQA (0-shot) 76.6 77.5 AGI-Eval (0-shot) 37.9 46.5 Winogrande (0-shot) 66.7 66.7 CommonsenseQA (0-shot) 64.5 69.8

Math & Code

GSM-8K (5-shot) 51.0 60.9 HumanEval (0-shot) 29.9 33.5 MBPP (0-shot) 35.0 40.6

###### Average Score 58.7 62.5

- Table 6 Pre-training Results (1T Tokens). Comparison of a 512-layer Pre-LN baseline vs. Keel. Keel capitalizes on its superior stability to train with a 50% higher learning rate, resulting in substantial gains across all categories, particularly in reasoning-intensive tasks like GSM-8K and AGI-Eval.

knowledge [3, 6, 11, 18, 20, 23, 32, 34], reasoning [7], coding [1, 5], and multilingual understanding [12, 13].

#### 5.7 Main Results

Pre-training Performance. Table 6 summarizes the zero-shot and few-shot performance after the 1T token pretraining phase. Keel demonstrates a decisive advantage over the Pre-LN baseline, achieving a global average improvement of +3.8 points (62.5 vs. 58.7). A deeper analysis reveals that while Keel provides consistent gains across general knowledge tasks (e.g., MMLU, HellaSwag), the performance gap widens dramatically on tasks requiring complex, multi-step reasoning. For instance, on the GSM-8K math benchmark, Keel outperforms the baseline by nearly +10 points (60.9 vs. 51.0). Similarly, we observe significant improvements in code generation, with MBPP and HumanEval scores increasing by +5.6 and +3.6 respectively. This suggests that the improved gradient flow in Keel is particularly beneficial for learning the deep, hierarchical representations necessary for algorithmic reasoning, rather than simple pattern matching.

Supervised Fine-Tuning (SFT) Results. We further investigate whether the advantages of Keel persist after supervised fine-tuning. Table 7 presents the results on a suite of challenging benchmarks, including MMLU-Pro and BBH (BIG-Bench Hard), which are designed to probe the limits of model capabilities. The results confirm that the “pre-training advantage” is effectively transferred to the SFT stage. The performance delta in reasoning tasks is preserved and, in some cases, amplified. On GSM-8K, the gap remains over 10 points (68.8 vs. 58.7), and on MMLU-Pro which requires nuanced understanding and robust instruction following, Keel achieves a score of 35.6 compared to the baseline’s 26.6. This indicates that Keel not only learns better representations during pre-training but also possesses a more amenable optimization landscape for fine-tuning,

Benchmark Pre-LN KEEL Reasoning & Hard Tasks

MMLU-Pro (5-shot) 26.6 35.6 BBH (3-shot) 46.4 51.7 GSM-8K (5-shot) 58.7 68.8 HumanEval (0-shot) 31.7 37.2 MBPP (0-shot) 37.6 42.6

Knowledge & Multilingual

MMLU (5-shot) 60.0 62.5 CMMLU (5-shot) 66.5 71.8 C-Eval (5-shot) 66.5 68.4

###### Average Score 49.3 54.8

- Table 7 SFT Performance. After fine-tuning, Keel maintains its dominance, particularly on “Hard” benchmarks (BBH, MMLU-Pro) that test the limits of model reasoning, demonstrating that the architectural improvements translate directly to downstream tasks.

allowing it to adapt to complex instructions without catastrophic forgetting or optimization difficulties.

### 6 Conclusion

In this work, we propose that, as conventional scaling strategies plateau, depth is a promising axis for improving Transformer expressivity, which remained under-exploited due to optimization instability at extreme depths. We revisit Post-LayerNorm (Post-LN) formulation and identify its primary breakdown mechanism: the ResNet-style residual pathway induces gradient vanishing as depth grows, preventing reliable end-to-end signal propagation. To address this, we introduce Keel, a Post-LN Transformer that replaces the ResNet residual path with a Highway-style connection. This simple architectural change directly targets the source of vanishing by preserving gradient flow through the residual branch, enabling stable optimization without relying on specialized initialization schemes or elaborate training heuristics. Empirically, Keel trains robustly at depths exceeding 1000 sub-layers, and it delivers consistent improvements and more favorable depth-scaling behavior compared to widely used Pre-LN baselines. This reopens the design space for extreme-depth Transformers and points toward a practical path for exploring infinite-depth architectures in future work.

### 7 Limitation and Future Work

This work primarily focuses on how to stably train Post-LN LLMs when scaling depth. However, training instability is not driven by depth alone. As model width increases (e.g., hidden size, number of experts, or FFN dimension), LLM training can also become more unstable. In such wider settings, Post-LN may require a larger α (or stronger stabilization mechanisms) to maintain stable optimization. We leave a thorough investigation of LLM stability under width scaling to future work.

In addition, Keel addresses the effective-depth issue in Pre-LN LLMs. When the width-to-depth ratio is already high, this issue may be less pronounced, and the gains from Keel may therefore be less substantial. Finally, Keel typically requires substantial training data to be effective and is not recommended in low-data regimes.

### References

- [1] Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models.

- CoRR, abs/2108.07732, 2021.
- [2] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2024.

- [3] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.

- [4] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

- [5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. CoRR, abs/2107.03374, 2021.

- [6] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.

- [7] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021.

- [8] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/ records/12608602.
- [9] Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min Lin. When attention sink emerges in language models: An empirical view. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

- [10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

- [11] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

- [12] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

- [13] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. CMMLU: measuring massive multitask language understanding in chinese. CoRR, abs/2306.09212, 2023. URL https://doi.org/10.48550/arXiv.2306.09212.

- [14] Pengxiang Li, Lu Yin, and Shiwei Liu. Mix-ln: Unleashing the power of deeper layers by combining pre-ln and post-ln. arXiv preprint arXiv:2412.13795, 2024.

- [15] Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, and Jiawei Han. Understanding the difficulty of training transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 5747–5763. Association for Computational Linguistics, 2020.

- [16] Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. Fineweb-edu: the finest collection of educational content, 2024. URL https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu.
- [17] Toan Q Nguyen and Julian Salazar. Transformers without tears: Improving the normalization of self-attention. arXiv preprint arXiv:1910.05895, 2019.

- [18] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), August 2016.

- [19] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21:140:1–140:67, 2020.

- [20] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: an adversarial winograd schema challenge at scale. Commun. ACM, 64(9):99–106, 2021.

- [21] Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber. Highway networks. CoRR, abs/1505.00387, 2015.

- [22] Wenfang Sun, Xinyuan Song, Pengxiang Li, Lu Yin, Yefeng Zheng, and Shiwei Liu. The curse of depth in large language models. CoRR, abs/2502.05795, 2025.

- [23] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4149–4158. Association for Computational Linguistics, 2019.

- [24] Llama Team. The llama 3 herd of models. CoRR, abs/2407.21783, 2024.

- [25] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [26] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- [27] Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, and Furu Wei. Deepnet: Scaling transformers to 1,000 layers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(10):6761–6774, 2024.

- [28] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

- [29] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On layer normalization in the transformer architecture. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 10524–10533. PMLR, 2020.

- [30] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. CoRR, abs/2412.15115, 2024.

- [31] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. Advances in neural information processing systems, 37:115491–115522, 2024.

- [32] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 4791–4800. Association for Computational Linguistics, 2019.

- [33] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025.

- [34] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. In Findings of the Association for Computational Linguistics: NAACL 2024, Mexico City, Mexico, June 16-21, 2024, pages 2299–

2314. Association for Computational Linguistics, 2024.

- [35] Zhijian Zhuo, Yutao Zeng, Ya Wang, Sijun Zhang, Jian Yang, Xiaoqing Li, Xun Zhou, and Jinwen Ma. Hybridnorm: Towards stable and efficient transformer training via hybrid normalization. arXiv preprint arXiv:2503.04598, 2025.

## Appendix

### A Layer Redundancy in Deep LLMs

Layer redundancy constitutes a major challenge in training deep LLMs. Prior work [14, 22] suggests that layer redundancy is correlated with the placement of layer normalization. To quantify the redundancy of a specific layer in an LLM, we design a set of controlled ablation experiments. Concretely, we train the Pre-LN and Keel on the same training data (250B tokens). We then measure layer redundancy by the performance degradation (∆PPLi = PPLremove i-th layer −PPLfull model) induced by removing an individual layer: a smaller drop indicates higher redundancy, while a larger drop indicates that the layer is more essential. For the open-source models, we select Qwen2.5-72B-Instruct [30] and LLaMA-3.3-70B-Instruct [24]. Both models comprise 80 layers. We evaluate these models by computing perplexity on the C4 validation dataset [19]. The results are demonstrated in Figure 6 and Figure 7.

104

Pre-LN

Keel

PPL

101

10 2

0 50 100 150 200 250 # Layers

###### Figure 6 Performance degradation caused by the removal of each individual layer in Pre-LN and Keel.

102

PPL

100

0 10 20 30 40 50 60 70 80 # Layers

###### (a) Qwen2.5-72B-Instruct

104

PPL

101

0 10 20 30 40 50 60 70 80 # Layers

(b) LLaMA-3.3-70B-Instruct

- Figure 7 Performance degradation caused by the removal of each individual layer in Qwen2.5-72B-Instruct and LLaMA-3.3-70B-Instruct.

We observe that deep LLMs exhibit a phenomenon analogous to the “attention sink” reported in sequence modeling [9, 28]. Specifically, the first few layers are exceptionally critical: removing it increases perplexity to above 104, destroying model performance. Beyond the first layer, for both the Pre-LN model and the

Post-LN-style Keel model, the performance degradation induced by layer removal increases with depth. We observe the same pattern for Qwen2.5-72B-Instruct and LLaMA-3.3-70B-Instruct. These results suggest that shallower layers are more redundant, whereas deeper layers are increasingly pivotal. Furthermore, compared to Pre-LN, removing Keel layers produces a larger PPL increase in the shallow region, indicating that Keel substantially reduces shallow-layer redundancy and yields a greater effective depth.

### B Discrepancy Between Training Loss and Downstream Evaluation

Training loss is commonly used as a key metric for assessing the quality of LLM pre-training, as it is relatively simple to measure and often directly correlates with downstream task performance. As a result, it is frequently considered a critical indicator of model pre-training success. However, during the training of deep LLMs, we observe that a lower training loss does not necessarily correspond to better downstream performance.

3.0

2.4

- 1k hidden dim, 256 layers

- 2k hidden dim, 64 layers

Keel, Peak LR = 1.5e-3

Pre-LN, Peak LR = 1.5e-3

2.8

2.3

2.6

Loss

Loss

2.4

2.2

2.2

2.1

2.0

1.8

2.0

10k 20k 30k 40k 50k 60k Step

10k 20k 30k 40k 50k 60k Step

(a) Training loss of shallow and deep Pre-LN.

(b) Training loss of Pre-LN and Keel.

- Figure 8 Comparison of training loss in Pre-LN models and between Pre-LN and Keel models, illustrating the mismatch between training loss and downstream performance.

Figure 8a shows the training loss curves of shallow and wide (2048 hidden size, 64 layers) and deep and narrow (1024 hidden size, 256 layers) Pre-LN models. Both models were trained on the same 250 billion tokens. While the shallow and wide Pre-LN model exhibits a significantly lower training loss, as shown in Table 5, it performs slightly worse than the deep and narrow model on downstream tasks.

A similar phenomenon was observed in the comparison between Pre-LN and Keel models at a smaller learning rate. As shown in Figure 8b, at a peak learning rate of 1.5 × 10−3, the Pre-LN model exhibits a slightly lower training loss than Keel, yet Keel demonstrates better performance on downstream tasks (see Table 2). Above all, we find that in the training of deep LLMs, training loss and end-task performance are not always positively correlated. Therefore, evaluating end-task performance during pre-training is essential. We aim to further investigate the mechanisms behind this phenomenon in future work.

### C Model Configuration

Configuration Hyper-parameters

Hidden dimension 1024 2048 Intermediate dimension 3072 6144 Attention heads 16 16 KV heads 8 8 Initializer N(0,0.022) N(0,0.022) Dropout 0.0 0.0 Attention dropout 0.0 0.0 RMS Norm ϵ 1e-5 1e-5 RoPE θ 10000.0 10000.0

Table 8 Model configuration of Keel and Pre-LN baselines.

