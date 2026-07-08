# arXiv:2509.26030v2[cs.LG]5Oct2025

## Muon Outperforms Adam in Tail-End Associative Memory Learning

Shuche Wang1,∗ Fengzhuo Zhang1,∗,† Jiaxiang Li2,∗ Cunxiao Du3 Chao Du3 Tianyu Pang3 Zhuoran Yang4 Mingyi Hong2 Vincent Y. F. Tan1

1National University of Singapore 2University of Minnesota 3Sea AI Lab 4Yale University shuche.wang@u.nus.edu fzzhang@u.nus.edu li003755@umn.edu cnsdunm@gmail.com {duchao, tianyupang}@sea.com zhuoran.yang@yale.edu mhong@umn.edu vtan@nus.edu.sg

Abstract

The Muon optimizer is consistently faster than Adam in training Large Language Models (LLMs), yet the mechanism underlying its success remains unclear. This paper demystifies this mechanism through the lens of associative memory. By ablating the transformer components optimized by Muon, we reveal that the associative memory parameters of LLMs, namely the Value and Output (VO) attention weights and Feed-Forward Networks (FFNs), are the primary contributors to Muon’s superiority. Motivated by this associative memory view, we then explain Muon’s superiority on real-world corpora, which are intrinsically heavy-tailed: a few classes (tail classes) appear far less frequently than others. The superiority is explained through two key properties: (i) its update rule consistently yields a more isotropic singular spectrum than Adam; and as a result, (ii) on heavy-tailed data, it optimizes tail classes more effectively than Adam. Beyond empirical evidence, we theoretically confirm these findings by analyzing a one-layer associative memory model under class-imbalanced data. We prove that Muon consistently achieves balanced learning across classes regardless of feature embeddings, whereas Adam can induce large disparities in learning errors depending on embedding properties. In summary, our empirical observations and theoretical analyses reveal Muon’s core advantage: its update rule aligns with the outer-product structure of linear associative memories, enabling more balanced and effective learning of tail classes in heavy-tailed distributions than Adam.

### 1 Introduction

The effectiveness of Adam (Kingma & Ba, 2015) across diverse training scenarios has made it one of the most widely used optimizers for neural networks, serving as a cornerstone of the tremendous successes of Large Language Models (LLMs). Building on this foundation, Muon (Jordan et al., 2024) has emerged as a matrixparameter optimizer designed to surpass Adam. Empirical studies show that Muon is nearly 2 times faster than Adam across a wide range of model sizes and architectures (Liu et al., 2025; Jordan et al., 2024). Its key innovation is to replace the raw gradient with the sum of its normalized orthogonal factors, which can be interpreted as performing steepest descent with respect to the spectral norm (Bernstein & Newhouse, 2024).

However, despite its empirical success, a rigorous understanding of why and how Muon outperforms Adam in transformers remains incomplete. In particular, the steepest gradient descent interpretation does not clarify why optimization with respect to the spectral norm, as in Muon, should outperform optimization with respect to the infinity norm (for vectors), as in Adam. Consequently, convergence analyses of Muon derived from this interpretation fail to account for its observed superiority over Adam (Li & Hong, 2025; Shen et al., 2025).

∗ Equal contribution. † Project Lead. Fengzhuo conducted this work during his internship at Sea AI Lab.

This paper takes the first step toward understanding the mechanisms underlying Muon’s superiority over Adam in training LLMs. Specifically, we first ask the question:

Which transformer components benefit most from Muon’s matrix-norm–based optimization compared to Adam?

To address this question, we apply Muon to different transformer components. Our experiments consistently show that the more rapid convergence of the validation loss using the Muon optimizer compared to Adam is primarily due to the former’s focus on the value-output (VO) matrices of the attention mechanism and the Feed-Forward Networks (FFN) blocks. This leads to our first key insight: VO and FFN blocks, which serve as the primary associative memory stores in the model (Geva et al., 2020; Bietti et al., 2023), are the main beneficiaries of Muon’s optimization strategy. This naturally raises the following question:

What structural features of the transformer allow Muon to optimize these components more effectively?

Building on the previous finding, we address this question by linking Muon’s update mechanism to the learning dynamics of associative memory. Prior work suggests that the behavior of these memory components can be modeled as a sum of outer products representing stored facts (Meng et al., 2022a). Since Muon’s update assigns equal update magnitudes to each outer product of the gradient corresponding to orthogonal singular directions, we hypothesize that it optimizes associative memories more effectively than Adam because: (i) Muon’s spectral normalization procedure balances the rates of learning of these outer products. (ii) Thus, when training on heavy-tailed data (i.e., where a few facts appear much more frequently than the rest), Muon reduces the dominance of frequent (head) facts and enables more effective learning from infrequent (tail) facts compared to Adam.

We validate these hypotheses through a combination of empirical analysis and theoretical modeling. Empirically, we conduct two experiments. First, we measure the singular value spectra of the weight matrices and show that Muon consistently yields more isotropic representations than Adam, indicating that its normalization prevents spectral energy from concentrating in dominant components. Second, we evaluate the performance of both optimizers on a knowledge-intensive, heavy-tailed task to demonstrate the practical benefit of Muon’s more balanced updates: while both optimizers perform well on head classes (frequent in training data), Muon outperforms Adam on tail classes (rare in training data), leading to more stable and uniform convergence.

Theoretically, we focus on a one-layer linear associative memory model to rigorously explain these empirical findings. Under class imbalance in the training data, mimicking a heavy-tailed distribution, we show that Muon maintains balanced learning across classes, regardless of the feature embeddings. In contrast, we prove that Adam’s performance is unstable and strongly dependent on the embedding structure, which can lead to large disparities in learning error across classes. By closely examining the parameter updates, we find that the singular spectrum of weight matrices trained by Muon is nearly isotropic, whereas Adam’s is uneven.

Summarizing the empirical and theoretical findings, we identify a clear mechanism underlying Muon’s superiority:

The Muon update rule is aligned with the outer-product structure of linear associative memories, enabling more balanced and effective learning of tail classes in heavy-tailed distributions as compared with Adam.

### 2 Related Works

Adam, proposed by Kingma & Ba (2015), was designed to make Gradient Descent (GD) adaptive to the complex optimization landscape of neural networks. Existing works analyze Adam from two primary perspectives: online optimization and feature learning. The online convex optimization view focuses on Adam’s properties when optimizing convex or non-convex loss functions. From this perspective, Chen et al. (2019) and Zhou et al. (2018) derive non-convex convergence results for Adam, and a series of subsequent works continuously relaxed the required assumptions for Adam’s convergence while tightening its convergence rate. For instance, Zou et al. (2019) proposes a set of easy-to-verify sufficient conditions for Adam’s update rules to guarantee convergence. D´efossez et al. (2020) derives the tightest dependency on the heavy ball momentum

parameters. More recently, Zhang et al. (2022) demonstrates that Adam can converge without modification of its procedures, and Li et al. (2023) relaxes the smoothness assumption by employing an adaptive Lipschitz constant for gradients. The feature learning view, on the other hand, highlights the relationship between deep learning characteristics and Adam, focusing more on how Adam’s mechanisms influence the properties of learned features within deep networks. For example, Pan & Li (2023) examines the sharpness of GD and Adam and relates Adam’s superiority to its low sharpness. Kunstner et al. (2024) finds that Adam is better at learning heavy-tailed distributions than GD. Furthermore, Zhang et al. (2024a) shows that Adam is adaptive to heterogeneous Hessian structures, thus optimizing faster than GD. More literature on Adam is included in the survey by Abdulkadirov et al. (2023).

Muon, proposed by Jordan et al. (2024), applies spectral normalization of the gradient to update parameters. At a high level, Muon can be understood as steepest descent with respect to the matrix operator norm (Bernstein & Newhouse, 2024). Alternatively, it can be viewed as maximizing the feature update subject to a parameter update constraint (Yang et al., 2023). Experiments show that Muon consistently outperforms Adam across diverse model sizes and architectures, including dense transformers and Mixtureof-Experts (Liu et al., 2025; Jordan et al., 2024). Building on this, Si et al. (2025) introduces an adaptive variant of Muon. To explain its advantages, Lau et al. (2025) introduces a unifying preconditioning framework, distinguishing optimizers that address curvature anisotropy (like Adam) from those that address gradient anisotropy (like Muon), and proposes a generalized optimizer class named PolarGrad. Sato et al. (2025) and Shah et al. (2025) examine the critical batch size of Muon, while other works analyze its convergence in convex and non-convex settings (Li & Hong, 2025; An et al., 2025; Kovalev, 2025; Pethick et al., 2025; Shen et al., 2025). Concurrently, Vasudeva et al. (2025) study Muon on shallow ViTs for computer vision, grounding their results for gradient descent and Muon in linear regression. In contrast, we investigate Muon in the context of LLMs, focusing on its effects on associative memory in next-token prediction.

Associative Memories have a long history in neural network design and knowledge storage (Hopfield, 1982; Kohonen, 2009; Willshaw et al., 1969). They have inspired architectures capable of retaining long histories, including RNNs (Orvieto et al., 2023) and Mamba (Zhang et al., 2024b). With the success of transformers, recent work has examined them through the lens of associative memories. Geva et al. (2020) and Dai et al. (2021) show that feed-forward modules store knowledge in Wout, while Bietti et al. (2023) demonstrates that the attention output matrix WO also encodes associations of knowledge. Building on these findings, a series of works edit knowledge directly by modifying these weights (Meng et al., 2022b; Fang et al., 2024). Beyond empirical results, theoretical analyses have further clarified how transformers leverage associative memories: Bietti et al. (2023) conducts a dynamic analysis of memory formation, while Nichani et al. (2024) constructs explicit associative memory mechanisms in both attention and feed-forward modules.

### 3 Preliminaries

In this section, we first introduce the notations and then present the Muon optimizer, the transformer architecture, and their associative memory components.

Notations. Let [N] for the set {1,...,N}. For a matrix X ∈ Rd×N, Xi is its i-th column and X:,−1 is its last column. IK,K is the K × K identity matrix, IK is all-ones vector and JK,K is the all-ones matrix. ⊙ denotes the element-wise product.

Muon is an optimizer tailored for matrix parameters that replaces the raw (or momentum) gradient with the sum of its normalized orthogonal factors, producing a scale-invariant, norm-controlled update direction (Jordan et al., 2024). For a weight matrix W ∈ Rm×n at step t, let Gt = ∇WL(Wt) denote its gradient. Muon maintains a momentum accumulator of gradients as Bt = µBt−1 + Gt with B0 = 0, and µ ∈ [0,1). At each step, Muon computes the Singular Value Decomposition (SVD) of Bt as Bt = UtStVt⊤ with Ut ∈ Rm×r

, Vt ∈ Rn×r

, where rt = rank(Bt), and form the nearest (semi)–orthogonal matrix Ot = UtVt⊤. Then Muon updates the parameter as Wt+1 = Wt−ηtOt. In practice, one can approximate Ot using a fixed number (e.g., 5) of Newton–Schulz iterations applied to Bt(Bt⊤Bt)−1/2, which avoids the full SVD while preserving the scale normalization effect. Bernstein & Newhouse (2024) interprets Muon as steepest gradient descent with respect to the matrix operator norm. Concretely, the Muon update Ot can be characterized (up to a

t

t

scalar factor) as the solution to

λ 2∥W∥2ℓ2→ℓ2 ,

⟨Bt,W⟩F +

argmin

W

where ∥·∥ℓ2→ℓ2 denotes the matrix operator norm, i.e., the largest singular value, and λ ∈ R determines the step size. By contrast, as explained in Appendix A, Adam can be viewed as steepest gradient descent with respect to the vector norm. However, this perspective alone does not explain why using the matrix operator norm rather than the vector norm leads to better performance.

Transformers serve as the backbone of LLMs. It predicts the probability of the next token given a sequence of N tokens (Radford et al., 2019). A sequence of N tokens is embedded into a matrix X(0) ∈ Rd×N. The first layer takes X(0) as the input, and each subsequent layer takes the previous layer’s output as its input. Every layer ℓ ∈ [L] processes its input through two sequential components: an attention module and a FFN module. The attention module computes

H(ℓ) = X(ℓ−1) +

H

WO,h(ℓ) WV,h(ℓ) X(ℓ−1)sm X(ℓ−1),⊤WK,h(ℓ),⊤WQ,h(ℓ) X(ℓ−1) , (3.1)

h=1

where sm(·) is the column-wise softmax operator, H is the number of attention heads, WQ,h(ℓ) ,WK,h(ℓ) ∈ Rd

k×d capture token relationships, and WV,h(ℓ) ∈ Rd

v×d,WO,h(ℓ) ∈ Rd×d

apply linear transformations. The feedforward module then updates the representation as

v

X(ℓ) = H(ℓ) + ff(H(ℓ),Win(ℓ),Wout(ℓ)) = H(ℓ) + Wout(ℓ)σ(Win(ℓ)H(ℓ)), (3.2)

where σ(·) is the element-wise activation function, and Win(ℓ) ∈ Rd

f×d,Wout(ℓ) ∈ Rd×d

are learnable parameters. In addition to the FFN in Eqn. (3.2), a gated variant is widely used in Large Language Models (LLM)s (Touvron et al., 2023; Hui et al., 2024), which replaces the standard form with

f

ffgate(H(ℓ),Win(ℓ),Wout(ℓ),Wgate(ℓ) ) = Wout(ℓ) σ(Win(ℓ)H(ℓ)) ⊙ (Wgate(ℓ) H(ℓ)) ,

where ⊙ is the Hadamard product, and Wgate(ℓ) ∈ Rd

f×d is an additional mapping. After L layers, the final hidden state of the last token, X−(L1), is projected by the language model head Ehead ∈ RK×d to produce logits EheadX−(L1), which has a vocabulary of size K.

Associative memory refers to architectures that store and retrieve patterns based on learned associations between inputs and outputs. Recent research has examined linear associative memory in LLMs. Specifically, consider a triplet (s,r,o), where s is the subject, r the relation, and o the object (e.g., s =“The United Nations headquarters”, r =“is located in”, o =“New York City”). A linear associative memory W maps a key vector es encoding (s,r) to a value vector eo encoding o, such that eo = Wes holds for all possible (s,r,o) (Nichani et al., 2024). Under the orthogonality of embeddings es and eo, W can be expressed as W = Ki=1 eo

e⊤s

, where the summation is taken over the index i of K facts. Prior work has investigated associative memory in both attention and FFN modules. For the attention module, Bietti et al. (2023) showed that the parameter WO can serve as a linear associative memory when WV is fixed. Since WO and WV play symmetric roles, we also treat WV as part of the associative memory parameters. For FFN, prior work on knowledge editing (Geva et al., 2020; Dai et al., 2021; Meng et al., 2022a,b) has shown that this module functions as an associative memory and can be well approximated by linear associative memory models. Thus, throughout this paper, we refer to WO, WV , and FFN in LLMs as the associative memory parameters.

i

i

### 4 Main Results

#### 4.1 Associative Memories Are Main Beneficiaries of Muon

In the Muon implementation (Jordan et al., 2024), the token embedding and language model head parameters are optimized with Adam rather than Muon. This observation motivates a closer examination of the

7.0

Muon(All Attn, FFN)

All Adam

6.5

Muon(QK Attn)&Adam(VO Attn, FFN) Muon(VO Attn)&Adam(QK Attn, FFN) Muon(V Attn)&Adam(QKO Attn, FFN) Muon(O Attn)&Adam(QKV Attn, FFN) Muon(Win)&Adam(All Attn, Wout)

6.0

ValidationLoss

5.5

Muon(Wout)&Adam(All Attn, Win, Wgate)

5.0

4.01

4.5

3.57

8000 10000

4.0

3.5

0 2000 4000 6000 8000 10000

Training Steps

(a) Independent Blocks with Non-gated FFN

Muon(All Attn, FFN)

7.0

All Adam

Muon(QK Attn)&Adam(VO Attn, FFN) Muon(VO Attn)&Adam(QK Attn, FFN) Muon(V Attn)&Adam(QKO Attn, FFN) Muon(O Attn)&Adam(QKV Attn, FFN)

6.5

6.0

ValidationLoss

Muon(Win)&Adam(All Attn, Wout, Wgate) Muon(Wgate)&Adam(All Attn, Wout, Win) Muon(Wout)&Adam(All Attn, Win, Wgate)

5.5

5.0

3.99

4.5

3.51

8000 10000

4.0

3.5

0 2000 4000 6000 8000 10000

Training Steps

###### (b) Independent Blocks with Gated FFN

7.0

Muon(All Attn, FFN)

All Adam

6.5

Muon(VO Attn, FFN)&Adam(QK Attn)

Muon(VO Attn, Win)&Adam(QK Attn, Wout) Muon(VO Attn, Wout)&Adam(QK Attn, Win)

6.0

ValidationLoss

Muon(V Attn, FFN)&Adam(QKO Attn) Muon(O Attn, FFN)&Adam(QKV Attn)

5.5

5.0

4.01

4.5

3.57

8000 10000

4.0

3.5

0 2000 4000 6000 8000 10000

Training Steps

(c) Combined Configuration with Non-gated FFN

7.0

Muon(All Attn, FFN)

All Adam

Muon(VO Attn, FFN)&Adam(QK Attn)

6.5

Muon(VO Attn, Win, Wgate)&Adam(QK Attn, Wout) Muon(VO Attn, Wout, Wgate)&Adam(QK Attn, Win) Muon(VO Attn, Win, Wout)&Adam(QK Attn, Wgate)

6.0

ValidationLoss

Muon(V Attn, FFN)&Adam(QKO Attn) Muon(O Attn, FFN)&Adam(QKV Attn)

5.5

5.0

3.99

4.5

3.51

8000 10000

4.0

3.5

0 2000 4000 6000 8000 10000

Training Steps

(d) Combined Configuration with Gated FFN

- Figure 1: Validation loss comparison on the 160M NanoGPT model with ungated and gated FFN. Panels (a) and (b) show the “Independent Blocks” results, where individual components are optimized separately, for models with ungated and gated FFN, respectively. Panels (c) and (d) show the “Combined Configurations” results, where multiple components are optimized jointly, again for ungated and gated FFN models.

efficacy of Muon across different components of the transformer architecture. In this section, we identify the transformer components that benefit most from Muon by measuring validation loss on the FineWeb dataset (Penedo et al., 2024) using a 160M NanoGPT model. We adopt a two–stage protocol. First, in the “Independent Blocks” setting, we apply Muon to a single block at a time while keeping all other blocks on Adam, covering the attention projections WQ,WK,WV ,WO and the feed-forward matrices Win, Wout. Second, in the “Combined Configurations” setting, we apply Muon to the most impactful subsets identified in the first stage to examine whether a partial application can recover the performance gains of full Muon. As introduced in Section 3, we evaluate both gated and non-gated FFN variants of NanoGPT. The experimental details are in Appendix B.

Figure 1 and Table 1 present our results. We first examine the independent-block experiments for attention. In both non-gated and gated FFN settings (Figures 1(a) and 1(b)), the VO weights WV ,WO (Muon on VO & Adam on QK and FFN) show substantially larger gains under Muon than the QK weights WQ,WK (Muon on QK & Adam on VO and FFN). Notably, applying Muon to only WV or only WO already yields much larger gains than applying it to QK. Between the two, WO performs comparably in the gated FFN setting and better in the non-gated setting. For the FFN, we find that Win, Wgate, and Wout all benefit from Muon, with Wout yielding stronger improvements than Win.

After identifying the importance of each module, the combined configurations aim to quantify their contributions to the full Muon. Guided by the independent-block findings, we first observe that VO+FFN

Table 1: Validation loss at 10000 training steps. The order of methods follows the legends in the original figure. The best result within each configuration block is highlighted in bold. A dash (—) indicates the configuration was not compatible for that FFN type.

##### Method Description Non-gated FFN Gated FFN

(from Fig. 1(a) and 1(c)) (from Fig. 1(b) and 1(d))

Baselines Muon(All Attn, FFN) 3.5654 3.5125 All Adam 3.9242 3.8837

Independent Blocks Muon(QK Attn)& Adam(VO Attn, FFN) 3.8925 3.8518 Muon(VO Attn)& Adam(QK Attn, FFN) 3.7644 3.6874 Muon(V Attn)& Adam(QKO Attn, FFN) 3.8301 3.7482 Muon(O Attn)& Adam(QKV Attn, FFN) 3.7712 3.7604 Muon(Win)& Adam(All Attn, Wout) 3.7170 Muon(Wout)& Adam(All Attn, Win, Wgate) 3.7023 3.7843 Muon(Win)& Adam(All Attn, Wout, Wgate) — 3.7918 Muon(Wgate)& Adam(All Attn, Wout, Win) — 3.7847 Combined Configuration Muon(VO Attn, FFN)& Adam(QK Attn) 3.5858 3.5312

Muon(VO Attn, Win)& Adam(QK Attn, Wout) 3.6778 Muon(VO Attn, Wout)& Adam(QK Attn, Win) 3.6054 Muon(VO Attn, Win, Wgate)& Adam(QK Attn, Wout) — 3.5681 Muon(VO Attn, Wout, Wgate)& Adam(QK Attn, Win) — 3.5833 Muon(VO Attn, Win, Wout)& Adam(QK Attn, Wgate) — 3.5482 Muon(V Attn, FFN)& Adam(QKO Attn) 3.6702 3.7185 Muon(O Attn, FFN)& Adam(QKV Attn) 3.6042 3.5634

already closely tracks—and in our runs nearly recovers—the full-Muon trajectory in Figures 1(c) and 1(d). This indicates that applying Muon to QK contributes little to its overall performance. The small remaining gap between full Muon and VO+FFN may arise because VO+FFN uses the same learning rate as full Muon without further tuning. This gap could likely be reduced by adjusting the learning rate specifically for VO+FFN. Importantly, the underperformance of QK is not attributable to the logit explosion observed by Team et al. (2025) in large Mixture-of-Experts (MoE) models; in our setting, logit values remain stable, as shown in Appendix C.1.

To isolate the contributions of WO and WV within VO+FFN, we perform ablations starting from the VO+FFN setting: we keep Muon on FFN and on only one of WO or WV , reverting the other to Adam (i.e., V+FFN and O+FFN). Both ablations degrade performance, with the V+FFN variant dropping more, indicating that WO is more influential than WV . We apply the same analysis to FFN. The results reveal architectural sensitivity: in the ungated setting (Figure 1(c)), VO+Wout nearly recovers the fullMuon trajectory, whereas in the gated setting (Figure 1(d)) the same combination falls short. Nevertheless, both analyses underscore the central role of Wout in FFN. Overall, applying Muon to VO+FFN is critical for recovering full-Muon performance, though the extent of recovery still depends on architectural design (ungated vs gated). The results from training a 0.7B model in Appendix C.2 show similar findings.

Observation 1: Muon is most effective when applied to VO and FFN; in particular, applying Muon to only VO+FFN almost recovers the full-Muon trajectory.

We emphasize that this observation is not a trivial consequence of parameter counting; although QK and VO have the same number of parameters, VO proves substantially more influential in our results.

As introduced in Section 3, prior works discover that the common role of VO and FFN is that they both

serve as the associative memories for transformers, which store facts and knowledge. Furthermore, Bietti et al. (2023) and Meng et al. (2022a) show that the linear associative memories well approximate them. Specifically, for a set of facts represented by key-value pairs {(es

)}, the memory matrix W can be constructed as a sum of outer products, i.e., W = Ki=1 eo

,eo

i

i

e⊤s

, where the summation is taken over the index i of K facts.

i

i

Learning linear associative memories is particularly well-suited to Muon’s update mechanism. Intuitively, the gradient G ∈ Rd×d of the loss with respect to the linear associative memory weight W can be expressed as a sum of outer products. Muon computes its update (without momentum) by taking the SVD of the gradient,

G = USV ⊤ = di=1 siuivi⊤, and forming the orthogonal factor O = UV ⊤ = di=1 uivi⊤. Comparing this with the linear associative memory Ki=1 eo

e⊤s

, we see that Muon updates all “orthogonal” facts at the same rate. Later, we will see that the singular values S encode the frequencies of knowledge in the training data in Sections 4.3 and 5. This implies that Muon can learn both frequent and infrequent facts uniformly.

i

i

We verify this insight from two perspectives. First, from the view of weight spectra, the weight matrices learned with Muon exhibit a more isotropic singular-value spectrum than those learned with Adam, indicating that knowledge, regardless of its frequency, is represented with comparable magnitude. Second, at the level of overall knowledge acquisition, Muon yields more balanced learning across entities and frequencies (head and tail) than Adam. We examine these two consequences in the following sections.

#### 4.2 Muon Consistently Learns More Isotropic Weights Than Adam

To validate that Muon can shape the weight matrices more evenly across directions, we conducted a spectral analysis of them. For a weight matrix with n non-zero singular values σ = (σ1,σ2,...,σn) arranged in descending order, we define the normalized singular energy distribution q = (q1,q2,...,qn), where each component qi is qi = σi2/ nj=1 σj2. This distribution represents the fraction of energy captured by each corresponding singular vector. Based on this, we introduce several metrics to characterize the isotropy of the spectrum:

- • Normalized SVD Entropy. This metric, adapted from Alter et al. (2000), quantifies the uniformity of the singular energy distribution. A higher entropy value indicates a more isotropic matrix where energy is distributed evenly across many directions. It is defined as the Shannon entropy of the distribution q, normalized by the maximum possible entropy: Hnorm(σ) = −log1n ni=1 qi log qi.

- • Effective Rank. The effective rank (Roy & Vetterli, 2007) provides a continuous measure of the number of significant singular dimensions used by the matrix. It is calculated as the exponentiation of the unnormalized Shannon entropy, which corresponds to the perplexity of the energy distribution: eRank(σ) = exp(− ni=1 qi log qi).
- • Top-k Energy Fraction. This metric measures the concentration of energy within the Top-k principal singular components. Assuming the singular values are sorted in descending order (σ1 ≥ σ2 ≥ ··· ≥

σn), it is the cumulative sum of the first k energy fractions: TopEk(σ) =

k

- i=1 σi2 n

- j=1 σj2.

- • Eigenvalue Quantile Ratio. To measure the spread of the singular energy distribution while being robust to extreme outliers, we compute the ratio of the 75th percentile (Q3) to the 25th percentile

2 i })

(Q1) of the eigenvalues {σi2}ni=1: Q75/25(σ) = Q3({σ

Q1({σi2}).

These metrics assess the isotropy of the distribution by capturing both the evenness of values (normalized SVD entropy, effective rank) and the rate of decay (Top-k energy fraction, quantile ratio). Intuitively, more isotropic weights correspond to larger values of normalized SVD entropy and effective rank, and smaller Top-k energy fraction and eigenvalue quantile ratio.

The spectral analysis in Figure 2, focusing on the key associative memory components from Observation 1, shows that Muon systematically reshapes the learned weight matrices relative to Adam. The results, averaged over 10 random seeds, demonstrate that: (i) In both gated and ungated FFN architectures, Muon produces a much more isotropic singular spectrum than Adam from the start of training, whereas Adam’s isotropy fluctuates significantly over the course of optimization. (ii) The isotropy of Muon is stable across random initializations, as indicated by the negligible error bars in Figure 2, while Adam is highly sensitive to initialization. These findings suggest that Muon consistently promotes richer and more diverse features

SVD entropy ( )

Top10E ( )

1.0

0.8

0.8

SVDentropy()

Top10E()

0.6

0.6

0.4

0.4

0.2

Muon(All Attn, FFN)

All Adam

0.2

0.0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

eRank ( )

Q75/Q25 Ratio ( )

- 101

- 102

- 103

400

Q75/Q25Ratio()

300

eRank()

200

100

0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

(a) VO(Non-gated FFN)

SVD entropy ( )

Top10E ( )

1.0

1.0

0.8

0.8

SVDentropy()

Top10E()

0.6

0.6

0.4

0.4

0.2

0.2

Muon(All Attn, FFN)

All Adam

0.0

0.0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

###### eRank ( )

Q75/Q25 Ratio ( )

- 101

- 102

- 103

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

600

Q75/Q25Ratio()

400

eRank()

200

0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

(c) Wout(Non-gated FFN)

SVD entropy ( )

Top10E ( )

1.0

0.8

0.8

SVDentropy()

Top10E()

0.6

0.6

0.4

0.4

0.2

Muon(All Attn, FFN)

0.2

All Adam

0.0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

eRank ( )

Q75/Q25 Ratio ( )

400

- 101

- 102

- 103

Q75/Q25Ratio()

300

eRank()

200

100

0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

###### (b) VO(Gated FFN)

SVD entropy ( )

Top10E ( )

1.0

0.8

0.8

SVDentropy()

0.6

Top10E()

0.6

0.4

0.4

0.2

0.2

Muon(All Attn, FFN)

All Adam

0.0

0.0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

eRank ( )

Q75/Q25 Ratio ( )

- 101

- 102

- 103

600

Q75/Q25Ratio()

400

eRank()

200

0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

(d) Wout(Gated FFN)

- Figure 2: Spectral Dynamics of Transformer Weight Matrices During Training. Each panel reports four metrics characterizing singular value distributions: SVD entropy, Top10E, eRank, and Q75/Q25 ratio. The four subplots correspond to different weight matrix groups: (a) VO, (b) VO (Gated FFN), (c) Wout, and (d) Wout (Gated FFN).

in the model’s most critical memory components, a conclusion we summarize below. The results for other weights are in Appendix C.3.

Observation 2: Muon consistently yields more isotropic weight matrices with broadly distributed spectral energy than Adam, both throughout training and across random initializations, thereby supporting richer feature representations.

Empirically, we also find that Muon learns more isotropic QK weights than Adam. However, as discussed in Section 4.1, QK weights are not part of the linear associative memory mechanism and are therefore not expected to benefit from the isotropic property of the weight matrices.

Our results differ fundamentally from the spectral analysis in Liu et al. (2025) for three reasons. First, we decompose the parameters according to associative memories, whereas Liu et al. (2025) aggregates them, obscuring the essential components driving Muon’s behavior. Second, we investigate the instability of Adam under random initialization (i.e., random seeds), which we further establish theoretically in Section 5. Finally, our analysis focuses on dense architectures, while Liu et al. (2025) centers on Mixture-of-Experts (MoE) models.

Class Groups

- 101

- 102

- 103

- 104

- 105

- Group 0

- Group 1

- Group 2

- Group 3

- Group 4

- Group 5

- Group 6

- Group 7

- Group 8

- Group 9

- Group 10

- Group 11

- Group 12

- Group 13

- Group 14

- Group 15

#QA Samples

100 101 102 103 104

Class index (sorted)

(a) Sample/class

1.0

0.8

Accuracy(FTA)

0.6

0.4

0.2

0.0

0 2000 4000 6000 8000 10000

Training Steps

(d) SGD+Momentum

1.0

0.8

Accuracy(FTA)

0.6

0.4

0.2

0.0

0 2000 4000 6000 8000 10000

Training Steps

(b) Muon

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1.0

0.8

Accuracy(FTA)

0.6

0.4

0.2

0.0

0 2000 4000 6000 8000 10000

Training Steps

(e) Muon(VO,FFN)& Adam(QK)

1.0

0.8

Accuracy(FTA)

0.6

0.4

0.2

0.0

0 2000 4000 6000 8000 10000

Training Steps

###### (c) Adam

1.0

0.8

Accuracy(FTA)

0.6

0.4

0.2

0.0

0 2000 4000 6000 8000 10000

Training Steps

(f) Muon(QK)& Adam(VO,FFN)

- Figure 3: Performance comparison of different optimizers for transformers with non-gated FFN on a heavytailed knowledge task. (a) Sample distribution per class, following a power law. (b–d) Performance of Muon, Adam, and SGD+Momentum. (e) Muon applied to VO and FFN, with Adam on QK. (f) Muon applied to QK, with Adam on VO and FFN.

#### 4.3 Muon Acquires Knowledge More Evenly Compared To Adam

Our previous findings indicate that the Muon optimizer is particularly important for the associative memory components of the model, where it learns more isotropic weights. To examine the overall effects of learning associative memories, we turn to a knowledge-intensive question-answering (QA) task. The task is based on a synthetic QA dataset containing biographical information (e.g., name, birthday, and company) for over 200,000 individuals (Allen-Zhu & Li, 2024). To capture the heavy-tailed nature of real-world knowledge, we control the frequency of each individual’s appearance in the training set so that it follows a power-law distribution (Figure 3(a)), thereby inducing varying levels of difficulty in learning knowledge about different individuals. A 160M NanoGPT model is trained to answer questions about this biographical information. The performance is evaluated via the First Token Accuracy (FTA) on the answers, following Allen-Zhu & Li (2024). Further details on the dataset are provided in Appendix B.2. We include SGD as a baseline for Adam and Muon.

The results in Figure 3 lead to an unequivocal conclusion about the efficacy of different optimizers under data imbalance. In high-frequency (head) classes, all optimizers perform well, with Muon, Adam, and even SGD+Momentum rapidly reaching near-perfect accuracy (Figure 3(b–d)). Consistent with prior work on heavy-tailed distributions (Kunstner et al., 2024), Adam maintains a clear advantage over SGD, which struggles with tail classes. Our key finding, however, is that Muon substantially outperforms Adam on low-frequency (tail) data, achieving faster and more uniform convergence across all frequencies. Moreover, the consistently tighter error bars for Muon—especially relative to Adam—reflect lower variance and a more stable learning process.

Furthermore, the hybrid configurations in Figure 3(e–f) clarify where Muon matters most. Applying Muon to VO+FFN (with QK on Adam) yields strong gains on rare classes and markedly reduces the head–tail gap, whereas applying Muon only to QK (with VO+FFN on Adam) yields only limited improvement. This mirrors Observation 1: VO+FFN is the most effective target set, as it concentrates the model’s associative memory. Results for the gated FFN, which show the same pattern, are provided in Appendix C.5. We summarize these findings as Observation 3.

Avg.AnglesBetweenEmbeddings(Degrees)

90

75

60

45

30

Ei Ei

15

- 0

5 10 15 20 25 Layer Index

1e+0

1e-1

1e-2

1e-3

()W

1e-4

GD, De/Coupled

10 3

1e-5

SignGD, Coupled

SignGD, Decoupled

1e-6

Muon, Decoupled

Muon, Coupled

10 4

5 × 10 2 10 1

0

10 1 100 101 Population Loss

1e+0

1e-1

1e-2

1e-3

()W

1e-4

GD, De/Coupled

10 3

1e-5

SignGD, Coupled

SignGD, Decoupled

1e-6

Muon, Decoupled

Muon, Coupled

10 4

5 × 10 2 10 1

0

10 1 100 Population Loss

(a) Average Angles Between Ei/ Ei

(b) One-step Optimization Results

(c) Multi-step Optimization Results

- Figure 4: (a) Average angles between Ei or Ei in FFN at layers 5, 10, 15, 20, 25 of Llama3-8b-instruct. (b) Results of one-step GD, SignGD, and Muon with both coupled and decoupled embeddings. For GD, the outcomes under the two embedding types coincide. (c) Results of multi-step GD, SignGD, and Muon with both coupled and decoupled embeddings.

Observation 3: In heavy-tailed, knowledge-intensive tasks, Muon matches Adam’s strong performance in the head classes while substantially improving learning on tail classes, narrowing the head-tail gap and accelerating convergence.

- 5 Case Study of One-Layer Models

We now analyze three optimizers—Adam, Muon, and GD (as a baseline)—to complement the preceding empirical observations. We first introduce an abstraction that captures their key dynamics and then present both empirical and theoretical results. As shown in Eqns. (3.1) and (3.2), a structural property of associative memory parameters is that their output is added directly to the hidden states, which are subsequently processed by the language model head. Motivated by this property, our abstraction retains the associative memory and language model head, while replacing all preceding modules with given feature embeddings.

Consider K triplets {(si,ri,oi)}Ki=1, where subject-relation pairs (si,ri) and objects oi are embedded into the columns of matrices E ∈ Rd

s×K and E ∈ Rd

o×K, respectively. A linear associative memory W ∈ Rd

o×ds predicts the object for a query Ek with probabilities fW(Ek) = sm( E⊤WEk) ∈ RK. The objective is to minimize the population cross-entropy loss L(W) = − Kk=1 pk log[fW(Ek)]k, where pk is the frequency or probability of the k-th triplet. We consider three optimizers: GD, Adam, and Muon.

- • GD updates the parameters according to the gradient ∇WL(W) as WtGD+1 = WtGD − ηt+1∇WL(WtGD).
- • For Adam, we switch off the exponential moving averages (EMA), i.e., β1 = β2 = 0, following the practice in existing theoretical works (Kunstner et al., 2024; Bernstein & Newhouse, 2024). Under this

setting, Adam reduces to sign-GD as WtSignGD+1 = WtSignGD −ηt+1 sign ∇WL(WtSignGD) , where sign(·) denotes the element-wise sign operator.

- • For Muon, we also disable its momentum and analyze the update WtMuon+1 = WtMuon−ηt+1Utnorm(Σt)Vt⊤, where norm(·) normalizes all non-zero elements to 1 (element-wise), and UtΣtVt⊤ is the SVD of the gradient ∇WL(WtMuon).

All these optimizers adopt the zero initialization that W0 = 0d

o,ds. We then state the assumptions for our results.

- Assumption 5.1. The embeddings E and E are orthonormal, i.e., E⊤E = E⊤ E = IK,K.

The unit-norm requirement rules out feature-level imbalance, which would otherwise couple with the imbalance induced by pk and complicate the analysis. Our techniques can be directly applied even without this unit-norm requirement. The orthogonality assumption is intuitively plausible, as different concepts are

independent and do not influence one another. We empirically verify this on Llama3-8b-instruct (Dubey et al., 2024). Following Fang et al. (2024), we extract Ei and Ei in FFN across layers for 3,000 knowledge items of Counterfact (Meng et al., 2022a) and compute average angles between them (see Appendix B.3 for details). As shown in Figure 4(a), these angles are near 90◦, confirming approximate orthogonality. For K independent concepts, orthogonality requires dr,ds ≥ K. For simplicity, we set dr = ds = K in what follows.

- Assumption 5.2. The first L triplets share the same probability and together contribute a total mass of α, i.e., pk = α/L for k ∈ [L]. The remaining triplets also share the same probability and together contribute a total mass of 1 − α, i.e., pk = (1 − α)/(K − L) for k > L.

This assumption states that the data imbalance is between two classes among the K triplets. Defining β = L/K, the ratio α/β quantifies the degree of balance: if α > β, the first L triplets appear more frequently during learning, and vice versa. This simplified two-class setting is sufficient to capture the primary differences between optimizers; the multi-class case follows directly from our proof by extending the SVD calculation.

#### 5.1 Experimental Results

Under Assumptions 5.1 and 5.2, we evaluate GD, SignGD, and Muon for α = 0.8, β = 0.2, considering two embeddings for E and E: (i) support-decoupled: the supports (indices of non-zero entries) of different Ei or Ei are disjoint; (ii) support-coupled: supports may overlap. We study two optimization protocols, initializing W0 = 0d

o×ds: (i) one-step: take a single update with a scaled step size to obtain a range of L(W) values; (ii) multi-step: run multiple updates to reduce L(W), varying the number of steps. Experimental details are in Appendix B.4. To quantify learning imbalance across K knowledge items, we examine the relationship between population loss L(W) and maximal probability gap ∆(W) := maxi,j∈[K][fW(Ei)]i−[fW(Ej)]j, where [fW(Ei)]i denotes the probability assigned to the correct item i. A larger ∆(W) indicates greater imbalance.

Across both optimization-step protocols and embeddings (Figures 4(b), 4(c)), we observe that

- • For all optimizers, ∆(W) first increases and then decreases as L(W) decreases. Early in training, when correct probabilities are near 0, imbalance is pronounced; later, when all items are well learned (e.g., probabilities ≥ 0.9), imbalance diminishes.
- • For both embedding regimes, GD and Muon behave consistently: GD exhibits a substantial imbalance, whereas Muon remains much more balanced across items.
- • SignGD also demonstrates unstable behavior; its imbalance resembles GD in the coupled embedding case and Muon in the decoupled embedding case.

Because one-step and multi-step experiments align qualitatively, we first analyze the one-step setting for clarity. This simplification is common in theoretical studies of neural network dynamics (Ba et al., 2022; Dandi et al., 2023), and our techniques extend directly—albeit with more algebra—to the multi-step case. As a demonstration, Theorem 5.4 provides a multi-step analysis of Muon.

#### 5.2 Theoretical Results

For the one-step analysis, define the smallest correct-class probability across all knowledge items, under the condition that at least one item achieves the correct-class probability of at least 1 − ϵ as

ϱϵopt = inf η≥0

(Ek)]k ≥ 1 − ϵ, Wη = W0 − η · Gopt(W0) , (5.1)

min

[fW

(Ek)]k max k∈[K]

[fW

η

η

k∈[K]

where opt ∈ {GD,SignGD,Muon} and Gopt(W0) denotes the parameter update of the optimizer “opt” at W0; and Wη denotes the parameter obtained after one step of optimizer “opt” with step size η starting from W0, i.e., Wη = W0 − η · Gopt(W0). Specifically, we denote

GGD(W0) = ∇WL(W0), GSignGD(W0) = sign(∇WL(W0)), GMuon(W0) = U0norm(Σ0)V0⊤, where U0Σ0V0⊤ is the SVD of ∇WL(W0). Note that ϱϵopt ∈ [0,1 − ε] and ∆(W) are related as ∆(W) =

- 1 − ϵ − ϱϵopt ≥ 0. When ϱϵopt ≈ 1 − ϵ, opt achieves balanced learning across facts; in contrast, when ϱϵopt ≈ 0, imbalanced learning ensues.

- Theorem 5.3. If Assumptions 5.1 and 5.2 hold, with fixed α,β such that α ̸= β, and K goes to infinity, we obtain the following results for one-step GD, Muon, and Adam.

- • For GD, for any E and E satistifying Assumption 5.1, we have

ϱϵGD = O(ϵ−r(α,β)Kr(α,β)−1), where r(α,β) = min

α(1 − β) β(1 − α)

,

β(1 − α) α(1 − β)

< 1.

- • For Muon, for any E and E satistifying Assumption 5.1, we have

ϱϵMuon ≥ 1 − ϵ 1 + O

log K K

, and GMuon(W0) = − EE⊤ + O

1 K

EJK,KE⊤ ,

where JK,K ∈ RK×K is the matrix with all elements equal to 1. The big-O notation for matrices means that for A = O(B), each entry satisfies Aij = O(Bij) for all i,j.

- • For Adam, there exist E and E satisfying Assumption 5.1 such that ϱϵSignGD ≥ 1 − ϵ. There also exist E′ and E′ satisfying Assumption 5.1 such that

σmin GSignGD(W0) σmax GSignGD(W0) ≤ 25%,

ϱϵSignGD = O(ϵ−0.7K−0.3), and

where σmax and σmin are the largest and smallest singular values, respectively.

Interpretation of Theorem 5.3. The proof of Theorem 5.3 is provided in Appendix D. We now explain the results for the three optimizers separately. For GD, the quantity r(α,β) ≤ 1 measures the imbalance of the data distribution: r(α,β) = 1 corresponds to perfectly balanced data, while r(α,β) ≪ 1 indicates severe imbalance. The results show that if one set of (s,r,o) triplets is learned with the correct-class probability [fW(Ek)]k of at least 1 − ϵ, then there exists another triplet whose correct-class probability is O(ϵ−r(α,β)Kr(α,β)−1). Thus, GD is highly sensitive to data imbalance: as the training distribution becomes more imbalanced, the dispersion of correct-class probabilities across items increases, i.e., the maximal probability gap ∆(W) grows and mink∈[K][fW(Ek)]k decreases. This mirrors the message in Figure 4(b), 4(c), and Figure 3(d) in Section 4.3.

In contrast, Muon learns in a balanced fashion, unaffected by data imbalance for any embeddings E and E. Our results show that when the best-learned triplet achieves a correct-class probability of at least 1 − ϵ, the worst-learned triplet has a comparable correct-class probability at least 1 − ϵ(1 + O(log K/K)). This justifies Observation 3. Furthermore, consistent with Observation 2, Muon’s update GMuon rule allocates equal strength to all update directions; equivalently, the singular values of GMuon(W0) are nearly identical.

Our analysis shows that Adam’s performance is unstable with respect to the embeddings E and E, as reflected by the large error bars in Observations 2 and 3. Adam’s element-wise normalization disrupts the inherent matrix structure of the gradient. When embeddings of different triplets have disjoint supports (e.g., E = E = IK,K), Adam can optimize parameters in a balanced manner. However, when embeddings overlap, the sign operator in Adam can introduce imbalance. In particular, the worst-optimized triplet may then have correct-class probability O(ϵ−0.7K−0.3). These exponents (0.3,0.7) are intrinsic to Adam’s update under certain embeddings and are independent of α or β. Moreover, the Adam update GSignGD(W0) exhibits pronounced spectral decay—for example, its smallest singular value can be less than 25% of the largest—unlike the nearly uniform singular values of Muon. This spectral decay explains the poor isotropy reported in Observation 2.

In the following, we extend our techniques of one-step analysis to the multi-step analysis of Muon. Parallel to (5.1), we define the infimum correct-class probability for the multi-step optimizer as

ϱϵopt = inf

(Ek)]k ≥ 1 − ϵ, where Wt = Wt−1 − ηt · Gopt(Wt−1) .

min

###### [fW

(Ek)]k max k∈[K]

[fW

t

t

t

k∈[K]

Here, we assume that the learning rates {ηt}t≥1 are determined by a fixed schedule prior to optimization. Although the quantity implicitly depends on this schedule, we omit it from the notation for ϱϵopt for brevity. We emphasize that different schedules may affect the value of t that attains the infimum in ϱϵopt, but they do not influence the balance behavior that we present.

- Theorem 5.4. If Assumptions 5.1 and 5.2 hold, then multi-step Muon achieves

ϱϵMuon ≥ 1 − ϵ 1 + O

log K K

, and GMuon(Wt) = − EE⊤ + O

1 K

EJK,KE⊤ for any t ≥ 0.

The proof is provided in Appendix E. We note that the multi-step analysis of Muon exhibits similar properties to the one-step case presented in Theorem 5.3. Specifically, for any embedding, Muon achieves balanced learning across all items, and its update at each step remains nearly isotropic.

### 6 Conclusion

Our work takes the first step toward unveiling why and how Muon outperforms Adam. Through ablations of Muon’s effect on different Transformer components and by relating these results to the balanced learning of associative memories, we conclude that the Muon update rule is aligned with the outer-product structure of linear associative memories, enabling more balanced and effective learning of tail classes in heavy-tailed distributions. Intuitively, this property of Muon may extend beyond outer products to higher-order tensor products, an exciting direction for future work.

### References

Ruslan Abdulkadirov, Pavel Lyakhov, and Nikolay Nagornov. Survey of optimization algorithms in modern neural networks. Mathematics, 11(11):2466, 2023.

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.3, knowledge capacity scaling laws. arXiv preprint arXiv:2404.05405, 2024.

Orly Alter, Patrick O Brown, and David Botstein. Singular value decomposition for genome-wide expression data processing and modeling. Proceedings of the National Academy of Sciences, 97(18):10101–10106, 2000.

Kang An, Yuxing Liu, Rui Pan, Yi Ren, Shiqian Ma, Donald Goldfarb, and Tong Zhang. Asgo: Adaptive structured gradient optimization. arXiv preprint arXiv:2503.20762, 2025.

Jimmy Ba, Murat A Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, and Greg Yang. High-dimensional asymptotics of feature learning: How one gradient step improves the representation. Advances in Neural Information Processing Systems, 35:37932–37946, 2022.

Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology. arXiv preprint arXiv:2409.20325, 2024.

Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, and Leon Bottou. Birth of a transformer: A memory viewpoint. Advances in Neural Information Processing Systems, 36:1560–1588, 2023.

Xiangyi Chen, Sijia Liu, Ruoyu Sun, and Mingyi Hong. On the convergence of a class of adam-type algorithms for non-convex optimization. In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=H1x-x309tm.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. arXiv preprint arXiv:2104.08696, 2021.

Yatin Dandi, Florent Krzakala, Bruno Loureiro, Luca Pesce, and Ludovic Stephan. How two-layer neural networks learn, one (giant) step at a time. arXiv preprint arXiv:2305.18270, 2023.

Alexandre D´efossez, Le´on Bottou, Francis Bach, and Nicolas Usunier. A simple convergence proof of adam and adagrad. arXiv preprint arXiv:2003.02395, 2020.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024.

Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Shi Jie, Xiang Wang, Xiangnan He, and TatSeng Chua. Alphaedit: Null-space constrained knowledge editing for language models. arXiv preprint arXiv:2410.02355, 2024.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. arXiv preprint arXiv:2012.14913, 2020.

John J Hopfield. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the national academy of sciences, 79(8):2554–2558, 1982.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.

Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cecista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan. github. io/posts/muon, 6, 2024.

Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Rame´, Morgane Rivi`ere, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

Diederik P Kingma and Jimmy Lei Ba. Adam: A method for stochastic gradient descent. In ICLR:

international conference on learning representations, pp. 1–15, 2015. Teuvo Kohonen. Correlation matrix memories. IEEE transactions on computers, 100(4):353–359, 2009. Dmitry Kovalev. Understanding gradient orthogonalization for deep learning via non-euclidean trust-region

optimization. arXiv preprint arXiv:2503.12645, 2025.

Frederik Kunstner, Alan Milligan, Robin Yadav, Mark Schmidt, and Alberto Bietti. Heavy-tailed class imbalance and why adam outperforms gradient descent on language models. Advances in Neural Information Processing Systems, 37:30106–30148, 2024.

Tim Tsz-Kit Lau, Qi Long, and Weijie Su. Polargrad: A class of matrix-gradient optimizers from a unifying preconditioning perspective. arXiv preprint arXiv:2505.21799, 2025.

Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. Zero-shot relation extraction via reading comprehension. arXiv preprint arXiv:1706.04115, 2017.

Haochuan Li, Alexander Rakhlin, and Ali Jadbabaie. Convergence of adam under relaxed assumptions. Advances in Neural Information Processing Systems, 36:52166–52196, 2023.

Jiaxiang Li and Mingyi Hong. A note on the convergence of muon and further. arXiv e-prints, pp. arXiv– 2502, 2025.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982, 2025.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372, 2022a.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. arXiv preprint arXiv:2210.07229, 2022b.

Eshaan Nichani, Jason D Lee, and Alberto Bietti. Understanding factual recall in transformers via associative memories. arXiv preprint arXiv:2412.06538, 2024.

Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. Resurrecting recurrent neural networks for long sequences. In International Conference on Machine Learning, pp. 26670–26698. PMLR, 2023.

Yan Pan and Yuanzhi Li. Toward understanding why adam converges faster than sgd for transformers. arXiv preprint arXiv:2306.00204, 2023.

Guilherme Penedo, Hynek Kydlı´ˇcek, Anton Lozhkov, Margaret Mitchell, Colin A Raffel, Leandro Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 37:30811–30849, 2024.

Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, and Volkan Cevher. Training deep learning models with norm-constrained lmos. arXiv preprint arXiv:2502.07529, 2025.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Olivier Roy and Martin Vetterli. The effective rank: A measure of effective dimensionality. In 2007 15th European signal processing conference, pp. 606–610. IEEE, 2007.

Naoki Sato, Hiroki Naganuma, and Hideaki Iiduka. Analysis of muon’s convergence and critical batch size. arXiv preprint arXiv:2507.01598, 2025.

Ishaan Shah, Anthony M Polloreno, Karl Stratos, Philip Monk, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, Anil Thomas, Ashish Tanwer, Darsh J Shah, et al. Practical efficiency of muon for pretraining. arXiv preprint arXiv:2505.02222, 2025.

Wei Shen, Ruichuan Huang, Minhui Huang, Cong Shen, and Jiawei Zhang. On the convergence analysis of muon. arXiv preprint arXiv:2505.23737, 2025.

Chongjie Si, Debing Zhang, and Wei Shen. Adamuon: Adaptive muon optimizer. arXiv preprint arXiv:2507.11005, 2025.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Bhavya Vasudeva, Puneesh Deora, and Christos Thrampoulidis. On generalization of spectral gradient descent: A case study on imbalanced data. In High-dimensional Learning Dynamics 2025, 2025.

David J Willshaw, O Peter Buneman, and Hugh Christopher Longuet-Higgins. Non-holographic associative memory. Nature, 222(5197):960–962, 1969.

Greg Yang, James B Simon, and Jeremy Bernstein. A spectral condition for feature learning. arXiv preprint arXiv:2310.17813, 2023.

Yushun Zhang, Congliang Chen, Naichen Shi, Ruoyu Sun, and Zhi-Quan Luo. Adam can converge without any modification on update rules. Advances in neural information processing systems, 35:28386–28399, 2022.

Yushun Zhang, Congliang Chen, Tian Ding, Ziniu Li, Ruoyu Sun, and Zhiquan Luo. Why transformers need adam: A hessian perspective. Advances in neural information processing systems, 37:131786–131823,

- 2024a.

Zeyu Zhang, Akide Liu, Ian Reid, Richard Hartley, Bohan Zhuang, and Hao Tang. Motion mamba: Efficient and long sequence motion generation. In European Conference on Computer Vision, pp. 265–282. Springer,

- 2024b.

Dongruo Zhou, Jinghui Chen, Yuan Cao, Ziyan Yang, and Quanquan Gu. On the convergence of adaptive gradient methods for nonconvex optimization. arXiv preprint arXiv:1808.05671, 2018.

Fangyu Zou, Li Shen, Zequn Jie, Weizhong Zhang, and Wei Liu. A sufficient condition for convergences of adam and rmsprop. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pp. 11127–11135, 2019.

### A Steepest Descent View Understanding Muon and Adam

Bernstein & Newhouse (2024) showed that many popular deep learning optimizers can be understood through the unifying framework of steepest descent, once their exponential moving averages (EMAs) are disabled. This perspective shifts the focus from heuristic or second-order motivations to a more fundamental, geometric view: the choice of an optimizer is equivalent to choosing a specific norm to measure the “size” of the weight update.

The Steepest Descent Framework. The core idea is to find a weight update, ∆w, that minimizes a local quadratic approximation of the loss function. This is formulated as the following optimization problem:

∆w∗ = argmin

∆w

λ 2∥∆w∥2 ,

g⊤∆w +

where g is the gradient of the loss, λ > 0 is a “sharpness” parameter that controls the step size, and ∥ · ∥ is a chosen norm.

The solution to this problem can be expressed as:

∆w∗ = −η · d, where the step size η = ∥g∥∗

λ and the update direction d = arg max∥t∥=1 g⊤t. Here, ∥ · ∥∗ denotes the dual norm of ∥ · ∥ (defined as ∥y∥∗ = sup∥x∥≤1 y⊤x). The key insight is that different choices of the norm ∥ · ∥ lead to different update directions d, recovering the update rules of well-known optimizers.

Muon as Steepest Descent under Spectral Norm. The update rule of the Muon optimizer is derived by applying the steepest descent framework to weight matrices equipped with the spectral norm, denoted in the paper as the ∥ · ∥ℓ2→ℓ2 operator norm (defined as its largest singular value, ∥A∥ℓ2→ℓ2 = σmax(A) = sup∥x∥

2=1 ∥Ax∥2). For a gradient matrix G, the problem is to find the update ∆W that solves: ∆W∗ = argmin

λ 2∥∆W∥2ℓ2→ℓ2 .

⟨G,∆W⟩F +

∆W

The solution to this problem is directly determined by the Singular Value Decomposition (SVD) of the gradient, G = UΣV ⊤. The resulting update direction, which maximizes alignment with the gradient under the spectral norm constraint, is shown to be UV ⊤. The corresponding dual norm of the gradient, ∥G∥∗ℓ2→ℓ2, which scales the step size, is found to be tr(Σ), the sum of the singular values. Combining these components yields the final steepest descent update rule:

tr(Σ) λ · UV ⊤.

∆W∗ = −

This demonstrates that Muon’s core operation is a principled descent step where the singular vectors of the gradient determine the direction, and the sum of its singular values scales the step size.

Adam as Steepest Descent under ℓ∞ Norm. Adam can be understood as steepest descent on the flattened parameter vector w when the space is equipped with the vector infinity norm (ℓ∞) (defined as the maximum absolute value of its elements, ∥x∥∞ = maxi |xi|). For a gradient vector g, the optimization problem is to find the update ∆w that solves:

∆w∗ = argmin

∆w

λ 2∥∆w∥2∞ .

g⊤∆w +

The update direction that maximizes alignment with the gradient g under the infinity norm constraint is the sign of the gradient, sign(g). The corresponding dual norm of the gradient, ∥g∥∗∞, which scales the step size, is the ℓ1 norm, ∥g∥1 (the sum of the absolute values of its elements, ∥x∥1 = i |xi|). Combining these components yields the final steepest descent update rule:

∥g∥1 λ · sign(g).

∆w∗ = −

This reveals that Adam’s fundamental operation corresponds to a descent step where each parameter moves with the same magnitude, determined only by its gradient’s sign.

### B Experimental Details

#### B.1 Experimental Details of Training on FineWeb

When training 160M models on FineWeb, we disable weight decaying and Nesterov acceleration for both Adam and Muon. Thus, we only compare their performance along. To set the learning rate, we conduct a grid search on 1 × 10−1,5 × 10−2,2 × 10−2,1 × 10−2,5 × 10−3,2 × 10−3,1 × 10−3,5 × 10−4,2 × 10−4. When conducting the “Independent Blocks ” and “Combined Configuration ” experiments in Section 4.1, we just fix the learning rate of Muon. We set β1 = 0.8, β2 = 0.95 for Adam and set β = 0.95 for Muon. When training 0.7B models on FineWeb, we conduct a grid search of learning rate on 2×10−3,1×10−3,5×10−4,2×10−4. We set β1 = 0.9, β2 = 0.95 for Adam and set β = 0.95 for Muon. We do not adopt group query attention in the structure; thus, the parameter sizes of WQ, WK, WV , and WO are the same. We conduct experiments on 8 A100 with 80 GB memory.

#### B.2 Dataset Details for the Heavy-Tail Knowledge Task

Following Allen-Zhu & Li (2024), the foundation of our knowledge-intensive task is a set of question-answering (QA) pairs derived from synthetically generated biographies. Each biography is constructed from a combination of seven key attributes: name, birthdate, birthplace, educational institution, major, employer, and workplace. The attribute values are sampled from predefined lists, creating a diverse set of entities. Specifically, we use approximately 400 first names, 1000 surnames, 300 educational institutions, 100 majors, and 300 employers. Each synthetic individual is assigned a unique combination of these attributes, forming a distinct biographical profile. For example, a generated biography might look like this:

Ashton Hilda Older has a birthday that falls on February 01, 2063. Miami, FL is the birthplace of he. He is an alumnus of Saddleback College. He has a General Literature education. He works closely with BlockFi. For professional growth, he chose to relocate to Jersey City.

This text is generated by combining the structured attributes (name, date, location, etc.) with a set of sentence templates.

A predefined set of QA templates is then used to generate the final training data. These templates contain placeholders corresponding to the biographical attributes. By formatting these templates with the information from each synthetic biography, we generate a collection of concrete QA pairs for each entity. For example, for the entity “Ashton Hilda Older”, we can generate the following six QA pairs:

- 1. What is the birth date of Ashton Hilda Older? Answer: February 01, 2063.
- 2. What is the birth city of Ashton Hilda Older? Answer: Miami, FL.
- 3. Which university did Ashton Hilda Older study? Answer: Saddleback College.

- 4. What major did Ashton Hilda Older study? Answer: General Literature.
- 5. Which company did Ashton Hilda Older work for? Answer: BlockFi.
- 6. Where did Ashton Hilda Older work? Answer: Jersey City.

To evaluate the optimizers on a knowledge-intensive task with data imbalance, we constructed a synthetic dataset where the number of question-answering (QA) samples per class follows a power-law distribution. This is designed to simulate real-world scenarios where a few entities (the “head”) are highly represented, while most entities (the “tail”) are rare.

The generation process is controlled by an integer parameter, m. The classes are organized into m + 1 groups, indexed from g = 0 to m.

- • Group g contains Ng classes, where N0 = 1 and Ng = 2g−1 for g > 0.
- • Each class within group g is allocated a specific number of “selections,” Sg = 2m−g.

- • For each selection, we generate nqa unique QA pairs by formatting templates with biographical information corresponding to that class.

Thus, the total number of QA samples for any given class in group g is Sg ×nqa. This structure ensures that the single class in group 0 has the most samples, while the numerous classes in group m have the fewest.

In our experiment, we set the parameters to m = 15 and nqa = 6. This results in a dataset with a total of 215 = 32,768 classes. The number of samples per class ranges from 196,608 for the head class (group 0) down to just 6 for each of the 16,384 tail classes (group 15). The final distribution is visualized in Figure 3(a) in the main text.

To evaluate the model’s performance on this pure memory task, we measure the First Token Accuracy (FTA) on the answers. This metric assesses the model’s ability to correctly recall information by checking if the first generated token of the answer matches the ground truth. Furthermore, to understand how optimizers handle data imbalance, we analyze the FTA across different data frequency groups, from highfrequency (head) to low-frequency (tail) data.

#### B.3 Experimental Details About Angles Between Associative Memories Embeddings

Following Fang et al. (2024), we analyze the associative memories in the FFN modules. To obtain Ei, we use the activations within the feed-forward modules, and for Ei, we take the corresponding module outputs. We evaluate knowledge items from two widely used datasets: Counterfact (Meng et al., 2022a) and ZsRE (Levy et al., 2017). Results on Counterfact are shown in Figure 4(a), while results on ZsRE are provided in Figure 10 in Appendix C.6.

#### B.4 Experimental Details of One-layer Models

We set the hyperparameters as K = d = 999, α = 0.8, β = 0.2. For the support-decoupled setting, we set E and E as identity matrices. For the support-coupled setting, we set E and E according to the construction presented in the proof of Theorem 5.3 in Appendix D.

### C Additional Experimental Results

#### C.1 MaxLogit per Layer on the 160M NanoGPT model via Muon Optimizer

In this subsection, we present the MaxLogit values for each layer of the 160M NanoGPT model trained using the Muon Optimizer. Following Gemma 3 (Kamath et al., 2025), we introduce RMSNorm to the attention mechanism. The attention mechanism in our model is defined as follows:

O = softmax( Q KT)V, Q = RMSNorm(Q), K = RMSNorm(K) where RMSNorm is defined as RMSNorm(x) = √ x

, with d being the dimension of x. MaxLogit is defined as:

d i=1 x2i

1 d

qi · kj representing the maximum value in the attention scores before softmax normalization. The MaxLogit values for each layer are summarized in Table 2. Table 2: MaxLogit values per layer on the 160M NanoGPT model via Muon Optimizer.

Smax = max

i,j

|Layer<br><br>|1 2 3 4 5 6 7 8 9 10 11 12|
|---|---|
|MaxLogit|8.396 6.880 6.009 7.676 6.349 5.890 7.688 6.314 6.205 5.613 6.033 6.371|

Recent reports Team et al. (2025) have shown a potential “MaxLogit explosion” phenomenon, where Smax grows steadily (often near-linearly) during training, leading to overly peaked attention, gradient spikes,

and degraded optimizer comparisons. We included this measurement to rule out the possibility that Muon’s comparatively smaller impact on the QK blocks (relative to VO/FFN) is simply due to suppressing such an instability. In our 160M setting, with RMSNorm applied to both Q and K (following Gemma 3), the per-layer MaxLogit values remain moderate and show no runaway growth. Thus, for this model size and normalization scheme, differences in Muon’s effectiveness across components cannot be attributed to avoiding a MaxLogit explosion in attention.

#### C.2 Scaling to the 0.7B NanoGPT Model

###### To evaluate the scalability of our findings, we extend our experiments from the 160M model to a larger 0.7B parameter model. This section presents the results of this scaled-up analysis, examining whether the advantages of Muon observed in the smaller model persist at a larger scale.

Muon(All Attn, FFN)

6.5

All Adam

Muon(QK Attn)&Adam(VO Attn, FFN) Muon(VO Attn, FFN)&Adam(QK Attn) Muon(O Attn, FFN)&Adam(QKV Attn) Muon(V Attn, FFN)&Adam(QKO Attn)

6.0

5.5

ValidationLoss

5.0

3.07

4.5

4.0

2.91

3.5

8000 10000

3.0

0 2000 4000 6000 8000 10000

Training Steps

(a) Non-gated FFN

Muon(All Attn, FFN)

5.5

All Adam

Muon(QK Attn)&Adam(VO Attn, FFN) Muon(VO Attn, FFN)&Adam(QK Attn) Muon(O Attn, FFN)&Adam(QKV Attn) Muon(V Attn, FFN)&Adam(QKO Attn)

5.0

ValidationLoss

4.5

3.15

4.0

3.5

2.96

8000 10000

3.0

0 2000 4000 6000 8000 10000

Training Steps

(b) Gated FFN

Figure 5: Validation loss comparison on the 0.7B NanoGPT model. (a) Combined configuration with nongated feed-forward networks.(b) Combined configuration with gated feed-forward networks.

Win - SVD Entropy ( )

###### Wout - SVD Entropy ( )

VO - SVD Entropy ( )

1.0

0.9

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.98

0.8

0.8

0.96

SVDEntropy()

SVDEntropy()

SVDEntropy()

0.6

0.94

0.7

0.4

0.92

0.6

0.2

0.90

Muon(All Attn, FFN)

0.5

All Adam

0.0

0.88

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

Training Steps

Win - eRank ( )

Wout - eRank ( )

###### VO - eRank ( )

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

600

1000

1100

1000

800

500

eRank()

eRank()

eRank()

900

600

800

400

400

700

200

300

600

0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

Training Steps

###### Figure 6: Spectral Dynamics of Weight Matrices During Training on the 0.7B NanoGPT model.

Wgate - SVD Entropy ( )

Win - SVD Entropy ( )

###### Wout - SVD Entropy ( )

VO - SVD Entropy ( )

1.0

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

- 0.5

- 0.6

- 0.7

- 0.8

- 0.9

0.982

0.980

0.8

SVDEntropy()

SVDEntropy()

SVDEntropy()

SVDEntropy()

0.980

0.975

0.6

0.978

0.970

0.4

0.976

0.965

0.2

0.974

Muon(All Attn, FFN)

0.960

All Adam

0.972

0.0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

Training Steps

Training Steps

###### Wgate - eRank ( )

###### Win - eRank ( )

Wout - eRank ( )

###### VO - eRank ( )

650

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1000

600

1120

1100

800

550

eRank()

eRank()

eRank()

eRank()

1100

600

1050

500

400

1080

450

1000

200

1060

400

0

950

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

Training Steps

Training Steps

- Figure 7: Spectral Dynamics of Weight Matrices During Training on the 0.7B NanoGPT model with the Gated FFN.

Figure 5 shows the validation loss curves for various optimizer configurations. Consistent with our findings on the 160M model, applying Muon to all components achieves the lowest validation loss, outperforming Adam baseline. The hybrid experiments further reinforce our earlier conclusions: applying Muon to only the VO and FFN components yields performance nearly identical to that of the full Muon optimizer, whereas applying it only to the QK components offers little advantage over Adam.

The spectral dynamics, shown in Figures 6 and 7, also align with Observation 2. For the VO, Win, Wgate (in model with Gated FFN) and Wout matrices, Muon leads to higher SVD entropy and eRank compared to Adam, indicating that it encourages the learning of more distributed, higher-dimensional representations. Overall, these results demonstrate that the benefits of Muon and the underlying mechanisms scale to larger models.

#### C.3 Additional Results about Spectral Dynamics of Transformer Weight Matrices During Training

To complement the main-text analysis (Fig. 2), we also evaluate spectral dynamics during training for the 160M NanoGPT model with both non-gated and gated feed-forward networks (Fig. 8). The analysis includes Win for both configurations, as well as the gate matrix Wgate for the gated version. The conclusions are consistent across all three matrices and mirror the non-gated setting: with Muon, SVD entropy and eRank increase, while Top-k energy and the Q75/25 ratio decrease, consistent with Observation 2 in the main text.

SVD entropy ( )

Top10E ( )

1.0

0.8

SVDentropy()

0.6

0.8

Top10E()

0.4

0.6

0.2

Muon(All Attn, FFN)

All Adam

0.4

0.0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

eRank ( )

Q75/Q25 Ratio ( )

2 × 101

600

Q75/Q25Ratio()

101

eRank()

400

6 × 100

200

- 3 × 100

- 4 × 100

0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

(a) Win (Non-Gated FFN)

SVD entropy ( )

Top10E ( )

1.0

0.8

SVDentropy()

0.8

0.6

Top10E()

0.6

0.4

0.2

Muon(All Attn, FFN)

0.4

All Adam

0.0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

eRank ( )

Q75/Q25 Ratio ( )

600

Q75/Q25Ratio()

101

eRank()

400

6 × 100

200

- 3 × 100

- 4 × 100

0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

(b) Win (Gated FFN)

SVD entropy ( )

Top10E ( )

1.0

0.8

SVDentropy()

0.8

Top10E()

0.6

0.6

0.4

0.2

Muon(All Attn, FFN)

0.4

All Adam

0.0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

eRank ( )

Q75/Q25 Ratio ( )

600

Q75/Q25Ratio()

101

eRank()

400

6 × 100

200

- 3 × 100

- 4 × 100

0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Training Steps

Training Steps

(c) Wgate

- Figure 8: Spectral Dynamics of FFN Weight Matrices During Training on the 160M NanoGPT model. Each panel reports four metrics characterizing singular value distributions: SVD entropy, Top10E, eRank, and Q75/Q25 ratio. The subplots correspond to different weight matrices: (a) Win (non-gated), (b) Win (gated), and (c) Wgate (gated).

#### C.4 Detailed Experiment Results about Heavy-Tail Imbalance Knowledge Task

To complement the qualitative trends shown in Section 4.3 (Fig. 3), we report the exact First Token Accuracy (FTA) for selected tail groups at three training checkpoints (2k, 5k, 10k steps). We focus on groups g = 11,13,15, which represent increasingly rare (mid–tail, tail, extreme tail) frequency bands in the power-law distribution (recall that larger g implies fewer samples per class). The tables contrast full Muon, Adam, SGD+Momentum, and two hybrid configurations (Muon applied only to VO&FFN or only to QK). The numbers highlight: (i) Muon’s rapid convergence on rare groups (already strong by 2k, near-saturated by 5k), (ii) Adam’s persistent head–tail gap, and (iii) the dominant contribution of applying Muon to VO&FFN for tail generalization (the VO&FFN hybrid closely tracks full Muon, whereas the QK-only hybrid lags). These quantitative results substantiate Observation 3 that Muon delivers more balanced learning.

Table 3: Heavy-tail knowledge task: Group performance by optimizer (2,000 steps)

Optimizer Muon Adam SGD+Mom. Muon(VO, FFN) Muon(QK)

Group

11 0.854 ± 0.029 0.312 ± 0.043 0.156 ± 0.037 0.814 ± 0.022 0.472 ± 0.041 13 0.386 ± 0.029 0.146 ± 0.015 0.120 ± 0.012 0.256 ± 0.030 0.154 ± 0.032 15 0.140 ± 0.027 0.090 ± 0.031 0.082 ± 0.013 0.114 ± 0.023 0.086 ± 0.037

Table 4: Heavy-tail knowledge task: Group performance by optimizer (5,000 steps)

Optimizer Muon Adam SGD+Mom. Muon(VO, FFN) Muon(QK)

Group

11 0.996 ± 0.006 0.936 ± 0.039 0.314 ± 0.021 0.992 ± 0.005 0.970 ± 0.007 13 0.964 ± 0.023 0.298 ± 0.074 0.148 ± 0.013 0.934 ± 0.015 0.354 ± 0.032 15 0.320 ± 0.028 0.110 ± 0.027 0.084 ± 0.011 0.254 ± 0.026 0.118 ± 0.019

Table 5: Heavy-tail knowledge task: Group performance by optimizer (10,000 steps)

Optimizer Muon Adam SGD+Mom. Muon(VO, FFN) Muon(QK)

Group

11 1.000 ± 0.000 1.000 ± 0.000 0.422 ± 0.023 1.000 ± 0.000 1.000 ± 0.000 13 1.000 ± 0.000 0.890 ± 0.042 0.294 ± 0.013 0.998 ± 0.002 0.940 ± 0.034 15 0.976 ± 0.006 0.264 ± 0.048 0.126 ± 0.021 0.954 ± 0.021 0.286 ± 0.039

#### C.5 Additional Experiment Results about Heavy-Tail Imbalance Knowledge Task with Gated Feed-Forward Networks

This subsection complements the main heavy-tail results in Section 4.3 by studying the gated feed-forward networks (Gated FFN) variant. We follow the same presentation order as in the main text: first an overview figure (sample distribution and learning curves under different optimizers), then tables reporting the exact First-Token Accuracy (FTA) for tail groups g ∈ {11,13,15} at three training checkpoints (2k, 5k,

- 10k steps). The findings mirror the non-gated setting: (i) full Muon consistently outperforms Adam and SGD+Momentum on rare classes and reaches high accuracy earlier; (ii) the VO&FFN-hybrid (Muon applied to VO and FFN while Adam is used for QK) closely tracks full Muon, indicating that VO&FFN are the primary levers for tail generalization; (iii) the QK-only hybrid offers limited gains. Overall, the gated FFN does not change the qualitative conclusions about where Muon helps most. See Fig. 9 and Tables 6–8 for details.

Class Groups

- 101

- 102

- 103

- 104

- 105

- Group 0

- Group 1

- Group 2

- Group 3

- Group 4

- Group 5

- Group 6

- Group 7

- Group 8

- Group 9

- Group 10

- Group 11

- Group 12

- Group 13

- Group 14

- Group 15

#QA Samples

100 101 102 103 104

Class index (sorted)

(a) Sample/class

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1.0

0.8

Accuracy(FTA)

0.6

0.4

0.2

0.0

0 2000 4000 6000 8000 10000

Training Steps

(b) Muon

1.0

0.8

Accuracy(FTA)

0.6

0.4

0.2

0.0

0 2000 4000 6000 8000 10000

Training Steps

###### (c) Adam

1.0

0.8

Accuracy(FTA)

0.6

0.4

0.2

0.0

0 2000 4000 6000 8000 10000

Training Steps

(d) SGD+Momentum

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1.0

0.8

Accuracy(FTA)

0.6

0.4

0.2

0.0

0 2000 4000 6000 8000 10000

Training Steps

(e) Muon(VO, FFN)&Adam(QK)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1.0

0.8

Accuracy(FTA)

0.6

0.4

0.2

0.0

0 2000 4000 6000 8000 10000

Training Steps

(f) Muon(QK)&Adam(VO,FFN)

- Figure 9: Performance comparison of different optimizers on a heavy-tailed knowledge task with gated feedforward networks. (a) The distribution of samples per class follows a power law. (b-d) Performance of Muon, Adam, and SGD+Momentum optimizers. (e) Muon (VO, FFN)/Adam (QK). (f) Muon (QK)/Adam (VO, FFN).

- Table 6: Heavy-tail knowledge task with the Gated FFN: Group performance by optimizer (2,000 steps)

Group

Optimizer Muon Adam SGD+Mom. Muon(VO, FFN) Muon(QK)

11 0.896 ± 0.009 0.214 ± 0.063 0.146 ± 0.018 0.892 ± 0.021 0.330 ± 0.042 13 0.478 ± 0.034 0.116 ± 0.030 0.110 ± 0.007 0.458 ± 0.037 0.140 ± 0.019 15 0.178 ± 0.018 0.086 ± 0.013 0.074 ± 0.009 0.166 ± 0.017 0.090 ± 0.020

- Table 7: Heavy-tail knowledge task with the Gated FFN: Group performance by optimizer (5,000 steps)

Group

Optimizer Muon Adam SGD+Mom. Muon(VO, FFN) Muon(QK)

11 0.998 ± 0.002 0.928 ± 0.024 0.252 ± 0.016 0.990 ± 0.010 0.960 ± 0.032 13 0.990 ± 0.010 0.216 ± 0.052 0.156 ± 0.024 0.968 ± 0.028 0.290 ± 0.046 15 0.510 ± 0.039 0.092 ± 0.015 0.080 ± 0.016 0.468 ± 0.016 0.098 ± 0.013

- Table 8: Heavy-tail knowledge task with the Gated FFN: Group performance by optimizer (10,000 steps)

Optimizer Muon Adam SGD+Mom. Muon(VO, FFN) Muon(QK)

Group

11 1.000 ± 0.000 0.998 ± 0.002 0.322 ± 0.011 1.000 ± 0.000 1.000 ± 0.000 13 1.000 ± 0.000 0.948 ± 0.027 0.304 ± 0.017 1.000 ± 0.000 0.946 ± 0.026 15 0.994 ± 0.006 0.244 ± 0.085 0.148 ± 0.015 0.990 ± 0.010 0.274 ± 0.042

#### C.6 Additional Results about Angles Between Associative Memories Embeddings

Avg.AnglesBetweenEmbeddings(Degrees)

90

75

60

45

30

Ei Ei

15

0

5 10 15 20 25 Layer Index

- Figure 10: Average angles between es or eo for items in ZsRE at layers 5, 10, 15, 20, 25 of Llama3-8b-instruct.

### D Proof of Theorem 5.3

We separately derive the results for GD, Muon, and Adam in the following proof. For all of them, we define

ηoptϵ = inf η ≥ 0 1 − max k∈[K]

fW(Ek) k ≤ ϵ, where W = W0 − η · Gopt(W0) . (D.1)

The quantity ηoptϵ represents the minimal step size for at least one triplet to be learned with error probability less than ϵ. From the definition, we have that

ϱϵopt ≤ min

k∈[K]

optGopt(Ek) k.

f−ηϵ

Step 1: Calculations of GD. We define the score of k′-th object for the k-th subject-relation pair with the parameter W as

At W0 = 0d

o,ds, we have that

exp( Ek⊤′WEk)

s(k′,k,W) =

.

K k′′=1 exp( E⊤

k′′WEk)

1 K

s(k′,k,W0) =

for all k,k′ ∈ [K]. Proposition F.1 shows that the gradient is

1 − α K − L

α LK

α L

E1:LE1:⊤L +

EL+1:KEL⊤+1:K −

EJK,LE1:⊤L −

−∇WL(W0) =

1 − α (K − L)K

EJK,K−LEL⊤+1:K. (D.2)

From the gradient, it is easy to see that the first L triplets (s,r,o) share the same learning behavior, and the last K −L triplets also share the same behavior. Thus, we calculate the results for k = 1 and k = L+1. The calculation for k = 1 depends on evaluating its score function, which takes the form η · Ek⊤′′[−∇WL(W0)]E1, for k′′ ∈ {1,...,K}. Based on the gradient in (D.2) and the orthonormality of the embeddings, it evaluates to α/L for the case k′′ = 1, and to 0 for all k′′ ̸= 1.

This leads to a numerator in the softmax score of exp(η · α/L), while the denominator sum consists of one term exp(η · α/L) and K − 1 terms of exp(0) = 1. A similar calculation for k = L + 1 shows that the argument of the exponent for the correct object, η· EL⊤+1[−∇WL(W0)]EL+1, evaluates to η·(1−α)/(K −L). By defining γ1 = α/(βK) and γ2 = (1 − α)/((1 − β)K) based on the problem setup (L = βK), we have that

exp(ηγ1) exp(ηγ1) + K − 1

exp(ηγ2) exp(ηγ2) + K − 1

, f−η∇

WL(EL+1) L+1 =

,

f−η∇

WL(E1) 1 =

where γ1 and γ2 are defined as

Then we derive that

1 − α (1 − β)K

α βK

γ1 =

, γ2 =

.

1 max{γ1,γ2}

log (ϵ−1 − 1)(K − 1) . (D.3)

ηGDϵ =

To calculate the desired quantity, we define the quantity r(α,β) to evaluate the balance of data as

α(1 − β) β(1 − α)

β(1 − α) α(1 − β)

r(α,β) = min{γ1/γ2,γ2/γ1} = min

,

.

Some basic calculations show that 1 − min

ϵ

ϵ + (1 − ϵ)r(α,β)ϵ1−r(α,β)(K − 1)r(α,β)−1 . (D.4) When r < 1, with the fact that x+11 = 1 − x + O(x2) as x → 0, we have that

f−ηϵ

GDGGD(Ek) k =

k∈[K]

GDGGD(Ek) k = O(ϵ−r(α,β)Kr(α,β)−1).

min

f−ηϵ

Thus, the proof for the convergence of GD has been established. Step 2: Calculations of Muon. For Muon, we first calculate the SVD of the gradient. In fact, we can write the gradient in Eqn. (D.2) as

−∇WL(W0) = E diag

= EXE⊤.

1 − α K − L

α L

IL,

IK−L −

1 K

IK ·

1 − α K − L

α L

I⊤L,

I⊤K−L

⊤

E⊤

The SVD calculation of X = UΣV ⊤ can be directly derived from Proposition F.3. Thus, the SVD of the gradient is −∇WL(W0) = ( E ·U)Σ(E ·V )⊤. The update quantity GMuon(W0) = U0norm(Σ0)V0⊤ of Muon is

− GMuon(W0)

= E1:LRL,L−1RL,L⊤ −1E1:⊤L + EL+1:KRK−L,K−L−1RK⊤−L,K−L−1EL⊤+1:K

1 K α2(K − L)3 + (1 − α)2L3

(K − L) E1:LIL − L EL+1:KIK−L

+

L(1 − α) K − L

(K − L)α L

I⊤LE1:⊤L −

I⊤K−LEL⊤+1:K

·

= E1:LE1:⊤L + EL+1:KEL⊤+1:K

(1 − β)2α λ − 1 E1:LJL,LE1:⊤L

1 K

1 β

+

β2(1 − α) λ − 1 EL+1:KJK−L,K−LEL⊤+1:K

1 1 − β

+

− β(1 − α) E1:LJL,K−LEL⊤+1:K − α(1 − β) EL+1:KJK−L,LE1:⊤L , (D.5)

where λ = α2(1 − β)3 + (1 − α)2β3, the matrices RL,L−1 and RK−L,K−L−1 are defined in Proposition F.3, and the second equality results from the following facts

1 L

ILI⊤L, RK−L,K−L−1RK⊤−L,K−L−1 = IK−L,K−L −

RL,L−1RL,L⊤ −1 = IL,L −

1 K − L

IK−LI⊤K−L.

Although the gradient is composed of heterogeneous components from E1:L,E1:L and EL+1:K,EL+1:K, we can bound the convergence rate of [f−ηG

(Ek)]k for any k: an upper (resp. lower) bound is obtained by

Muon

increasing (resp. decreasing) the coefficient of EkEk⊤ while decreasing (resp. increasing) that of Ek′Ek⊤ for k′ ̸= k. In fact, Eqn. (D.5) implies that there exists a constant C > 0 such that the dynamics of the fastest-

and slowest-learning triplets are bounded by those along the following two update directions.

2C K

−G+Muon(W0) = 1 +

2C K

−G−Muon(W0) = 1 −

C K · EJK,KE⊤

( E1:LE1:⊤L + EL+1:KEL⊤+1:K) −

C K · EJK,KE⊤.

( E1:LE1:⊤L + EL+1:KEL⊤+1:K) +

Concretely, the rate of score increase for the correct object of the k-th triplet, which is given by the term Ek⊤[−GMuon(W0)]Ek in the exponent of the softmax score, is bounded. The rate for the fastest-learning triplet is lower-bounded by the corresponding rate derived from −G+Muon(W0), while the rate for the slowestlearning triplet is upper-bounded by that from −G−Muon(W0). Thus, we only need to focus on G+Muon(W0) and G−Muon(W0) to calculate the desired quantity. Following the similar procedures for GD to derive Eqn. (D.4), we have that for any η such that maxk∈[K] fW

(Ek) k ≥ 1 − ϵ (where Wη = W0 − η · GMuon(W0)), the following holds

η

ϵ ϵ + (1 − ϵ)r(K)ϵ1−r(K)(K − 1)r(K)−1, (D.6)

1 − min

(Ek) k ≤

fW

η

where r(K) = (K − 2C)/(K + 2C). We further have that (1 − ϵ)r(K)ϵ1−r(K)(K − 1)r(K)−1

4C K + 2C

ϵ 1 − ϵ − log(K − 1)

= (1 − ϵ)exp

log

(log K)2 K2

4C K + 2C

ϵ 1 − ϵ − log(K − 1) + O

= (1 − ϵ) 1 +

log

log K K

= (1 − ϵ) + O

, (D.7)

where the first equality results from the basic calculations, the second equality results from that exp(x) = 1 + x + O(x2) when x → 0. Combining Eqn. (D.6) and (D.7), we have that

log K K

ϱϵMuon ≥ 1 − ϵ 1 + O

.

Thus, we prove the desired results for Muon. Step 3: Calculations of Adam. The proof of the results for Adam is conducted under two cases. We will construct different embeddings

- E and E in these two cases. In the first case, we set E = E = IK,K. With such embedding and sufficiently large K, we have that

−GSignGD(W0) = −sign(∇WL(W0)) = 2IK,K − JK,K. Under such a setting, all triplets share the same dynamic. Thus, we have that

ϱϵSignGD = 1 − ϵ.

In the second case, we set E and E as block-wise diagonal matrices. Here we set the block size as 3, i.e., requiring that K mod3 = 0. Such a requirement can be satisfied infinitely often when K → ∞. Then the sufficient and necessary condition of Assumption 5.1 is that each 3 × 3 block contains an orthonormal basis. To achieve this, we define the following matrix.

 

 .

cosacosbcosc − sinasinc −cosacosbsinc − sinacosc cosasinb sinacosbcosc + cosasinc −sinacosbsinc + cosacosc sinasinb

R(a,b,c) =

−sinbcosc sinbsinc cosb

It is obvious that R(a,b,c)⊤R(a,b,c) = I3,3. Then we set E and E as E = IK/3,K/3 ⊗ R(3.638,2.949,5.218), E = IK/3,K/3 ⊗ R(1.715,0.876,3.098),

where ⊗ is the Kronecker product. With these specifications and sufficiently large K, the Adam update matrix is

−GSignGD(W0) = IK/3,K/3 ⊗ A + JK/3,K/3 ⊗ B, where A and B are specified as

 

 , B =

 

 .

−1 −1 −1 −1 −1 −1 1 1 1

2 0 0 2 0 2 −2 −2 −2

A =

These show that the diagonal block of −GSignGD(W0) is

 

 .

1 −1 −1 1 −1 1 −1 −1 −1

A + B =

Since the k-th and (k + 3)-th triplets share the same learning dynamics for all k ∈ [K − 3], we focus on the learning dynamics of k = 1,2,3. We have that

R(3.638,2.949,5.218)⊤ · (A + B) · R(1.715,0.876,3.098)

 

 ,

1.46552253 1.0132908 −0.11179563 −0.0732561 1.00709257 −1.26935805 0.0544114 0.89611102 1.54147329

=

R(3.638,2.949,5.218)⊤ · B · R(1.715,0.876,3.098)

 

 .

−0.19288146 −1.24460331 −1.4058011 −0.20112175 −1.2977753 −1.46585978 −0.12780259 −0.82466989 −0.93147899

=

From the last columns of these two matrices, following the similar procedures for GD to derive Eqn. (D.3), we have that

1 1.541 + 0.930

log (ϵ−1 − 1)(K − 1) =

ηSignGDϵ ≤

- 1

- 2.471

log (ϵ−1 − 1)(K − 1) .

Then, from the first columns of these matrices, we have that 1 − min

ϵ ϵ + (1 − ϵ)rϵ1−r(K − 1)r−1,

SignGDGSignGD(Ek) k ≥

f−ηϵ

k∈[K]

where r = 1.466+02.471.202 = 12..668471. Thus, we have that

ϱϵSignGD ≤ O(ϵ−rKr−1) ≤ O(ϵ−0.7K−0.3).

Then we calculate the singular values of −GSignGD(W0). We define the eigen vectors of IK,K as U, i.e., U⊤IK/3,K/3 U = diag(K/3,0··· ,0). Using the orthogonal invariance of singular values, −GSignGD(W0) shares the singular values with the following matrix

( U⊤ ⊗ I3,3) − GSignGD(W0) ( U ⊗ I3,3) = IK/3,K/3 ⊗ A + ( U⊤IK/3,K/3 U) ⊗ B

= diag(A − KB/3,A,··· ,A). Thus, the singular values of A are also the singular values of GSignGD(W0). We have that

σmin GSignGD(W0) σmax GSignGD(W0) ≤

σmin(A) σmax(A) ≤ 25%.

Thus, we conclude the proof of Theorem 5.3.

### E Proof of Theorem 5.4

The proof of Theorem 5.4 takes two steps. In the first step, we derive the share form of Wt along the whole optimization trajectory. In the second step, we build the desired results on the basis of step 1. Throughout

the proof, we will write WtMuon as Wt for the ease of presentation. Step 1: Derive the shared forms of Wt and GMuon. We will derive the forms of Wt along the optimization trajectory via the induction method. We first

state our hypothesis and then prove it. Hypothesis 1 . For any optimization step index t ∈ [T], the parameters Wt can be expressed as

Wt = EXtE, Xt = Λt + Ct,

where Λt and Ct are

c11t · JL,L c12t · JL,K−L c21t · JK−L,L c22t · JK−L,K−L

Λt = diag(at · IL,bt · IK−L), Ct =

,

where at,bt,c11t ,c12t ,c21t ,c22t ∈ R are real numbers such that (1) at = bt ≥ 0, and (2) cijt = O(at/K) for i,j ∈ [2].

o,ds satisfying this hypothesis with at = bt = c11t = c12t = c21t = c22t = 0. Then we assume that this hypothesis holds for {1,··· ,t}, and we will prove that it holds for t + 1. Since Wt+1 = Wt − ηt+1Utnorm(Σt)Vt⊤, we need to show that −ηt+1Utnorm(Σt)Vt⊤ satisfies the hypothesis. We define the score of k′-th object for the k-th subject-relation pair with the parameter W as

When t = 0, it is obvious to verify that W0 = 0d

exp( Ek⊤′WEk)

s(k′,k,W) =

.

K k′′=1 exp( E⊤

k′′WEk)

According to the symmetry of Wt, we have that

- • s(k,k,Wt) = s(1,1,Wt) for all k ≤ L.
- • s(k,k,Wt) = s(K,K,Wt) for all k > L.
- • s(k′,k,Wt) = s(2,1,Wt) for all k,k′ ≤ L,k′ ̸= k.
- • s(k′,k,Wt) = s(K,1,Wt) for all k ≤ L,k′ > L.
- • s(k′,k,Wt) = s(K − 1,K,Wt) for all k,k′ > L,k′ ̸= k.
- • s(k′,k,Wt) = s(1,K,Wt) for all k > L,k′ ≤ L.

- Thus, Proposition F.1 shows that the gradient of Wt is −∇WL(Wt) = E(Γt + Bt)E⊤,

where Γt and Bt are defined as

Γt = diag

α L

1 + s(2,1,Wt) − s(1,1,Wt) IL,

1 − α K − L

1 + s(K − 1,K,Wt) − s(K,K,Wt) IK−L ,

Bt = −Lαs(2,1,Wt) · JL,L −K1−−αLs(1,K,Wt) · JL,K−L −Lαs(K,1,Wt) · JK−L,L −K1−−αLs(K − 1,K,Wt) · JK−L,K−L

.

- Thus, Proposition F.2 shows that

where

−GMuon(Wt) = E diag(IK) +

C11 · JL,L C12 · JL,K−L C21 · JK−L,L C22 · JK−L,K−L

E⊤,

U1,1 V1,1 + U1,2 V1,2 − 1 βK

U1,1 V2,1 + U1,2 V2,2 β(1 − β)K

C11 =

, C12 =

,

U2,1 V2,1 + U2,2 V2,2 − 1 (1 − β)K

U2,1 V1,1 + U2,2 V1,2 β(1 − β)K

C21 =

, C22 =

.

where U, V ∈ R2×2 are the orthonormal matrices defined in Proposition F.2. Since Wt+1 = Wt−ηt+1GMuon(Wt), it is obvious that at+1 = bt+1. The orthonormality of U and V implies that | Ui,j|,| Vi,j| ≤ 1. Thus, we have

U1,1 V1,1 + U1,2 V1,2 − 1 βK

1 K

= O

.

This further implies that c1t+1,1 = O(at+1/K). The proofs for other cijt+1 are similar. This completes the proof. Step 2: Establish the convergence results. We note that this analysis is very similar to the proof of Muon in Theorem 5.3. Concretely, for Wt, the

coefficients at,bt,c11t ,c12t ,c21t ,c22t from multiple-step optimization share the same property with those of the one-step results. It means that there exists a constant C > 0 such that the dynamics of the fastest- and slowest-learning triplets are bounded by those along the following two update directions in only one step.

2C K

−G+Muon = 1 +

2C K

−G−Muon = 1 −

C K · EJK,KE⊤

( E1:LE1:⊤L + EL+1:KEL⊤+1:K) −

C K · EJK,KE⊤.

( E1:LE1:⊤L + EL+1:KEL⊤+1:K) +

The remaining analysis is then exactly the same as that of Theorem 5.3. Thus, we conclude the proof of Theorem 5.4.

### F Supporting Propositions

- Proposition F.1. We define the score of k′-th object for the k-th subject-relation pair with the parameter W as

s(k′,k,W) =

exp( Ek⊤′WEk)

K k′′=1 exp( E⊤

k′′WEk)

.

When the parameter W is trained with loss

L(W) = −

K

k=1

pk · log fW(Ek) k,

the gradient of W is

∇WL(W) = −

K

k=1

pk 1 − s(k,k,W) EkEk⊤ −

k′̸=k

s(k′,k,W) Ek′Ek⊤ .

- Proof of Proposition F.1. The proof just follows from the basic calculus. Thus, we omit them here.

| |
|---|

- Proposition F.2. Let X = Λ + C ∈ RK×K. The matrix Λ = diag(a · IL,b · IK−L) is a diagonal matrix whose first L diagonal elements are a and the last K − L elements are b with a,b > 0. The matrix C is a block-wise constant matrix defined as

c11 · JL,L c12 · JL,K−L c21 · JK−L,L c22 · JK−L,K−L

C =

.

Then X = UΣV ⊤. Here Σ,V,U are defined as follows. All of them can be decomposed into three blocks, each corresponding to a subspace. The first subspace is

S1 =

x 0K−L

x⊤IL = 0, and x ∈ RL .

The dimension of this space is L−1. The singular value of X corresponding to this subspace is a. The block of columns in both U and V that forms an orthonormal basis for this subspace is given by

RL,L−1 0K−L,L−1

,

where the columns of the matrix RL,L−1 ∈ RL×(L−1) form an orthonormal basis for the subspace {x ∈ RL|x⊤IL = 0}. The second subspace is

0L y

y⊤IK−L = 0, and y ∈ RK−L .

S2 =

The dimension of this space is K − L − 1. The singular value of X corresponding to this subspace is b. The block of columns in both U and V that forms an orthonormal basis for this subspace is given by

0L,K−L−1 RK−L,K−L−1

,

where the columns of the matrix RK−L,K−L−1 ∈ R(K−L)×(K−L−1) form an orthonormal basis for the subspace {y ∈ RK−L|y⊤IK−L = 0}. The remaining 2-dimensional subspace is induced by a 2 × 2 matrix M defined as

M =

α β γ δ

= Udiag(s1,s2) V ⊤,

where the elements of M are defined as

α = a + Lc11, β = L(K − L)c12, γ = L(K − L)c21, δ = b + (K − L)c22. The singular values s1,s2 are

s1,2 =

√T2 − 4∆ 2

T ±

, T = α2 + β2 + γ2 + δ2, ∆ = (αδ − βγ)2.

The singular values of X in this subspace are s1 and s2. The corresponding right singular vectors (vi) and left singular vectors (ui), which form columns of V and U respectively, are given by:

vi = V1,ie1 + V2,ie2,ui = U1,ie1 + U2,ie2 for i = 1,2, where the vectors e1 and e2 are defined as

In summary, the SVD of X is

e1 =

LIL 0K−L

√ 1

, e2 =

0L √K1−LIK−L

.

Σ = diag(a · IL−1,b · IK−L−1,s1,s2), V =

RL,L−1 0K−L,L−1

0L,K−L−1 RK−L,K−L−1

,

,v1,v2 ,

RL,L−1 0K−L,L−1

0L,K−L−1 RK−L,K−L−1

U =

,

,u1,u2 .

- Proof of Proposition F.2. We first prove the results for S1. For any vector v in S1, it is direct to verify that

v 0K−L

v 0K−L

X⊤X

= a2

.

Thus, the singular value of X corresponding to the subspace spanned by the vector [v⊤,0⊤K−L]⊤ is a, and the corresponding columns of V form an orthonormal basis for S1. For the U calculation, we have that

v 0K−L

v 0K−L

X

= a

.

Thus, the corresponding left singular vectors (columns of U) are identical to the right singular vectors for this subspace. A similar calculation can be done for S2. The remaining vectors are orthogonal to both S1 and S2 and thus take the form of

vi = p1e1 + p2e2, ui = p3e1 + p4e2 for i = 1,2 with p1,p2,p3,p4 ∈ R.

By solving the equation X⊤Xvi = λvi, we can show that the corresponding singular values and coefficients p1,p2,p3,p4 coincide with those in the SVD of M, as can be verified by simple calculations. Thus, we conclude the proof of Proposition F.2.

| |
|---|

- Proposition F.3. Let x = [a·I⊤L,b·I⊤K−L]⊤ ∈ RK, and X = diag(x)−K−1IK ·x⊤ ∈ RK×K, where a,b > 0. Then the SVD of X = UΣV T is that

a2 · (K − L) + b2 · L K

Σ = diag a · IL−1,b · IK−L−1,

,0 ,

RL,L−1 0K−L,L−1

0L,K−L−1 RK−L,K−L−1

V =

,

,v1,v2 ,

RL,L−1 0K−L,L−1

0L,K−L−1 RK−L,K−L−1

U =

,

,u1,u2 .

Here, the columns of the matrix RL,L−1 ∈ RL×(L−1) form an orthonormal basis for the subspace of vectors in RL orthogonal to IL. Similarly, the columns of RK−L,K−L−1 ∈ R(K−L)×(K−L−1) form an orthonormal basis for the subspace of vectors in RK−L orthogonal to IK−L. These correspond to the subspaces S1 and S2 defined as:

x 0K−L

0L y

x⊤IL = 0, and x ∈ RL , S2 =

y⊤IK−L = 0, and y ∈ RK−L . The vectors v1,v2,u1,u2 are

S1 =

√

a√K − L √

IL 0K−L −

L √K − L

b

1 a2(K − L) + b2L

0L IK−L v2 =

v1 =

L

IL 0K−L

1 a2(K − L) + b2L

0L IK−L

b

+ a

- u1 =

1 KL(K − L)

(K − L)

IL 0K−L − L

0L IK−L

- u2 =

1 √

IK.

K

- Proof of Proposition F.3. This proposition is a direct corollary of Proposition F.2. The matrix X = diag(x)− K−1IK · x⊤ is an instance of the general form Λ + C from Proposition F.2.

The diagonal part is Λ = diag(x) = diag(a·IL,b·IK−L). The off-diagonal part is C = −K−1IK ·x⊤. We can write C in block form:

IL IK−L

###### 1 K

1 K

aJL,L bJL,K−L aJK−L,L bJK−L,K−L

aI⊤L bI⊤K−L = −

C = −

. This corresponds to setting the block-wise constants in Proposition F.2 to:

c11 = −a/K, c12 = −b/K, c21 = −a/K, c22 = −b/K.

Substituting these into the formulas for α,β,γ,δ from Proposition F.2 gives: α = a + L(−a/K) = a(K − L)/K β = L(K − L)(−b/K) γ = L(K − L)(−a/K) δ = b + (K − L)(−b/K) = bL/K

These coefficients define the 2×2 matrix M from Proposition F.2 for this specific case. We now analyze this matrix M. A key observation is that its determinant is zero:

a(K − L) K

bL K −

det(M) = αδ − βγ =

L(K − L) K2

(−b)(−a) = 0.

Since the determinant is zero, one of its singular values must be zero. The other singular value, s1, can be calculated from the squared Frobenius norm (sum of squares of elements), which is also the sum of squared

singular values (s21 + s22):

L(K − L)b2 K2

a2(K − L)2 K2

s21 + 02 = α2 + β2 + γ2 + δ2 =

+

a2(K − L) + b2L K

=

.

L(K − L)a2 K2

+

+

b2L2 K2

This confirms the singular values stated in the proposition. The singular vectors v1,v2,u1,u2 can be derived by performing the SVD on this specific 2 × 2 matrix M.

| |
|---|

