## MolHIT: Advancing Molecular-Graph Generation with Hierarchical Discrete Diffusion Models

# arXiv:2602.17602v1[cs.AI]19Feb2026

Hojung Jung1,∗, Rodrigo Hormazabal2, Jaehyeong Jo1, Youngrok Park1, Kyunggeun Roh3, Se-Young Yun1, Sehui Han2, Dae-Woong Jeong2 1KAIST AI 2LG AI Research 3Seoul National University ghwjd7281@kaist.ac.kr

### Abstract

Molecular generation with diffusion models has emerged as a promising direction for AI-driven drug discovery and materials science. While graph diffusion models have been widely adopted due to the discrete nature of 2D molecular graphs, existing models suffer from low chemical validity and struggle to meet the desired properties compared to 1D modeling. In this work, we introduce MolHIT, a powerful molecular graph generation framework that overcomes long-standing performance limitations in existing methods. MolHIT is based on the Hierarchical Discrete Diffusion Model, which generalizes discrete diffusion to additional categories that encode chemical priors, and decoupled atom encoding that splits the atom types according to their chemical roles. Overall, MolHIT achieves new state-of-the-art performance on the MOSES dataset with near-perfect validity for the first time in graph diffusion, surpassing strong 1D baselines across multiple metrics. We further demonstrate strong performance in downstream tasks, including multi-property guided generation and scaffold extension.

### 1. Introduction

Molecular generation with AI has the potential to significantly speed up materials design (Sanchez-Lengeling and Aspuru-Guzik, 2018) and drug discovery (Zhang et al., 2025). While this promise has led to many different modeling strategies, generating valid and novel molecules is challenging due to the vast combinatorial search space (Dobson, 2004). Here, the primary challenge is not generating novel structures, but ensuring the structures remain chemically valid and relevant. Even a minor atom-level error can produce a structure that is chemically impossible or

∗Work done during an internship at LG AI Research.

99.1

###### Validity(%)

100

92.8

90.5

88.3

90

85.7

80

DiGress DisCo Cometh DeFoG MolHIT (Ours)

MolHIT (Ours) 2D Models 1D Models

100%

95%

Quality↑

90%

85%

80%

10% 20% 30% 40%

Scaffold Novelty ↑

Figure 1. MolHIT achieves SOTA result on MOSES dataset. (Top) Near-perfect validity, outperforming existing graph diffusion models. (Bottom) Pareto-optimal in quality-novelty trade-off.

synthetically inaccessible. Consequently, it is necessary to develop generative models that efficiently explore this immense chemical space while generating valid and synthesizable molecules.

One common approach is to treat molecules as 1D sequences, most commonly through the SMILES representation (Weininger, 1988). By representing molecular graphs into strings, these models can leverage powerful natural language processing techniques to learn patterns of characters. While this simpler learning objective results in generating valid molecules, they suffer from memorization, often reproducing patterns or common subsequences in the training set. This limited exploration capability creates a performance plateau as shown in Fig. 1, where high validity is achieved at the expense of a reduced number of new structures.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

- Figure 2. Overview of MolHIT. (a) Markov chain of Hierarchical Discrete Diffusion Model (HDDM). Clean states (S0) are transited to the mid-level states (S1) and finally to the masked state (S2). (b) Generation process of MolHIT. From the masked prior, atoms are denoised into mid-level states and then to atomic tokens in a coarse-to-fine manner. (c) Phase diagram of HDDM showing the transition probability of the forward process. (d) Decoupled atom encoding scheme, separately encoding the aromatic and charged atom types.

To overcome the exploration limits of sequence-based approaches, graph generative models (Jo et al., 2022; Liu et al., 2023) treat molecules as interconnected systems of atoms and bonds. Unlike 1D models that often overfit to specific textual patterns, graph-based architectures are designed to internalize the underlying topological principles of chemical structures, allowing them to generalize beyond the training set and discover novel structures. In particular, discrete diffusion models (Austin et al., 2021) have been widely studied for molecular graph generation as they naturally align with the categorical nature of atoms and bonds (Vignac et al., 2022; Xu et al., 2024; Qin et al., 2024; Seo et al., 2025).

While these models excel at structural exploration, they are prone to generating invalid or chemically unstable samples compared to well-optimized 1D models. This creates a performance gap that raises a fundamental research question: Can we leverage the inductive biases of graph modeling to match the validity of sequence models while maintaining their superior capacity for structural novelty?

We identify two critical limitations in existing molecular graph generation with discrete diffusion. (1) First, current uniform or absorbing transition treats each atom category as an independent category, even though there is well known chemical relationship that some atoms are easier to be replaced with another. Neglecting well-established domain priors often makes the learning unnecessarily hard, especially in molecular settings where high-quality molecule data is scarce. (2) Second, existing graph models rely on naive atom encodings, ignoring the fact that a single atom can have different characteristics when it has a formal charge

or consists of a ring (aromaticity). We reveal that this makes molecular graph generation tasks ill-posed and unnecessarily challenging, which we demonstrate in the reconstruction experiments in Fig. 3 where previous atom encoding fails.

In light of these observations, we introduce MolHIT, a hierarchical discrete diffusion framework designed to bridge the gap between structural innovation and chemical validity. Our framework is built upon the Hierarchical Discrete Diffusion Model (HDDM), where additional categories are added to represent natural chemical groups into the diffusion process. This coarse-to-fine approach allows the model to establish high-level chemical identities before refining them into specific atom types, thereby capturing the meaningful dependencies of molecular structure that uniform or absorbing kernels often overlook. Furthermore, we introduce Decoupled Atom Encoding (DAE) to resolve the information loss found in naive representations by explicitly split atoms based on their specific chemical roles, such as formal charge and aromaticity. By providing a chemical role into each token, DAE not only resolves the reconstruction problem in original atom encoding, but also reduces the burden of differentiating atom roles solely with the (O(n2)) bond features. Combined together, MolHIT reaches a new Pareto frontier in generating novel structures with high quality, surpassing both existing 1D and 2D models (Fig. 1).

We extensively evaluate MolHIT with experiments on large molecular benchmarks, including unconditional generation tasks on MOSES (Polykovskiy et al., 2020) and GuacaMol (Brown et al., 2019) benchmarks and conditional generation tasks, including scaffold extension and multi-

Coarse

Decoupled (Ours)

| |
|---|

| |
|---|

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

120

12

SuccessRate(%)

100

Percentage(%)

Reconstruction

10

80

8

60

6

40

4

20

2

0

0

[nH] Sources Non-[nH] Sources

Reference DiGress DeFOG Ours

(a) Reconstruction Success

(b) Molecules Containing [nH]

- Figure 3. Existing atom encoding for molecular graph is ill-posed. (Left) Reconstruction success rate on the Moses dataset with previous encoding and our decoupled atom encoding. (Right) Proportion of generated molecules containing pyrrolic nitrogen [nH].

property guided generation tasks. Across all benchmarks and tasks, MolHIT shows significant improvements over previous graph diffusion models, resulting in a new state-ofthe-art that surpass 1D models.

Our contributions can be summarized as follows:

- • We introduce MolHIT, a molecular graph diffusion model built upon a novel Hierarchical Discrete Diffusion Model (HDDM) framework with a mathematically guaranteed ELBO.
- • We identify a critical limitation in the prior graph generative models’ atom encoding and propose a simple solution: Decoupled Atom Encoding (DAE). By representing atoms based on their specific chemical roles, we find DAE enhances both the model’s generative expressiveness and chemical reliability.
- • We achieve the SOTA performance on the MOSES benchmarks in multiple metrics, significantly outperforming both existing graph diffusion models and 1D sequence-level baselines.
- • We test our algorithm on practical downstream tasks including multi-property guided generation and scaffold extension, achieving the highest performance compared to the previous graph diffusion approach.

### 2. Preliminaries

#### 2.1. Discrete Diffusion Models

Given a discrete state space S with K categories, discrete diffusion models define a noising and denoising process within a discrete space. Specifically, for x ∈ S, the noising process is described by a Markov chain as follows:

q(xt|xt−1) = Cat(xt;xt−1Qt). (1)

Here, marginal probability of xt in timestep t, given clean data x0 can be calculated with

q(xt|x0) = Cat(xt;x0Q¯t), Q¯t = QtQt−1 ···Q1. (2)

As shown by Austin et al. (2021), one can design multiple types of diffusion process, where two types of processes are

widely used because of the closed-form calculation of the forward process and natural noising process.

Uniform transition Uniform transition assumes uniform prior pT(xT = c) = K1 for all c ∈ {1,...,K}. Then one could define a forward noising process by interpolating clean data x0 with the prior in the following way:

1 K

q(xt|x0) = Cat(xt;(1 − α¯t)

11T + α¯tx0), (3)

where α¯t is monotonic decreasing function with α¯0] = 1,α¯T = 0, which we call diffusion scheduler.

Marginal transition To facilitate the diffusion learning, marginal transition assumes data prior π to be an optimal probability distribution that approximates the empirical data distribution from the training set. This has been primarily adopted for graph diffusion models DiGress (Vignac et al., 2022; Siraudin et al., 2024), where further details are in Appendix B.2.

Absorbing transition Unlike uniform and marginal transition where diffusion process operates on the given K categories, one can introduce an additional masked (absorbing) state m with prior em being a one-hot vector of the masked state. Then, one can naturally define a diffusion process as an absorbing process in a Markov chain, which results in the following forward form:

q(xt|x0) = Cat(xt;α¯tx0 + (1 − α¯t)em). (4)

⊤ t|s ⊙ x0Q¯t−1

Given q(xt−1|xt,x0) = Cat(xt−1; xtQ

x0Q¯txTt ), one could estimate posterior pθ(xt−1|xt,x0) by learning to estimate the clean data xˆ0 given the noisy data xt. This enables training the diffusion models with simple cross-entropy loss, where the loss function becomes directly linked to the negative evidence lower bound (NELBO) (Austin et al., 2021; Sahoo et al., 2024).

2.2. Molecular Graph Generation with Discrete Diffusion

Given molecular graph G = (X,E), denote X ∈ Rn×d

X, E ∈ Rn×n×d

E for the atom matrix and adjacency matrix (bond matrix) where n is the number of atoms and dX,dE are feature dimensions of atoms and edges. The forward process of discrete diffusion operates independently on the atom and bond matrices:

#### Gt = (Xt,Et) : Xt = X0Q¯X,t, Et = E0Q¯E,t. (5)

where, we define Q¯X,t = QX,t ···QX,1,Q¯E,t = QE,t ···QE,1 are forward transition matrix usually calculated in a closed form for efficiency.

Given a noisy graph Gt, a neural network is trained to estimate a clean graph G0 = (X0,E0) through predicting clean atoms and bonds independently, which in practice results in the following cross-entropy (CE) loss:

n

−log pXθ (X0,i | Gt,t)

Lθ = Et, G

t∼q(·|G0)

i=1

(6)

−log pEθ (E0,ij | Gt,t) ,

+ λ

1≤i<j≤n

where λ > 0 is a weighting factor that balances the relative contribution of node and edge loss.

### 3. MolHIT Framework

#### 3.1. Hierarchical Discrete Diffusion Models

We introduce Hierarchical Discrete Diffusion Models (HDDM), which generalize the discrete diffusion framework into a multi-stage setting. Unlike standard discrete diffusion (Austin et al., 2021), where the forward transitions operate either within the clean vocabulary space or toward an absorbing (masked) state, HDDM introduces additional mid-level states that bridge the corruption process.

To design a discrete diffusion in this augmented space, we first show that there exists a simple forward process that admits a tractable closed-form transition kernel. Specifically, for clean state space S0 with K categories, suppose we add additional G+1 categories such that we have an augmented state space T with cardinality D = K+G+1. As illustrated in Figure 2-(a), we partition T into three disjoint subsets: S0, mid-level states S1 with G categories, and the masked state S2 = {m}.

Now, we define the transition kernel via a row-stochastic matrix Φ ∈ [0,1]K×G, where Φij represents the probability of mapping any element i ∈ S0 to a mid-level element j ∈ S1. This operator induces a transition matrix Q(1) ∈ [0,1]D×D on the full space T , structured as a block matrix relative to the partition (S0,S1,S2):

 

 

0K×K Φ 0K×1

Q(1) =

- 0G×K IG 0G×1
- 01×K 01×G 1

Here, IG is the identity matrix, indicating that states in S1 are absorbing the clean states in S0 under Q(1). Similarly, we define the masking operation via the transition matrix Q(2), which maps all states in S0 ∪ S1 to the unique absorbing state in S2:

- 0(K+G)×(K+G) 1(K+G)×1
- 01×(K+G) 1

Q(2) =

These transition matrices form the basis of the HDDM forward process as in the following lemma:

Lemma 3.1. Define diffusion schedules αt,βt to be monotonically decreasing functions satisfying the boundary conditions α0 = β0 = 1 and α1 = β1 = 0, such that αt ≤ βt for all t. We define the forward diffusion process of the hierarchical Markov chain via the transition kernel Qt|s from timestep s to t as:

Qt|s = αt|sI+(βt|s−αt|s)Q(1)+(1−βt|s)Q(2), (7)

where αt|s := αt/αs,βt|s := βt/βs. Then, the transition kernels satisfy the Chapman–Kolmogorov equation,

such that Qt|sQs|r = Qt|r for any r < s < t. Consequently, the cumulative forward transition from the initial state to timestep t is given by:

Qt = αtI + (βt − αt)Q(1) + (1 − βt)Q(2), (8) where I denotes the identity matrix in T .

Note that the above forward transition operators can be naturally extended to arbitrary hierarchies in state space. We provide a proof of the above lemma with a generalized forward process in Appendix C.1.

Now for training guarantee, one can derive negative ELBO (NELBO), which we prove in Theorem C.2 in Appendix. In practice, one can define Φ as a deterministic projection that clusters clean atom categories into meaningful groups. We show in this special case, NELBO of HDDM can be further simplified as in the following.

Theorem 3.2. If the forward transition kernels Qt in Eq. 8 is induced from the deterministic projection Φ, the continuous time NELBO of HDDM is given as:

L∞NELBO(θ) = EQ,t

αt(βt′ − αt′) βt − αt

log ⟨xθ,x⟩ · I[zt ∈ S1]+ βt′(βt − αt) βt(1 − βt)

(9)

log⟨Q(1)xθ,Q(1)x)⟩ · I[zt = m]

αtβt′ βt(1 − βt)

log ⟨xθ,x⟩ · I[zt = m] + C,

+

for some constant C and the denoiser xθ(zt,t).

We provide a proof in Appendix C.2. For a sanity check, one can observe that Eq. 9 reduces to the NELBO of the original masked diffusion models when βt = αt (i.e, no S1). With Theorem 3.2, we can design a simple cross-entropy loss for HDDM training in a principled way. We empirically find that regularization loss in Eq. 9 does not improve the performance, so we take the original loss in Eq. 6.

#### 3.2. Decoupled Atom Encoding

Existing graph diffusion frameworks (Vignac et al., 2022; Xu et al., 2024; Qin et al., 2024) typically rely on a coarse atom encoding scheme, where node identities are determined solely by their atomic numbers. While this simplifies the encoding, we identify that this one-to-many mapping between atomic tokens and their physical states (e.g., protonation or aromaticity) causes the generative task to be ill-posed. As illustrated in Fig. 3 (Left), this leads to a systematic reconstruction failure in molecules requiring finegrained atomic descriptors, such as specific nitrogen motifs found in drug-like scaffolds. Consequently, models using these coarse encodings suffer from a representational bias, struggling to generate essential motifs that are statistically prevalent in the training distribution (Fig. 3, Right).

To resolve these representational gaps and ensure the model can generalize across diverse chemical spaces, we introduce Decoupled Atom Encoding (DAE). DAE expands atomic state space by explicitly encoding aromaticity and formal charge as primary node attributes. This results in a nearperfect reconstruction ratio both on the MOSES and GuacaMol dataset. Furthermore, by providing the model with necessary structural priors, MolHIT successfully recovers the distribution of complex motifs such as pyrrolic nitrogen ([nH]), which baselines using coarse encoding struggle to capture (Fig. 3, Right). Further details are in Appendix D.1.

Algorithm 1 PN-sampler with temperature sampling

- 1: Input: Sample size S, Timesteps T, Temperature τ, Nucleus threshold p
- 2: for i = 1 to S do
- 3: Sample N ∼ Ptrain(N)
- 4: GT ∼ pT(GT) {G = (X,E)}
- 5: for t = T down to ∆t with step ∆t do
- 6: pˆ0(X),pˆ0(E) ← fθ(Gt,t)
- 7: pˆ′0(X) ← TopP(Softmax(ˆp0(X)/τ),p)
- 8: Xˆ 0 ∼ Categorical(ˆp′0(X))
- 9: Eˆ0 ∼ Categorical(ˆp0(E))
- 10: Gt−∆t ∼ q(Gt−∆t|Gˆ0) where Gˆ0 = (Xˆ 0,Eˆ0)
- 11: end for
- 12: Store Gfinal
- 13: end for

properties and aromaticity. For instance, in the MOSES dataset, we partition 12 atom types into four semantic groups: {C}, {N,O,S}, {F,Cl,Br}, and {c,o,n,nH,s}. This hierarchical structure simplifies the initial stages of diffusion by focusing on broad chemical categories before refining specific identities. We extend this strategy to other datasets, such as GuacaMol, by adapting the groupings to their respective atom vocabularies. Full details of these partitions are provided in Appendix D.2

#### 3.3. Forward and Reverse Process of MolHIT

Forward process of MolHIT Since the atom and bond are perturbed independently throughout the forward process, we decouple their transition dynamics in graph diffusion. This flexibility is particularly advantageous for molecular graph modeling. We empirically observe that a uniform transition kernel is essential for edge generation, whereas HDDM yields superior performance for atom types compared to a uniform approach. Therefore, we employ an HDDM process for atoms and a uniform transition for edges, resulting in the following forward process dynamics:

QX,t = αX,tI + (βX,t − αX,t)Q(1)X,t + (1 − βX,t)Q(2)X,t, QE,t = αE,tI + (1 − αE,t)1d

1Td

, (10)

E

E

Our preliminary experiments show robustness on the HDDM scheduler αX,t, βX,t, and therefore we simply opt for linear schedule for αX,t = αE,t = 1 − t and βX,t = 1 − t2 for the experiments.

Grouping strategy Given the mid-level states S1, HDDM allows for the design of arbitrary transition kernels from S0 to S1. We implement a deterministic grouping kernel that clusters atom elements based on their intrinsic chemical

Project and Noise (PN-sampler) Due to the standard ELBO guarantee as we prove in Theorem 3.2, one can sample from the original posterior update as in prior works (Austin et al., 2021). While standard posterior updates follow the transition q(Gt−∆t|Gt,G0) as justified by the ELBO guarantee in Theorem 3.2, we empirically find that this approach often restricts the structural exploration necessary for complex molecular generation. To address this, we design a Project-and-Noise (PN) sampler. PN sampler projects model’s denoising prediction pθ(G0|Gt) onto the clean manifold M (one-hot vector) via categorical sampling to obtain a discrete candidate Gˆ0. This candidate is then directly re-noised to the preceding timestep s = t−∆t using the cumulative transition kernel Qt, effectively bypassing posterior constraints of Gt to encourage greater diversity in the generated graph. The overall algorithm is illustrated in Alg. 1.

Temperature sampling While temperature and top-p sampling have become standard techniques for managing the quality-diversity trade-off in generative domains (Holtzman et al., 2019; Ficler and Goldberg, 2017; Hashimoto et al., 2019), their application to molecular graph generation remains largely unexplored. We evaluate the impact of these sampling strategies and demonstrate that our PNsampler effectively controls this trade-off. We empirically

Table 1. Comprehensive MOSES benchmark results. Scaffold Novelty (Scaf-Novel) measures the ratio of novel scaffold molecules to the number of generated molecules, while Scaffold Retrieval (Scaf-Ret.) quantifies test scaffold retrievals. All of the results are the averaged value over 3 runs of 25,000 samples. Bold denotes the best in each category, and underline indicates SOTA performance within the 2D Graph models. Empty values are due to the absence of publicly available checkpoints or samples.

Category Model Quality ↑ Scaf-Novel ↑ Scaf-Ret. ↑ Valid ↑ Unique ↑ Novel ↑ Filters ↑ FCD ↓ SNN ↑ Scaf ↑

- Training set 95.4 — — 100.0 100.0 — 100.0 0.48 0.59 -

- 1D Sequence VAE[25] 92.8 0.22 0.031 97.7 99.7 69.5 99.7 0.57 0.58 5.9 CharRNN [42] 92.6 0.29 0.035 97.5 99.9 84.2 99.4 0.52 0.56 11.0 SAFE-GPT [33] 92.8 0.12 0.015 99.8 98.9 43.7 97.7 0.72 0.57 6.3 GenMol [27] 62.1 0.05 0.012 99.7 64.0 68.9 98.1 16.4 0.64 1.6

- 2D Graph DiGress [47] 82.5 0.26 0.031 87.1 100.0 94.2 97.5 1.25 0.53 12.8 DisCo [51] - - - 88.3 100.0 97.7 95.6 1.44 0.50 15.1 Cometh [45] 82.1 0.36 0.023 87.2 100.0 96.4 97.3 1.44 0.51 16.8 DeFoG [38] 88.5 0.26 0.031 92.8 99.9 92.1 98.9 1.95 0.55 14.4 MolHIT 94.2 0.39 0.033 99.1 99.8 91.4 98.0 1.03 0.55 14.4

find that temperature sampling can be naturally adopted for PN-sampler, where doing temperature sampling only for the atom prediction (line 7 in Alg. 1) results in the best performance.

#### 3.4. Conditional Modeling

To enable conditional modeling, we train a conditional model by modifying the original graph transformer architecture in DiGress (Vignac et al., 2022) by adding adaptive layer normalization (adaLN) for node attention only. For sampling, we adopt classifier-free guidance (CFG) (Ho and Salimans, 2022). We provide the details in Appendix D.8.

Table 2. Full GuacaMol benchmark results using unfiltered dataset. Metric abbreviations: Val. (Validity), V.U. (Unique), V.U.N. (Novel). DiGress (org.) is original DiGress trained with filtered dataset and DiGress (full) values are from the re-implementation of DiGress on unfiltered, full GuacaMol dataset.

Model Val. V.U. V.U.N. KL Div. FCD Training set 100.0 100.0 — 99.9 92.8 DiGress [47] (org.) 85.2 85.2 85.1 92.9 68.0 DiGress [47] (full) 74.7 74.6 74.0 92.4 61.1 DiGress + DAE 65.2 65.2 64.9 87.0 49.2 MolHIT (Ours) 87.1 87.1 86.0 96.7 54.9

metrics are provided in Appendix D.5.

### 4. Experiments

We evaluate MolHIT on two large-scale molecular datasets: MOSES (Polykovskiy et al., 2020) and Guacamol (Brown et al., 2019). The MOSES dataset consists of 1.9M molecules containing 7 heavy atom types, which we augment into 12 tokens using DAE (Sec. 3.2). Similarly, the GuacaMol (Brown et al., 2019) dataset, which originally contains 12 heavy atom types, is decoupled into 56 tokens via DAE. For the model architecture, we utilize the original graph transformer from DiGress (Vignac et al., 2022), maintaining the same model size. All reported results represent the average of three independent runs, and standard deviations are provided in Appendix D.3.

Scaffold novelty metrics While the standard MOSES benchmark provides a foundation for evaluating molecular generative models, simple metrics like novelty may not reflect the capability for generating new molecules. For instance, a high novelty score itself can come from merely generating novel-looking noise outside the manifold of drug-like molecules, while high uniqueness may not reflect true structural diversity if the model is trapped in a narrow chemical subspace. To address this, we introduce two metrics given ntotal generated molecules: (1) Scaffold Novelty = |Sgen \ Strain|/ntotal, which quantifies the efficiency of structural extrapolation; and (2) Scaffold Retrieval = |Sgen ∩Stest|/ntotal, which measures distributional fidelity. Further details are in Appendix D.6.

#### 4.1. Unconditional Generation on MOSES

Evaluation Following previous graph diffusion works (Vignac et al., 2022; Qin et al., 2024), we measure with official benchmarks for Moses (Polykovskiy et al., 2020) which includes 7 metrics: Validity (%), Uniqueness (%), Novelty (%), Filters (%), FCD, SNN, Scaf. We also measure Quality (Lee et al., 2025), which is defined by the proportion of molecules that are valid, unique, synthetic accessibility (SA (Bickerton et al., 2012) ≤ 4), and drug-like (QED (Ertl and Schuffenhauer, 2009) ≥ 0.6). Formal definitions of the

Baselines For graph generative models, we compare with DiGress (Vignac et al., 2022), DisCo (Xu et al., 2024), Cometh (Siraudin et al., 2024), DeFoG (Qin et al., 2024), which are previous SOTA in atom-level graph diffusion. We also compare with 1D baselines; VAE (Kingma and Welling, 2013), Char-RNN (Segler et al., 2018), SAFE-GPT (Noutahi et al., 2024), GenMol (Lee et al., 2025).

Result As shown in Table 1, MolHIT significantly outperforms previous graph-based baselines across nearly all

- Table 3. Multi-property guided generation on MOSES with four different conditions. We report mean absolute error (MAE), Pearson correlation (Pearson r), and validity. Avg. denotes the macro-average across four properties. Bold denotes best performances. All results are the averaged value over 3 runs of 10,000 samples.

MAE ↓ Pearson r ↑ Validity (%) ↑ Method QED SA logP MW Avg. QED SA logP MW Avg.

Marginal 0.117 0.115 0.067 0.272 0.143 0.489 0.570 0.802 0.396 0.564 75.03 Marginal + DAE 0.107 0.094 0.061 0.227 0.122 0.565 0.559 0.836 0.437 0.599 87.85

MolHIT (Ours) 0.061 0.040 0.049 0.081 0.058 0.804 0.790 0.950 0.685 0.807 96.31

- Table 4. Scaffold extension results on the MOSES dataset. Results are averaged over 3 runs of 10,000 targets.

Table 5. Incremental performance gains on the MOSES dataset by integrating DAE, the PN Sampler, and HDDM into the DiGress.

Method Quality ↑ FCD ↓ Validity (%) ↑

Model Validity (%) ↑ Diversity ↑ Hit@1 ↑ Hit@5 ↑

DiGress [47] 82.5 1.25 87.1 + DAE 87.6 0.89 96.2 + PN Sampler 92.9 1.65 99.4 + HDDM (MolHIT) 94.2 1.03 99.1

DiGress 50.8 44.8 2.07 6.41 Marginal + DAE 64.8 58.0 1.67 6.37 MolHIT (Ours) 83.9 57.4 3.92 9.79

key metrics, including Quality, Validity, FCD, and Scaffold Novelty. While 1D sequence-based models (SAFE-GPT, GenMol) excel in Validity, they exhibit a clear tendency toward memorization, evidenced by their lower Scaf-Novelty and novelty scores. On the other hand, MolHIT achieves a new state-of-the-art both for Quality (94.2%) and Scaffold Novelty (0.39) while achieving near perfect validity score (99.1). The above results validate that MolHIT effectively navigates the valid drug-like manifold without sacrificing its ability to explore novel chemical space.

#### 4.2. Unconditional Generation on GuacaMol

Setup Compared to MOSES where molecules contain charged atoms are filtered, GuacaMol benchmark (Brown et al., 2019) contains a broader chemical space, including compounds with formal charges that are not eliminated by neutralization. Previous atom encoding (Vignac et al., 2022) fails to reconstruct these properties (Fig. 5), and they train model only with a manually filtered dataset which are failed to be reconstructed. This helps improve the validity measure, but making models learn from the imperfect, biased distribution. In contrast, we utilize the full GuacaMol dataset for training to evaluate the robustness of our model. We run 3 run of generating 10,000 samples for each experiment.

#### 4.3. Multi-property guided generation

Generating molecules with targeted chemical properties is important for practical applications in materials science and drug discovery. For this, we evaluate the capacity of MolHIT under the multi-conditional generation scenario.

Setup We train a conditional graph transformer (Sec. 3.4) on the MOSES dataset, labeled with four key chemical properties: Quantitative Estimate of Drug-likeness (QED), Synthetic Accessibility (SA), Molecular Weight (MW), and the lipophilicity (logP). We utilize RDKit (Landrum, 2006), an open-source cheminformatics toolkit, for all property labeling and condition evaluation.

Evaluation For inference, we generate 10,000 samples conditioned on target properties of the molecules that are randomly sampled from the test split. We measure Mean Absolute Error (MAE) and the Pearson correlation coefficient (r) for conditioning and validity for the structural fidelity of the samples. We compare MolHIT against two baselines: (1) Marginal transition (effectively a DiGress without a geometric prior) and (2) Marginal transition with a DAE (incorporating decoupled atom encoding into the marginal transition baseline).

Results Table 2 shows that MolHIT achieves the highest performance among all metrics except FCD. For FCD, the strong performance of original DiGress without DAE indicates that using DAE does not always lead to the generative task being easier since it can be hard to model with differentiate extended atom vocabulary. However, as in Appendix D.1, we find that using DAE substantially increase the amount of molecules having charged or special atoms, which is not rare in the GuacaMol. Note that the original DiGress is trained for 1,000 epochs, while our results are from training only with 40 epochs, so further training will improve the metrics.

Results Table 3 shows that MolHIT significantly outperforms all baselines across every metric. For conditioning precision, MolHIT achieves a macro-averaged MAE of 0.058, a 52.4% reduction compared to the Marginal+DAE baseline. MolHIT also exhibits high reliability, reaching a Pearson r of 0.807 on average, including a near-perfect 0.950 for log P and 0.804 for QED. The results also show this improved conditioning does not come with the cost of lower validity, where MolHIT achieves validity higher than 95%, outperforming baselines with a large gap. We provide more experimental details of the multi-property guided generation in Appendix D.8.

MolHIT (Ours) 2D Graph Baselines

1D Seq. Baselines

| |
|---|

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

95%

Quality↑

90%

85%

0.1 0.2 0.3 0.4

Scaffold Novelty ↑

Figure 4. Effect of top-p sampling in MolHIT.

#### 4.4. Scaffold Extension

Setup We evaluate pretrained unconditional model’s generative capability when conditioned on a given substructure. For this, we use Bemis-Murcko scaffold (Bemis and Murcko, 1996) of the molecules in the test split and treat fix this part during the diffusion sampling. For each experiment, we utilize 10,000 unique target scaffolds, generating multiple candidates per target to assess the model’s distributional coverage. Specifically, we measure the Hit@1 and Hit@5 ratios, which are the probability that the ground-truth extension is recovered within the top k samples along with standard metrics of validity and diversity. Further experimental setup is provided in Appendix D.9.

Result Table 4 shows that MolHIT significantly outperforming DiGress in all metrics. Interestingly, applying DAE to DiGress improves validity and diversity while reducing in Hit@1, which may due to the extended expressivity of the model. However, DAE results in higher diversity, which results in matched Hit@5 ratio for original DiGress.

#### 4.5. Ablation Studies

Component analysis To show the contribution of each component on MolHIT’s performance, we conduct an ablation study by testing on the Moses dataset. The result in Table 5 shows that our atom encoding method (DAE), sampler (PN sampler), and diffusion algorithm (HDDM) all contributes to get to the highest value of Quality, FCD, and Validity among graph diffusion models.

Effect of temperature sampling In Fig. 4, we analyze MolHIT trained on the MOSES dataset across a range of topp values. Our results demonstrate that as top-p decreases, a clear trade-off emerges between sample quality and scaffold novelty. Specifically, lowering the top-p value from

1.0 down to 0.8 consistently improves the quality and validity of generated structures, while further reducing the p threshold leads to a sharp decline in both chemical metrics and structural diversity. Notably, when sampling with top-p, MolHIT achieves a high validity of 99.4% and a quality score of 95.1%, demonstrating the effectiveness of the nucleus sampling in MolHIT.

### 5. Related Works

Discrete diffusion models Along with the success of continuous diffusion models (Ho et al., 2020; Song et al., 2020), discrete diffusion models formulate a noise process within a discrete state space. Hoogeboom et al. (2021) investigate uniform transition of the discrete diffusion models, while D3PM (Austin et al., 2021) explore different types of transition mechanism which include absorbing transition. Recently and independently developed alongside our work, Zhou et al. (2025) propose a hierarchical discrete diffusion approach to language modeling. While similar in spirit, our HDDM is derived from a semigroup-consistent family of closed-form transition kernels Q(1),Q(2) parameterized by explicit diffusion scheduler αt,βt while Zhou et al. (2025) is developed in the CTMC framework (Campbell et al., 2022). Moreover, HDDM supports an arbitrary row-stochastic projection Φ, which generalizes the deterministic hierarchical mapping used in Zhou et al. (2025).

Diffusion models for molecular generation Various diffusion models and its techniques have been applied for molecular graph generation. GDSS (Jo et al., 2022) formulate continuous diffusion modeling through the system of SDE with a score matching objective. DiGress (Vignac et al., 2022) utilize primary form of discrete diffusion models with uniform-style transition with data dependent prior. Siraudin et al. (2024); Xu et al. (2024); Qin et al. (2024) apply CTMC framework as in Campbell et al. (2022) to further boost the performance. Another axis for molecular generation is to model 1D sequence. SAFE-GPT (Noutahi et al., 2024) trains an Autoregressive model with their unique representation of molecule while GenMol (Lee et al., 2025) adopts masked diffusion framework in a wide range of drug discovery tasks. We defer further related works in Appendix B.1.

### 6. Conclusion

In this work, we present MolHIT, a novel molecular diffusion model with a hierarchical discrete diffusion framework. Our algorithm results in state-of-the-art performance in large molecular datasets. It unlocks new capacity for end-to-end atom-level molecular generation, directly generating atoms with formal charges or explicit nH for the first time, moving us towards more realistic molecule generation.

### Impact Statement

This paper presents work whose goal is to advance the field of molecule generation. We hope our work accelerates the discovery of useful drugs and materials, improving human lives. However, one might maliciously use our model to generate harmful substances to humans and environments.

Rick Durrett. Probability: theory and examples, volume 49. Cambridge university press, 2019. 14

Peter Ertl and Ansgar Schuffenhauer. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal of cheminformatics, 1(1):8, 2009. 6, 22

### References

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021. 2, 3, 4, 5, 8

Guy W Bemis and Mark A Murcko. The properties of known drugs. 1. molecular frameworks. Journal of medicinal chemistry, 39(15):2887–2893, 1996. 8, 21

G Richard Bickerton, Gaia V Paolini, J´er´emy Besnard, Sorel Muresan, and Andrew L Hopkins. Quantifying the chemical beauty of drugs. Nature chemistry, 4(2):90–98, 2012. 6

Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, et al. Llada2. 0: Scaling up diffusion language models to 100b. arXiv preprint arXiv:2512.15745, 2025. 13

Nathan Brown, Marco Fiscato, Marwin HS Segler, and Alain C Vaucher. Guacamol: benchmarking models for de novo molecular design. Journal of chemical information and modeling, 59(3):1096–1108, 2019. 2, 6, 7, 18

Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022. 8, 12

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11315–11325, 2022. 12

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019. 12

Christopher M Dobson. Chemical space and biology. Nature, 432(7019), 2004. 1

Jessica Ficler and Yoav Goldberg. Controlling linguistic style aspects in neural language generation. arXiv preprint arXiv:1707.02633, 2017. 5

Nate Gruver, Samuel Stanton, Nathan Frey, Tim GJ Rudner, Isidro Hotzel, Julien Lafrance-Vanasse, Arvind Rajpal, Kyunghyun Cho, and Andrew G Wilson. Protein design with guided discrete diffusion. Advances in neural information processing systems, 36:12489–12517, 2023. 12

Tatsunori B Hashimoto, Hugh Zhang, and Percy Liang. Unifying human and statistical evaluation for natural language generation. arXiv preprint arXiv:1904.02792, 2019. 5

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6, 22

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 8, 22

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751, 2019. 5

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forr´e, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in neural information processing systems, 34:12454–12465, 2021. 8

Emiel Hoogeboom, Vıctor Garcia Satorras, Cl´ement Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3d. In International conference on machine learning, pages 8867–8887. PMLR, 2022. 12

John J Irwin and Brian K Shoichet. Zinc- a free database of commercially available compounds for virtual screening. Journal of chemical information and modeling, 45(1): 177–182, 2005. 18

Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. In International conference on machine learning, pages 2323–2332. PMLR, 2018. 12

Jaehyeong Jo, Seul Lee, and Sung Ju Hwang. Score-based generative modeling of graphs via the system of stochastic differential equations. In International conference on machine learning, pages 10362–10383. PMLR, 2022. 2, 8

Hojung Jung, Youngrok Park, Laura Schmid, Jaehyeong Jo, Dongkyu Lee, Bongsang Kim, Se-Young Yun, and Jinwoo Shin. Conditional synthesis of 3d molecules with time correction sampler. Advances in Neural Information Processing Systems, 37:75914–75941, 2024. 12

Seo Hyun Kim, Sunwoo Hong, Hojung Jung, Youngrok Park, and Se-Young Yun. Klass: Kl-guided fast inference in masked diffusion models. arXiv preprint arXiv:2511.05664, 2025. 12

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 6, 19

Greg Landrum. RDKit: Open-source cheminformatics. ht tps://www.rdkit.org, 2006. 7

Seul Lee, Karsten Kreis, Srimukh Prasad Veccham, Meng Liu, Danny Reidenbach, Yuxing Peng, Saee Paliwal, Weili Nie, and Arash Vahdat. Genmol: A drug discovery generalist with discrete diffusion. arXiv preprint arXiv:2501.06158, 2025. 6, 8, 19, 20

Chengyi Liu, Wenqi Fan, Yunqing Liu, Jiatong Li, Hang Li, Hui Liu, Jiliang Tang, and Qing Li. Generative diffusion models on graphs: Methods and applications. arXiv preprint arXiv:2302.02591, 2023. 2

Gang Liu, Jiaxin Xu, Tengfei Luo, and Meng Jiang. Graph diffusion transformers for multi-conditional molecular generation. Advances in Neural Information Processing Systems, 37:8065–8092, 2024. 13, 22

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023. 12

David Mendez, Anna Gaulton, A Patr´ıcia Bento, Jon Chambers, Marleen De Veij, Eloy F´elix, Mar´ıa Paula Magari˜nos, Juan F Mosquera, Prudence Mutowo, Michał Nowotka, et al. Chembl: towards direct deposition of bioassay data. Nucleic acids research, 47(D1):D930– D940, 2019. 18

Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling up masked diffusion models on text. arXiv preprint arXiv:2410.18514, 2024. 13

Emmanuel Noutahi, Cristian Gabellini, Michael Craig, Jonathan SC Lim, and Prudencio Tossou. Gotta be safe: a

new framework for molecular design. Digital Discovery, 3(4):796–804, 2024. 6, 8, 19

Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024. 13, 15

Youngrok Park, Hojung Jung, Sangmin Bae, and SeYoung Yun. Temporal alignment guidance: Onmanifold sampling in diffusion models. arXiv preprint arXiv:2510.11057, 2025. 12

Daniil Polykovskiy, Alexander Zhebrak, Benjamin SanchezLengeling, Sergey Golovanov, Oktai Tatanov, Stanislav Belyaev, Rauf Kurbanov, Aleksey Artamonov, Vladimir Aladinskiy, Mark Veselov, et al. Molecular sets (moses): a benchmarking platform for molecular generation models. Frontiers in pharmacology, 11:565644, 2020. 2, 6, 20, 21

Kristina Preuer, Philipp Renz, Thomas Unterthiner, Sepp Hochreiter, and Gunter Klambauer. Fr´echet chemnet distance: a metric for generative models for molecules in drug discovery. Journal of chemical information and modeling, 58(9):1736–1741, 2018. 21

Yiming Qin, Manuel Madeira, Dorina Thanou, and Pascal Frossard. Defog: Discrete flow matching for graph generation. arXiv preprint arXiv:2410.04263, 2024. 2, 5, 6, 8, 19, 20, 21

Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024. 3, 12, 15

Benjamin Sanchez-Lengeling and Al´an Aspuru-Guzik. Inverse molecular design using machine learning: Generative models for matter engineering. Science, 361(6400): 360–365, 2018. 1

Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dalla-torre, Bernardo P de Almeida, Alexander Rush, Thomas Pierrot, and Volodymyr Kuleshov. Simple guidance mechanisms for discrete diffusion models. arXiv preprint arXiv:2412.10193, 2024. 13

Marwin HS Segler, Thierry Kogej, Christian Tyrchan, and Mark P Waller. Generating focused molecule libraries for drug discovery with recurrent neural networks. ACS central science, 4(1):120–131, 2018. 6, 19

Hyunjin Seo, Taewon Kim, Sihyun Yu, and SungSoo Ahn. Learning flexible forward trajectories for masked molecular diffusion. arXiv preprint arXiv:2505.16790, 2025. 2

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37:103131–103167, 2024. 13, 15

Antoine Siraudin, Fragkiskos D Malliaros, and Christopher Morris. Cometh: A continuous-time discrete-state graph diffusion model. arXiv preprint arXiv:2406.06449, 2024. 3, 6, 8, 19, 20, 21

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 8

Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. Digress: Discrete denoising diffusion for graph generation. arXiv preprint arXiv:2209.14734, 2022. 2, 3, 5, 6, 7, 8, 19, 20, 21, 22

David Weininger. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. Journal of chemical information and computer sciences, 28(1):31–36, 1988. 1

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025. 12

Minkai Xu, Alexander S Powers, Ron O Dror, Stefano Ermon, and Jure Leskovec. Geometric latent diffusion models for 3d molecule generation. In International Conference on Machine Learning, pages 38592–38610. PMLR, 2023. 12

Zhe Xu, Ruizhong Qiu, Yuzhong Chen, Huiyuan Chen, Xiran Fan, Menghai Pan, Zhichen Zeng, Mahashweta Das, and Hanghang Tong. Discrete-state continuoustime diffusion for graph generation. Advances in Neural Information Processing Systems, 37:79704–79740, 2024. 2, 5, 6, 8, 19

Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025. 13

Kang Zhang, Xin Yang, Yifei Wang, Yunfang Yu, Niu Huang, Gen Li, Xiaokun Li, Joseph C Wu, and Shengyong Yang. Artificial intelligence in drug development. Nature medicine, 31(1):45–59, 2025. 1

Cai Zhou, Chenyu Wang, Dinghuai Zhang, Shangyuan Tong, Yifei Wang, Stephen Bates, and Tommi Jaakkola. Next semantic scale prediction via hierarchical diffusion language models. arXiv preprint arXiv:2510.08632, 2025. 8

### Table of contents

- A Limitation and future directions 12
- B Further background 12

- B.1 Further related works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- B.2 Details of masked diffusion models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- C Mathematical derivations 13

- C.1 Generalized HDDM forward process . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- C.2 Proof of Theorem 3.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- D Experiment details 17

- D.1 Decoupled Atom Encoding (DAE) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- D.2 Grouping in HDDM . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- D.3 Full experimental results with standard deviations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- D.4 Implementation of baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.5 Unconditional generation with MOSES and GuacaMol . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.6 Structure novelty metric . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.7 Unconditional generation with GuacaMol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.8 Multi-property guided generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.9 Scaffold extension . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

### A. Limitation and future directions

Limitations While our models improve the traditional diffusion based molecular generation, we have not tested with the model size increase or architectural improvement, in which we believe have further room for better performance. Moreover, we have not fully trained the model until performance saturation on GuacaMol dataset and we believe performance improvement with further training.

Future directions There are many interesting future directions. One is to apply Hierarchical Discrete Diffusion Models to the language domains (Sahoo et al., 2024), or image models (Chang et al., 2022) and combining with different sampling schemes for diffusion models (Jung et al., 2024; Park et al., 2025; Wu et al., 2025; Kim et al., 2025). Another direction is to further improve MolHIT’s framework with more advanced tokenization incorporating motifs or functional groups (Jin et al.,

- 2018) and apply into the 3D molecular generation (Hoogeboom et al., 2022; Xu et al., 2023) and proteins (Gruver et al., 2023).

### B. Further background

#### B.1. Further related works

Further backgrounds on discrete diffusion models Recently, Masked Language Models are actively studied due to its simple form and potential of bi-directional modeling (Devlin et al., 2019). Campbell et al. (2022) formulate the discrete diffusion using Continuous Time Markov Chain (CTMC) framework and propose correction sampler leveraging tau-leaping. SEDD (Lou et al., 2023) introduce score entropy loss in analogous to the score matching loss in the continuous diffusion models and show scalability in language modeling. Recently, masked diffusion models is further simplified (Sahoo et al.,

2024; Ou et al., 2024; Shi et al., 2024) and reaching to the level that is comparable to the standard AR modeling in large scale (Nie et al., 2024; Bie et al., 2025) and even in multi-modal setting (Yang et al., 2025).

Conditional generation with discrete diffusion For conditional generation, Liu et al. (2024) propose graph diffusion transformer for multi-conditional generation on polymer dataset and shows the effectiveness of the classifier-free guidance. Schiff et al. (2024) propose simple mechanism for conditional sampling which is in analogous with CFG in continuous diffusion.

#### B.2. Details of masked diffusion models

Marginal transition Let X = (X1,...,XN) be a discrete random vector with Xk ∈ {1,...,K}, and let P denote the empirical data distribution on {1,...,K}N. Define the family of product of distributions:

C = q : q(x) =

N

qk(xk), qk ∈ ∆K−1 , (11)

k=1

where ∆K−1 is the (K−1)-simplex. Then, one can define the marginal terminal distribution π as the product of the data marginals:

π(x) =

N

πk(xk), πk(a) = PX∼P[Xk = a], a ∈ {1,...,K}. (12)

k=1

Equivalently, π is the (unique) KL projection of P onto C:

KL(P ∥q). (13)

π = arg min

q∈C

Then, marginal transition defines forward process of discrete diffusion as follows:

q(xt|x0) = Cat(xt;α¯tx0 + (1 − α¯t)π). (14)

### C. Mathematical derivations

#### C.1. Generalized HDDM forward process

We now derive a generalized forward process that incorporates arbitrary multi-level hierarchies. Let T be the total discrete state space with dimension D = nk=0 Kk. We partition T into n + 1 disjoint subsets S0,S1,...,Sn, where S0 represents the clean atomic states (with |S0| = K0) and Sk represents the k-th level of intermediate hierarchical states (with |Sk| = Kk). We further define the cumulative subspace up to level i as Ti := ik=0 Sk. For each hierarchical stage i ∈ {1,...,n}, we define the transition kernel as a row-stochastic matrix Φi ∈ [0,1]|T

i−1|×Ki. This kernel encodes the probabilistic mapping from the cumulative lower-level states in Ti−1 to the specific higher-level states in Si. To characterize the evolution in full space T , we induce a global transition matrix Q(i) ∈ [0,1]D×D on T which embeds the local kernel Φi into the full space as follows:

 

Φi(xnext | x) if x ∈ Ti−1 and xnext ∈ Si, 1 if x ∈ T \ Ti−1 and xnext = x, 0 otherwise.

Q(i)(xnext | x) =

(15)



In matrix notation, Q(i) forms a block structure where the transitions from Ti−1 are governed by Φi, while the remaining diagonal blocks form an identity matrix. Under this formulation, each Q(i) represents the probabilistic projection onto the i-th hierarchical level, enabling us to design the diffusion forward process with the following lemma:

Proposition C.1. Suppose monotonically decreasing functions αt(i) := αi(t) (i = 1,2,...,n) defined in 0 ≤ t ≤ 1 are satisfying 0 ≤ αt(1) ≤ ··· ≤ αt(n) ≤ 1 and boundary conditions α0(i) = 1,α1(i) = 0 for all i. We define the transition matrix from timestep s to timestep t (s ≤ t) as:

Qt|s := αt(1)|sI + (αt(2)|s − αt(1)|s)Q(0) + ··· + (1 − αt(|ns))Q(n), (16) where αt(|is) = α

(i) t

for every i. Then, transition kernel defined by Eq. 16 satisfies Chapman–Kolmogorov consistency (Durrett,

α(si)

##### 2019) as follows: Qt|sQs|r = Qt|r ∀0 ≤ r ≤ s ≤ t ≤ 1. (17)

Moreover, one could represent cumulative forward transition from initial timestep 0 to t in the following form:

Qt = αt(1)I + (αt(2) − αt(1))Q(1) + ···(1 − αt(n))Q(n). (18)

Proof. First, we can observe Q(i) is a projection operator; i.e, Q(i)2 = Q(i) for all i by definition. In fact, this can be generalized as Q(i)Q(j) = Qmax(i,j) for any 1 ≤ i,j ≤ n by the definition of the Q(i) and φi. Now, suppose Eq. 16 holds for some j ∈ N. Then, one can observe:

Qt|sQs|r

= αt(1)|sI + (αt(2)|s − αt(1)|s)Q(0) + ··· + (1 − αt(|js+1))Q(j+1) αs(1)|rI + (αs(2)|r − αs(1)|r)Q(0) + ··· + (1 − αs(j|r+1))Q(j+1)

= αt(1)|sI + (αt(2)|s − αt(1)|s)Q(0) + ··· + (αt(|js+1) − αt(|js))Q(j) αs(1)|rI + (αs(2)|r − αs(1)|r)Q(0) + ··· + (αs(j|r+1) − αs(j|r))Q(j)

+ αt(1)|s + ··· + (1 − αt(|js+1)) 1 − αs(j|r+1) Q(j+1) + 1 − αt(|js+1) αs|r + ···(1 − αs(j|r+1)) Q(j+1)

+ 1 − αt(|js+1))(1 − αs(j|r+1)) (Q(j+1))2

 

  · αs(j|r+1)

 

 

αs(1)|r αs(j|r+1)

(αt(|js+1) − αt(|js)) αt(|js+1)

(αs(j|r+1) − αs(j|r)) αs(j|r+1)

αt(1)|s αt(|js+1)

= αt(|js+1)

Q(j)

Q(j)

I + ··· +

I + ··· +

+ αt(1)|s + ··· + (1 − αt(|js+1)) 1 − αs(j|r+1) Q(j+1) + 1 − αt(|js+1) αs|r + ···(1 − αs(j|r+1)) Q(j+1)

+ 1 − αt(|js+1))(1 − αs(j|r+1)) (Q(j+1))2

= αt(1)|rI + (αt(2)|r − αt(1)|r)Q(0) + ··· + (1 − αt(|jr))Q(j) + αt(1)|s + ··· + (1 − αt(|js+1)) 1 − αs(j|r+1) Q(j+1)

+ 1 − αt(|js+1) αs|r + ···(1 − αs(j|r+1)) Q(j+1) + 1 − αt(|js+1))(1 − αs(j|r+1)) (Q(j+1))2

= αt(1)|rI + (αt(2)|r − αt(1)|r) + ··· + (1 − αt(|jr+1))Q(j+1),

(19) where the second to the last equation comes from the inductive assumption on j. Since j = 1 case is trivial, the result follows by mathematical induction on j.

| |
|---|

Proof of Lemma 3.1 Lemma 3.1 is now just a special case of above generalized formula in Proposition C.1.

#### C.2. Proof of Theorem 3.2

Let m be the one-hot representation of the masked state which we take as a prior, and let x ∈ S0 denote an element in a clean state. For a forward transition kernel Qt|s defined induced by the row stochastic matrix Φ ∈ [0,1]K×G and the masking operation as in Lemma 3.1, the conditional transition is defined as q(zt|zs) = Cat(zt;zsQt|s). By applying the closed-form transition from Lemma 3.1 and Bayes’ rule, the posterior distribution q(zs|zt,x) can be derived as follows:

αt|sI + (βt|s − αt|s)Q(1) + (1 − βt|s)Q(2) ⊺ zt ⊙ αsx + (βs − αs)Q(1)x + (1 − βs)m zTt [αtI + (βt − αt)Q(1) + (1 − βt)Q(2)]Tx

q(zs|zt,x) = Cat zs;

(20)

We can divide into the following 3 cases depending on which state zt belongs to.

- Case 1. zt ∈ S0 q(zs|zt,x) = Cat(zs;x). (21)
- Case 2. zt ∈ S1 When the observed state at time t belongs to the mid-level space S1, the posterior depends on the state of zt under the stochastic transition. Using the block structure of the transition kernels, we can obtain:

q(zs|zt,x) = Cat zs;

(αsβt|s − αt)x + (βt − βt|sαs)zt βt − αt

, (22)

- Case 3. zt ∈ S2 = {m} When the observed state is the mask token m ∈ S2, the posterior distribution q(zs|zt,x) becomes a weighted combination of the clean state, its stochastic projection, and the mask prior. Using the normalization constant 1 − βt, the posterior is given by:

αs(1 − βt|s)x + (1 − βt|s)(βs − αs)Q(1)x + (1 − βs)m 1 − βt

q(zs|zt,x) = Cat zs;

. (23)

Parameterization Inspired by the masked diffusion literature (Sahoo et al., 2024; Shi et al., 2024; Ou et al., 2024), we derive simplified loss form through the parameterizing a neural network θ to estimate only the probability in clean final state in S0. This leads to the posterior pθ(zs|zt) in following closed forms depending on the current state zt.

- Case 1. zt ∈ S0 pθ(zs|zt) = Cat(zs;zt). (24)
- Case 2. zt ∈ S1

pθ(zs|zt) = Cat zs;

(αsβt|s − αt)Q(1)⊺zt ⊙ xθ(zt,t) + (βt − βt|sαs)zt ⊙ Q(1)xθ (βt − αt)zTt Q(1)xθ

. (25)

- Case 3. zt ∈ S2 = {m}

pθ(zs|zt)

[αt|sm + (βt|s − αt|s)Q(1)⊺m + (1 − βt|s)Q(2)⊺m] ⊙ [αsxθ + (βs − αs)Q(1)xθ + (1 − βs)m] 1 − βt

= Cat zs;

αs(1 − βt|s)xθ + (1 − βt|s)(βs − αs)Q(1)xθ + (1 − βs)m 1 − βt

= Cat zs;

.

.

(26)

ELBO analysis Now, we start with analyzing the ELBO of the Hierarchical Discrete diffusion models in discrete timesteps.

t|x) [T · DKL (q(zs|zt,x)∥pθ(zs|zt))] (27)

LT = Et∈{ 1

T ,T2 ,...,1}Eq(z

- Case 1. zt ∈ S0 DKL (q(zs|zt,x)∥pθ(zs|zt)) = 0. (28)

- Case 2. zt ∈ S1 DKL (q(zs|zt,x)∥pθ(zs|zt))

=

zs∈{x,φ(x)}

q(zs|zt,x)log

q(zs|zt,x) pθ(zs|zt)

= q(zs = x|zt,x)log

q(zs = x|zt,x) pθ(zs = x|zt)

+ q(zs = φ(x)|zt,x)log

q(zs = Q(1)x|zt,x) pθ(zs = Q(1)x|zt)

=

(αsβt|s − αt) βt − αt

log

z⊺t Q(1)xθ ⟨x,Q(1)⊺zt⟩ · ⟨xθ,x⟩

+ 0.

(29)

- Case 3. zt ∈ S2 = {m} DKL (q(zs|zt,x)∥pθ(zs|zt))

q(zs|zt,x) pθ(zs|zt)

q(zs|zt,x)log

=

zs∈{x,φ(x),m}

q(zs = Q(1)x|zt = m,x) pθ(zs|zt)

q(zs = x|zt = m,x) pθ(zs|zt)

+ q(zs = Q(1)x|zt = m,x)log

= q(zs = x|zt = m,x)log

q(zs = m|zt = m,x) pθ(zs|zt)

+ q(zs = m|zt = m,x)log

(αs − αsβt|s) 1 − βt

(1 − βt|s)(βs − αs) 1 − βt

1 ⟨xθ,x⟩

DKL Q(1)x∥Q(1)xθ(zt,t) + 0.

=

log

+

Combined together, each term in Eq. 27 can be expressed as follows:

(30)

DKL (q(zs|zt,x)∥pθ(zs|zt))

(αsβt|s − αt) βt − αt

log⟨zt,Q(1)xθ⟩ − log ⟨xθ,x⟩ ⟨zt,Q(1)x⟩

=

(1 − βt|s)(βs − αs) 1 − βt

(αs − αsβt|s) 1 − βt

log⟨Q(1)x,Q(1)xθ(zt,t)⟩ +

−

log ⟨xθ,x⟩ ⟨zt,m⟩ + Cs,t,

for some constant Cs,t, which leads into following continuous time NELBO of HDDM:

(31)

L∞NELBO(θ)

T

DKL q(zt(i−1)|zt(i),x)∥pθ(zt(i−1)|zt(i))

= lim

T→∞

i=2

T

(αt(i)βt(i)|t(i−1) − αt(i)) βt(i) − αt(i)

log⟨zt(i),Q(1)xθ⟩ − log ⟨xθ,x⟩ ⟨zt(i),Q(1)x⟩

= lim

T→∞

i=2

(1 − βt(i)|t(i−1))(βt(i) − αt(i−1)) 1 − βt(i)

(αt(i−1) − αt(i−1)βt(i)|t(i−1)) 1 − βt(i)

log⟨Q(1)x,Q(1)xθ(zt(i),t(i))⟩ +

−

log ⟨xθ,x⟩ ⟨zt(i),m⟩

+ Ct(i−1),t(i)

1

αt(βt′ − αt′) βt − αt

log⟨zt,Q(1)xθ⟩ − log ⟨xθ,x⟩ ⟨zt,Q(1)x⟩

=

t(1)

βt′(βt − αt) βt(1 − βt)

αtβt′ βt(1 − βt)

log⟨Q(1)x,Q(1)xθ(zt,t)⟩ +

log ⟨xθ,x⟩ ⟨zt,m⟩ + Ct dt.

−

(32) As a result, we obtain the general NELBO of HDDM in the following Theorem:

General NELBO in HDDM

Theorem C.2. Suppose forward process of two-level HDDM Q is defined with stochastic operator Q(1),Q(2), which are induced from the Φ and masking operator, respectively as in Lemma 3.1. Then, for some constant C1, the negative evidence lower bound (NELBO) can be expressed as follows:

αt(βt′ − αt′) βt − αt

L∞NELBO(θ) = EQ,t

log⟨zt,Q(1)xθ⟩ − log ⟨xθ,x⟩ ⟨zt,Q(1)x⟩

(33)

βt′(βt − αt) βt(1 − βt)

αtβt′ βt(1 − βt)

log⟨Q(1)x,Q(1)xθ(zt,t)⟩ +

−

log ⟨xθ,x⟩ ⟨zt,m⟩ + C1.

Note that above theorem holds for arbitrary stochastic row matrix Φ, which means we can design any stochastic mapping from S0 to S1.

When Q(1) is deterministic (i.e, its rows are composed of one-hot vectors), we can parameterize model to estimate the categories that are in the same mid-level state. This parameterization leads to further simplified form of Theorem C.2 as follows:

NELBO in HDDM with deterministic grouping Corollary C.3.

αt(βt′ − αt′) βt − αt

L∞NELBO(θ) = EQ,t

log ⟨xθ(zt,t),x⟩ · I[zt ∈ S1]

(34)

βt′ βt(1 − βt)

(βt − αt)log⟨Q(1)x,Q(1)xθ(zt,t)⟩ + αt log ⟨xθ(zt,t),x⟩ I[zt = m] + C2,

−

for some constant C2.

### D. Experiment details

#### D.1. Decoupled Atom Encoding (DAE)

While standard graph-based diffusion models typically adopt a coarse node encoding based solely on atomic numbers (Z), decoupled atom encoding (DAE) expands the original token vocabulary by explicitly decoupling three standards: aromaticity, hydrogen saturation, and formal charge magnitude. Decoupled Atom Encoding (DAE) expands the token vocabulary by explicitly decoupling three critical chemical descriptors: aromaticity, hydrogen saturation, and formal charge magnitude. Unlike previous methods that rely on implicit hydrogen estimation (e.g., via RDKit’s valence rules), DAE treats these attributes as primary node features to be explicitly encoded and decoded. This approach resolves the one-to-many mapping problem between atomic tokens and their physical states, enabling the near-perfect reconstruction of drug-like scaffolds from the MOSES and GuacaMol datasets. Furthermore, this extended vocabulary facilitates the reliable generation of complex heteroaromatics and zwitterionic species which are extremely rare for baselines using coarse tokenization. Specifically, we emphasize that tokenizing [nH] as a distinct state is fundamentally different from modeling explicit hydrogen atoms as separate nodes. While the latter can significantly increases graph complexity and computational overhead, DAE preserves graph sparsity while maintaining chemical precision.

- Table 6. Comparison of Atom Vocabularies on the MOSES Dataset. DAE resolves structural ambiguities by decoupling elements into specific aromatic and hydrogen-locked states.

Method Elemental Basis Unique Tokens (Vocabulary) Size Standard Encoding {C, N, S, O, F, Cl, Br} C, N, S, O, F, Cl, Br 7 DAE (Ours) {C, N, S, O, F, Cl, Br} Aliphatic: C, N, S, O, F, Cl, Br 12

###### Aromatic: c, n, nH, s, o

Coarse

DAE (Ours)

| |
|---|

120

Relax

Identity Changed

100.0% 100.0% 100.0% 100.0%

100.0%

96.5%

100

80.4%

###### SuccessRate(%)

80

60

40

20

2.0%

1.9%

0

[nH] Group Charged Group Standard Group

- Figure 5. The ratios of generated molecules having formal charge. MolHIT can reach to the training level proportion, while models with previous coarse encoding (left two) barely generate the charged atoms.

DAE in MOSES The MOSES dataset consists primarily of stable, neutral drug-like molecules which is clean lead filtered from the ZINC dataset (Irwin and Shoichet, 2005). In this context, the reconstruction bottleneck is primarily structural. Previous coarse-grained encodings fail to resolve the placement of pyrrolic hydrogens ([nH]), a critical motif in heteroaromatic rings like indole or imidazole. By explicitly decoupling aromaticity and hydrogen counts, DAE enables model can explicitly distinguish these motifs, resulting in improved generation quality.

- Table 7. Vocabulary expansion for the Guacamol dataset. DAE scales from 12 elemental types to 56 semantic tokens including aromatic and charged atoms.

Category Standard Encoding (Size: 12) DAE Tokens (Size: 56) Neutral Aliphatic {C, N, O, F, B, Br, Cl, I, P, S, Se, Si} C, N, O, F, B, Br, Cl, I, P, S, Se, Si Aromatic States (None / Implicit) c, c+, c-, n, nH, n+, nH+, n-, s, s+, o, o+, se, se+, p Charged & (None / Implicit) C+, C-, N+, NH+, NH2+, NH3+, N-, NH-, O+, O-, F+, F-, Hypervalent B-, Br+2, Br-, Cl+, Cl+2, Cl+3, Cl-, I+, I+2, I+3,

P+, P-, S+, S-, Se+, Se-, Si-

DAE in GuacaMol GuacaMol (Brown et al., 2019) is constructed from a standardized subset of ChEMBL (Mendez et al., 2019), restricted to common medicinal-chemistry elements. In this unconstrained space, previous models suffer from a fundamental reconstruction failure; for instance, standard coarse-grained encoding achieves only a 1.88% success rate on the [nH] group, with a negligible 0.09% identity preservation rate. While previous models implicitly rely on the relaxation technique which can improve the success rates (e.g., increasing charged group success from 80.43% to 96.54%), this it only preserves 80.07% of total molecules, indicating a failure to maintain the original chemical identity. As illustrated in Figure 5, MolHIT addresses this through Decoupled Atom Encoding (DAE), which expands the vocabulary to 56 tokens by encode-decode the atoms with extended vocabulary space, resulting in 100 % success rate and over 99.98% in identity preservation rate. Moreover, as illustrated in Fig. 6, the effect of DAE also happens in generative performance, where it enables generating molecules with formal charge which consist of about 6% in GuacaMol dataset.

#### D.2. Grouping in HDDM

Grouping Details for MOSES and GuacaMol We employ dataset-specific grouping strategies to align the intermediate state space S1 with the underlying chemical distribution of each corpus. Table 8 summarizes these partitions.

0.08

0.0743

Fractionofmolecules

0.0596

0.06

0.0382

0.04

0.02

0.0001 0.0005

0.00

DiGress (full) DiGress (org.) DiGress + DAE GuacaMol train MolHIT

- Figure 6. Reconstruction Fidelity and Identity Preservation. We measure the proportion of generated molecules that have at least one atom with formal charge.

Table 8. Deterministic grouping kernels for node state space partitioning in MOSES and GuacaMol. Dataset Group ID Atom Elements (S0)

- Group 1 {C}
- Group 2 {N, O, S}
- Group 3 {F, Cl, Br}
- Group 4 {c, o, n, [nH], s}

MOSES

- Group 1 {F, Cl, Br, I, F−, Cl−, Br−}
- Group 2 {C, N, O, P, S, Se}
- Group 3 {c, n, [nH], o, s, se, p}
- Group 4 {N+, n+, [nH]+, P+, [NH]+, [NH2]+, [NH3]+, Br+2, Cl+2, Cl+3, I+2, I+3}
- Group 5 {O−, N−, [NH]−, O+, S+, B−, C+, C−, c+, c−, n−, s+, o+, se+, F+, Cl+, I+, P−, S−, Se+, Se−, Si−}
- Group 6 {B, Si}

GuacaMol

- Table 9. Unconditional generation on MOSES dataset with full statistics. We bring the reported value from Cometh and DeFoG from their work.

Category Model Quality ↑ Scaf-Novel ↑ Scaf-Ret. ↑ Valid ↑ Unique ↑ Novel ↑ Filters ↑ FCD ↓ SNN ↑ Scaf ↑

- Training set 95.4 — — 100.0 100.0 — 100.0 0.48 0.59 -

- 1D Sequence VAE[25] 92.8 ± 0.2 0.22 ± 0.01 0.031 ± 0.003 97.7 ± 0.1 99.7 ± 0.0 69.5 ± 0.6 99.7 ± 0.0 0.57 ± 0.00 0.58 ± 0.01 5.9 ±1.0 CharRNN [42] 92.6 ± 2.5 0.29 ± 0.04 0.035 ± 0.003 97.5 ± 2.6 99.9 ± 0.0 84.2 ± 5.1 99.4 ± 0.3 0.52 ± 0.03 0.56 ± 0.01 11.0 ± 0.8 SAFE-GPT [33] 92.8 ± 0.0 0.12 ± 0.00 0.015 ± 0.000 99.8 ± 0.0 98.9 ± 0.0 43.7 ± 0.3 97.7 ± 0.0 0.72 ± 0.02 0.57 ± 0.01 6.3 ± 0.7 GenMol [27] 62.1 ± 0.0 0.05 ± 0.00 0.012 ± 0.001 99.7 ± 0.1 64.0 ± 0.5 68.9 ± 0.4 98.1 ± 0.1 16.36 ± 0.07 0.64 ± 0.01 1.6 ± 0.1

- 2D Graph DiGress [47] 82.5 ± 0.7 0.26 ± 0.00 0.031 ± 0.000 87.1 ± 0.9 100.0 ± 0.0 94.2 ± 0.2 97.5 ± 0.0 1.25 ± 0.03 0.53 ± 0.00 12.8 ± 1.4 DisCo [51] - - - 88.3 100.0 97.7 95.6 1.44 0.50 15.1 Cometh [45] 82.1 ± 0.1 0.36 ± 0.00 0.023 ± 0.000 87.2 ± 0.0 100.0 ± 0.0 96.4 ± 0.1 97.3 ± 0.0 1.44 ± 0.02 0.51 ± 0.00 16.8 ± 0.7 DeFoG [38] 88.5 ± 0.0 0.26 ± 0.00 0.031 ± 0.000 92.8 ± 0.0 99.9 ± 0.0 92.1 ± 0.0 98.9 ± 0.0 1.95 ± 0.00 0.55 ± 0.00 14.4 ± 0.0 MolHIT 94.2 ± 0.2 0.39 ± 0.00 0.033 ± 0.001 99.1 ± 0.0 99.8 ± 0.0 91.4 ± 0.2 98.0 ± 0.00 1.03 ± 0.02 0.55 ± 0.00 14.4 ± 1.0

#### D.3. Full experimental results with standard deviations

For statistical significance, we run 3 experiments for every experiment. We put the result including standard deviation of unconditional MOSES generation in Table 9, GuacaMol experiment in Table 10, multi-property guided generation result in Table 11, and scaffold extension result in Table 12.

- Table 10. Full statistics of GuacaMol benchmark results (unfiltered). Val.: Validity, V.U.: Unique, V.U.N.: Novel. All results are averaged over 3 runs.

Model Val. ↑ V.U. ↑ V.U.N. ↑ KL ↑ FCD ↓ Training set 100.0 100.0 — 99.9 92.8 DiGress (org.) 85.2 85.2 85.1 92.9 68.0 DiGress (full) 74.7 ± 0.4 74.6 ± 0.5 74.0 ± 0.4 92.4 ± 0.5 61.1 ± 0.2 DiGress+DAE 65.2 ± 0.4 65.2 ± 0.4 64.9 ± 0.4 87.0 ± 0.4 49.2 ± 0.6

MolHIT (Ours) 87.1 ± 0.5 87.1 ± 0.3 86.0 ± 0.5 96.7 ± 0.1 54.9 ± 0.2

- Table 11. Full statistics of multi-property guided generation on MOSES with 4 different conditions. We report mean absolute error (MAE; ↓), Pearson correlation (r; ↑), and validity. Avg. is the macro-average across properties. Bold denotes best values.

MAE ↓ Pearson r ↑ Validity (%) ↑ Method QED SA LogP MW Avg. QED SA LogP MW Avg.

Marginal 0.117(±0.002) 0.115(±0.003) 0.067(±0.001) 0.272(±0.009) 0.143(±0.004) 0.489(±0.003) 0.570(±0.012) 0.802(±0.003) 0.396(±0.001) 0.564(±0.005) 75.03(±0.74) Marginal + DAE 0.107(±0.001) 0.094(±0.001) 0.061(±0.000) 0.227(±0.004) 0.122(±0.001) 0.565(±0.005) 0.559(±0.009) 0.836(±0.005) 0.437(±0.015) 0.599(±0.002) 87.85(±0.46) MolHIT 0.061(±0.001) 0.040(±0.001) 0.049(±0.001) 0.081(±0.005) 0.058(±0.002) 0.804(±0.009) 0.790(±0.011) 0.950(±0.004) 0.685(±0.024) 0.807(±0.011) 96.31(±0.23)

- Table 12. Full statistics of scaffold extension results on MOSES. Results are averaged over 3 runs of 10,000 targets. Hit@k denotes the recovery of ground-truth within k samples.

Model Valid (%) ↑ Diversity ↑ Hit@1 ↑ Hit@5 ↑

DiGress 50.8 ± 0.5 44.8 ± 1.8 2.07 ± 0.09 6.41 ± 0.21 Marginal + DAE 64.8 ± 0.2 58.0 ± 0.1 1.67 ± 0.10 6.37 ± 0.24

MolHIT (Ours) 83.9 ± 0.4 57.4 ± 0.6 3.92 ± 0.23 9.79 ± 0.09

#### D.4. Implementation of baselines

For all baselines, we use released checkpoints when available. Otherwise, we train the models using their official codebase, following the training hyperparameters reported in the paper or provided in the codebase. Note that the original GenMol (Lee et al., 2025) model was trained on a much larger molecule dataset, so we train the model on the MOSES dataset for a fair comparison.

#### D.5. Unconditional generation with MOSES and GuacaMol

Training details For our model backbone, we adopt the graph transformer proposed by Vignac et al. (2022), which simultaneously predicts node and edge features. To ensure a fair comparison across all experimental settings, we maintain a consistent architecture of 12 transformer blocks without altering any internal dimensional configurations. The total trainable parameter count is approximately 16.2M. The introduction of additional token indices (DAE and HDDM) adds negligible overhead where representing a variance of less than 0.01% in total parameters. For training stability, we employ gradient clipping with a threshold of 2.0 and an Exponential Moving Average (EMA) rate of 0.999. We early stop with 100 epoch training with MOSES and 50 epochs with GuacaMol, compared to the original 300 epoch training of other graph diffusion baselines (Vignac et al., 2022; Siraudin et al., 2024; Qin et al., 2024). We also remove calculating geometric prior originally used in Vignac et al. (2022), where they use extra graph features as conditional information. In our experiments, this has negligible effects on the performance.

Evaluation of MOSES The following metrics are utilized to evaluate the generative performance on the MOSES dataset, following the standardized protocols established by Polykovskiy et al. (2020).

- • Validity (↑): The fraction of generated molecules that pass RDKit’s sanitization checks and basic chemical valency rules. High validity is a primary indicator that the DAE system successfully constrains the sampling process to the chemically feasible manifold.
- • Uniqueness (↑): The proportion of valid molecules that are not duplicates. This measures the model’s ability to avoid mode collapse and explore a diverse structural space.
- • Novelty (↑): The fraction of valid, unique molecules that were not present in the training set. This differentiates between a model that has memorized the data and one that has learned the underlying generative distribution.

- • Filters (↑): The percentage of generated molecules that pass common medicinal chemistry filters (e.g., MCULE, BRENK, and PAINS). This evaluates the drug-likeness and synthetic viability of the generated samples.
- • Fr´echet ChemNet Distance (FCD (Preuer et al., 2018), ↓): A measure of the distance between the multivariate distributions of generated and test molecules in the feature space of ChemNet. FCD captures both chemical and biological similarity, serving as the most rigorous metric for distributional fidelity.
- • Similarity to Nearest Neighbor (SNN, ↑): The average Tanimoto similarity between a generated molecule and its closest neighbor in the test set. High SNN indicates that the model has captured the specific structural motifs and chemical space of the benchmark.
- • Scaffold Similarity (Scaf, ↑): The cosine similarity between the frequencies of Bemis–Murcko scaffolds (Bemis and Murcko, 1996) in the generated and test sets. This assesses whether the model’s learned distribution of backbone structure matches the architectural diversity of real-world leads.

Baselines

- D.6. Structure novelty metric

- • Scaffold Novelty: We evaluate the model’s ability to innovate beyond the training distribution using Bemis–Murcko scaffolds (Bemis and Murcko, 1996). The absolute number of unique generated scaffolds absent from the training set:

Scaf-Novel = |Sgen \ Strain|

ntotal

. (35)

This metric quantifies the model’s capacity for structural extrapolation, measuring its efficiency in exploring the beyond the molecular structure of the given dataset.

- • Scaffold Retrieval: This assesses the model’s ability to rediscover known, high-quality frameworks from the held-out test set. This is defined as the absolute number of unique test-set scaffolds successfully generated: from the training set:

Scaf-Ret = |Sgen ∩ Stest|

ntotal

. (36)

Scaffold retrieval serves as a rigorous test of distributional accuracy. A high retrieval density demonstrates that the model has not merely learned to generate novel-looking noise, but has accurately captured the underlying manifold of valid, drug-like molecules defined by the test distribution.

- D.7. Unconditional generation with GuacaMol

GuacaMol experiment For experiment on GuacaMol, we test our algorithm on the unfiletered, full dataset. Previous graph diffusion model baselines (Vignac et al., 2022; Siraudin et al., 2024; Qin et al., 2024) train the model on the filtered dataset, where they filter out the molecules that are failed to be reconstructed back. This can bias the training data distribution. In contrast, we use full, unfiltered dataset for the experiment and since there is no graph diffusion baseline, we compare with the original DiGress trained on a full GuacaMol dataset with coarse atom encoding, Discrete Diffusion using marginal transition with DAE, and compare them with MolHIT.

- D.8. Multi-property guided generation

Data construction For conditional generation, we augment the MOSES dataset (Polykovskiy et al., 2020) with four continuous molecular descriptors: Quantitative Estimate of Drug-likeness (QED), Synthetic Accessibility (SA) score, Octanol-Water Partition Coefficient (logP), and Molecular Weight (MW).

- • Quantitative Estimate of Drug-likeness (QED, ↑): A widely used composite score that summarizes multiple molecular properties (e.g., lipophilicity, polarity, and molecular size) into a single measure of drug-likeness; higher values indicate more drug-like compounds.

- • Synthetic Accessibility (SA, ↓): An empirical estimate of synthetic difficulty that combines fragment-based contributions with a complexity penalty; lower values indicate molecules that are easier to synthesize.
- • Octanol–Water Partition Coefficient (logP): A measure of lipophilicity that is informative of solubility and membrane permeability; excessively high logP is typically associated with poor solubility and unfavorable ADMET profiles.
- • Molecular Weight (MW): The molecular mass in Daltons. Consistency with the training distribution (e.g., MOSES) helps ensure generated molecules remain within a drug-like regime.

All properties are calculated using the RDKit library and the sascorer module (Ertl and Schuffenhauer, 2009). To ensure stable convergence of the conditioning vector C within our AdaLayerNorm layers, we perform min-max normalization on these values using the global statistics of the training split, which are in Table 13.

Table 13. Min and max values for molecular property conditioning in MOSES training / test split. Split Statistic QED SA Score logP MW (Da) Training

Min 0.1912 1.2694 -5.3940 250.017 Max 0.9484 7.4831 5.5533 349.999

Min 0.2265 1.3339 -4.2894 250.042 Max 0.9484 6.6916 5.7255 349.990

Test

Conditional graph transformer While we maintain the core node-edge attention mechanism of the original graph transformer (Vignac et al., 2022), we introduce several key modifications to enable conditional modeling. First, we remove the persistent global feature vector y—which in the original framework is updated at every layer—and replace it with a centralized conditioning vector C. This vector is composed of a sinusoidal timestep embedding (Ho et al., 2020) and an optional MLP-encoded external property condition c. Second, to integrate C into the denoising process, we replace standard Layer Normalization with Adaptive Layer Normalization (AdaLayerNorm) for node features. Specifically, for a node embedding x, the normalization is defined as:

AdaLN(x,C) = (1 + γ(C)) · LayerNorm(x) + β(C)

where γ and β are affine transformations of the conditioning vector. This allows the global context (time and properties) to directly modulate the scale and shift of node representations. Finally, we implement Classifier-Free Guidance (CFG) support by incorporating a dropout mechanism on the property embedding during training, while ensuring the temporal signal remains persistent to maintain denoising stability. Our conditional graph transformer naturally inherits permutation equivariance, which is different from the Liu et al. (2024).

Evaluation details For sampling, we employ Classifier-Free Guidance (CFG) (Ho and Salimans, 2022) with a guidance scale of w = 1.0. We observe that in our discrete graph-diffusion framework, increasing the guidance weight beyond unity did not consistently yield better property alignment. We leave the better design the sampler or models to be effective in higher guidance strength w as a promising avenue for future research.

#### D.9. Scaffold extension

Task Formulation Given a ground-truth molecule G from the test split, we use RDKit to extract its scaffold S ⊂ G. The task is to generate a completed molecule M such that S ⊆ M. To isolate the generative capability from size prediction errors, we bound the generation size (number of atoms) to match |G|.

Sampling Protocol At each reverse timestep t, the region corresponding to S is forced to be the same (i.e, q(xt|xscaffold) = xscaffold, ensuring the scaffold region is strictly fixed during the generation. The extension region is initialized from the limit distribution (i.e, prior) pprior and evolved via the standard reverse process. We generate K = 1,5 independent samples per scaffold to capture the model’s exploration capability.

Metric Definitions Metrics are computed per scaffold and then averaged across the test set. Let M1,...,MK be the generated graphs for a single scaffold.

- • Validity: The fraction of Mi that are chemically valid according to RDKit sanitization.
- • Diversity: Calculated on the unique valid set U. We define diversity as 1 − |U|12 u,v∈U Sim(u,v), where Sim is the Tanimoto similarity using Morgan fingerprints (r = 2, 2048 bits).

- • Exact Match (Hit@K): A binary indicator, set to 1 if the ground truth G (canonical SMILES) is present in the generated set {M1,...,MK}, and 0 otherwise.

