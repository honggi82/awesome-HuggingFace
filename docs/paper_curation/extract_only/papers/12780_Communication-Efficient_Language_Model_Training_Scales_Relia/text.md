# arXiv:2503.09799v1[cs.LG]12Mar2025

## Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo

Zachary Charles1∗ Gabriel Teston2 Lucio Dery3 Keith Rush1

Nova Fallen1 Zachary Garrett1 Arthur Szlam3 Arthur Douillard3 1Google Research 2Google Search 3Google DeepMind

Abstract

As we scale to more massive machine learning models, the frequent synchronization demands inherent in data-parallel approaches create significant slowdowns, posing a critical challenge to further scaling. Recent work [Douillard et al., 2023, 2025] develops an approach (DiLoCo) that relaxes synchronization demands without compromising model quality. However, these works do not carefully analyze how DiLoCo’s behavior changes with model size. In this work, we study the scaling law behavior of DiLoCo when training LLMs under a fixed compute budget. We focus on how algorithmic factors, including number of model replicas, hyperparameters, and token budget affect training in ways that can be accurately predicted via scaling laws. We find that DiLoCo scales both predictably and robustly with model size. When well-tuned, DiLoCo scales better than data-parallel training with model size, and can outperform data-parallel training even at small model sizes. Our results showcase a more general set of benefits of DiLoCo than previously documented, including increased optimal batch sizes, improved downstream generalization with scale, and improved evaluation loss for a fixed token budget.

Harder: DiLoCo’s hyperparameters are robust and predictable across model scales.

Better: DiLoCo further improves over data-parallel training as model size increases.

Faster: DiLoCo uses orders of magnitude less bandwidth than data-parallel training.

Stronger: DiLoCo tolerates a significantly larger batch size than data-parallel training.

### 1 Introduction

The default approach to training large language models (LLMs) continues to be large-batch distributed data-parallel training. However, bandwidth and communication constraints, which can be negligible at smaller scales, become dominant factors at larger scales. The frequent synchronization

∗Correspondence to zachcharles@google.com.

demands inherent in classical distributed approaches create significant slowdowns, posing a critical challenge to further scaling. As a remedy, Douillard et al. [2023] propose DiLoCo (Distributed LowCommunication), a generalization of algorithms like Local SGD [Mangasarian and Solodov, 1993, Stich, 2018] and FedAvg [McMahan et al., 2017], which enables training of LLMs in parallel across “islands” of compute (such as datacenters connected via low-bandwidth networks) by performing parallel training of models with only periodic synchronization.

Unlike communication-reduction methods like quantization and sparsification, DiLoCo fundamentally alters training dynamics [Rush et al.]. While Douillard et al. [2023], Jaghouar et al. [2024b] show that DiLoCo yields comparable or better downstream evaluation metrics to data-parallel training at moderate model scales (up to 1.1 billion parameters), it is unclear how data-parallel training and DiLoCo compare at larger model scales. Moreover, DiLoCo has extra hyperparameters not present in data-parallel training that may be computationally prohibitive to tune at large enough scales.

This points to the need for DiLoCo scaling laws. Scaling laws generally give data-backed mechanisms for predicting facets of a model trained via a chosen algorithm (such as the evaluation loss after training on a given number of tokens) [Kaplan et al., 2020, Hoffmann et al., 2022]. We focus on two specific scaling laws: (1) predictions for evaluation loss as a function of model size and (2) predictions for optimal hyperparameter choices for a given model size (which can obviate the need to perform expensive hyperparameter tuning). In both cases, we are explicitly interested in how these compare to analogous scaling laws for data-parallel training.

Setting. Throughout, we focus on the task of pre-training “from scratch” a model of size N on some total number of tokens D. We are primarily concerned with predicting, for both data-parallel training and DiLoCo training, and as a function of N: (1) the evaluation loss L after training, computed on a held-out set, and (2) optimal hyperparameter settings.

One path towards scaling laws for DiLoCo would be to develop them as a modification or functional transformation of scaling laws for data-parallel training. However, the facets of DiLoCo that are key to its communication-efficiency also make such a direct adaptation infeasible. First, DiLoCo operates by training M models in parallel, with periodic synchronization every H steps. The values of M and H depend on the ecosystem of compute available (such as the number of datacenters we are training a model over and the network bandwidth between them), and are absent in scaling laws for data-parallel training. Second, DiLoCo uses a bi-level optimization framework; each model replica performs data-parallel training, but upon synchronization we apply an “outer” optimization step [Douillard et al., 2023]. This means that DiLoCo has “outer” hyperparameters not present in Data-Parallel training that cannot be inferred from data-parallel hyperparameter scaling laws.

Instead, we develop scaling laws for data-parallel training and DiLoCo from the ground up, using similar methodology to work of Kaplan et al. [2020], Hoffmann et al. [2022]. Fixing the number of tokens D to be the “Chinchilla-optimal” number of tokens [Hoffmann et al., 2022], we model the evaluation loss and optimal hyperparameters of data-parallel training as functions of model size N, and we model the loss and optimal hyperparameters of DiLoCo as functions of model size N and number of replicas M.1 We empirically estimate these functions using the final evaluation loss

1While we fix H = 30 for these scaling laws, we also provide extensive ablations on the role of H in Section 5.1. This value of H is large enough to imply a very substantial communication reduction, but (as our experiments show), DiLoCo with this setting of H is still highly competitive with data-parallel training, sometimes even outperforming it.

attained by models trained with both algorithms for varying hyperparameters (including learning rate, batch size, and “outer” learning rate for DiLoCo), model sizes (varying N over 9 model sizes ranging from 35 million to 2.4 billion parameters), and numbers of DiLoCo replicas M.

Contributions. Our core contribution are scaling laws for evaluation loss and optimal hyperparameters for both data-parallel training and DiLoCo based on the data described above. We show that these scaling laws provide good estimates of loss and optimal hyperparameters when extrapolating to larger model sizes. To our surprise, these scaling laws predict that in many settings, the more communication-efficient DiLoCo algorithm would actually yield better evaluation loss than data-parallel training for the same token budget. Utilizing our scaling laws to predict the hyperparameters for DiLoCo, we tested these predictions when training models with 4 billion and 10 billion parameters. The scaling laws proved accurate, with DiLoCo outperforming data-parallel training as predicted, even while reducing total communication by a factor of over 100.

Our large amount of experimental data also enables us to analyze these algorithms at a deeper level, including an examination of their system characteristics. For each experiment, we provide an idealized end-to-end wall-clock training time under networks of varying bandwidth and latency. We show that DiLoCo incurs a variety of benefits in comparison to data-parallel training, including (1) increased optimal batch size, allowing for greater horizontal scalability, (2) greater reductions in evaluation loss as model size increases, and (3) significantly less wall-clock training time.

One potentially surprising artifact of this data: DiLoCo improves training even when communication is not a bottleneck. Our experiments include DiLoCo with M = 1. This algorithm, effectively an enhanced version of the Lookahead optimizer [Zhang et al., 2019], does not incur any communication reduction. However, it actually does better than data-parallel training across model sizes in terms of both evaluation loss and tolerance for larger batch sizes, via its use of an infrequent momentum operation. Notably, DiLoCo, M = 1 outperforms data-parallel training on both evaluation loss and training time. We show that DiLoCo with M = 1 achieves lower evaluation loss at all model scales, and is more robust to larger batch sizes, greatly reducing wall-clock training time.

### 2 Preliminaries

Throughout, we let θ denote the model parameters. We let θ(t) denote the model at step t. Since DiLoCo operates on M parallel model parameters, we will use subscript notation θm to denote the m-th model. When there is no subscript, the parameters are assumed to be replicated across all DiLoCo replicas. For a batch of data x, we let f(θ,x) denote the loss of θ on the batch of data.

#### 2.1 DiLoCo: Distributed Low-Communication

DiLoCo [Douillard et al., 2023] is a technique designed for training models in the presence of communication constraints. It is especially motivated by the training of large models across devices that are not all connected by low-latency bandwidth. To avoid incurring latency costs, DiLoCo trains M models in parallel (ideally, training each one with co-located compute connected via low latency bandwidth), only synchronizing the models every H steps. This is similar to the FedOpt algorithm used in federated learning [Reddi et al., 2021], but with the important difference that the replicas maintain their inner optimizer state across rounds.

Table 1: General Notation

Table 2: Algorithm-Specific Notation

Symbol Meaning

Symbol Data-Parallel DiLoCo

θ Model weights N Model size L Evaluation loss T Training steps D Token budget C Total FLOPs

γ Learning rate Inner learning rate η – Outer learning rate B Batch size Global batch size M – DiLoCo replicas H – Synchronization cadence

[Figure 1]

[Figure 2]

Parallel Inner Optimization

Parallel Inner Optimization

…

…

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Outer Optimization

Model …

Outer

Initial Optimization

…

…

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

##### … …

… …

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Figure 1: DiLoCo. Each DiLoCo model replica trains independently for H inner optimization steps. These models are synchronized via an outer optimization step, usually involving momentum across outer optimization steps. In this figure, there are M = 4 replicas.

In more detail, DiLoCo applies a bi-level optimization framework across multiple models: each DiLoCo replica has its own model θm(t), and there is a global model θ(t). At every step, each replica takes an inner optimization step (InnerOpt). Every H steps, each replica computes the ∆(mt) = θ(t−H) − θm(t), the difference in parameter space between the replica’s current model and the most recently updated global model. We average these deltas across replicas, resulting in a vector ∆(t) which we refer to as an outer gradient. We treat this as a gradient estimate of the outer model2 and an outer optimization step (OuterOpt) to the outer model θ(t−H). This yields an updated outer model θ(t) which is broadcast to all replicas and set as their current inner model. We give full pseudo-code for DiLoCo in Algorithm 1.

#### 2.2 Data-Parallel versus DiLoCo

Throughout our work, we will perform model training via distributed data-parallel training, which we refer to as Data-Parallel for brevity, and DiLoCo. We discuss how they differ, as well as operational details we will use throughout. In Data-Parallel, at each step we take some batch of data of size B. In our work, like that of Kaplan et al. [2020], Hoffmann et al. [2022], the batch size will refer to the number of tokens in a batch (as opposed to the number of sequences). We then compute a batch

2While empirically useful, it is worth noting that this operation is not completely grounded theoretically, as ∆(t) is generally not a gradient of any function [Charles and Rush, 2022].

Algorithm 1 DiLoCo Require: Loss function f(θ,x), batch size B, number of replicas M, synchronization cadence H. Require: Initial model weights θ(0), data shards {D1,...,DM} Require: Optimizers InnerOpt and OuterOpt

- 1: ∀m, θm(0) ← θ(0)
- 2: for step t = 1...T do
- 3: parallel for replica m = 1...M do
- 4: Receive a batch x(mt) ∼ Dm of size B/M
- 5: gm(t) ← ∇θf(θm(t−1),x(mt))
- 6: θm(t) ← InnerOpt(θm(t−1),gm(t))
- 7: end parallel for
- 8: if t mod H = 0 then
- 9: ∆(mt) ← θ(t−H) − θm(t)
- 10: ∆(t) ← M1 Mm=1 ∆(mt)

- 11: θ(t) ← OuterOpt(θm(t−H),∆(t))
- 12: ∀m,θm(t) ← θ(t)

gradient and apply an optimization with a learning rate of γ.

In DiLoCo, at each step t, we take a global batch of data of size B, and evenly partition it at the sequence level across the M DiLoCo replicas. Thus, the global batch size is B, but each of the DiLoCo replicas uses a local batch of size B/M. Similarly to Data-Parallel, each replica then computes a batch gradient and applies an inner optimization step with a learning rate of γ. Unlike Data-Parallel, DiLoCo does outer optimization (on outer-gradients computed in parameter space) every H steps with learning rate η.

An important comparison is Data-Parallel versus DiLoCo with M = 1. While similar, they are not identical. DiLoCo with M = 1 still has an outer optimizer step using OuterOpt, and is thus a variant of the Lookahead optimizer [Zhang et al., 2019]. In our notation, [Zhang et al., 2019] corresponds to the setting when OuterOpt is SGD with learning rate η. In DiLoCo OuterOpt is often set to SGD with Nesterov momentum [Douillard et al., 2023] in which case DiLoCo with M = 1 becomes a variant of Data-Parallel with a momentum operation only applied every H steps.3

When comparing between Data-Parallel and DiLoCo, we ensure that the model size N and total token budget D is the same. To compute an evaluation loss L on some held-out set, for Data-Parallel we use the current model, and for DiLoCo we use the most recent global model. We summarize the algorithm-independent notation in Table 1 and the algorithm-specific notation in Table 2.

Table 3: Model details. We present the size, number of layers, layer dimensions, and token budgets for each model trained. For the larger models (4B and 10B) we use scaling laws to predict optimal hyperparameters, rather than performing extensive hyperparameter tuning.

Model Transformer Attention QKV Hidden Token Hyperparameter Scale Layers Heads Dimension Dimension Budget Sweep

35M 6 8 512 2,048 70M ✓ 90M 9 12 768 3,072 1.8B ✓

180M 12 16 1,024 4,096 3.6B ✓ 330M 15 20 1,280 5,120 6.6B ✓ 550M 18 24 1,536 6,144 11B ✓

- 1.3B 24 32 2,048 8,192 26B ✓

- 2.4B 30 40 2,560 10,240 48B ✓ 4B 36 48 3,072 12,288 80B ✗

10B 48 64 4,096 16,384 200B ✗

### 3 Experimental Methodology

Model architecture. We use a Chinchilla-style decoder-only transformer architecture [Hoffmann et al., 2022]. As suggested by Wortsman et al. [2023] and Jaghouar et al. [2024b], we use QKLayerNorm to reduce sensitivity to learning rate. We also use z-loss regularization [Chowdhery et al.,

- 2023] to increase training stability. We use a vocabulary size of 32,768: 32,000 in-vocabulary words, and extra tokens for BOS and out-of-vocabulary (extended so that the vocabulary size is a power of 2, which is useful for sharding computations across accelerators). We pack multiple sequences into each batch, with a maximum sequence length of 2,048 throughout. We train all of our models from scratch, as we are primarily interested in scaling laws for pre-training regimes in this work.

We train on a family of models varying the number of transformer layers, number of attention heads, QKV dimension, and feed-forward layer hidden dimension. Details on the architecture for each model scale are given in Table 3. As we will discuss in Section 3.1, we use the Chinchilla token budget unless otherwise noted and do extensive hyperparameter sweeps on all models except the two largest (4B and 10B).

Datasets. In most experiments, we train our models using the train split of the C4 dataset [Raffel et al., 2020] throughout. We report evaluation metrics on C4’s held-out validation set. Additionally, we compute downstream zero-shot evaluation metrics on 3 tasks: HellaSwag [Zellers et al., 2019], Piqa [Bisk et al., 2020], and Arc-Easy [Clark et al., 2018]. When performing overtraining ablations, we use the Dolma dataset [Soldaini et al., 2024].

Algorithms and optimizers. We use AdamW [Loshchilov, 2017] as the optimizer for DataParallel and the inner optimizer for DiLoCo. For both algorithms, we set β1 = 0.9 and β2 = 0.99. We do 1000 steps of warmup followed by cosine learning rate decay. Following [Wang and Aitchison,

- 2024], we set the weight decay parameter λ to T−1 where T is the total number of training steps

3This is similar in spirit to the fast- and slow-momentum steps in AdEMAMix [Pagliardini et al., 2024], but yields different training dynamics since it uses a gradient estimate computed by linearizing across multiple training steps.

(which crucially depends on the batch size and token budget). We decay to 5% of the peak learning rate by the end of training. For training stability, we clip (inner) gradients to a global ℓ2 norm of

- 1. We do not clip outer gradients. For DiLoCo, we use SGD with Nesterov momentum [Sutskever et al., 2013] as the outer optimizer, as suggested by Douillard et al. [2023]. We use a momentum term of 0.9, and a constant outer learning rate. Unless otherwise specified, we set H = 30.

Implementation. We use a modified version of NanoDO [Liu et al., 2024c] that uses DrJAX [Rush et al.] to parallelize inner training steps across replicas and surface the model replica axis for explicit programming. This was crucial for better scaling performance in JAX [Bradbury et al., 2018], as DrJAX provides an enriched version of jax.vmap that provides more explicit sharding information about the DiLoCo replicas. The outer optimization is implemented using an all-reduce. Data-parallel training is implemented as a special case with a single model replica, and no outer optimization step. We use bfloat16 representation of model weights and gradients throughout. We perform most of our experiments on TPUv5e and TPUv6e, and for the largest scales on TPUv5.

Idealized wall-clock time. For each experiment, we also compute an idealized wall-clock time for training, that factors in both an idealized computation time and idealized communication time. We specifically measure the end-to-end wall-clock time (sometimes referred to as elapsed real time). Greater horizontal parallelization, such as via doubling the batch size, will therefore reduce wall-clock time. Our model assumes that we are training a model across multiple datacenters. Within a datacenter, we have a high-bandwidth network. Across datacenters, we either have a high-, medium-, or low-bandwidth network. For the idealized communication time, we always use the high-bandwidth network for the within-datacenter network, and one of the three for the cross-datacenter network. For details on the idealized wall-clock time, see Appendix A.

#### 3.1 Scaling Law Experiments

We perform comprehensive hyperparameter sweeps for Data-Parallel and DiLoCo on models ranging from 35M to 2.4B We sweep over the learning rate γ using integer powers of √2 and batch size B using powers of 2. For DiLoCo, we train using M = 1,2,4,8 and sweep the outer learning rate η over {0.2,0.4,0.6,0.8,1.0}. The initial grids for (inner) learning rate and batch size depend on the model size, and were extended as needed until the minimum loss value was obtained on an interior point in all hyperparameter grids.

Using this data, we derive scaling laws to predict evaluation loss and optimal hyperparameters for larger models. We use the predicted hyperparameters to train models with 4B and 10B parameters, in order to validate the scaling laws empirically. Following the Chinchilla scaling laws [Hoffmann et al., 2022] we assume that the optimal token budget is given by D = 20N. Unless otherwise indicated, in all experiments we train using this number of tokens. This means that for a fixed model size, if we double the batch size B, we halve the number of training steps.

### 4 Empirical Findings

Before discussing our process for fitting the specific scaling laws, we talk about the empirical results and four critical findings that are worth highlighting independently.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

3.6

- 2.6

2.8

3.0

- 3.2
- 3.4

Algorithm

EvalLoss

Data Parallel

- DiLoCo, M=1

- DiLoCo, M=2

DiLoCo, M=4 DiLoCo, M=8

2.4

225 226 227 228 229 230 231

Model Size

(a) Evaluation loss for various algorithms, as a function of N.

- 0
- 1
- 2
- 3
- 4

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

RelativeEvalLoss(%)

225 226 227 228 229 230 231

Model Size

Algorithm

Data Parallel

- DiLoCo, M=1

- DiLoCo, M=2

DiLoCo, M=4 DiLoCo, M=8

(b) Percentage difference in evaluation loss, relative to Data-Parallel.

Figure 2: DiLoCo does better with scale. We compare Data-Parallel to DiLoCo for varying model sizes N. For all M, DiLoCo improves monotonically wrt Data-Parallel as N increases.

Finding 1: Scale. DiLoCo’s evaluation loss improves relative to Data-Parallel as N increases (Figure 2 and Table 4). Scaling laws predict DiLoCo with M = 2 achieves lower loss than Data-Parallel above several billion parameters, a phenomenon validated by both the largest model over which we performed tuning and our 4B and 10B model training runs (Table 5).

Table 4: Evaluation loss for Data-Parallel and DiLoCo at varying model sizes. We vary model size N and number of replicas M, and report evaluation loss L along with the percentage difference with respect to Data-Parallel (DP). We indicate settings where DiLoCo achieves a lower loss than Data-Parallel in bold.

DiLoCo M = 1 M = 2 M = 4 M = 8

N DP

35M 3.485 3.482 (−0.1%) 3.508 (+0.7%) 3.554 (+2.0%) 3.621 (+3.9%) 90M 3.167 3.162 (−0.1%) 3.182 (+0.5%) 3.213 (+1.5%) 3.265 (+3.1%)

180M 2.950 2.943 (−0.2%) 2.957 (+0.3%) 2.981 (+1.1%) 3.019 (+2.3%) 335M 2.784 2.777 (−0.3%) 2.788 (+0.1%) 2.808 (+0.9%) 2.841 (+2.0%) 550M 2.653 2.645 (−0.3%) 2.657 (+0.2%) 2.673 (+0.7%) 2.698 (+1.7%)

- 1.3B 2.460 2.451 (−0.4%) 2.464 (+0.2%) 2.472 (+0.5%) 2.493 (+1.3%)

- 2.4B 2.326 2.317 (−0.4%) 2.323 (−0.1%) 2.332 (+0.3%) 2.351 (+1.1%)

This finding has two separate, but related, components. First, as noted above, DiLoCo with M = 1 seems to attain lower evaluation loss than Data-Parallel on all model sizes. Moreover, the gap between Data-Parallel and DiLoCo, M = 1 widens as N increases. Second, on most model sizes, DiLoCo with M ≥ 2 achieves a higher evaluation loss. However, if we look at the signed percentage difference between DiLoCo and Data-Parallel, we see that as N increases, DiLoCo does better and better relative to Data-Parallel, and outperforms Data-Parallel at M = 2 when N = 2.4B.

Table 5: DiLoCo outperforms Data-Parallel at larger scales. Here we show the evaluation results on 4B and 10B models, using hyperparameters predicted by scaling laws. We indicate settings where DiLoCo reaches lower loss than Data-Parallel in bold.

Algorithm Loss 4B 10B Data-Parallel 2.224 2.090

- DiLoCo, M = 1 2.219 (-0.22%) 2.086 (-0.19%)

- DiLoCo, M = 2 2.220 (-0.18%) 2.086 (-0.19%) DiLoCo, M = 4 2.230 (+0.18%) 2.096 (+0.29%)

For example, we give the evaluation loss achieved by Data-Parallel and DiLoCo for each model size N in Table 4. We see that for all values of M, the percentage difference strictly decreases with N. DiLoCo with M = 4 goes from attaining 2% higher evaluation loss for N = 35M parameters to only a 0.26% higher evaluation loss at N = 2.4B parameters. This same information is plotted in Figure

- 2. We see that as N increases, the relative evaluation loss of DiLoCo decreases.

We validated this by training 4B and 10B models with hyperparameters set via our scaling laws. Though Figure 2 show to the ‘interpolation’ regime, the result of extensive sweeps, these findings qualitatively carry over to the extrapolation regime, allowing us to train 4B and 10B models to lower evaluation losses using DiLoCo when M = 1,2. Table 5 shows the results of training with our extrapolated hyperparameters, with deeper commentary on the extrapolation regime in Section 6.

Finding 2: Single Replica DiLoCo. DiLoCo with M = 1 attains lower evaluation loss than Data-Parallel across model scales (see Figure 3).

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
| | | | | | |
| | | | | | |
| | | | | | |

2.9

2.8

Model Scale

2.7

335M 550M

EvalLoss

- 1.3B

- 2.4B

2.6

Algorithm

Data Parallel DiLoCo, M=1

2.5

2.4

2.3

216 217 218 219 220

Global Batch Size

(a) Evaluation loss.

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

0.55

0.50

HellaSwagZeroshot

Model Scale

335M 550M

0.45

- 1.3B

- 2.4B

0.40

Algorithm

Data Parallel DiLoCo, M=1

0.35

0.30

216 217 218 219 220

Global Batch Size

(b) Zero-shot accuracy on HellaSwag.

Figure 3: DiLoCo with M = 1 generalizes better than Data-Parallel. We present the evaluation loss and downstream accuracy of Data-Parallel and DiLoCo with M = 1 for varying model and global batch sizes (measured in tokens). In all settings, DiLoCo with M = 1 does better than Data-Parallel, and the gap between them increases with batch size. We see similar results for other model sizes, but omit for the sake of brevity.

Another key findings is that in virtually all settings, DiLoCo with M = 1 attained lower evaluation loss and higher downstream zero-shot evaluation accuracy than Data-Parallel. Moreover, the performance of DiLoCo with M = 1 exhibited much greater stability with respect to batch size; doubling or quadrupling the batch size greatly reduced performance of Data-Parallel, but had little effect on DiLoCo, M = 1, as depicted in Figure 3.

Finding 3: Optimal batch size. DiLoCo increases the optimal batch size and moreover, the optimal global batch size increases with M (see Figures 4 and 5). This means that DiLoCo improves horizontal scalability relative to Data-Parallel (see Figure 6).

Model Scale = 550M

Model Scale = 1.3B

Model Scale = 2.4B

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

2.38

2.85

2.50

Algorithm

2.80

2.36

EvalLoss

Data Parallel

- DiLoCo, M=1

- DiLoCo, M=2

2.75

2.48

2.34

DiLoCo, M=4 DiLoCo, M=8

2.70

2.46

2.32

2.65

215 216 217 218 219 220 221

216 217 218 219 220 221

216 217 218 219 220 221

Global Batch Size

Global Batch Size

Global Batch Size

###### Figure 4: DiLoCo increases optimal batch size, part 1. Evaluation loss of Data-Parallel and DiLoCo as a function of global batch size (in tokens). For all M, DiLoCo exhibits larger optimal batch size than Data-Parallel. Moreover, the optimal batch size increases as a function of M. We see similar results for other model sizes, but omit for conciseness.

Model Scale = 550M

Model Scale = 1.3B

Model Scale = 2.4B

0.48

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.55

0.38

HellaSwagZeroshot

0.47

Algorithm

0.54

Data Parallel

0.36

0.46

- DiLoCo, M=1

- DiLoCo, M=2

0.53

0.45

0.34

DiLoCo, M=4 DiLoCo, M=8

0.52

0.44

0.51

0.32

215 216 217 218 219 220 221

216 217 218 219 220 221

216 217 218 219 220 221

Global Batch Size

Global Batch Size

Global Batch Size

Figure 5: DiLoCo increases optimal batch size, part 2. Zero-shot accuracy on HellaSwag of Data-Parallel and DiLoCo as a function of global batch size (in tokens). Even at smaller model sizes, DiLoCo with M = 2 attains higher accuracy for larger global batch sizes. We see similar results for other model sizes, but omit for conciseness.

While DiLoCo with M > 1 often did slightly worse in terms of evaluation loss when picking the best experiment across all hyperparameters, it exhibited significantly improved performance with respect to batch size. While Data-Parallel and DiLoCo, M = 1 did well with small batch sizes, Data-Parallel’s performance degrades quickly as batch size increases. By contrast, DiLoCo with any

- M exhibits much more stable performance with respect to batch size. Examples of this phenomenon are in Figure 4 for evaluation loss, and Figure 5 for zero-shot accuracy on HellaSwag. As batch size

###### increases, Data-Parallel becomes worse than DiLoCo with M = 2,4, and eventually, M = 8.

3.0

Algorithm

180M

Data Parallel

- DiLoCo, M=1

- DiLoCo, M=2

| |
|---|

335M

2.8

550M

2.6

EvalLoss

1.3B

2.4

2.4B

4B

2.2

10B

103 104 105 106 107

(Idealized) Total Wallclock Time (s)

(a) Network with a bandwidth of 10 gigabits/s and a latency of 10−2 seconds (low-bandwidth).

3.0

Algorithm

180M

Data Parallel

- DiLoCo, M=1

- DiLoCo, M=2

| |
|---|

335M

2.8

550M

2.6

EvalLoss

1.3B

2.4

2.4B

4B

2.2

10B

103 104 105 106

(Idealized) Total Wallclock Time (s)

(b) Network with a bandwidth of 100 gigabits/s and a latency of 10−3 seconds (medium-bandwidth).

3.0

Algorithm

180M

Data Parallel

- DiLoCo, M=1

- DiLoCo, M=2

| |
|---|

335M

2.8

550M

2.6

EvalLoss

1.3B

2.4

2.4B

4B

2.2

10B

103 104 105 106

(Idealized) Total Wallclock Time (s)

(c) Network with a bandwidth of 400 gigabits/s and a latency of 10−4 seconds (high-bandwidth).

Figure 6: DiLoCo trains faster, with or without communication bottlenecks. We plot idealized wall-clock time (see Appendix A) for training by Data-Parallel and DiLoCo across compute nodes connected via high-, medium-, and low-bandwidth networks, for varying model sizes. For models up to 2.4B, we also vary global batch size. For 4B and 10B models, we use the batch size predicted by scaling laws, with discussion of fitting these scaling laws in Section 6. DiLoCo is faster in almost all settings, due to its reduced communication and tolerance to larger batch sizes. Even for high-bandwidth networks, larger batch sizes reduce wall-clock time. We see similar results for M ≥ 4 and for smaller models, but omit for visual clarity.

One important consequence is that DiLoCo results in much more natural horizontal scalability. Recall that in all cases, the token budget D is a function only of N. This means that when using a batch size that is 4× larger (for example), we do 4× fewer training steps. For DiLoCo, this yields quite good performance, and can use more resources all at once, reducing total training time. By contrast, Data-Parallel seems to require much more serial training. This reduction in training time is compounded by reducing communication. To show these effects, we plot an idealized wall-clock time when training under networks of varying bandwidth in Figure 6. We see that DiLoCo’s tolerance for larger batch sizes allows it to achieve comparable loss to Data-Parallel significantly faster, and that this is only amplified in low-bandwidth settings.

Finding 4: Outer learning rate. The optimal outer learning rate is essentially constant with respect to the model size N, but varies with M (see Figure 7).

While optimal inner learning rate varies with model size N, the optimal outer learning rate η for DiLoCo is independent of N and depends only on M. As shown in Figure 7, for sufficiently large models (N ≥ 335M), the best η for each M is constant. Larger values of M seem to necessitate larger η. This is consistent with prior findings that outer learning rate should increase as a function of number of clients in federated learning settings [Charles et al., 2021].

Best Outer Learning Rate

1.0

|1| |0.|8|0.|8|0.|8|0.|8|0.|8|0.|8|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
|0.|6|0.|8|0.|6|0.|6|0.|6|0.|6|0.|6|
| | | | | | |0.|6| | | | | | |
|0.|6|0.|6|0.|6| | |0.|6|0.|6|0.|6|
| | | | | | | | | | | | | | |
|0.|6|0.|4|0.|6|0.|4|0.|4|0.|4|0.|4|
| | | | | | | | | | | | | | |

[Figure 26]

8

0.9

0.8

4

0.7

0.6

M

2

0.5

0.4

1

0.3

0.2

35M 90M 180M 335M 550M 1.3B 2.4B

Model Scale

Figure 7: Optimal outer learning rate is independent of model size. We present the best outer learning rate for DiLoCo for varying M and model sizes N. We select the best outer learning rate over {0.2,0.4,0.6,0.8,1.0}, optimizing over inner learning rate γ and global batch size B. For sufficiently large models, the best outer learning rate is clearly constant.

### 5 Ablations

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.0

0.8

OuterLearningRate

0.6

0.4

0.2

1 5 10 30 100 300

Synchronization Cadence

(a) Optimal outer learning rate for each synchronization cadence. Shaded regions represent the variance across model sizes.

1.0

0.8

OuterLearningRate

0.6

Algorithm

- DiLoCo, M=1

- DiLoCo, M=2

0.4

DiLoCo, M=4

0.2

225 226 227 228 229

Model Size

(b) Optimal outer learning rate for each model size. Shaded regions represent the variance across synchronization cadences.

Figure 8: Outer learning rate scales with M and H, not N. The optimal outer learning rate η is a monotonically increasing function of synchronization cadence H and M (left), but essentially independent of model size N (right). As in Figure 7, this means that we can tune outer learning rate at smaller model scales.

#### 5.1 Synchronization Cadence

Our experiments above all used a synchronization cadence H of 30 (ie. after every H = 30 inner optimization steps, DiLoCo performs an outer optimization step). We now study an ablation designed to ensure that our results are consistent with other values of H. To that end, we apply DiLoCo with varying M and across various model sizes N. For each such setting, we take the optimal inner learning rate and global batch size from above, but perform a sweep over H ∈ {1,5,10,30,100,300}

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

3.8

3.6

Model Scale

3.4

90M

EvalLoss

180M 550M

3.2

Algorithm

- DiLoCo, M=1

- DiLoCo, M=2

3.0

DiLoCo, M=4

2.8

2.6

1 5 10 30 100 300

Synchronization Cadence

Figure 9: Infrequent synchronization works better for larger models. Outside of H = 1, which performs the worst, evaluation loss increases with H. However, the rate of increase is less pronounced for DiLoCo with M = 1 and for larger models, suggesting that large models can be synchronized quite infrequently.

and η ∈ {0.05,0.1,0.2,0.4,0.6,0.8,1.0}. We first study whether the observation in Section 4, that η should be tuned independently of N, holds for other values of H.

We give results in the affirmative in Figure 8. We see that across model scales, the optimal learning rate is essentially only a function of the number of replicas M and the synchronization cadence H, and is essentially independent of model size N. There is some slight variation, though this is likely due to not re-tuning the inner learning rate. Moreover, our results actually show a potentially surprising phenomenon: The optimal outer learning rate increases with H. This is potentially counter-intuitive; as H increases, the DiLoCo replicas may diverge more, and so one might expect that a more conservative learning rate is warranted. This is not the case.

Next, we analyze how H impacts the evaluation loss of DiLoCo in Figure 9. For all models, synchronizing every step (H = 1) performs the worst, but after this point all values of H perform somewhat comparably. For a fixed N and M, evaluation loss increases as H increases. However, this increase is less pronounced for M = 1 and larger models. This yields an important finding: As the model size N increases, we can actually perform synchronization across DiLoCo replicas less frequently, while nearly maintaining evaluation performance.

Compute utilization. The synchronization cadence of DiLoCo is critical to training large-scale models distributed across the world. Indeed, less frequent synchronization (larger H) diminishes the bandwidth requirements of training. Following Douillard et al. [2025], we simulate the amount of bandwidth required to have a compute utilization (computetotal timetime) as large as possible for three types of LLMs: a 10B Chinchilla-style transformer [Hoffmann et al., 2022] in Fig.10a, a 405B Llama3 model [Dubey et al., 2024] in Fig.10b, and a 671B DeepSeek-v3 MoE [Liu et al., 2024a] in Fig.10c. We also report raw numbers in Table 6.

[Figure 27]

[Figure 28]

[Figure 29]

(a) 10B Chinchilla (b) 405B Llama3 (c) 671B DeepSeek-V3

Figure 10: DiLoCo greatly increases compute utilization. Here we present simulated compute utilization for DiLoCo and Data-Parallel across a range of bandwidth and synchronization cadences H. A compute utilization of 0.8 means 80% of the time is spent in computation, and 20% in communication. A higher value is better. We see similar results for other model sizes, but omit for visual clarity.

Gbit/s to reach a compute utilization CU =?

Architecture Size Step time Method

50% 80% 90% 95% 99%

Data-Parallel 104.8 184.2 222.3 222.3 390.7 DiLoCo, H = 1 104.8 184.2 222.3 222.3 390.7 DiLoCo, H = 10 16.0 49.4 86.8 152.6 222.3 DiLoCo, H = 50 3.0 11.0 23.3 41.0 126.5

Chinchilla 10B 0.8s

DiLoCo, H = 100 1.4 6.2 13.3 23.3 86.8 DiLoCo, H = 300 0.5 2.0 4.3 9.1 41.0

Data-Parallel 126.5 222.3 268.3 323.8 323.8 DiLoCo, H = 1 126.5 222.3 268.3 323.8 323.8 DiLoCo, H = 10 19.3 72.0 126.5 184.2 268.3 DiLoCo, H = 50 3.6 13.3 28.1 59.6 184.2 DiLoCo, H = 100 2.0 7.5 16.0 33.9 126.5

Llama3 405B 26s

- DiLoCo, H = 300 0.7 3.0 6.2 13.3 59.6

DeepSeek-V3 671B 20s

Data-Parallel 323.8 569.0 686.6 686.6 1000.0+ DiLoCo, H = 1 323.8 569.0 686.6 686.6 1000.0+

DiLoCo, H = 10 49.4 152.6 268.3 390.7 686.6 DiLoCo, H = 50 7.5 33.9 72.0 126.5 390.7 DiLoCo, H = 100 4.3 16.0 41.0 72.0 268.3

- DiLoCo, H = 300 1.7 6.2 13.3 28.1 126.5

Table 6: Simulated compute utilization. We estimate the step time based on the required flops using the rule proposed by Kaplan et al. [2020] and a max flop utilization of 60%. We estimate the bandwidth (in Gbit/s) required to reach a level of compute utilization using [Douillard et al., 2025]’s simulator. We highlight in light blue 10× reduction of bandwidth, and in dark blue 100× reduction.

#### 5.2 Overtraining

3.8

3.6

| |
|---|

3.4

Algorithm

3.2

Data Parallel

EvalLoss

- DiLoCo, M=1

- DiLoCo, M=2

| |
|---|
| |
| |

3.0

DiLoCo, M=4

Overtrain Multiplier

1 4 16

2.8

2.6

| |
|---|

2.4

1017 1018 1019 1020 1021 1022

Total FLOPs

Figure 11: DiLoCo scales reliably with overtraining. For each overtrain multiplier, the curves of loss with respect to FLOPs are all essentially parallel lines in log-space. We see that variations in loss from changing the algorithm is dominated by changing the model size or amount of overtraining. Moreover, DiLoCo with M = 1 is slightly better than all other algorithms, including Data-Parallel, at all overtraining amounts and model sizes.

In all of the above experiments, we use the Chinchilla-optimal amount of tokens for each model size [Hoffmann et al., 2022]. However, in many settings (e.g. when training a model with inference costs in mind) it is beneficial to perform overtraining, where we use more tokens [Gadre et al., 2024]. We want to make sure that our results are robust to the amount of overtraining. To that end, we perform a swath of ablation studies on various overtraining multipliers. Given an overtraining multiplier λ ≥ 1, we train on D = 20Nλ tokens, so that λ = 1 corresponds to the Chinchilla-optimal number of tokens.

For our experiments, for varying model sizes N and algorithms, we take the best-performing hyperparameters from our results above (ie. (global) batch size B, (inner) learning rate γ, and outer learning η when using DiLoCo). We then train for D = 20Nλ tokens on the Dolma dataset [Soldaini et al., 2024], as it has more tokens than C4. Since we use QK-layernorm, we generally avoid the need to re-tune learning rates [Gadre et al., 2024]. We take the resulting models and evaluate them on the same evaluation set as before, the validation split of C4. We vary the overtrain multiplier λ over {1,4,16}. Note that retraining for λ = 1 is necessary since we changed the underlying dataset. We train with Data-Parallel as well as DiLoCo with M ∈ {1,2,4}.

Our results are presented in Figure 11. We see that qualitatively, the scaling remains essentially unchanged as we do more overtraining. We emphasize that we did not re-tune any hyperparameter in these experiments. For each model size and algorithm, we simply took the best-performing hyperparameters from our Chinchilla-optimal experiments on C4. This means that the consistency of DiLoCo as we overtrain held despite the fact that for M > 1, we used much larger batch sizes than Data-Parallel.

To illustrate this, we plot idealized training time of Data-Parallel and DiLoCo with M = 2 for different overtraining amounts in Figure 12. DiLoCo speeds up overtraining by reducing communication costs,

3.4

3.2

| |
|---|

| |
|---|

3.0

EvalLoss

| |
|---|

| |
|---|

2.8

2.6

| |
|---|

| |
|---|

| |
|---|

2.4

103 104 105 106 107

(Idealized) Total Wallclock Time (s)

(a) Network with a bandwidth of 10 gigabits/s and a latency of 10−2 seconds (low-bandwidth).

3.4

3.2

| | |
|---|---|
| | |

3.0

EvalLoss

| | |
|---|---|

2.8

2.6

| |
|---|

| |
|---|

| |
|---|

2.4

103 104 105 106

(Idealized) Total Wallclock Time (s)

(b) Network with a bandwidth of 100 gigabits/s and a latency of 10−3 seconds (medium-bandwidth).

3.4

3.2

| |
|---|
| |

Algorithm

3.0

EvalLoss

Data Parallel DiLoCo, M=2

Overtrain Multiplier

| |
|---|

1 4 16

2.8

2.6

| |
|---|

| |
|---|

| |
|---|

2.4

103 104 105 106

(Idealized) Total Wallclock Time (s)

(c) Network with a bandwidth of 400 gigabits/s and a latency of 10−4 seconds (high-bandwidth).

Figure 12: DiLoCo speeds up overtraining by leveraging horizontal scalability. For DataParallel and DiLoCo, M = 2, we plot idealized wall-clock time (see Appendix A) for training by Data-Parallel and DiLoCo, M = 2 across compute nodes connected via high-, medium-, and low-bandwidth networks. For each algorithm and overtraining amount, we display lines represent varying model sizes, from 335M parameters to 2.4B. DiLoCo is faster in all settings, due to both its reduced communication and its tolerance to larger batch sizes. Even in the high-bandwidth setting, the larger batch sizes increase horizontal scalability, reducing end-to-end wallclock time. We see similar results for M ≥ 4 and for smaller models, but omit for visual clarity.

and utilizing larger batch sizes, therefore requiring fewer serial training steps. This suggests that DiLoCo is a significant boon in overtraining, as we can amortize compute time (which can be quite long for overtraining) via horizontal scalability.

### 6 Scaling Laws

We now discuss the process we used to fit scaling laws to our empirical results. Recall that for each algorithm (Data-Parallel or DiLoCo with M ∈ {1,2,4,8}) we ran extensive hyperparameter sweeps on models of size N up to 2.4B. To fit a scaling law for loss L as a function of N, we pick the best hyperparameters (in terms of evaluation loss) for each N, aggregate the loss values, and fit some kind of function for L(N), such as a power law [Kaplan et al., 2020]. We will also fit scaling laws to the optimal hyperparameters.

We will fit two types of scaling laws for DiLoCo: independent fits for each value of M, and joint fits where we fit a single scaling law as a function of N and M simultaneously.

#### 6.1 Independent scaling laws

Scaling laws for loss. We first fit scaling laws for the loss obtained by Data-Parallel training. We fit a power law to the evaluation loss of Data-Parallel as a function of N via the power law approximation L(N) ≈ ANα. Note that this can easily be done via applying linear fit techniques to log(L), and is not sensitive to things like initial values of A,α. The resulting power law is in the first row of Table 7.

We mirror this above for DiLoCo when doing independent fits. For each value of N,M, we record the lowest loss value across all hyperparameters. We can then fit power law LM(N) := L(N,M) ≈ ANα for each M. The results are given in Table 7.

Table 7: Power law approximations for loss L(N) ≈ ANα.

A α Data-Parallel 18.129 −0.0953

- DiLoCo, M = 1 18.363 −0.0961
- DiLoCo, M = 2 18.768 −0.0969 DiLoCo, M = 4 19.762 −0.0992 DiLoCo, M = 8 21.051 −0.1018

The results show that Data-Parallel and DiLoCo see similar predicted reductions in loss as a function of N. Notably, the fit parameters suggest that DiLoCo, M = 1 outperforms Data-Parallel at essentially all but the absolute smallest model scales. This mirrors the results discussed in Section 4.

Scaling laws for hyperparameters. For Data-Parallel, we fit scaling laws for learning rate γ

and batch size B. For DiLoCo, we fit scaling laws for inner learning rate γ and global batch size B. Given their analogous role in the algorithms, we fit them in the same way. For (inner) learning rate, we use the same approach as fitting scaling laws for loss: for each N (and M, for DiLoCo), we select the best hyperparameters, and fit a power law. The results are in Table 8.

For (global) batch size, we alter this slightly. As discussed in Section 3, our sweeps use powers of 2 for batch size, in order to saturate compute. However, the optimal batch size may be between these values. To account for this, we first fit a quadratic approximation to the batch size. Specifically, for each value of N we look at the loss as a function of log2(B) (when using the best learning rate for that B), and fit a quadratic to this function. We select the minima of those quadratics and fit a power law to them, as a function of N. The results are in Table 9.

DiLoCo has a third hyperparameter we could fit scaling laws to: the outer learning rate. However, as shown in Section 4, the optimal outer learning rate is (for sufficiently large models) seemingly constant. Therefore, a scaling law would seemingly not yield any improved predictive performance over simply using the best outer learning rate for each M (see Figure 7).

#### 6.2 Joint scaling laws

Alternatively, we can fit joint power laws to various facets of DiLoCo, using a two-variable power law f(N,M) ≈ ANαMβ. We do this for loss L, inner learning rate γ and global batch size B. For the first two, we select, for each value of N,M, the best learning rate and loss. For batch size we do the same, but using the quadratic approximations from the section above. We can then fit a joint power law via standard linear regression techniques. The resulting power laws are in Table 10. Just as with the independent fits, we do not attempt to model outer learning, as the optimal value is independent of N.

Table 8: Power law approximations for (inner) learning rate γ(N) ≈ ANα.

A α

Data-Parallel 16319.2 −0.819 DiLoCo, M = 1 74620.6 −0.945 DiLoCo, M = 2 3978.82 −0.780 DiLoCo, M = 4 4512.99 −0.789 DiLoCo, M = 8 618986 −1.102

Table 9: Power law approximations for (global) batch size B(N) ≈ ANα.

A α Data-Parallel 0.22592 0.281

- DiLoCo, M = 1 0.01361 0.435
- DiLoCo, M = 2 0.00769 0.479 DiLoCo, M = 4, 0.00535 0.510 DiLoCo, M = 8 0.01859 0.455

Table 10: Joint power law approximations f(N,M) = ANαMβ for the loss L, inner learning rate γ, and batch size B of DiLoCo.

A α β

L 19.226 −0.0985 0.0116 γ 22256 −0.8827 0.2929 B 0.00709 0.4695 0.3399

#### 6.3 Measuring goodness-of-fit

Now that we have two different ways of developing scaling laws for DiLoCo, we can attempt to ask which one yields better predictions. First we do this via leave-one-out validation. Specifically, we use the same methodology as above to fit scaling laws for L, γ, and B, but only using data up to N = 1.3B parameters, leaving out our data on N = 2.4B parameters. We then use the scaling law to predict the optimal value for L,γ, and B at N = 2.4B parameters, across different values of M.

Given a predicted value y˜ and an actual optimal value of y, we compute the residual of our prediction as the mean absolute error of the logarithm: res(y,y˜) = |log(y) − log(˜y)|. We use this measure as it works well for all three variables simultaneously, despite the fact that they vary greatly in their scale. For each M ∈ {1,2,4,8}, we compute the predicted value of the three parameters above, and measure the residual relative the actual optimal value at N = 2.4B. We report these values, as well

- as their average across M, in Table 11.

Our results generally show that both approach is generally valid (as there is no clear winner) but also that there is significant variation in residuals between M. That being said, we see that on average, while the individual fit is slightly better at predicting the loss and global batch size, the independent fit is significantly better at predicting inner learning rate.

#### 6.4 Extrapolating to larger models

We use the independent and joint fits to predict optimal hyperparameters for Data-Parallel and DiLoCo with M ∈ {1,2,4} at 4B and 10B model scales. Note that for Data-Parallel, we can only use independent fits. We run training on these models with these hyperparameters, using a Chinchilla-optimal token budget, and compare the results.

We see two important facets. First, unlike results above, at 4B and 10B scales we see that DiLoCo with M = 2 actually outperforms both Data-Parallel and DiLoCo, M = 1, regardless of using

Table 11: Joint fit scaling laws match or beat independent fit. Here we give the residuals for scaling law predictions for N = 2.4B and varying M. We compare the residual of independent and joint fitting strategies in predicting loss L, inner learning rate γ, and global batch size B. For the average residuals, we highlight which of independent or joint achieved a lower residual. We see that the joint fit matches independent for L and B, but does better at predicting γ.

Fit L γ B

- M = 1

Independent 0.011 0.35 0.00088 Joint 0.019 0.14 0.19

- M = 2

Independent 0.0099 0.18 0.44

Joint 0.013 0.29 0.28 M = 4

Independent 0.012 0.051 0.25

Joint 0.0082 0.086 0.11 M = 8

Independent 0.014 0.62 0.076

Joint 0.0076 0.23 0.19 Average over M

Independent 0.012 0.30 0.19 Joint 0.012 0.19 0.19

individual or joint fit approaches. Second, we see that DiLoCo, M = 1 requires the joint fit to do better than Data-Parallel. Other than in this case, joint and independent fits perform comparably throughout. All in all, the joint fit approach to hyperparameters appears to have a slight edge over individual fit in extrapolating. Combined with its ability to also extrapolate to larger M, we generally recommend the joint fit approach for all hyperparameters.

We now use these loss values to see how they compare to the scaling laws fit above. We generally find that the loss values are predicted very well, within a few percentage points of the loss predicted by the scaling laws. We present the fit scaling law and extrapolated loss values in Figure 13.

Data Parallel

DiLoCo, M=1

DiLoCo, M=2

DiLoCo, M=4

3.5

3.5

Scaling Law Fit Points

3.5

3.5

Extrapolated Fit Points

3.0

3.0

3.0

3.0

EvalLoss

EvalLoss

EvalLoss

EvalLoss

2.5

2.5

2.5

2.5

2.0

2.0

2.0

2.0

225 226 227 228 229 230 231 232 233 234

225 226 227 228 229 230 231 232 233 234

225 226 227 228 229 230 231 232 233 234

225 226 227 228 229 230 231 232 233 234

Model Size

Model Size

Model Size

Model Size

###### Figure 13: DiLoCo scaling laws extrapolate well to larger models. We present loss scaling laws for Data-Parallel and DiLoCo. Pictured are both the loss values to form the scaling law (by training models up to a scale of 2.4B) and loss values attained on larger models (4B and 10B). While we present the individual-fit scaling laws for simplicity, the joint fit also predicts loss well.

Table 12: Joint fit hyperparameters extrapolate well to larger models. Here we show the evaluation results on 4B and 10B models, using hyperparameters predicted by individual and joint scaling laws. We highlight DiLoCo evaluation results that were better (ie. lower) than Data-Parallel. We see that while DiLoCo with M = 2 does better than Data-Parallel with either independent or joint fit rules, DiLoCo M = 1 only does better when using joint fit.

Loss 4B 10B

Algorithm Fit Method

Data-Parallel Independent 2.224 2.090

- DiLoCo, M = 1

Independent 2.229 2.103 Joint 2.219 2.086

- DiLoCo, M = 2

Independent 2.218 2.083

Joint 2.220 2.086 DiLoCo, M = 4

Independent 2.232 2.098 Joint 2.230 2.096

#### 6.5 Parametric Function Fitting

While power laws have useful properties, various works on scaling laws have often found it useful to fit more complex functions to the data [Hoffmann et al., 2022]. In particular, a power law such as ANα will only ever tend towards 0,1, or diverge to ∞ as N → ∞. In many cases, it makes sense to fit more complex functions.

We are particularly interested in parametric forms for our joint scaling laws. While Hoffmann et al. [2022] use a risk decomposition argument to decompose loss as a function of N and D, it is not immediately clear how to do decompose L(N,M). To that end, we use an empirical approach. We develop candidate functions, and determine which does best in an extrapolative sense. We use the following functional forms:

- 1. L(N,M) ≈ ANαMβ
- 2. L(N,M) ≈ ANαMβ + C
- 3. L(N,M) ≈ ANα+βM + C
- 4. L(N,M) ≈ ANα + BMβ + C

The first is included for comparison’s sake, as it recovers the power law scaling law used above. Fitting more sophisticated functions can be much more sensitive to initial values of parameters, and also sensitive to outlier data. Therefore, when fitting these functions to our loss values above, we use the general strategy proposed by Hoffmann et al. [2022].

In detail, let fQ(N,M) denote one of the functional forms above, where Q represents the set of parameters to be fit (e.g. Q = {A,α,β} for the first). Let Huberδ denote the Huber loss with parameter δ. Let N,M denote the set of values of N and M considered. For each N,M, we have an empirical loss L(N,M), and some estimate of the loss fQ(N,M). We then solve the following

minimization problem:

Huberδ log fQ(N,M) − log L(N,M) .

min

Q

N∈N M∈M

We minimize this via L-BFGS, using some initialization Q0 for the parameters. We repeat this process for 256 random initializations Q0 of the parameters. We hold out all loss values at the N = 2.4B scale, and select the parameters Q that best fit the held-out data, measured in terms of the average residual |log fQ(N,M) − log L(N,m)| over all M.

Table 13: Parametric function fitting improves joint scaling laws. We showcase various parametric approximations to the empirical loss function L(N,M), along with their validation error on held-out loss data at the N = 2.4B scale. We see that joint power laws (the first) and classical loss decomposition (the last) are significantly worse at predicting loss on the held-out data.

Parametric form Average Residual

L(N,M) ≈ ANαMβ 0.0044 L(N,M) ≈ ANαMβ + C 0.0035 L(N,M) ≈ ANα+βM + C 0.0025 L(N,M) ≈ ANα + BMβ + C 0.0043

We see that the power law (row 1) and additive decomposition (row 4) are significantly worse

- at extrapolating loss values than more nuanced parametric forms. We note that the additive decomposition resembles the decomposition of loss as a function of model size N and token budget D used by Hoffmann et al. [2022], but does not seem to reflect how M affects loss for DiLoCo. We leave it as an open problem to determine what parametric forms better predict loss, and can be explained by theoretical understanding of communication-efficient training.

### 7 Related Work

Distributed training of LLMs. Due to their increasingly large sizes, the advancement of language models has necessitated advancements in distributed training methods. One vein of work uses data-parallel training, and attempts to shard its constituent computations across accelerators in efficient ways. This includes advancements in things like distributed data parallelism [Shazeer et al., 2018, Li et al., 2020], ZeRO parallelism [Rajbhandari et al., 2020, Ren et al., 2021], fully-sharded data parallelism [FairScale authors, 2021, Zhao et al., 2023], and pipeline parallelism [Petrowski et al., 1993, Huang et al., 2019, Narayanan et al., 2019]. Conceptual models of the impact of batch size on training time [McCandlish et al., 2018] aid in making the most effective trade-offs between training time and compute cost when using data-parallel training methods.

As scale continues to increase, the need to all-reduce gradients between data-parallel replicas becomes a bottleneck in training, causing accelerators to ‘wait’ on this allreduce for an unacceptably long time. This observation has motivated extensive work in pipeline parallelism and scheduling, communicating activations rather than gradients. An alternative line of work keeps the basic data processing pattern of data parallelism while directly minimizing communication requirements.

Three broad families of algorithms exist in that space: 1) sparse updates (including CocktailSGD [Wang et al., 2023], PowerSGD [Vogels et al., 2019], and DeMo [Peng et al., 2024]), 2) fast asynchronous updates (including Hogwild [Niu et al., 2011], WASH [Fournier et al., 2024], and Sparta [Baioumy and Cheema, 2025]), and 3) infrequent updates (including LocalSGD [Stich, 2018], FedOpt [Reddi et al., 2021], and DiLoCo [Douillard et al., 2023]).

In this work, we focus on the third category, which Douillard et al. [2023] proved recently it can reduce communication costs in LLM training significantly more by training multiple models independently with infrequent synchronization. Their method, DiLoCo, massively reduces communication overhead when training LLMs with a moderate numbers of model replicas. It has also shown great promise in training LLMs up to 10 billion parameters [Jaghouar et al., 2024b,a]. This work has also been extended to asynchronous overlapped updates [Liu et al., 2024b, Douillard et al., 2025], and low-communication expert sharding [Douillard et al., 2024].

Federated learning. There is an enormous body of work on communication-efficient training methods for machine learning. In that vein, DiLoCo is closely related to algorithms used in federated learning to perform communication-efficient training over decentralized data, often (but not exclusively) on edge devices [Kairouz et al., 2021]. The prototypical algorithm used in federated learning, FedAvg [McMahan et al., 2017], reduces communication costs by training models in parallel, with periodic model averaging. This algorithm has been invented and reinvented throughout machine learning, and is also known as Local SGD [Stich, 2018], parallel SGD [Zinkevich et al., 2010], and parallel online backpropagation [Mangasarian and Solodov, 1993]. The use of inner and outer optimization steps (as in DiLoCo, see Algorithm 1) was first used by Hsu et al. [2019] and Reddi et al. [2021] for federated learning, focusing on SGD as the inner optimizer and SGDM or Adam [Kingma and Ba, 2014] as the outer optimizer in order to leverage more sophisticated optimizers in resourceconstrained settings. DiLoCo is also closely related to many other federated optimization methods, though the huge amount of work in this area makes it impossible to summarize succinctly here. We instead refer the interested reader to the survey of Wang et al. [2021], though the field has of course progressed since then. While federated learning is often applied to more moderately sized models, Charles et al. [2024] and Sani et al. [2024] show that federated learning can be used to good effect for LLM training.

Scaling laws. Scaling laws work often aims to estimate how empirical generalization error scales with various facets, including model size and training set size. Empirical scaling analyses with power law behavior date have existed for decades (see [Banko and Brill, 2001]). Hestness et al. [2017] developed power laws for model and dataset size across various tasks and model architectures (including encoder-decoder LSTM models). More recently, scaling laws for transformer-based LLMs were proposed by Kaplan et al. [2020] and Hoffmann et al. [2022], who exhibited power law relationships between LLM performance and model size. Sine then, there has been a large number of works developing scaling laws for other facets of LLMs, including (among many others) inference costs [Sardana et al., 2023], data-constrained training [Muennighoff et al., 2023], and overtraining [Gadre et al., 2024].

Scaling laws for DiLoCo were previously studied by He et al. [2024], who show that DiLoCo with 8 replicas exhibits analogous scaling behavior to Data-Parallel. He et al. [2024] use a fixed number

of replicas, batch size, and outer learning rate, and an unspecified total token budget.4 Our work expands on many aspects of their work and explores others that were not considered, including but not limited to: 10× larger models, varying the number of replicas (including single-replica DiLoCo), varying token budgets and overtraining, parametric function fitting, scaling laws for hyperparameters, and optimal batch size analysis.

### 8 Conclusions and Future Work

Our results above all show that like Data-Parallel, DiLoCo scales predictably with model size in ways that make it simpler to tune hyperparameters and train models at extremely large scales. Moreover, DiLoCo can offer significant benefits over Data-Parallel, including superior evaluation loss when using a single model replica, and increased optimal batch size for any number of model replicas. Moreover, these benefits are robust to model scale, overtraining amount, and synchronization frequency.

There are at least three promising veins of future work in this space. First, the scaling law analyses can be augmented by various facets already used for Data-Parallel scaling laws. Examples include downstream task analysis [Gadre et al., 2024], data-constrained scaling laws [Muennighoff et al., 2023], and inference costs [Sardana et al., 2023]. Second, the scaling laws can be adapted to encompass proposed improvements to DiLoCo and related methods, including asynchronous updates [Liu et al., 2024b], streaming DiLoCo [Douillard et al., 2025], and modular architectures co-designed with training methods [Douillard et al., 2024, Huh et al., 2024]. Last, there is a clear need for systems and software that can be used to deploy methods like DiLoCo at scale and attain its communication-efficiency benefits in practical extremely large settings [Jaghouar et al., 2024b].

### Acknowledgments

We would like to thank Lechao Xiao for invaluable advice on scaling law methodology and using the NanoDO codebase; Adhiguna Kuncoro, Andrei Rusu, Jeffrey Pennington, Jiajun Shen, Mark Kurzeja, Nicole Mitchell, Qixuan Feng, Rachita Chhaparia, Ran Tian, Satyen Kale, Sean Augenstein, Vincent Roulet, Yani Donchev, and Zohar Yahav for helpful comments, advice, and conversations; Brendan McMahan, Daniel Ramage, Marc’Aurelio Ranzato, and Prateek Jain for feedback, guidance, and leadership support.

### References

Mohamed Baioumy and Alex Cheema. Sparta. blog.exolabs.net, 2025. URL https://blog.exolabs. net/day-12/.

Michele Banko and Eric Brill. Scaling to very very large corpora for natural language disambiguation. In Proceedings of the 39th annual meeting of the Association for Computational Linguistics, pages 26–33, 2001.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.

4At the time of writing, He et al. [2024] only say that “Each model was trained to achieve adequate convergence”.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http: //github.com/jax-ml/jax.

Zachary Charles and Keith Rush. Iterated vector fields and conservatism, with applications to federated learning. In Sanjoy Dasgupta and Nika Haghtalab, editors, Proceedings of The 33rd International Conference on Algorithmic Learning Theory, volume 167 of Proceedings of Machine Learning Research, pages 130–147. PMLR, 29 Mar–01 Apr 2022. URL https://proceedings.

mlr.press/v167/charles22a.html.

Zachary Charles, Zachary Garrett, Zhouyuan Huo, Sergei Shmulyian, and Virginia Smith. On large-cohort training for federated learning. Advances in neural information processing systems, 34:20461–20475, 2021.

Zachary Charles, Nicole Mitchell, Krishna Pillutla, Michael Reneer, and Zachary Garrett. Towards federated foundation models: Scalable dataset pipelines for group-structured learning. Advances in Neural Information Processing Systems, 36, 2024.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Arthur Douillard, Qixuan Feng, Andrei A Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc’Aurelio Ranzato, Arthur Szlam, and Jiajun Shen. DiLoCo: Distributed lowcommunication training of language models. arXiv preprint arXiv:2311.08105, 2023.

Arthur Douillard, Qixuan Feng, Andrei A Rusu, Adhiguna Kuncoro, Yani Donchev, Rachita Chhaparia, Ionel Gog, Marc’Aurelio Ranzato, Jiajun Shen, and Arthur Szlam. DiPaCo: Distributed path composition. arXiv preprint arXiv:2403.10616, 2024.

Arthur Douillard, Yanislav Donchev, Keith Rush, Satyen Kale, Zachary Charles, Zachary Garrett, Gabriel Teston, Dave Lacey, Ross McIlroy, Jiajun Shen, et al. Streaming DiLoCo with overlapping communication: Towards a distributed free lunch. arXiv preprint arXiv:2501.18512, 2025.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

FairScale authors. Fairscale: A general purpose modular pytorch library for high performance and large scale training. https://github.com/facebookresearch/fairscale, 2021.

Louis Fournier, Adel Nabli, Masih Aminbeidokhti, Marco Pedersoli, Eugene Belilovsky, and Edouard Oyallon. WASH: train your ensemble with communication-efficient weight shuffling, then average. Advances in Neural Information Processing Systems (NeurIPS) Workshop, 2024. URL https: //arxiv.org/abs/2405.17517.

Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, et al. Language models scale reliably with over-training and on downstream tasks. arXiv preprint arXiv:2403.08540, 2024.

Qiaozhi He, Xiaomin Zhuang, and Zhihua Wu. Exploring scaling laws for local SGD in large language model training. arXiv preprint arXiv:2409.13198, 2024.

Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335, 2019.

Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, et al. GPipe: Easy scaling with micro-batch pipeline parallelism. proceeding of Computer Science> Computer Vision and Pattern Recognition, 2019.

Minyoung Huh, Brian Cheung, Jeremy Bernstein, Phillip Isola, and Pulkit Agrawal. Training neural networks from scratch with parallel low-rank adapters. arXiv preprint arXiv:2402.16828, 2024.

Sami Jaghouar, Jack Min Ong, Manveer Basra, Fares Obeid, Jannik Straube, Michael Keiblinger, Elie Bakouch, Lucas Atkins, Maziyar Panahi, Charles Goddard, Max Ryabinin, and Johannes Hagemann. INTELLECT-1 technical report, 2024a. URL https://arxiv.org/abs/2412.01152.

Sami Jaghouar, Jack Min Ong, and Johannes Hagemann. OpenDiLoCo: An open-source framework for globally distributed low-communication training. arXiv preprint arXiv:2407.07852, 2024b.

Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. Foundations and trends® in machine learning, 14(1–2): 1–210, 2021.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, et al. PyTorch distributed: Experiences on accelerating data parallel training. arXiv preprint arXiv:2006.15704, 2020.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024a.

Bo Liu, Rachita Chhaparia, Arthur Douillard, Satyen Kale, Andrei A Rusu, Jiajun Shen, Arthur Szlam, and Marc’Aurelio Ranzato. Asynchronous local-SGD training for language modeling. arXiv preprint arXiv:2401.09135, 2024b.

Peter J. Liu, Roman Novak, Jaehoon Lee, Mitchell Wortsman, Lechao Xiao, Katie Everett, Alexander A. Alemi, Mark Kurzeja, Pierre Marcenac, Izzeddin Gur, Simon Kornblith, Kelvin Xu, Gamaleldin Elsayed, Ian Fischer, Jeffrey Pennington, Ben Adlam, and Jascha-Sohl Dickstein. NanoDO: A minimal transformer decoder-only language model implementation in JAX., 2024c. URL http://github.com/google-deepmind/nanodo.

I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. Olvi L Mangasarian and Mikhail V Solodov. Backpropagation convergence via deterministic

nonmonotone perturbed minimization. Advances in Neural Information Processing Systems, 6, 1993.

Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model of large-batch training. arXiv preprint arXiv:1812.06162, 2018.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pages 1273–1282. PMLR, 2017.

Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel. Scaling data-constrained language models. Advances in Neural Information Processing Systems, 36:50358–50376, 2023.

Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R Devanur, Gregory R Ganger, Phillip B Gibbons, and Matei Zaharia. PipeDream: Generalized pipeline parallelism for DNN training. In Proceedings of the 27th ACM symposium on operating systems principles, pages 1–15, 2019.

Feng Niu, Benjamin Recht, Christopher Re, and Stephen J. Wright. HOGWILD!: A lock-free approach to parallelizing stochastic gradient descent, 2011. URL https://arxiv.org/abs/1106.5730.

Matteo Pagliardini, Pierre Ablin, and David Grangier. The AdEMAMix optimizer: Better, faster, older. arXiv preprint arXiv:2409.03137, 2024.

Pitch Patarasuk and Xin Yuan. Bandwidth optimal all-reduce algorithms for clusters of workstations. J. Parallel Distrib. Comput., 69(2):117–124, February 2009. ISSN 0743-7315. doi: 10.1016/j.jpdc. 2008.09.002. URL https://doi.org/10.1016/j.jpdc.2008.09.002.

Bowen Peng, Jeffrey Quesnelle, and Diederik P Kingma. Decoupled momentum optimization. arXiv preprint arXiv:2411.19870, 2024.

Alain Petrowski, Gerard Dreyfus, and Claude Girault. Performance analysis of a pipelined backpropagation parallel algorithm. IEEE Transactions on Neural Networks, 4(6):970–981, 1993.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.

Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konečný, Sanjiv Kumar, and Hugh Brendan McMahan. Adaptive federated optimization. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= LkFG3lB13U5.

Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He. {Zero-offload}: Democratizing {billion-scale} model training. In 2021 USENIX Annual Technical Conference (USENIX ATC 21), pages 551–564, 2021.

Keith Rush, Zachary Charles, Zachary Garrett, Sean Augenstein, and Nicole Elyse Mitchell. DrJAX: Scalable and differentiable mapreduce primitives in JAX. In 2nd Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@ ICML 2024).

Lorenzo Sani, Alex Iacob, Zeyu Cao, Royson Lee, Bill Marino, Yan Gao, Dongqi Cai, Zexi Li, Wanru Zhao, Xinchi Qiu, et al. Photon: Federated LLM pre-training. arXiv preprint arXiv:2411.02908, 2024.

Nikhil Sardana, Jacob Portes, Sasha Doubov, and Jonathan Frankle. Beyond chinchilla-optimal: Accounting for inference in language model scaling laws. arXiv preprint arXiv:2401.00448, 2023.

Noam Shazeer, Youlong Cheng, Niki Parmar, Dustin Tran, Ashish Vaswani, Penporn Koanantakool, Peter Hawkins, HyoukJoong Lee, Mingsheng Hong, Cliff Young, et al. Mesh-TensorFlow: Deep learning for supercomputers. Advances in neural information processing systems, 31, 2018.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al. Dolma: An open corpus of three trillion tokens for language model pretraining research. arXiv preprint arXiv:2402.00159, 2024.

Sebastian U Stich. Local SGD converges fast and communicates little. arXiv preprint arXiv:1805.09767, 2018.

Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and momentum in deep learning. International Conference on Machine Learning (ICML), 2013. URL https://proceedings.mlr.press/v28/sutskever13.html.

Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. PowerSGD: practical low-rank gradient compression for distributed optimization. Advances in Neural Information Processing Systems (NeurIPS), 2019. URL https://arxiv.org/abs/1905.13727.

Jianyu Wang, Zachary Charles, Zheng Xu, Gauri Joshi, H Brendan McMahan, Maruan Al-Shedivat, Galen Andrew, Salman Avestimehr, Katharine Daly, Deepesh Data, et al. A field guide to federated optimization. arXiv preprint arXiv:2107.06917, 2021.

Jue Wang, Yucheng Lu, Binhang Yuan, Beidi Chen, Percy Liang, Christopher De Sa, Christopher Re, and Ce Zhang. CocktailSGD: fine-tuning foundation models over 500mbps networks. International

Conference on Machine Learning (ICML), 2023. URL https://openreview.net/forum?id= w2Vrl0zlzA.

Xi Wang and Laurence Aitchison. How to set AdamW’s weight decay as you scale model and dataset size. arXiv preprint arXiv:2405.13698, 2024.

Mitchell Wortsman, Peter J Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, et al. Small-scale proxies for large-scale transformer training instabilities. arXiv preprint arXiv:2309.14322, 2023.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Michael Zhang, James Lucas, Jimmy Ba, and Geoffrey E Hinton. Lookahead optimizer: k steps forward, 1 step back. Advances in neural information processing systems, 32, 2019.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch FSDP: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

Martin Zinkevich, Markus Weimer, Lihong Li, and Alex Smola. Parallelized stochastic gradient descent. Advances in neural information processing systems, 23, 2010.

### A Wall-Clock Time Model

In this section, we present an idealized model for the wall-clock times of Data-Parallel and DiLoCo. We measure the total elapsed time, which means that parallelization (e.g. via increasing the batch size) reduces wall-clock time.

#### A.1 Computation Time

Here, we mean the time expended by floating point operations in model training, ignoring communication time across nodes (which we treat separately in the section below). We use the idealized model where the total FLOPs C = 6ND. Given some number of chips R, each of which can perform Q floating point operations per second, the total computation time is bounded below by C/RQ. The number of chips R is purely a function of model size N and global batch size B. The number of chips does not depend on the algorithm (Data-Parallel or DiLoCo) or number of model replicas when using DiLoCo.

#### A.2 Communication Time

The network connectivity is characterized by a bandwidth W and latency ϵ. When performing an all-reduce of N parameters over R compute nodes, the lower bound on the amount of traffic sent and received by at least one of the compute nodes participating in the all-reduce is 2N(1−R−1) [Patarasuk and Yuan, 2009]. Such algorithms are called bandwidth-optimal. Since communication across the nodes is done synchronously but in parallel, in a network with bandwidth W and latency ε between each pair of nodes, the time to complete the all-reduce is at least

2N W

1 R

1 −

+ ε.

DiLoCo [Douillard et al., 2023] was designed for settings where models are too large to fit in a single datacenter, so they must be trained across compute islands connected by low bandwidth. To model this, we will assume that we are training over R compute nodes (typically, GPUs or TPUs). Some of these are connected by networks within a datacenter, and others are connected across datacenters. We let W0,ε0 denote the bandwidth and latency of the within-datacenter network, and W1,ε1 analogously defined for the cross-datacenter network. Typically, W0 ≥ W1,ε0 ≤ ε1.

Data-Parallel: At every training step T, we have to perform an all-reduce over all R compute nodes. Since some nodes are connected across datacenters, the total communication time is at least

2N W1

1 R

1 −

+ ε1 T

DiLoCo, M = 1: At every inner step T, we perform an all-reduce over all R devices as in Data-Parallel training. We also do an all-reduce every H steps for the outer optimization. Some of these nodes are connected across datacenters, so the communication time per all-reduce is at least 2N(1 − R−1)W1−1 + ε1. The total communication time is therefore at least

2N W1

1 R

1 H

1 −

+ ε1 T 1 +

.

DiLoCo, M ≥ 2: We assume that each of the M model replicas is trained on compute nodes connected within the same datacenter. In each inner step T, each model replica is trained by R/M devices which need to do an all-reduce. However, no communication occurs between datacenters, so the communication time of each inner step is bounded by 2N(1 − MR−1)W0−1 + ε0. Each outer optimization step involves all-reducing over all R devices, connected across datacenters. This incurs a communication time of at least 2N(1 − R−1)W1−1 + ε1. Since it occurs only every H steps, the total communication time is bounded below by:

2N W0

M R

1 −

+ ε0 T +

2N W1

1 R

1 −

+ ε1

T H

.

Note that this suggests that as long as H ≥ W0/W1, the outer communication steps incur at most half of the total communication cost.

Streaming DiLoCo. We note that the computed cost above applies to the Streaming DiLoCo [Douillard et al., 2025] as well. While the inner step remains the same, the outer step is smoothed such that each fragment p ∈ {1,...,P} is every H steps. However, fragment communication is offset such that some fragment is communicated every H/P steps, resulting in the communication amortizing to the same per-step cost. This is expected as Streaming DiLoCo reduces peak communication over any step, but does not reduce total communication across training.

Overlapping communications. Another contribution of Douillard et al. [2025] is the ability to overlap communications required for the outer optimizer with computation by using a stale version of the fragment in the outer optimizer, and merging the result of this outer optimization with the locally

optimized fragment. This would allow, for example, the communication term to be omitted from the calculation for wall-clock-time, if computation time dominates communication time. This setting is different from an algorithmic perspective, so its impact on scaling would need to be examined independently.

#### A.3 Total Wall-Clock Time

The total wall-clock time is a sum of the computation and communication times above. To measure the communication time, we must know the number of chips R used for each experiment, the number of FLOPs per chip per second Q, the bandwidth and the latency of the within-datacenter and cross-datacenter networks. For computation costs, we use a slightly idealized number of chips R based on our experiments, but ensuring that doubling the global batch size would double the number of chips. We base the constant Q on publicly available information about the FLOPs capabilities of the TPU v5e and v6e chips5, which have peak compute per chip (in bfloat16) of 197 teraflops and 918 teraflops, respectively. Assuming a maximum FLOPs usage of 50%, these chips have an actual compute of approximately 100 and 408 teraflops, respectively. When computing idealized compute time, we set Q = 300 teraflops, somewhere in-between the two.

For bandwidth and latency, we consider three archetypes of networks:

- 1. High-bandwidth network with bandwidth Whigh = 400 gigabits per second and a latency of εhigh = 10−4 seconds.
- 2. Medium-bandwidth network with bandwidth Wmed = 100 gigabits per second and a latency of εmed = 10−3 seconds.
- 3. Low-bandwidth network with bandwidth Wlow = 10 gigabits per second and a latency of εlow = 10−2 seconds.

We stress that these are not based on any actual systems, and are simply designed as instructive archetypes of networks. For the idealized communication time, we always use the high-bandwidth network for the within-datacenter network, and one of the three for the cross-datacenter network.

5See https://cloud.google.com/tpu/docs/v6e.

### B Additional Experimental Results

Model Scale = 35M

Model Scale = 90M

Model Scale = 180M

Model Scale = 335M

Model Scale = 550M

Model Scale = 1.3B

Model Scale = 2.4B

3.80

3.06

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

3.35

2.95

2.38

2.85

3.75

3.04

2.50

Algorithm

3.30

3.70

2.90

2.80

3.02

2.36

EvalLoss

Data Parallel

3.65

- DiLoCo, M=1

- DiLoCo, M=2

3.00

2.75

3.25

2.48

2.85

3.60

2.34

2.98

DiLoCo, M=4 DiLoCo, M=8

2.70

3.55

3.20

2.96

2.46

2.80

3.50

2.32

2.65

2.94

214 215 216 217

215 216 217 218

216 217 218 219

216 217 218 219 220 221

215 216 217 218 219 220 221

216 217 218 219 220 221

216 217 218 219 220 221

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

###### Figure 14: Evaluation loss of Data-Parallel and DiLoCo as a function of global batch size (in tokens).

Model Scale = 35M

Model Scale = 90M

Model Scale = 180M

Model Scale = 335M

Model Scale = 550M

Model Scale = 1.3B

Model Scale = 2.4B

0.48

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.272

0.305

0.55

0.252

0.34

0.38

HellaSwagZeroshot

0.270

0.47

Algorithm

0.300

0.54

0.33

0.268

Data Parallel

0.250

0.36

0.46

- DiLoCo, M=1

- DiLoCo, M=2

0.53

0.295

0.266

0.32

0.45

0.34

DiLoCo, M=4 DiLoCo, M=8

0.248

0.52

0.264

0.290

0.31

0.262

0.44

0.51

0.32

0.285

0.246

0.30

214 215 216 217

215 216 217 218

216 217 218 219

216 217 218 219 220 221

215 216 217 218 219 220 221

216 217 218 219 220 221

216 217 218 219 220 221

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

###### Figure 15: Zero-shot evaluation accuracy on HellaSwag of Data-Parallel and DiLoCo as a function of global batch size (in tokens).

Model Scale = 35M

Model Scale = 90M

Model Scale = 180M

Model Scale = 335M

Model Scale = 550M

Model Scale = 1.3B

Model Scale = 2.4B

0.660

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.640

0.590

0.715

0.695

0.655

0.610

0.67

0.635

PIQAZeroshot

Algorithm

0.585

0.650

0.690

0.710

0.630

Data Parallel

0.605

0.66

- DiLoCo, M=1

- DiLoCo, M=2

0.645

0.580

0.625

0.685

0.600

0.705

DiLoCo, M=4 DiLoCo, M=8

0.640

0.65

0.620

0.575

0.680

0.595

0.700

0.635

0.615

0.570

214 215 216 217

215 216 217 218

216 217 218 219

216 217 218 219 220 221

215 216 217 218 219 220 221

216 217 218 219 220 221

216 217 218 219 220 221

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

###### Figure 16: Zero-shot evaluation accuracy on Piqa of Data-Parallel and DiLoCo as a function of global batch size (in tokens).

Model Scale = 35M

Model Scale = 90M

Model Scale = 180M

Model Scale = 335M

Model Scale = 550M

Model Scale = 1.3B

Model Scale = 2.4B

0.345

0.330

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.43

0.38

0.39

0.37

0.340

0.325

0.41

ARCEasyZeroshot

Algorithm

0.42

0.335

0.320

0.38

0.36

0.37

Data Parallel

0.40

0.330

0.315

- DiLoCo, M=1

- DiLoCo, M=2

0.41

0.35

0.37

0.36

0.310

0.325

DiLoCo, M=4 DiLoCo, M=8

0.39

0.305

0.320

0.40

0.34

0.36

0.35

0.300

0.38

0.315

214 215 216 217

215 216 217 218

216 217 218 219

216 217 218 219 220 221

215 216 217 218 219 220 221

216 217 218 219 220 221

216 217 218 219 220 221

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Global Batch Size

Figure 17: Zero-shot evaluation accuracy on Arc-Easy of Data-Parallel and DiLoCo as a function of global batch size (in tokens).

In this section, we give additional experimental results that expand on those in Section 4. In Figures 14, 15, 16, and 17, we present evaluation loss and evaluation accuracy on various downstream zero-shot tasks, as a function of algorithm, model size, and global batch size. The results consistently show that Data-Parallel’s evaluation performance degrades quickly as batch size increases. By contrast DiLoCo’s performance degrades more slowly, or even improves, as batch size increases. We note that Arc-Easy was quite noisy as an evaluation task.

In Figures 18 and 19, we compare Data-Parallel and DiLoCo with M = 1 in terms of their evaluation loss and zero-shot evaluation accuracy on HellaSwag, Piqa and Arc-Easy. As above, we note that DiLoCo with M = 1 has an improved tolerance to larger batch sizes.

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
| | | | | | |
| | | | | | |
| | | | | | |

2.9

2.8

Model Scale

2.7

335M 550M

EvalLoss

- 1.3B

- 2.4B

2.6

Algorithm

Data Parallel DiLoCo, M=1

2.5

2.4

2.3

216 217 218 219 220

Global Batch Size

(a) Evaluation loss.

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

0.55

0.50

HellaSwagZeroshot

Model Scale

335M 550M

0.45

- 1.3B

- 2.4B

0.40

Algorithm

Data Parallel DiLoCo, M=1

0.35

0.30

216 217 218 219 220

Global Batch Size

(b) Zero-shot accuracy on HellaSwag.

Figure 18: Evaluation loss and zero-shot accuracy of Data-Parallel and DiLoCo with M = 1 for varying model and global batch sizes (measured in tokens). In all settings, DiLoCo with M = 1 does better than Data-Parallel, and the gap between them increases with batch size.

0.72

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.70

Model Scale

PIQAZeroshot

335M 550M

0.68

1.3B 2.4B

Algorithm

0.66

Data Parallel DiLoCo, M=1

0.64

216 217 218 219 220

Global Batch Size

(a) Zero-shot accuracy on Arc-Easy.

0.42

ARCEasyZeroshot

Model Scale

0.40

335M 550M

- 1.3B

- 2.4B

0.38

Algorithm

Data Parallel DiLoCo, M=1

0.36

216 217 218 219 220

Global Batch Size

(b) Zero-shot accuracy on Arc-Easy.

Figure 19: Zero-shot evaluation accuracy of Data-Parallel and DiLoCo with M = 1 for varying model and global batch sizes (measured in tokens), on Piqa and Arc-Easy. In nearly all settings, DiLoCo with M = 1 does better than Data-Parallel, and the often the gap increases with batch size.

