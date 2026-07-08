# arXiv:2501.18512v1[cs.CL]30Jan2025

[Figure 1]

## Streaming DiLoCo with overlapping communication:

### Towards a Distributed Free Lunch

[Figure 2]

###### Arthur Douillard*,1, Yanislav Donchev*,1, Keith Rush2, Satyen Kale†,2, Zachary Charles2, Zachary Garrett2, Gabriel Teston3, Dave Lacey1, Ross McIlroy1, Jiajun Shen1, Alexandre Ramé1, Arthur Szlam1, Marc’Aurelio Ranzato1 and Paul Barham1

1Google DeepMind, 2Google Research, 3Google, *Equal core contributions, †Currently at Apple.

Training of large language models (LLMs) is typically distributed across a large number of accelerators to reduce training time. Since internal states and parameter gradients need to be exchanged at each and every single gradient step, all devices need to be co-located using low-latency high-bandwidth communication links to support the required high volume of exchanged bits. Recently, distributed algorithms like DiLoCo (Douillard et al., 2024a) have relaxed such co-location constraint: accelerators can be grouped into “workers”, where synchronizations between workers only occur infrequently. This in turn means that workers can afford being connected by lower bandwidth communication links without affecting learning quality. However, in these methods, communication across workers still requires the same peak bandwidth as before, as the synchronizations require all parameters to be exchanged across all workers. In this paper, we improve DiLoCo in three ways. First, we synchronize only subsets of parameters in sequence, rather than all at once, which greatly reduces peak bandwidth. Second, we allow workers to continue training while synchronizing, which decreases wall clock time. Third, we quantize the data exchanged by workers, which further reduces bandwidth across workers. By properly combining these modifications, we show experimentally that we can distribute training of billion-scale parameters and reach similar quality as before, but reducing required bandwidth by two orders of magnitude.

Keywords: large-scale, language modeling, distributed learning

#### 1. Introduction

Scaling deep learning has led to significant leaps in capability (Grattafiori et al., 2024; OpenAI et al., 2024; Team et al., 2024). Although neural architectures have evolved over the past decade, the standard approach to optimization remains essentially unchanged from the days of Alexnet (Krizhevsky et al., 2012). Practitioners use minibatch stochastic gradient descent, with backpropagation through the model’s layers to compute the gradients. As in Alexnet, which already combined two hardware accelerators for parallel training, models are trained with multiple hardware accelerators.

However, modern training runs, for example for large language models (LLM), may use tens of thousands of accelerators, and this number increases year after year. Building and maintaining a data-center that can co-locate that many

accelerators is expensive and leads to increasingly complex engineering challenges. Beyond the physical infrastructure, orchestrating the passage of gradients, parameters and intermediate states between these devices at each optimization step, while keeping all devices fully utilized is technically challenging from a software engineering perspective. Furthermore, the more devices that are used for each synchronous training step, the more chances there are that one of them fails, risking halting training, or introducing subtle numerical issues.

Recent publications (Douillard et al., 2024a; Jaghouar et al., 2024a), building on work by McMahan et al. (2017), have demonstrated that the co-location requirements of all accelerators can be loosened. These methods allow highly-performant training when accelerators are grouped into several “workers” with fast band-

Corresponding author(s): douillard@google.com © 2025 Google DeepMind. All rights reserved

width intra-worker but with slow bandwidth interworkers. The basic approach is to allow each worker to continue training for many minibatches, independently of the other workers; and then synchronize the parameters of the workers after a set number of these “inner” steps. The synchronization, in its simplest form (McMahan et al., 2017), is to average the parameters of the workers (Wortsman et al., 2022); but more sophisticated methods (Huo et al., 2020; Reddi et al., 2021) use the workers’ parameters to form a pseudogradient to update the shared parameters. The details of the formulation of the weight synchronization is important for machine-learning efficiency; see Section 2.1.

However, in these approaches, the synchronization typically requires an all-reduce operation which fully synchronizes the model parameters on some step. This all-reduce results in two main issues: 1) a large peak bandwidth, and 2) a blocking of the workers while they wait to receive updated weights. In this work, dubbed Streaming DiLoCo, we propose three modifications to these approaches to practically reduce the peak bandwidth and mitigate worker-blocking without loss of learning efficiency:

- Contribution 1: Synchronization. We synchronize subsets of parameters on a schedule, rather than all parameters at once. This contribution reduces the peak required bandwidth.

- Contribution 2: Overlapping. We overlap worker computation and communication of synchronizations. This contribution increases the tolerated latency of communication.

- Contribution 3: Quantization. We compress the outer gradients to four bits per parameters without loss of performance. This contribution reduces the total amount of bits exchanged.

We show experimentally that our model, Streaming DiLoCo, is strictly superior to the original DiLoCo (Douillard et al., 2024a), and achieves

similar performance to the bandwidth-costly dataparallelism. Since we attain the same quality at negligible bandwidth, we consider our approach as an important stepping stone towards a form of distributed free lunch.

#### 2. Model

For all algorithms, we denote the model parameters as 𝜃. We use the superscript notation 𝜃(𝑡) to indicate the parameters at a given step 𝑡, and the subscript notation 𝜃𝑚 to denote a particular shard of the DiLoCo replica. For example, 𝜃𝑚(𝑡) indicates the parameters of DiLoCo replica 𝑚 at step 𝑡. If no subscript is used, the parameters are replicated across DiLoCo replicas. Note that it is possible for parameters to not be replicated and yet to be of the same value.

Algorithm 1 FedOpt / DiLoCo Require: 𝑀 replicas Require: Synchronization frequency 𝐻 Require: Model replicas {𝜃1(𝑡−1), . . . , 𝜃𝑀(𝑡−1)} Require: Data shards {D1, . . . , D𝑀} Require: Optimizers InnerOpt and OuterOpt

- 1: parallel for replica 𝑚 = 1. . . 𝑀 do
- 2: for step 𝑡 = 1. . .𝑇 do
- 3: 𝑥 ∼ D𝑚
- 4: L ← 𝑓 (𝑥, 𝜃𝑚(𝑡−1))
- 5: 𝜃𝑚(𝑡) ← InnerOpt(𝜃𝑚(𝑡−1),∇L)
- 6: if 𝑡 mod 𝐻 == 0 then
- 7: Δ𝑚(𝑡) ← 𝜃𝑚(𝑡−𝐻) − 𝜃𝑚(𝑡)
- 8: Δ(𝑡) ← async-send[𝑀1

𝑀 𝑚=1(Δ𝑚(𝑡))]

- 9: block-receive[Δ(𝑡)]
- 10: 𝜃𝑚(𝑡) ← OuterOpt(𝜃𝑚(𝑡−𝐻), Δ(𝑡))
- 11: end parallel for

###### 2.1. Context: FedOpt and DiLoCo

FedOpt (Reddi et al., 2021) is a generic framework to perform federated learning with a bi-level optimization. 𝑀 local replicas perform 𝐻 steps of inner independent optimizations on a different subset of the data (L3 to L5 in Algorithm 1). Every 𝐻 steps, each replica computes an outer gradient Δ𝑚𝑡 = 𝜃𝑚(𝑡−𝐻) − 𝜃𝑚(𝑡) (L7), a delta in the parameters space, and communicates to all other

replicas. This communication can be performed through a central parameter server or through direct communication of each worker to the others (e.g. with a ring all-reduce), and results in each worker obtaining Δ𝑡 = 1/𝑀

𝑀 𝑚=1 Δ𝑚𝑡 (L7-9). This outer gradient is applied on a set of outer parameters, the previously synchronized parameters 𝜃𝑚(𝑡−𝐻), with an outer optimizer (L10). The full algorithm is shown in Algorithm 1.

The costly communication between noncolocated devices happens during the averaging of outer gradients, in L8-9 of Algorithm 1. It is as costly as in Data-Parallel, but instead of being executed at every step, it is done every 𝐻 (e.g. one hundred) steps, thus amortizing the communication cost.

DiLoCo is a successful instantiation of FedOpt applied to language models where the inner optimizer is Adam (Kingma and Ba, 2014) and the outer optimizer is SGD with Nesterov momentum (Sutskever et al., 2013). In this work, which focuses on distributed optimization, and unlike in the federated learning literature (as discussed in FedOpt), the workers aren’t sampled; but instead all workers will be present at each step.

###### 2.2. Streaming partial updates

Instead of communicating the full outer gradient vector (Δ𝑚(𝑡),∀𝑚 ∈ {1, ...𝑀}) every 𝐻 steps, we propose to share only a fragment 𝑝 of it (Δ𝑚,𝑝(𝑡) ) more frequently, as highlighted in Figure 1. There is a huge possible space of choices for these fragments and specification of “more frequently”1; here we consider the simple partition of our network into 𝑃 fragments made of several transformer blocks. Specifically, we study two fragment patterns, as shown in Figure 2: 1) sequential where each fragment comprises consecutive transformer blocks and 2) strided where each fragment is composed of interleaved transformer blocks. We will demonstrate in subsection 3.3 that the algorithm is robust to the particular choice of fragments. Since the stride version offers in practice slightly better compute utilization (less

1For example, an extreme version might be to send a constant bitstream of random choices (according to some optimization-useful distribution) of parameters.

time spent communicating instead of computing), we will use it as the default choice in our experiments. As we increase model scale, the fragment definition (e.g., how many transformer blocks comprise a fragment) is maintained, which means that larger models have more fragments.

The resulting algorithm in shown is Algorithm 2 (and contrasted with the original version shown in Algorithm 1), where only a fragment 𝑝 of the replica 𝑚 is shared. We denote a fragment with a new lower script, thus 𝜃𝑚,𝑝(𝑡) is the parameters of fragment 𝑝 of the replica 𝑚 at step time 𝑡.

The Streaming DiLoCo’s inner optimization (L3-5 of Algorithm 2) is identical to DiLoCo’s (L35 of Algorithm 1). However, the outer optimization (L12) is done per fragment. If a fragment 𝑝 satisfies the condition 𝑡+𝑡𝑝 mod 𝐻 = 0, where 𝑡𝑝 is a time offset fragment-dependent, then it is synchronized. In this way, each fragment will always do 𝐻 steps before being synchronized, but overall the model is synchronizing some fragment more frequently than every 𝐻 steps. For example, with 𝐻 = 100 and 𝑃 = 2 fragments, the first fragment will be synchronized at step 𝑡 = 100, 𝑡 = 200, ... (𝑡𝑝=1 = 0); the second fragment will be synchronized at step 𝑡 = 150, 𝑡 = 250, ... (𝑡𝑝=2 = 50). While in practice, given an equal 𝐻, streaming DiLoCo communicates more often than DiLoCo, the peak communication is reduced by a factor of |𝑝|/𝐿 with |𝑝| the size of a fragment in layers and 𝐿 the total number of layers.

2.3. Overlapping communication with computation

To further maximize the time spent doing computation v.s. communication, we propose to overlap the communication of the outer gradient fragment with the inner optimization computation; the overlap happens with a strictly positive 𝜏 in Algorithm 2, lines 10-12. At the beginning of outer step 𝑡 + 1, instead of waiting for the communication of the fragment block-receive, we immediately start the new round of optimization. After 𝜏 − 1 inner steps (L3-5), we block-wait for the exchanged fragment (L10), apply the outer optimizer on the previously synchronized frag-

ment (𝜃𝑚,𝑝(𝑡−𝜏(𝑡−−𝜏𝐻))), and merge it with the currently

t = 1

t = 2

Parallel Inner Optimization

Parallel Inner Optimization

[Figure 3]

[Figure 4]

Model randomly initialized

…

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Outer Optimization

Outer Optimization

[Figure 14]

[Figure 15]

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

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

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

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

- Figure 1 | Streaming DiLoCo: each replica trains independently for dozen of inner optimization steps, and then synchronize a single fragment during outer optimization. In this figure, there are 𝑀 = 4 replicas with 𝑝 = {1, 2, 3} fragments. Each fragment can be made of several transformer layers. Note that this figure only showcases the streaming partial updates (subsection 2.2) and not the quantized communication overlapping (subsection 2.3 and 2.4).

28

Embeddings

- Layer 1

- Layer 2

- Layer 3

- Layer 4

- Layer 5

- Layer 6

- Layer 7

- Layer 8

Embeddings

- Layer 1

- Layer 2

- Layer 3

- Layer 4

- Layer 5

- Layer 6

- Layer 7

- Layer 8 Sequential Strided

time

Layers

- Figure 2 | Streaming pattern: sequential (left) and strided (right). Colors denotes the fragment. A different fragment is synchronized each time.

Algorithm 2 Streaming DiloCo Require: 𝑀 replicas Require: Number of inner steps 𝐻 Require: Fragments 𝑝 ∈ {1, . . . , 𝑃} with their

respective synchronization offset 𝑡𝑝

Require: Model replicas {𝜃1(𝑡), . . . , 𝜃𝑀(𝑡)} Require: Inner overlap delay 𝜏 < 𝐻

Require: Data shards {D1, . . . , D𝑀} Require: Optimizers InnerOpt and OuterOpt

- 1: parallel for replica 𝑚 = 1. . . 𝑀 do
- 2: for step 𝑡 = 1. . .𝑇 do
- 3: 𝑥 ∼ D𝑚
- 4: L ← 𝑓 (𝑥, 𝜃𝑚(𝑡−1))
- 5: 𝜃𝑚(𝑡) ← InnerOpt(𝜃𝑚(𝑡−1),∇L)
- 6: if ∃𝑝 s.t. 𝑡 − 𝑡𝑝 mod 𝐻 == 0 then
- 7: Δ𝑚,𝑝(𝑡) ← 𝜃𝑚,𝑝(𝑡−𝐻) − 𝜃𝑚,𝑝
- 8: Δ𝑝(𝑡) ← async-send[𝑀1

𝑀 𝑚=1(Δ𝑚,𝑝(𝑡) )]

- 9: if ∃𝑝 s.t. 𝑡 − 𝑡𝑝 − 𝜏 mod 𝐻 == 0 then
- 10: block-receive[Δ𝑝(𝑡−𝜏)]
- 11: 𝜃˜𝑚,𝑝(𝑡) ← OuterOpt(𝜃𝑚,𝑝(𝑡−𝜏−𝐻), Δ𝑝(𝑡−𝜏))
- 12: 𝜃𝑚,𝑝(𝑡) ← 𝛼𝜃𝑚,𝑝(𝑡) + (1 − 𝛼)𝜃˜𝑚,𝑝(𝑡)
- 13: end parallel for

optimized fragment with a mixing factor 𝛼. 𝛼 = 1 is equivalent to no communication between replicas, 𝛼 = 0 discards any updates done in the first 𝜏 steps on the fragment 𝑝, and 𝛼 = 0.5 does a uniform average between the local fragment parameters and the globally shared one.

###### 2.4. Low-precision outer gradients

The previous methods, streaming and overlapping communication with computation, reduce peak bandwidth and wall-clock time, respectively.

To reduce total amount of bits exchanged, we use lower-precision in the outer gradients exchanged by workers (while still using FP32 for computing gradients), up to a float with 4 bits (1 sign bit, 3 exponent bits, and 0 mantissa bit) called E3M0 (Agrawal et al., 2024). Across a wide variety of experiments, we found no sign of performance regression when employing such low precision numbers during communication, even at the billion scale. This compression is applied when sending each replica’s unreduced outer gradients to miminize the amount bits communicated (L8 of Algorithm 2), but once received by a replica, importantly, the accumulation is done in FP32 for stability.

###### 2.5. Discussion on the memory overhead

In an SPMD2 model, the memory overhead of the Data-Parallel baseline is the parameters (1×) + Adam state (2×). (Streaming or not) DiLoCo’s memory overhead is the parameters (1×), the Adam state (2×), the outer global parameters (1×), and the outer Nesterov state (1×). Thus, our method requires 66% (5/3) more memory compared to Data-Parallel. However, in the case of Streaming DiLoCo, only a subset of the outer parameters and outer optimizer state is needed at a given time. Therefore, this overhead can be alleviated by offloading the additional bits onto CPU memory (Beaumont et al., 2022). The memory overhead to hold in HBM at any point in time is the size of a fragment |𝑝| times two, for the outer parameters and outer optimizer state. For a 100 billion parameter model for instance, with |𝑝| = three layers and with a total of 108 layers, that amounts to a 2% increase of memory (additional 20 GB to 1,117 GB3). This extra memory is used when there are no activations or gradients in live memory, and thus should fit in HBM without any problem.

Furthermore, the communication schedule is deterministic and known prior to training. Thus,

- 2https://en.wikipedia.org/wiki/

Single_program,_multiple_data

- 3The size in GB of the parameters & inner Adam opti-

mizer state is the number of parameters × 3 × 4 (FP32). The size in GB of the additional fragment and its outer Nesterov

optimizer state is the number of parameters × 2 × 4 ×1083 .

we can start the transfer from RAM to HBM of a fragment (and its associated outer optimizer state) while finishing the previous (inner) gradients passes. Given that only a small subset is required at a given time, the memory transfer cost is negligible. With an H100 with PCIe4, characterized by 2 TB/s of bandwidth speed, and without any sharding, this transfer is done in less than 10 milliseconds.

#### 3. Experiments

We run experiments demonstrating the compute utilization benefits of our approach in a a bandwidth and compute simulation in Section 3.1. In Section 3.2 we show in practice the model learning outcomes. Finally, in Section 3.3 we show results with variations of the three main contributions of the paper, ablating their relative importance.

###### 3.1. Compute utilization

To highlight the impact of our contributions in a controlled setting, we built a simulator to estimate the compute utilization of each method: how much time is spent doing computation v.s. communication. The simulation is a DAG with four different types of nodes as seen in Figure 3: forward in blue, backward w.r.t activations and parameters in green, (outer or not, for resp. DiLoCo and Data-Parallel) gradients reduction in purple. Each node represent a single layer. Therefore, the total number of nodes, for a single step, is 4×𝐿−1 (because we don’t need the backward w.r.t activations of the first layer). The overall training is represented by a DAG made of such nodes. We use this simulator to estimate the compute utilization of each method: how much time is spent doing computation v.s. communication. Ideally this number is close to 1.0: No time is spent waiting for communication. It is more useful than just reporting the reduction in data transferred because our overlapping method (subsection 2.3) reduces the latency while keeping the amount of data exchanged constant.

4https://www.nvidia.com/en-gb/data-center/ h100/

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

- Figure 3 | Simulation of a schedule interleaving forward passes (in blue), backward passes w.r.t. activations and parameters (resp. in light and dark green), and (outer) gradient reduction (in purple).

[Figure 57]

(a) 1B parameters model. (b) 10B parameters model (c) 100B parameters model

[Figure 58]

[Figure 59]

- Figure 4 | Compute Utilization simulated across a range of bandwidth. A compute utilization of 0.8 means 80% of the time is spent in computation, and 20% in communication. Our best method reaches a compute utilization of 95% for models 1B, 10B, and 100B with a bandwidth roughly constant between 1 and 5 Gbit/s. Data-Parallel on the other hand requires 100, 200, and 300Gbit/s.

In Figure 3, we report:

- • Data-Parallel: the baseline which communicates gradients of the full model at every step;
- • DiLoCo: which communicating outer gradients of the full model once in a while (in this example, every 𝐻 = 2 steps);
- • Streaming DiLoCo: which communicates outer gradients only for a subset of the model (here the fragment size is a single layer and there are two fragments) every 𝐻 = 2 steps;
- • Streaming DiLoCo with overlapping communication and computation: This is similar to the above but gradients sent across workers are only needed after 𝜏 steps (in this example 𝜏 = 1).

The simulated compute utilization (CU) depends on some factors, listed as columns in Table 4. For the model scales 10B and 100B, we estimate step time (pure compute) based on the flops profile, a reasonable MFU (40%), and hardware theoretical flops per seconds. We simulate training of each method, across three scales (1B, 10B, and 100B) under various bandwidth profiles Figure 4.

We make several observations: 1) Streaming DiLoCo (in green) improves the CU of DiLoCo (in orange) despite exchanging as much data, because it reduces the latency by splitting the communication of the outer gradients across fragments. 2) only overlapping communication with computation can reach full 100% compute utilization. 3) the required bandwidth can become

lower as the model scale gets larger when overlapping communication with computation, because longer compute step time (forward & backward) will provide more time to perform the synchronization across workers.

The last point may seem counter-intuitive at first glance, but is the main advantage of our method, exploiting the square-cube law of distributed training (Ryabinin et al., 2023) where computation scales worse than communication (𝑂(𝑛3) vs 𝑂(𝑛2) for a square matrix 𝑛 × 𝑛). We provide in the appendix, in Figure 15, the simulated compute utilization for a 100B model across various compute step time.

Remark. Of course this is only a simulation of what we expect to happen in practice. Such simulation is not perfect because for instance we consider only the bandwidth between datacenters and not the local bandwidth between devices. We believe however, that this is still a useful tool5 to estimate device utilization.

###### 3.2. LLM Scaling Experiments

We perform our experiments with a Chinchilla architecture (Hoffmann et al., 2022). Following Wortsman et al. (2023) and Jaghouar et al. (2024a), we use QKNorm (Henry et al., 2020) and a Z-loss (Chowdhery et al., 2023) with a factor of 1𝑒 − 4 to stabilize training. We report in Table 2 the architecture hyperparameters and token budget at each scale. Unlike recommended in Post-Local SGD (Lin et al., 2020), we train all our models from scratch. The main hyperparameter of DiLoCo is its outer learning rate; we tuned it to be optimal at small scale at 0.4, and kept it fixed across all scales. Likewise, for the simplicity, and to show that Streaming DiLoCo is a drop-in replacement of DiLoCo, we used the same outer learning rate, without further hyperparameters tuning.

Except mentioned otherwise, we use the C4 dataset (Raffel et al., 2020) and train models from 35 million to 4 billion parameters, all with a sequence length of 1,024. Each scale is trained with the chinchilla-optimal number of steps. We use 2 DiLoCo replicas, each of them performing

5https://en.wikipedia.org/wiki/Bonini% 27s_paradox

FSDP (Zhao et al., 2023) across their respective closely located devices.

For training we use a modified version of the NanoDO codebase (Liu et al., 2024b) that uses DrJax (Rush et al., 2024) to parallelize inner steps across replicas. The inner optimization is done with an annotated variant of jax.vmap for the optimization step, with parameters having an extra leading axis for the DiLoCo replicas. The outer optimization is implemented with an allreduce, without any central parameter server.

###### 3.2.1. Scaling

We perform scaling experiments on C4, with models ranging from 35 millions parameters to 1 billion parameters, all with a sequence length of 1,024. For Data-Parallel and Streaming DiLoCo with 𝐻 = 100, we also provide results on a 4 billion parameter model. At each scale, we use the Chinchilla-optimal (Hoffmann et al., 2022) number of steps. We highlight in Figure 5 the evaluation loss (lower is better) and HellaSwag (Zellers et al., 2019) accuracy (higher is better).

First, we observe in Figure 5 that Data-Parallel (in blue), DiLoCo with 𝐻 = 30 inner steps (in orange), and Streaming DiLoCo with 𝐻 = 30 (green) perform all similarly across both loss (at 1B parameters, respectively 2.49, 2.49, and

- 2.48) and accuracy metrics (resp. 46.6%, 46.5%, and 46.6%). Streaming DiLoCo with more inner steps 𝐻 = 100 (in red) has slightly worse performance initially but use significantly less bandwidth and the loss improves proportionally better as we scale: scaling law slope for Data-Parallel is

-0.13149 while -0.13539 for Streaming DiLoCo. We report in the appendix Table 5 all metrics, and include two more downstream tasks: Piqa (Bisk et al., 2020) and Arc-Easy (Clark et al., 2018). Moreover Table 6 considers an increased number of DiLoCo replicas.

- 3.2.2. Overtraining on Dolma

The previous experiments were performed on C4 dataset using the chinchilla-optimal number of tokens. Using a 1 billion parameter model, this yields a token budget of 25 billion. However, language models are now usually overtrained (Gadre

[Figure 60]

[Figure 61]

(a) Evaluation loss on C4 (b) HellaSwag accuracy

- Figure 5 | Scaling models from 35M (1.49e17 flops) to 4B parameters (2e21 flops) on C4.

Method Token Budget Hours spent w/ +∞ Gbits/s Hours spent w/ 1 Gbits/s Terabytes exchanged Eval Loss ↓ HellaSwag ↑ Piqa ↑ Arc Easy ↑

25B 0.67 109 441 2.67 42.09 67.35 40.42 100B 2.7 438 1,767 2.52 49.78 69.15 44.03 250B 6.75 1097 4,418 2.45 53.86 70.45 44.21

Data-Parallel

25B 0.67 0.88 1.10 2.66 42.08 67.46 38.42 100B 2.7 3.5 4.42 2.51 49.98 69.96 44.03 250B 6.75 8.75 11.05 2.45 54.24 71.38 41.92

Streaming DiLoCo with overlapped FP4 communication

Table 1 | Overtraining Data-Parallel and our method on Dolma with a 1 billion parameters model. The latter performs slightly better despite exchanging in total 400× fewer bits, reducing the peak bandwidth by 8×, and with a significantly relaxed training communication latency constraint: allow communication to be as long as a full compute step.

et al., 2024). Therefore we perform a comparison of a Data-Parallel baseline vs our full model (streaming DiLoCo with overlapped FP4 communication) on the Dolma dataset (Soldaini et al., 2024) with a 1 billion parameter model and with a token budget of 25, 100, and 250 billions tokens (resp. 1.9e20, 7.6e20, and 1.9e21 flops) using a sequence length of 2,048. In that larger, more realistic setting, we set the number of inner steps between synchronization to 𝐻 = 100 to further minimize communication.

We report results in Table 1, and note that both our method and the baseline perform similarly w.r.t loss and accuracy on downstream tasks (HellaSwag, Piqa, Arc-Easy). In addition of being neutral in term of ML performance: 1) the amount of bits exchanged between non-colocated devices over the course of training is 400× higher for Data-Parallel; 2) the peak bandwidth (amount of

bits exchanged at given moment) is reduced by

num layers = 24

fragment size = 3 = 8×; and 3) while Data-Parallel ideally hopes for a 0 second latency when communicating, our overlapping scheme allows us a latency as long as a full forward/backward pass, which is several seconds at large scale. For those reasons, we believe our work is step towards a truly “distributed free lunch”.

###### 3.3. Ablations

To ablate the importance of each component of Streaming DiLoCo, we perform all our ablations on a model of size 500 million parameters using the C4 dataset with the chinchilla-optimal number of steps and a token budget of 11 billions.

We split our ablations section in three parts, corresponding to the three improvements brought in this paper: namely 1) Streaming synchronization

in section 3.3.1, 2) overlapping communication with computation in section 3.3.2, and 3) finally quantized communication in section 3.3.3.

###### 3.3.1. Ablating the streaming synchronization

In this ablation section, we consider different settings for streaming DiLoCo presented in subsection 2.2.

Number of synced layers per fragment. We ablate in Figure 6 the fragment size, i.e., how many transformer blocks are included in a fragment. Based on this analysis, we choose a fixed fragment size of 3 layers, striking a desirable tradeoff between ML performance and reduction of peak bandwidth (for which the smaller the fragments the better). We also consider whether to have a sequential or strided pattern (see illustration in Figure 2 for a reference). We choose the latter for several reasons: 1) ML performance is slightly better for the fragment size we consider, 2) deeper networks, with a small fragment size (e.g. 3), should benefit more from striding by spreading out up-to-date synchronized layers across the full depth of the network. Finally, 3) it slightly improves the compute utilization (see Figure 7) by allowing better overlapping schedule,

- as clearly seen in Figure 14 in the appendix.

[Figure 62]

(a) C4 eval loss

[Figure 63]

(b) Peak bandwidth reduction

Figure 6 | The fragment’s size will determine the peak bandwidth but also the learning dynamics. We choose in practice 3 layers per fragment across all model scales.

significant flop inefficiency, which might not be acceptable in training-compute bound settings (which are typical in current large scale training of LLMs).

Comparison to FedPart. Streaming DiLoCo bears similarity with concurrent work dubbed FedPart (Wang et al., 2024), where a subset of the model is also exchanged at each round. However, FedPart argues that non-shared layers should be be frozen during inner optimization. We believe this is rather flops-inefficient: For an 18 layer model, with 3 layers per fragment, 15 layers (83%) are frozen at any point in time despite doing forward/backward computation. We ran comparison of Streaming DiLoCo with and without the frozen pattern proposed by FedPart, reaching respectively on the C4 eval loss 3.2145 and 2.6749. Freezing the 18−3 = 15 layers that won’t be synchronized at the given round therefore results in a 20% increase of the evaluation loss. These results confirm our intuition that while freezing layers may help merging, this incurs a

###### 3.3.2. Ablating the communication overlap

In this ablation section, we investigate how to overlap communication with computation, see subsection 2.3 for reference.

Overlapping. We first vary the number of inner steps, 𝜏 we use to overlap communication with computation (see subsection 2.3). Figure 8 shows results varying 𝜏 from 1 to 20, with 𝛼 = 0 (discarding any intermediary inner updates) and 𝛼 = 0.5 (uniform merging). We can see that the degradation in evaluation loss is negligible up to an overlap of 10 inner steps (< 0.2%). By checking the compute utilization of a 100B model in Figure 9, we observe little gain in compute time after an overlap of 5 inner steps. Therefore, we advise practitioners to limit the overlap to 5 inner

[Figure 64]

- Figure 7 | Compute utilization profile of sequential vs strided pattern for a 100 billion parameters model.

[Figure 65]

- Figure 8 | Varying the number of overlapped inner steps 𝜏 for 𝛼 = {0, 0.5}. A larger 𝜏 requires a significantly lower bandwidth, see also Figure 9.

steps. In our main experiments, we used 1 step for simplicity.

Overlapping with some slack between workers. If workers use heterogeneous device types (e.g. TPUv5e vs TPUv6e) or are just placed in different environments, their execution speed might vary and it would be inefficient to force them to synchronize at the same optimization step. In this case, we could grant workers with some slack. For instance, in a 2 DiLoCo replicas setting, both workers could send their respective outer gradients at the same step (but not necessarily at the same time) and they could receive the update

- at a different step (Liu et al., 2024a). We accomplish this by simply using a different 𝜏 per

worker (𝜏1 and 𝜏2) as shown in line 7-9 in Algorithm 2. We show in Figure 10 the evaluation loss when 𝜏1 = 1 and varying 𝜏2. Similarly to

[Figure 66]

- Figure 9 | Estimated compute utilization for a 100B model when increasing 𝜏, the number of inner steps which overlap with communication.

[Figure 67]

- Figure 10 | Varying the number of overlapped inner steps 𝜏2 for the second worker while keeping 𝜏1 = 1. For all data points, 𝛼 = 0.5. Training is very robust for values of 𝜏2 less than 5.

Figure 8, the loss degradation is limited under a delay of up to 5 inner steps. This result suggests that Streaming DiLoCo is rather robust and could support training of large models on several heterogeneous workers.

3.3.3. Ablating the quantized communication Finally, in this section, we consider various schemes to quantize our communication, as proposed in subsection 2.4.

Compressing outer gradients. We ablate in

- Figure 11 two ways of compressing the outer gradients: either by setting to zero some values (FedDropout (Wen et al., 2022), Dare (Yu et al., 2024), and Top-K selection) or by lowering the precision. In all cases, we accumulate the outer

[Figure 68]

- (a) C4 evaluation loss

[Figure 69]

- (b) HellaSwag accuracy

Figure 11 | Compressing the outer gradients with either value dropping (FedDropout, Dare) or using lower-precision floating point numbers.

update in float32. We report both the evaluation loss on C4 and the accuracy on HellaSwag. Interestingly, lowering the precision, from float32 to float4 does not affect the performance, while setting some values to zero is significantly worse, particularly when zero-ing out at random. We also considered Ties-Merging’s pruning method (Yadav et al., 2023) but preliminary experiments showed it also underperformed; however this approach might become advantageous with larger number of replicas 𝑀.

#### 4. Related Works

Model merging. Model merging, a subfield within the broader study of linear mode connectivity, explores the potential of linearly interpolating between parameters of multiple models to synthesize a unified model that inherits the strengths of its constituents (Matena and Raffel,

2021; Wortsman et al., 2021). A key finding in this domain is the existence of low-loss pathways within the parameter space (Frankle et al., 2020; Neyshabur et al., 2020) that connect independently trained models, effectively circumventing the anticipated loss barriers. For instance, Wortsman et al. (2022) demonstrated that averaging the parameters of models fine-tuned from a common pre-trained initialization on diverse tasks (Ramé et al., 2023a) or with varying hyperparameters (Wortsman et al., 2022) yields a performant merged model. This approach, initially demonstrated in computer vision, has been successfully extended to natural language processing (Li et al., 2022), reinforcement learning with human feedback (Ramé et al., 2023b), noisy label learning (Rebuffi et al., 2022), and out-of-distribution generalization (Ramé et al., 2023c). Recent research has further investigated alternative strategies for mitigating loss barriers in model merging, including techniques based on parameter space transformations and other model surgery methods aiming to resolve merging conflicts (Ainsworth et al., 2023; Jin et al., 2023; Jordan et al., 2023; Stoica et al., 2023; Yadav et al., 2023; Yu et al., 2024).

Federated learning / local SGD. While model merging proposes to combine several models once, FedAvg (McMahan et al., 2017) and Local SGD (Stich, 2019) do it multiple times with the goal of reducing bandwidth requirements: they operate by performing local training (typically via SGD) across workers for some number of steps before doing some kind of synchronization of worker parameters, or aggregation of parameters across workers. In their original forms, both FedAvg and Local SGD simply averaged the parameters across workers. As shown by Reddi et al. (2021), the synchronization is more effective when each worker calculates a “model delta”, and these are aggregated over workers to produce a pseudo-gradient (Ilharco et al., 2022; Reddi et al., 2021) or outer gradient, which is then fed to a first-order optimizer. This yields a bi-level optimization scheme with inner optimizers and an outer optimizer, referred to as FedOpt by Reddi et al. (2021), who propose using SGD as the inner optimizer and adaptive methods like Adam (Kingma and Ba, 2014) as the outer opti-

mizer in resource-constrained FL settings.

Distributed training for LLMs. The increased requirements of training large language models (LLMs) hastened the need for distributed methods, for both inference (Borzunov et al., 2023) and training (Diskin et al., 2021; Presser, 2020; Ryabinin et al., 2021). More recently, DiLoCo

- (Douillard et al., 2024a) proposed a particular instantiation of FedOpt (Reddi et al., 2021) with AdamW (Loshchilov and Hutter, 2019) as inner optimizer and Nesterov (Sutskever et al., 2013) as outer optimizer (Huo et al., 2020). This simple formulation proved to be effective for distributed training with LLMs, where the number of replicas is small (<100) and without replica sampling, closer to cross-silo federated learning (Kairouz et al., 2021). The FedOpt algorithm was also shown to be effective in training LLMs in settings that looked more like cross-device federated learning (Charles et al., 2024). The empirical success of DiLoCo has been reproduced multiple times (Jaghouar et al., 2024b; Sani et al., 2024b) and has been successfully scaled up to 10 billion parameter models (Jaghouar et al., 2024a). Related, a simple change on how the outer Nesterov accumulates outer gradients proved to handle well asynchronicity between workers of different speeds (Liu et al., 2024a). DiLoCo adds a new axis of paralellism to distributed training (Shoeybi et al., 2020), and is compatible (Jaghouar et al., 2024a) with other existing axes like FSDP (Zhao et al., 2023), or even another level of federated learning (Sani et al., 2024a).

Partial communication. Communicating a subset of the network is often used in federated learning to provide personalized models per user, see FedPart (Arivazhagan et al., 2019). DiPaCo

- (Douillard et al., 2024b) recently proposed a distributed mixture-of-experts where subsets of the model is synchronized with subsets of the replicas, according to a sharing pattern that is optimized with Expectation-Maximization style of algorithm during training. WASH (Fournier et al., 2024) and later Sparta (Baioumy and Cheema, 2025) propose to frequently exchange a random subset of the neurons. Finally FedPart (Wang et al., 2024),

developed at the same time as Streaming DiLoCo, also proposes to share per-layer fragments. However, they argue that for a given communication round, non-shared fragments should not undergo inner optimization, a strategy which we show slows down convergence. Note that all partial communication methods can be seen as a form of structured (outer) gradients compression.

Gradient compression. Data-Parallel (with gradients) and Federated learning (with outer gradients) often share similar methods to compress the communication (Lin et al., 2018): from randomly dropping values (Wen et al., 2022), to combining multiple compression schemes (e.g. dropping, top-k, low-precision) (Wang et al., 2023), to use low-rank compression (Vogels et al., 2019; Zhao et al., 2024), or recently to keep only the fast moving components with DCT but communicates via an all-gather collective instead of an all-reduce (Peng et al., 2024).

#### 5. Conclusion and Future Work

In this paper, we introduced three improvements over DiLoCo: we synchronize a only subset of the parameters at a time, we overlap the communication of this synchronization over several computation steps, and we compress the outer gradients to communicate to low-precision with only four bits. All these innovations combined together leads to a training with similar ML-performance as a classical Data-Parallel training, while using 400× less bandwidth, reducing the peak bandwidth compared to DiLoCo’s bursts of communication, and allowing communication to have an ideal nonzero latency by overlapping it with computation.

In sum, we can reach a similar compute utilization as the widely used Data-Parallel using two orders of magnitude less Gbit/s bandwidth, while performing comparably in term of training loss and downstream evaluation accuracies as DataParallel. For those reasons, we claim that this work in a first step towards what we call a distributed free lunch, paving the way for a new way to train distributed networks with reduced bandwidth and yet without trading-off model quality.

Next. In our view, the ubiquity of co-located Data-Parallel training is likely due to the hardware lottery (Hooker, 2020), when “a research idea wins because it is suited to the available software and hardware and not because the idea is superior to alternative research directions”. DataParallel training has been extensively studied, tuned, and scaled (Kaplan et al., 2020), and it is hard to beat the wisdom-of-the-crowd of thousands of researchers. In contrast, the federated learning literature has mainly studied smaller scale models, primarily due to its focus on edge devices. There are huge opportunities for bringing the ideas from the federated learning literature to the new world of large scale training for LLMs. A critical next work is to study how new distributed methods like ours should be tuned and scaled across multiple axes (e.g. model size, overtraining factor, number of replicas). In particular, how to scale efficiently the number of DiLoCo replicas given an equivalent token budget is most needed.

More generally, reducing the communication problem to a minor obstacle allows new classes of co-designed architectures and training paradigms (for example Douillard et al. (2024b)) maximizing available compute (Sutton, 2019): we hope to see the training of modular constellations of small models loosely connected (Dean, 2021) across heterogeneous devices, using compute arbitrage spread world-wide.

#### Acknowledgements

We would like to thank Alban Rrustemi, Jeff Dean, Michael Isard, Sebastian Borgeaud, Rohan Anil, Koray Kavukcuoglu, and Raia Hadsell for their feedback and leadership support; Andrei Rusu, Adhiguna Kuncoro, Lucio Dery, Rachita Chhaparia, Zohar Yahav, Qixuan Feng, Zack Nado, Nova Fallen, Nicole Mitchell, Sean Augenstein, and Stephen Roller for the helpful advices; finally Alberto Magni, Juliana Vincente Franco, Joel Wee, James Lotte, Matthew Johnson, and Blake Hechtman for unblocking us through so many engineering hurdles alongside our journey.

#### References

Aditya Agrawal, Matthew Hedlund, and Blake Hechtman. exmy: A data type and technique for arbitrary bit precision quantization. arXiv preprint library, 2024. URL https:// arxiv.org/abs/2405.13938.

Samuel K. Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa. Git re-basin: Merging models modulo permutation symmetries. arXiv preprint library, 2023. URL https:// arxiv.org/abs/2209.04836.

Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, and Sunav Choudhary. Federated learning with personalization layers, 2019. URL https://arxiv.org/abs/ 1912.00818.

Mohamed Baioumy and Alex Cheema. Sparta. blog.exolabs.net, 2025. URL https:// blog.exolabs.net/day-12/.

Olivier Beaumont, Lionel Eyraud-Dubois, Alena Shilova, and Xunyi Zhao. Weight offloading strategies. HAL repository, 2022. URL https: //inria.hal.science/hal-03580767/.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2020. URL https://arxiv.org/abs/1911.11641.

Alexander Borzunov, Dmitry Baranchuk, Tim Dettmers, Max Ryabinin, Younes Belkada, Artem Chumachenko, Pavel Samygin, and Colin Raffel. Petals: Collaborative inference and fine-tuning of large models. Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL Short Papers), 2023. URL https://arxiv.org/abs/ 2209.01188.

Zachary Charles, Nicole Mitchell, Krishna Pillutla, Michael Reneer, and Zachary Garrett. Towards federated foundation models: Scalable dataset pipelines for group-structured learning. Advances in Neural Information Processing Systems, 2024. URL https://arxiv.org/abs/ 2307.09619.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy MeierHellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 2023. URL http://jmlr.org/ papers/v24/22-1144.html.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018. URL https://arxiv.org/ abs/1803.05457v1.

Jeff Dean. Pathways: A next-generation ai architecture. Google’s blog, 2021. URL https://blog.google/technology/ ai/introducing-pathways-nextgeneration-ai-architecture/.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo

Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. Deepseek-v3 technical report. arXiv preprint library, 2024. URL

##### https://arxiv.org/abs/2412.19437.

Michael Diskin, Alexey Bukhtiyarov, Max Ryabinin, Lucile Saulnier, Quentin Lhoest, Anton Sinitsin, Dmitry Popov, Dmitry Pyrkin, Maxim Kashirin, Alexander Borzunov, Albert Villanova del Moral, Denis Mazur, Ilia Kobelev, Yacine Jernite, Thomas Wolf, and Gennady Pekhimenko. Distributed deep learning in open collaborations. Advances in Neural Information Processing Systems (NeurIPS), 2021. URL https://arxiv.org/abs/2106.10207.

Arthur Douillard, Qixuan Feng, Andrei A. Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc’Aurelio Ranzato, Arthur Szlam, and Jiajun Shen. DiLoCo: Distributed lowcommunication training of language models. International Conference on Machine Learning (ICML) Workshop, 2024a. URL https:// arXiv.org/abs/2311.08105.

Arthur Douillard, Qixuan Feng, Andrei A. Rusu, Adhiguna Kuncoro, Yani Donchev, Rachita Chhaparia, Ionel Gog, Marc’Aurelio Ranzato, Jiajun Shen, and Arthur Szlam. Dipaco: Distributed path composition. arXiv preprint library, 2024b. URL https://arxiv.org/ abs/2403.10616.

Louis Fournier, Adel Nabli, Masih Aminbeidokhti, Marco Pedersoli, Eugene Belilovsky, and Edouard Oyallon. Wash: Train your ensemble with communication-efficient weight shuffling, then average. Advances in Neural Information Processing Systems (NeurIPS) Workshop, 2024. URL https://arxiv.org/abs/ 2405.17517.

Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. International Conference on Machine Learning (ICML), 2020. URL https: //arxiv.org/abs/1912.05671.

Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, Rui Xin, Marianna Nezhurina, Igor Vasiljevic, Jenia Jitsev, Luca Soldaini, Alexandros G. Dimakis, Gabriel Ilharco,

Pang Wei Koh, Shuran Song, Thomas Kollar, Yair Carmon, Achal Dave, Reinhard Heckel, Niklas Muennighoff, and Ludwig Schmidt. Language models scale reliably with over-training and on downstream tasks. arXiv preprint library, 2024. URL https://arxiv.org/abs/ 2403.08540.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren RantalaYeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin,

Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew

Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrst-

edt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi

Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. Query-key normalization for transformers. Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020. URL https://arxiv.org/abs/2010.04245.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models. Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2203.15556.

Sara Hooker. The hardware lottery. Communication for the ACM, 2020. URL https:// arxiv.org/abs/2009.06489.

Zhouyuan Huo, Qian Yang, Bin Gu, and Lawrence Carin. Heng Huang. Faster on-device training using new federated momentum algorithm. arXiv preprint library, 2020. URL https:// arxiv.org/abs/2002.02090.

Gabriel Ilharco, Mitchell Wortsman, Samir Yitzhak Gadre, Shuran Song, Hannaneh Hajishirzi, Simon Kornblith, Ali Farhadi, and Ludwig Schmidt. Patching open-vocabulary models by interpolating weights. Advances in Neural Information Processing Systems (NeurIPS), 2022.

Sami Jaghouar, Jack Min Ong, Manveer Basra, Fares Obeid, Jannik Straube, Michael Keiblinger, Elie Bakouch, Lucas Atkins, Maziyar Panahi, Charles Goddard, Max Ryabinin, and Johannes Hagemann. Intellect-1 technical report. arXiv preprint library, 2024a. URL https://arxiv.org/abs/2412.01152.

Sami Jaghouar, Jack Min Ong, and Johannes Hagemann. Opendiloco: An open-source framework for globally distributed low-communication training. arXiv preprint library, 2024b. URL https://arxiv.org/abs/2407.07852.

Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng. Dataless knowledge fusion by merging weights of language models. Proceedings of the International Conference on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2212.09849.

Keller Jordan, Hanie Sedghi, Olga Saukh, Rahim Entezari, and Behnam Neyshabur. Repair: Renormalizing permuted activations for interpolation repair. arXiv preprint library, 2023. URL https://arxiv.org/abs/ 2211.08403.

Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. Foundations and trends in machine learning, 2021. URL https://arxiv.org/ abs/1912.04977.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint library, 2020. URL https://arxiv.org/abs/2001.08361.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. Proceedings of the International Conference on Learning Representations (ICLR), 2014. URL https: //arxiv.org/abs/1412.6980.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems (NeurIPS), 2012. URL https://papers.nips.cc/ paper_files/paper/2012/hash/ c399862d3b9d6b76c8436e924a68c45bAbstract.html.

Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A. Smith, and Luke Zettlemoyer. Branch-train-merge: Embarrassingly parallel training of expert language models. arXiv preprint library, 2022. URL https://arxiv.org/abs/2208.03306.

Tao Lin, Sebastian U. Stich, Kumar Kshitij Patel, and Martin Jaggi. Don’t use large minibatches, use local sgd. Proceedings of the International Conference on Learning Representations (ICLR), 2020. URL https://arxiv.org/ abs/1808.07217.

Yujun Lin, Song Han, Huizi Mao, Yu Wang, and William J. Dally. Deep gradient compression: Reducing the communication bandwidth for distributed training, 2018. URL https:// arxiv.org/abs/1712.01887.

Bo Liu, Rachita Chhaparia, Arthur Douillard, Satyen Kale, Andrei A. Rusu, Jiajun Shen, Arthur Szlam, and Marc’Aurelio Ranzato. Asynchronous local-sgd training for language modeling. International Conference on Machine Learning (ICML) Workshop, 2024a. URL https: //arxiv.org/abs/2401.09135.

Peter J. Liu, Roman Novak, Jaehoon Lee, Mitchell Wortsman, Lechao Xiao, Katie Everett, Alexander A. Alemi, Mark Kurzeja, Pierre Marcenac, Izzeddin Gur, Simon Kornblith, Kelvin Xu, Gamaleldin Elsayed, Ian Fischer, Jeffrey Pennington, Ben Adlam, and Jascha-Sohl Dickstein. Nanodo: A minimal transformer decoderonly language model implementation in JAX., 2024b. URL http://github.com/googledeepmind/nanodo.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. Proceedings of the International Conference on Learning Representations (ICLR), 2019. URL https:// arxiv.org/abs/1711.05101.

Michael Matena and Colin Raffel. Merging models with fisher-weighted averaging. Advances in Neural Information Processing Systems (NeurIPS), 2021. URL https://arxiv.org/ abs/2111.09832.

H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. Communication-efficient learning of deep networks from decentralized data. International Conference on Artificial Intelligence and Statistics (AISTATS), 2017. URL https://arxiv.org/ abs/1602.05629.

Behnam Neyshabur, Hanie Sedghi, and Chiyuan Zhang. What is being transferred in transfer learning? Advances in Neural Information Processing Systems (NeurIPS), 2020. URL https://arxiv.org/abs/2008.11687.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha GontijoLopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Bil-

lie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya,

Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774.

Bowen Peng, Jeffrey Quesnelle, and Diederik P. Kingma. Demo: Decoupled momentum optimization. arXiv preprint library, 2024. URL https://arxiv.org/abs/2411.19870.

##### Shawn Presser. Swarm training, 2020. URL https://battle.shawwn.com/swarmtraining-v01a.pdf.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 2020. URL https:

##### //arxiv.org/abs/1910.10683.

Alexandre Ramé, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, Léon Bottou, and David LopezPaz. Model ratatouille: Recycling diverse models for out-of-distribution generalization. International Conference on Machine Learning (ICML), 2023a. URL https://arxiv.org/ abs/2212.10445.

Alexandre Ramé, Guillaume Couairon, Mustafa Shukor, Corentin Dancette, Jean-Baptiste Gaya, Laure Soulier, and Matthieu Cord. Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. Advances in Neural Information Processing Systems (NeurIPS), 2023b. URL https: //arxiv.org/abs/2306.04488.

Alexandre Ramé, Matthieu Kirchmeyer, Thibaud Rahier, Alain Rakotomamonjy, Patrick Gallinari, and Matthieu Cord. Diverse weight averaging for out-of-distribution generalization. Ad-

vances in Neural Information Processing Systems (NeurIPS), 2023c. URL https://arxiv.org/ abs/2205.09739.

Sylvestre-Alvise Rebuffi, Francesco Croce, and Sven Gowal. Revisiting adapters with adversarial training. Proceedings of the International Conference on Learning Representations (ICLR), 2022. URL https://arxiv.org/ abs/2210.04886.

Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konečný, Sanjiv Kumar, and H. Brendan McMahan. Adaptive federated optimization. Proceedings of the International Conference on Learning Representations (ICLR), 2021. URL https:// arxiv.org/abs/2003.00295.

Keith Rush, Zachary Charles, Zachary Garrett, Sean Augenstein, and Nicole Mitchell. Drjax: Scalable and differentiable mapreduce primitives in jax. International Conference on Machine Learning (ICML) Workshop, 2024. URL https://arxiv.org/abs/2403.07128.

Max Ryabinin, Eduard Gorbunov, Vsevolod Plokhotnyuk, and Gennady Pekhimenko. Moshpit sgd: Communication-efficient decentralized training on heterogeneous unreliable devices. Advances in Neural Information Processing Systems (NeurIPS), 2021. URL https: //arxiv.org/abs/2103.03239.

Max Ryabinin, Tim Dettmers, Michael Diskin, and Alexander Borzunov. Swarm parallelism: Training large models can be surprisingly communication-efficient. International Conference on Machine Learning (ICML), 2023. URL https://arxiv.org/abs/2301.11913.

Lorenzo Sani, Alex Iacob, Zeyu Cao, Royson Lee, Bill Marino, Yan Gao, Dongqi Cai, Zexi Li, Wanru Zhao, Xinchi Qiu, and Nicholas D. Lane. Photon: Federated llm pre-training. arXiv preprint library, 2024a. URL https: //arxiv.org/abs/2411.02908.

Lorenzo Sani, Alex Iacob, Zeyu Cao, Bill Marino, Yan Gao, Tomas Paulik, Wanru Zhao, William F. Shen, Preslav Aleksandrov, Xinchi Qiu, and Nicholas D. Lane. The future of large language

model pre-training is federated. arXiv preprint library, 2024b. URL https://arxiv.org/ abs/2405.10853.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-ofexperts layer. arXiv preprint library, 2017. URL https://arxiv.org/abs/1701.06538.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multibillion parameter language models using model parallelism. arXiv preprint library, 2020. URL https://arxiv.org/abs/1909.08053.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: an open corpus of three trillion tokens for language model pretraining research, 2024. URL https://arxiv.org/ abs/2402.00159.

Sebastian U. Stich. Local SGD converges fast and communicates little. Proceedings of the International Conference on Learning Representations (ICLR), 2019. URL https://arxiv.org/ abs/1805.09767.

George Stoica, Daniel Bolya, Jakob Bjorner, Taylor Hearn, and Judy Hoffman. Zipit! merging models from different tasks without training. arXiv preprint library, 2023. URL https:// arxiv.org/abs/2305.03053.

Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and momentum in deep learning. International Conference

on Machine Learning (ICML), 2013. URL https://proceedings.mlr.press/v28/ sutskever13.html.

Rich Sutton. The bitter lesson. incompleteideas.net, 2019. URL http: //www.incompleteideas.net/IncIdeas/ BitterLesson.html.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul R. Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, Jack Krawczyk, Cosmo Du, Ed Chi, HengTze Cheng, Eric Ni, Purvi Shah, Patrick Kane, Betty Chan, Manaal Faruqui, Aliaksei Severyn, Hanzhao Lin, YaGuang Li, Yong Cheng, Abe Ittycheriah, Mahdis Mahdieh, Mia Chen, Pei Sun, Dustin Tran, Sumit Bagri, Balaji Lakshminarayanan, Jeremiah Liu, Andras Orban, Fabian Güra, Hao Zhou, Xinying Song, Aurelien Boffy, Harish Ganapathy, Steven Zheng, HyunJeong Choe, Ágoston Weisz, Tao Zhu, Yifeng Lu, Siddharth Gopal, Jarrod Kahn, Maciej Kula, Jeff Pitman, Rushin Shah, Emanuel Taropa, Majd Al Merey, Martin Baeuml, Zhifeng Chen, Laurent El Shafey, Yujing Zhang, Olcan Sercinoglu, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, Alexandre Frechette, Charlotte Smith, Laura Culp, Lev Proleev, Yi Luan, Xi Chen, James Lottes, Nathan Schucher, Federico Lebron, Alban Rrustemi, Natalie Clay, Phil Crone, Tomas Kocisky, Jeffrey Zhao, Bartek Perz, Dian Yu, Heidi Howard, Adam Bloniarz, Jack W. Rae, Han Lu, Laurent Sifre, Marcello Maggioni, Fred Alcober, Dan Garrette, Megan Barnes, Shantanu Thakoor, Jacob Austin, Gabriel Barth-

Maron, William Wong, Rishabh Joshi, Rahma Chaabouni, Deeni Fatiha, Arun Ahuja, Gaurav Singh Tomar, Evan Senter, Martin Chadwick, Ilya Kornakov, Nithya Attaluri, Iñaki Iturrate, Ruibo Liu, Yunxuan Li, Sarah Cogan, Jeremy Chen, Chao Jia, Chenjie Gu, Qiao Zhang, Jordan Grimstad, Ale Jakse Hartman, Xavier Garcia, Thanumalayan Sankaranarayana Pillai, Jacob Devlin, Michael Laskin, Diego de Las Casas, Dasha Valter, Connie Tao, Lorenzo Blanco, Adrià Puigdomènech Badia, David Reitter, Mianna Chen, Jenny Brennan, Clara Rivera, Sergey Brin, Shariq Iqbal, Gabriela Surita, Jane Labanowski, Abhi Rao, Stephanie Winkler, Emilio Parisotto, Yiming Gu, Kate Olszewska, Ravi Addanki, Antoine Miech, Annie Louis, Denis Teplyashin, Geoff Brown, Elliot Catt, Jan Balaguer, Jackie Xiang, Pidong Wang, Zoe Ashwood, Anton Briukhov, Albert Webson, Sanjay Ganapathy, Smit Sanghavi, Ajay Kannan, Ming-Wei Chang, Axel Stjerngren, Josip Djolonga, Yuting Sun, Ankur Bapna, Matthew Aitchison, Pedram Pejman, Henryk Michalewski, Tianhe Yu, Cindy Wang, Juliette Love, Junwhan Ahn, Dawn Bloxwich, Kehang Han, Peter Humphreys, Thibault Sellam, James Bradbury, Varun Godbole, Sina Samangooei, Bogdan Damoc, Alex Kaskasoli, Sébastien M. R. Arnold, Vijay Vasudevan, Shub-

- ham Agrawal, Jason Riesa, Dmitry Lepikhin, Richard Tanburn, Srivatsan Srinivasan, Hyeontaek Lim, Sarah Hodkinson, Pranav Shyam, Jo-
- han Ferret, Steven Hand, Ankush Garg, Tom Le Paine, Jian Li, Yujia Li, Minh Giang, Alexander Neitz, Zaheer Abbas, Sarah York, Machel Reid, Elizabeth Cole, Aakanksha Chowdhery, Dipanjan Das, Dominika Rogozińska, Vitaliy Nikolaev, Pablo Sprechmann, Zachary Nado, Lukas Zilka, Flavien Prost, Luheng He, Marianne Monteiro, Gaurav Mishra, Chris Welty, Josh Newlan, Dawei Jia, Miltiadis Allamanis, Clara Huiyi Hu, Raoul de Liedekerke, Justin Gilmer, Carl Saroufim, Shruti Rijhwani, Shaobo Hou, Disha Shrivastava, Anirudh Baddepudi, Alex Goldin, Adnan Ozturel, Albin Cassirer,

- Yunhan Xu, Daniel Sohn, Devendra Sachan, Reinald Kim Amplayo, Craig Swanson, Dessie Petrova, Shashi Narayan, Arthur Guez, Siddhartha Brahma, Jessica Landon, Miteyan Pa-

tel, Ruizhe Zhao, Kevin Villela, Luyu Wang, Wenhao Jia, Matthew Rahtz, Mai Giménez, Legg Yeung, James Keeling, Petko Georgiev, Diana Mincu, Boxi Wu, Salem Haykal, Rachel Saputro, Kiran Vodrahalli, James Qin, Zeynep Cankara, Abhanshu Sharma, Nick Fernando, Will Hawkins, Behnam Neyshabur, Solomon Kim, Adrian Hutter, Priyanka Agrawal, Alex Castro-Ros, George van den Driessche, Tao Wang, Fan Yang, Shuo yiin Chang, Paul Komarek, Ross McIlroy, Mario Lučić, Guodong Zhang, Wael Farhan, Michael Sharman, Paul Natsev, Paul Michel, Yamini Bansal, Siyuan Qiao, Kris Cao, Siamak Shakeri, Christina Butterfield, Justin Chung, Paul Kishan Rubenstein, Shivani Agrawal, Arthur Mensch, Kedar Soparkar, Karel Lenc, Timothy Chung, Aedan Pope, Loren Maggiore, Jackie Kay, Priya Jhakra, Shibo Wang, Joshua Maynez, Mary Phuong, Taylor Tobin, Andrea Tacchetti, Maja Trebacz, Kevin Robinson, Yash Katariya, Sebastian Riedel, Paige Bailey, Kefan Xiao, Nimesh Ghelani, Lora Aroyo, Ambrose Slone, Neil Houlsby, Xuehan Xiong, Zhen Yang, Elena Gribovskaya, Jonas Adler, Mateo Wirth, Lisa Lee, Music Li, Thais Kagohara, Jay Pavagadhi, Sophie Bridgers, Anna Bortsova, Sanjay Ghemawat, Zafarali Ahmed, Tianqi Liu, Richard Powell, Vijay Bolina, Mariko Iinuma, Polina Zablotskaia, James Besley, Da-Woon Chung, Timothy Dozat, Ramona Comanescu, Xiance Si, Jeremy Greer, Guolong Su, Martin Polacek, Raphaël Lopez Kaufman, Simon Tokumine, Hexiang Hu, Elena Buchatskaya, Yingjie Miao, Mohamed Elhawaty, Aditya Siddhant, Nenad Tomasev, Jinwei Xing, Christina Greer, Helen Miller, Shereen Ashraf, Aurko Roy, Zizhao Zhang, Ada Ma, Angelos Filos, Milos Besta, Rory Blevins, Ted Klimenko, ChihKuan Yeh, Soravit Changpinyo, Jiaqi Mu, Oscar Chang, Mantas Pajarskas, Carrie Muir, Vered Cohen, Charline Le Lan, Krishna Haridasan, Amit Marathe, Steven Hansen, Sholto Douglas, Rajkumar Samuel, Mingqiu Wang, Sophia Austin, Chang Lan, Jiepu Jiang, Justin Chiu, Jaime Alonso Lorenzo, Lars Lowe Sjösund, Sébastien Cevey, Zach Gleicher, Thi Avrahami, Anudhyan Boral, Hansa Srinivasan, Vittorio Selo, Rhys May, Konstantinos Aiso-

pos, Léonard Hussenot, Livio Baldini Soares, Kate Baumli, Michael B. Chang, Adrià Recasens, Ben Caine, Alexander Pritzel, Filip Pavetic, Fabio Pardo, Anita Gergely, Justin Frye, Vinay Ramasesh, Dan Horgan, Kartikeya Badola, Nora Kassner, Subhrajit Roy, Ethan Dyer, Víctor Campos Campos, Alex Tomala,

- Yunhao Tang, Dalia El Badawy, Elspeth White, Basil Mustafa, Oran Lang, Abhishek Jindal, Sharad Vikram, Zhitao Gong, Sergi Caelles, Ross Hemsley, Gregory Thornton, Fangxiaoyu Feng, Wojciech Stokowiec, Ce Zheng, Phoebe Thacker, Çağlar Ünlü, Zhishuai Zhang, Mohammad Saleh, James Svensson, Max Bileschi, Piyush Patil, Ankesh Anand, Roman Ring, Katerina Tsihlas, Arpi Vezer, Marco Selvi, Toby Shevlane, Mikel Rodriguez, Tom Kwiatkowski, Samira Daruki, Keran Rong, Allan Dafoe, Nicholas FitzGerald, Keren Gu-Lemberg, Mina Khan, Lisa Anne Hendricks, Marie Pellat, Vladimir Feinberg, James Cobon-Kerr, Tara Sainath, Maribeth Rauh, Sayed Hadi Hashemi, Richard Ives, Yana Hasson, Eric Noland, Yuan Cao, Nathan Byrd, Le Hou, Qingze Wang, Thibault Sottiaux, Michela Paganini, JeanBaptiste Lespiau, Alexandre Moufarek, Samer Hassan, Kaushik Shivakumar, Joost van Amersfoort, Amol Mandhane, Pratik Joshi, Anirudh Goyal, Matthew Tung, Andrew Brock, Hannah Sheahan, Vedant Misra, Cheng Li, Nemanja Rakićević, Mostafa Dehghani, Fangyu Liu, Sid Mittal, Junhyuk Oh, Seb Noury, Eren Sezener, Fantine Huot, Matthew Lamm, Nicola De Cao, Charlie Chen, Sidharth Mudgal, Romina Stella, Kevin Brooks, Gautam Vasudevan, Chenxi Liu, Mainak Chain, Nivedita Melinkeri, Aaron Cohen, Venus Wang, Kristie Seymore, Sergey Zubkov, Rahul Goel, Summer Yue, Sai Krishnakumaran, Brian Albert, Nate Hurley, Motoki Sano, Anhad Mohananey, Jonah Joughin, Egor Filonov, Tomasz Kępa, Yomna Eldawy, Jiawern Lim, Rahul Rishi, Shirin Badiezadegan, Taylor Bos, Jerry Chang, Sanil Jain, Sri Gayatri Sundara Padmanabhan, Subha Puttagunta, Kalpesh Krishna, Leslie Baker, Norbert Kalb, Vamsi Bedapudi, Adam Kurzrok, Shuntong Lei, Anthony Yu, Oren Litvin, Xiang Zhou, Zhichun Wu, Sam Sobell, Andrea Siciliano, Alan Papir, Robby Neale, Jonas Bragagnolo,

Tej Toor, Tina Chen, Valentin Anklin, Feiran Wang, Richie Feng, Milad Gholami, Kevin Ling, Lijuan Liu, Jules Walter, Hamid Moghaddam, Arun Kishore, Jakub Adamek, Tyler Mercado, Jonathan Mallinson, Siddhinita Wandekar, Stephen Cagle, Eran Ofek, Guillermo Garrido, Clemens Lombriser, Maksim Mukha, Botu Sun, Hafeezul Rahman Mohammad, Josip Matak, Yadi Qian, Vikas Peswani, Pawel Janus, Quan Yuan, Leif Schelin, Oana David, Ankur Garg, Yifan He, Oleksii Duzhyi, Anton Älgmyr, Timothée Lottaz, Qi Li, Vikas Yadav, Luyao Xu, Alex Chinien, Rakesh Shivanna, Aleksandr Chuklin, Josie Li, Carrie Spadine, Travis Wolfe, Kareem Mohamed, Subhabrata Das, Zihang Dai, Kyle He, Daniel von Dincklage, Shyam Upadhyay, Akanksha Maurya, Luyan Chi, Sebastian Krause, Khalid Salama, Pam G Rabinovitch, Pavan Kumar Reddy M, Aarush Selvan, Mikhail Dektiarev, Golnaz Ghiasi, Erdem Guven, Himanshu Gupta, Boyi Liu, Deepak Sharma, Idan Heimlich Shtacher, Shachi Paul, Oscar Akerlund, François-Xavier Aubet, Terry Huang, Chen Zhu, Eric Zhu, Elico Teixeira, Matthew Fritze, Francesco Bertolini, Liana-Eleonora Marinescu, Martin Bölle, Dominik Paulus, Khyatti Gupta, Tejasi Latkar, Max Chang, Jason Sanders, Roopa Wilson, Xuewei Wu, Yi-Xuan Tan, Lam Nguyen Thiet, Tulsee Doshi, Sid Lall, Swaroop Mishra, Wanming Chen, Thang Luong, Seth Benjamin, Jasmine Lee, Ewa Andrejczuk, Dominik Rabiej, Vipul Ranjan, Krzysztof Styrc, Pengcheng Yin, Jon Simon, Malcolm Rose Harriott, Mudit Bansal, Alexei Robsky, Geoff Bacon, David Greene, Daniil Mirylenka, Chen Zhou, Obaid Sarvana, Abhimanyu Goyal, Samuel Andermatt, Patrick Siegler, Ben Horn, Assaf Israel, Francesco Pongetti, Chih-Wei "Louis" Chen, Marco Selvatici, Pedro Silva, Kathie Wang, Jackson Tolins, Kelvin Guu, Roey Yogev, Xiaochen Cai, Alessandro Agostini, Maulik Shah, Hung Nguyen, Noah Ó Donnaile, Sébastien Pereira, Linda Friso, Adam Stambler, Adam Kurzrok, Chenkai Kuang, Yan Romanikhin, Mark Geller, ZJ Yan, Kane Jang, ChengChun Lee, Wojciech Fica, Eric Malmi, Qijun Tan, Dan Banica, Daniel Balle, Ryan Pham, Yanping Huang, Diana Avram, Hongzhi Shi,

Jasjot Singh, Chris Hidey, Niharika Ahuja, Pranab Saxena, Dan Dooley, Srividya Pranavi Potharaju, Eileen O’Neill, Anand Gokulchandran, Ryan Foley, Kai Zhao, Mike Dusenberry, Yuan Liu, Pulkit Mehta, Ragha Kotikalapudi, Chalence Safranek-Shrader, Andrew Goodman, Joshua Kessinger, Eran Globen, Prateek Kol-

- har, Chris Gorgolewski, Ali Ibrahim, Yang Song, Ali Eichenbaum, Thomas Brovelli, Sahitya Potluri, Preethi Lahoti, Cip Baetu, Ali Ghorbani, Charles Chen, Andy Crawford, Shalini Pal, Mukund Sridhar, Petru Gurita, Asier Mujika, Igor Petrovski, Pierre-Louis Cedoz, Chenmei Li, Shiyuan Chen, Niccolò Dal Santo, Siddharth Goyal, Jitesh Punjabi, Karthik Kappaganthu, Chester Kwak, Pallavi LV, Sarmishta Velury, Himadri Choudhury, Jamie Hall, Premal Shah, Ricardo Figueira, Matt Thomas, Minjie Lu, Ting Zhou, Chintu Kumar, Thomas Jurdi, Sharat Chikkerur, Yenai Ma, Adams Yu, Soo Kwak, Victor Ähdel, Sujeevan Rajayogam, Travis Choma, Fei Liu, Aditya Barua, Colin Ji, Ji Ho Park, Vincent Hellendoorn, Alex Bailey, Taylan Bilal, Huanjie Zhou, Mehrdad Khatir, Charles Sutton, Wojciech Rzadkowski, Fiona Macintosh, Konstantin Shagin, Paul Medina, Chen Liang, Jinjing Zhou, Pararth Shah, Yingying Bi, Attila Dankovics, Shipra Banga, Sabine Lehmann, Marissa Bredesen, Zifan Lin, John Eric Hoffmann, Jonathan Lai, Raynald Chung, Kai Yang, Nihal Balani, Arthur Bražinskas, Andrei Sozanschi, Matthew Hayes, Héctor Fernández Alcalde, Peter Makarov, Will Chen, Antonio Stella, Liselotte Snijders, Michael Mandl, Ante Kärrman, Paweł Nowak, Xinyi Wu, Alex Dyck, Krishnan Vaidyanathan, Raghavender R, Jessica Mallet, Mitch Rudominer, Eric Johnston, Sushil Mittal, Akhil Udathu, Janara Christensen, Vishal Verma, Zach Irving, Andreas Santucci, Gamaleldin Elsayed, Elnaz Davoodi, Marin Georgiev, Ian Tenney, Nan Hua, Geoffrey Cideron, Edouard Leurent, Mahmoud Alnahlawi, Ionut Georgescu, Nan Wei, Ivy Zheng, Dylan Scandinaro, Heinrich Jiang, Jasper Snoek, Mukund Sundararajan, Xuezhi Wang, Zack Ontiveros, Itay Karo, Jeremy Cole, Vinu Rajashekhar, Lara Tumeh, Eyal Ben-David, Rishub Jain, Jonathan Uesato, Romina Datta, Oskar Bunyan, Shimu Wu, John

Zhang, Piotr Stanczyk, Ye Zhang, David Steiner, Subhajit Naskar, Michael Azzam, Matthew Johnson, Adam Paszke, Chung-Cheng Chiu, Jaume Sanchez Elias, Afroz Mohiuddin, Faizan Muhammad, Jin Miao, Andrew Lee, Nino Vieillard, Jane Park, Jiageng Zhang, Jeff Stanway, Drew Garmon, Abhijit Karmarkar, Zhe Dong, Jong Lee, Aviral Kumar, Luowei Zhou, Jonathan Evens, William Isaac, Geoffrey Irving, Edward Loper, Michael Fink, Isha Arkatkar, Nanxin Chen, Izhak Shafran, Ivan Petrychenko, Zhe Chen, Johnson Jia, Anselm Levskaya, Zhenkai Zhu, Peter Grabowski, Yu Mao, Alberto Magni, Kaisheng Yao, Javier Snaider, Norman Casagrande, Evan Palmer, Paul Suganthan, Alfonso Castaño, Irene Giannoumis, Wooyeol Kim, Mikołaj Rybiński, Ashwin Sreevatsa, Jennifer Prendki, David Soergel, Adrian Goedeckemeyer, Willi Gierke, Mohsen Jafari, Meenu Gaba, Jeremy Wiesner, Diana Gage Wright, Yawen Wei, Harsha Vashisht, Yana Kulizhskaya, Jay Hoover, Maigo Le, Lu Li, Chimezie Iwuanyanwu, Lu Liu, Kevin Ramirez, Andrey Khorlin, Albert Cui, Tian LIN, Marcus Wu, Ricardo Aguilar, Keith Pallo, Abhishek Chakladar, Ginger Perng, Elena Allica Abellan, Mingyang Zhang, Ishita Dasgupta, Nate Kushman, Ivo Penchev, Alena Repina, Xihui Wu, Tom van der Weide, Priya Ponnapalli, Caroline Kaplan, Jiri Simsa, Shuangfeng Li, Olivier Dousse, Fan Yang, Jeff Piper, Nathan Ie, Rama Pasumarthi, Nathan Lintz, Anitha Vijayakumar, Daniel Andor, Pedro Valenzuela, Minnie Lui, Cosmin Paduraru, Daiyi Peng, Katherine Lee, Shuyuan Zhang, Somer Greene, Duc Dung Nguyen, Paula Kurylowicz, Cassidy Hardin, Lucas Dixon, Lili Janzer, Kiam Choo, Ziqiang Feng, Biao Zhang, Achintya Singhal, Dayou Du, Dan McKinnon, Natasha Antropova, Tolga Bolukbasi, Orgad Keller, David Reid, Daniel Finchelstein, Maria Abi Raad, Remi Crocker, Peter Hawkins, Robert Dadashi, Colin Gaffney, Ken Franko, Anna Bulanova, Rémi Leblond, Shirley Chung, Harry Askham, Luis C. Cobo, Kelvin Xu, Felix Fischer, Jun Xu, Christina Sorokin, Chris Alberti, Chu-Cheng Lin, Colin Evans, Alek Dimitriev, Hannah Forbes, Dylan Banarse, Zora Tung, Mark Omernick, Colton Bishop, Rachel Sterneck, Rohan Jain, Jiawei Xia, Ehsan Amid,

Francesco Piccinno, Xingyu Wang, Praseem Banzal, Daniel J. Mankowitz, Alex Polozov, Victoria Krakovna, Sasha Brown, MohammadHossein Bateni, Dennis Duan, Vlad Firoiu, Meghana Thotakuri, Tom Natan, Matthieu Geist, Ser tan Girgin, Hui Li, Jiayu Ye, Ofir Roval, Reiko Tojo, Michael Kwong, James LeeThorp, Christopher Yew, Danila Sinopalnikov, Sabela Ramos, John Mellor, Abhishek Sharma, Kathy Wu, David Miller, Nicolas Sonnerat, Denis Vnukov, Rory Greig, Jennifer Beattie, Emily Caveness, Libin Bai, Julian Eisenschlos, Alex Korchemniy, Tomy Tsai, Mimi Jasarevic, Weize Kong, Phuong Dao, Zeyu Zheng, Frederick Liu, Fan Yang, Rui Zhu, Tian Huey Teh, Jason Sanmiya, Evgeny Gladchenko, Nejc Trdin, Daniel Toyama, Evan Rosen, Sasan Tavakkol, Linting Xue, Chen Elkind, Oliver Woodman, John Carpenter, George Papamakarios, Rupert Kemp, Sushant Kafle, Tanya Grunina, Rishika Sinha, Alice Talbert, Diane Wu, Denese OwusuAfriyie, Cosmo Du, Chloe Thornton, Jordi PontTuset, Pradyumna Narayana, Jing Li, Saaber Fatehi, John Wieting, Omar Ajmeri, Benigno Uria, Yeongil Ko, Laura Knight, Amélie Héliou, Ning Niu, Shane Gu, Chenxi Pang, Yeqing Li, Nir Levine, Ariel Stolovich, Rebeca SantamariaFernandez, Sonam Goenka, Wenny Yustalim, Robin Strudel, Ali Elqursh, Charlie Deck, Hyo Lee, Zonglin Li, Kyle Levin, Raphael Hoffmann, Dan Holtmann-Rice, Olivier Bachem, Sho Arora, Christy Koh, Soheil Hassas Yeganeh, Siim Põder, Mukarram Tariq, Yanhua Sun, Lucian Ionita, Mojtaba Seyedhosseini, Pouya Tafti, Zhiyu Liu, Anmol Gulati, Jasmine Liu, Xinyu Ye, Bart Chrzaszcz, Lily Wang, Nikhil Sethi, Tianrun Li, Ben Brown, Shreya Singh, Wei Fan, Aaron Parisi, Joe Stanton, Vinod Koverkathu, Christopher A. Choquette-Choo, Yunjie Li, TJ Lu, Abe Ittycheriah, Prakash Shroff, Mani Varadarajan, Sanaz Bahargam, Rob Willoughby, David Gaddy, Guillaume Desjardins, Marco Cornero, Brona Robenek, Bhavishya Mittal, Ben Albrecht, Ashish Shenoy, Fedor Moiseev, Henrik Jacobsson, Alireza Ghaffarkhah, Morgane Rivière, Alanna Walton, Clément Crepy, Alicia Parrish, Zongwei Zhou, Clement Farabet, Carey Radebaugh, Praveen Srinivasan, Claudia van der Salm, Andreas Fid-

jeland, Salvatore Scellato, Eri Latorre-Chimoto, Hanna Klimczak-Plucińska, David Bridson, Dario de Cesare, Tom Hudson, Piermaria Mendolicchio, Lexi Walker, Alex Morris, Matthew Mauger, Alexey Guseynov, Alison Reid, Seth Odoom, Lucia Loher, Victor Cotruta, Madhavi Yenugula, Dominik Grewe, Anastasia Petrushkina, Tom Duerig, Antonio Sanchez, Steve Yadlowsky, Amy Shen, Amir Globerson, Lynette Webb, Sahil Dua, Dong Li, Surya Bhupatiraju, Dan Hurt, Haroon Qureshi, Ananth Agarwal, Tomer Shani, Matan Eyal, Anuj Khare, Shreyas Rammohan Belle, Lei Wang, Chetan Tekur, Mihir Sanjay Kale, Jinliang Wei, Ruoxin Sang, Brennan Saeta, Tyler Liechty, Yi Sun, Yao Zhao, Stephan Lee, Pandu Nayak, Doug Fritz, Manish Reddy Vuyyuru, John Aslanides, Nidhi Vyas, Martin Wicke, Xiao Ma, Evgenii Eltyshev, Nina Martin, Hardie Cate, James Manyika, Keyvan Amiri, Yelin Kim, Xi Xiong, Kai Kang, Florian Luisier, Nilesh Tripuraneni, David Madras, Mandy Guo, Austin Waters, Oliver Wang, Joshua Ainslie, Jason Baldridge, Han Zhang, Garima Pruthi, Jakob Bauer, Feng Yang, Riham Mansour, Jason Gelman, Yang Xu, George Polovets, Ji Liu, Honglong Cai, Warren Chen, XiangHai Sheng, Emily Xue, Sherjil Ozair, Christof Angermueller, Xiaowei Li, Anoop Sinha, Weiren Wang, Julia Wiesinger, Emmanouil Koukoumidis, Yuan Tian, Anand Iyer, Madhu Gurumurthy, Mark Goldenson, Parashar Shah, MK Blake, Hongkun Yu, Anthony Urbanowicz, Jennimaria Palomaki, Chrisantha Fernando, Ken Durden, Harsh Mehta, Nikola Momchev, Elahe Rahimtoroghi, Maria Georgaki, Amit Raul, Sebastian Ruder, Morgan Redshaw, Jinhyuk Lee, Denny Zhou, Komal Jalan, Dinghua Li, Blake Hechtman, Parker Schuh, Milad Nasr, Kieran Milan, Vladimir Mikulik, Juliana Franco, Tim Green, Nam Nguyen, Joe Kelley, Aroma Mahendru, Andrea Hu, Joshua Howland, Ben Vargas, Jeffrey Hui, Kshitij Bansal, Vikram Rao, Rakesh Ghiya, Emma Wang, Ke Ye, Jean Michel Sarr, Melanie Moranski Preston, Madeleine Elish, Steve Li, Aakash Kaku, Jigar Gupta, Ice Pasupat, Da-Cheng Juan, Milan Someswar, Tejvi M., Xinyun Chen, Aida Amini, Alex Fabrikant, Eric Chu, Xuanyi Dong, Amruta Muthal, Senaka Buthpitiya, Sarthak

Jauhari, Nan Hua, Urvashi Khandelwal, Ayal Hitron, Jie Ren, Larissa Rinaldi, Shahar Drath, Avigail Dabush, Nan-Jiang Jiang, Harshal Godhia, Uli Sachs, Anthony Chen, Yicheng Fan, Hagai Taitelbaum, Hila Noga, Zhuyun Dai, James Wang, Chen Liang, Jenny Hamer, ChunSung Ferng, Chenel Elkind, Aviel Atias, Paulina Lee, Vít Listík, Mathias Carlen, Jan van de Kerkhof, Marcin Pikus, Krunoslav Zaher, Paul Müller, Sasha Zykova, Richard Stefanec, Vitaly Gatsko, Christoph Hirnschall, Ashwin Sethi, Xingyu Federico Xu, Chetan Ahuja, Beth Tsai, Anca Stefanoiu, Bo Feng, Keshav Dhandhania, Manish Katyal, Akshay Gupta, Atharva Parulekar, Divya Pitta, Jing Zhao, Vivaan Bhatia, Yashodha Bhavnani, Omar Alhadlaq, Xiaolin Li, Peter Danenberg, Dennis Tu, Alex Pine, Vera Filippova, Abhipso Ghosh, Ben Limonchik, Bhargava Urala, Chaitanya Krishna Lanka, Derik Clive, Yi Sun, Edward Li, Hao Wu, Kevin Hongtongsak, Ianna Li, Kalind Thakkar, Kuanysh Omarov, Kushal Majmundar, Michael Alverson, Michael Kucharski, Mohak Patel, Mudit Jain, Maksim Zabelin, Paolo Pelagatti, Rohan Kohli, Saurabh Kumar, Joseph Kim, Swetha Sankar, Vineet Shah, Lakshmi Ramachandruni, Xiangkai Zeng, Ben Bariach, Laura Weidinger, Tu Vu, Alek Andreev, Antoine He, Kevin Hui, Sheleem Kashem, Amar Subramanya, Sissie Hsiao, Demis Hassabis, Koray Kavukcuoglu, Adam Sadovsky, Quoc Le, Trevor Strohman, Yonghui Wu, Slav Petrov, Jeffrey Dean, and Oriol Vinyals. Gemini: A family of highly capable multimodal models, 2024. URL https: //arxiv.org/abs/2312.11805.

Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. Powersgd: Practical low-rank gradient compression for distributed optimization. Advances in Neural Information Processing Systems (NeurIPS), 2019. URL https: //arxiv.org/abs/1905.13727.

Haolin Wang, Xuefeng Liu, Jianwei Niu, Wenkai Guo, and Shaojie Tang. Why go full? elevating federated learning through partial network updates. Advances in Neural Information Processing Systems (NeurIPS), 2024. URL https://arxiv.org/abs/2410.11559.

Jue Wang, Yucheng Lu, Binhang Yuan, Beidi

Chen, Percy Liang, Christopher De Sa, Christopher Re, and Ce Zhang. Cocktailsgd: finetuning foundation models over 500mbps networks. International Conference on Machine Learning (ICML), 2023. URL https:// openreview.net/forum?id=w2Vrl0zlzA.

Dingzhu Wen, Ki-Jun Jeon, and Kaibin Huang. Federated dropout – a simple approach for enabling federated learning on resource constrained devices. IEEE Wireless Communications Letters, 2022. URL https://arxiv.org/ abs/2109.15258.

Mitchell Wortsman, Maxwell Horton, Carlos Guestrin, Ali Farhadi, and Mohammad Rastegari. Learning neural network subspaces. International Conference on Machine Learning (ICML), 2021. URL https://arxiv.org/ abs/2102.10472.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael GontijoLopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. International Conference on Machine Learning (ICML), 2022. URL https: //arxiv.org/abs/2203.05482.

Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohldickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, and Simon Kornblith. Small-scale proxies for large-scale transformer training instabilities. arXiv preprint library, 2023. URL https://arxiv.org/abs/2309.14322.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. Ties-merging: Resolving interference when merging models. Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https: //arxiv.org/abs/2306.01708.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models

- as a free lunch. International Conference on Machine Learning (ICML), 2024. URL https: //arxiv.org/abs/2311.03099.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL Short Papers), 2019. URL https://arxiv.org/abs/ 1905.07830.

Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian. Galore: Memory-efficient llm training by gradient low-rank projection. International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/ abs/2403.03507.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. Pytorch fsdp: Experiences on scaling fully sharded data parallel, 2023. URL https://arxiv.org/abs/ 2304.11277.

#### Supplementary Materials

Architecture hyperparameters. We detail the architecture across model scales in Table 2. The token budget per scale is computed from the chinchilla-optimal amount of flops (Hoffmann et al., 2022).

Model scale Hidden dim Num layers Num heads Token budget 35M 2,048 6 8 700M 100M 3,072 9 12 1.5B 200M 4,096 12 16 3.5B 300M 5,120 15 20 6B 500M 6,144 18 24 11B 1B 8,192 24 32 25B 4B 12,288 36 48 83B

- Table 2 | Architecture hyperparameters: we consider model from 35M to 4B with the following hyperameters and chinchilla-optimal token budget. For all model scale, the vocabulary size is 32,000.

Number of replicas. We perform our main experiments with 2 replicas for simplicity but showcase in Figure 12 an ablation of DiLoCo vs Streaming DiLoCo where the number of replicas 𝑀 vary from 2 to 8. Contrarely to (Douillard et al., 2024a), we keep the total token budget constant. In Figure 12a, we keep the global batch size constant, and thus reduce the local per-replica batch size). In Figure 12b, we keep the local per-replica batch size constant, and thus increase the global batch size but also reduce the total number of steps.

Number of inner steps. The number of inner steps 𝐻, has an engineering effect and a learning effect: a larger 𝐻 means less frequent synchronization and thus less required bandwidth. On the other hand, a too small 𝐻 produce noisy small-normed outer gradients and a too high 𝐻 will see replicas drifting apart. Therefore, some middle ground needs to be found. We ablate in Figure 13 and find that while Streaming DiLoCo has similar behavior as DiLoCo when 𝐻 increases, it is more robust to low values of 𝐻.

Which parameters to evaluate. We considered multiple subset of the parameters to use for evaluation: 1) the arbitrarily chosen first replica (𝜃1), 2) an average of all replicas (𝑀1 𝑚 𝑀=1 𝜃𝑚), or 3) the globally shared outer parameters (𝜃). Note that the latter is made of fragments that were synchronized

- at different points in time. We show the performance of each subset in Table 3: The difference here between these methods is small, but the outer parameters yield slightly better performance.

Sequential vs strided patterns. The choice of the synchronization pattern (Figure 2), has a slight impact on the ML performance (Figure 6a) but also on the compute utilization (Figure 7). Indeed, as better seen in Figure 14, the strided pattern will never have multiple early layers to be synchronize together. Therefore, it is easier to overlap their communication with the first few layers’ forward of the next step.

Parameters evaluated Eval Loss HellaSwag

First replica 2.77 37.77 Replicas average 2.68 37.72 Outer parameters 2.67 37.78

- Table 3 | Which parameters to evaluate?: Evaluating the outer parameters, where each fragment

has been synchronized at a different moment in time, yields better performance than any inner parameters.

Model size # layers Step time Method

Gbit/s to reach a compute utilization CU =?

50% 80% 90% 95% 99%

1B 24 0.1s

Data-Parallel 86.8 152.6 184.2 222.3 569.0

Vanilla DiLoCo 1.4 6.2 13.3 23.3 86.8 Streaming DiLoCo 1.4 5.2 9.1 16.0 28.1 Streaming DiLoCo w/ overlapped com. 1.4 4.3 6.2 9.1 11.0

Streaming DiLoCo w/ overlapped FP4 com. 0.4 0.9 1.7 2.0 3.0

10B 48 0.8s

Data-Parallel 104.8 222.3 222.3 268.3 471.5 Vanilla DiLoCo 1.7 7.5 16.0 33.9 104.8

Streaming DiLoCo 1.7 5.2 9.1 13.3 19.3

Streaming DiLoCo w/ overlapped com. 1.7 3.6 5.2 6.2 7.5

- Streaming DiLoCo w/ overlapped FP4 com. 0.4 0.9 1.4 1.4 1.7

100B 108 4.9s

Data-Parallel 184.2 323.8 390.7 390.7 471.5 Vanilla DiLoCo 3.0 11.0 23.3 49.4 184.2

Streaming DiLoCo 2.4 6.2 9.1 11.0 19.3

Streaming DiLoCo w/ overlapped com. 1.7 3.6 4.3 5.2 5.2

- Streaming DiLoCo w/ overlapped FP4 com. 0.5 0.9 1.1 1.1 1.4

- Table 4 | Simulation: we estimate the step time (pure compute) of 10B and 100B based on the required flops using Kaplan et al. (2020) rule and using a MFU of 60%. For all DiLoCo and Streaming DiLoCo-variants, we use 𝐻 = 100. For all Streaming DiLoCo-variants, we use a fragment size of 3 layers.

Compute utilization. We report in Table 4 the amount of Gbit/s required, per method, to reach a certain level of compute utilization. See Figure 4 for a figure view of this table. For DiLoCo and Streaming DiLoco (and variants thereof), we use 𝐻 = 100 inner steps. For Streaming DiLoCo (and variants thereof), we use a fixed fragment size of 3 layers; therefore, deeper networks have more fragments: for 1B, 10B, and 100B model scales, it is respectively 8, 16, and 36 fragments. Also, respectively per model scales, a fragment is synchronized every 11, 5, and 2 steps. While the synchronization seems to be more frequent for deeper networks, from the perspective of particular fragment, it is synchronized roughly every 𝐻 = 100 steps. To estimate the compute utilization in

- Table 4 and Figure 4, the time spent per step doing computation (forward & backward) is critical: we report respectively 0.1s, 0.8s, and 4.9s based on each model scale flops profile, a reasonable amount of chips, and a MFU of 60%.

Compute utilization with various speeds. Varying the time spent per step to do pure computation (forward & backward) affects the compute utilization: e.g. for a fixed bandwidth and thus fixed communication time, longer step time, will improve compute utilization. We report in Figure 15, simulated compute utilization when using, at 100B model scale, a compute step time of 1 second, 5 seconds, and 10 seconds.

Model size Flops Method Eval Loss ↓ HellaSwag ↑ Piqa ↑ Arc Easy ↑

Data-Parallel 3.51 24.62 57.89 29.65 DiLoCo H=30 3.54 24.53 58.11 29.65 Streaming DiLoCo with overlapped FP4 com., H=30 3.53 24.46 57.67 30.53 Streaming DiLoCo with overlapped FP4 com., H=100 3.56 24.80 57.89 29.12

35M 1.5e17

Data-Parallel 3.19 26.94 60.12 30.35 DiLoCo H=30 3.21 26.59 60.50 29.12 Streaming DiLoCo with overlapped FP4 com., H=30 3.21 26.97 59.58 31.40 Streaming DiLoCo with overlapped FP4 com., H=100 3.22 26.68 60.39 31.93

100M 9.4e17

Data-Parallel 2.97 29.86 63.71 35.44 DiLoCo H=30 2.98 29.71 62.30 33.68 Streaming DiLoCo with overlapped FP4 com., H=30 2.98 29.67 61.92 34.39 Streaming DiLoCo with overlapped FP4 com., H=100 3.00 29.27 62.13 34.21

200M 4e18

Data-Parallel 2.80 33.46 64.69 34.91 DiLoCo H=30 2.81 33.87 64.74 34.74 Streaming DiLoCo with overlapped FP4 com., H=30 2.81 33.66 63.49 35.09 Streaming DiLoCo with overlapped FP4 com., H=100 2.83 33.00 63.71 34.39

300M 1.4e19

Data-Parallel 2.67 38.68 66.49 37.19 DiLoCo H=30 2.68 38.37 65.61 36.32 Streaming DiLoCo with overlapped FP4 com., H=30 2.67 38.10 66.21 34.91 Streaming DiLoCo with overlapped FP4 com., H=100 2.69 37.40 65.51 34.74

500M 4.7e19

Data-Parallel 2.49 46.60 68.93 39.65 DiLoCo H=30 2.49 46.56 68.82 36.84 Streaming DiLoCo with overlapped FP4 com., H=30 2.48 46.60 69.04 39.12 Streaming DiLoCo with overlapped FP4 com., H=100 2.50 46.00 68.82 38.42

1B 1.9e20

Data-Parallel 2.25 59.56 72.42 43.51

DiLoCo H=30 - - - Streaming DiLoCo with overlapped FP4 com., H=30 - - - -

4B 2e21

Streaming DiLoCo with overlapped FP4 com., H=100 2.26 59.02 72.52 43.16

- Table 5 | Scaling from 35 million parameters to 4 billion parameters using a chinchilla-optimal number of flops/tokens. We train on the C4 dataset, and report the evaluation loss on its validation set.

Compute Utilization on Llama and DeepSeek. We estimate in Figure 16, the compute utilization of our method vs baselines on top of Llama405 (Grattafiori et al., 2024) and DeepSeek-V3 (DeepSeek-AI et al., 2024). For each, we estimate their step time from the respective paper: 26.9 seconds for Llama (first stage of pretraining) and 20.1 seconds for DeepSeek, using the most charitable estimation everytime. Notably, for DeepSeek-V3 (Figure 16b), only 35 billion parameters are activated per token due to their MoE architecture (Shazeer et al., 2017). However, the total 671 billion parameters are synchronized between replicas, massively increasing the amount of bits to transfer. In that case, in our simulation, our method (in red) can be close to 100% compute utilization with 4 Gbits per second vs 1 Tbit per second for Data-Parallel.

Scaling performance. We report in Table 5, the evaluation loss on C4 and accuracy on HellaSwag (Zellers et al., 2019), Piqa (Bisk et al., 2020), and Arc-Easy (Clark et al., 2018), for four different methods across 6 model scales. See subsubsection 3.2.1 for the initial discussion. Performance across scales are roughly similar among all considered methods, with usually a slight advantage for Data-Parallel. We found in practice this advantage to disappear when doing a more realistic overtraining with larger token budget in subsubsection 3.2.2.

Model size Flops 𝑀 𝐻 Eval Loss ↓ HellaSwag ↑ Piqa ↑ Arc Easy ↑

2 30 3.53 24.46 57.67 30.53 4 30 3.60 24.50 56.09 28.60 2 100 3.56 24.80 57.89 29.12 4 100 3.64 24.67 56.75 26.84

35M 1.5e17

2 30 3.21 26.97 59.58 31.40 4 30 3.25 26.24 59.74 32.63 2 100 3.22 26.68 60.39 31.93 4 100 3.29 26.54 60.34 29.82

100M 9.4e17

2 30 2.98 29.67 61.92 34.39 4 30 3.02 29.09 62.89 35.44 2 100 3.00 29.27 62.13 34.21 4 100 3.05 28.53 61.10 33.51

200M 4e18

2 30 2.81 33.66 63.49 35.09 4 30 2.84 32.54 64.42 34.74 2 100 2.83 33.00 63.71 34.39 4 100 2.87 32.02 64.25 35.44

300M 1.4e19

2 30 2.67 38.10 66.21 34.91 4 30 2.70 36.95 65.72 35.26 2 100 2.69 37.40 65.51 34.74 4 100 2.73 36.02 66.27 35.09

500M 4.7e19

2 30 2.48 46.60 69.04 39.12 4 30 2.50 45.25 67.95 39.12 2 100 2.50 46.00 68.82 38.42 4 100 2.53 44.74 68.34 38.25

1B 1.9e20

- Table 6 | Scaling from 35 million parameters to 1 billion parameters Streaming DiLoCo with overlapped FP4 communication and with two different synchronization frequencies 𝐻 = {30, 100} and number of DiLoCo replicas 𝑀 = {2, 4}.

Scaling with variable number of replicas. Contrarely to Data-Parallel, changing the number of replicas for DiLoCo is not mathematically equivalent due to the local training, happening independentely for each replicas. We display in Table 6, a scaling from 35 million parameters to 1 billion parameters on the C4 dataset of our method, Streaming DiLoCo with overlapped FP4 communication, with different number of replicas 𝑀 = {2, 4} and different frequencies of synchronization 𝐻 = {30, 100}. Likewise, in Table 7, we showcase token budget overtraining at 1 billion parameters on the Dolma dataset.

Outer gradients’ cosine similarity. We observe in Figure 17 the cosine similarity per scale between each replica’s outer gradients for respectively all parameters but the embeddings (Figure 17a) and only the embeddings (Figure 17b). For both, the cosine similarity starts from slightly correlated (≈ 0.1), spends of the training time to be close to orthogonal (≈ 0.0), and ends slightly inversely correlated (≈ −0.1) as we reach the fluctuation phase. Note also that the larger the model size, the lower is overall the cosine similarity.

We also plot in Figure 18 the cosine similarity per scale and per transformer layer. Notably, the first

Method Token Budget Terabytes exchanged ↓ Eval Loss ↓ HellaSwag ↑ Piqa ↑ Arc Easy ↑

25B 441 2.67 42.09 67.35 40.42 100B 1,767 2.52 49.78 69.15 44.03 250B 4,418 2.45 53.86 70.45 44.21

Data-Parallel

###### 25B 1.10 2.66 42.08 67.46 38.42 100B 4.42 2.51 49.98 69.96 44.03 250B 11.05 2.45 54.24 71.38 41.92

Our method, M=2

25B 0.55 2.73 38.93 66.92 39.64 100B 2.21 2.54 48.35 69.42 40.52 250B 5.52 2.47 52.20 70.29 42.45

Our method, M=4

- Table 7 | Overtraining on the Dolma dataset with a 1 billion parameters model, and with an increasing token budgets (25B, 100B, and 250B). We report here for our model both with 𝑀 = 2 and 𝑀 = 4 DiLoCo replicas. With twice more replicas, the global batch size is doubled, and twice less steps are done. It is also thus roughly twice faster, but come with slightly worse performance. Our method is the final model: Streaming DiLoCo with overlapped FP4 communication.

transformer layer at each scale has a significantly higher similarity, at every model scales.

[Figure 70]

- (a) Keeping the global batch size constant, and thus decreasing the local per-replica batch size.

[Figure 71]

- (b) Keeping the local per-replica batch size constant, and thus increasing the global batch size.

- Figure 12 | Scaling the number of DiLoCo replicas 𝑀 from 𝑀 = 2 to 𝑀 = 4. For all experiments, the token budget is kept constant.

[Figure 72]

###### Figure 13 | Varying the number of inner steps 𝐻 for DiLoCo and Streaming DiLoCo while keeping the total number of steps constants. A lower 𝐻 means more communication rounds to be done.

[Figure 73]

[Figure 74]

###### Figure 14 | Simulation of a schedule interleaving forward passes (in blue), backward passes w.r.t. activations and weights (resp. in light and dark green), and (outer) gradient reduction (in purple) for Streaming DiLoCo, respectively with a sequential and strided pattern.

[Figure 75]

[Figure 76]

[Figure 77]

(a) 1s step time (b) 5s step time (c) 10s step time

###### Figure 15 | Compute Utilization for a 100 billion parameters when the step time (pure compute) is 1 second, 5 seconds, and 10 seconds.

[Figure 78]

[Figure 79]

(a) Llama405B. (b) DeepSeek-V3 (671B total, 35B activated).

- Figure 16 | Compute Utilization simulated across a range of bandwidth for Llama405 and DeepSeekV3, using step time estimated from respective papers.

[Figure 80]

(a) All fragments but the embedding (b) Embedding fragment

[Figure 81]

- Figure 17 | Cosine similarity between the outer gradients across scales.

[Figure 82]

- Figure 18 | Cosine similarity between the outer gradients across scales. Each line is a transformer layer, with darker colors being earlier layers and lighter colors later layers.

