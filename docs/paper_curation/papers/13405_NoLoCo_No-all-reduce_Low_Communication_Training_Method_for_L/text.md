arXiv:2506.10911v1[cs.LG]12Jun2025

# NoLoCo: No-all-reduce Low Communication Training Method for Large Models

Jari Kolehmainen1, Nikolay Blagoev1, John Donaghy1, Oğuzhan Ersoy1 and Christopher Nies1 1Gensyn

Training large language models is generally done via optimization methods on clusters containing tens of thousands of accelerators, communicating over a high-bandwidth interconnect. Scaling up these clusters is expensive and can become impractical, imposing limits on the size of models that can be trained. Several recent studies have proposed training methods that are less communication intensive, avoiding the need for a highly connected compute cluster. These state-of-the-art low communication training methods still employ a synchronization step for model parameters, which, when performed over all model replicas, can become costly on a low-bandwidth network. In this work, we propose a novel optimization method, NoLoCo, that does not explicitly synchronize all model parameters during training and, as a result, does not require any collective communication. NoLoCo implicitly synchronizes model weights via a novel variant of the Nesterov momentum optimizer by partially averaging model weights with a randomly selected other one. We provide both a theoretical convergence analysis for our proposed optimizer as well as empirical results from language model training. We benchmark NoLoCo on a wide range of accelerator counts and model sizes, between 125M to 6.8B parameters. Our method requires significantly less communication overhead than fully sharded data parallel training or even widely used low communication training method, DiLoCo. The synchronization step itself is estimated to be one magnitude faster than the all-reduce used in DiLoCo for few hundred accelerators training over the internet. We also do not have any global blocking communication that reduces accelerator idling time. Compared to DiLoCo, we also observe up to 4% faster convergence rate with wide range of model sizes and accelerator counts.

## 1. Introduction

Large transformer models have recently shown impressive performance on a wide variety of tasks, including natural language understanding Liu et al. (2024); Team et al. (2023); Touvron et al. (2023); Zhang et al. (2022); image related tasks Zhang et al. (2025); Gao et al. (2024); Zhu et al. (2023); Lin et al. (2023); or speech recognition and generation Maiti et al. (2024); Gourav et al. (2024); Rubenstein et al. (2023); Xu et al. (2025). These large models are usually trained by a combination of different distributed training methods such as data parallelism Rasley et al. (2020), pipeline parallelism Huang

- et al. (2019); Narayanan et al. (2021); Sun et al. (2024), and others Shoeybi et al. (2019); Rasley
- et al. (2020); Liu et al. (2023); Shyam et al. (2024); Fujii et al. (2024); Liu et al. (2024); Cai et al.

(2024). The aforementioned training methods are bandwidth intensive and require a high-bandwidth interconnection fabric available between individual compute nodes that is generally only available in dedicated machine learning clusters Team et al. (2023); Grattafiori et al. (2024); Duan et al. (2024); Intelligence (2024). This requirement increases the cost of training and poses a limit on the training scale as highly connected computer clusters cannot be scaled easily beyond a data center.

Recently, a number of studies have aimed to address this issue by proposing algorithms that scale better

Corresponding author(s): jari@gensyn.ai © 2025 Gensyn. All rights reserved.

than traditional distributed training algorithms in low-bandwidth and high latency network Douillard et al. (2023, 2024); Ryabinin et al. (2023); Li et al. (2022); Biswas et al. (2024); Kale et al. (2025); Charles et al. (2025). Most of these methods use an explicit step to synchronize the data parallel instances of the model during training, typically using all-reduce operations. This synchronization step can take several minutes in highly distributed network and lead to poor overall scaling efficiency of the training algorithm Jaghouar et al. (2024); Yuan et al. (2022).

The main contribution of this paper is to propose a novel optimization method, NoLoCo, for training large neural models that does not use any explicit all-to-all communication. NoLoCo is built upon the inner-outer optimizer paradigm together with the idea of epidemic learning where averaging is done among subset of accelerators instead of all. Specifically, in NoLoCo, outer synchronization is done via only pairs of the accelerators, rather than all of them. Moreover, each inner optimizer step is done via random pipeline routing of accelerators which implicitly helps different pipelines to converge with less synchronisation. Furthermore, we modify the Nesterov momentum optimizer to prevent the weights of the same stage from diverging. We also provide a theoretical convergence analysis on our modified optimizer step.

We test our method with a state of the art low communication method, DiLoCo, and with a traditional distributed model training in the language modeling task with two datasets (Pushshift Reddit and C4) and several model sizes (125M, 1.3B and 6.8B parameters). Our experimental results show that our method is more communication efficient and also converges up to 4% faster than DiLoCo for wide range worker counts, and model sizes. The speed-up from omitting the all-to-all communication increases with increasing number of workers and network latency variance.

We also study how the random pipeline routing between different model stages impacts the convergence. We show that while it can aid in load balancing across different workers Ryabinin et al. (2023), it also hinders the convergence of validation loss slightly. This effect become less pronounced for larger models.

The paper is structured as follows. We first describe relevant work in Section 2. This is followed by Section 3 presenting our method, NoLoCo. In Section 4 we describe - for ease of reproducibility - the details of experiments. Results are presented and discussed in Section 5. Source code for running the experiments is available in GitHub.

## 2. Related Work

### 2.1. Decentralized Training Methods

The common data-parallel optimization algorithm keeps all model parameters synchronized across data-parallel workers by always performing an all-reduction on the gradients before the optimizer step Rasley et al. (2020). Decentralized training methods such as DiLoCo have relaxed this assumption by allowing the model parameters to diverge during local steps and only synchronizing them at the outer optimizer steps, performed less frequently Douillard et al. (2023). Specifically, DiLoCo and its variations Douillard et al. (2024); Li et al. (2022); Biswas et al. (2024); Kale et al. (2025); Peng et al. (2024); Charles et al. (2025) divide the optimization process into inner and outer optimizer steps similar to look ahead optimizer Zhang et al. (2019). During the inner optimizer steps, only local weights are updated, and different copies of the model can have different weights. The outer optimization step uses local weights to compute an outer gradient that is applied to update global weights shared between all copies of the model Douillard et al. (2023).

DiLoCo greatly reduces the frequency of all-reduces compared with regular data parallel training as they are only performed during the outer optimizer steps as opposed to every optimizer step. When accelerator connections are slow or there are enormous number of accelerators, the DiLoCo outer optimizer steps can consume a significant amount of time Jaghouar et al. (2024); Yuan et al. (2022).

Recent studies have aimed to address this by staggering the all-reduce communication with compute Kale et al. (2025) or only synchronizing a subset of model parameters Biswas et al. (2024).

Methods using outer optimizer steps Douillard et al. (2023); Kale et al. (2025); Biswas et al. (2024) still require all-to-all communication during the outer optimizer steps. To overcome this, several studies in the federated learning paradigm Hegedűs et al. (2019); De Vos et al. (2023); Du et al. (2024); Dhasade et al. (2025); Sad et al. (2025) have proposed various training methods that aim to avoid explicit all-to-all communication by replacing it with local communication. For example, epidemic learning De Vos et al. (2023) proposed the use of local averaging to synchronize the subset of weights during training instead of all weights to avoid all-reduces.

### 2.2. Dynamic Pipeline Routing

Pipeline parallelism Huang et al. (2019); Narayanan et al. (2021); Sun et al. (2024) is a popular choice for model parallelism in a low bandwidth environment as it requires less network bandwidth and involves less blocking communication than the fully sharded data parallel training with ZeRO-3 Rasley et al. (2020) or the tensor parallelism Shoeybi et al. (2019). In pipeline parallel training, model is split to consecutive stages where each model passes it’s compute outputs to the next stage and receive inputs from the prior stage Huang et al. (2019). During forward pass model stages have to wait for inputs from the prior stage and during backward pass for gradients from the subsequent stage. This leads to formation of the computation bubble where certain devices will be inherently idle.

Swarm Ryabinin et al. (2023) is one of the earliest works allowing pipeline stages receive inputs from arbitrary replicas of the prior model stage and vice-versa. In this setting, a model stage can start computing immediately once inputs are available from any replica of the prior stage as opposed to waiting for a dedicated model replica, which is the case for regular pipeline parallelism. The approach effectively reduces blocking communication and renders itself well for load balancing Ryabinin et al. (2023). In SWARM, pipeline routing is done based on the load balancing using a message queue like setup. For equal workers and uniform network topology, this becomes essentially random routing. In this study, we will employ random routing as it is a good proxy for the SWARM routing process from optimizer convergence point of view.

Later DiPaCo Douillard et al. (2024) proposed similar setup to SWARM, but used an explicit routing model loosely related to mixture-of-expert (MoE) parallelism Liu et al. (2024); Cai et al. (2024). The main difference to between DiPaCo and the MoE parallelism is that in DiPaCo the routing is done at the sample level while MoE parallelism is generally done at the token level; and that the routing model in DiPaCo is a separate model while in MoE models it is part of the model itself Cai et al. (2024). The routing model in DiPaCo also need to be trained before the actual training.

DiPaCo can theoretically produce less correlated outer gradients due to the router, which is desirable for estimating the expected value using sample means. However, it can also lead to load balancing issues similar to standard MoE parallelism Cai et al. (2024). In addition, having different token counts used within the inner optimizer steps by different workers can degrade outer gradient estimates. DiPaCo aimed to address this by using weighted averaging where the weights are given as the ratio of tokens used by the worker and all tokens used by all workers.

## 3. NoLoCo

NoLoCo utilizes Data Parallelism (DP) and Pipeline Parallelism (PP) methods with the following modifications: (i) for inner optimization step of PP, at each iteration, different pipeline paths are chosen among the accelerators rather than having a fixed path, (ii) for outer optimization step of DP, only pairs of accelerators are synchronized rather than all. To prevent the weights of the same stage from

##### (A) (B)

[Figure 1]

[Figure 2]

- Figure 1 (A): Illustration of dynamic PP routing with DP. Model is split to consecutive pipeline stages (shown vertically), and each stage is replicated to process data in parallel (shown horizontally). (B): Illustration of different terms of the outer momentum term for group consisting of two workers. Red dots show the current slow weights. Yellow dot shows the average of the slow weights.

diverging, we modify the Nesterov momentum optimizer step.

### 3.1. Inner optimizer step via dynamic pipeline routing

NoLoCo uses dynamic PP where an accelerator can receive inputs from any instance of the prior pipeline stage and forwards outputs to any instance of next pipeline stage. We use random permutations to group workers, and perform the routing based on the random groups that guarantees good load balancing. This is illustrated in Fig. 1A. During the backward pass, gradients follow the same path that was chosen during the forward pass.

The random routing allows mixture of the weights of different DP components as their inputs and outputs might be used by the same model pipeline stages. We hypothesize this creates an implicit effect to drive the weights of different DP models closer without requiring frequent synchronization.

### 3.2. Outer optimizer step with modified Nesterov momentum

We aim to relax the outer synchronisation even further by synchronizing the model parameters with a smaller local group, rather than the all-to-all reduction performed in DiLoCo-like methods.

Let us assume there are 𝑁 model instances in the whole network, and we synchronize among a randomly chosen local subgroup of 𝑛 instances where 𝑁 ≫ 𝑛. For each iteration we update the local subgroup to obtain information from different workers. In the local subgroup, we have 𝑛 model instances at step 𝑡, 𝜙𝑡,𝑖, where 𝑖 < 𝑛 indicates the model instance. We will refer to 𝜙𝑡,𝑖 as slow weights as in the look-ahead optimizer Zhang et al. (2019). To progress to the next step, we perform 𝑚 local optimizer steps on 𝜙𝑡,𝑖 to obtain fast model weights 𝜃𝑡+1,𝑖. This step is the same to the inner optimizer steps in DiLoCo and similar to look-ahead optimizer inner steps. We use the fast model weights to compute a local outer gradient:

Δ𝑡,𝑖 = 𝜃𝑡+1,𝑖 − 𝜙𝑡,𝑖. (1) We modify the expression for Nesterov momentum by computing it over a local group as opposed to all

data parallel workers and introduce a term to account for difference in the local slow weights:

#### ∑︁

#### ∑︁

1

𝛽 𝑛

𝜙𝑡,𝑗 , (2)

𝛿𝑡,𝑖 = 𝛼𝛿𝑡−1,𝑖 −

Δ𝑡,𝑗 − 𝛾 𝜙𝑡,𝑖 −

𝑛

𝑗

𝑗

where 𝛼 is Nesterov momentum parameter; 𝛽 is the outer learning rate; 𝛾 is a local weight averaging parameter. The expected values are taken over a random sub groups of size 𝑛. If the sub group consists of all model instances, Eq. 2 simplifies to regular DiLoCo outer optimizer momentum and the last term

diminishes: 𝛿𝑡 = 𝛼𝛿𝑡−1 − 𝑛𝛽 𝑗 Δ𝑡,𝑗 . The third term can also be viewed as a rolling average over weight differences between the worker and random workers observed throughout the updates.

Finally, local weights are updated by the momentum 𝛿𝑡,𝑖 the same as in look ahead optimizer: 𝜙𝑡+1,𝑖 = 𝜙𝑡,𝑖 + 𝛿𝑡,𝑖. (3)

- Fig. 1B illustrates relationship between different terms in the method for outer gradient computation involving two workers.

In our experiments, we use the minimum group size for random subgroups, which is 𝑛 = 2. During the outer optimizer step, workers have to share their outer gradients (Eq. 1) and prior slow weights 𝜙𝑡,𝑖. The prior slow weights can be communicated already at the end of the prior outer step. This allows overlapping communication of the slow weights with the computation for the next fast weights.

Intuitively, the method should have convergence properties very similar to those of DiLoCo as the mean term in Eq. 2 is a rolling average of the slow weights over the duration of training. We prove this by showing that the modified Nesterov optimizer given by Eq. 2 converges to the optima 𝜃 = 0 for a quadratic loss of the form L(𝜃) = 12(𝜃 − 𝑐)TA(𝜃 − 𝑐), where 𝑐 ∼ N(0, Σ) with a constant covariance matrix Σ, and A is a positive definite symmetric matrix Schaul et al. (2013); Wu et al. (2018); Zhang et al. (2019). We also assume that the inner optimizer uses stochastic gradient descent with a constant learning rate 𝜔. With these assumptions following theorem hold:

Theorem 1. When the outer iteration step count 𝑡 → ∞, the expected value of the slow weights 𝔼(𝜙𝑡,𝑖) → 0, and the variance 𝕍(𝜙𝑡,𝑖) ∝ 𝜔2.

Proof of the above theorem is given in Appendix A. The variance part of the theorem also hints that the inner learning rate - learning rate schedules in particular - can be an effective way to control the divergence of the weights during the training. One can initiate the training with a large learning rate and decay it towards end of the training to essentially obtain a very tight cluster of models. We present empirical evidence for this in Section 5.

## 4. Experimental Setup

We study the optimization methods in the context of next token prediction task from language modeling. We use two datasets, pushshift reddit data Baumgartner et al. (2020) and C4 Dodge et al. (2021) to probe the training approaches. For benchmarking, we use 10 million tokens hold out from training data for reddit and the validation partition for C4. All data is tokenized by the Llama sentencepiece tokenizer with a vocabulary size of 128, 000 tokens and formatted to sequences of 1024 tokens.

Global batch size and learning rate are taken from OPT Zhang et al. (2022). Each run has a linear warm-up for learning rate with 1000 steps and cosine learning rate schedule applied after the warm-up period to reduce the learning rate by one magnitude compared to the maximum learning rate. All training runs are done over 25, 000 optimizer steps. We explore 3 different model sizes: small, medium

|Parameter Name|Small Medium Large|
|---|---|
|Hidden size Layers Intermediate size Attention heads (Inner) Learning-rate Global batch size Transformer Parameters<br><br>|768 2048 4096<br><br>12 24 32 3072 8192 16, 384<br><br>16 32 32 0.0006 0.0002 0.00012<br><br>0.5M 1M 2M 125M 1.3B 6.8B|

- Table 1 Summary of model hyper-parameters. Batch sizes are expressed in tokens.

and large Llama models with 125M, 1.3B and 6.8B parameters respectively, and all models sharing the same vocabulary size of 128, 000 tokens. Model hyper-parameters are outlined in Table 1.

We use Adam as the (inner) optimizer for all experiments and applied gradient clipping for gradients larger than unity. Both methods use the same outer learning rate, 𝛽 = 0.7. For DiLoCo we use a momentum value of 𝛼 = 0.3 that was found to produce better results in our setting than standard 𝛼 = 0.9; and we apply the outer optimizer step at every 100 steps. For NoLoCo, we use a higher momentum value of 𝛼 = 0.5; the group size of two workers; and employ the optimizer step every at 50 steps. It’s note worthy that with the doubled frequency of outer optimizer steps, NoLoCo still requires much less communication than DiLoCo since 𝑁 ≫ 2 · 𝑛 := 4. All computations are done in bfloat16 numerical precision and multi-head attentions are computed using the flash-attention Dao et al. (2022). Finally, the source for running the experiments is available in GitHub.

## 5. Results and Discussion

### 5.1. Training results

Validation perplexities at the end of the training are shown in Table 2 for both Reddit and C4 datasets. We observe that both methods are slightly worse than fully sharded data parallel (FSDP) training for all the cases considered, typically few percent worse, but in some cases even 30% (C4, Small model, 16 data parallel world size). The perplexity gap is generally larger with smaller models and large data parallel world sizes; and decreases when the model size becomes larger or the data parallel world size becomes smaller. This finding is consistent with the findings of recent study in DiLoCo scaling Charles et al. (2025). In this regard, both NoLoCo and DiLoCo convergence scale similarly respect to data parallel world size and model size.

The perplexity numbers presented here are subject to the training hyperparameters. In particular, the batch size and learning rate were chosen from a study optimized for FSDP and hence are likely sub-optimal for both DiLoCo and NoLoCo. We found that increasing the batch size improved the training method performance (see Appendix C). However, extensive hyper parameter search for all the methods and model sizes is beyond scope of this study and we report most of the results using hyper-parameters from the study Zhang et al. (2022).

Comparing our method with DiLoCo, we observe that NiLoCo is slightly better than DiLoCo in most experiments (all reddit experiments, and most experiments in C4). This seems counterintuitive, as one would expect the variations in model weights during the training to slow the convergence, not improve it. We hypothesize that this could be due to small perturbations in model weights having a regularization effect on training. All training data batches in this study were within the first epoch, but larger datasets have been shown to contain similar text sequences that can cause slight over-training already within the first epoch. This could also explain why we observe better performance in the pushshift reddit data as opposed to C4 which has more variety of topics and hence less likely to experience some level of

Pushshift Reddit C4 Model Total DP PP FSDP DiLoCo NoLoCo FSDP DiLoCo NoLoCo

8 8 1 25.5 27.6 27.3 29.1 35.4 34.5 8 4 2 25.7 26.8 26.4 29.1 32.1 31.3

Small

16 8 2 25.5 27.6 27.2 29.1 34.0 33.1 32 16 2 25.5 29.7 29.1 29.1 39.1 37.7

16 8 2 19.6 21.0 20.5 18.8 21.8 21.1 32 16 2 19.6 21.2 20.7 18.8 23.2 23.4 64 16 4 19.6 21.0 20.9 18.8 22.6 22.9

Medium

Large 64 16 4 16.1 18.0 17.5 15.7 17.3 16.6

- Table 2 Validation perplexity values for different world sizes and models at the end of the training. DP stands for the data parallel world size and PP for the number of pipeline stages. Best perplexity results are highlighted with a bold font. FSDP shows perplexity from fully sharded data parallel training.

over-fitting within the first epoch.

We also compared the methods in a pure data parallel setting without the pipeline parallelism and the random routing. We observe that for NoLoCo there was a minor degradation in the final perplexity and the convergence rate was slightly slower compared with the pipeline parallel case. For DiLoCo, the opposite was true: perplexity was unchanged but we did note a negligible increase in the convergence rate (not shown). We conclude that the random routing has a minor impact on the convergence rate.

| |(A)<br><br>FSDP<br><br>DiLoCo<br><br>NoLoCo| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

|(B)<br><br>FSDP<br><br>DiLoCo<br><br>NoLoCo| | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

50

FSDP

45

(C)

DiLoCo

40

NoLoCo

35

###### PPL

30

25

20

15

0 5000 10000 15000 20000 25000

0 5000 10000 15000 20000 25000

0 5000 10000 15000 20000 25000

Steps

Steps

Steps

- Figure 2 Reddit validation perplexity of different training methods at different optimizer step counts. Solid blue curve is NoLoCo, dashed red curve show FSDP, and dashed green curve is DiLoCo. (A): Small model with data parallel world size of 8 and two pipeline stages; (B): Medium model with data parallel world size of 8 and two pipeline stages. (C): Large model with data parallel world size of 16 and four pipeline stages.

- Fig. 2 show validation perplexity over the course of training for all model sizes. We can observe that the gap between the FSDP and the decentralized methods becomes less with model size and that NoLoCo has slightly lower perplexity towards the end of the training. This can be easily observed in Fig. 3A that shows the convergence of the relative validation perplexity difference between DiLoCo and NoLoCo for Reddit data. The shown relative perplexity difference is computed by

DiLoCo Perplexity − NoLoCo Perplexity FSDP Perplexity

Rel. PPL Diff =

. (4)

We find that NoLoCo has typically a few percent lower perplexity during the beginning and the end of the training while they are fairly close in the intermediate stage of training. The difference in the late stage perplexity depended on the dataset. For pushshift reddit, we observe that NoLoCo is converging

###### (A)

###### (B)

1.0

Small DP=4 PP=2

8

Small DP=8 PP=2

3.0

Small DP=16 PP=2

0.8

2.5

Medium DP=8 PP=2

6

2.0

Medium DP=16 PP=2

0.6

1.5

Large DP=16 PP=4

max

4

σσ

23000 24000 25000

0.4

Rel.PPLDiff(%)

2

0.2

0

0.0

0 5000 10000 15000 20000 25000

0 5000 10000 15000 20000 25000

Steps

Steps

- Figure 3 (A): Relative validation perplexity difference between DiLoCo and NoLoCo throughout training. Perplexity numbers are normalized by the FSDP perplexity at the same optimizer step count. Positive number indicate faster convergence compared to DiLoCo. (B): Standard deviation of the model weights across the data parallel world size normalized by the largest value within the training run.

faster at the late stage while for C4 the results are varying and depended on the model size and the data parallel world size.

Finally, Fig. 3B shows convergence of the different model replicas during the training. We observe that the standard deviation between different data parallel model replicas peaks after the warm-up and converges throughout the training. Theorem 1 suggests that the model instance variance is proportional to the square of the inner learning rate once convergence is reached. We find empirically that the Pearson correlation coefficient of the standard deviation and the learning rate ranges indeed between 0.91 and 0.97 suggesting that the weight variance across different instances is controlled to a high degree by the inner learning rate as predicted by the theorem. Hence, learning rate scheduler can be used effectively to obtain eventual consistency of the weights in NoLoCo.

### 5.2. Effects of Random Pipeline Routing

To better understand the effect of random routing in the pipeline parallel communication, we perform an ablation study comparing random routing with fixed routing. For the fixed routing, nodes only send and receive values from a fixed prior and subsequent model stage, as in typical setups Huang et al. (2019); Narayanan et al. (2021); Sun et al. (2024). We also remove the outer optimizer synchronization (for both routing methods), thus nodes in the same stage never exchange information directly. Thus, without random routing the setup is the same as running separate training jobs without data parallelism. Using this setup we repeat experiments for Reddit using the small and medium model sizes.

We present the results in Fig. 4. We observe that for small model the standard deviation is ∼ 15% lower than in the same run without random routing between different data parallel pipelines. This effect becomes less pronounced for the medium model with larger data parallel world size, namely ∼ 10% lower standard deviation. Thus we observe that through the pipeline parallel training, nodes in the same stage implicitly exchange information about their weights, without directly synchronizing. We attribute the faster convergence observed in Fig. 2 to this fact.

###### (A)

###### (B)

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

95

90

104

85

102

80

(%)

(%)

75

100

base

base

LL

σσ

70

98

65

Small, DP=4

60

96

Medium, DP=8

55

0 2000 4000 6000 8000 10000 12000

0 2000 4000 6000 8000 10000 12000

Steps

Steps

- Figure 4 Results from training without any outer optimizer steps. The baseline loss and model replica variance is computed from DP number of independent runs. The combined training is only using the random pipeline routing to connect the data parallel model instances. (A): Ratio of weight standard deviations between the random routing and without it. (B): Ratio of validation perplexity between the random pipeline routing and without it.

### 5.3. Latency Analysis

We will derive theoretical speed-up for the NoLoCo local averaging compared with a tree all-reduce - a common implementation for all-reduce in collective communication libraries. We consider 𝑛 workers with each having a message send time of 𝑡𝑐 to any other worker. Tree all-reduce is composed of two stages: a reduce to the root of the tree followed by a broadcast from the root of the tree to all leave nodes. The total time that it takes to do this will:

𝑡𝑎𝑙𝑙 ≈ 2𝑡𝑐 log2(𝑛). (5)

For the local averaging with groups of two, the overall time is simply 2𝑡𝑐 and hence the ratio will be ∼ log2(𝑛). The above formula ignores the fact that not all communication takes the same time and in practice the communication time follows a distribution. We model this by assuming that the communication time 𝑡 follows a log-normal distribution 𝑡 ∼ LogNormal(𝜇, 𝜎2). The time it takes for a parent node to receive message from it’s children is the maximum of the children nodes’ sending time:

𝑡𝑙𝑜𝑐𝑎𝑙 = max(𝑡1, 𝑡2). (6)

If 𝑡1 and 𝑡2 are independent identically distributed log-normal random variables, the expected value of 𝑡𝑙𝑜𝑐𝑎𝑙 is given by:

𝜎2

𝜎

2) exp 𝜇 +

. (7)

𝔼(𝑡𝑙𝑜𝑐𝑎𝑙) = 1 + erf(

2

Also, 2𝔼(𝑡𝑙𝑜𝑐𝑎𝑙) is the mean time it takes for the local averaging as that can be viewed as a single step of the tree reduce at the bottom leaf level. Fig. 5A show the ratio of tree-reduce expected time to local averaging expected time in terms of different world sizes and message sending time variances. We can see that the tree-reduce slows significantly when the message sending time standard deviation increases, which is a typical case for public networks. This effect becomes larger for the larger world sizes as expected.

Real networks generally do not have a constant expected value for message times, but this depends on multiple factors such as the relative topology of the communicating nodes. This increases the observed variance of the message send times effectively widening the gap between the all-reduce and local-averaging.

###### (A)

(B)

40

n = 4

2.4

35

n = 16

2.2

30

n = 64

NoLoCo

2.0

n = 256

25

local

)/

n = 1024

)/

1.8

20

1.6

all

15



DiLoCo

1.4

10



1.2

5

1.0

0.0 0.5 1.0 1.5 2.0 2.5

0 20 40 60 80 100 120

std ( t )/ t

Inner Steps in Outer Step

c

- Figure 5 (A): Simulated ratio tree-reduce expected time to local averaging expected time. 𝑛 denotes here a world

size and 𝑡𝑐 = exp(𝜇 + 𝜎2/2) is the expected time for sending single message. (B): Simulated ratio of total training times between DiLoCo and NoLoCo. Each training run consisted of 500 outer optimizer steps and variable number of inner optimizer steps. The inner optimizer step latency is modeled as log-normal distribution with 𝜇 = 1 and 𝜎2 = 0.5.

The above analysis also assumes that all the workers call the all-reduce operation at the same time, but this is in practice not true and workers would arrive to the communication call at different times. We performed similar analysis by modeling the time each inner optimizer step takes as log-normal distribution and observed how long it will take for all processes to finish 250 outer iterations. We omitted the local averaging and all-reduce time to highlight the effect of the global blocking communication present in DiLoCo (and it’s variations), but not in NoLoCo. Results are shown in Fig. 5B. One can observe that the difference between NoLoCo and DiLoCo increases with increasing world size and is ∼ 20% for 100 inner steps within an outer step using 1024 accelerators. Performing outer optimizer steps more often increases the overhead. This overhead originating from the blocking communication is present in addition to the overhead originating from all-reduce latency.

- 6. Summary

We proposed a novel low communication decentralized training method, NoLoCo, that requires only local subgroup synchronization in outer optimizer steps and thereby avoids any collective all-to-all communication. While reducing the synchronisation group, to prevent diverge of the model weights, we introduced a modification on the Nesterov optimizer used in the outer step. We proved that NoLoCo converges in theory and showed in practice that its convergence speed is comparable with fully synchronized DP methods like FSDP. We also compared NoLoCo with a less frequently synchronizing method (DiLoCo) via various model sizes ranging from 125M to 6.8B parameters; two different language modeling datasets (C4 and pushshift reddit); and a number of parallel worker counts. We found that NoLoCo converges up to 4% faster than DiLoCo in our experiments while not using any all-to-all communication. We hypothesize that this is because of the regularization effect coming from slight variations in the different model instances. Speed-up from removing the all-to-all communication outer optimizer steps depends on the standard deviation of the message sending latency and logarithmically on the data parallel worker count.

NoLoCo - unlike DiLoCo - produces an ensemble of models as the weights are not explicitly synchronized. We found that the standard deviation of the model weights across different instances is controlled to a high degree by the inner learning rate. Hence, a learning rate scheduler can be used effectively to

obtain eventual consistency of different model instances to an arbitrary numerical accuracy.

Finally, we explored how random pipeline routing affects the convergence of validation loss and model weight variance across different data parallel instances. We found that random pipeline routing can reduce the variance across different instances up to 15% compared with fixed routing where the data parallel instances do not exchange any information. Interestingly, this synchronization comes at the cost of reduced convergence rate of the validation loss which is up to 4% higher than the fixed routing case. This effect of random routing became less pronounced for larger model size.

We have demonstrated that local averaging is a viable option for training models with low bandwidth and high latency networks. Further work is needed to establish optimal hyper-parameters for our proposed method with different model sizes and accelerator counts, which is beyond the scope of this study. Our experiments are done within a private cluster. In future work, we plan to extend this to geo-distributed locations to benchmark the exact latency improvements of our proposed method.

## References

Jason Baumgartner, Savvas Zannettou, Brian Keegan, Megan Squire, and Jeremy Blackburn. The pushshift reddit dataset. In Proceedings of the international AAAI conference on web and social media, volume 14, pages 830–839, 2020.

Sayan Biswas, Anne-Marie Kermarrec, Alexis Marouani, Rafael Pires, Rishi Sharma, and Martijn De Vos. Boosting asynchronous decentralized learning with model fragmentation. arXiv preprint arXiv:2410.12918, 2024.

Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, and Jiayi Huang. A survey on mixture of experts. arXiv preprint arXiv:2407.06204, 2024.

Zachary Charles, Gabriel Teston, Lucio Dery, Keith Rush, Nova Fallen, Zachary Garrett, Arthur Szlam, and Arthur Douillard. Communication-efficient language model training scales reliably and robustly: Scaling laws for diloco. arXiv preprint arXiv:2503.09799, 2025.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

Martijn De Vos, Sadegh Farhadkhani, Rachid Guerraoui, Anne-Marie Kermarrec, Rafael Pires, and Rishi Sharma. Epidemic learning: Boosting decentralized learning with randomized communication. Advances in Neural Information Processing Systems, 36:36132–36164, 2023.

Akash Dhasade, Anne-Marie Kermarrec, Erick Lavoie, Johan Pouwelse, Rishi Sharma, and Martijn de Vos. Practical federated learning without a server. In Proceedings of the 5th Workshop on Machine Learning and Systems, pages 1–11, 2025.

Jesse Dodge, Maarten Sap, Ana Marasović, William Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, and Matt Gardner. Documenting large webtext corpora: A case study on the colossal clean crawled corpus. arXiv preprint arXiv:2104.08758, 2021.

Arthur Douillard, Qixuan Feng, Andrei A Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc’Aurelio Ranzato, Arthur Szlam, and Jiajun Shen. Diloco: Distributed low-communication training of language models. arXiv preprint arXiv:2311.08105, 2023.

Arthur Douillard, Qixuan Feng, Andrei A Rusu, Adhiguna Kuncoro, Yani Donchev, Rachita Chhaparia, Ionel Gog, Marc’Aurelio Ranzato, Jiajun Shen, and Arthur Szlam. Dipaco: Distributed path composition. arXiv preprint arXiv:2403.10616, 2024.

Haizhou Du, Yijian Chen, Ryan Yang, Yuchen Li, and Linghe Kong. Hyperprism: An adaptive non-linear aggregation framework for distributed machine learning over non-iid data and time-varying communication links. Advances in Neural Information Processing Systems, 37:11814–11836, 2024.

Jiangfei Duan, Shuo Zhang, Zerui Wang, Lijuan Jiang, Wenwen Qu, Qinghao Hu, Guoteng Wang, Qizhen Weng, Hang Yan, Xingcheng Zhang, et al. Efficient training of large language models on distributed infrastructures: a survey. arXiv preprint arXiv:2407.20018, 2024.

Kazuki Fujii, Kohei Watanabe, and Rio Yokota. Accelerating large language model training with 4d parallelism and memory consumption estimator. arXiv preprint arXiv:2411.06465, 2024.

Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. Clip-adapter: Better vision-language models with feature adapters. International Journal of Computer Vision, 132(2):581–595, 2024.

Aditya Gourav, Jari Kolehmainen, Prashanth Shivakumar, Yile Gu, Grant Strimel, Ankur Gandhe, Ariya Rastrow, and Ivan Bulyko. Multi-modal retrieval for large language model based speech recognition. In Findings of the Association for Computational Linguistics ACL 2024, pages 4435–4446, 2024.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

István Hegedűs, Gábor Danner, and Márk Jelasity. Gossip learning as a decentralized alternative to federated learning. In Distributed Applications and Interoperable Systems: 19th IFIP WG 6.1 International Conference, DAIS 2019, Held as Part of the 14th International Federated Conference on Distributed Computing Techniques, DisCoTec 2019, Kongens Lyngby, Denmark, June 17–21, 2019, Proceedings 19, pages 74–90. Springer, 2019.

Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, et al. Gpipe: Efficient training of giant neural networks using pipeline parallelism. Advances in neural information processing systems, 32, 2019.

Amazon Artificial General Intelligence. The amazon nova family of models: Technical report and model card. 2024.

Sami Jaghouar, Jack Min Ong, Manveer Basra, Fares Obeid, Jannik Straube, Michael Keiblinger, Elie Bakouch, Lucas Atkins, Maziyar Panahi, Charles Goddard, et al. Intellect-1 technical report. arXiv preprint arXiv:2412.01152, 2024.

Satyen Kale, Arthur Douillard, and Yanislav Donchev. Eager updates for overlapped communication and computation in diloco. arXiv preprint arXiv:2502.12996, 2025.

Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A Smith, and Luke Zettlemoyer. Branch-train-merge: Embarrassingly parallel training of expert language models. arXiv preprint arXiv:2208.03306, 2022.

Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.

Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise transformers for near-infinite context. arXiv preprint arXiv:2310.01889, 2023.

Soumi Maiti, Yifan Peng, Shukjae Choi, Jee-weon Jung, Xuankai Chang, and Shinji Watanabe. Voxtlm: Unified decoder-only models for consolidating speech recognition, synthesis and speech, text continuation tasks. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 13326–13330. IEEE, 2024.

Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, et al. Efficient large-scale language model training on gpu clusters using megatron-lm. In Proceedings of the international conference for high performance computing, networking, storage and analysis, pages 1–15, 2021.

Bowen Peng, Jeffrey Quesnelle, and Diederik P Kingma. Decoupled momentum optimization. arXiv preprint arXiv:2411.19870, 2024.

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 3505–3506, 2020.

Paul K Rubenstein, Chulayuth Asawaroengchai, Duc Dung Nguyen, Ankur Bapna, Zalán Borsos, Félix de Chaumont Quitry, Peter Chen, Dalia El Badawy, Wei Han, Eugene Kharitonov, et al. Audiopalm: A large language model that can speak and listen. arXiv preprint arXiv:2306.12925, 2023.

Max Ryabinin, Tim Dettmers, Michael Diskin, and Alexander Borzunov. Swarm parallelism: Training large models can be surprisingly communication-efficient. In International Conference on Machine Learning, pages 29416–29440. PMLR, 2023.

Christos Sad, George Retsinas, Dimitrios Soudris, Kostas Siozios, and Dimosthenis Masouros. Towards asynchronous peer-to-peer federated learning for heterogeneous systems. In Proceedings of the 5th Workshop on Machine Learning and Systems, pages 261–268, 2025.

Tom Schaul, Sixin Zhang, and Yann LeCun. No more pesky learning rates. In International conference on machine learning, pages 343–351. PMLR, 2013.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatronlm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Vasudev Shyam, Jonathan Pilault, Emily Shepperd, Quentin Anthony, and Beren Millidge. Tree attention: Topology-aware decoding for long-context attention on gpu clusters. arXiv preprint arXiv:2408.04093, 2024.

Ao Sun, Weilin Zhao, Xu Han, Cheng Yang, Xinrong Zhang, Zhiyuan Liu, Chuan Shi, and Maosong Sun. Seq1f1b: Efficient sequence-level pipeline parallelism for large language model training. arXiv preprint arXiv:2406.03488, 2024.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Yuhuai Wu, Mengye Ren, Renjie Liao, and Roger Grosse. Understanding short-horizon bias in stochastic metaoptimization. arXiv preprint arXiv:1803.02021, 2018.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025.

Binhang Yuan, Yongjun He, Jared Davis, Tianyi Zhang, Tri Dao, Beidi Chen, Percy S Liang, Christopher Re, and Ce Zhang. Decentralized training of foundation models in heterogeneous environments. Advances in Neural Information Processing Systems, 35:25464–25477, 2022.

Michael Zhang, James Lucas, Jimmy Ba, and Geoffrey E Hinton. Lookahead optimizer: k steps forward, 1 step back. Advances in neural information processing systems, 32, 2019.

Shaolei Zhang, Qingkai Fang, Zhe Yang, and Yang Feng. Llava-mini: Efficient image and video large multimodal models with one vision token. arXiv preprint arXiv:2501.03895, 2025.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

## A. Convergence Analysis

### A.1. Expected Value and Variance of Inner Iteration

The proposed method in this study aims to lower the communication overhead of data parallel training via the modified Nesterov momentum expression given by Eq. 2. This appendix shows proof that the modified version converges to the vicinity of the real optima when the hyperparameters are chosen appropriately. We structure the proof as follows: this section gives background context and derives expected value and variance for the inner iterations weights; Section A.2 provides convergence proof for the expected value of slow weights; and finally A.3 shows proof for the variance of the slow weights.

To show this we assume a stochastic loss from prior studies Schaul et al. (2013); Wu et al. (2018); Zhang et al. (2019):

- 1

- 2(𝜃 − 𝑐)TA(𝜃 − 𝑐), (8)

L(𝜃) =

where 𝑐 ∼ N(0, Σ) is a vector valued random variable; A is a positive definite symmetric matrix; and the covariance matrix Σ is a constant Zhang et al. (2019). With the above loss function, the minimum value is obtained at 𝜃 = 0. Gradient of the loss function is given by:

∇𝜃L(𝜃) = A(𝜃 − 𝑐) ∼ N(A𝜃, AΣA). (9)

For this analysis, we assume that the inner optimizer is using the stochastic gradient descent with a constant learning rate. Stochastic gradient descent updates the (fast) weights according to following rule:

𝜃𝑡,𝑖(𝑘+1) = (I − 𝜔A) 𝜃𝑡,𝑖(𝑘) + 𝜔A𝑐, (10) where (𝑘) denotes the (inner) iteration index and 𝜔 is the inner learning rate. With these assumptions, expected value and variance of the fast model weights after one inner iteration are Wu et al. (2018):

𝔼(𝜃𝑡(+𝑗+11,𝑖)) = (I − 𝜔A) 𝔼(𝜃𝑡(+𝑗1),𝑖), (11) 𝕍(𝜃𝑡(+𝑗+11,𝑖)) = (I − 𝜔A) 𝕍(𝜃𝑡(+𝑗1),𝑖) (I − 𝜔A) + 𝜔2AΣA, (12)

where I is the identity matrix, and 𝜃𝑡(+01),𝑖 = 𝜙𝑡,𝑖. To simplify the notation, we define following shorthands: B = I − 𝜔A, (13) U = 𝜔2AΣA. (14)

Solving Eqs. 11 and 12 for expected value and variance, we obtain:

𝔼(𝜃𝑡(+𝑗+11,𝑖)) = B𝑗𝔼(𝜃𝑡(+01),𝑖), (15) 𝕍(𝜃𝑡(+𝑗+11,𝑖)) = B𝑗𝕍(𝜃𝑡(+01),𝑖)B𝑗 + F −1 U − B𝑗UB𝑗 , (16)

where F is a linear function given by:

F (X) = X − BXB. (17) F is invertible when 𝜔 > 0 and F (X) ≡ 0 when 𝜔 = 0. From Eqs. 15 and 16 we can solve for the outer gradient:

𝔼(Δ𝑡,𝑖) = 𝔼(𝜃𝑡(+𝑚1),𝑖 − 𝜙𝑡,𝑖) (18) = (I − 𝜔A)𝑚 𝔼(𝜙𝑡,𝑖) − 𝔼(𝜙𝑡,𝑖) (19) = ((I − 𝜔A)𝑚 − I) 𝔼(𝜙𝑡,𝑖), (20)

where 𝑚 is the number of inner optimizer steps in one outer optimizer step. Similarly for variance,

∑︁ 𝑚

#### 𝑚∑︁−1

𝜃𝑡(+𝑘1),𝑖 − 𝜃𝑡(+𝑘1−,𝑖1) = 𝕍(−𝜔

∇𝜃L(𝜃𝑡(+𝑘1),𝑖)) (21)

𝕍(Δ𝑡,𝑖) = 𝕍

𝑘=1

𝑘=0

#### 𝑚∑︁−1

≈ 𝜔2

𝕍(∇𝜃L(𝜃𝑡(+𝑘1),𝑖)) (22)

𝑘=0

𝑚∑︁−1

= 𝜔2A

𝕍(𝜃𝑡(+𝑘1),𝑖) + Σ A (23)

𝑘=0

𝑚∑︁−1

B𝑘−1𝕍(𝜃𝑡(+01),𝑖)B𝑘−1 + F −1 U − B𝑘−1UB𝑘−1 A + 𝑚U (24)

= 𝜔2A

𝑘=0

𝑚∑︁−1

B𝑘𝕍(𝜃𝑡(+01),𝑖)B𝑘 A + R′ (25)

= 𝜔2A

𝑘=0

= 𝜔2AF −1 𝕍(𝜃𝑡(+01),𝑖) − B𝑘𝕍(𝜃𝑡(+01),𝑖)B𝑘 A + R′, (26)

where R′ is a constant matrix depending on 𝜔, Σ, A, and 𝑚. In Eq. 26 we neglected covariance of non-consecutive fast weights similar to study Zhang et al. (2019). Eq. 26 can be simplified by writing the variance matrix as a vector using following notation:

𝕌(𝜃) ≡ vec(𝕍(𝜃)) = [𝕍(𝜃)1, 𝕍(𝜃)2, · · · ], (27)

where 𝕍(𝜃)1 and 𝕍(𝜃)2 are column vectors of the covariance matrix 𝕍(𝜃); and vec(·) denotes converting a matrix to a vector by concatenating all the column vectors. With this notation we obtain:

𝕌(Δ𝑡,𝑖) = vec 𝜔2AF −1 𝕍(𝜃𝑡(+01),𝑖) − B𝑘𝕍(𝜃𝑡(+01),𝑖)B𝑘 A + R′ (28)

= 𝜔2A ⊗ A (I − B ⊗ B)−1 I − B𝑘 ⊗ B𝑘 𝕌(𝜃𝑡(+01),𝑖) + vec(R′), (29) where ⊗ denotes Kronecker product. We also define:

B𝕍 = 𝜔2A ⊗ A (I − B ⊗ B)−1 I − B𝑘 ⊗ B𝑘 . (30)

- A.2. Expected Value of Outer Iteration We proceed to show following:

- Theorem 2. When the outer iteration step count 𝑡 → ∞, the expected value of the slow weights 𝔼(𝜙𝑡,𝑖) → 0.

We remind the reader that the modified Nesterov momentum used in this study is given by:

#### ∑︁

#### ∑︁

1

𝛽 𝑛

𝜙𝑡,𝑗 , (31)

𝛿𝑡,𝑖 = 𝛼𝛿𝑡−1,𝑖 −

Δ𝑡,𝑗 − 𝛾 𝜙𝑡,𝑖 −

𝑛

𝑗

𝑗

where 𝑛 is the group size used for the sample means.

The different realizations of the slow weights, 𝜙𝑡,𝑗, are generally not independent due to the path decomposition mechanism and due to the previous outer iterations. However, we will assume that they

would be independent for convergence analysis, which introduces an error in the expected value and variance expressions. This error will become smaller when the data parallel world size becomes larger as the coupling between slow weights becomes weaker.

The optimization method will start from the same instance of slow model weights; hence 𝜙0,𝑖 ≡ 𝜙0 as the first starting point. All instances of slow weights are updated by the same algorithm and hence the value of 𝜙𝑡,𝑖 depends on two things: (1) what data is used to compute the inner gradients; and (2) what other instances are used in the sample averages. Both of these processes are identically random regarding different instances, and hence we assume that the slow weights have identical distributions. This has following corollaries.

Lemma 1. The expected values of the slow weights 𝜙𝑡,𝑖 satisfy: 𝔼(𝜙𝑡,𝑖 − 1𝑛 𝑗 𝜙𝑡,𝑗) = 0.

We present a formal proof Lemma 1 in Appendix B. For the variances situation is more complex, and we assume following conjecture based on the above informal reasoning:

2

Conjecture 1. The variances of the slow weights 𝜙𝑡,𝑖 satisfy: 𝕍(𝜙𝑡,𝑖 − 1𝑛 𝑗 𝜙𝑡,𝑗) ≈ 2 𝑛−𝑛1

𝕍(𝜙𝑡,𝑖).

The outer gradients are fundamentally dependent on the current slow weights. We neglect the dependency of the instances outer gradients on other gradients slow weights as before, which will introduce another error in the approximation. This error should diminish as the data parallel world size becomes larger and the coupling between different instances of slow weights becomes weaker. Similar to reasoning with the slow weights, we assume that the outer gradients have identical distributions. With these assumptions, the expected value of 𝛿𝑡,𝑖 becomes:

𝔼(𝛿𝑡,𝑖) = 𝛼𝔼(𝛿𝑡−1,𝑖) + 𝛽𝔼(Δ𝑡,𝑖), (32)

∑︁𝑡

𝛼𝑡−𝑘𝔼(Δ𝑘,𝑖) (33)

= 𝛽

𝑘=0

∑︁𝑡

𝛼𝑡−𝑘 (B𝑚 − I) 𝔼(𝜙𝑘,𝑖) (34)

= 𝛽

𝑘=0

that is the same expression as for regular Nesterov momentum. Expected value of the next iteration slow weights becomes:

𝔼(𝜙𝑡+1,𝑖) = 𝔼(𝜙𝑡,𝑖) + 𝔼(𝛿𝑡,𝑖) (35)

#### ∑︁𝑡

𝛼𝑡−𝑘 (B𝑚 − I) 𝔼(𝜙𝑘,𝑖) (36)

= 𝔼(𝜙𝑡,𝑖) + 𝛽

𝑘=0

∑︁𝑡

𝛼−𝑘𝔼(𝜙𝑘,𝑖) (37)

= 𝔼(𝜙𝑡,𝑖) + 𝛽𝛼𝑡 (B𝑚 − I)

𝑘=0

∑︁𝑡−1

= 𝔼(𝜙𝑡,𝑖) + 𝛽𝛼𝑡 (B𝑚 − I) 𝛼−𝑡𝔼(𝜙𝑡,𝑖) +

𝛼−𝑘𝔼(𝜙𝑘,𝑖) (38) = (I + 𝛽(B𝑚 − I)) 𝔼(𝜙𝑡,𝑖) (39)

𝑘=0

∑︁𝑡−1

+𝛼 −𝔼(𝜙𝑡−1,𝑖) + 𝔼(𝜙𝑡−1,𝑖) + 𝛽𝛼𝑡−1 (B𝑚 − I)

𝛼−𝑘𝔼(𝜙𝑘,𝑖) (40)

𝑘=0

= (I + 𝛽(B𝑚 − I)) 𝔼(𝜙𝑡,𝑖) + 𝛼 𝔼(𝜙𝑡,𝑖) − 𝔼(𝜙𝑡−1,𝑖) (41) = (I + 𝛼I + 𝛽(B𝑚 − I)) 𝔼(𝜙𝑡,𝑖) − 𝛼𝔼(𝜙𝑡−1,𝑖) (42) = D𝔼(𝜙𝑡,𝑖) − 𝛼𝔼(𝜙𝑡−1,𝑖), (43)

where D = I + 𝛼I + 𝛽(B𝑚 − I). Solving 𝔼(𝜙𝑡,𝑖) from the above recursive equation using method of characteristics yields:

𝔼(𝜙𝑡,𝑖) = C1𝑟1𝑡 + C2𝑟2𝑡 , (44)

where C1 and C2 are constants independent of 𝑡; 𝑟1 and 𝑟2 are the matrix roots of the characteristic polynomial given by:

- 𝑟1 =

- 1

- 2

D + √︁D2 − 4𝛼I , (45)

- 𝑟2 =

D − √︁D2 − 4𝛼I . (46)

- 1

- 2

We note that 0 < 𝑟2 ≤ 𝑟1 ≤ D when 0 ≤ 𝛼 < 1, hence it is sufficient to show that D𝑡 → 0 that happens if and only if all eigen values of D, D𝑖, have less than unit absolute value. Recall, that A is symmetric and positive definite and hence has eigen value decomposition: A = QΛQ𝑇 where Q is an orthogonal matrix and Λ is a diagonal matrix with positive non-zero elements at diagonal. This yields:

B𝑚 = I − 𝜔QΛQ𝑇 𝑚 (47)

𝑚

= Q(I − 𝜔Λ)QT

(48)

= Q (I − 𝜔Λ)𝑚 QT. (49) Substituting Eq. 49 to definition of D gives:

D = I + 𝛼I + 𝛽(B𝑚 − I) (50) = I + 𝛼I + 𝛽Q ((I − 𝜔Λ)𝑚 − I) QT (51) = Q (I + (𝛼 − 𝛽)I + 𝛽 (I − 𝜔Λ)𝑚) QT, (52)

where we can identify that the eigen values are:

D𝑖 = 1 + 𝛼 − (1 − (1 − 𝜔Λ𝑖)𝑚)𝛽, (53)

where Λ𝑖 > 0 is the 𝑖th eigen value of A. Convergence of the expected value depends on the hyperparameters 𝛼, 𝛽, 𝜔, and 𝑚. When 𝑚 is sufficiently large and inner learning rate is chosen to satisfy

- 0 < 𝜔Λ𝑖 ≤ 1, sufficient condition is 𝛽 > 𝛼 and 𝔼(𝜙𝑡,𝑖) → 0 when 𝑡 → ∞. Thus the method converges to optimal solution.

- A.3. Variance of Outer Iteration Finally, we will show following:

- Theorem 3. When the outer iteration step count 𝑡 → ∞, the expected value of the slow weights 𝕍(𝜙𝑡,𝑖) ∝ 𝜔2.

The variance of slow weights is given by:

𝕍(𝜙𝑡+1,𝑖) = 𝕍(𝜙𝑡,𝑖) + 𝕍(𝛿𝑡,𝑖) + 2Cov(𝜙𝑡,𝑖, 𝛿𝑡,𝑖) (54) We only consider direct dependencies of slow weights 𝜙𝑡,𝑖 for the covariance term that becomes:

𝑛 − 1 𝑛

Cov(𝜙𝑡,𝑖, 𝛿𝑡,𝑖) ≈ −𝛾2

𝕍(𝜙𝑡,𝑖), (55)

where we omitted the covariance between the outer gradients and corresponding slow weights. The variance of the momentum term becomes:

2

𝛽2 𝑛

𝑛 − 1 𝑛

𝕌(Δ𝑡,𝑖) + 2𝛾2

𝕌(𝛿𝑡,𝑖) ≈ 𝛼2𝕌(𝛿𝑡−1,𝑖) +

𝕌(𝜙𝑡,𝑖) (56)

2

𝛽2B𝕍 𝑛 + 2𝛾2

𝛽2 𝑛

𝑛 − 1 𝑛

= 𝛼2𝕌(𝛿𝑡−1,𝑖) +

vec(R′) (57)

𝕌(𝜙𝑡,𝑖) +

∑︁𝑡−1

2

𝛽2B𝕍 𝑛 + 2𝛾2

𝛽2 𝑛

𝑛 − 1 𝑛

𝛼2(𝑡−1−𝑘)

vec(R′) (58)

=

𝕌(𝜙𝑘,𝑖) +

𝑘=0

#### ∑︁𝑡−1

= 𝛼2𝑡−2

𝛼−2𝑘(C𝕍𝕌(𝜙𝑘,𝑖) + R′′), (59)

𝑘=0

where we assumed that initial momentum is 𝕍(𝛿0,𝑖) ≡ 0, and used shorthands:

2

𝛽2B𝕍 𝑛 + 2𝛾2

𝑛 − 1 𝑛

, (60)

C𝕍 =

𝛽2 𝑛

R′′ =

vec(R′). (61)

Substituting Eqs. 55 and 59 into Eq. 54 yields:

#### ∑︁𝑡−1

𝑛 − 1 𝑛

𝕌(𝜙𝑡+1,𝑖) = 𝕌(𝜙𝑡,𝑖) − 2𝛾2

𝕌(𝜙𝑡,𝑖) + 𝛼2𝑡−2

𝛼−2𝑘(C𝕍𝕌(𝜙𝑘,𝑖) + R′′) (62)

𝑘=0

#### ∑︁𝑡−1

𝑛 − 1 𝑛

= 1 − 2𝛾2

𝕌(𝜙𝑡,𝑖) + 𝛼2𝑡−2

𝛼−2𝑘(C𝕍𝕌(𝜙𝑘,𝑖) + R′′). (63)

𝑘=0

The last sum-term can be rearranged as follows:

#### ∑︁𝑡−1

𝛼2𝑡−2

𝛼−2𝑘(C𝕍𝕌(𝜙𝑘,𝑖) + R′′) (64)

𝑘=0

∑︁𝑡−2

= C𝕍𝕌(𝜙𝑡−1,𝑖) + R′′ + 𝛼2 𝛼2(𝑡−2)

𝛼−2𝑘(C𝕍𝕌(𝜙𝑘,𝑖) + R′′) (65)

𝑘=0

𝑛 − 1 𝑛

= C𝕍𝕌(𝜙𝑡−1,𝑖) + R′′ + 𝛼2 𝕌(𝜙𝑡,𝑖) − 1 − 2𝛾2

𝕌(𝜙𝑡−1,𝑖) (66)

𝑛 − 1 𝑛

= 𝛼2𝕌(𝜙𝑡,𝑖) + C𝕍 − 𝛼2 1 − 2𝛾2

I 𝕌(𝜙𝑡−1,𝑖) + R′′. (67)

Substituting Eq. 67 to Eq. 63 gives:

𝕌(𝜙𝑡+1,𝑖) = 𝑑𝕍𝕌(𝜙𝑡,𝑖) + E𝕍𝕌(𝜙𝑡−1,𝑖) + R′′, (68) where 𝑑𝕍 and E𝕌 are given by:

𝑛 − 1 𝑛

𝑑𝕍 = 1 + 𝛼2 − 2𝛾2

, (69)

𝑛 − 1 𝑛

E𝕍 = C𝕍 − 𝛼2 1 − 2𝛾2

I. (70)

Solving 𝕌(𝜙𝑡,𝑖) by method of characteristics gives following solution: 𝕌(𝜙𝑡,𝑖) = C′

1𝑣1𝑡 + C′

2𝑣2𝑡 + R′′, (71)

where C1′ and C2′ are constants independent of 𝑡; 𝑣1 and 𝑣2 are the matrix roots of the characteristic polynomial given by:

- 𝑣1 =

- 1

- 2

𝑑𝕍I +

√︃𝑑𝕍2I − 4E𝕍 , (72)

- 𝑣2 =

√︃𝑑𝕍2I − 4E𝕍 . (73)

- 1

- 2

𝑑𝕍I −

For real roots, we have following bounds: 0 ≤ ∥𝑣2∥2 ≤ ∥𝑣1∥2 ≤ |𝑑𝕍|. For the variance to remain bounded as 𝑡 → ∞ we must have |𝑑𝕍| < 1. Solving for 𝛾 we arrive at condition:

#### √︂ 𝑛

𝛼 < 𝛾 < √︂ 𝑛 2(𝑛 − 1)

(2 + 𝛼2). (74)

2(𝑛 − 1)

When hyper-parameters satisfy Eq. 74, 𝕌(𝜙𝑡,𝑖) → R′′ when 𝑡 → ∞. The leading order term of R′′ in terms of inner learning rate is ∝ 𝜔2. Hence, the variance ∥𝕌(𝜙𝑡,𝑖)∥2 ∝ 𝜔2 when 𝜔 → 0 which proofs that the method converges to the correct optima and the variance of the optima is proportional to square of inner learning rate.

- B. Proof for Identical Model Expected Values Reminder that after 𝑘 + 1 steps, the expected fast weights are:

𝔼[𝜃𝑡𝑘++11,𝑖] = B𝑘𝔼[𝜙𝑡,𝑖], (75) where B = I − 𝜔A. We prove that for any arbitrary step 𝑡 the expected difference between an arbitrary 𝜙𝑡,𝑖 and the average of several other 1𝑛 𝑛𝑗 𝜙𝑡,𝑗 is 0:

#### ∑︁𝑛

#### ∑︁𝑛

1

1

𝔼[𝜙𝑡,𝑗] = 0. (76)

𝜙𝑡,𝑗] = 𝔼[𝜙𝑡,𝑖] −

𝔼[𝜙𝑡,𝑖 −

𝑛

𝑛

𝑗

𝑗

Reminder about some of the variables:

Δ𝑡,𝑖 = 𝜃𝑡𝑘++11,𝑖 − 𝜙𝑡,𝑖 (77) 𝜙𝑡+1,𝑖 = 𝜙𝑡,𝑖 + 𝛿𝑡,𝑖 (78)

#### ∑︁𝑛

#### ∑︁𝑛

1

𝛽 𝑛

𝜙𝑡,𝑗 . (79)

𝛿𝑡,𝑖 = 𝛼𝛿𝑡−1,𝑖 −

Δ𝑡,𝑗 − 𝛾 𝜙𝑡,𝑖 −

𝑛

𝑗

𝑗

It is trivial to see that at step 1 the property holds: 𝜙0,𝑖 = 𝜙0 (80)

- 𝛿0,𝑖 = 0 (81)
- 𝛿1,𝑖 = −

#### ∑︁𝑛

𝛽 𝑛

𝜃𝑡𝑘++11,𝑖 − 𝜙0 𝔼[𝛿1,𝑖] (82)

𝑗

#### ∑︁𝑛

𝛽 𝑛

B𝑘𝔼[𝜙0] − 𝔼[𝜙0] (83)

= −

𝑗

= −𝛽(B𝑘 − I)𝔼[𝜙0], and (84) 𝔼[𝜙1,𝑖] = 𝔼[𝜙0] − 𝛽(B𝑘 − I)𝔼[𝜙0]. (85)

Since the value does not depend on 𝑖, then the value will be identical across all replicas, thus 𝔼[𝜙1,𝑖] − 𝑛 𝑗 𝔼[𝜙1,𝑗] = 0.

- 1 𝑛

We proceed next to show that if the property holds for an arbitrary step 𝑇 and for all steps prior to that 𝑡 ∈ [0,𝑇], then it also holds for 𝑇 + 1.

At step 𝑇 + 1 we have:

#### ∑︁𝑛

#### ∑︁𝑛

1

𝛽 𝑛

𝜙𝑇,𝑗 . (86)

𝛿𝑇+1,𝑖 = 𝛼𝛿𝑡,𝑖 −

Δ𝑇,𝑗 − 𝛾 𝜙𝑇,𝑖 −

𝑛

𝑗

𝑗

Taking expected value of Eq. 86 yield:

#### ∑︁𝑛

#### ∑︁𝑛

1

𝛽 𝑛

𝔼[𝜙𝑇,𝑗] . (87)

𝔼[𝛿𝑇+1,𝑖] = 𝛼𝔼[𝛿𝑇,𝑖] −

𝔼[Δ𝑇,𝑗] − 𝛾 𝔼[𝜙𝑇,𝑖] −

𝑛

𝑗

𝑗

The right term reduces to 0 and thus we are left with:

∑︁𝑛

𝛽 𝑛 (B𝑘 − I)

𝔼[𝜙𝑇] (88)

𝔼[𝛿𝑇+1,𝑖] = 𝛼𝔼[𝛿𝑇,𝑖] −

𝑗

= 𝛼𝔼[𝛿𝑇,𝑖] − 𝛽(B𝑘 − I)𝔼[𝜙𝑇], (89) and

𝔼[𝜙𝑇+1,𝑖] = 𝔼[𝜙𝑇,𝑖] + 𝛼𝔼[𝛿𝑡,𝑖] − 𝛽(B𝑘 − I)𝔼[𝜙𝑇]. (90) Finally,

#### ∑︁𝑛

1

𝔼[𝜙𝑇+1,𝑗] = 𝔼[𝜙𝑇,𝑖] + 𝛼𝔼[𝛿𝑇,𝑖] − 𝛽(B𝑘 − I)𝔼[𝜙𝑇] (91)

𝔼[𝜙𝑇+1,𝑖] −

𝑛

𝑗

#### ∑︁𝑛

1

𝔼[𝜙𝑇,𝑗] + 𝛼𝔼[𝛿𝑇,𝑗] − 𝛽(B𝑘 − I)𝔼[𝜙𝑇] (92)

−

𝑛

𝑗

#### ∑︁𝑛

1

𝛼𝔼[𝛿𝑇,𝑗] (93)

= 𝛼𝔼[𝛿𝑇,𝑖] −

𝑛

𝑗

#### ∑︁𝑛

1

𝔼[𝜙𝑇−1,𝑗] (94)

= 𝔼[𝜙𝑇−1,𝑖] −

𝑛

𝑗

#### ∑︁𝑛

1

𝔼[𝜙𝑇−1,𝑗] = 0. (95)

−𝔼[𝜙𝑇,𝑖] +

𝑛

𝑗

Thus the property holds for step 𝑇 + 1, and by proof by induction it holds for all 𝑇 since the first step holds.

## C. Hyper-parameter Ablations

DiLoCo and NoLoCo are sensitive to the used batch size. Table 3 present how increasing the batch size affects the results. Increasing the batch size will also increase the number of tokens models observe during training as well as linearly increase the training cost.

|Method|1M 2M<br><br>|
|---|---|
|FSDP DiLoCo NoLoCo|19.6 18.0<br><br>21.0 19.7<br><br>20.9 19.3<br>|

###### Table 3 Summary of final perplexity numbers from Reddit data with varying global batch size. All results are from the medium model size, 64 accelerators, four pipeline stages, and data parallel world size of sixteen.

