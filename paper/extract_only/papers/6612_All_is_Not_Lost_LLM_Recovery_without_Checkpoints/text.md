## arXiv:2506.15461v2[cs.DC]4Apr2026

# All is Not Lost: LLM Recovery without Checkpoints

###### Nikolay Blagoev1,2, Oğuzhan Ersoy1 and Lydia Yiyu Chen2

1Gensyn, 2Université de Neuchâtel

Training LLMs on decentralized nodes or on-spot instances lowers the training cost and enables model democratization. The inevitable challenge here is the transient churns of nodes due to failures and the operator’s scheduling policies, leading to losing parts of the model (some layers). The conventional approaches to recover from failures is to either use checkpointing, where periodically a copy of the entire model is sent to an additional storage, or redundant computation. These approaches yield significant communication and/or computation overhead even in non-failure cases and scale poorly in settings with large models. In this paper we propose CheckFree, an efficient recovery method where a failing stage (in a pipeline) is substituted by weighted averaging of the closest neighboring stages. In contrast to the state of the art, CheckFree requires no additional computation or storage. However, because of the nature of averaging neighbouring stages, it can only recover failures of intermediate stages. We further extend our method to CheckFree+ with out-of-order pipeline execution to tolerate crashes of the first and last stages. Thanks to out-of-order pipelining, behaviour of the first and last stages are mimicked by their neighboring ones, which allows CheckFree+ to recover them by copying the neighboring stages. To recover the (de-)embedding layers, CheckFree+ copies those layers in the neighboring stages, which requires relatively small storage overhead. We extensively evaluate our method on LLaMa models of model sizes from 124M to 1.5B with varying failure frequencies. In the case of low and medium failure rates (5-10%), CheckFree and CheckFree+ outperform both checkpointing and redundant computation in terms of convergence wall-clock time, achieving up to 12% improvement over redundant computation. Both of our proposals can be ran via our code available at: https://github.com/gensyn-ai/CheckFree.

### 1. Introduction

Training LLMs can take several months even on specialized high performance clusters [6, 24]. Recent works have made use of decentralized or cloud on-spot instances (preemptible GPUs) to train models [11, 20, 17], as they are cheaper to rent per hour. However, due to the high training time, (hardware) failures during the training are inevitable [26]. Moreover, the number of failures is expected to be higher for training on decentralized or on-spot instances since the failures can also occur due to devices becoming unavailable [20].

Large Language Model (LLM) training utilizes multiple Graphics Processing Units (GPUs) to parallelize and shard the training. In pipeline parallelism (PP), the model is split across several GPUs/nodes where each node (usually having a single GPU) executes a stage of consecutive layers and communicates activations with the nodes of the neighbouring stages in forward and backward passes. Typically, PP is combined with data parallelism (DP) where several pipelines are trained in parallel on different data. In DP, several nodes concurrently perform the gradient descent of the same stage and then synchronize their gradients at the end of an iteration with others in the same stage.

Stage Failure. With data parallel training, a stage can have some or all nodes failing. For the purposes of this work, we ignore the former scenario where at least one node remains active for a given stage, since failing nodes can be trivially recovered by using the alive nodes of the same

© 2026 Gensyn. All rights reserved.

External Storage

New node

New node

New node

Stage2

Stage3

Stage1

Stage2

Stage1

Stage3

Stage1

Stage3

Stage1

Stage2

Stage2

Stage3

Node 1

Node 2

Node 3

Node 1

Node 2

Node 3

Node 1

Node 2

Node 3

(a) Checkpointing with external storage.

(b) Redundant computation (and storage).

(c) Our recovery method(s).

Figure 1 Three strategies to recover the stage failure using a new node. A stage failure can happen when either each stage is only run by a single node or all the nodes of the same stage in DP fails simultaneously. a) In checkpointing strategy, an external non-faulty storage is used to periodically store checkpoints. Upon a failure, a new node receives stale weights from the storage. b) In redundant computation (RC), each node redundantly stores and performs the computations of the subsequent stage as well [20], and in the case of failure, new node can obtain the stage from the previous one. c) In our methods, CheckFree and CheckFree+, we use the weights of the neighbouring stages to initialize the failed stage. CheckFree+ also recovers the first and last stages by copying the corresponding neighbouring stage which is trained to simulate the same behaviour.

stage as the checkpoints. For example, when new nodes join, the most recent weights can be downloaded from a peer in the same stage, due to the data parallelism within the stage [11]. We thus focus on entire stage failures, i.e. when all nodes within a stage become unavailable. This could happen when either (i) there is no DP and each stage is only run by a single node or (ii) all the nodes of the same stage fails simultaneously. The latter case is plausible for a real deployment via spot instances since such nodes of the same stage would be rented from the same geographic location (to minimize the DP communication overhead) [24], and in the case of high demand on that location, all spot instances could become unavailable simultaneously [20].

To deal with stage failures, conventional approaches make use of checkpointing - periodically saving the model state to a non-volatile storage, which should remain reachable throughout the training [22]. When a failure occurs, the state of the model is reverted to the previously stored one, thus losing several GPU hours of training time. A careful tradeoff is chosen between checkpointing frequency and training throughput, as checkpointing too frequently to a remote storage can slow down the training time [22]. For example, LLaMa 70B with optimizer state requires roughly 700GBs when serialized and even on high-bandwidth CPUto-GPU bus and 500Mb/s connection, it would require upwards of 20 minutes per checkpoint, thus

impacting training at high checkpointing frequencies. Alternatively to checkpointing, Bamboo [20] uses redundant computation across the pipeline to recover failures where each node stores and does redundant computation for the following stage (in addition to their assigned stage).

Both recovery methods require significant overhead either in storage, communication or computation.

In this work, we propose an efficient recovery method, CheckFree, that can recover even when an entire stage (consisting of multiple layers) has failed without the need for checkpointing or redundant computations. To achieve this, we exploit layer stacking [1] and LLMs’ natural resilience to layer omission [8]. When a stage has crashed, we reinitialize it with a weighted average of its two neighbouring stages, on a new node. While this proves a lossy recovery mechanism, we extensively demonstrate that the benefit of cheaper recovery outweighs the negatives of slower convergence. Still, this strategy fails to recover the first and last stages without suffering from significant convergence drop. For this case, we introduce CheckFree+ which utilizes the idea of out of order PP execution [3], such that the second and second to last stages learn the behaviour of the first and last stage respectively. A visualization of the different approaches is presented in Fig. 1, and their comparison is given in Table 1.

- Table 1 Comparison of failure recovery strategies regarding the additional costs required even in the non-failure cases. F is the model, |𝑆| is the parameter size per stage, E is the embedding layer, and 𝑁 are the number of nodes.

|Recovery method<br><br>|Checkpoint [22]<br><br>|RC [20]<br><br>|CheckFree|CheckFree+|
|---|---|---|---|---|

|Cost<br><br>|Memory Communication|𝑂(|F |)|𝑂(|F |)<br><br>|0 0|𝑂(|E|)|
|---|---|---|---|---|---|
| | |𝑂(|F |)<br><br>|𝑂(|F |)<br><br>| |𝑂(|E|)|
| |Computational<br><br>|0|Forward pass|0<br><br>|0|
| |Non-faulty storage<br><br>|Yes<br><br>|No<br><br>|No|No|
|Recovery|non-consecutive stages consecutive stages<br><br>|Yes Yes<br><br>|Yes No<br><br>|Yes No|Yes No|
| | | | | | |
| |first and last stages|Yes|Yes<br><br>|No|Yes|

The rest of the paper is structured as follows. We first present relevant background on fault tolerance and checkpointing. We then present our recovery method, with theoretical justification of its stability. We further extend our work with a novel use of out of order pipelining for knowledge encoding. Lastly, we extensively evaluate our solution across various models and failure ratios and demonstrate superior performance of our method, which improves training time by over 12% relative to redundant computation in low failure frequencies.

### 2. Related Work

Failure Recovery As the models are growing and training process becoming ever more expensive, fault-tolerant distributed LLM training has been receiving greater recognition [17, 20, 11, 13]. SWARM [17] and DMoE [16] studied efficient recovery in PP training in the presence of failures. Recovery of weights and optimizer states has typically been performed through checkpointing [22], where periodically the weights and optimizer states are stored in some non-faulty remote storage. In case of a failure, new coming nodes download the corresponding weights from the remote storage. The training proceeds from the previous checkpoint.

In decentralized training via spot instances, churn rates are even higher and there may not be an available non-faulty storage to save the checkpoints.

Alternatively, Bamboo [20] makes use of redun-

dant computation, where each node computes the forward pass for itself and redundantly for the next node in the pipeline. In the case of a failure, the previous node can immediately continue the training with the redundant weights. However, such mechanism can prove too costly for large models, as each node performs redundant forward pass [11]. Instead, Oobleck [11] uses of the data parallel replication to recover from failures, where new nodes can get the weights from any of the nodes servicing the same weights as it. While efficient, such a recovery mechanism fails when an entire stage is gone.

Layer Stacking Layer (or model) stacking has emerged as an efficient method for training LLMs by starting from a small model and periodically initializing new weights [1, 5, 18]. Across commonly used strategies such as copying a layer to increase the depth and random initialization, it is shown that copying the last layer provides the biggest speed up and has the best over all performance, even outperforming training the full model [5]. In [1], authors further demonstrated that such a method of initialization approximates accelerated Nesterov gradient descent. Finally, in [18], authors showed the resemblance of models trained with stacking to looped LLMs [12] and further found that stacking new layers in the middle greatly improves the model’s reasoning for downstream tasks.

Layer Omission LayerSkip showed that during training layers can be skipped to improve early

exit inference and to speed up training [8]. SkipPipe further demonstrated that arbitrary layer skipping and out of order execution are a viable methods to speed up distributed training [3]. Both of these are reminiscent of earlier work on Stochastic Depth, which has been shown to work as gradient noise regularization [21]. Moreover, the results of [14, 27] showed that models can converge even with layers removed/skipped during training. This implies a certain redundancy between neighbouring layers of LLMs.

Random

- 3

- 4

- 5

- 6

- 7

- 8

- 9

Naive copy

Average

Gradient Average

ValidationLoss

0 5000 10000 15000 20000 Iteration

Figure 2 Varying reinitialization strategies for failed stages of a LLaMa 500M model.

### 3. Problem Statement

We assume a LLM model F, composed of an embedding layer E, several layers with residual connections 𝑓1, . . . , 𝑓𝐿, and a deembedding layer E−1, F = E−1 ◦ (𝐼 + 𝑓𝐿) ◦. . .◦ (𝐼 + 𝑓2) ◦ (𝐼 + 𝑓1) ◦E, where ◦ is a composing operation and 𝐼 is the identity matrix. The weights of each layer 𝑙 ∈ [1, 𝐿] are given by 𝑊𝑙. The goal of the model (F : X → Y) is to minimize some loss function (𝔼(𝑥,𝑦)∼𝐷L) over some dataset (𝐷 with distribution X × Y).

Distributed Training The model is split over stages 𝑆1, . . . , 𝑆𝑠, each holding a non overlapping and consecutive partition of the layers. The weights of a stage 𝑖 ∈ [1, 𝑆] is given by 𝑊𝑖𝑆, a shorthand for the combination of weights of the layers in that stage. Nodes serve a single stage and communicate with their peers to train the model. Communication occurs over the network and is associated with delays dependent on the bandwidth and latency between pairs of nodes. No central non-faulty storage exists to which model weights can be offloaded.

Failure pattern Nodes can fail at arbitrary points. A failing node is disconnected from the training and its local progress is lost. While it may come back at some point in the future, its local state will be too outdated to be useful.

Upon failure of a node, we assume new one can be made available within a negligible amount of time (e.g. starting a new on-spot node to join the training). We distinguish between failures where at least one node remains per stage or none. Dealing with the former type of failures is trivial as

new nodes can download the weights of the stage from a node alive serving that stage [17, 11] due to data parallelism. We henceforth focus only on recovering from failures where a stage is entirely disconnected.

### 4. Propose Recovery Method

##### 4.1. Motivation

Our recovery method draws inspiration from several key (known and experimented) observations regarding training of LLMs in the presence of layer omissions and stacking.

LLMs are resilient to layer omission. In both training and inference time, LLMs are resilient to layer omission [3, 8, 25]. This is partially attributed to the residual connections [21] and suggests a degree of redundancy encoded in the layers (neighbouring layers can perform each other’s functionality).

Layer stacking can improve LLM’s performance. Initializing new layers through a neighbouring one improves training efficiency as well as the model’s reasoning [18, 5, 1].

Based on these observations, we consider recovering a failed stage with a combination of neighboring ones. In Fig. 2, we present our early results on comparing four strategies: random (randomly initializing the stage), copy (copying the previous stage), and average (initializing by averaging of neighbouring stages), gradient average (initializing by weighted averaging of neighbouring stages wrt their gradients). Here, we train a LLaMa 500M model on the OpenWebText dataset [9] with a failure probability of 16% for any stage (but those

holding embedding and de-embedding). It can be seen that though copying a neighbouring stage is significantly better than random reinitialization (of the failed stage), averaging methods significantly outperform them. In the following sections, we explain how we choose the weights of the averaging (for CheckFree) as well as how we can recover the first and last stages as they have only a single neighbour (via CheckFree+).

#### 4.2. CheckFree: Memoryless Recovery

In the case of no checkpointing or redundant computation, if a stage failure happens, the weights of the failed stages are lost. Our method does not directly recover the exact weights of the failed stage, but recovers (maintains) the model’s performance by replacing the failed stages with a combination of the remaining ones. Specifically, by using the observations given in the previous section, we replace the failed stage with the weighted average of the weights of its neighbouring stages, i.e.

𝜔𝑖−1 ∗ 𝑊𝑖𝑆−1 + 𝜔𝑖+1 ∗ 𝑊𝑖𝑆+1 𝜔𝑖−1 + 𝜔𝑖+1

𝑊𝑖𝑆 =

.

Now, we discuss how to select the weights 𝜔𝑖−1 and 𝜔𝑖+1. A known approach is to copy the previous stage when initializing new weights (𝜔𝑖+1 = 0). However, as seen in Fig. 2, this proves inferior to aggregated averaging. A naive way of averaging would be a uniform average of the two stages, i.e. (𝜔𝑖−1 = 𝜔𝑖+1). Such averaging does not distinguish the importance and convergence of the stages, and thereby leads to slower convergence of the overall model (as seen in Fig. 2). For that reason, we use the weights of the last gradient norm of the given stage, i.e. 𝜔𝑖−1 = ||∇𝑊𝑖𝑆−1||2 and 𝜔𝑖+1 = ||∇𝑊𝑖𝑆+1||2. Conceptually, this gives more weight to stages which have not converged as much yet, thus partially offloading their functionality to the new stage. To achieve weighted gradient averaging, for each stage 𝑆𝑖, the nodes need to store the gradient norm ||∇𝑊𝑖𝑆||2 and send it to the new-coming nodes. The communication and storage overhead of this is negligible, as ||∇𝑊𝑖𝑆||2 is a single scalar, which is awfully smaller than the size of the weights of the stage. Once the weights are reinitialized, to further assist the new-formed stages in diverging from their (possibly) inferior state, we scale up the starting learning rate by a

small amount (in our experiments by 1.1), which can change over iterations subject to the learning rate scheduler. Our solution is presented in Algorithm 1.

Algorithm 1 Recovery algorithm for stage 𝑖

Require: new node assigned to stage 𝑖, 𝜆 learning rate

- 1: Receive 𝑊𝑖𝑆−1 and 𝜔𝑖−1 := ||∇𝑊𝑖𝑆−1||2 of stage 𝑖 − 1
- 2: Receive 𝑊𝑖𝑆+1 and 𝜔𝑖+1 := ||∇𝑊𝑖𝑆+1||2 of stage 𝑖 + 1
- 3: Initialize the weights of the failed stage and update

its learning rate 𝑊𝑖𝑆 ← 𝜔𝑖−1𝑊

𝑆 𝑖−1+𝜔𝑖−1𝑊𝑖𝑆+1

𝜔𝑖−1+𝜔𝑖+1 , 𝜆𝑖 ← 1.1𝜆𝑖

- 4: Continue training from the current batch

Such a memoryless averaging approach, however, cannot recover the first and last stages, as there is no second neighbour to average with. Additionally, as identified in several other works [18, 2], the first and last stages perform different functionality than the other stages (the first stage especially) and are critical to the performance of the model. Therefore, a naive copy of one neighbour results in a significant drop in performance. For these reasons, CheckFree is suitable for the recovery of intermediate stages but not for the first or last ones. In the following section we present an improvement that can recover those stages as well.

4.3. CheckFree+: First and Last Stage Recov-

ery

CheckFree+ extends CheckFree by also recovering the first and last stage failures. Here, we take inspiration from out-of-order pipeline parallelism presented in [3] to mimic redundant computation [20] without actually duplicating computation. Specifically, for half the microbatches, we run the stages in standard order (E, 𝑆1, 𝑆2...𝑆𝑠−1, 𝑆𝑠, E−1) and for the other half, we swap the order of the first two and last two stages excluding the (de)embedding layers (i.e. E, 𝑆2, 𝑆1...𝑆𝑠, 𝑆𝑠−1, E−1). In this way, the neighboring stages of the first and last stages, 𝑆2 and 𝑆𝑠−1, partially learn the behaviour of 𝑆1 and 𝑆𝑠 respectively, without any additional computation, as half the time they take their places in the pipeline. For the same reason, this would result in the two layers having similar weights (within some noise range of each other), meaning they can easily be recovered from one

another. Out-of-order training, however, comes with a small degradation to convergence, as noted in [3]. As such, if the failure chance is almost negligible, this solution would prove slower than no checkpointing, due to the longer time needed to converge.

Finally, we explain how CheckFree+ recovers the embedding and deembedding (LM head) layers. While some existing works explore sharing the weights between the two [10], thus suggesting that in case of failure of either E or E−1 the weights can be reinitialized by copying the other one’s, in this work we simply send their weights to the previous and following stages. Thus in case of a failure the weights can be recovered exactly without loss of data. While this may resemble some form of decentralized checkpointing, we note that the sizes of these layers are significantly smaller compared to the other stages, thus presenting minimal communication and storage overhead.

#### 4.4. Convergence Analysis of CheckFree

We base our proof on [14]. The model is comprised of an embedding layer E, deembedding layer E−1, and a series of residual functions F = E−1 ◦ (𝐼 + 𝑓𝐿) ◦ ... ◦ (𝐼 + 𝑓1) ◦ E. Post failure of layer 𝑘1, the model is modified as F′ = (𝐼 + 𝑓𝑙) ◦ ... ◦ (𝐼 + 𝑓𝑘+1) ◦ (𝐼 + 𝜔1 𝑓𝑘+1 + 𝜔2 𝑓𝑘−1) ◦ (𝐼 + 𝑓𝑘−1)... ◦ (𝐼 + 𝑓1) ◦ E. We let F𝑡 denote the loss function at 𝑡𝑡ℎ iteration.

- Assumption 1. The loss function L(F) is L-smooth

and convex: ||L(F𝑡) − L(F𝑡+1)|| ≤ 𝐿 · ||F𝑡 − F𝑡+1||, ∀𝑡.

- Assumption 2. For some 𝛿 ∈ [0, 1) at any timestep 𝑡 the model reduction error is bounded by ||F𝑡 − 𝑚 ⊙ F𝑡||2 ≤ 𝛿2||F𝑡||2 where 𝑚 is a 0-1 vector selecting given layers of the model.

This is a common assumption in literature [27, 14]. Given these assumptions, the convergence of a model past a failure with our recovery method is given by:

∑︁

1 𝑡 )+∑︁

𝔼||F𝑡′−1−F0||2 ≤ O(

2𝔼||𝜔1 𝑓𝑘+1+𝜔2 𝑓𝑘−1−𝑓𝑘||2.

𝑡

𝑡

(1)

1Equivalently can be grouping of layers from 𝑘 to 𝑘 + 𝑗, i.e. a stage

This implies that every failure slows down convergence with the error incurred by the initialization. The details of the proof are provided in Appendix C.

### 5. Evaluation

In this section, we evaluate CheckFree and CheckFree+ on models ranging from 124M to 1.5B (details of each model can be found in Appendix A.1). We demonstrate that our solutions outperform the state of the art in terms of failure recovery and does not suffer from significant decrease in convergence, despite incurring lower storage and computation cost compared to existing failure recovery strategies.

Setup. We perform our tests on a single H100 node with 8 GPUs. Communication delays between nodes are simulated based on realistic bandwidth and latency measurements between 5 geodistributed locations from Google Cloud. We use the rates of 5%, 10%, or 16% as the probability of a stage failure within an hour. These values are inspired from Bamboo [20] where they use double of these failure rates per node, while for us they represent stage failures.2 We expect in practice these probabilities to be significantly lower. However, this choice of higher probabilities is meant to stress test our solution in highly challenging churn rates. As we see in Fig. 4, our solution works better at lower failure frequencies, which are more realistic.

Baselines. We compare against two baselines - checkpointing and redundant computation. For checkpointing, we create checkpoints for the small models every 50 iterations, for the medium models every 100 iterations, and for the large models - every 40 iterations (roughly corresponding to every 3 hours, as per [19]). The effect of checkpointing frequency on convergence can be found in our ablation study in Appendix B. For redundant computation, each stage computes redundantly the next stage in the forward pass as per [20]. To account

2These probabilities could be significantly lower. Under the assumption that there are 𝑘 devices each with an independent probability of disconnecting 𝑝, the chance of an entire stage failing is 𝑝𝑘. However, in practice one would use a datacenter responsible per stage. Thus, failure of a stage is plausible if the datacenter becomes fully occupied and spot instances simultaneously become unavailable.

Small Model 10% Convergence

10

Checkpointing

CheckFree

CheckFree+

8

Redundant computation

ValidationLoss

6

4

2

0 500 1000 1500 2000 2500 3000 3500 4000 Iteration

(a) Small model at 10% failure rate.

Medium Model 10% Convergence

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

Checkpointing

Checkpointing

- 3

- 4

- 5

- 6

- 7

- 8

- 9

CheckFree

CheckFree

CheckFree+

CheckFree+

Redundant Computation

Redundant computation

ValidationLoss

ValidationLoss

0 5000 10000 15000 20000 25000 Iteration

0 6000 12000 18000 24000 30000 Iteration

(b) Medium model at 10% failure rate.

(c) Large model at 16% failure rate.

Figure 3 Convergence of the models with 10% failure rate for small and medium models & 16% for large ones.

- Table 2 Performance of three different recovery strategies for three different failure rates until convergence for medium models (specifically, when the validation loss reaches 2.85).

| |Checkpointing<br><br>| | |Redundant Comp.| | |CheckFree<br><br>| | |CheckFree+| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Failure rate<br><br>|5%|10%<br><br>|16%|5%<br><br>|10%|16%|5%<br><br>|10%<br><br>|16%|5%<br><br>|10%<br><br>|16%|

|Iteration time (s)<br><br>|91.35|91.35|92.14<br><br>|151.04<br><br>|151.04|151.04<br><br>|91.32|91.32|92.12<br><br>|91.32|91.32<br><br>|92.12|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Train time (h)|558.23|621.67|634.35<br><br>|419.56|419.56|419.56<br><br>|367.81<br><br>|405.87|562.96|355.13|367.81|460.6|

- Table 3 Perplexity of 1.5B models after 30K iterations.

est failure rate (16%) as a means of stress testing our solution for it. Despite that, CheckFree+ converges close to the no-failure case. This can be explained by the fact that larger models typically have a higher degree of redundancy, thus allowing for such recovery mechanisms [15], hinting at the potential of our solution scaling better with model sizes.

|Perplexity ↓|No failures<br><br>|CheckFree<br><br>|CheckFree+|
|---|---|---|---|

|OpenWebText<br><br>|16.07|17.59<br><br>|17.07|
|---|---|---|---|
|Gutenberg<br><br>|13.32|14.58|13.46|
|Stack Exchange|19.85<br><br>|21.36<br><br>|20.86|
|Arxiv|12.25<br><br>|13.33|12.46|

for the higher memory requirement, we use half the microbatch size, but double the microbatch count, thus keeping the same batch size. Details of all experiments can be found in Appendix A.

We evaluate the perplexity of the large pretrained models across four different datasets OpenWebText [9], GutenBerg books, and Stack Exchange and Arxiv as presented in [4]. We present the results in Table 3. We see similar performance between a model trained with no faults and with our recovery strategies, despite the drastically different resultant weights. The models produced from our method are trained much faster wallclock time wise, as redundant computation has a high per-iteration time.

##### 5.1. Convergence and Downstream Evaluation

We evaluate the models trained via our recovery method against redundant computation (which is convergence-wise equivalent to the case without failures) and checkpointing for the same iteration count. We report the validation loss over the iterations in Fig. 3. The medium and small models are trained until convergence. For the large models, due to resource constraints, we trained only for 30 thousand iterations and we chose the high-

##### 5.2. Throughput with Failures

Here, we evaluate the training throughput of different recovery strategies - checkpointing, redundant

computation, CheckFree and CheckFree+.

We perform the throughput tests for 500 iterations, simulating the failures of different stages across iterations, so that the failure patterns between tests are the same. In Table 2, we report the iteration and train time. The reported train time is the wall-clock time needed to reach the "converged" iterations reported in Fig. 3b. Specifically, for the convergence point, we use the validation loss reaching 2.85 as it saturates around that value.

We observe similar iteration time as checkpointing as the frequency of checkpointing is high enough that it does not impact iteration time. Combining the number of iterations needed to reach the convergence, checkpointing experiences a significantly higher train time due to the need to restart training. For the same convergence point, CheckFree+ requires a higher number of iterations than redundant computing and a lower number than checkpointing. Due to a lower iteration time for CheckFree+ and CheckFree, the resulting training time of redundant computing is higher. Thus, our solution demonstrates a clear improvement in performance, being 12% faster at 5% failure rate than redundant computation and significantly faster than checkpointing. Additionally, our method has better scalability than redundant computation in low-bandwidth networks and larger models, due to the minimal overhead. Also, in case of stage failure, the recovery time of that stage is around 30 seconds. Ablation experiments can be found in Appendix B.

### 6. Conclusion

Failure recovery is critical for enabling training of LLMs on distributed and unreliable nodes. Prior works recover stage failures within a pipeline through redundant computation or checkpointing. We propose CheckFree which recovers stage failures by reinitializing the failed stages with the weighted average of the neighboring stages. We further extend the protocol with CheckFree+ to tolerate first and last stage failures, via out of order pipelining. CheckFree exploits the inherent resilience of LLMs to layer omissions and efficient layer initialization techniques to recover from stage failures without requiring non-faulty storage or redundant computation. We extensively demon-

strate the convergence of our method via empirical tests. We evaluate the effectiveness of CheckFree on LLaMa models of different sizes, showing that such lightweight recovery reduces the training time compared to the state of the art in 5% stage churn rate by over 12%. CheckFree+ also exhibits robust and steady convergence results for varying failure frequencies.

Limitations and Future Work. Our proposed solutions CheckFree and CheckFree+ cannot recover from cases of consecutive stages failing together, as there is no neighboring stages for the reinitialization strategy. We believe that extending our methods with a lightweight checkpointing scheme will address this limitation and will explore this extension in our future work. Moreover, we will work on improving the convergence of our methods (especially CheckFree+) in non-faulty case to reduce the number of iterations required for training.

### References

- [1] Naman Agarwal, Pranjal Awasthi, Satyen Kale, and Eric Zhao. Stacking as accelerated gradient descent. arXiv preprint arXiv:2403.04978, 2024.
- [2] Srinadh Bhojanapalli, Ayan Chakrabarti, Daniel Glasner, Daliang Li, Thomas Unterthiner, and Andreas Veit. Understanding robustness of transformers for image classification. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 10211–10221. IEEE, 2021. doi: 10.1109/ICCV48922.2021.

01007. URL https://doi.org/10.1109/ ICCV48922.2021.01007.

- [3] Nikolay Blagoev, Lydia Yiyu Chen, and Oguzhan Ersoy. Skippipe: Partial and reordered pipelining framework for training llms in heterogeneous networks. CoRR, abs/2502.19913, 2025. doi: 10.48550/ARXIV.2502.19913. URL https:// doi.org/10.48550/arXiv.2502.19913.
- [4] Together Computer. Redpajama: An open source recipe to reproduce llama training dataset, 2023. URL https://github.com/ togethercomputer/RedPajama-Data.
- [5] Wenyu Du, Tongxu Luo, Zihan Qiu, Zeyu Huang, Yikang Shen, Reynold Cheng, Yike Guo, and Jie Fu. Stacking your transformers: A closer look at model growth for efficient LLM pre-training.

- In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.
- [6] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. The llama 3 herd of models. CoRR, abs/2407.21783, 2024. doi: 10.48550/ARXIV.2407.21783. URL https: //doi.org/10.48550/arXiv.2407.21783.
- [7] Ronen Eldan and Yuanzhi Li. Tinystories: How small can language models be and still speak coherent english? CoRR, abs/2305.07759, 2023. doi: 10.48550/ARXIV.2305.07759. URL https: //doi.org/10.48550/arXiv.2305.07759.
- [8] Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed A Aly, Beidi Chen, and Carole-Jean Wu. Layerskip: Enabling early exit inference and self-speculative decoding. In

Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 12622– 12642. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.681. URL https://doi.org/10.18653/v1/2024.

acl-long.681.

- [9] Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext corpus. http://Skylion007.github.io/ OpenWebTextCorpus, 2019.
- [10] Dylan Hillier, Leon Guertler, Cheston Tan, Palaash Agrawal, Ruirui Chen, and Bobby Cheng. Super tiny language models. CoRR, abs/2405.14159, 2024. doi: 10.48550/ARXIV. 2405.14159. URL https://doi.org/10. 48550/arXiv.2405.14159.
- [11] Insu Jang, Zhenning Yang, Zhen Zhang, Xin Jin, and Mosharaf Chowdhury. Oobleck: Resilient distributed training of large models using pipeline templates. In Jason Flinn, Margo I. Seltzer, Peter Druschel, Antoine Kaufmann, and Jonathan Mace, editors, Proceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023, pages 382–395. ACM, 2023. doi: 10.1145/3600006.

3613152. URL https://doi.org/10.1145/ 3600006.3613152.

- [12] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30,

2020. OpenReview.net, 2020. URL https:// openreview.net/forum?id=H1eA7AEtvS.

- [13] Yuhang Liang, Xinyi Li, Jie Ren, Ang Li, Bo Fang, and Jieyang Chen. Attnchecker: Highlyoptimized fault tolerant attention for large language model training. In Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming, PPoPP 2025, Las Vegas, NV, USA, March 1-5, 2025, pages 252–266. ACM, 2025. doi: 10.1145/3710848.

3710870. URL https://doi.org/10.1145/ 3710848.3710870.

- [14] Xiaolong Ma, Minghai Qin, Fei Sun, Zejiang Hou, Kun Yuan, Yi Xu, Yanzhi Wang, Yen-Kuang Chen, Rong Jin, and Yuan Xie. Effective model sparsification by scheduled grow-and-prune methods. In The Tenth International Conference on

- Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id= xa6otUDdP2W.
- [15] Sameera Ramasinghe, Thalaiyasingam Ajanthan, Gil Avraham, Yan Zuo, and Alexander Long. Protocol models: Scaling decentralized training with communication-efficient model parallelism. CoRR, abs/2506.01260, 2025. doi: 10.48550/ ARXIV.2506.01260. URL https://doi.org/ 10.48550/arXiv.2506.01260.
- [16] Max Ryabinin and Anton Gusev. Towards crowdsourced training of large neural networks using decentralized mixture-of-experts. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.
- [17] Max Ryabinin, Tim Dettmers, Michael Diskin, and Alexander Borzunov. SWARM parallelism: Training large models can be surprisingly communication-efficient. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 29416–29440. PMLR, 2023. URL https://proceedings. mlr.press/v202/ryabinin23a.html.
- [18] Nikunj Saunshi, Stefani Karp, Shankar Krishnan, Sobhan Miryoosefi, Sashank Jakkam Reddi, and Sanjiv Kumar. On the inductive bias of stacking towards improving reasoning. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.
- [19] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo

- Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, and et al. BLOOM: A 176b-parameter open-access multilingual language model. CoRR, abs/2211.05100, 2022. doi: 10.48550/ARXIV.2211.05100. URL https: //doi.org/10.48550/arXiv.2211.05100.
- [20] John Thorpe, Pengzhan Zhao, Jonathan Eyolfson, Yifan Qiao, Zhihao Jia, Minjia Zhang, Ravi Netravali, and Guoqing Harry Xu. Bamboo: Making preemptible instances resilient for affordable training of large dnns. In Mahesh Balakrishnan and Manya Ghobadi, editors, 20th USENIX Symposium on Networked Systems Design and Implementation, NSDI 2023, Boston, MA, April 17-19, 2023, pages 497–513. USENIX Association, 2023. URL https://www.usenix.org/conference/ nsdi23/presentation/thorpe.
- [21] Andreas Veit, Michael J. Wilber, and Serge J. Belongie. Residual networks behave like ensembles of relatively shallow networks. In Daniel D. Lee, Masashi Sugiyama, Ulrike von Luxburg, Isabelle Guyon, and Roman Garnett, editors, Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, pages 550–558, 2016.
- [22] Zhuang Wang, Zhen Jia, Shuai Zheng, Zhen Zhang, Xinwei Fu, T. S. Eugene Ng, and Yida Wang. GEMINI: fast failure recovery in distributed training with in-memory checkpoints. In Jason Flinn, Margo I. Seltzer, Peter Druschel, Antoine Kaufmann, and Jonathan Mace, editors, Proceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023, pages 364–381. ACM, 2023. doi: 10.1145/3600006.3613145. URL https: //doi.org/10.1145/3600006.3613145.
- [23] Maurice Weber, Daniel Y. Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, Ben Athiwaratkun, Rahul Chalamala, Kezhen Chen, Max Ryabinin, Tri Dao, Percy Liang, Christopher Ré, Irina Rish, and Ce Zhang. Redpajama: an open dataset for training large language models. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing

- Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.
- [24] Binhang Yuan, Yongjun He, Jared Davis, Tianyi Zhang, Tri Dao, Beidi Chen, Percy Liang, Christopher Ré, and Ce Zhang. Decentralized training of foundation models in heterogeneous environments. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.
- [25] Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. Draft& verify: Lossless large language model acceleration via self-speculative decoding. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 11263–

11282. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.607. URL https://doi.org/10.18653/v1/2024.

acl-long.607.

- [26] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. OPT: open pre-trained transformer language models. CoRR, abs/2205.01068, 2022.
- [27] Hanhan Zhou, Tian Lan, Guru Venkataramani, and Wenbo Ding. Every parameter matters: Ensuring the convergence of federated learning with dynamic heterogeneous models reduction. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

### A. Reproducibility information

This section describes relevant information for reproducing our results.

##### A.1. Models

We train three different model sizes, all of the LLaMa family. We provide details of their hyperparameters in Table 4.

Table 4 Model hyperparameters.

|Size|Small|Medium<br><br>|Large [8]|
|---|---|---|---|
|Parameters Dim Heads Layers Stages Context Learning rate<br><br>|124M 512 8 12 4 512 6 × 10−4<br><br>|500M 1024 16 24 6 1024 3 × 10−4|1.5B 2048 16 24 6 4096 3 × 10−4<br><br>|

##### A.2. Optimizers

All tests were performed with the Adam optimizer with no weight decay and betas (0.9,0.999)

##### A.3. Datasets

- • TinyStories [7] - Available with license CDLAsharing-1.0. 100 texts were reserved for validation. Used for the small models.
- • OpenWebText [9] - Available with license Creative Commons Zero. 100 texts (≈1M tokens) were reserved for validation. Used for the medium models.
- • RedPajamas v2 [23] - Available with license Apache-2.0 license. 100 texts (≈10M tokens) were reserved for validation. Used for the large models.

##### A.4. Hardware

For convergence tests we run on 2, 4, and 8 H100s for the small, medium, and large models respectively. We simulate the faults and recoveries across nodes without actual communication (other than data-parallel aggregation) to speed up iteration time. This has no difference from a test performed on geo-distributed nodes, except that runtime will be longer wall-clock wise due to the communication overhead.

Small models converged in roughly 8 hours, medium - in 2 days, and large models - in 2 weeks.

For throughput results we test on 8 H100s spawning 20 separate nodes. Communication be-

tween all 20 nodes is simulated based on realistic latency and bandwidth taken from profiling Google Cloud nodes in 5 different locations. Thus our results on throughput accurately reflect a real deployment situation on geo-distributed nodes.

### B. Ablation Studies

Here we investigate the effects of several components individually on a model’s convergence with various hyperparameters and failure rates.

Varying model sizes. We empirically demonstrate the convergence of our methods on three models and dataset pairs: small 120M LLaMa with TinyStories [7], medium 500M LLaMa with the OpenWebText dataset [9], and large 1.5B LLaMa with the RedPyjamas dataset [23]. For the small and medium experiments we fix the crash rate at 10%. We evaluate 4 recovery strategies: checkpointing, redundant computation, CheckFree and CheckFree+. We present the results of the small, medium and large models in Fig. 3. These figures demonstrate that for different model sizes, our solution is superior to checkpointing convergence-wise. Note that the figures plot over iteration count, not wall-clock time. As such, despite redundant computation converging faster in terms of iterations, they incur a high overhead, which decreases their throughputs.

While our solution converges slower iterationwise than redundant computation, it significantly outperforms it wall-clock-wise.

Varying failure frequency. One important question to consider is how our method scales with different failure frequencies. Here, we investigate this by repeating the tests on the medium (500M) model in 3 settings - 5%, 10%, and 16% stage failure chance. We summarize the results in Fig. 4a. As expected, CheckFree+ performs the best in low-failure settings. Yet, it can be seen that the performance (validation loss) slightly degrades even when the failure rate is tripled, which demonstrates the robustness of our recovery method.

Checkpointing frequency. Our initial comparison against checkpointing assumes a checkpoint roughly every 100 iterations (corresponding to around every 3 hours [19]). A higher checkpointing rate is expected to yield better convergence

results (due to the significantly smaller loss of training) but incurs higher communication overhead to send the model weights to the external non-faulty storage. Here, we investigate the empirical tradeoff of checkpointing strategies, against the proposed CheckFree+. We repeat the experiments of the medium model, comparing against 3 checkpointing frequencies - every 10, every 50, and every 100 iterations, at 10% failure chance. The results are plotted in Fig. 4b. We observe that our solution outperforms even high frequency checkpointing (every 10 iterations) case, due to the need to rollback the model after every failure with checkpointing.

Effects of swapping on convergence. While CheckFree+ performs well in cases with failures, it incurs a non-negligible affect to its convergence in 0% failure case, due to the swapping. Here we empirically quantify this effect on a medium model. We compare the convergence of a model trained with and without swapping. The results are plotted in Fig. 4c. We observe a slowdown in convergence when swapping is used.

### C. Extended Proof

From the assumptions given in the main body, it follows:

###### ||F𝑡 − F𝑡′||2 = ||F𝑡 − 𝑚 ⊙ F𝑡 + 𝑚 ⊙ F𝑡 − F𝑡′||2

Where 𝑚 this time selects all layers that are the same (non-failed ones).

###### ||F𝑡 − 𝑚 ⊙ F𝑡 + 𝑚 ⊙ F𝑡 − F𝑡′||2 ≤ 𝛿21||F𝑡||2 − 𝛿22||F𝑡′||2 ≤ 𝛿(||F𝑡||2 − ||F𝑡′||2) ≤ 𝛿(||𝜔1 𝑓𝑘+1 + 𝜔2 𝑓𝑘−1||2 − || 𝑓𝑘||2)

We can divide the training past a random failure of model F0 into two parts - standard optimization with the modified model F′ and post-recovery reduction error:

∑︁

###### 𝔼||F𝑡′−1 − F0||2 = ∑︁

###### 𝔼||F𝑡′−1 − F0′ + F0′ − F0||2 ≤ ∑︁

###### 𝔼||F𝑡′−1 − F0′||2 + ∑︁

𝑡

𝑡

###### 𝔼||F0′ − F0||2.

𝑡

𝑡

The left hand side is standard optimization problem. This depends on the optimizer, but here we

Varying Failure Frequencies

CheckFree+ 16% CheckFree+ 10% CheckFree+ 5%

- 3

- 4

- 5

- 6

- 7

- 8

- 9

ValidationLoss

0 3500 7000 10500 14000 17500 21000 Iteration

(a) Varying failure frequencies

Varying Checkpoint Frequencies

Effects of swapping

Checkpointing 100

CheckFree+

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 3

- 4

- 5

- 6

- 7

- 8

- 9

Checkpointing 50 Checkpointing 10 CheckFree+

No swaps

ValidationLoss

ValidationLoss

0 4000 8000 12000 16000 20000 Iteration

0 2500 5000 7500 10000 12500 15000 Iteration

(b) Varying checkpoint frequencies.

(c) Impact of swapping

Figure 4 Ablation studies of CheckFree+ on medium models.

assume it converges inversely proportional to 𝑡 O(1𝑡 ).

The right hand is the error due to replacement of stages. What this tells us is that if the error is small, the convergence is not as affected (which intuitively makes sense). The right side can be rewritten as: ||𝜔1 𝑓𝑘+1 + 𝜔2 𝑓𝑘−1 − 𝑓𝑘||2, thus:

###### ∑︁

1 𝑡 )+∑︁

𝔼||F𝑡′−1−F0||2 ≤ O(

2𝔼||𝜔1 𝑓𝑘+1+𝜔2 𝑓𝑘−1−𝑓𝑘||2

𝑡

𝑡

QED.

