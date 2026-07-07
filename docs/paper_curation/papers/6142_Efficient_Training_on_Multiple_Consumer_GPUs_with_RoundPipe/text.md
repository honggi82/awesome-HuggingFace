## arXiv:2604.27085v1[cs.DC]29Apr2026

# Efficient Training on Multiple Consumer GPUs with RoundPipe

Yibin Luo

Shiwei Gao

Huichuan Zheng

luo-yb25@mails.tsinghua.edu.cn Tsinghua University Beijing, China

zhenghuichuan@mail.tsinghua.edu.cn Tsinghua University Beijing, China

gsw23@mails.tsinghua.edu.cn Tsinghua University Beijing, China

Youyou Lu

luyouyou@tsinghua.edu.cn Tsinghua University Beijing, China

### Abstract

Fine-tuning Large Language Models (LLMs) on consumergrade GPUs is highly cost-effective, yet constrained by limited GPU memory and slow PCIe interconnects. Pipeline parallelism combined with CPU offloading mitigates these hardware bottlenecks by reducing communication overhead. However, existing PP schedules suffer from an inherent limitation termed the weight binding issue. Binding uneven model stages (e.g., the LM head is large) to GPUs limits the pipeline’s throughput to that of the GPU with the heaviest load, leading to severe pipeline bubbles.

Inthispaper,weproposeRoundPipe,anovelpipelineschedule that breaks the weight binding constraint on consumer GPU servers. RoundPipe treats GPUs as a pool of stateless execution workers and dynamically dispatches computation stages across devices in a round-robin manner, achieving a near-zero-bubble pipeline. To ensure training correctness and system efficiency, RoundPipe integrates a priority-aware transfer scheduling engine, a fine-grained distributed event-based synchronization protocol, and an automated layer partitioning algorithm. Evaluations on an 8× RTX 4090 server demonstrate that RoundPipe achieves 1.48–2.16× speedups over state-of-the-art baselines when fine-tuning 1.7B to 32B models. Remarkably, RoundPipe enables LoRA fine-tuning of the Qwen3-235B model with 31K sequence length on a single server.

RoundPipe is publicly available as an open-source Python library with comprehensive documentation.

Github Repo: https://github.com/ITcarrot/RoundPipe Documentation: https://itcarrot.github.io/RoundPipe/

[Figure 1]

[Figure 2]

### 1 Introduction

Fine-tuning Large Language Models (LLMs) has become the cornerstone of modern AI applications, enabling open-source foundation models to master complex reasoning and domainspecific tasks [34, 42]. Unlike pre-training from scratch, finetuning requires significantly less computation [9]. For such

Jiwu Shu

shujw@tsinghua.edu.cn Tsinghua University Beijing, China

workloads, consumer-grade GPUs offer a compelling alternative. The NVIDIA RTX 4090 delivers computation capabilities comparable to the datacenter A100 GPUs, while costing roughly 80% less.1 Efficiently fine-tuning massive LLMs on consumer GPU servers is vital for democratizing AI for small companies and researchers.

However, consumer-grade GPUs impose two hardware constraints that restrict model scalability and training efficiency. (1) Limited memory capacity: The VRAM of a typical consumer-grade GPU falls short of training demands. For instance, training an 8B model needs 128 GB for model states alone [37], far larger than 24GB for NVIDIA RTX 4090 or 32GB for NVIDIA RTX 5090. Furthermore, expanding context windows for complex tasks [8, 47] (e.g., reasoning, video generation) also inflates activation memory. (2) Slow inter-GPU communication: Consumer-grade GPUs use PCIe interconnects, offering less than 20% of NVLink bandwidth. This physical limitation is further compounded by root complex contention in PCIe topologies [12, 21].

To alleviate VRAM constraints during consumer-grade GPU training, existing methods commonly employ CPU offloading [38, 40] to shift parameters, optimizer states, and activations to host memory, often combined with activation recomputation [4, 22] to further reduce the memory footprint. When scaling out across multiple consumer-grade GPUs connected via PCIe, data parallelism techniques like ZeROInfinity [38] distribute model states across devices. However, they require frequent collective communications at each layer to reconstruct parameters. Prior studies [12] indicate that these communications can consume up to 70% of the training time. To mitigate such PCIe bottlenecks, state-of-the-art systems like Mobius [12] integrate pipeline parallelism (PP) with offloading, replacing extensive multi-GPU collectives with more efficient P2P communication.

However, existing pipeline schedules, designed mainly for datacenter training, suffer from significant pipeline bubbles. In these schedules, the weights of a stage (and the corresponding forward and backward computation) are fixed to a specific

1Prices from Amazon in March 2026.

Forward Recompute Backward Structural Bubbles Imbalance Bubbles

- GPU1
- GPU2
- GPU3
- GPU4

- F1
- F2
- F3
- F4

- F9
- F10
- F11
- F12,13 B12,13

B5

- F5
- F6
- F7
- F8

B9

B1

B6

B10

B2

B11

B3

B7

B8

B4

Time

(a) Looped BFS schedule (standard partition)

- GPU1
- GPU2
- GPU3
- GPU4

F13 B13

- F1
- F2
- F3
- F4

- F5
- F6
- F7
- F8

- F9
- F10
- F11 B12

B11

B10

B9

B8

B7

B6

B5

B4

B3

B2

B1

(b) Looped BFS schedule (flexible partition) Time

- F12

B9

- GPU1
- GPU2
- GPU3
- GPU4

FB13 B12

B10 B9

B6

F1,2,3 F4,5,6 F7,8,9 F10,11,12

B1 F1,2,3 F4,5,6 F7,8,9

B2

B5

F10,11,12 FB13 B12

B8

B4

B1

B5

B11 B10

B7 B6

B3

B8

B4

B3

B2

B7

B11

Time

(c) RoundPipe schedule

Figure 1. Looped BFS schedule and RoundPipe schedule when training a 12-layer model with language model head (layer 13) on 4 GPUs. F/B denotes forward/backward, and numbers indicate the index of layers contained in the stage. Looped BFS schedule processes 8 microbatches at a time while RoundPipe processes them in two rounds.

GPU. We term this the weight binding issue. No matter how the pipeline is partitioned (standard looped partitioning in

- Figure 1 (a) or flexible partitioning in Figure 1 (b)), GPUs have to wait for the slowest stage (e.g., the LM Head), causing structured bubbles between stages or imbalanced bubbles within stages. In the case of training the Llama-3.1-8B model, pipeline bubbles can reach up to 30%.

Our core observation is that CPU offloading offers an opportunity to break the weight binding issue. As the master weights and activations reside in host memory and are transferred to the GPU for execution on demand, the forward pass of the same layer can be executed on a different GPU. GPUs are viewed as a stateless execution worker pool; the stage is not bound to a specific GPU, but is dispatched dynamically.

Based on this insight, we propose RoundPipe, whose pipeline schedule is shown in Figure 1 (c). RoundPipe adopts an asymmetric stage splitting strategy to ensure that different stages share similar execution time. For example, we combine three layers into a forward stage or one layer into a backward stage.2 RoundPipe leverages a round-robin task dispatching schedule and assigns stages to different GPUs in a sequential manner. Thus, the forward and backward stages can be pipelined with almost no bubbles.

Fully unleashing the potential of RoundPipe needs to tackle several challenges. First, massive amounts of data must continuously move between the host and GPUs. To prevent large parameter transfers from blocking critical-path activation transfers and the compute stream, RoundPipe introduces a priority-aware transfer scheduling engine, packing parameter

transfers into idle windows between critical-path activation transfers, ensuring seamless overlap with computation.

Second,RoundPipeexecutesoptimizerupdatesasynchronously

to hide latency. This introduces race conditions over hostresident data, where the GPU and the CPU-side optimizer simultaneously read and write this data. To guarantee data consistency without reintroducing pipeline-stalling barriers, RoundPipeimplementsafine-grained,distributedevent-based protocol that enforces strict execution ordering at the individual layer level.

Finally, to eliminate manual tuning and maintain nearoptimal load balancing across asymmetrically partitioned stages, RoundPipe features an automated stage-splitting algorithm that efficiently (with 𝑂(𝐿3) complexity) computes a pipeline partition.

We evaluate RoundPipe across 8×RTX 4090 and 8×A800 SXM servers using 1.7B to 235B models [28, 33, 46]. Compared with state-of-the-art baselines, RoundPipe delivers: (1) on 4090s, up to 2.16× higher throughput and 7.3× longer sequences; (2) on A800s, matching their throughput for small models, with up to 1.47× speedups and 5.6× longer sequences on large ones. Remarkably, RoundPipe is the only system capable of LoRA fine-tuning a 235B MoE model on 24GB GPUs. Furthermore, its 4090 throughput reaches at least 76% of existing A800 solutions across all models, effectively bridging the performance gap between consumer and data-center hardware.

In summary, the main contributions of this paper are as follows:

2Due to activation recomputation, their execution times are equal.

- • We identify the limitations of pipeline parallelism on consumer-grade GPUs and propose RoundPipe as a solution.
- • We propose a set of system designs, including a multi-stream architecture for data overlap and a fine-grained event-based consistency protocol, alongside a stage-splitting algorithm to fully unleash the potential of RoundPipe.
- • We evaluate RoundPipe comprehensively to demonstrate its effectiveness and training efficiency compared to stateof-the-art solutions.

### 2 Background and Motivation

Consumer-grade GPUs offer high computational power at a fraction of the cost of datacenter accelerators. However, their limited hardware memory capacity and low-bandwidth PCIe interconnects pose bottlenecks for training large models. In this section, we first review the memory pressure and existing mitigation techniques (§2.1). We then discuss multi-GPU parallelism under the offloading setting (§2.2), highlighting the limitations of current pipeline schedules in resolving structural and imbalance bubbles (§2.3).

#### 2.1 Memory-Consuming Training and Mitigations

On memory-constrained consumer GPUs, one primary challenge in training or fine-tuning large models is memory pressure. This pressure stems from the massive footprint of static model states (including parameters, gradients, and optimizer states) and runtime intermediate activations.

First, the continuous scaling of model size to unlock emergent capabilities [20] increases the basic memory required for model states. Under standard mixed-precision [29] training with the Adam optimizer [37], a Φ-parameter model state occupies 16Φ bytes. For a 32B model, the basic memory usage has reached 512GB.

Second, the continuous expansion of context windows for complex tasks like reasoning [8], video generation [47], and document-level understanding [1] drastically inflates the runtime activation that must be stored for backpropagation. Theactivationfootprintpertransformerlayerincreaseslinearly with input sequence length. For example, training a LLaMA-

##### 3.1-8B [28] model with a single 16k-token sequence generates 68 GB of activations from all layers (detailed in Appendix B.1). Furthermore, the use of micro-batches (which linearly scale up the activation) is common during distributed training.

Several techniques can mitigate the training memory pressure on consumer-grade GPU servers. We will review representative approaches below and discuss their trade-offs.

- 2.1.1 Activation Recomputation. Activation recomputation or gradient checkpointing [4] avoids storing intermediate activations during the forward pass. Instead, only the input to each transformer layer is retained. The remaining activations are recomputed on-the-fly by re-executing the forward pass with the stored layer input before the backward pass of every

Recompute vs Activation Reload Time on 4090

Recompute Time

50

Activation Reload Time

40

Time(ms)

30

20

10

0

Qwen3 1.7B

Llama-3.1 8B

gpt-oss 20B

Qwen3 32B

Qwen3 235B

Figure 2. Theoretical time of recomputing and reloading activations of a transformer layer. Calculation details in Appendix B.2.

layer. This reduces the intermediate activations’ size to 2𝑠𝑏ℎ bytes per layer [22].

While the extra computation may seem expensive, it is often more efficient than swapping activations to host memory over PCIe. Figure 2 compares the execution time of activation recomputation and reloading activations from host memory on 4090. Recomputing a transformer layer is 2.37× ∼ 5.75× faster than reloading activations from host memory. Hence, the reloading overhead cannot even be mitigated by layer-by-layer computation-transmission overlap. Activation recomputation has therefore become a common practice in large-scale training.ItisfavoredbymainstreamLLMtrainingframeworks[53], and is employed even on datacenter clusters with sufficient GPU memory [19, 27]. In this paper, we adopt full activation recomputation as a basic assumption. We also offload the checkpointed activation to host DRAM.

- 2.1.2 Weight and Optimizer Offloading. To reduce the static memory footprint of model states, prior work has proposed to offload optimizer states, gradients, and often parameters to host memory, and to execute the optimizer on the CPU [38, 40]. Model weights are transmitted to the GPU layer by layer for gradient computation, and the resulting gradients are sent back to the host memory for optimizer updates. A host-side optimizer step is typically slow for large models, 9.6s for a 32B model [40]. Since the execution time of the CPU optimizer is comparable to the scale of GPU computation, synchronous execution of both reduces training efficiency. Prior work has therefore adopted staleness-1 asynchronous updates. Training iteration 𝑇+1 will read the weights produced after iteration 𝑇−1, while the CPU applies iteration-𝑇 gradients to model weights in the background [24, 40]. These works demonstrate that one-step staleness does not harm convergence, making asynchronous optimizer a practical option for memory-constrained training systems.
- 2.2 Parallelism with Offloading

Given the limited PCIe interconnect of consumer GPUs, another critical factor for training efficiency is how to parallelize across multiple consumer GPUs.

DP-based parameter offloading. A simple approach is to combine data parallelism with host memory offload. DeepSpeed ZeRO-Offload [40] and ZeRO-Infinity [38] partition model states across data-parallel ranks, and offload them to CPU memory or NVMe storage when GPU memory is insufficient.

However, as a data-parallel approach, ZeRO requires exchanging the full set of parameters between every GPU for each forward and backward computation. It is usually completed through a high volume of collective communications like all-gather. On consumer-grade servers with low PCIe bandwidth, these communications become the major performance bottleneck. A previous study shows that DeepSpeed spends about 70% of training time on communication on consumer GPU servers [12], limiting the efficiency of multi-GPU scaling on cost-effective hardware.

PP-based parameter offloading. To overcome the communication bottleneck inherent in data-parallel offloading, Mobius [12] proposes a pipeline-parallel approach to offloaded training. It partitions a model’s layers into pipeline stages and assigns each stage to a different GPU. These stages are stored in DRAM, and each stage’s weights are loaded to its assigned GPU before executing it. This reduces communication traffic, as only activations and gradients pass between GPUs. However, Mobius inherits the pipeline bubble problem from existing pipeline parallelism.

#### 2.3 Motivation: Pipeline Schedules and Bubbles

Pipeline parallelism typically suffers from two types of performance-degrading bubbles. As shown in Figure 1 (a), Structural bubbles arise from the data-dependency of forward and backward propagation under balanced stages. Imbalance bubbles arise when some stages run longer than others, forcing dependent stages to stall. Existing pipeline schedules struggle to mitigate both types of bubbles simultaneously when training large models.

Structural Bubbles. Structural bubbles come from data dependencies in pipeline execution. Non-looped schedules such as GPipe [16] and 1F1B [15] assign one stage to each GPU. The first stage starts first and finishes last, so some GPUs are idle at the beginning and end of the iteration. Given a model with 𝑀 layers and split into 𝑆 stages, this results in a pipeline bubble of 𝑆−1

𝑀+𝑆−1, which can be substantial when 𝑀 is not much larger than 𝑆.

Zero-Bubble schedules [36] attempt to fill these idle slots by reordering computations. However, they require holding activations for extended periods. This conflicts with activation recomputation and leads to prohibitive memory consumption for large-scale models. Looped schedules (e.g., Interleaved 1F1B [32], Looped BFS [23]) offer a more memory-efficient alternative. They assign an equal number of stages to each GPU, creating a finer-grained pipeline of total 𝑆 = 𝑣𝑁 stages

Ideal vs Real Bubble Ratio (microbatch=16)

30

ideal

interleaved-1f1b

BubbleRatio(%)

looped-bfs

20

10

0

Qwen3-1.7B Llama-3.1-8B gpt-oss-20B Qwen3-32B Qwen3-235B

Figure 3. Bubble ratio of Looped schedules under ideal balanced partition and real-world imbalanced partition on 8 GPUs. Real-world bubble ratios are collected in §5.6.1

across all 𝑁 GPUs. This reduces the bubble ratio to approximately 𝑆·𝑀𝑁+·(𝑁𝑁·(−𝑁1)−1) , and decreases as the number of stages grows.

Imbalance Bubbles. Theoretical analyses of pipeline bubbles assume uniform stage execution times, which are frequently violated by real-world models. In practice, stage latencies diverge significantly; the slowest bottleneck stage stalls subsequent operations, inducing imbalance bubbles within the pipeline. Consequently, this results in a significant disparity between the actual and ideal bubble ratios for the looped schedule, as shown in Figure 3.

Limitations. Crucially, existing schedules face a dilemma because they force the total stage count to be an exact integer multiple of the GPU count. Coarse-grained partitioning with fewer stages (e.g., GPipe) incurs high structural bubbles. Conversely, fine-grained partitioning with more stages (e.g., looped schedules) leads to severe load imbalance: as each stage contains fewer layers, compute-heavy components, such as the LM head, introduce significant inter-stage imbalance regardless of where they are assigned. Consequently, as demonstrated in Figure 3, the overall bubble ratio can reach up to 30% in current pipeline schedules.

### 3 Introuding RoundPipe

In this section, we introduce RoundPipe. By leveraging the opportunities presented by offloading, we decouple pipeline stages from specific physical GPUs and propose a novel computation dispatch paradigm (§3.1). Building on this foundation, we employ round-robin dispatch and asymmetric splitting to synthesize a new pipeline schedule supporting an arbitrary number of stages (§3.2). We analyze the bubble ratio of this pipeline and the overhead introduced by the computation dispatch paradigm (§3.3).

#### 3.1 Computation Dispatch Paradigm

Strawman: flexible pipeline partition. Recall that existing pipeline schedules require partitioning stages into integer multiples of the number of devices 𝑆 = 𝑣𝑁, resulting in imbalance bubbles (as shown in Figure 1 (a)). An intuitive

approach to reduce the imbalance bubble is to apply a flexible stage partitioning (such as 13 stages in Figure 1 (b)). This approach can reduce imbalance bubbles by reducing the execution time imbalance between layers.

However, this approach increases structural bubbles. In

- Figure 1 (b), GPU 1 is assigned more stages, while GPUs 2-4 have fewer stages. Unfortunately, the weights (and corresponding computations) of a stage in existing pipeline schedules are fixed to specific GPUs and do not move during training. Therefore, both the forward and backward passes of a stage reside on the same physical device. Since each GPU can only process its pre-assigned and bound stages, GPUs 2-4 are underutilized, creating structural bubbles between their forward and backward passes when waiting for data from the busiest GPU (GPU 1).

Opportunity: computation dispatching. We find that offloading model states to the CPU presents a unique opportunity to increase the pipeline schedule efficiency. In PP-based offloading, the model states are offloaded to host memory and transferred to the GPU on demand for execution. It is no longer necessary to bind a stage to a fixed GPU. This allows us to freely relocate the stage computation to an underutilized GPU. Intuitively, this incurs no extra communication cost; given that weight transfers between the CPU and GPU are already necessary, this method essentially reassigns the target GPU for these weights and their associated computations. We evaluate its cost in §3.3.

Thus, we propose the computation dispatch paradigm. Model states and activations reside on the host, and computations (with corresponding model stage and activation) are dispatched to GPUs for execution. In this way, any GPU can execute any stage as soon as the requisite data is ready. By doing so, we can execute a flexible pipeline with a small structural bubble, achieving a simultaneous reduction in both imbalance and structural bubbles.

#### 3.2 RoundPipe Schedule

We concretize the idea of computation dispatch with the RoundPipe schedule (Figure 1c). RoundPipe incorporates a round-robin stage dispatch pattern and an asymmetric stage splitting strategy.

Round-robindispatch. Similarlytoloopedschedules,Round-

Pipe distributes stages across GPUs in a round-robin fashion; however, it unifies both forward and backward stages into a single continuous dispatch sequence. RoundPipe divides the 𝑀 micro-batches into multiple rounds, processing 𝑀𝑅 (𝑀𝑅 ≥ 𝑁) micro-batches per round on 𝑁 GPUs. Within each round, the 𝑆𝑓 forward stages are followed by the 𝑆𝑏 backward stages, forming a linear sequence of 𝑆 = 𝑆𝑓 + 𝑆𝑏 stage slots. Stage slot 𝑖 (counting from 0 across the concatenated forwardthen-backward sequence) is dispatched to 𝐺𝑃𝑈(𝑔0+𝑖) mod 𝑁, where 𝑔0 is the starting GPU index for the round. Each GPU

GPUs Iter T-1

Iter T Iter T+1

CPU

Optim T-2 Optim T-1

Optim T

###### Time

Figure 4. Asynchronous optimizer update in RoundPipe.

executes all 𝑀𝑅 micro-batches of the current round for its assigned stage before the next stage slot is dispatched.

Between rounds, the dispatch seamlessly resumes where the previous round left off. That is, the starting index updates to 𝑔0 ← (𝑔0 + 𝑆) mod 𝑁, assigning the first stage of the new round to the GPU next in line. This dispatch logic ensures a continuous, zero-bubble flow of stages across rounds. Stage dispatch will repeat 𝑅 = 𝑀/𝑀𝑅 rounds until all 𝑀 microbatches have completed both their forward and backward passes.

Asymmetric stage splitting. The forward pass of a transformer layer is faster than its backward pass with recomputation. The symmetric stage splitting adopted by conventional pipeline parallelism enforces identical layer partitions across both passes, creating idle bubbles at the transition boundary between the faster forward and slower backward stages. In contrast, RoundPipe maintains separate partitions for forward and backward pass, which enables a balanced partition that equalizes per-stage execution time across the entire forwardto-backward sequence.

To formalize this asymmetric partitioning, RoundPipe defines (𝐹1, 𝐹2, . . ., 𝐹𝑆𝑓 ,𝐵1) as the forward partition, where 𝐹𝑖 is the number of consecutive layers in forward stage 𝑖; and defines (𝐵1,𝐵2, . . .,𝐵𝑆𝑏) as the backward partition, where 𝐵𝑗 is the number of consecutive layers in backward stage 𝑗. To reduce redundant computation, RoundPipe fuses the forward and backward pass for the 𝐵1 layers. In this fused stage, these forward computations serve as the recomputation required by the backward pass, saving one forward pass’s worth of computation for these layers. All stages should have approximately equal computation time across both directions, satisfying 𝑖 𝐹𝑖 + 𝐵1 = 𝐿 and 𝑗 𝐵𝑗 = 𝐿, where 𝐿 is the number of model layers.

By breaking the symmetry between forward and backward partitions, asymmetric splitting bridges the gap between the forward and backward phases, eliminating RoundPipe’s pipeline bubbles at phase boundaries.

Supporting asynchronous optimizer. RoundPipe natively supports staleness-1 asynchronous optimizer update (§2.1.2). RoundPipe’s round-robin dispatch is defined solely in terms of the stage sequence, not the iteration boundary. As shown in Figure 4, iteration 𝑇 continues the round-robin assignment from where iteration 𝑇 − 1 left off without pipeline flush.

With asynchronous updates enabled, the warm-up and cooldown bubbles that appear at iteration boundaries are thereby eliminated.

#### 3.3 Benefits and Tradeoff Analysis

Bubble analysis. We now quantify the pipeline bubble ratio of the RoundPipe schedule. Let 𝑡 denote the execution time per micro-batch for each stage. With the stage count 𝑆 = 𝑆𝑓 +𝑆𝑏, the workload of all 𝑀 micro-batches costs 𝑀 ·𝑆 ·𝑡 GPU time. The bubble of RoundPipe comes from the startup cost of filling the pipeline in the first round and the cool-down cost of draining the pipeline after the last round, which are symmetric and together contribute 𝑁 · (𝑁 − 1) · 𝑡 GPU time. The total GPU time is therefore (𝑀 · 𝑆 + 𝑁 · (𝑁 − 1)) · 𝑡 and the bubble ratio is 𝑀·𝑆𝑁+·(𝑁𝑁·(−𝑁1)−1) . Although RoundPipe shares the same ratio formula with looped schedules (Interleaved 1F1B, Looped bfs), the number of stages in RoundPipe is the sum of forward and backward stages, which is around

###### 4

- 3× larger than looped schedules. 3 Thus, the bubble ratio of RoundPipe is smaller than that of looped schedules. In addition, RoundPipe’s flexible stage partitioning allows it to achieve better time balance and therefore smaller imbalanceinduced bubbles than looped schedules, as we verify in §5.6.1.

Roofline analysis. We conducted a roofline analysis [49] to evaluate whether the data transfers of the computation dispatch paradigm introduce bottlenecks that block GPU execution in RoundPipe. We conclude that the PCIe transfer time can be entirely overlapped by computation simply by using typical training batch sizes (as small as 𝐵 = 8 for dense models and 𝐵 = 80 for MoE models). Detailed analysis is available in Appendix C.

By establishing that dispatching layer computations over PCIe does not reduce GPU utilization, we show that the computation dispatch paradigm is not a throughput–memory trade-off. Rather, it is a throughput-preserving reorganization that unlocks a strictly larger scheduling search space, which we exploit in §3.2. The actual realization of this overlap depends on the implementation’s ability to pipeline transfers with kernel execution, which we detail in §4.2.

- 4 Design and Implementation

The previous section presented RoundPipe’s scheduling algorithm in the abstract. Realizing this abstraction as a practical training framework involves mapping the logical pipeline to physical hardware efficiently. §4.1 introduces the overall system architecture and workflow, followed by the core technical challenges. Subsequent subsections detail how RoundPipe addresses data transfer overlap (§4.2), parameter consistency (§4.3), and stage partitioning (§4.4).

Model

GPU Worker 0

Dispatch

Stage Partition (§4.4)

…

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

GPU Worker N

Data Transfer (§4.2)

Optimizer Worker

step()

Consistency Protocol (§4.3)

Model Optimizer Copy (High Precision)

Model Master Copy (Low Precision)

Figure 5. RoundPipe system overview.

#### 4.1 RoundPipe Overview

RoundPipe adopts a single-controller architecture inspired by Ray [30] and veRL/HybridFlow [43], separating the control plane, which manages task scheduling and ordering, from the data plane, which handles hardware-level execution and device data transfers. As shown in Figure 5, RoundPipe consists of a controller, an optimizer worker, and one GPU worker for each GPU. RoundPipe controller transparently maps the computation onto the available hardware via GPU workers, and handles asynchronous optimizer updates on the optimizer worker.

- 4.1.1 RoundPipe Workflow. Users write a sequential training script using two APIs. During each RoundPipe invocation, the user’s thread acts as the controller to orchestrate all activity, directing per-device GPU workers and the optimizer worker that execute concurrently.

forward_backward(). From the caller’s perspective, this is identical to a standard PyTorch forward and backward pass. It accepts input tensors, returns a scalar loss, and accumulates gradients in place. Internally, RoundPipe runs a complete pipelined execution across all available GPUs. The controller constructs micro-batches from the supplied inputs and dispatches the stage to a GPU worker in round-robin order. The GPU workers execute in parallel with the controller and with one another, handling the data movement and computation asynchronously.

step(). This API dispatches gradient post-processing and weight updates (e.g., gradient scaling, clipping, and the optimizer step) to the optimizer worker. The optimizer worker applies these updates to the full-precision optimizer copy of the model, while the GPU workers concurrently compute the next iteration using the master copy without interference.

- 4.1.2 Challenges. There are several software-design challenges that, if handled carelessly, would introduce subtle performance and correctness bugs:

• Data Transfer Overlap: Under Computation Dispatch Paradigm, multiple kinds of data moves between the host

31 stage = 3 forward = 1 backward, given that the forward of a transformer layer is typically 3× faster than its backward pass with recomputation

Category (1) lies on the critical path. The next stage cannot begin until the current stage’s output activation has arrived in host memory and been re-uploaded to the target GPU. Category (2), in contrast, has lower priority and can be scheduled into any available execution window during the preceding or subsequent stages.

Stage 1 Stage 2 MB1 MB2 MB1 MB2

Parameter Upload

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

head-of-line blocking

Activation Upload

[Figure 23]

|[Figure 24]<br><br>[Figure 25]|
|---|

|[Figure 26]<br><br>[Figure 27]|
|---|

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Compute

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

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

- 4.2.1 The Multi-Stream Architecture. To exploit this overlap and fully utilize the bidirectional PCIe link, RoundPipe maintains four dedicated communication streams per device alongside the default compute stream. These four streams independently handle the two transfer categories in both directions (upload and download). Figure 7 illustrates this overlap. Activation uploads and downloads are performed one micro-batch early/delayed to avoid blocking compute, while parameter uploads and gradient downloads are interleaved into idle intervals between activation transfers. The compute stream synchronizes with the activation transfer streams at micro-batch granularity using CUDA events. This divides the timeline into bounded data-transfer windows, ensuring dependencies are met while capping in-flight data to avoid out-of-memory errors.
- 4.2.2 Scheduling Parameter Transfers. To avoid bandwidth contention on the shared PCIe link, RoundPipe confines parameter and gradient transfers to the idle intervals between activation transfers. A stage processing 𝑀 microbatches creates 𝑀 data-transfer windows, each processing one micro-batch activation and one chunk of parameter/gradient transfers. RoundPipe partitions parameter/gradient transfers into𝑀 chunksbyapplyinglongest-processing-time-firstscheduling [13]. Specifically, we sort the parameter/gradient tensors by size in descending order and assign each tensor to the interval with the smallest current total assigned transfer size. In case of very large tensors (e.g., language model head), we split them into smaller chunks before scheduling to ensure they can fit within the available data-transfer windows without causing contention.
- 4.3 Fine-Grained Parameter Consistency

Time

- Figure 6. Illustrated example of simple compute-transfer overlap on two consecutive stages with two microbatches (MB) each. Each color represents one stage.

Stage 1 Stage 2 Stage 3 MB1 MB2 MB1 MB2 MB1 MB2

Time

Chunked Parameter Upload

Activation Upload

Compute

Activation Download

Chunked Gradient Download

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

|[Figure 86]<br><br>[Figure 87]|
|---|

|[Figure 88]<br><br>[Figure 89]|
|---|

|[Figure 90]<br><br>[Figure 91]|
|---|

|[Figure 92]<br><br>[Figure 93]|
|---|

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

|[Figure 137]<br><br>[Figure 138]|
|---|

|[Figure 139]<br><br>[Figure 140]|
|---|

|[Figure 141]<br><br>[Figure 142]|
|---|

|[Figure 143]<br><br>[Figure 144]|
|---|

|[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]|
|---|

|[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]|
|---|

|[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]|
|---|

|[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]|
|---|

|[Figure 161]<br><br>[Figure 162]|
|---|

|[Figure 163]<br><br>[Figure 164]|
|---|

- Figure 7. RoundPipe multi-stream workflow over three consecutive stages with two microbatches (MB) each. Each color represents one stage.

and GPUs continuously. As shown in Figure 6, a simple overlap strategy suffers from head-of-line blocking, where large parameter/gradient transfers delay the critical-path activation transfers for the current stage. §4.2 addresses this via a priority-aware transfer scheduling engine that packs non-critical data into the idle windows of critical-path communications.

- • Asynchronous Consistency: Operating concurrently on the optimizer copy and the master copy of the model requires bridging the gap between them. The system must propagate updates between these two copies to preserve staleness-1 asynchronous optimizer update semantics without stalling pipeline execution. §4.3 addresses this via an event-based parameter consistency protocol.
- • Pipeline Stage Partition: To balance the load across the pipeline without manual tuning, the system requires an efficient mechanism for partitioning layers automatically. However, a naive search for the optimal partition incurs an unacceptable exponential time complexity. §4.4 addresses this via a 𝑂(𝐿3) two-stage partitioning algorithm.

4.3.1 The Consistency Problem. Mixed-precision training with CPU offloading maintains three representations of the parameters and gradients [40]: (1) a short-lived GPU transient copy for active computation, (2) a low-precision CPU master copy in host memory, and (3) a full-precision Optimizer copy synchronized with the master.

To coordinate iterations, RoundPipe utilizes the step() call as a logical barrier during asynchronous optimizer updates. RoundPipe must precisely control the synchronization of updated weights into the master copy (P copy) and the extraction of gradients from the master copy for the optimizer (G copy). To guarantee staleness-1 training semantics, correctness reduces to enforcing five ordering constraints among these synchronization operations, GPU data transfers, and optimizer step:

#### 4.2 Data Transfer Overlap

Under the Computation Dispatch Paradigm, every pipeline stage requires two categories of data movement: (1) permicro-batch activation uploads and downloads between host and GPU, and (2) per-stage parameter uploads and gradient downloads between host and GPU.

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

Parameter Upload Compute Gradient Download

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |

GPUs

|forward_backward step T|
|---|

barrier

controller

P cp G cp O forward_backward step T+1

Time

optimizer worker

optim step

optim step

(a) Blocking Copy

| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |

GPUs

| | | |
|---|---|---|
|forward_backward step T|O|forward_backward step T+1|
| | | |

controller

[Figure 165]

[Figure 166]

… optimizer worker …

P cp Gcp Time

optim step optim step

[Figure 167]

[Figure 168]

(b) Distributed, Event-Based Protocol

- Figure 8. (a) The blocking approach copies weights (P cp) and gradients (G cp) on the main thread. (b) RoundPipe’s event-based protocol offloads copies to the optimizer worker (O) and uses per-layer events. The four arrows from left to right correspond to ordering constraints (1)-(4), respectively.

- (1) Protects weight integrity. P copy must wait until the GPU has finished uploading parameters for the previous iteration.
- (2) Protects against torn weights. GPU must wait until P copy completes before uploading parameters for the next iteration.
- (3) Protects against incomplete gradients. G copy must wait until the GPU has finished downloading the previous iteration’s gradients into the master buffer.
- (4) Protects gradient integrity. GPU must wait until G copy completes before writing next iteration’s gradients into master buffer.
- (5) Protects optimizer step semantics. Data copies must be scheduled between the just-completed optimizer step and the next incoming step.

- 4.3.2 Blocking vs. Event-Based Protocol. A straightforward blocking copy (Figure 8(a)) treats the step() call as a concrete barrier, halting all workers until floating-point synchronization concludes on the CPU, artificially reintroducing pipeline bubbles.

To eliminate this, RoundPipe uses a Event-Based Protocol (Figure 8(b)). RoundPipe offloads weight and gradient copies to the optimizer worker and coordinates access through a set of threading events. The ordering constraints (1)-(4) should be preserved by four dependency edges, as shown in Figure 8(b), while (5) is naturally enforced by the optimizer worker’s internal sequential execution. RoundPipe utilizes point-topoint event signaling to satisfy these edges. The controller dynamically creates and dispatches events for workers to wait for or to set, allowing them to resolve ordering constraints directly in a peer-to-peer manner.

Crucially, synchronization applied to the whole model would still stall the shallowest layer (Layer 1), which sits at the

tight turnaround between the end of the backward pass and the beginning of the next forward pass. Therefore, RoundPipe binds these synchronization events to individual layers. This fine-grained protocol allows the optimizer worker to release events for early layers immediately after processing them, empowering GPU workers to begin the next iteration’s forward pass on Layer 1 while deeper layers are still synchronizing.

#### 4.4 Automatic Stage Partitioning

- 4.4.1 Problem Formulation. We collect per-layer execution time and memory consumption for stage partitioning during the first few iterations. Since RoundPipe overlaps most communication with computation, we simplify the stage partitioning problem to find a near-optimal partition. We model the execution time of all stages as the maximum stage runtime to calculate the total GPU time of the pipeline. Our goal is to

minimize (𝑀 · 𝑆 + 𝑁 · (𝑁 − 1)) · 𝑡max (§3.3) while preserving GPU memory constraints on every stage.

- 4.4.2 Algorithm. RoundPipe exploits the observation that, because each stage must contain contiguous layers, the set

of possible values for 𝑡max is bounded to all contiguous subsequence sums of forward and backward times (𝑂(𝐿2) possibilities). For each candidate 𝑡max value, the problem reduces to: find the minimum number of contiguous partitions such that no partition’s total time exceeds 𝑡max and no partition’s memory consumption exceeds the GPU memory size. This is a classic greedy problem solvable in 𝑂(𝐿) time by scanning layers and packing them into stages until a constraint is violated, leading to an overall 𝑂(𝐿3) partitioning algorithm. During the greedy process, RoundPipe fills the first backward stage first to ensure it is filled as fully as possible. This greedy choice maximizes the size of the first backward stage, which

is beneficial because this stage skips activation recomputation (§3.2), directly increasing the total computational savings.

### 5 Evaluation

In this section, we first evaluate RoundPipe on both consumergrade and datacenter-grade GPU servers, then study its scaling with sequence length and GPU count. We finally investigate the pipeline bubbles of RoundPipe and the effectiveness of RoundPipe design.

#### 5.1 Experimental Setup

Hardware. We use two server configurations. Unless otherwise stated, all experiments use 8 GPUs on a single server.

- • 4090 server: 8× NVIDIA RTX 4090 GPUs (24 GB VRAM each), Intel Xeon Gold 6330 CPU, 800 GB available DDR4 host memory, PCIe 4.0 (32GB/s) interconnect.
- • A800 server: 8× NVIDIA A800 SXM GPUs (80GB HBM2e each), Intel Xeon Platinum 8352Y CPU, 800GB available DDR4 host memory, NVLink 3.0 (200GB/s) interconnect.

Baselines. We compare against six representative training frameworks: (1) DeepSpeed ZeRO-2 [37], (2) PyTorch FSDP[35](PyTorch’simplementationofZeRO-3DP),(3)DeepSpeed ZeRO-Infinity [38] (Offloading to CPU memory),

- (4) Megatron-LM Pipeline Parallelism (Megatron-PP) [32],
- (5) Megatron-LM Tensor Parallelism (Megatron-TP) [44], and
- (6) Mobius [12]. We also report RoundPipe-sync, a variant of RoundPipe that disables asynchronous optimizer updates (§3.2) and performs a synchronous CPU optimizer step.

Workloads. We evaluate five open-source models: Qwen31.7B [46], LLaMA-3.1-8B [28], GPT-OSS-20B [33], Qwen332B, and Qwen3-235B-A22B. GPT-OSS-20B and Qwen3235B-A22B are Mixture-of-Experts (MoE) models, while the other three are dense Transformers. We perform fullparameter training on the first four models and apply LoRA fine-tuning (𝑟 = 32) exclusively to the massive Qwen3-235B. The sequence length is fixed at 2048, with global batch sizes of 512, 256, 128, 128, and 64 for each model, respectively. All frameworks employ mixed-precision (FP16) training, full activation recomputation, and identical global batch sizes. We maximize the micro-batch size for each framework and model individually to ensure full GPU utilization, subject to two constraints: avoiding device out-of-memory (OOM) and maintaining a sufficient number of micro-batches (e.g., ≥ 8) to sustain high pipeline or data parallel throughput.

Metrics. We report two metrics: (1) Training throughput (tokens/s): the number of tokens processed per second during steady-state training, averaged over 10 iterations after warmup. (2) Maximum sequence length: the longest sequence length at which training completes without out-of-memory errors, searched at micro-batch size=1.

Training Throughput on 8×RTX 4090

ZeRO-2 FSDP ZeRO-Infinity Megatron-TP

Megatron-PP Mobius RoundPipe-sync RoundPipe

| |
|---|

60000

Throughput(tokens/s)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50000

| |
|---|

| |
|---|

40000

30000

20000

10000

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

N/A

0

Qwen3 1.7B

LLaMA-3.1 8B

GPT-OSS 20B

Qwen3 32B

Qwen3 235B-LoRA

Figure 9. Training Throughput on 8×RTX 4090.

Max Sequence Length on 8×RTX 4090

ZeRO-2 FSDP ZeRO-Infinity

Megatron-TP Megatron-PP

Mobius RoundPipe

| |
|---|

| |
|---|

MaxSequenceLength

| |
|---|

| |
|---|

| |
|---|

60K

| |
|---|

40K

20K

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

OOM

N/A

0K

Qwen3 1.7B

LLaMA-3.1 8B

GPT-OSS 20B

Qwen3 32B

Qwen3 235B-LoRA

Figure 10. Maximum trainable sequence length on 8×RTX 4090.

5.2 End-to-End Performance on 4090 Figures 9 and 10 present end-to-end throughput and maximum sequence length results on the 4090 server. 4

Throughput. BothRoundPipeandRoundPipe-syncachieve

the highest throughput across all five models. RoundPipe outperforms the fastest existing systems across 1.7-32B models by 1.48 ∼ 2.16× on training throughput, and RoundPipe-sync outperforms by 1.15 ∼ 1.63×. This performance advantage is driven by our novel pipeline design, which substantially mitigates pipeline bubbles and yields improved communicationcomputation overlap compared with existing frameworks. Furthermore, by strictly bounding the number of layers resident on the GPU, RoundPipe is the only system capable of successfully LoRA-finetuning the massive 235B model on 24GB GPUs.

Maximum Sequence Length. RoundPipe consistently supports the longest sequences. Excluding Megatron-TP, whose throughput becomes impractical under PCIe, RoundPipe extends the maximum sequence length by 4.7 ∼ 7.3× over the next-best baseline. Optimizer asynchrony does not affect this metric because it does not change GPU memory usage. On Qwen-1.7B, RoundPipe and Megatron-TP both handle substantially longer sequences than other baselines. TP achieves this by sharding activations across GPUs, while

4N/A on Megatron-TP when LoRA finetuning Qwen3-235B because the model has four key value heads, which does not support TP=8 on Megatron.

Training Throughput on 8×A800 SXM

ZeRO-2 FSDP ZeRO-Infinity

Megatron-PP RoundPipe-sync RoundPipe

| |
|---|

80000

Throughput(tokens/s)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Megatron-TP

RoundPipe-4090

60000

40000

20000

OOM

OOM

OOM

OOM

OOM

N/A

0

Qwen3 1.7B

LLaMA-3.1 8B

GPT-OSS 20B

Qwen3 32B

Qwen3 235B-LoRA

Figure 11. Training Throughput on 8×A800.

Qwen3-1.7B

60000

LLaMA-3.1-8B

GPT-OSS-20B

Throughput(tokens/s)

Qwen3-32B

50000

Qwen3-235B-LoRA

40000

30000

20000

10000

0

0 1 2 3 4 5 6 7 8

Number of RTX 4090 GPUs

Figure 13. RoundPipe throughput on 1∼8 RTX 4090 GPUs.

Max Sequence Length on 8×A800 SXM

300K

ZeRO-2

ZeRO-Infinity

Megatron-PP

| |
|---|

| |
|---|

MaxSequenceLength

FSDP

Megatron-TP

RoundPipe

250K

200K

150K

100K

50K

OOM

OOM

OOM

OOM

OOM

N/A

0K

Qwen3 1.7B

LLaMA-3.1 8B

GPT-OSS 20B

Qwen3 32B

Qwen3 235B-LoRA

Figure 12. Maximum trainable sequence length on 8×A800.

RoundPipe achieves it by storing stage-boundary activations in host memory and recomputing layer-internal activations on demand. RoundPipe avoids TP’s heavy communication overhead, thereby achieving higher training throughput than TP. As model size scales up, non-offloading systems are bottlenecked by model footprint and fail to scale sequence lengths, whereas existing offloading systems (e.g., ZeRO-Infinity) still OOM early because they do not offload activations. RoundPipe even supports a slightly longer sequence on Qwen3-235B than on Qwen3-32B because LoRA fine-tuning produces fewer intermediate activations and gradients than full-parameter training during recompute and backward pass.

#### 5.3 End-to-End Performance on A800

To evaluate the overhead of RoundPipe and the Computation Dispatch Paradigm, we compare RoundPipe with SOTA trainingsystemsondatacenterGPUswithampleGPUmemory and interconnect. Figure 11 and 12 present the results. We omit Mobius because it reports lower performance on datacenter GPUs compared to DeepSpeed ZeRO-Infinity [12].

Throughput. Even on hardware-rich servers, RoundPipe delivers 0.98 ∼ 1.47× the throughput of SOTA systems. For smaller models (1.7B and 8B), Data Parallelism (DP) performs best by fully exploiting the high-bandwidth NVLink. Remarkably, RoundPipe achieves highly competitive performance without utilizing any GPU peer-to-peer communication (NVLink), relying entirely on PCIe host-to-device

transfers under the Computation Dispatch Paradigm. Meanwhile, RoundPipe-sync matches the speed of DeepSpeed ZeRO-Infinity, with their slowdown attributed to CPU optimizer step overhead. For larger models (20B+), TP and PP overtake DP by avoiding full-parameter synchronization, but RoundPipe still leads by generating less communication than DP/TP and fewer bubbles than existing PP systems. On Qwen-32B, Megatron-PP encounters an OOM error because the 64-layer model must be partitioned across 8 GPUs. The final rank is burdened with 8 layers plus the LM head, encountering OOM during backward. Furthermore, Figure 11 plots RoundPipe’s performance on the 4090 server. Although the 4090’s lower VRAM bandwidth inherently caps its peak compute utilization compared to the A800, RoundPipe still consistently achieves over 76% of the throughput of existing SOTA frameworks running on the A800 across all models. This proves that RoundPipe elevates consumer GPUs beyond cost-effective alternatives; they now offer absolute training times comparable to the expensive datacenter hardware.

Maximum Sequence Length. On the A800 server, RoundPipe increases the maximum sequence length by 1.19 ∼ 5.62× over the baselines, following the same trend as on the 4090. For smaller models (≤ 20B), Megatron-TP uses NVLink to support 108K∼242K tokens, but RoundPipe pushes this further to 192K∼288K by storing model states and stage-boundary activations in CPU memory. For larger models, TP and the other baselines again become limited by GPU memory, whereas RoundPipe maintains its lead.

#### 5.4 Scalability

We conducted strong-scaling experiments with fixed workloads to evaluate RoundPipe’s multi-GPU efficiency. Figure 13 illustrates that RoundPipe achieves near-linear throughput scaling from 1 to 8 GPUs across all model sizes. The stability of this scaling curve underscores the robustness of the RoundPipe, showing that its partitioning and overlap mechanisms remain effective across GPU counts.

Crucially, RoundPipe exhibits a unique architectural advantage: the maximum sequence length is independent of the GPU count. From 1 to 8 GPUs, the maximum sequence

Qwen3-1.7B on 8×RTX 4090

Throughput(tokens/s)

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

60000

40000

20000

0

512 1K 2K 4K 8K 16K 32K 64K

Sequence Length

Figure 14. Throughput vs. sequence length for Qwen3-1.7B.

Bubble Ratio by Model and Schedule

GPipe

Interleaved-1F1B

RoundPipe-Sync

1F1B

Looped-BFS

RoundPipe

40

BubbleRatio(%)

20

0

Qwen3 1.7B

Llama-3.1 8B

GPT-OSS 20B

Qwen3 32B

Qwen3 235B

80

Timeperiteration(s)

Blocking Copy

Fine-Grained Consistency

60

40

20

0

Qwen3 1.7B

LLaMA 3.1-8B

GPT-OSS 20B

Qwen3 32B

Qwen3 235B-LoRA

Figure 15. Simulated pipeline bubble rate on each schedule with different models.

Figure 16. Blocking copy vs. finegrained consistency protocol.

length remains 73K, 49K, 39K, 28K, and 31K for the five models. This invariant is a direct product of our Computation Dispatch Paradigm. A GPU holds data required for its currently assigned stage only, while the remaining model state and stage-boundary tensors for recompute are stored in host memory.

#### 5.5 Sequence Length Sensitivity

Figure 14 shows how RoundPipe’s throughput changes as the sequence length increases over two orders of magnitude on 8×RTX 4090 with Qwen3-1.7B. RoundPipe maintains robust performance across the entire range, with throughput decreasing smoothly as attention cost grows. This shows that RoundPipe supports both short and very long contexts without prohibitive overhead.

#### 5.6 Ablation Studies

- 5.6.1 Pipeline Schedule. To isolate the contribution of RoundPipe’s pipeline schedule from the end-to-end performance, we simulate the bubble ratio of five pipeline schedules using per-layer timing data collected from the 4090 server. For each model, we measure forward and backward execution times of each transformer layer and the language model head using a micro-batch size of 4 × 2048 tokens per micro-batch. We write a pipeline simulator that faithfully implements each schedule’s logic. We simulate the pipeline execution for 16 micro-batches on 8 GPUs.

Figure 15 shows the simulation result. RoundPipe-sync and the four baselines include start-up and cool-down bubbles, whereas RoundPipe with asynchronous optimizer updates hides those costs in the adjacent iteration. RoundPipe-sync reduces bubbles by 23% ∼ 55% relative to the best baseline, and its bubble ratio further decreases as deeper models are split into more stages. RoundPipe with asynchronous optimizer updates virtually eliminates inter-iteration bubbles, driving the absolute bubble ratio down to below 4.5%. This remaining idle time is attributable to stage execution imbalances, which our automatic partitioning algorithm (§4.4) successfully bounds to a very low level.

We measured the wall-clock time for RoundPipe’s automatic stage-splitting algorithm (§4.4). Partitioning Qwen31.7B, LLaMA-3.1-8B, GPT-OSS-20B, and Qwen3-32B takes

2.9, 2.9, 2.6, and 5.0 milliseconds, respectively, proving the algorithm is exceedingly fast in practice. Partitioning 94 layers Qwen3-235Btakes1.47seconds,whichisstillinconsequential compared to the hours required for model training.

- 5.6.2 Fine-GrainedParameterConsistencyProtocol. RoundPipe’s event-based parameter-consistency protocol (§4.3.2) avoids blocking the main thread during the FP32→FP16 weight copy and gradient collection that bridges the asynchronous optimizer with the GPU workers. To quantify the benefit, we compare it with the blocking copy baseline that inserts a global barrier before performing the FP32→FP16 weight cast on the main thread (§4.3.2). We measure the per-iteration training time of both methods to quantify the performance difference.

Figure 16 presents the results. The consistency protocol reduces a substantial overhead of 2.6 ∼ 14 seconds per iteration, growing roughly with the size of trainable parameters. Qwen3-235B-LoRA benefits less because LoRA updates only a small subset of weights and therefore copies less data. These results confirm that the event-based parameter-consistency protocol is essential for realizing the full benefit of asynchronous optimizer updates. By replacing the iteration barrier with fine-grained, per-layer event signaling, RoundPipe effectively overlaps weight and gradient synchronization with GPU computation, converting what would otherwise be idle pipeline stalls into productive work.

6 Related Work

- 6.1 Pipeline Parallel Schedules

Synchronouspipelineschedules[10,16,17,25,44]aresubject to the pipeline bubble problem. Asynchronous approaches [3, 5, 14, 15, 31, 50, 51] reduce bubbles via weight stashing, and backward-splitting schedules [36] reduce bubbles via delayed weight updates, but both techniques trade memory consumption for efficiency. Looped pipeline methods [23, 32] increase stage counts to reduce bubbles, but they require the stage count to be a multiple of the GPU count, making balanced partitioning increasingly difficult. RoundPipe takes a different approach by utilizing heterogeneous memory to decouplestagesfromGPUs,providingasynchronousschedule

with flexible stage partitions, and an asynchronous schedule with no GPU memory overhead.

#### 6.2 Offloading Training Frameworks

Heterogeneous memory offloading has been widely explored to enable training models that exceed GPU memory. One line of work offloads model weights and optimizer states to CPU or NVMe [11, 38, 45]. Another line targets activation offloading, swapping intermediate activations to the host to reduce peak GPU memory [2, 41, 48, 54]. More recent systems manage data at tensor granularity to achieve better transfer–compute overlaps [26, 39, 52]. However, these approaches are predominantly designed for single-GPU or data-parallel settings; scaling them to multiple GPUs incurs substantial communication overhead [12]. RoundPipe co-designs distributed training with host-memory offloading, achieving scaling and offloading with negligible overhead.

### 7 Conclusion

This paper presents RoundPipe, a system that designs new pipeline parallelism schedule for training large models on consumer GPU servers. We introduced the Computation Dispatch Paradigm and confirmed that it preserves full compute-bound throughput. Building on this paradigm, the RoundPipe schedule uses asymmetric stage splitting and round-robin dispatch to decouple stages from GPUs, mitigate stage imbalance, and improve pipeline efficiency. Experiment results demonstrate its performance gain.

### References

- [1] Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2023. L-Eval: Instituting Standardized Evaluation for Long Context Language Models. arXiv:2307.11088 [cs.CL] https://arxiv.org/abs/2307.11088
- [2] Jonghyun Bae, Jongsung Lee, Yunho Jin, Sam Son, Shine Kim, Hakbeom Jang, Tae Jun Ham, and Jae W Lee. 2021. {FlashNeuron}:{SSDEnabled}{Large-Batch} training of very deep neural networks. In 19th USENIX conference on file and storage technologies (FAST 21). 387–401.
- [3] Chi-Chung Chen, Chia-Lin Yang, and Hsiang-Yun Cheng. 2018. Efficient and robust parallel dnn training through model parallelism on multi-gpu platform. arXiv preprint arXiv:1809.02839 (2018).
- [4] TianqiChen,BingXu,ChiyuanZhang,andCarlosGuestrin.2016. Training Deep Nets with Sublinear Memory Cost. arXiv:1604.06174 [cs.LG] https://arxiv.org/abs/1604.06174
- [5] Zihao Chen, Chen Xu, Weining Qian, and Aoying Zhou. 2023. Elastic averaging for efficient pipelined DNN training. In Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming. 380–391.
- [6] Tri Dao. 2023. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691 [cs.LG] https: //arxiv.org/abs/2307.08691
- [7] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré.

2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv:2205.14135 [cs.LG] https://arxiv.org/abs/2205. 14135

- [8] DeepSeek-AI. 2025. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature 645, 8081 (Sept. 2025),

- 633–638. doi:10.1038/s41586-025-09422-z
- [9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.

2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics (NAACL). 4171–4186.

- [10] Shiqing Fan, Yi Rong, Chen Meng, Zongyan Cao, Siyu Wang, Zhen Zheng, Chuan Wu, Guoping Long, Jun Yang, Lixue Xia, et al. 2021. DAPPLE: A pipelined data parallel approach for training large models. In Proceedings of the 26th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming. 431–445.
- [11] Jiarui Fang, Zilin Zhu, Shenggui Li, Hui Su, Yang Yu, Jie Zhou, and Yang You. 2022. Parallel training of pre-trained models via chunk-based dynamic memory management. IEEE Transactions on Parallel and Distributed Systems 34, 1 (2022), 304–315.
- [12] Yangyang Feng, Minhui Xie, Zijie Tian, Shuo Wang, Youyou Lu, and Jiwu Shu. 2023. Mobius: Fine Tuning Large-Scale Models on CommodityGPUServers.InProceedingsofthe28thACMInternational Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (Vancouver, BC, Canada) (ASPLOS 2023). Association for Computing Machinery, New York, NY, USA, 489–501. doi:10.1145/3575693.3575703
- [13] R. L. Graham. 1969. Bounds on Multiprocessing Timing Anomalies. SIAM J. Appl. Math. 17, 2 (1969), 416–429. arXiv:https://doi.org/10.1137/0117039 doi:10.1137/0117039
- [14] Lei Guan, Wotao Yin, Dongsheng Li, and Xicheng Lu. 2019. XPipe: Efficient pipeline model parallelism for multi-GPU DNN training. arXiv preprint arXiv:1911.04610 (2019).
- [15] Aaron Harlap, Deepak Narayanan, Amar Phanishayee, Vivek Seshadri, Nikhil Devanur, Greg Ganger, and Phil Gibbons. 2018. PipeDream: Fast and Efficient Pipeline Parallel DNN Training. arXiv:1806.03377 [cs.DC] https://arxiv.org/abs/1806.03377
- [16] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, YonghuiWu,andZhifengChen.2019. GPipe:EfficientTrainingofGiant Neural Networks using Pipeline Parallelism. arXiv:1811.06965 [cs.CV] https://arxiv.org/abs/1811.06965
- [17] Arpan Jain, Ammar Ahmad Awan, Asmaa M Aljuhani, Jahanzeb Maqbool Hashmi, Quentin G Anthony, Hari Subramoni, Dhableswar K Panda, Raghu Machiraju, and Anil Parwani. 2020. Gems: Gpu-enabled memory-aware model-parallelism system for distributed dnn training. In SC20: international conference for high performance computing, networking, storage and analysis. IEEE, 1–15.
- [18] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas,EmmaBouHanna,FlorianBressand,GiannaLengyel,Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, MarieAnne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of Experts. arXiv:2401.04088 [cs.LG] https://arxiv.org/abs/2401.04088
- [19] Chao Jin, Ziheng Jiang, Zhihao Bai, Zheng Zhong, Juncai Liu, Xiang Li, Ningxin Zheng, Xi Wang, Cong Xie, Qi Huang, Wen Heng, Yiyuan Ma, Wenlei Bao, Size Zheng, Yanghua Peng, Haibin Lin, Xuanzhe Liu, Xin Jin, and Xin Liu. 2025. MegaScale-MoE: Large-Scale CommunicationEfficient Training of Mixture-of-Experts Models in Production. arXiv:2505.11432 [cs.LG] https://arxiv.org/abs/2505.11432
- [20] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling Laws for Neural Language Models. arXiv:2001.08361 [cs.LG] https://arxiv.org/abs/2001.08361
- [21] Heehoon Kim, Junyeol Ryu, and Jaejin Lee. 2024. TCCL: Discovering Better Communication Paths for PCIe GPU Clusters. In Proceedings of the 29th ACM International Conference on Architectural Support for

- Programming Languages and Operating Systems, Volume 3 (La Jolla, CA, USA) (ASPLOS ’24). Association for Computing Machinery, New York, NY, USA, 999–1015. doi:10.1145/3620666.3651362
- [22] Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. 2022. Reducing Activation Recomputation in Large Transformer Models. arXiv:2205.05198 [cs.LG] https://arxiv.org/abs/2205.05198
- [23] Joel Lamy-Poirier. 2023. Breadth-First Pipeline Parallelism. arXiv:2211.05953 [cs.DC] https://arxiv.org/abs/2211.05953
- [24] Tingfeng Lan, Yusen Wu, Bin Ma, Zhaoyuan Su, Rui Yang, Tekin Bicer, Masahiro Tanaka, Olatunji Ruwase, Dong Li, and Yue Cheng. 2025. ZenFlow: Enabling Stall-Free Offloading Training via Asynchronous Updates. arXiv:2505.12242 [cs.DC] https://arxiv.org/abs/2505.12242
- [25] Shigang Li and Torsten Hoefler. 2021. Chimera: efficiently training large-scale neural networks with bidirectional pipelines. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 1–14.
- [26] Changyue Liao, Mo Sun, Zihan Yang, Jun Xie, Kaiqi Chen, Binhang Yuan, Fei Wu, and Zeke Wang. 2024. LoHan: Low-Cost HighPerformance Framework to Fine-Tune 100B Model on a Consumer GPU. arXiv:2403.06504 [cs.DC] https://arxiv.org/abs/2403.06504
- [27] Xinzhang Liu, Chao Wang, Zhihao Yang, Zhuo Jiang, Xuncheng Zhao, Haoran Wang, Lei Li, Dongdong He, Luobin Liu, Kaizhe Yuan, Han Gao, Zihan Wang, Yitong Yao, Sishi Xiong, Wenmin Deng, Haowei He, Kaidong Yu, Yu Zhao, Ruiyu Fang, Yuhao Jiang, Yingyan Li, Xiaohui Hu, Xi Yu, Jingqi Li, Yanwei Liu, Qingli Li, Xinyu Shi, Junhao Niu, Chengnuo Huang, Yao Xiao, Ruiwen Wang, Fengkai Li, Luwen Pu, Kaipeng Jia, Fubei Yao, Yuyao Huang, Xuewei He, Zhuoru Jiang, Ruiting Song, Rui Xue, Qiyi Xie, Jie Zhang, Zilu Huang, Zhaoxi Zhang, Zhilong Lu, Yanhan Zhang, Yin Zhang, Yanlei Xue, Zhu Yuan, Teng Su, Xin Jiang, Shuangyong Song, Yongxiang Li, and Xuelong Li.

2025. Training Report of TeleChat3-MoE. arXiv:2512.24157 [cs.CL] https://arxiv.org/abs/2512.24157

- [28] AI @ Meta Llama Team. 2024. The Llama 3 Herd of Models. arXiv:2407.21783 [cs.AI] https://arxiv.org/abs/2407.21783
- [29] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. 2018. Mixed Precision Training. arXiv:1710.03740 [cs.AI] https://arxiv.org/abs/1710.03740
- [30] Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I. Jordan, and Ion Stoica. 2018. Ray: A Distributed Framework for Emerging AI Applications. arXiv:1712.05889 [cs.DC] https://arxiv.org/abs/1712.05889
- [31] Deepak Narayanan, Amar Phanishayee, Kaiyu Shi, Xie Chen, and Matei Zaharia. 2021. Memory-Efficient Pipeline-Parallel DNN Training. arXiv:2006.09503 [cs.LG] https://arxiv.org/abs/2006.09503
- [32] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Anand Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. 2021. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. arXiv:2104.04473 [cs.CL] https://arxiv.org/abs/2104.04473
- [33] OpenAI. 2025. gpt-oss-120b & gpt-oss-20b Model Card. arXiv:2508.10925 [cs.CL] https://arxiv.org/abs/2508.10925
- [34] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems (NIPS) 35 (2022), 27730–27744.
- [35] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward

- Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. arXiv:1912.01703 [cs.LG] https://arxiv.org/abs/1912.01703
- [36] Penghui Qi, Xinyi Wan, Guangxing Huang, and Min Lin. 2023. Zero Bubble Pipeline Parallelism. arXiv:2401.10241 [cs.DC] https://arxiv. org/abs/2401.10241
- [37] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. arXiv:1910.02054 [cs.LG] https://arxiv.org/abs/ 1910.02054
- [38] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, and Yuxiong He. 2021. ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning. arXiv:2104.07857 [cs.DC] https://arxiv.org/abs/2104.07857
- [39] Jie Ren, Jiaolin Luo, Kai Wu, Minjia Zhang, Hyeran Jeon, and Dong Li.

2021. Sentinel: Efficient tensor migration and allocation on heterogeneous memory systems for deep learning. In 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA). IEEE, 598–611.

- [40] Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He.

2021. ZeRO-Offload: Democratizing Billion-Scale Model Training. arXiv:2101.06840 [cs.DC] https://arxiv.org/abs/2101.06840

- [41] Minsoo Rhu, Natalia Gimelshein, Jason Clemons, Arslan Zulfiqar, and Stephen W Keckler. 2016. vDNN: Virtualized deep neural networks for scalable, memory-efficient neural network design. In 2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO). IEEE, 1–13.
- [42] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. 2023. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950 (2023).
- [43] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. HybridFlow: A Flexible and Efficient RLHF Framework. In Proceedings of the Twentieth European Conference on Computer Systems (EuroSys

’25). ACM, 1279–1297. doi:10.1145/3689031.3696075

- [44] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2020. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv:1909.08053 [cs.CL] https://arxiv.org/abs/1909.08053
- [45] XiaoyangSun,WeiWang,ShenghaoQiu,RenyuYang,SongfangHuang, Jie Xu, and Zheng Wang. 2022. StrongHold: fast and affordable billionscale deep learning model training. In Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (Dallas, Texas) (SC ’22). IEEE Press, Article 71, 17 pages.
- [46] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https://arxiv.org/abs/2505.09388
- [47] Team Wan. 2025. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv:2503.20314 [cs.CV] https://arxiv.org/abs/ 2503.20314
- [48] Linnan Wang, Jinmian Ye, Yiyang Zhao, Wei Wu, Ang Li, Shuaiwen Leon Song, Zenglin Xu, and Tim Kraska. 2018. Superneurons: Dynamic GPU memory management for training deep neural networks. In Proceedings of the 23rd ACM SIGPLAN symposium on principles and practice of parallel programming. 41–53.
- [49] Samuel Williams, Andrew Waterman, and David Patterson. 2009. Roofline: an insightful visual performance model for multicore architectures. Commun. ACM 52, 4 (April 2009), 65–76. doi:10.1145/1498765. 1498785
- [50] Bowen Yang, Jian Zhang, Jonathan Li, Christopher Ré, Christopher Aberger, and Christopher De Sa. 2021. Pipemare: Asynchronous pipeline parallel dnn training. Proceedings of Machine Learning and

Table 1. Summary of notations.

Notation Meaning

𝑠 sequence length 𝑏 micro-batch size ℎ hidden dimension 𝑚 intermediate dimension in MLP 𝑎 number of attention heads 𝑘 number of key-value heads

𝐸act number of active experts 𝐸 number of experts in total

Table 2. Hardware specs of consumer-grade and datacenter GPUs.

###### 4090 5090 A100 H100

FP16 (TFLOPS) 330 419 312 989.5 Memory (GB) 24 32 80 80 Interconnect PCIe4 PCIe5 NVLink3 NVLink4 Speed (GB/s) 32 64 300 450

Table 3. Model Configs

Model ℎ 𝑎 𝑘 𝑚 𝐸act 𝐸 Qwen3-1.7B 2048 16 8 6144 1 1 Llama-3.1-8B 4096 32 8 14336 1 1 GPT-OSS-20B 2880 64 8 2880 4 32 Qwen3-32B 5120 64 8 25600 1 1 Qwen3-235B 4096 64 4 1536 8 128

Systems 3 (2021), 269–296.

- [51] PengCheng Yang, Xiaoming Zhang, Wenpeng Zhang, Ming Yang, and Hong Wei. 2022. Group-based interleaved pipeline parallelism for large-scale DNN training. In International Conference on Learning Representations.
- [52] Haoyang Zhang, Yirui Zhou, Yuqi Xue, Yiqi Liu, and Jian Huang.

2023. G10: Enabling an efficient unified gpu memory and storage architecture with smart tensor migrations. In Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture. 395–410.

- [53] Pinxue Zhao, Hailin Zhang, Fangcheng Fu, Xiaonan Nie, Qibin Liu, Fang Yang, Yuanbo Peng, Dian Jiao, Shuaipeng Li, Jinbao Xue, Yangyu Tao, and Bin Cui. 2025. MEMO: Fine-grained Tensor Management For Ultra-long Context LLM Training. Proceedings of the ACM on Management of Data 3, 1 (Feb. 2025), 1–28. doi:10.1145/3709703
- [54] Zan Zong, Li Lin, Leilei Lin, Lijie Wen, and Yu Sun. 2023. Str: Hybrid tensor re-generation to break memory wall for dnn training. IEEE Transactions on Parallel and Distributed Systems 34, 8 (2023), 2403–2418.

### A Summary of notations

Table 1 defines the notation used in the appendix. Table 2 shows the hardware specs of GPUs used in the following analysis. Table 3 shows the configs of models used in the

following analysis. These configs are consistent with their open-source weights.

### B Recomputation Analysis Details

This appendix provides the derivation for the activation-sizerelated conclusions in §2.1 and Figure 2.

#### B.1 Activation Size

We derive an approximate formula for the memory required to store activations in the forward pass of a single GQA [28, 33, 46] transformer layer. We use the same calculation method as in [22], considering only the main contributors to the memory and ignoring small buffers. We also assume that the network and the activations are stored in a 16-bit floating point format, and therefore each element requires 2 bytes for storage.

Each transformer layer consists of an attention and an MLP block connected with two layer norms. Below, we derive the memory required to store activations for each of these elements:

Attention block. It includes a self-attention followed by a linear projection.

- • Query (Q), Key (K), and Value (V ) matrix multiplies: We only need to store their shared input with size 2𝑠𝑏ℎ.
- • FlashAttention [6, 7]: It requires storage of Q, K, V with a total size 2𝑠𝑏ℎ + 4𝑘𝑎𝑠𝑏ℎ.

- • Output projection: It has an input size of 2𝑠𝑏ℎ.

MLP. Dense models compute a single MLP while MOE models compute𝐸act MLPs. The two linear layers store their inputs with sizes 2𝑠𝑏ℎ and 2𝑠𝑏𝑚𝐸act. The SwiGLU non-linearity also needs its input with size 4𝑠𝑏𝑚𝐸act for back-propagation. In total, the MLP block requires 2𝑠𝑏ℎ + 6𝑠𝑏𝑚𝐸act bytes of storage.

Layer norm. Each layer norm stores its input with size 2𝑠𝑏ℎ and therefore in total, we will need 4𝑠𝑏ℎ of storage. Summing the memory required for attention, MLP, and the layer-norms, the memory required to store the activations for a single layer of a transformer network is:

4𝑘 𝑎 )𝑠𝑏ℎ + 6𝑠𝑏𝑚𝐸act bytes. (1)

Activations per layer = (12 +

By substituting the actual configuration of a LLaMA-3.18B [28] model (𝑠 = 16384,𝑏 = 1,ℎ = 4096,𝑚 = 14336,𝑎 = 32,𝑘 = 8,𝐸act = 1, and 32 layers) into Equation 1, training with a single 16k-token sequence generates 68 GB of activations from all layers.

#### B.2 Activation Recompute v.s. Reload Analysis

Consider a single transformer layer. The forward/recompute pass performs four self-attention projections (𝑄, 𝐾,𝑉, output), the attention computation, and three feed-forward networks (FFN) projections (gate, up, down). The total FP16 forward

FLOPS are:5 FLOPSfwd = 4𝑠𝑏ℎ2

𝑘 𝑎

+ 4𝑠𝑏2ℎ attention

+4𝑠𝑏ℎ2

+6𝑠𝑏ℎ𝑚𝐸act

. (2)

Q, out proj.

FFN

K, V proj.

For dense transformer models, it is equivalent to 𝐸act = 1.

By substituting micro-batch size 𝑏 = 4, sequence length 𝑠 = 2048, model configs in Table 3, and hardware specs of 4090 in Table 2 to Equation 2 and 1, we calculate the activation recompute and reload time in Figure 2.

### C Roofline Analysis Details

This appendix provides the full derivation for the roofline analysis summarized in §3.3.

#### C.1 OI of Dense Transformer Layer Forwarding

The data movement consists of uploading the layer’s parameters and the input activation and downloading the output activation. Because PCIe is full-duplex (uploads and downloads proceed simultaneously), the effective transfer time is determined by the larger direction. The upload volume is:

𝑘 𝑎

Bytesfwd upload = 4ℎ2

+ 4ℎ2

+ 6ℎ𝑚

+ 2𝑏𝑠ℎ

. (3)

input act.

Q, out proj.

FFN

K, V proj.

The download volume is 2𝑏𝑠𝐻 bytes for output activation. Since the upload side always exceeds the download side, the operational intensity of the dense transformer layer forward pass is therefore:

4𝑠𝑏ℎ2 + 4𝑠𝑏ℎ2𝑘𝑎 + 4𝑠𝑏2ℎ + 6𝑠𝑏ℎ𝑚 4ℎ2 + 4ℎ2𝑘𝑎 + 6ℎ𝑚 + 2𝑏𝑠ℎ

. (4)

OIfwd =

##### C.2 OI of a Mixture-of-Experts Layer For MoE transformer layers [18], the attention computation is identical, but each token is routed to 𝐸act active experts out of 𝐸 total. Computation scales with 𝐸act while the weight transfer must cover all 𝐸 expert matrices, since all experts are expected to be active when processing multiple tokens. The operational intensity becomes:

4𝑠𝑏ℎ2 + 4𝑠𝑏ℎ2𝑘𝑎 + 4𝑠𝑏2ℎ + 6𝑠𝑏ℎ𝑚𝐸act 4ℎ2 + 4ℎ2𝑘𝑎 + 6ℎ𝑚𝐸 + 2𝑏𝑠ℎ

. (5)

OImoe =

The key distinction is that all𝐸 expert weight sets (6ℎ𝑚𝐸 bytes) must be transferred, but only 𝐸act experts contribute FLOPS in the numerator. This makes MoE layers lower in OI than their dense counterparts of comparable parameter count.

5We count only matrix-multiplication FLOPS, which dominate both compute and transfer time. Each multiply-add accounts for two FLOPS. Elementwise operations (LayerNorm, softmax, activation functions) are negligible compared to matrix operations and therefore omitted.

Operational Intensity vs Batch Size (S=2048)

30000

Qwen3-1.7B

Qwen3-32B

OperationalIntensity(FLOPS/Byte)

Llama-3.1-8B

Qwen3-235B

25000

gpt-oss-20B

20000

H100

15000

4090 A800 5090

10000

5000

0

50 100 150 200 250 Batch Size

Figure 17. Operational intensity vs. batch size for representative dense and MoE models at 𝑠=2048. Horizontal lines mark the ridge-point OI for GPUs in Table 2.

#### C.3 Backward Pass Has Even Higher OI

The analysis above covers the forward pass. During the backward pass with activation recomputation, the total computation consists of three components: the recomputed forward pass, the gradient with respect to activations, and the gradient with respect to weights, each contributing approximately 1× the forward FLOPS. The total backward FLOPS are therefore approximately 3× the forward FLOPS [36].

The data movement in the backward pass, accounting for full-duplex PCIe, is as follows. The upload direction carries parameters, the input activation, and the output gradient. The download direction carries the parameter gradient and the input gradient. The effective transfer is the maximum of the two directions, which is the upload side:

Bytesbwd = Bytesfwd upload + 2𝑏𝑠ℎ. (6)

With 2𝑏𝑠ℎ < Bytesfwd upload, the ratio of backward to forward transfer lies strictly between 1 and 2. Therefore, we have:

3 · FLOPSfwd 2 · Bytesfwd upload

OIbwd >

> OIfwd. (7)

If the forward pass is compute-bound, the backward pass is guaranteed to be compute-bound as well.

#### C.4 OI of Representative Models

Figure 17 plots these equations for representative models at sequence length 𝑠=2048. For dense models, OI exceeds the ridge point at batch size 𝐵 = 8, the smallest batch size for 8-GPU pipeline parallel training. For MoE models, the ridge point is crossed when 𝐵 < 100, well within the typical training regime. Furthermore, the backward pass (with activation recomputation) has ∼3× the forward FLOPS but less than 2× the data movement, so its OI is strictly higher than the forward pass’s.

