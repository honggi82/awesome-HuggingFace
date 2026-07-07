# arXiv:2410.00531v1[cs.DC]1Oct2024

## TPI-LLM: SERVING 70B-SCALE LLMS EFFICIENTLY ON LOW-RESOURCE EDGE DEVICES

Zonghang Li Department of Machine Learning MBZUAI zonghang.li@mbzuai.ac.ae

Mohsen Guizani Department of Machine Learning MBZUAI mohsen.guizani@mbzuai.ac.ae

Wenjiao Feng ∗ School of Info & Comm Engineering UESTC wenjiaofeng@std.uestc.edu.cn

Hongfang Yu School of Info & Comm Engineering UESTC yuhf@uestc.edu.cn

ABSTRACT

Large model inference is shifting from cloud to edge due to concerns about the privacy of user interaction data. However, edge devices often struggle with limited computing power, memory, and bandwidth, requiring collaboration across multiple devices to run and speed up LLM inference. Pipeline parallelism, the mainstream solution, is inefficient for single-user scenarios, while tensor parallelism struggles with frequent communications. In this paper, we argue that tensor parallelism can be more effective than pipeline on low-resource devices, and present a compute- and memory-efficient tensor parallel inference system, named TPILLM, to serve 70B-scale models. TPI-LLM keeps sensitive raw data local in the users’ devices and introduces a sliding window memory scheduler to dynamically manage layer weights during inference, with disk I/O latency overlapped with the computation and communication. This allows larger models to run smoothly on memory-limited devices. We analyze the communication bottleneck and find that link latency, not bandwidth, emerges as the main issue, so a star-based allreduce algorithm is implemented. Through extensive experiments on both emulated and real testbeds, TPI-LLM demonstrated over 80% less time-to-first-token and token latency compared to Accelerate, and over 90% compared to Transformers and Galaxy, while cutting the peak memory footprint of Llama 2-70B by 90%, requiring only 3.1 GB of memory for 70B-scale models.

1 INTRODUCTION

Recently, Large Language Models (LLMs) have been widely deployed in the cloud for inference. User inputs are uploaded to the cloud, where high-performance GPUs are used to compute output sequences, and then sent back to user devices for display. This process poses privacy risks, as user prompts are exposed to network intermediaries and clouds. Therefore, there is an increasing need to shift LLM services to the network edge, such as on laptops, hand phones, tablets, and desktop computers. However, edge devices have very limited memory (4-16 GB) and computing power (often CPU-only). Even with quantization, running a Llama 3.1-70B model still requires at least 40 GB of memory, which far exceeds the capacity of most edge devices. Besides, running Bert-L on one Nano-M device results in a latency that is 120× longer than on one A100 GPU. This gap requires the use of more edge devices to support and speed up LLM inference on the network edge.

While advanced LLM serving systems (Shoeybi et al., 2019; Rasley et al., 2020; Li et al., 2023; Agrawal et al., 2024; Miao et al., 2024) have been designed for high-performance GPU clusters, recent efforts (Zhang et al., 2024; Mei et al., 2024; Borzunov et al., 2024) are adapting these systems to edge environments, by adaptively partitioning model between edge devices and optimizing

∗Zonghang Li and Wenjiao Feng contribute equally.

schedulers to boost token throughput. However, in smart home scenarios like smart speaker, edge LLM systems often handle one user request at a time, making them degrade from pipeline to model parallelism and leaving devices idle most of the time. Thus, tensor parallelism is preferred for better efficiency. For instance, Ye et al. (2024) combine tensor and sequence parallelism to reduce token latency and Wei et al. (2024) introduce block parallelism to restructure Transformer layers.

However, even with 8 devices sharing the load, running full-precision Llama 2-70B still requires 35 GB per device, memory remains a shortage. Solutions like memory block paging (Kwon et al., 2023) and optimized KVCache storage (Jin et al., 2023; Lee et al., 2024) help schedule data between GPUs and CPUs, but unfortunately, GPUs are not available on most edge devices. As a popular alternative, Accelerate (Gugger et al., 2022) can offload model data from a CPU to a disk to run larger models, but its blocking I/O drastically slows inference, with token latency increases to 30 seconds per token on Llama 3.1-8B.

In this work, we analyze why tensor parallelism is more effective than model parallelism on lowresource edge devices and present TPI-LLM, a computing- and memory-efficient tensor parallel inference framework to serve LLM models. Constrained by the high link latency, a star-based allreduce algorithm is implemented. To address the memory shortage, a sliding window memory scheduler is further introduced. We build a prototype of TPI-LLM with 3K LoC and two testbeds using Klonet (Ma et al., 2024) and 4 laptops. Extensive results on Llama 3.1-8B/70B (Dubey et al., 2024), Llama 2-3B/7B/13B/70B (Touvron et al., 2023) and Yi-34B (AI et al., 2024) demonstrate the significant reduction of the memory footprint and faster inference speed compared to Transformers (Wolf et al., 2020), Acclerate (Gugger et al., 2022), and Galaxy (Ye et al., 2024).

We summarize the main contributions of this work as follows:

- • We design a TPI-LLM for edge LLM serving, which keeps prompt privacy in mind to allow edge devices with limited computing power collaborate to deliver faster inference.
- • We find that network bandwidth is no longer an issue. Instead, link latency causes high delays in advanced allreduce algorithms. Thus, a star-based allreduce algorithm is implemented, which greatly outperforms ring- and tree-based methods.
- • We introduce a sliding window memory scheduler, which asynchronously loads and unloads layer weights and overlaps disk I/O latency with computations and communications, enabling the inference of larger models on low-memory devices.
- • We prototype TPI-LLM and show that it reduces time-to-first-token and token latency by over 80% compared to Accelerate and over 90% compared to Transformers and Galaxy. It serves Llama 2-70B with a peak memory footprint of 3.1 GB across 8 low-resource devices.

2 OBSERVATIONS AND MOTIVATIONS

Before presenting our TPI-LLM system, we address two questions that guide our design:

- Q1: On low-resource edge devices, which dominate inference time: computation or communication? Which is more efficient, tensor parallelism or model parallelism?

On the network edge, the balance between computation and communication differs from that in high-performance GPU clusters. To determine whether tensor or model parallelism offers more benefits, it is essential to identify which—computation or communication—takes up more time. For this purpose, we examine the Llama 3.1-8B model on a LAN network with 4 laptops of 8 cores. The network bandwidth between them is 178 Mbps, and the devices implement allreduce communications using a parameter server architecture (Li et al., 2014).

Figures 1a and 1b show the timeline and computing-communication time ratio for model and tensor parallelism during inference. In model parallelism, communication accounts for only 2% of the time, with most spent on computation. However, when one device is computing, others are idle, creating pipeline bubbles and resource waste. In tensor parallelism, communication rises to 70%, but all devices compute simultaneously, and the speed boost outweighs the communication cost, leading to less overall inference time. This makes tensor parallelism the preferred choice.

- Q2: Is tensor parallelism enough for edge LLM serving?

Computation

Communication

Model Parallelism

Tensor Parallelism

0 2 4 6 8 10

Runtime (s)

(a)

Computation Time

120

Communication Time

100

Percentage(%)

80

60

40

20

0

Model Parallel Tensor Parallel Type of Parallelism

(b)

MemoryFootprintPerDevice(GB)

Tensor Parallelism

30

Theoretical

TPI-LLM

25

20

15

10

5

0

1 2 4 Number of Devices

(c)

Figure 1: Comparison of (a,b) tensor and model parallelism in terms of computational and communication time and (c) memory footprint each device with increasing tensor parallel nodes.

Tensor parallelism does reduce memory footprint each device by sharing model parameters across multiple devices, but it doesn’t fully address the memory shortage. Figure 1c shows that even with 4 tensor parallel nodes, memory footprint remains at 12 GB—still too high for most edge devices. This is because memory footprint includes not just model parameters but also intermediate results, key value cache, libraries, etc., causing the actual usage to exceed the theoretical value. Besides, other apps on the device also compete for memory, which worsens the shortage. Thus, even with tensor parallelism, a memory scheduler is still needed to avoid out-of-memory (OOM) issues.

- 3 TPI-LLM FRAMEWORK WITH SLIDING WINDOW MEMORY SCHEDULING

In a typical inference workflow, many users send their prompts to a cloud-based service. These prompts are pooled and scheduled in batches, undergoing dozens of Transformer layers, and converted into probabilities to predict the next token. This process repeats until the generated sequence is finished. While the fundamental workflow on the cloud and edge are similar, key differences arise:

- (a) Keep prompts and generated sequences on users’ device. In a cloud setup, user prompts are sent to remote servers for processing, which result in exposure of private data. Edge LLM serving systems are required to keep prompts and generated sequences in users’ own devices to ensure raw data never get exposed to external unknown environments.
- (b) More single-prompt serving. Current LLM serving systems are typically optimized for batched prompts using pipeline scheduling. However, these optimizations lead to resource underutilization in edge scenarios like smart speakers, where only one prompt is processed at a time.
- (c) Low-resource devices without CUDA support. Edge devices, unlike cloud GPUs, have very limited memory and low computing power. Many of them lack CUDA support or do not have GPUs at all, and they often prioritize full precision to ensure faster computations.

3.1 THE PARALLEL FRAMEWORK DESIGN OF TPI-LLM SYSTEM

The proposed tensor parallel inference system (TPI-LLM) tackles these challenges by using a tensor parallel framework that distributes attention heads across multiple nodes. As depicted in Figure 2, it involves a master node, typically the user’s device that initiates the prompt, and several worker nodes that share the computational load. Their pseudo codes are given in Algorithms 1 and 2.

- Step 1: The master node partitions and distributes model weights. Before inference begins, the master node partitions the pretrained model weights W, such as attention heads and FFN weights, among the worker nodes. Workers with greater computing power and larger memory are allocated more attention heads and FFN weights. This ensures no single device bears the full burden.
- Step 2: The master node initiates prompt and broadcast the input embedding to workers. The inference process starts at the master node, where a user prompt is tokenized into a list of token indices x and then transformed into input embeddings H0 = xWemb. The embedding is then broadcast to all worker nodes Hffn0 = H0 to initiate the tensor parallel workflow.

Q

|Tokenizer| |
|---|---|
| | |

|TokenEmb| |
|---|---|
| | |

|RotaryEmb| |
|---|---|
| | |
| | |

|LayerNorm| |
|---|---|
| | |

|MLP| |
|---|---|
| | |

|Softmax| |
|---|---|
| | |

|RoPE| |
|---|---|
| |K|
| | |

|Attention| |
|---|---|
| | |

RotaryEmb

|LayerNorm| |
|---|---|
| | |

|q| |
|---|---|
| | |

|LayerNorm| |
|---|---|
| | |

|g|
|---|

|SiLU| |
|---|---|
| | |

TokenEmb

LayerNorm

LayerNorm

LayerNorm

Tokenizer

RoPE

Attention

Softmax

Input Emb Position Emb

MLP

|d|
|---|

SiLU

|k| |
|---|---|
| | |

|o| |
|---|---|
| | |

prompt

next token

|u|
|---|

V

|v| |
|---|---|
| | |

Attention Block

FFN Block

Preprocess

Postprocess

KVCache 0

[Figure 1]

+(K0,0 , V0,0)

+(K0,2 , V0,2)

+(K0,1 , V0,1)

user device (master)

Postprocess

|Attention<br><br>Block| |
|---|---|
| | |

|FFNBlock| |
|---|---|
| | |

|Postprocess|
|---|

|Attention<br><br>Block| |
|---|---|
| | |

|Preprocess| |
|---|---|
| | |

|FFNBlock| |
|---|---|
| | |

|Attention<br><br>Block| |
|---|---|
| | |

Preprocess

FFNBlock

FFNBlock

Attention

Attention

Attention

[Figure 2]

prompt

Block

Block

Block

next token

…

Broadcast

Allreduce

Allreduce

Allreduce

Allreduce

Reduce

|FFNBlock| |
|---|---|
| | |

|Attention<br><br>Block| |
|---|---|
| | |

|FFNBlock| |
|---|---|
| | |

|Attention<br><br>Block| |
|---|---|
| | |

|Attention<br><br>Block| |
|---|---|
| | |

FFNBlock

FFNBlock

Attention

Attention

Attention

[Figure 3]

Block

Block

Block

…

other devices i (worker)

+(Ki,1 , Vi,1)

+(Ki,2 , Vi,2)

+(Ki,0 , Vi,0)

[Figure 4]

KVCache i

Transformer Layer Tl Transformer Layer Tl+1 Transformer Layer Tl+2…L

Figure 2: Overview of the TPI-LLM parallel framework.

- Step 3: All nodes perform tensor parallel computing. The tensor parallel computing follows a cycle of four operations: attention computing, allreduce, FFN computing, and allreduce. These operations together constitute a Transformer block. Devices compute attention and FFN with partitioned weights in parallel, reducing the computing delays on low-power devices. In the attention computation phase of the l-th Transformer block, device h processes only a subset of attention heads Qh,l = Hnorml WQh,l,Kh,l = Hnorml WKh,l,V h,l = Hnorml WVh,l, where Hnorml =

norm(Hffnl−1) is the normed hidden state and weight partitions WQh,l,WKh,l,WVh,l are downloaded from the master node in Step 1. Once Qh,l,Kh,l,V h,l are computed, we apply the scaled dotproduct attention to calculate the attention score, and the result is then synchronized across devices:

Hattnl = all reduce(softmax(

Qh,l(Kh,l)⊤

√

d

)V h,l) + Hffnl−1, (1)

where d is the dimension for attention head. Here, attention is computed in parallel across devices, followed by an allreduce to aggregate their hidden states and a shortcut connection. The key-value pair (Kh,l,V h,l) is cached locally on device h to reduce redundant computations. This distributed KVCache partitions the cache across devices, so memory cost is reduced on individual device.

After the attention computation and allreduce, the process continues with the FFN computation:

Hffnl = all reduce(Wdownh,l · (σ(Wgateh,l · Hnorml ) ⊙ (Wuph,l · Hnorml ))) + Hattnl , (2)

where FFN weights Wgateh,l ,Wuph,l,Wdownh,l are also partitioned weights, Hnorml = norm(Hattnl ), σ represents the activation function such as SiLU (Elfwing et al., 2018). Similar to the attention computation stage, the FFN is computed in parallel, followed by an allreduce and a shortcut connection.

- Step 4: The master node reduces tensor parallel results and calculates the next token. After each node h completes its part of computation within the backbone network, the result is sent to the

master node. The summed results HffnL are then passed through a task head Whead and softmax to obtain the probability distribution of the next token z = softmax(HffnL Whead), which is then sampled. Steps 2 to 4 repeat until an EOS token is generated or the length limit is reached.

TPI-LLM provides three benefits: (i) The user prompt {x1,x2,···} and generated sequence {z1 ∼ z1,z2 ∼ z2,···} are processed only on the master node, keeping them hidden from workers. Even if workers reverse-engineer input embeddings H0, they cannot recover the raw prompt x or next token z ∼ z since the weights of input embedding Wemb and task head Whead reside solely on master. (ii) The inference speed is often limited by the computational latency, but in TPI-LLM, it is accelerated via parallel computing. (iii) Unlike other systems that use a mix of communication primitives (reduce & broadcast (Shoeybi et al., 2019), reducescatter & allgather (Ye et al.,

Algorithm 1: Master (with rank 0):

Algorithm 2: Worker (with rank k):

- 1 Split and distribute pretrained weight files to worker nodes;
- 2 Tokenize user prompt into indices;
- 3 Start memory scheduler;
- 4 while generated sequence not finished do

- 5 Preprocess: Convert indices to input and position embeddings, calculate causal mask and cache position;
- 6 Broadcast: Send embeddings, causal mask, and cache position to workers;
- 7 foreach decoder layer l do

- 8 Attention: Execute layernorm,

self-attention, and store (Kl0, Vl0) in KVCache D0;

- 9 Allreduce: Aggregate attention outputs;
- 10 FFN: Execute layernorm and FFN;
- 11 Allreduce: Aggregate FFN outputs;
- 12 end
- 13 Reduce: Sum final outputs with others;
- 14 Postprocess: Execute layernorm, MLP, softmax, and sample next token;
- 15 end

- 1 Download sliced weight files from the master node;
- 2 Start memory scheduler;
- 3 while generated sequence not finished do

- 4 Broadcast: Receive embeddings, causal mask, and cache position from master;
- 5 foreach decoder layer l do

- 6 Attention: Execute layernorm,

self-attention, and store (Klk, Vlk) in KVCache Dk;

- 7 Allreduce: Aggregate attention outputs;
- 8 FFN: Execute layernorm and FFN;
- 9 Allreduce: Aggregate FFN outputs;
- 10 end
- 11 Reduce: Send final output to master;
- 12 end

2024), etc.), TPI-LLM standardizes communications to allreduce. This enhances compatibility with broader communication libraries like PS-LITE (Chen et al., 2015) and NetStorm (Li et al., 2024), leveraging their optimized implementations for edge conditions.

- 3.2 ALLREDUCE LATENCY ANALYSIS

Given the dynamic and heterogeneous nature of edge networks, we tested NetStorm (Li et al., 2024) as the communication backend, but unfortunately, it resulted in high token latency. After further validation, we confirmed that this latency was not due to network bandwidth, but due to link latency.

To analyze the impact of network bandwidth and link latency, we make the following assumption.

Assumption 1. Assume that the edge network adopts a physical topology as shown in Appendix A.7, the network links have the same latency τ, the allreduce algorithm follows a tree-based structure of depth 2 for aggregation, and each device has the same computing power.

The allreduce latency can be expressed as tall reduce = 2L(tdata + tlink + tbarrier + taggr), where L is the number of Transformer layers, tdata is the cumulative data transfer latency, tlink is the cumulative link latency, tbarrier is the cumulative barrier latency during aggregation, and taggr is the cumulative latency for aggregation calculation. Here we ignore taggr as it takes only 0.1 ms and thus negligible compared to other factors.

- Proposition 1. The bottleneck in allreduce is not network bandwidth, but link latency.

##### 32|H|

Proof. The data transfer latency tdata = 2 {i→j}∈P

Bij depends on the size 32|H| of the data being transmitted and the bandwidth Bij of the links in the path Ph, here Ph is an index sequence from device h to the master device. For example, in the case of Llama 2-70B with a hidden size

h

|H| = 8192 and a network bandwidth of 300 Mbps, the data transfer latency is only tdata = 3.4 ms, which is negligible compared to other latencies. In addition, experiment results in Figure 5 show that increasing the network bandwidth does not significantly reduce token latency, further confirming that data transfer and network bandwidth is not the bottleneck.

The link latency tlink, which is often neglected, emerges as the main issue. For example, the path from device h2 to h1 via h8 follows the route h2 → r2 → r9 → r8 → h8 → r8 → r9 → r1 → h1, resulting in a total link latency of 16τ, where τ is the per-hop link latency. To isolate the impact of

link latency, we ran allreduce with only 4 bytes of data, excluding data transfer tdata and barrier latencies tbarrier. The results, shown in Figure 3, demonstrate that the per-link latency τ significantly impacts allreduce latency. This indicates that an inefficient allreduce algorithm, where multiple hops are required (e.g., ring (Ye et al., 2024; Shoeybi et al., 2019) or tree-based (Zhou et al., 2021; Li et al., 2024) algorithms), will further amplifies this impact. For example, with the ring algorithm, allreduce requires 7 communication steps for reducescatter and 7 for allgather, resulting in a total link latency of 56τ, which is 3.5× higher than the tree-based setup.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

250

AllreduceLatency(ms)

200

150

100

50

0

0 2 4 6 8 10 Per-link Latency (ms)

Figure 3: Impact of link latency τ. delays during data aggregation. Given the assumption that all devices have equal computing power and network links have equal latencies, the barrier latency can be approximated as negligible:

The barrier latency, tbarrier, arises from synchronization

32|H| Bij

32|H| Bij

tbarrier = max{

,∀P} − min{

(i→j)∈P

(i→j)∈P

Thus, link latency tlink emerges as the key factor in allreduce latency.

,∀P} ≈ 0. (3)

| |
|---|

- Proposition 2. The star-based allreduce is more effective for TPI-LLM in high-latency networks.

Despite past criticism, the star-based allreduce, where workers push data directly to the master for aggregation and pull the result back (Chen et al., 2015), stands out as the best choice (see Appendix A.1 for a detailed proof). It has minimal hops (8), lowest link latency (8τ), zero intermediate barriers, and avoids the single-point issue due to the small data size (256 KB per device), making it the preferred allreduce algorithm for TPI-LLM.

- 3.3 SLIDING WINDOW MEMORY SCHEDULING

Quantizations like FP16 and INT8 are common for NVIDIA GPUs with CUDA support, but most edge devices lack CUDA and prefer full precision for faster computation due to their general-purpose CPU design. As a result, while tensor parallelism helps distribute memory costs across devices, the memory load remains high. Thus, memory scheduling is still required to manage these loads.

We introduce a memory scheduler, which manages memory by dynamically loading and unloading model weights during inference, ensuring that only the necessary parts are kept in memory (see Appendix A.2 for potential use). The memory scheduler operates on a daemon thread to asynchronously handle memory operations. To maintain the peak memory footprint, it uses a sliding window and preloads weights for upcoming layers while unloading those that have been processed.

As mentioned in Section 3.1, each Transformer layer is divided into attention computing, allreduce, FFN computing, and allreduce. For simplicity, in Figure 4, we assume the delays for these stages and weight loading to be equal. In each time slot, the memory scheduler asynchronously loads weights for either an attention or FFN block. By overlapping weight loading with ongoing computations and communications, it hides the I/O latency associated with loading weights from disk. For example, in Figure 4, the memory scheduler loads one more block during each allreduce until the sliding window reaches its size. As computations and communications proceed, we ensure weights are always ready when needed, allowing for seamless inference without computational stalls.

Next, we provide the conditions for this mechanism to reach a steady state, under which all required weights are loaded before computation starts.

- Proposition 3 (Loose Steady Condition). The memory scheduler reaches a steady state when the following condition is met:

tattn + tffn + 2tall reduce ≥ τffn + τattn, (4) and one of the following conditions is met:

l · tattn + (l − 1) · tffn + (2l − 1) · tall reduce ≥ l · τffn + (l − 1) · τattn, ∀l ∈ {1,··· ,L}, (5) (l − 1) · tattn + l · tffn + (2l − 1) · tall reduce ≥ (l − 1) · τffn + l · τattn, ∀l ∈ {1,··· ,L}, (6)

|1| |2| |3| |4| |5| |6|
|---|---|---|---|---|---|---|---|---|---|---|

|1| |2| |3| |4| |5| |6|
|---|---|---|---|---|---|---|---|---|---|---|

|1| |2| |3| |4| |5| |6|
|---|---|---|---|---|---|---|---|---|---|---|

- Time 0: Memory:

|1| |2| |3| |4| |5| |6|
|---|---|---|---|---|---|---|---|---|---|---|

- Time 1: Memory:

|1| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|

| |2| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|

|1| |2| |3| |4| |5| |6|
|---|---|---|---|---|---|---|---|---|---|---|

- Time 2: Memory:

- Time 3: Memory:

| | |3|4| | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|

|1| |2| |3| |4| |5| |6|
|---|---|---|---|---|---|---|---|---|---|---|

- Time 4: Memory:

| | |3|4|5| | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|

|1| |2| |3| |4| |5| |6|
|---|---|---|---|---|---|---|---|---|---|---|

- Time 5: Memory:

- Time 6: Memory:

| | | |4|5|6|7| | | | |
|---|---|---|---|---|---|---|---|---|---|---|

|1| |2| |3| |4| |5| |6|
|---|---|---|---|---|---|---|---|---|---|---|

- Time 7: Memory:

| | | | |5|6|7|8| | | |
|---|---|---|---|---|---|---|---|---|---|---|

|1| |2| |3| |4| |5| |6|
|---|---|---|---|---|---|---|---|---|---|---|

- Time 8: Memory:

| | | |4|5|6| | | | | |
|---|---|---|---|---|---|---|---|---|---|---|

| |2|3| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|

| | | | |5|6|7|8| | | |
|---|---|---|---|---|---|---|---|---|---|---|

Time

- Figure 4: An illustration of the sliding window memory scheduling. Blue blocks indicate the blocks currently executed, with numbered blocks for attention or FFN computing and unnumbered blocks for allreduce communication. Green blocks indicate loaded model weights. The dashed box represents the sliding window, with size 4 in this case.

where tattn and tffn are times required for attention and FFN computation, tall reduce is the allreduce latency, τffn and τattn are times required to load attention and FFN weights, and L is the number of Transformer layers.

This condition is loose but a bit hard to assess, so we present a tighter, more intuitive condition.

### Proposition 4 (Tight Steady Condition). tattn + tall reduce ≥ τffn and tffn + tall reduce ≥ τattn.

The proofs can be found in Appendices A.3 and A.4. This conclusion is straightforward. If the previous block’s computation and allreduce time cover the current block’s weight loading time, the memory scheduler can fully hide the disk I/O latency. As an example, in Section 4.4, we use 4 laptops with Llama 2-7B, setting pi = 0.25 and w = 4. We measured tattn = 11 ms, tffn = 17 ms, tall reduce = 14 ms, τattn = 18 ms, and τffn = 30 ms. While the tight steady condition is not met, the loose steady condition is met, allowing the memory scheduler to achieve steady state.

#### Proposition 5 (Peak Memory Footprint). If the memory scheduler reaches a steady state, the peak memory footprint of the master and worker can be expressed as

 

hv + h, if w = 1 2hv + h, if w = 2 2hv + h + w−2 2 2(1 + ab)h2pi + h + w−2 1 (3hspi + h), if w ≥ 3

(7)

Mmaster = γ ×



w 2

b a

w + 1 2

)h2pi + h) +

(3hspi + h) , (8)

Mworker = γ ×

(2(1 +

where h is the hidden size, v is the vocabulary size, a is the number of attention heads, b is the number of key-value heads, s is the intermediate size, pi is the proportion of parameters handled by device i, w is the memory window size, and γ is a memory scaling factor.

- The proof can be found in Appendix A.5. However, if a slow disk I/O disrupts the steady state, the memory scheduler will retain some FFN blocks in memory to reduce disk access frequency.

Proposition 6 (Loose Steady Condition with Block Retention). Let the memory scheduler retain one FFN block in memory every T FFN blocks, the condition to reach a steady state is then

l · tattn + l · tffn + 2l · tall reduce ≥ (l −

l T

) · τffn + l · τattn, (9)

l · tattn + (l − 1) · tffn + (2l − 1) · tall reduce ≥ (l −

l T

) · τffn + (l − 1) · τattn. (10)

- The proof can be found in Appendix A.6. By setting an appropriate T, idle memory can help the scheduler reach a steady state, thus achieving a tradeoff between memory use and inference speed.

4 EXPERIMENTS

Prototype and Testbed. We implemented the prototype of TPI-LLM1 with 3K LoC using PyTorch and Transformers to provide flexible support for various sizes and versions of pretrained LLMs. Our testbed, illustrated in Appendix A.7, was built upon Klonet (Ma et al., 2024) to create an edge network environment, emulating realistic conditions with configurable properties like network topology, bandwidth, and latency. By default, 8 edge devices were emulated on 2 Intel Xeon Gold 5220R CPUs, each limited to 8 logical cores, 8 GB of memory, and 4 GB of swap. Network bandwidth between devices was set to 300 Mbps with a 1 ms latency.

Models. The inference speed of TPI-LLM is significantly affected by the model architecture. Deeper layers, more parameters, larger hidden sizes, and more attention heads increase the computational latency. Additionally, deeper layers result in more allreduce communications, and a larger hidden size leads to greater traffic. We tested with various models of different sizes, including Llama 23B/7B/13B/70B, Llama 3-8B/70B, and Yi-34B. See Appendix A.8 for their configuration details.

- 4.1 OVERVIEW OF TPI-LLM PERFORMANCE

Fit 70B LLMs into edge devices and run in high efficiency. We tested the performance of TPILLM with a focus on 3 key metrics: time-to-first-token (TTFT), token latency, and peak memory footprint per device. The memory window size is set to 2 by default. As shown in Table 1, without the memory scheduler, the full weights are loaded into the memory at once. Despite that these weights have been distributed across multiple devices, the memory is still insufficient for larger models like Yi-34B and Llama 2/3/3.1-70B. Instead, enabling our memory scheduler significantly reduces the peak memory footprint, allowing larger models to run efficiently. For example, the Llama 2-70B model requires just 3.1 GB of memory per device, and the Llama 3.1-70B model fits within device limits. The results are summarized in Table 1.

- Table 1: The TTFT, token latency, and peak memory footprint per device of TPI-LLM.

|Model (FP32)|Memory Scheduler Disabled TTFT Latency Memory<br><br>|Memory Scheduler Enabled TTFT Latency Memory|
|---|---|---|
| | | |

|Llama 2-3B Llama 2-7B Llama 2-13B Llama 2-70B|2.3 s 1.0 s/token 2.8 GB<br><br>3.1 s 1.2 s/token 4.5 GB<br><br><br>5.1 s 1.9 s/token 8.1 GB OOM OOM 34.9 GB<br><br>|2.0 s 1.9 s/token 1.4 GB<br><br>3.0 s 2.6 s/token 1.7 GB<br><br><br>5.8 s 2.9 s/token 2.1 GB 29.4 s 26.1 s/token 3.1 GB<br><br>|
|---|---|---|
|Llama 3.1-8B Llama 3.1-70B<br><br>|4.5 s 1.5 s/token 8.5 GB OOM OOM 42.3 GB<br><br>|4.5 s 4.3 s/token 5.4 GB 32.9 s 29.9 s/token 11.3 GB|
|Yi-34B|OOM OOM 20.4 GB<br><br>|15.7 s 13.7 s/token 4.9 GB|

- Table 2: Peak memory footprint per device with the memory window size set to 2.

| |Memory Scheduler Disabled (GB)<br><br>|Memory Scheduler Enabled (GB)|
|---|---|---|
|Model (FP32)<br><br>|N = 2 N = 4 N = 6 N = 8<br><br>|N = 2 N = 4 N = 6 N = 8|

|Llama 2-3B Llama 2-7B Llama 2-13B Llama 2-70B<br><br>|7.3 4.3 3.2 2.8 13.7 7.7 5.5 4.5 25.7 13.9 9.8 8.1 130 66.6 46.6 34.9<br><br>|1.5 1.4 1.4 1.4<br>2.0 1.8 1.7 1.7<br><br><br>2.3 2.2 2.2 2.1<br>3.7 3.3 3.3 3.1<br>|
|---|---|---|
|Llama 3.1-8B Llama 3.1-70B<br><br>|18.4 11.8 9.4 8.5 137.7 74.0 51.1 42.3<br><br>|5.9 5.6 5.5 5.4 10.8 10.5 11.4 11.3|
|Yi-34B<br><br>|67 36.4 23.9 20.4|5.0 5.0 5.0 4.9|

No need for dozens of devices, one or two are enough to run 70B models. We used 8 devices by default, but can fewer devices run 70B-scale models? Table 2 gives detailed peak memory footprints with varying number of devices. Without the memory scheduler, full weights are loaded onto the devices, and with fewer devices, the memory load increases. For instance, using only 2 devices limits users to smaller models, like those between 3B and 7B. However, with the memory scheduler enabled, only a few layers’ weights are loaded and distributed across devices. This allows even

1Open available at: https://anonymous.4open.science/r/tpi-llm.

50

50

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

50

TokenLatency(s/token)

TokenLatency(s/token)

TokenLatency(s/token)

40

40

40

30

30

30

20

20

20

10

10

10

0

0

0

4 6 8 Number of Devices

4 6 8 CPU Cores Per Device

300M 500M 1G Bandwidth

#### Figure 5: Token latency over varying number of devices, CPU cores, and network bandwidth on Llama 2-70B.

Llama 2-3B Llama 2-7B

Llama 2-13B Llama 2-70B

| |Llama 2-3B Llama 2-13B<br><br>| | | |
|---|---|---|---|---|
| |Llama 2|-7B<br><br>|Llama 2-70B| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
|OOM| |OOM|OOM| |

50

50

TokenLatency(s/token)

40

40

30

30

TTFT(/s)

20

20

10

10

OOM OOM OOM

0

0

Standalone MP Galaxy TPI-LLM

Standalone MP Galaxy TPI-LLM

Figure 6: Comparison of TPI-LLM with three benchmarks.

larger models, such as 70B, to run smoothly on just 2 devices. Appendix A.9 shows the case with a memory window size of 4, which requires slightly more memory but faster speed. The peak memory footprint in TPI-LLM is primarily determined by the product of vocabulary size and hidden size, which is detailed in equation (7) and can be further reduced in our future work.

- 4.2 SCALING OVER VARYING EDGE CONDITIONS

Computation remains the bottleneck, not network bandwidth. In this experiment, we examined the token latency of TPI-LLM under different edge conditions, the results are shown in Figure 5. As expected, increasing the number of devices reduces the computing load on each device, significantly lowering token latency, and more CPU cores also contribute to a reduced latency. Instead, a limited network bandwidth was no longer a bottleneck, boosting it from 300 Mbps to 1 Gbps had little effect on latency due to the tiny data size (only 256 KB) during each allreduce. Thus, the main bottleneck remains in the computation, which our future work should focus on.

- 4.3 COMPARISON WITH BENCHMARKS

We compared the TPI-LLM with 3 benchmarks: (a) Standalone: LLM inference is executed only on a single edge device using Transformers (Wolf et al., 2020). (b) Model Parallelism (MP): Since only one user is served at a time, the pipeline parallelism (Zhang et al., 2024; Mei et al., 2024; Borzunov et al., 2024) degrades to the model parallelism, where different layer sequences are distributed across multiple devices. Each device computes its layers and passes the result to the next device until the entire inference is complete. (c) Galaxy (Ye et al., 2024) combines tensor and sequence parallelism and overlaps communication and computation to accelerate inference. They all run in FP32 mode.

Run larger models with lower latency and memory usage. As shown in Figure 6, a limited memory on a single device makes it challenging to run even 3B models in a standalone mode. MP addresses this by the collaboration of 8 devices, allowing models up to 13B, but suffers from high latency due to pipeline bubbles. Galaxy tries to reduce such latency by combining tensor and sequence parallelism. However, in Section 3.2, we concluded that the network bandwidth was no longer the issue, and the real problem is the link latency. Galaxy’s use of a ring algorithm for reducescatter and allgather forces each link to be traversed at least 14 times. This causes high link

Table 3: Comparison of Transformers, Accelerate, Transformers+MS, and TPI-LLM on 4 laptops.

|Model (FP32)<br><br>|Transformers TTFT Latency|Accelerate TTFT Latency<br><br>|Transformers + MS TTFT Latency|TPI-LLM TTFT Latency|
|---|---|---|---|---|
| |(s) (s/token)|(s) (s/token)<br><br>|(s) (s/token)|(s) (s/token)|

|Llama 2-3B Llama 2-7B Llama 2-13B<br><br>|61 30<br><br>115 56 OOM OOM|24 16 30 26 OOM OOM<br><br>|4 3 13 8<br><br>22 18<br><br>|2.5 2<br><br>6 5 10 9|
|---|---|---|---|---|
|Llama 3.1-8B|133 65<br><br>|37 31|20 12<br><br>|11 8|
|Yi-34B<br><br>|OOM OOM|OOM OOM|185 55<br><br>|33 29|

latency and outweighs the benefits of parallel computing, ultimately resulting in a higher token latency than MP. In contrast, TPI-LLM adopts a star-based allreduce algorithm, minimizing hops and cumulative link latency. Combined with the blocking-free memory scheduler, TPI-LLM delivers significantly lower token latency and memory footprint, even with larger 70B models.

- 4.4 REAL CASE STUDY

In this study, we used 4 laptops with different CPU architectures and memory capacities, connected via a local Wi-Fi router. The testbed and configurations are detailed in Appendix A.10. Macbook Pro was used by default. Due to the lack of CUDA, all computations were performed in full precision. As shown in Table 3, Transformers loaded the entire model into the CPU memory, and when memory was insufficient, the operating system offloaded data to the swap. This frequent swap exchange significantly increased TTFT and token latency, even for smaller 3B models. As the model size grows, the swap space overflowed, finally leading to OOM errors. As a more efficient alternative, Accelerate (Gugger et al., 2022) instantly loads layer weights only when required for the computation and reduces unnecessary data I/O. While it speeds up inference, due to implementation flaws on disk offloading, it still requires loading full weights before splitting and offloading them to disk. This results in OOM errors when the model size reaches 13B.

TPI-LLM stands out in TTFT, token latency, and model size. Our memory scheduler (Transformers+MS) outperforms Transformers and Accelerate in both TTFT and token latency across all model sizes. This is because our memory scheduler employs a sliding window mechanism, where a daemon thread asynchronously preloads the weights needed for upcoming computations. By overlapping data I/O with computations and communications, the scheduler avoids delays caused by disk I/O blocks, ensuring smoother and faster inference. To further speed up inference, we integrate the computing power of 4 laptops to serve TPI-LLM. By distributing the computational load across

- 4 laptops, the reduction in computing time far exceeds communication delays, so both TTFT and token latency are further reduced. The results from using 3 laptops are shown in Appendix A.11, indicating a slightly higher latency due to reduced parallelism.
- 5 CONCLUSION

In this paper, we concluded that tensor parallelism can be more effective than pipeline parallelism on low-resource devices, and presented a compute- and memory-efficient tensor parallel inference system, named TPI-LLM, to serve 70B-scale LLMs. TPI-LLM is designed with user prompt and generated sequence privacy in mind, by keeping sensitive raw data local in the users’ devices. It leverages a sliding window memory scheduler to dynamically manage layer weights during inference with disk I/O latency overlapped by onging computations and communications, allowing larger models to run smoothly on devices with very limited memory. Our analysis showed that link latency, not bandwidth, emerges as the main issue, so TPI-LLM implements a star-based allreduce algorithm, rather than the commonly used ring- and tree-based algorithms. Through extensive experiments on emulated and real testbeds, TPI-LLM demonstrated significantly lower TTFT, token latency, and peak memory footprint compared to Transformers, Accelerate, Galaxy, and enabled serving largerscale LLMs such as Yi-34B and Llama 2/3/3.1-70B on low-memory devices.

REPRODUCIBILITY

We have made efforts to ensure reproducibility by providing the source code at https:// anonymous.4open.science/r/tpi-llm, with a detailed README for guidance included. To ease the use, a prebuilt Docker image is also provided. Key experimental setups are given in Section 4 of the paper.

REFERENCES

Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. Taming Throughput-Latency tradeoff in LLM inference with Sarathi-Serve. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pp. 117–134, Santa Clara, CA, July 2024. USENIX Association. ISBN 978-1-939133-40-3. URL https://www.usenix.org/conference/osdi24/ presentation/agrawal.

01. AI, :, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. Yi: Open foundation models by 01.ai, 2024.

Alexander Borzunov, Max Ryabinin, Artem Chumachenko, Dmitry Baranchuk, Tim Dettmers, Younes Belkada, Pavel Samygin, and Colin A Raffel. Distributed inference and fine-tuning of large language models over the internet. Advances in Neural Information Processing Systems, 36, 2024.

Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, and Zheng Zhang. Mxnet: A flexible and efficient machine learning library for heterogeneous distributed systems. arXiv preprint arXiv:1512.01274, 2015.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural networks, 107:3–11, 2018.

Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, and Benjamin Bossan. Accelerate: Training and inference at scale made simple, efficient and adaptable. https://github.com/huggingface/accelerate, 2022.

Yunho Jin, Chun-Feng Wu, David Brooks, and Gu-Yeon Wei. s3: Increasing gpu utilization during generative inference for higher throughput. Advances in Neural Information Processing Systems, 36:18015–18027, 2023.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pp. 611–626, 2023.

Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. Infinigen: Efficient generative inference of large language models with dynamic kv cache management. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pp. 155–172, 2024.

Mu Li, David G Andersen, Jun Woo Park, Alexander J Smola, Amr Ahmed, Vanja Josifovski, James Long, Eugene J Shekita, and Bor-Yiing Su. Scaling distributed machine learning with the parameter server. In 11th USENIX Symposium on operating systems design and implementation (OSDI 14), pp. 583–598, 2014.

Zhuohan Li, Lianmin Zheng, Yinmin Zhong, Vincent Liu, Ying Sheng, Xin Jin, Yanping Huang, Zhifeng Chen, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. AlpaServe: Statistical multiplexing with model parallelism for deep learning serving. In 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23), pp. 663–679, Boston, MA, July 2023. USENIX Association. ISBN 978-1-939133-34-2. URL https://www.usenix.org/conference/ osdi23/presentation/li-zhouhan.

Zonghang Li, Wenjiao Feng, Weibo Cai, Hongfang Yu, Long Luo, Gang Sun, Hongyang Du, and Dusit Niyato. Accelerating geo-distributed machine learning with network-aware adaptive tree and auxiliary route. IEEE/ACM Transactions on Networking, 2024.

Tie Ma, Long Luo, Hongfang Yu, Xi Chen, Jingzhao Xie, Chongxi Ma, Yunhan Xie, Gang Sun, Tianxi Wei, Li Chen, et al. Klonet: an easy-to-use and scalable platform for computer networks education. In 21st USENIX Symposium on Networked Systems Design and Implementation, pp. 2025–2046, 2024.

Yixuan Mei, Yonghao Zhuang, Xupeng Miao, Juncheng Yang, Zhihao Jia, and Rashmi Vinayak. Helix: Distributed serving of large language models via max-flow on heterogeneous gpus. arXiv preprint arXiv:2406.01566, 2024.

Xupeng Miao, Chunan Shi, Jiangfei Duan, Xiaoli Xi, Dahua Lin, Bin Cui, and Zhihao Jia. Spotserve: Serving generative large language models on preemptible instances. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pp. 1112–1127, 2024.

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 3505–3506, 2020.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Yuanxin Wei, Shengyuan Ye, Jiazhi Jiang, Xu Chen, Dan Huang, Jiangsu Du, and Yutong Lu. Communication-efficient model parallelism for distributed in-situ transformer inference. In 2024 Design, Automation & Test in Europe Conference & Exhibition, pp. 1–6. IEEE, 2024.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, et al. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations, pp. 38–45, 2020.

Shengyuan Ye, Jiangsu Du, Liekang Zeng, Wenzhong Ou, Xiaowen Chu, Yutong Lu, and Xu Chen. Galaxy: A resource-efficient collaborative edge ai system for in-situ transformer inference. arXiv preprint arXiv:2405.17245, 2024.

Mingjin Zhang, Jiannong Cao, Xiaoming Shen, and Zeyang Cui. Edgeshard: Efficient llm inference via collaborative edge computing. arXiv preprint arXiv:2405.14371, 2024.

Huaman Zhou, Weibo Cai, Zonghang Li, Hongfang Yu, Ling Liu, Long Luo, and Gang Sun. Tsengine: Enable efficient communication overlay in distributed machine learning in wans. IEEE Transactions on Network and Service Management, 18(4):4846–4859, 2021.

A APPENDIX

- A.1 PROOF OF PROPOSITION 2

In conventional data parallel systems, each device sends several gigabytes of data, putting significant pressure on network bandwidth. This makes data transfer latency a major concern, while link latency becomes negligible. Then, tree and ring-based algorithms are introduced to optimize the data transfer. However, they do not apply to our case. In TPI-LLM, each device only sends a small amount of data, usually just tens of kilobytes. This tiny data size does not strain the network, so data transfer latency is minimal. Instead, in edge networks where wireless communication causes higher transmission delays, link latency becomes more significant than data transfer latency. As a result, the commonly used tree and ring-based allreduce algorithms are less effective.

Let us consider 1 master and 2 workers connected via a router. In Figure 7, we compare the traffic models of star, tree, and ring-based algorithms. In star-based allreduce, worker 1 sends data directly to the master via the router, and the allreduce latency (includes reduce and broadcast) is tstar = 2(tdata + tlink) + tbarrier + taggr. In this model, the router only forwards data packets.

worker 1 router master

Barrier & Aggr

star-based allreduce

Barrier & Aggr

Barrier & Aggr

tree-based allreduce ring-based allreduce

Barrier & Aggr

Barrier & Aggr

Barrier & Aggr

Barrier & Aggr

Barrier Barrier

worker 1 worker 2 master worker 1 worker 2 master

ttdatalink

barrieraggrt+t

Figure 7: Comparison of traffic models for star, tree, and ring-based allreduce algorithms.

In tree-based allreduce, data from worker 1 must first go through worker 2 before reaching the master, so there are 2 hops involved. In this process, worker 1 sends its data to worker 2, which aggregates it and forwards the result to the master. Once the global aggregation is complete, the final result is broadcast back to all workers. The total time for this process is ttree = 3tdata + 4tlink + 2tbarrier + 2taggr.

In ring-based allreduce, each device communicates directly with its neighbors in a ring topology. Data is divided and sent in a sequence around the ring, with each device receiving, aggregating, and passing the data to the next device. Unlike star or tree-based methods, there is no central device, and data flows continuously between the devices. The total time for the ring-based allreduce is

tring = 34tdata + 4tlink + 3tbarrier + 32taggr.

Assume that all devices are homogeneous, i.e., tbarrier ≈ 0, and tdata ≈ 0,taggr ≈ 0 because the data size is very small. Then we have latencies simplified as follows:

tstar = 2tlink < ttree = tring = 4tlink. (11) Thus, the star-based allreduce is the most efficient method because it minimizes link latency.

- A.2 A SIMPLE-TO-USE MEMORY SCHEDULER

In our implementation, a context manager is used to ensure that the required block weights are loaded correctly and unload the used weights to free up memory for subsequent blocks. This simplifies the deployment of large-scale LLMs on low-memory edge devices, requiring just one additional line of code:

- 1

|with memory_manager.wait_and_release(f"self_attn.0"):<br><br>hidden_states = self_attn(hidden_states)<br><br>|
|---|

- 2

- A.3 PROOF OF PROPOSITION 3 We start with the first attention block and end with the final FFN block.

- Time slot 1 (attention computation): In this initialization step, Wattn1 must be loaded before computing the first attention block, taking τattn + tattn. During the computation time tattn, the next FFN weights, Wffn1 , are loading in parallel.
- Time slot 2 (allreduce): The attention block is followed by allreduce communication, which takes tall reduce, with the next FFN weights, Wffn1 , loading in parallel.

- Time slot 3 (FFN computation): By this time, the FFN weights Wffn1 should be fully loaded. If not, the computation must wait for loading to complete. Let t′ = tattn + tall reduce − τffn, if t′ ≥ 0, no blocking occurs; otherwise, the computation is delayed by |t′|. Once loaded, compute the FFN block in tffn.

During this time slot, the waiting, computation of the current FFN block and the weight loading of the next attention block occur simultaneously. By the time the current FFN block finishes, the next attention block’s weights Wattn2 have been loading for max{0,tattn + tall reduce − τffn} + tffn.

- Time slot 4 (allreduce): The FFN block is followed by allreduce communication, which takes tall reduce, with the next attention weights, Wattn2 , loading in parallel.

- Time slot 5 (attention computation): Ensure that the attention weights Wattn2 are fully loaded. Let t′ = max{0,tattn + tall reduce − τffn} + tffn + tall reduce − τattn. If t′ ≥ 0, the computation proceeds without blocking. Then, Wattn2 is computed in tattn, and the next FFN weights Wffn2 have been loading for max{0,max{0,tattn + tall reduce − τffn} + tffn + tall reduce − τattn} + tattn.

- Time slot 6 (allreduce): The allreduce communication takes tall reduce, while the next FFN weights Wffn2 are loading in parallel.

- Time slot 7 (FFN computation): Ensure that the FFN weights Wffn2 are fully loaded. Let t′ = max{0,max{0,tattn + tall reduce − τffn} + tffn + tall reduce − τattn} + tattn + tall reduce − τffn. If t′ ≥ 0, the computation proceeds without blocking. This process repeats, until the generation task is finished.

For the system to reach a steady state where computation is not blocked by weight loading at any time, the following conditions must hold.

- Case 1: tattn + tall reduce − τffn ≥ 0. Time slot 3 (l = 1): tattn + tall reduce − τffn ≥ 0, (12) Time slot 5 (l = 1): tattn + tffn + 2tall reduce − τffn − τattn ≥ 0, (13) Time slot 7 (l = 2): 2tattn + tffn + 3tall reduce − 2τffn − τattn ≥ 0, (14) Time slot 9 (l = 2): 2tattn + 2tffn + 4tall reduce − 2τffn − 2τattn ≥ 0. (15)

We repeat these conditions and derive the following patterns.

tattn + tffn + 2tall reduce ≥ τffn + τattn, (16) l · tattn + (l − 1) · tffn + (2l − 1) · tall reduce ≥ l · τffn + (l − 1) · τattn. (17)

- Case 2: tattn + tall reduce − τffn < 0.

Time slot 3 (l = 1): tattn + tall reduce − τffn < 0, (18) Time slot 5 (l = 1): tffn + tall reduce − τattn ≥ 0, (19) Time slot 7 (l = 2): tattn + tffn + 2tall reduce − τattn − τffn ≥ 0, (20) Time slot 9 (l = 2): tattn + 2tffn + 3tall reduce − 2τattn − τffn ≥ 0, (21)

Time slot 11 (l = 3): 2tattn + 2tffn + 4tall reduce − 2τattn − 2τffn ≥ 0. (22)

Similarly, repeat these conditions and derive the following patterns.

tattn + tffn + 2tall reduce ≥ τffn + τattn, (23)

(l − 1) · tattn + l · tffn + (2l − 1) · tall reduce ≥ (l − 1) · τffn + l · τattn. (24) Thus, the proposition is proved.

- A.4 PROOF OF PROPOSITION 4 Let α = l · tattn + (l − 1) · tffn + (2l − 1) · tall reduce − l · τffn − (l − 1) · τattn > 0, and we derive the following inequality from inequality (16):

l · tattn + l · tffn + 2l · tall reduce − l · τffn − l · τattn > 0. (25)

By substituting α into this inequality, we have α + tffn + tall reduce − τattn > 0. Let α > 0 > τattn − tffn − tall reduce, we obtain the first condition:

tffn + tall reduce > τattn. (26)

Let β = tffn+tall reduce−τattn > 0, and substitute β into inequality (16), then we have β+tattn+ tall reduce − τffn > 0. Let β > 0 > τffn − tattn − tall reduce, we obtain the second condition:

tattn + tall reduce > τffn. (27) Thus, the proposition is proved.

- A.5 PROOF OF PROPOSITION 5

In this section, we analyze the peak memory footprint on both the master and worker nodes to estimate the largest model size that our memory scheduler can handle.

Let us use the Llama model as an example, assume the vocabulary size is v, hidden size is h, number of attention heads is a, number of key-value heads is b, and intermediate size is s. Let p = [p1,p2,··· ,pn] be a vector representing the proportion of parameters handled by n devices, and w be the window size of the memory scheduler. Following the block definition in Figure 2, the parameter counts for each block are detailed in Table 4:

Table 4: Parameter counts for the main blocks (e.g., n = 4, pi = 0.25, Llama 2-7B).

Block Parameters Block Size Preprocess hv 500 MB Attention 2(a + b)h2pi/a + h 64 MB

FFN 3hspi + h 129 MB Postprocess hv + h 500 MB

The memory footprint is affected by parameters, activation storage, temporary tensors, memory management, and caching, making precise quantification challenging. To estimate peak memory, we apply an empirical rule: multiply the parameter size by a scaling factor γ.

Token i Token i + 1

|Attn|FFN|Post|
|---|---|---|

|Attn|FFN|Post| |Pre|Attn|FFN|
|---|---|---|---|---|---|---|

|Pre|Attn|FFN|
|---|---|---|

- w = 1

|Attn|FFN|Post| |Pre|Attn|FFN|
|---|---|---|---|---|---|---|

- w = 2

|Attn|FFN|Post| |Pre|Attn|FFN|
|---|---|---|---|---|---|---|

Token i Token i + 1

- w = 3

- w = 4

|Attn|FFN|Post||Pre|Attn|FFN|
|---|---|---|
|
|---|---|---|---|

- w = 5

||Attn|FFN|Post|
|---|---|---|
<br><br>|Pre|Attn|FFN|
|---|---|---|
|
|---|

- w = 6

Figure 8: Illustration of the memory window at the peak memory footprint.

From the memory window at the peak memory footprint shown in Figure 8, we can derive the following equations.

 

hv + h, if w = 1 2hv + h, if w = 2 2hv + h + w−2 2 2(1 + ab)h2pi + h + w−2 1 (3hspi + h), if w ≥ 3

Mmaster = γ ×



For any worker node, the memory footprint does not include the preprocess and postprocess blocks. Therefore, the peak memory footprint Mworker can be expressed as:

w 2

b a

w + 1 2

)h2pi + h) +

Mworker = γ ×

(2(1 +

(3hspi + h) . Thus, the proposition is proved.

- A.6 PROOF OF PROPOSITION 6

When the memory scheduler reaches a steady state, the overlap between computation, communication, and disk I/O is optimized, ensuring that weights are always pre-loaded before they are needed for computations. However, if disk I/O becomes a bottleneck and disrupts the steady state (e.g., due to high disk latency), the scheduler must adapt by selectively retaining certain blocks in memory to reduce disk access frequency.

In our preliminary experiments, we measured tattn = 11 ms, tffn = 17 ms, τattn = 18 ms, τffn = 30 ms, and observed that FFN blocks generally exhibit higher computation and weight loading latency. By retaining some FFN blocks in memory, we can reduce the need to reload large weights.

Let the memory scheduler retain one FFN block in memory every T FFN blocks, and

1, if l = 1 + kT and k ∈ Z≥ 0, 0, otherwise.

I{l=1+kT} =

Similar to the analysis in Appendix A.3, we have Time slot 3 (l = 1): tattn + tall reduce − (1 − I{l=1+kT})τffn ≥ 0, Time slot 5 (l = 1): tattn + tffn + 2tall reduce − (1 − I{l=1+kT})τffn − τattn ≥ 0,

2

Time slot 7 (l = 2): 2tattn + tffn + 3tall reduce −

(1 − I{i=1+kT})τffn − τattn ≥ 0,

i=1

2

Time slot 9 (l = 2): 2tattn + 2tffn + 4tall reduce −

(1 − I{i=1+kT})τffn − 2τattn ≥ 0,

i=1

3

Time slot 11 (l = 3): 3tattn + 2tffn + 5tall reduce −

(1 − I{i=1+kT})τffn − 2τattn ≥ 0. By repeating these conditions, we derive the following patterns:

i=1

l

(1 − I{i=1+kT})τffn − l · τattn ≥ 0,

l · tattn + l · tffn + 2l · tall reduce −

i=1

l

(1 − I{i=1+kT})τffn − (l − 1) · τattn ≥ 0.

l · tattn + (l − 1) · tffn + (2l − 1) · tall reduce −

i=1

Since li=1 I{i=1+kT} = T l , we have

l T

l · tattn + l · tffn + 2l · tall reduce ≥ (l −

) · τffn + l · τattn,

l T

) · τffn + (l − 1) · τattn. Thus, the proposition is proved.

l · tattn + (l − 1) · tffn + (2l − 1) · tall reduce ≥ (l −

- A.7 KLONET TESTBED

One of our testbed, as shown in Figure 9, was built upon Klonet (Ma et al., 2024) to create an edge network environment. Klonet is a network emulation platform designed to support the development and testing of new network protocols and applications in a realistic environment. It can emulate various network scenarios, such as wireless, mobile, satellite, and optical networks, and provide fine-grained control over the network parameters, such as bandwidth, delay, jitter, and packet loss. It can also integrate with real devices and applications, such as routers, switches, sensors, and smartphones, to create hybrid network experiments. Klonet is based on the Linux operating system and uses virtualization and containerization technologies to create isolated network nodes and links. It provides both GUI and CLI to help users configure and manage their network experiments.

[Figure 5]

Figure 9: Testbed built upon Klonet.

This testbed includes 8 user devices (devices 1 to 8), 8 home gateways (routers 1 to 8), and 1 core router (router 9). User devices connect to their home gateways via wired or wireless connections, and these home gateways are interconnected through routers or switches in the edge network. This topology reflects real-world household network interconnections. In addition, the CPU cores, memory, swap limits, bandwidth, and latency settings in Section 4 are based on measurements from the authors’ edge network.

- A.8 CONFIGURATIONS OF THE USED MODELS

Table 5: Configurations of the used Llama models.

Model (FP32) Layers Params Hidden Size Heads KV Heads Required Mem

|Llama 2-3B Llama 2-7B Llama 2-13B Llama 2-70B<br><br>|26 32 40 80|3 billion 7 billion 13 billion 70 billion<br><br>|3200 4096 5120 8192<br><br>|32 32 40 64<br><br>|– – – 8|14 GB 26 GB 50 GB 257 GB<br><br>|
|---|---|---|---|---|---|---|
|Llama 3.1-8B Llama 3.1-70B<br><br>|32 80|8 billion 70 billion<br><br>|4096 8192<br><br>|32 64|8 8<br><br>|31 GB 266 GB|
|Yi-34B<br><br>|60<br><br>|34 billion|7168<br><br>|56|8<br><br>|130 GB|

- A.9 PEAK MEMORY FOOTPRINT WITH MEMORY WINDOW SIZE 4

Table 6: Peak memory footprint per device with the memory window size set to 4.

| |Memory Scheduler Disabled (GB)<br><br>|Memory Scheduler Enabled (GB)|
|---|---|---|
|Model (FP32)<br><br>|N = 2 N = 4 N = 6 N = 8|N = 2 N = 4 N = 6 N = 8|

|Llama 2-3B Llama 2-7B Llama 2-13B Llama 2-70B<br><br>|7.3 4.3 3.2 2.8 13.7 7.7 5.5 4.5 25.8 13.9 9.7 8.0 129.9 66.5 46.7 35.0<br><br>|1.7 1.5 1.5 1.5<br>2.4 2.1 1.8 1.8 2.8 2.5 2.3 2.2 4.5 3.1 3.1 3.1<br>|
|---|---|---|
|Llama 3.1-8B Llama 3.1-70B|18.4 11.8 9.4 8.5 137.8 74.0 51.4 42.5<br><br>|6.3 5.8 5.6 5.5 10.8 10.5 11.5 11.4|
|Yi-34B|67 36.4 23.9 20.4<br><br>|6.0 5.6 5.3 5.2|

- A.10 REAL TESTBED AND CONFIGURATIONS

The real testbed consists of 4 laptops, all connected via a local Wi-Fi router, as shown in Figure 10. Table 7 details the hardware and network configurations of these laptops. In this case study, the laptop in the lower right serves as the master, while the other three laptops act as workers. The workers are connected to the master, and the master is currently generating the output sequence. The generated sequence is identical to that of single-server inference.

[Figure 6]

Figure 10: A real testbed composed of 4 laptops connected via local Wi-Fi.

Table 7: Hardware and network configurations of the laptops.

Device CPU Model Cores Memory Bandwidth Latency CUDA Number

|Mac Pro|Apple M1<br><br>|8|8 GB|510 Mbps<br><br>|5 ms|No|1|
|---|---|---|---|---|---|---|---|
|Mac Air|Intel Core i5<br><br>|4|8 GB|320 Mbps<br><br>|7 ms|No|1|
|Dell|Intel i7-1165G7<br><br>|8|16 GB<br><br>|610 Mbps<br><br>|3 ms|No<br><br>|2|

- A.11 CASE STUDY WITH 3 LAPTOPS

In this case, only 3 out of the 4 laptops are used: one MacBook Pro, one MacBook Air, and one Dell laptop. The results are given in Table 8, indicating a slightly higher latency due to reduced parallelism.

Table 8: Comparison of Transformers, Accelerate, Transformers+MS, and TPI-LLM on 3 laptops.

|Model (FP32)<br><br>|Transformers TTFT Latency<br><br>|Accelerate TTFT Latency|Transformers + MS TTFT Latency|TPI-LLM TTFT Latency|
|---|---|---|---|---|
| |(s) (s/token)|(s) (s/token)<br><br>|(s) (s/token)|(s) (s/token)|

|Llama 2-3B Llama 2-7B Llama 2-13B|61 30<br><br>115 56 OOM OOM<br><br>|24 16 30 26 OOM OOM|4 3 13 8<br><br>22 18<br><br>|3 2 7 6 14 12|
|---|---|---|---|---|
|Llama 3.1-8B<br><br>|133 65|37 31<br><br>|20 12|13 9|
|Yi-34B<br><br>|OOM OOM<br><br>|OOM OOM|185 55<br><br>|48 41|

