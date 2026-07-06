# arXiv:2504.08791v2[cs.DC]26Sep2025

## PRIMA.CPP: FAST 30-70B LLM INFERENCE ON HETEROGENEOUS AND LOW-RESOURCE HOME CLUSTERS

### Zonghang Li1 Tao Li2 Wenjiao Feng2 Rongxing Xiao2 Jianshu She1 Hong Huang3 Mohsen Guizani1 Hongfang Yu2 Qirong Ho1 Wei Xiang4 Steve Liu1 1MBZUAI 2UESTC 3City University of Hong Kong 4La Trobe University

ABSTRACT

On-device inference offers privacy, offline use, and instant response, but consumer hardware restricts large language models (LLMs) to low throughput and capability. To overcome this challenge, we present prima.cpp, a distributed on-device inference system that runs 30-70B LLMs on consumer home clusters with mixed CPUs/GPUs, insufficient RAM/VRAM, slow disks, Wi-Fi links, and heterogeneous OSs. We introduce pipelined-ring parallelism (PRP) to overlap disk I/O with compute and communication, and address the prefetch-release conflict in mmap-based offloading. We further propose Halda, a heterogeneityaware scheduler that co-optimizes per-device CPU/GPU workloads and device selection under RAM/VRAM constraints. On four consumer home devices, a 70B model reaches 674 ms/token TPOT with <6% memory pressure, and a 32B model with speculative decoding achieves 26 tokens/s. Compared with llama.cpp, exo, and dllama, our proposed prima.cpp achieves 5-17× lower TPOT, supports fine-grained model sizes from 8B to 70B, ensures broader cross-OS and quantization compatibility, and remains OOM-free, while also being Wi-Fi tolerant, privacy-preserving, and hardware-independent. The code is available at https://gitee.com/zonghang-li/prima.cpp.

1 INTRODUCTION

As a representative of next-generation AI, embodied AI (Gupta et al., 2021) is acutely sensitive to privacy protection, stable connectivity, and long-term financial costs. Home surveillance, alwayslistening assistants, and companion robots cannot send raw interactions to the cloud. Cloud services also suffer from frequent network failures, queuing, timeouts, and their costs scale linearly with token usage and service time. Consequently, embodied AI is moving from the cloud to the device.

However, on-device inference is constrained by modest chips and small RAM, struggling to run models beyond 8B (MLC, 2025; Lugaresi et al., 2019; Ghorbani, 2025), yet reliable long-term planning and tool use require 32B or more. To support larger models, disk offloading helps capacity but at the cost of speed (Gerganov, 2025; Li, 2023), e.g., Qwen 2.5-14B Q4K on an 8 GiB Mac M1 laptop takes 10 s/token with llama.cpp. Recent efforts use dedicated devices (e.g., Jetson, Mac Studio) for larger models and faster speed (Ye et al., 2024; 2025), but they are too expensive for most households. This motivates our goal: Can we deliver fast 30-70B inference on user-owned consumer devices that meet embodied AI’s demands summarized in Table 1?

Distributed inference may be the only solution to improve model size and speed while preserving the original outputs. Users usually own multiple devices, e.g., laptops, PCs, phones, and tablets. Some laptops and PCs have low-end GPUs like NVIDIA 20-/30-/40-series, and some Macs have Apple Silicon GPUs. By pooling the computing and memory of home devices, we can run larger models.

Despite recent progress, four limitations remain: (a) Existing systems require sufficient aggregated RAM/VRAM to hold the full model (Ye et al., 2024; Exo, 2025; Tadych, 2025; Zhang et al., 2025; Lee et al., 2024; Zhao et al., 2023; Zhang et al., 2024), driving up hardware costs and restricting model size. (b) Disk offloading slows inference, causing time-per-output-token (TPOT) of tens of seconds (Li et al., 2025). (c) Layer partitioning relies on the strong assumption in (a) and overlooks heterogeneity in OS-specific memory reclamation and disk access. (d) All devices are assumed to

Table 1: Comparison of cloud and local LLM deployments. TPS1 indicates the per-request token rate, TPS2 denotes TPS values measured across 32-70B models. OA denotes open-access.

Cost ($) Privacy Speed (TPS) Queue Net Support Model Cloud

App < 100 ✗ 10-701 ✓ ✓ Specific models API < 100 ✗ 10-701 ✓ ✓ Specific models Dedicated < 1,000 ✗ 10-352 ✗ ✓ Any OA model

Dedicated > 1,000 ✓ 4-172 ✗ ✗ Any OA model (≤70B) Consumer Free ✓ 2-262 ✗ ✗ Any OA model (≤70B)

Local

be necessary, even though removing slow ones could improve speed. These limitations raise two questions that motivate our design: (Q1) How can memory constraints be relaxed to run larger models? If disk offloading is required, how can disk latency be hidden? (Q2) Given Q1, how can we design heterogeneity-aware layer partitioning and identify bottleneck devices?

We propose prima.cpp, the first distributed inference system for consumer home clusters, with mixed CPUs/GPUs, insufficient RAM/VRAM, slow disks, Wi-Fi links, heterogeneous OSs, and capable of running 30-70B models at practical speed. To address Q1, Section 3.1 proposes pipelined-ring parallelism with prefetching to overlap disk latency and resolve the prefetch-release conflict in mmap. To address Q2, Section 3.2 models heterogeneity in compute, communication, memory, and OS-specific memory reclamation and disk optimizations. Section 3.3 develops Halda to find the optimal layer partitioning and select the best-performing devices. Section 4 presents experiments on real home clusters and shows that, even under limited memory, a 70B model runs locally at 674 ms/token with <6% memory pressure. With speculative decoding (Leviathan et al., 2023), a 32B model achieves 26 tokens/s, which is fast enough and paves the way for household embodied AI.

As summarized in Table 1, prima.cpp runs on existing, free devices; all data stays local; no queueing or timeouts; offline; and supports models up to 70B. To the best of our knowledge, prima.cpp is the first on-device system to deliver practical performance for 30-70B models in such constrained environments, without relying on specialized hardware (e.g., Jetson, NPUs) or altering model outputs.

- 2 RELATED WORK

On-device LLM systems. Most of them run on a resource-constrained device and can only handle small models. MLC-LLM (MLC, 2025) and MediaPipe (Lugaresi et al., 2019) bring 7B models to mobile phones and browsers. PocketPal AI (Ghorbani, 2025), which runs on Android using llama.cpp (Gerganov, 2025), supports models up to 3.8B. Some efforts push for larger models, like AirLLM (Li, 2023), which loads only the needed layers to save memory but at the cost of speed. Others turn to high-end hardware, such as the Apple M2 Ultra (192 GiB RAM) for a 65B model or kTransformers (kvcache ai, 2025) (382 GiB RAM) for a 671B model (∼75 GiB for 70B). Like kTransformers, HeteGen (Zhao et al., 2024) also collaborates with the CPU and GPU on one device to accelerate inference. These setups (large RAM, advanced CPUs with specific instruction sets, dedicated hardware) go far beyond common home devices and are inaccessible to most households.

Distributed on-device LLM systems. Such distributed systems follow two main parallelism paradigms: tensor parallelism and pipeline parallelism.

- - Tensor Parallelism (TP). TP splits tensors across devices to share the load (Shoeybi et al., 2019). To speed up all-reduce, dllama (Tadych, 2025) uses USB4 and Thunderbolt 5 for fast connections, and AirInfer (Zhang et al., 2025) uses wireless analog superposition to perform all-reduce over the air. Due to device heterogeneity, Hepti (Lee et al., 2024) optimizes workload partitioning with three slicing strategies for different memory budgets, and Galaxy (Ye et al., 2024) prioritizes compute power first, then memory, to maximize speed and avoid OOM. These systems reside the full model in the aggregate memory. With limited aggregate memory, only small models can run. TPI-LLM (Li et al., 2025) loads model layers on demand and hides disk loading with prefetching. This enables low-end devices with only 4 GiB of RAM to run a 70B model, but at 30 s/token, making it impractical.

Table 2: Comparison of distributed on-device LLM systems. (Abbr.: Quantization, Heterogeneity)

Type Backends Mem Quant. Mem Stress Speed Hete. dllama TP CPU RAM Q4 Critical Slow ✗

AirInfer TP CPU RAM FP32 Critical Slow ✓

Hepti TP CPU RAM FP32 Critical Slow ✓ Galaxy TP+SP CPU/GPU RAM/VRAM FP32 Critical Slow ✓

TPI-LLM TP CPU RAM FP32 Medium Slow ✗ exo PP CPU/GPU RAM/VRAM Q4+FP32 Critical Slow ✓

LinguaLinked PP CPU RAM Q8/FP32 Critical Slow ✓ EdgeShard PP CPU/GPU RAM/VRAM FP32 Critical Fast ✓ prima.cpp PRP CPU&GPU RAM&VRAM Q4/IQ1 Low Fast ✓

- - Pipeline Parallelism (PP). Due to Wi-Fi’s high latency, pipeline parallelism becomes more suitable for home clusters as it requires less P2P communication. Exo (2025); Zhao et al. (2023); Zhang et al.

(2024) split the model into segments and assign them to devices based on memory, compute, and network conditions. Each device computes its segment and passes the result to the next, until the last device outputs the next token. Exo (2025) partitions model segments based on memory ratio; LinguaLinked (Zhao et al., 2023) uses linear optimization to solve the device assignment problem; and EdgeShard (Zhang et al., 2024) uses dynamic programming.

However, these systems either require dedicated hardware (e.g., Jetson AGX/Nano) or sufficient aggregate memory to reside the full model, support only CPU or GPU backends, overlook heterogeneity (especially in disk offloading), and impose high memory pressure. Table 2 summarizes their features. These limitations raise hardware costs, restrict use to small models, cause slow inference and device freezes, and discourage users from deploying LLMs at home.

Instead, prima.cpp runs on consumer devices, supports disk offloading to run larger models, and uses pipelined-ring parallelism (PRP) to overlap disk latency. It runs on both GPUs and CPUs, combines RAM and VRAM, and models system heterogeneity to optimize GPU-CPU layer partitioning per device. Moreover, it stores model weights in the OS page cache, allowing the OS to reclaim memory as needed to preserve user experience. These features distinguish prima.cpp from existing systems.

- 3 PRIMA.CPP: USE PIPELINED-RING PARALLELISM IN LLAMA.CPP

- 3.1 PIPELINED-RING PARALLELISM WITH PREFETCHING

PP is effective for batched inference when aggregated memory is abundant; however, this prerequisite rarely holds in households. In such cases, on-device LLMs typically serve very few users at low frequency, often a single request at a time1, which prevents mini-batching and leaves significant pipeline bubbles. Besides, home devices are few and have limited available memory2, making it difficult for users to build a cluster that can hold the full model in memory.

To relax memory constraints, we extend PP with mmap, building on llama.cpp (Gerganov, 2025): when computation requires the layers, mmap loads them from external storage on demand, and lets the OS evict them under memory pressure, so a small-memory cluster can run larger models. To hide disk loading latency, we employ prefetching, allowing devices to preload the next model segment.

This naive design supports larger models with low memory pressure but suffers from slow speed, because prefetching will fail due to prefetch-release conflict: if disk reads are fast, later-loaded layers will evict earlier prefetched layers from page cache, so when computation begins, the needed layers are no longer cached (see Appendix A.1 for details). This triggers page faults and reloads, negating the benefit of prefetching, and pipeline bubbles remain significant (see Fig. 6 d,e in Appendix A.2).

Pipelined-ring parallelism (PRP) with prefetching. To address the prefetch-release conflict, we further propose PRP, which connects devices end-to-end in a ring and runs multiple rounds to predict

- 1We target a single request, but this can extend to batches via dynamic batching, which we have implemented.
- 2We cannot occupy all memory, as this may disrupt other apps (e.g., TikTok) and cause users to kill the LLM.

one token. In each round, devices prefetch different layer segments from the disk, which overlap with other devices’ ongoing operations, such as computing, communication, and disk loading. Since only a small segment (whose size is referred to as the layer window size hereafter) is reloaded per round, memory overflow can be avoided, and prefetched layers are less likely to be evicted, thus mitigating the prefetch-release conflict (see Figs. 4 and 5 in Appendix A.2 for examples). In addition, PRP processes both input and output on the head device, providing enhanced interaction privacy.

In Fig. 1, all devices share a layer window size of 2. Each device processes two model layers per round and takes 3 rounds to predict one token. Fig. 6 a,b in Appendix A.2 show the PRP timeline on homogeneous devices with fast and slow disks. On fast disks, prefetching latency is fully overlapped; on slow disks, CPU and disk operate alternately, removing bubbles but leaving residual page-fault loading latency, since required layers may not be fully loaded when computation begins. Fig. 2 evaluates this multi-round design: for large models, PRP reduces PP’s TPOT by about 50% (PP is equivalent to PRP at k = 1), while for small models, PRP converges to PP with similar TPOT.

Model Segments (total 36 layers, layer window size 2)

|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Device 1 Device 2 Device 3

[Figure 1]

[Figure 2]

- Round 1
- Round 2
- Round 3

[Figure 3]

|3|
|---|

1

2

Hi

7 13

8 14

|9|
|---|

|15|
|---|

Hello

[Figure 4]

[Figure 5]

[Figure 6]

|12|
|---|

|18|
|---|

17 16

10

11

|6|
|---|

5

4

Device 6 Device 5 Device 4

Figure 1: Pipelined-ring parallelism. In this case, 6 devices handle a 36-layer model. With a layer window size of 2, the model is split into 18 segments and assigned to 6 devices in ring order, so each device needs 3 rounds to predict one token.

In practice, however, home devices are heterogeneous. Using a uniform layer window size still produces pipeline bubbles (see Fig. 6c in Appendix A.2), whereas assigning different window sizes per device can reduce them (see Fig. 6f in Appendix A.2). In general, stronger devices should be assigned larger window sizes. However, determining which devices are stronger and how large their window sizes should be is challenging. Beyond common heterogeneity in computing, communication, and memory, disk loading is heavily affected by OS-specific memory reclamation and disk read throughput. Such complex heterogeneity makes disk latency difficult to quantify.

- 3.2 LAYER-TO-DEVICE ASSIGNMENT PROBLEM

As mentioned in Table 2, prima.cpp is designed to use multiple backends3. Within a layer window, some layers reside in VRAM and execute on the GPU, while others are offloaded to RAM and executed on the CPU. If the layers assigned to the CPU exceed available RAM, the overflow is further offloaded to external storage and reloaded into RAM via mmap4. This leads to two challenges: How to set the layer window size for each device? Which layers run on the GPU and which on the CPU?

Prior work assumes sufficient aggregated VRAM and proposes heuristic partitioning strategies. For example, Exo (2025) partitions model layers based on memory ratio, so devices with more RAM/VRAM handle more layers. Ye et al. (2024) partition by compute power and migrate layers from OOM devices to those with free memory. These heuristics work on some testbeds but are not always optimal (see Fig. 10 in Appendix A.10), since memory size does not guarantee compute power, and under disk offloading, devices with larger memory but weaker CPUs/GPUs can perform worse than those with less memory but faster CPUs and disks.

To achieve optimal partitioning, TPOT must be quantified through an analytical model. Building such a model is challenging: beyond device heterogeneity, we must account for CPU-GPU coordination,

- 3Our prototype supports CUDA, Metal, and CPU backends, but with minor adaptation, it can also work with others such as Vulkan and ROCm.
- 4We pin model layers in VRAM and avoid paging them as in disk-RAM; otherwise, disk-RAM and RAMVRAM transfers would contend for PCIe (or shared) bandwidth, increase disk reload volume, and amplify disk loading latency, which offsets the gain from GPU acceleration.

memory contention, OS-specific memory reclamation, disk optimizations, and quantization. For instance, some PCs have dedicated GPUs with separate VRAM; others (e.g., Mac M-series) adopt a UMA architecture where CPU and GPU share memory, and OS reclamation will be more aggressive; some are NUMA systems without GPUs, and they differ in reclamation thresholds across OSs; even on the same device, OS reclamation differs with or without Metal; Linux optimizes sequential disk reads, making reloads faster; quantization influences compute latency, memory access, disk loading, and RAM/VRAM constraints, ultimately shaping the partitioning strategy.

After extensive performance analysis in Appendix A.3 and preliminary experiments, we formalize this as the layer-to-device assignment (LDA) problem as follows.

Definition 1 (Layer-to-device assignment). Assume there are M devices, wm is the layer window size on device dm and nm is the number of GPU layers within wm. Let the decision variables be wT = [w1,w2,··· ,wM] and nT = [n1,n2,··· ,nM]. Our objective is to find the optimal w and n that minimizes the TPOT:

aT · w + bT · n + eT · c eT · w

+ κ, (1)

L ·

min

w,n

s.t. wm ∈ Z>0,nm ∈ Z≥0,nm ≤ wm ≤ L, (2) L − k(eT · w) = 0,k ∈ Z>0, (3) Pw · w′ + Pn · n′ + eT · w · z ≤ 0, (4)

− Pngpu · zgpu · eT · w + Pngpu · n ≤ 0. (5)

where L is the number of model layers; a,b,c are latency coefficient vectors determined by compute latency, memory access latency, disk loading latency, and communication latency on each device; κ is a constant latency offset; k is the number of rounds to predict one token; w′ and n′ are the extended vectors of w and n; z and zgpu are constraint vectors for RAM and VRAM; Pw,Pn,Pngpu are diagonal matrices that activate or deactivate the decision variables; and e is an all-ones vector.

Constraint (2) ensures the layers do not exceed the limit. Constraint (3) enforces that all devices are assigned an equal number of windows and all windows are filled5. Constraints (4) and (5) ensure that RAM and VRAM usage stay within limits. Table 6 summarizes the key symbols. To construct a,b,c,κ,w′,n′,z,Pw,Pn, we categorize devices into four sets: (a) Set M1: macOS with Metal disabled and insufficient RAM; (b) Set M2: macOS with Metal enabled and insufficient RAM; (c) Set M3: Linux and Android with insufficient RAM; (d) Set M4: devices with sufficient RAM or slow disk. For Sets M1-M3 where devices should overload, RAM usage should stay above available RAM; for Set M4, where overloading is not allowed, RAM usage should stay below available RAM. We can extend other OSs to these sets, or create a new set and adjust variable dimensions.

This problem is an NP-hard integer linear fractional program (ILFP): both the numerator and denominator of the objective are linear in the decision variables, and all constraints are linear inequalities. Whether a device is overloaded depends on w and n, e.g., a large wm − nm will overload RAM. However, we cannot determine a device’s set before solving the LDA problem, and without the set assignment, we cannot solve the LDA problem. This traps us in a circular dependency.

- 3.3 HALDA: AUTOMATIC LAYER PARTITIONING AND DEVICE SELECTION

To solve this non-standard, NP-hard LDA problem, our core ideas include: (i) transform the original problem into a set of standard integer linear programs (ILPs) by enumerating over all valid k that divide L, and (ii) search for optimal set assignments M1-M4 by iterative optimization.

Transform into standard ILPs. Given that the number of layers L in LLMs is typically less than 100, the integer k has a limited range of values: at most 11 valid factors for any L ≤ 100. By

5This is not mandatory in our implementation, but it can simplify the problem model.

Algorithm 1: Heterogeneity-Aware Layer-to-Device Allocation (HALDA)

- 1 Initialize layer windows w proportionally to devices’ memory budgets, and GPU layers n ← 0;
- 2 Calculate platform-specific coefficients αm, βm, ξm for each device m;
- 3 Calculate valid factors KL of L (excluding L);
- 4 while true do

- 5 Calculate W = eT · w and k = L/W;
- 6 Reassign devices to sets M1,M2,M3,M4 based on the latest w,n,k and Mforce4 ;
- 7 if the assignment sets M1,M2,M3,M4 remain unchanged then

- 8 break;

- 9 Calculate the objective coefficients a, b, c, κ, the RAM upper bound z, and the VRAM/shared memory upper bound zgpu according to the updated assignment sets;
- 10 foreach k ∈ KL do

- 11 Solve the ILP for fixed k using a solver;
- 12 Update best solution (w⋆,n⋆) if the current objective is smaller;

- 13 if any device has free VRAM but another device is overloaded then

- 14 Force the device ms from {M1,M2,M3} with slowest disk reads into Mforce4 ;
- 15 continue;

- 16 Update w ← w∗ and n ← n∗;

- 17 return w⋆,n⋆;

enumerating these factors, we treat k and W as constants, and the problem becomes:

k(aT · w + bT · n + eT · c) + κ, (6)

min

w,n

s.t. wm ∈ Z>0,nm ∈ Z≥0,nm ≤ wm ≤ L, (7) eT · w = W, (8) Pw · w′ + Pn · n′ + Wz ≤ 0, (9) Pngpu · n − WPngpu · zgpu ≤ 0. (10)

Hence, for each fixed k, the objective and constraints boil down to linear functions/inequalities, and the problem becomes an ILP. Then, we can run a standard ILP solver (e.g., HiGHS (Huangfu & Hall, 2018)) to obtain the optimal w,n.

Iterative optimization for set assignment. The problem remains unsolvable because the sets M1M4 are unknown. To break this circular dependency, we adopt an iterative optimization procedure. We start by setting w proportional to available memory 6 and initializing n = 0, which gives an initial division of devices into M1-M4. We then solve the ILPs to update w and n, reassign devices, and iterate until the sets converge.

However, this approach still has defects. For example, if a device m ∈ {M1,M2,M3} is assigned 30 layers, although the GPU can host 20 layers, memory constraints may end up assigning only 10 to the GPU and 20 to the CPU, underutilizing the GPU. This issue arises from an improper set initialization: if the device was initialized in M4, its GPU could be fully utilized. Therefore, Algorithm 1 includes a calibration step: if a GPU is underutilized (VRAM not full) while another device is overloaded (VRAM full with layers offloaded to CPU, or RAM exceeded), we move the device with slowest disk reads from the set {M1,M2,M3} into Mforce4 (Mforce4 ⊂ M4) and solve again. This ensures convergence to the optimal set assignment and the corresponding optimal values of w, n, and k.

Complexity analysis. The main loop alternates between set assignment and LDA solving, repeated for T iterations with T = O(M) in the worst case. The set assignment takes O(M). For LDA, we solve a tiny ILP for each valid factor of L, giving K = O(log L) factors in total. Although ILPs are NP-hard, our instances are small and sparse, allowing modern solvers to finish quickly. Empirically, on 4-32 devices, the global scheduling latency is 10-12 ms. The overhead of rerunning Halda is

6davailm for macOS without Metal and Linux, davailm,metal for macOS with Metal, and davailm + dswapoutm for Android.

Table 3: Testbed configuration. Termux is used on D4 and D5.

D1 D2 D3 D4 D5 D6

Device Mac M1 Laptop Desktop Mate40Pro Honor Pad Mac Air OS macOS Linux Linux HarmonyOS Android macOS CPU Apple M1 Intel i9 Intel i9 Kirin 9000 Dimensity 8100 Intel i5 CPU Cores 8 8 16 8 8 4 RAM (avail.) 2.4 GiB 4.1 GiB 9.7 GiB 1.9 GiB 5.1 GiB 6.8 GiB Disk Read 0.7 GB/s 3.0 GB/s 3.0 GB/s 1.4 GB/s 2.0 GB/s 0.4 GB/s GPU Type Apple Silicon 3070 2080TI - - VRAM (avail.) - 8 GiB 11 GiB - - -

negligible, and since no cross-device layer migration is required, workloads can be repartitioned whenever the task queue is empty to adapt to environmental changes.

Device selection. Weak devices assigned only one layer are removed, since Halda indicates that excluding them would improve speed. Appendix A.7 illustrates this selection process. For small models, Halda prefers to allocate all layers to one most powerful device, reducing prima.cpp to llama.cpp. However, if removing a device would break the communication ring (e.g., it is the only reachable hop due to network policy), Halda keeps it in the ring as a relay node but assigns it no workload. This design frees users from device selection: we only need to discover more available devices, and Halda will select the optimal subset for us.

4 EXPERIMENTS

We implement prima.cpp with 20K lines of code based on llama.cpp and build a real testbed of heterogeneous, low-end home devices (see Table 3). These devices connect via a local Wi-Fi router, with inter-device bandwidth ranging from 320-610 Mbps and link latency from 3-7 ms. By default, four devices (D1-D4) with aggregated RAM+VRAM of 37 GiB (insufficient for a Q4K-quantized 70B model) are used. We evaluated Llama models from 8B to 70B (Q4K) in terms of time-per-outputtoken (TPOT), time-to-first-token (TTFT), and memory pressure. We select llama.cpp, exo, and dllama as baselines, as they are the most popular open-source on-device (distributed) LLM inference systems7. Since llama.cpp is a standalone on-device system, we run it on D3, which offers the largest RAM/VRAM and highest decoding efficiency8. Exo is a PP system, which we run on D1-D3, but D4 is unavailable because it requires root access. In contrast, dllama and prima.cpp run on D1-D4.

- 4.1 FAST INFERENCE ON LARGE MODELS

Table 4 presents TPOT and TTFT across model sizes of 8-70B9. As a result, prima.cpp achieves substantially lower TPOT and TTFT than all baselines, and matches llama.cpp on small models (<14B). Compared with llama.cpp, it reduces TPOT by up to 17× and TTFT by up to 8×; against exo and dllama, it achieves 5-8× lower TPOT and 12-24× lower TTFT, without OOMs.

For small models (<14B), Halda finds that D3 can hold the entire model in VRAM, so it removes the other devices and runs only on D3. This reduces prima.cpp to the single-device case (equivalent to llama.cpp), with both achieving the same TPOT and TTFT. At 30B, as shown in Fig. 9a in Appendix A.8, D3-GPU runs out of VRAM and forces layers onto the CPU, causing llama.cpp to deteriorate rapidly. In contrast, Fig. 9d shows that Halda offloads the overflow from D3-GPU to D2-GPU, keeping TPOT and TTFT stable. At 45B, both RAM and VRAM on D3 are exhausted, and mmap in llama.cpp begins frequent reloads, incurring disk latency. At this stage, only a few pages are reloaded, so the efficiency loss is minor. At 60B, high memory stress causes more active pages to be evicted earlier, sharply increasing TPOT and TTFT. This indicates that llama.cpp is ill-suited for large models on consumer-grade devices. In contrast, Halda balances workloads across devices according to their

- 7At submission, llama.cpp had 87K stars, exo 31K stars, and dllama 2.7K stars. We also tried to deploy

vLLM and SGLang, but as server-oriented systems, they failed to run on our home cluster.

- 8At small batch sizes, decoding is bandwidth-bound, and 2080TI offers higher memory bandwidth than 3070.
- 9At submission, exo and dllama didn’t support Llama models from 14-65B, so we mark their values with "-".

Table 4: TPOT (ms/token) and TTFT (ms) on Llama models for llama.cpp, exo, dllama, prima.cpp.

prima.cpp (w/o halda)

prima.cpp (w/o prefetch) prima.cpp

llama.cpp exo dllama

Size

TPOT TTFT TPOT TTFT TPOT TTFT TPOT TPOT TPOT TTFT

8B 15 18 263 960 459 1845 78 15 15 18 14B 20 25 - - - - 131 20 20 25 30B 202 611 - - - - 258 79 72 214 45B 328 712 - - - - 409 263 233 440 60B 7965 8350 - - - - 7053 532 468 990 65B 8807 9662 - - - - 12253 688 569 1770 70B 10120 10806 OOM OOM OOM OOM 20848 755 674 1793

capabilities. Even when disk offloading is required, it assigns workloads to devices with strong CPU and fast disk, while PRP with prefetching hides disk latency. As a result, prima.cpp maintains stable TPOT and TTFT growth and achieves sub-second TPOT even at 70B. To the best of our knowledge, prima.cpp is the first system to achieve this speed on real consumer-grade, low-end devices10.

For exo and dllama, despite only 8B and 70B being available, their limitations are already evident. Llama 3-8B (Q4K) requires 5.3 GiB of VRAM, so the entire model could fit into D3-GPU. However, exo allocates layers proportional to each device’s memory. D1 (8 GiB RAM), D2 (8 GiB VRAM), and D3 (11 GiB VRAM) are assigned 9, 10, 13 layers, respectively. While D1 has an Apple Silicon GPU, its efficiency is much lower than the 2080TI on D3-GPU, making it an efficiency bottleneck.

For dllama, inference is based on TP. For the 8B model, 64 all-reduces per token incur at least 192 ms of delay under Wi-Fi latency (3-7 ms), making its measured TPOT (459 ms/token) far above prima.cpp (15 ms/token). At 70B, dllama runs out of memory, but our independent tests show ∼150 ms per all-reduce, so 160 all-reduces add 24 s/token, yielding ∼30 s/token compared to prima.cpp’s 674 ms/token (442 ms/token with speculative decoding). These results highlight that Halda and PRP are better suited for heterogeneous, high-latency home environments than heuristic partitioning or TP.

Similar results are observed on another testbed in Appendix A.4 and a homogeneous testbed in Appendix A.5, confirming that prima.cpp’s speedups generalize across diverse testbeds. Appendix

- A.6 shows the results on more models, including Qwen 2.5, QwQ, and DeepSeek R1. Appendix
- A.7 explains the underlying rationale for device selection: more is not always better, and aggregated memory need not match the model’s needs. It also shows how Halda selects a subset of devices to build a best-performing cluster. Appendix A.9 shows that with speculative decoding, 32B inference achieves 26 tokens/s, offering both the speed and intelligence needed for LLM agents.

- 4.2 ABLATION STUDY ON PREFETCHING, HALDA, AND PIPELINED-RING PARALLELISM

Table 4 also presents ablations on Halda and prefetching. Exo is also a PP system, so we migrate exo’s layer partitioning (based on memory ratio) to prima.cpp (w/o halda), making it an exo variant. Unlike exo, this variant uses available instead of total memory, adds a broader range of models, adds cross-platform quantization and prefetching, and supports CPU/disk offloading to prevent OOMs.

Prefetching. To evaluate prefetching, we compare prima.cpp with and without it. It has little effect on small models that fit in RAM/VRAM, but for larger models, evicted layers cause frequent page faults and reloads. Without prefetching, all reloads are triggered on demand by page faults. With prefetching, upcoming layers are loaded in advance and their latencies overlapped, reducing page fault reloads and lowering TPOT by 9-17%.

Halda. A proper layer partitioning is critical for fast inference. As shown in Table 4, prima.cpp with Halda reduces TPOT by up to 31× over prima.cpp (w/o halda). For small models, Halda selects the most powerful device (D3) to run all layers, while exo partitions by memory ratio, assigning more layers to weak devices. For larger models, where disk offloading is unavoidable, Halda balances workloads to reduce I/O pressure or prioritizes devices with strong CPU and fast disk, whereas exo

10We exclude sparse inference systems, as they do not meet the conditions in Table 1: they require dedicated hardware such as NPUs, support only sparse models, and may degrade output quality.

Table 5: Memory pressure for each device on Llama models.

|llama.cpp<br><br>|exo<br><br>|dllama|prima.cpp|
|---|---|---|---|
|D3|D1 D2 D3<br><br>|D1 D2 D3 D4<br><br>|D1 D2 D3 D4|

Size

|8B 14B<br><br>|2.0% 2.5%<br><br>|20.0% 51.3% 42.5% - - -|13.5% 12.8% 55.8% 12.8% - - - -<br><br>|5.3% 5.4% 2.7% ≤1.0% 5.3% 4.3% 2.2% ≤1.0%|
|---|---|---|---|---|
|30B 45B 60B 65B 70B<br><br>|8.0% 3.9%<br><br>5.5% 15.6%<br><br>6.0%<br><br><br>|- - -<br>- - -<br>- - -<br>- - -<br><br><br>OOM OOM OOM|- - - -<br><br>- - - -<br><br>- - - -<br><br>- - - -<br><br><br>OOM OOM OOM OOM<br><br>|3.0% 5.7% 2.9% ≤1.0%<br>4.9% ≤1.0% 6.0% ≤1.0% 6.3% 4.7% 4.7% ≤1.0%<br><br><br>3.9% ≤1.0% ≤1.0% ≤1.0%<br>4.7% 4.8% 4.8% ≤1.0%<br>|

often overloads slow-disk devices, causing TPOT to spike. Appendix A.10 compares Halda with two heuristic baselines, further showing its effectiveness and novelty.

PRP with multiple per-token rounds. To evaluate PRP in isolation, we built a CPU cluster with

180

| || |
|---|
<br><br>Llama 3-8B<br><br>Llama 3-14B<br><br>DeepSeek R1-32B<br><br>Llama 3-45B Llama 3-60B<br><br>Llama 65B<br><br>DeepSeek R1-70B<br><br>Llama 3-70B<br><br>Qwen 2.5-72B| | | | | | | |
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

- 4 devices, each with 8 cores, 8 GiB RAM, and an SSD of 2 GB/s. We tested models from 8-72B, and assigned layers evenly across devices. For example, for models with 80 layers, at the per-token round k = 1, layers were split 20:20:20:20; at k = 2, 10:10:10:10, and so on. With large models (>60B) and k = 1, PRP degrades to PP and provides no benefit due to the prefetch-release conflict (see Appendix A.1). In contrast, with k ≥ 2, Fig. 2 shows that PRP halves TPOT by eliminating the prefetch-release conflict and allowing finer-grained overlap of disk loading (see Appendix A.2). With small models (<45B) and sufficient memory, the model fits in the cluster without disk offloading. Then, increasing k has minimal impact on TPOT, aside from minor overhead from additional communication and kernel launches. This shows that PRP is more effective than PP for 70B-scale models.

160

140

NormalizedTPOT

120

100

80

60

40

20

0

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 k

Figure 2: TPOT of PRP under different rounds k.

4.3 LOW MEMORY PRESSURE TO PRESERVE USER EXPERIENCE

Memory pressure is critical for user experience, as high pressure can slow apps or crash the device. For example, if a phone runs an LLM service while the user is browsing TikTok, memory competition may cause TikTok to lag or crash, prompting the user to terminate the LLM. An implicit advantage of prima.cpp is that it accounts for this by running at a lower memory priority than other apps: it reduces RAM usage when apps start and uses more when they stop. By giving up RAM to keep other apps responsive, prima.cpp preserves user experience and protects itself from being killed by the user.

We define memory pressure from the LLM as ∆mem_available/mem_total (e.g., 2 GiB used / 8 GiB total = 25%)11. Table 5 shows that exo and dllama pin model weights in RAM, raising memory pressure by over 50% on some devices. This can force the OS to reclaim memory from other apps (e.g., compression, swapping), causing lag or crashes, and can still lead to OOM for larger models. The root cause is that they prioritize the LLM at the cost of other apps, which users do not want. In contrast, llama.cpp and prima.cpp exhibit low pressure: both pin only minimal KV caches and compute buffers, while prima.cpp further distributes them across devices. Model weights are loaded via mmap and cached in pages that can be released instantly for other apps. These properties make prima.cpp a practical choice for consumer devices, where LLMs must run alongside user apps.

11mem_available includes free and reclaimable pages, so its reduction reflects only non-reclaimable pages, i.e., true memory stress. Appendix A.8 shows each device’s memory footprint, but not memory pressure, since reclaimable memory (e.g., page cache) is included and can be freed instantly.

- 5 CONCLUSION

This paper proposes prima.cpp, the first on-device distributed system to deliver practical speed for 3070B LLMs on consumer-grade home devices, achieving 26 tokens/s for 32B models and 2 tokens/s for 70B models with speculative decoding. We introduce PRP to run models exceeding the memory limit and resolve the prefetch-release conflict to reactivate prefetching. To handle device heterogeneity and reduce bubbles in PRP, we develop the LDA model, which captures the heterogeneity in computing, communication, memory, and OS-specific reclamation, disk optimizations. We solve it via Halda, which performs smart layer partitioning for minimal TPOT and automatically selects the bestperforming devices. The system is meticulously designed, and this level of detail surpasses prior work that often relies on simplified heuristics. It offers features desired by home users: no extra hardware cost, no specialized hardware, privacy, offline, no queueing or timeouts, models beyond memory limit, practical intelligence and speed, low memory pressure, cross-platform, heterogeneity awareness, automatic workload allocation and device selection, and Wi-Fi ready. These make prima.cpp a compelling choice for home deployment and drive broader adoption of embodied AI at home.

REFERENCES

Exo. exo: Run your own ai cluster at home with everyday devices. https://github.com/ exo-explore/exo, 2025.

Georgi Gerganov. llama.cpp: Llm inference in c/c++. https://github.com/ggerganov/ llama.cpp, 2025.

Asghar Ghorbani. Pocketpal ai: An app that brings language models directly to your phone. https: //github.com/a-ghorbani/pocketpal-ai, 2025.

Agrim Gupta, Silvio Savarese, Surya Ganguli, and Li Fei-Fei. Embodied intelligence via learning and evolution. Nature Communications, 12(1):5721, 2021.

Qi Huangfu and JA Julian Hall. Parallelizing the dual revised simplex method. Mathematical Programming Computation, 10(1):119–142, 2018.

kvcache ai. ktransformers: A flexible framework for experiencing cutting-edge llm inference optimizations, 2025. URL https://github.com/kvcache-ai/ktransformers.

Juhyeon Lee, Insung Bahk, Hoseung Kim, Sinjin Jeong, Suyeon Lee, and Donghyun Min. An autonomous parallelization of transformer model inference on heterogeneous edge devices. In Proceedings of the 38th ACM International Conference on Supercomputing, pp. 50–61, 2024.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pp. 19274–19286. PMLR, 2023.

Gavin Li. Airllm: scaling large language models on low-end commodity computers, 2023. URL https://github.com/lyogavin/airllm/.

Zonghang Li, Wenjiao Feng, Mohsen Guizani, and Hongfang Yu. Tpi-llm: Serving 70b-scale llms efficiently on low-resource mobile devices. IEEE Transactions on Services Computing, 2025.

Camillo Lugaresi, Jiuqiang Tang, Hadon Nash, et al. Mediapipe: A framework for perceiving and processing reality. In 3rd Workshop on Computer Vision for AR/VR at CVPR, 2019.

MLC. Mlc-llm: Universal llm deployment engine with ml compilation. https://github.com/ mlc-ai/mlc-llm, 2025.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, et al. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Bartłomiej Tadych. Distributed llama. https://github.com/b4rtaz/ distributed-llama, 2025.

Shengyuan Ye, Jiangsu Du, Liekang Zeng, Wenzhong Ou, Xiaowen Chu, Yutong Lu, and Xu Chen. Galaxy: A resource-efficient collaborative edge ai system for in-situ transformer inference. In

- IEEE INFOCOM 2024-IEEE Conference on Computer Communications, pp. 1001–1010, 2024.

Shengyuan Ye, Bei Ouyang, Liekang Zeng, Tianyi Qian, Xiaowen Chu, Jian Tang, and Xu Chen. Jupiter: Fast and resource-efficient collaborative inference of generative llms on edge devices. In

- IEEE INFOCOM 2025-IEEE Conference on Computer Communications, pp. 1–10, 2025.

Kai Zhang, Hengtao He, Shenghui Song, Jun Zhang, and Khaled B Letaief. Distributed on-device llm inference with over-the-air computation. arXiv preprint arXiv:2502.12559, 2025.

Mingjin Zhang, Xiaoming Shen, Jiannong Cao, Zeyang Cui, and Shan Jiang. Edgeshard: Efficient llm inference via collaborative edge computing. IEEE Internet of Things Journal, 2024.

Junchen Zhao, Yurun Song, Simeng Liu, Ian G Harris, and Sangeetha Abdu Jyothi. Lingualinked: A distributed large language model inference system for mobile devices. arXiv preprint arXiv:2312.00388, 2023.

Xuanlei Zhao, Bin Jia, Haotian Zhou, Ziming Liu, Shenggan Cheng, and Yang You. Hetegen: Efficient heterogeneous parallel inference for large language models on resource-constrained devices. In 6th Proceedings of Machine Learning and Systems, pp. 162–172, 2024.

A APPENDIX

- A.1 PREFETCH-RELEASE CONFLICT

As illustrated in Fig. 3, before computation starts, the OS prefetches 3 model layers to the available memory limit. However, it does not stop and continues to load the 4th layer, causing the 1st layer to be released. This prefetch-release cycle repeats, so by the end, the last 3 layers are in memory, while the first 3 are not. Then, when computation begins, the 1st layer, which is not in memory, triggers a page fault, prompting the OS to reload it and the 4th layer to be released. Finally, all layers are loaded twice, incurring unnecessary disk I/O without any benefit from prefetching.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

Prefetch Prefetch

OS Release

Prefetch

OS Release

Prefetch

OS Release

Prefetching:

Computing:

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

Page Fault

OS Release

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

Page Fault

OS Release

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

Page Fault

OS Release

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

Page Fault

OS Release

① ② ③ ④

⑤ ⑥ ⑦ ⑧

Figure 3: Illustration of model layers loaded into memory in PP with prefetching. In this case, the device handles 6 model layers, but its available memory can only hold 3. The green blocks show the layers loaded into memory, while white blocks indicate those not yet loaded.

- A.2 HOW PIPELINED-RING PARALLELISM SOLVES THE PREFETCH-RELEASE CONFLICT

- Fig. 4 illustrates a fast-disk device where prefetching is fast enough to complete before computation begins. In this case, with a fast disk and a layer window size of 2, ⃝1 prefetching is fast enough to load 2 layers before computation begins, then ⃝2 computation runs without page faults. Then, ⃝3 the next round of 2 layers is prefetched, replacing the used layers. Steps ⃝2 -⃝7 repeat until inference is complete. Prefetching overlaps with other devices’ operations, so its latency does not contribute to TPOT. Here, with no page faults, TPOT comes only from computation. In other words, disk loading latency is fully overlapped.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

Prefetch Compute

① ② ③

Prefetch

OS Release

④

Compute

⑤

Prefetch

OS Release

⑥

Compute

⑦

Prefetch

OS Release

Jump to ②

Figure 4: Illustration of model layers loaded into memory in PRP with a fast disk.

- Fig. 5 shows a common case with a slow disk. In this case, ⃝1 prefetching loads only one layer, then ⃝2 computation begins, ⃝3 a page fault is triggered upon reaching the 2nd layer, blocking until it loads. After computation, ⃝4 the device prefetches the next round of 2 layers, but only one layer loads due to the slow disk, and the OS releases the oldest layer. Then, ⃝5 the next round of computation begins, and ⃝6 at the 6th layer, another page fault occurs. This cycle of "loading (prefetch) - computing -

Compute (Page Fault)

Prefetch Compute

Prefetch

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

① ② ③ ④

OS Release

Compute

Compute (Page Fault)

Prefetch

Compute

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

⑤

⑥

⑦

⑧

OS Release

OS Release

Compute (Page Fault)

Prefetch

Compute (Page Fault)

Compute

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

⑪

⑫

⑨

⑩

OS Release

OS Release

OS Release

Figure 5: Illustration of model layers loaded into memory in PRP with a slow disk.

loading (page fault) - computing" repeats until inference completes. While page fault-induced loading blocks computation, prefetching helps overlap some latency.

We use a timeline to visualize this overlap. In Fig. 6, green blocks show prefetching that is overlapped, and orange blocks show page fault-induced loading that is not overlapped. In Fig. 6a, with a fast disk, disk loading is fully overlapped. In Figs. 6b and 6c, with a device that has a slow disk, only part of the disk loading is overlapped, while other devices are fully overlapped. In Figs. 6c, 6d, and 6e, although disk loading is not fully hidden, PRP significantly reduces TPOT compared with vanilla PP. In Fig. 6e, while prefetching is used, it exceeds memory limits and triggers prefetch-release, where the OS releases earlier prefetched layers as new ones are loaded, adding disk I/O cost without benefit. This underscores the need to combine PRP with prefetching for higher efficiency.

- A.3 LAYER-TO-DEVICE ASSIGNMENT: FROM LATENCY ANALYSIS TO VECTORIZED MODEL

Assume there are M devices, where the layer window size for device dm is wm. On device dm, the number of GPU layers nm is defined as follows: within a layer window of size wm, nm layers run on the GPU, while the remaining wm − nm layers run on the CPU (wm and nm can vary across devices). Our objective is to find a vector w = {w1,··· ,wM} and a vector n = {n1,··· ,nM} to minimize the TPOT T, which is the sum of latencies from computation Tmcomp, memory access Tmmem, disk loading Tmdisk, and communication Tmcomm on each device.

M

Tmcomp + Tmmem + Tmdisk + Tmcomm . (11)

T =

m=1

Here, we minimize T = Mm=1 Tm instead of T = max{Tm}, because Fig. 6f is an idealized illustration. In practice, the OS does not start prefetching immediately after computation ends, and the

timing is unknown. As a result, device 4 may experience more bubbles and higher page fault-induced latency than expected. This uncertainty prevents us from solving T = max{Tm} before deployment (which is also hard to measure), and historical data is useless due to fluctuating device conditions. Thus, we take a worst-case approach, assuming the OS has not started prefetching when computation begins, leading to our objective T = Mm=1 Tm. Next, we analyze these latencies in detail.

Estimation of computation latency Tmcomp. The computation latency on device dm is defined

- as the time taken to process lm model layers and the output layer (if dm is the head device), where lmgpu layers run on the GPU, and the remaining lm − lmgpu layers and output layer run on

the CPU. Here, we have lm = W L wm + min(wm,max(0,R − mj=1−1 min(wj,R))), lmgpu = L W nm + min(nm,max(0,R − mj=1−1 min(wj,R))), where nm ≤ wm, W = Mm=1 wm, and

Init Round 3

Round 1 Round 2 Round 1 Round 2 Round 3

Init

- Device 1
- Device 2
- Device 3
- Device 4

(a) Pipelined-ring parallelism: Homogeneous devices with fast disks (b) Pipelined-ring parallelism: Homogeneous devices with slow disks

Round 1 Round 2 Round 3 Round 4

Init

- Device 1
- Device 2*
- Device 3
- Device 4

(c) Pipelined-ring parallelism: Heterogeneous devices with a slow disk on device 2

Round 1~3

Init

- Device 1
- Device 2*
- Device 3
- Device 4 (d) Vanilla pipeline parallelism (w/o prefetching): Heterogeneous devices with a slow disk on device 2

Round 1~3

Init

- Device 1
- Device 2*
- Device 3
- Device 4

Prefetch-Release Prefetch-Release Prefetch-Release

(e) Vanilla pipeline parallelism (with prefetching): Heterogeneous devices with a slow disk on device 2

Init Round 1 Round 2 Round 3 Round 4 Round 5 Round 6

- Device 1
- Device 2*
- Device 3
- Device 4

(f) Pipelined-ring parallelism: Heterogeneous devices with different layer window sizes

| |
|---|

| |
|---|

Page Fault Loading

Computing

Prefetching

Device Idle

Figure 6: Timeline of (a,b) PRP on homogeneous devices with fast/slow disks; (c,e) PRP on heterogeneous devices with the same/different window sizes; and (d,e) vanilla PP on heterogeneous devices with/without prefetching.

R = L mod W. Since the input layer uses a lookup table, it does not contribute to computation latency.

To estimate the computation time, we develop a model profiler to count the floating-point operations (FLOPs) for each model layer and a device profiler to measure the floating-point throughput (FLOPS) of each device. Taking Q4K as an example, the model weights are primarily quantized in the Q4K format, though some weights use other formats. Specifically, we consider Q ={Q4_K, Q5_K, Q6_K, Q8_0, FP16, FP32} and three types of backends: CPU, CUDA, and Metal. The FLOPs for

each layer Fm and output layer Fmout consist of 6 values: Fm = {fmq4k,fmq5k,fmq6k,fmq80,fmfp16,fmfp32}, Fmout = {fm,q4kout,fm,q5kout,fm,q6kout,fm,q80out,fm,fp16out,fm,fp32out}, each representing the FLOPs under a specific quantization format. The FLOPS Sm consists of 3 sets: {Smcpu,Smcuda,Smmetal}, with each set consisting of 6 values (e.g., for CPU, Smcpu = {scpum ,q4k,scpum ,q5k,scpum ,q6k,scpum ,q80,scpum ,fp16,scpum ,fp32}) representing the floating-point throughput for a specific backend-quantization pair. Fm, Fmout, and Sm can be easily extended for more backends and quantization formats.

With these profilers, we can estimate the computation time as follows:

fm,q out scpum ,q

fmq sgpum ,q

fmq scpum ,q

+ lmgpu

Tmcomp = (lm − lmgpu)

. (12)

+ Im=1 ·

q∈Q

q∈Q

q∈Q

Here, sgpum ,q refers to GPU FLOPS. If prima.cpp is compiled with CUDA support, sgpum ,q corresponds to scudam ,q. If it runs on an Apple device with Metal enabled, sgpum ,q corresponds to smetalm ,q. In our implementation, the output layer is executed on the CPU by the master node (m = 1).

Estimation of memory access latency Tmmem. This latency consists of three components: (a) KV cache copy time Tmkv_cpy: the time taken to copy the new token’s cache to the KV cache storage on device m; (b) device copy time Tmdev_cpy: the time taken to copy hidden states between the CPU and the GPU (e.g., CUDA and Metal); (c) device loading time Tmdev_load: the time taken to load data from RAM or VRAM into the processing cores of the CPU or GPU.

For the KV cache copy time, in each token step, new key and value caches are generated with dimensions (hkek,1) and (hvev,1), respectively. Here, hk and hv are the number of attention heads for the key and value caches, ek and ev are the embedding size per head for the key and value vectors, respectively. Thus, for generating one token, each layer needs to copy hkek + hvev values to the KV cache storage. If values are stored in FP16, each takes 2 bytes, so the total number of bytes to be copied is 2(hkek + hvev) bytes. In the device profiler module, we measure the time of copying 2(hkek + hvev) bytes on CPU, CUDA, and Metal to obtain tkv_cpy,cpum and tkv_cpy,gpum . Then, the KV cache copy time Tmkv_cpy can be estimated by (lm − lmgpu)tkv_cpy,cpum + lmgputkv_cpy,gpum .

For the device copy time, this latency arises when the GPU is enabled, as it involves copying the input from RAM to VRAM and then copying the output from VRAM back to RAM. Both input and output have shape (e,1), where e is the embedding size. These values are typically stored in the FP32 format. In the device profiler module, we measure the latency for two operations: the time taken to copy 4e bytes of data from RAM to VRAM, denoted as tram-vramm , and the time taken to copy 4e bytes of data from VRAM to RAM, denoted as tvram-ramm . For a sequence of layers within a window, one RAM-to-VRAM copy and one VRAM-to-RAM copy are needed, so the device copy time for one window is tram-vramm + tvram-ramm . For device dm, it was assigned Wm = W L + min(1,max(0,R − mj=1−1 min(wj,R))) windows. Thus, the device copy time for dm is Tmdev_cpy = Wm(tram-vramm +tvram-ramm )(1−IUMAm ), where IUMAm = 1 indicates that device dm uses a unified memory architecture (UMA, e.g., Apple M-series) and the CPU and GPU share memory, so no explicit RAM-VRAM copy is needed.

For the device loading time, processing cores must load data from RAM/VRAM into registers before executing instructions, which incurs latency. However, the theoretical RAM/VRAM bandwidth does not determine this latency because applications often fail to fully utilize the bandwidth, and multi-level caching also has a significant influence. To capture these effects, our device profiler implements an operator to read data from RAM/VRAM into registers. By measuring its latency

- at data volumes similar to the tensor sizes, we obtain practical throughputs {Tmcpu,Tmcuda,Tmmetal}. Next, we calculate the data volume that needs to be loaded into registers during each token step, which typically consists of the weight data and the KV cache. In the model profiler, we record the total bytes of weight data for the input and output layers as bi,bo, and for each layer as b. Additionally, the KV cache size for each layer is 2(hkek + hvev)nkv, where nkv is the number of tokens for which the cache is stored. Then, the device loading time Tmdev_load can be expressed as

gpu m

gpu m

Tmcpu · Im=1, where Tmgpu depends on the hardware: it equals Tmmetal for Metal and Tmcuda for CUDA, and V is the vocabulary size. Now we can combine the three latency components and give the formal definition of the memory access latency Tmmem:

Tmdev_load = (lm−l

Tmcpu + l

Tmgpu )(b + 2(hkek + hvev)nkv) + b

i/V +bo

Tmmem =(lm − lmgpu)tkv_cpy,cpum + lmgputkv_cpy,gpum + Wm(tram-vramm + tvram-ramm )(1 − IUMAm ) (13)

lmgpu Tmgpu

lm − lmgpu Tmcpu

bi/V + bo Tmcpu

· Im=1. (14)

+

)(b + 2(hkek + hvev)nkv) +

+ (

Estimation of disk loading latency Tmdisk. Prima.cpp is designed to run on memory-constrained devices, so it cannot load the entire model into RAM. To address this, prima.cpp uses mmap to manage model weights. By using mmap, model weights are loaded into memory from disk only when needed for computation, and the OS will release inactive memory-mapped (mmapped) pages when memory pressure is high. This prevents OOM risks but incurs significant disk I/O latency, because evicted pages must be reloaded from disk the next time they are needed. To estimate this disk loading latency, it is necessary to determine the data volume that mmap needs to reload in each token

step. This is a challenging task because different OSs exhibit very different memory management behaviors and high variability.

On macOS (without Metal) and Linux, the OS gradually reclaims memory. When memory pressure is moderate, i.e., when blio+2(hkek+hvev)nkv(lm−lmgpu)+ccpu > davailm , mmapped pages are released incrementally until the pressure is alleviated. As a result, some weight data remain in the page cache, and the amount of data that mmap needs to reload is max(blio + 2(hkek + hvev)nkv(lm − lmgpu) + ccpu − davailm ,0), where blio = (lm − lmgpu)b + (bi/V + bo) · Im=1, 2(hkek + hvev)nkv(lm − lmgpu) is the KV cache size, and ccpu is the compute buffer size. If CUDA is enabled on Linux, the model weights in private VRAM are locked by the CUDA driver, keeping them resident so no disk I/O occurs. Therefore, the disk loading latency for macOS (without Metal) and Linux can be estimated as

max blio + 2(hkek + hvev)nkv(lm − lmgpu) + ccpu − davailm ,bi/V sdiskm

Tm,diskmacOS(no Metal) = Tm,diskLinux =

.

(15) For Tm,diskmacOS(no Metal), lmgpu = 0, and sdiskm is the random-read throughput of the disk. On Linux, mmap is configured for sequential access, so sdiskm is now the sequential-read throughput of the disk.

When Metal is enabled on macOS, the behavior changes. Metal loads mmapped pages into shared memory, and the OS prioritizes retaining these pages. That is, the OS is more inclined to swap out or compress active pages while keeping mmapped model weight pages in shared memory intact. However, when memory is exhausted (with free and inactive pages exhausted, the compression pool nearing saturation, and heavy swap usage), macOS will release these mmapped pages more aggressively. This may cause the entire model weights to be repeatedly reloaded and released. As a result, when the required memory exceeds the total available memory, i.e., when lmb + (bi/V + bo) · Im=1+2(hkek+hvev)nkvlm+ccpu+cgpu > davailm,metal, device dm needs to reload lmb+(bi/V +bo)· Im=1 bytes in each token step. Here, davailm,metal denotes the maximum working set size recommended by Metal. By measuring the random-read throughput sdiskm of the disk, we can calculate the disk loading latency for macOS (with Metal) as:

lmb + (bi/V + bo) · Im=1 sdiskm · I lmb + (bi/V + bo) · Im=1

Tm,diskmacOS (with Metal) =max

bi sdiskm V

+ 2(hkek + hvev)nkvlm + ccpu + cgpu − davailm,metal ,

. (16)

When running on Android devices, the OS prioritizes swapping out inactive pages to disk, such as memory used by background applications, to ensure that the active application runs smoothly. As a result, the available RAM for prima.cpp can be higher than expected because the OS swaps cold pages to disk, freeing some memory. Thus, on Android, the number of bytes that mmap needs to

reload is max(bbio + 2(hkek + hvev)nkv(lm − lmgpu) + ccpu − davailm − dswapoutm ,0), where dswapoutm = min(max(0,bbio + 2(hkek + hvev)nkv(lm − lmgpu) + ccpu − davailm ),min(dbytes_can_swapm ,dswap_availm )) represents the data bytes that are swapped to disk, dbytes_can_swapm is the data bytes of currently used memory that can be swapped out, and dswap_availm is the total available swap space. Then we have:

max(bbio + 2(hkek + hvev)nkv(lm − lmgpu) + ccpu − davailm − dswapoutm ,bi/V ) sdiskm

Tm,diskAndroid =

. (17)

By aggregating them, we obtain a unified expression compatible with cross-platform devices:

Tmdisk =Tm,diskmacOS (no Metal) · ImacOS (no Metal) + Tm,diskmacOS (with Metal) · ImacOS (with Metal)

+ Tm,diskLinux · ILinux + Tm,diskAndroid · IAndroid, (18)

where ImacOS (no Metal),ImacOS (with Metal),ILinux,IAndroid are indicator functions. This expression can be easily extended to include new OSs.

Estimation of network communication latency Tmcomm. In prima.cpp, devices are connected in a ring, and each device receives inputs from its predecessor, processes them, and forwards the outputs to the next device. After a device completes the computation for one layer window, it transmits the result (e values in the FP32 format, totaling 4e bytes) to the next device for further computation in the next

layer window. During each token step, the number of network communications on device dm equals the number of layer windows, which is Wm = W L + min(1,max(0,R − mj=1−1 min(wj,R))). By measuring the latency tcommm of transmitting 4e bytes between adjacent devices, we can estimate the network communication latency on device dm as:

Tmcomm =

L W

+ min(1,max(0,R −

m−1

min(wj,R))) tcommm . (19)

j=1

By aggregating these latencies, our objective becomes:

M

fm,q out scpum ,q

fmq sgpum ,q

fmq scpum ,q

+ lmgpu

(lm − lmgpu)

+ Im=1 ·

T =

m=1

q∈Q

q∈Q

q∈Q

+ (lm − lmgpu)tkv_cpy,cpum + lmgputkv_cpy,gpum + Wm(tram-vramm + tvram-ramm )(1 − IUMAm )

lm − lmgpu Tmcpu

lmgpu Tmgpu

bi/V + bo Tmcpu

· Im=1

+

(b + 2(hkek + hvev)nkv) +

+

max(lmb + (bi/V + bo) · Im=1 + 2(hkek + hvev)nkvlm + ccpu − davailm ,bi/V ) sdiskm · ImacOS (no Metal)

+

lmb + (bi/V + bo) · Im=1

sdiskm · I lmb + (bi/V + bo) · Im=1 + 2(hkek + hvev)nkvlm + ccpu + cgpu − davailm,metal , bi

+ max

sdiskm V · ImacOS (with Metal)

max((lm − lmgpu)b + (bi/V + bo) · Im=1 + 2(hkek + hvev)nkv(lm − lmgpu) + ccpu − davailm ,bi/V ) sdiskm · ILinux

+

max((lm − lmgpu)b + (bi/V + bo) · Im=1 + 2(hkek + hvev)nkv(lm − lmgpu) + ccpu − davailm − dswapoutm ,bi/V ) sdiskm · IAndroid

+

m−1

L W

min(wj,R))) tcommm

+ min(1,max(0,R −

+

j=1

M

fm,q out scpum ,q

fmq scpum ,q

bi/V + bo Tmcpu

b + 2(hkek + hvev)nkv Tmcpu

· Im=1 + (lm − lmgpu)

+ tkv_cpy,cpum +

+

=

m=1 q∈Q

q∈Q

fmq sgpum ,q

b + 2(hkek + hvev)nkv Tmgpu

+ lmgpu

+ tkv_cpy,gpum +

+ Wm (tram-vramm + tvram-ramm )(1 − IUMAm ) + tcommm

q∈Q

max(lmb + (bi/V + bo) · Im=1 + 2(hkek + hvev)nkvlm + ccpu − davailm ,bi/V ) sdiskm · ImacOS (no Metal)

+

lmb + (bi/V + bo) · Im=1

sdiskm · I lmb + (bi/V + bo) · Im=1 + 2(hkek + hvev)nkvlm + ccpu + cgpu − davailm,metal , bi

+ max

sdiskm V · ImacOS (with Metal)

(lm − lmgpu) b + 2(hkek + hvev)nkv sdiskm

(bi/V + bo) · Im=1 + ccpu − davailm − dswapoutm · IAndroid sdiskm

+

,

+ max

bi sdiskm V · (ILinux + IAndroid)

To remove the max operator, we decompose the disk loading latency into multiple terms, separately accounting for whether memory is sufficient or not. Let M be the set of all devices, and M1,M2,M3,M4 be the subsets of devices that satisfy the respective conditions in Cases 1-4, where M1 ∩ M2 ∩ M3 ∩ M4 = ∅ and M1 ∪ M2 ∪ M3 ∪ M4 = M.

- Case 1 (macOS with Metal disabled and insufficient RAM): If lmb + (bi/V + bo) ·

Im=1 + 2(hkek + hvev)nkvlm + ccpu > davailm and sdiskm > sdiskthreshold, then Tmdisk = lmb+(bi/V +bo)·Im=1+2(hkek+hvev)nkvlm+ccpu−davailm

sdiskm ,m ∈ M1.

- Case 2 (macOS with Metal enabled and insufficient RAM): If lmb + (bi/V + bo) · Im=1 + 2(hkek +

hvev)nkvlm + ccpu + cgpu > davailm,metal and sdiskm > sdiskthreshold, then Tmdisk = l

mb+(bi/V +bo)·Im=1

sdiskm ,m ∈ M2.

- Case 3 (Linux and Android with insufficient RAM): If (lm −lmgpu) b+2(hkek +hvev)nkv +(bi/V + bo) · Im=1 + ccpu > davailm + dswapoutm · IAndroid and sdiskm > sdiskthreshold, then Tmdisk = s1

disk m

(lm − lmgpu)[b + 2(hkek + hvev)nkv] + (bi/V + bo) · Im=1 + ccpu − davailm − dswapoutm · IAndroid ,m ∈ M3.

- Case 4 (OS with sufficient RAM or low disk speed): In these cases, the physical RAM is large enough to hold the model weights or the disk speed is slow (i.e., sdiskm < sdiskthreshold). As a result, no disk loading

is expected, except for the latency incurred during lookup table access, thus Tmdisk = b

#### sdiskm V ,m ∈ M4.

i

With these cases, we can rewrite the objective function as follows: T =

f1q,out scpu1 ,q

bi/V + bo T1cpu

bi/V sdisk1

bo sdisk1 · I1∈M/

(20)

+

+

+

4

q∈Q

fmq scpum ,q

b + 2(hkek + hvev)nkv Tmcpu

+ tkv_cpy,cpum +

(lm − lmgpu)

#### +

m∈M

q∈Q

fmq sgpum ,q

b + 2(hkek + hvev)nkv Tmgpu

+ lmgpu

+ tkv_cpy,gpum +

+ Wm (tram-vramm + tvram-ramm )(1 − IUMAm ) + tcommm

q∈Q

ccpu − davailm − dswapoutm · IAndroid sdiskm

(lm − lmgpu)[b + 2(hkek + hvev)nkv] sdiskm

lmb sdiskm

+

+

#### +

m∈M1∪M3

m∈M2

To further simplify the objective function, we make the following assumption. Assumption 1. Let WL be an integer (i.e., R = 0), where W = m∈M wm. Then all devices are assigned an equal number of windows, and all windows are filled. Thus, we have lm = w

mL W , lmgpu = n

mL W , Wm = WL . Let b′ = b + 2(hkek + hvev)nkv, αm =

fmq

q m

fmq

scpum,q +tkv_cpy,cpum + b

scpum,q +tkv_cpy,gpum −tkv_cpy,cpum + b

′ Tmcpu, ξm = (tram-vramm +tvram-ramm )(1−IUMAm )+tcommm , κ = q∈Q f

′

′

Tmcpu, βm = q∈Q f

Tmgpu − b

sgpum,q − q∈Q

q∈Q

q 1,out

scpu1 ,q + b

i/V +bo T1cpu

+ b

i/V sdisk1 + b

sdisk1 ·I1∈M/

#### +

o

4

ccpu−davailm −dswapoutm ·IAndroid

sdiskm , where αm,βm,ξm are platform-specific constants and κ is a global constant. Then, we add the first general term to the three platform-specific terms and obtain: T =

m∈M1∪M3

b′ sdiskm

L W m∈M

b sdiskm

L W m∈M

(αm +

(αm +

)wm + ξm +

)wm + βmnm + ξm

1

2

b′ sdiskm

b′ sdiskm

L W m∈M

L W m∈M

)wm + (βm −

(αm +

)nm + ξm +

#### αmwm + βmnm + ξm + κ.

+

3

4

This objective is a sum over device sets M1,M2,M3. Each summand involves expressions linear in wm and nm, plus platform-specific constant terms. To clarify the form, we define a linear function f(a,b,c) = awm+bnm+c, where the platform-specific constants a,b,c are independent of decision variables wm,nm. Consequently, the objective can be rearranged to a combination of linear functions:

b′ sdiskm

L W m∈M

b sdiskm

T =

f(αm +

,0,ξm) +

f(αm +

,βm,ξm)

m∈M2

1

b′ sdiskm

b′ sdiskm

,βm −

+

f(αm +

,ξm) +

f(αm,βm,ξm) + κ

m∈M3

m∈M4

Note that the objective T is nonlinear because W = m∈M wm depends on the decision variables, and the term W1 introduces nonlinearity. Now, we put everything together:

T (21)

min

wm,nm

s.t. wm ∈ Z>0,nm ∈ Z≥0,nm ≤ wm ≤ L, (22) L = kW,k ∈ Z>0, (23) W =

wm, (24)

m∈M

f(a,b,c) = awm + bnm + c, (25)

4

4

Mi = M, (26)

Mi = ∅,

i=1

i=1

W Lb′ (davailm − bciom ),m ∈ M1, (27)

wm >

W Lb′ (davailm,metal − bciom − cgpu),m ∈ M2, (28)

wm >

W Lb′(davailm + dswapoutm · IAndroid − bciom ),m ∈ M3, (29)

wm − nm >

#### W Lb′(davailm − bciom ),m ∈ M4, (30)

wm · ImacOS (no Metal) <

W Lb′ (davailm,metal − bciom − cgpu),m ∈ M4, (31)

wm · ImacOS (with Metal) <

W Lb′(davailm + dswapoutm · IAndroid − bciom ),m ∈ M4, (32)

(wm − nm)(ILinux + IAndroid) <

bciom = (bi/V + bo) · Im=1 + ccpu, (33) nm · Icuda ≤

W Lb′ (davailm,cuda − cgpu) · Icuda, (34)

#### W Lb′ (davailm,metal − cgpu − bo · Im=1) · Imetal, (35)

nm · Imetal ≤

nm = 0,if Icuda = 0 and Imetal = 0. (36) Constraint (22) requires that the window size wm must be a positive integer, the number of GPU layers nm must be a non-negative integer, and nm cannot exceed wm. Constraint (23) requires that all devices be assigned an equal number of windows and all windows be filled. Constraints (27-29) ensure that devices categorized into sets M1,M2,M3 meet the memory condition outlined in Cases 1-3. Similarly, Constraints (30-32) ensure that devices assigned to set M4 meet the memory condition outlined in Case 4. bciom in Eq. (33) is a platform-independent constant. Constraints (34-35) ensure that the VRAM used by CUDA or the shared memory used by Metal does not exceed the available capacity. Here, davailm,cuda denotes the available GPU private memory for CUDA, and davailm,metal denotes the maximum working set size recommended by Metal.

This is an integer linear fractional program (ILFP) because the numerator is a linear function of the decision variables wm,nw, and the denominator W is also a linear function of wm. Moreover, the constraints are linear inequalities. The platform indicators ImacOS,ILinux,IAndroid,Icuda,Imetal,Im=1 are known a priori, and they activate/deactivate corresponding linear constraints for each device.

Next, we transform the model into a vectorized form. Let the decision variables be wT = [w1,w2,··· ,wM], nT = [n1,n2,··· ,nM], and the coefficients a,b,c be:









′

αm + b





#### sdiskm | m ∈ M1 αm + sb

0 | m ∈ M1 βm | m ∈ M2 βm − b

ξm | m ∈ M1 ξm | m ∈ M2 ξm | m ∈ M3 ξm | m ∈ M4

| m ∈ M2

disk m

, c =

, b =

.

a =

′

 

 

sdiskm | m ∈ M3 βm | m ∈ M4

′

 

 

αm + b

sdiskm | m ∈ M3 αm | m ∈ M4

 

 

To apply constraints to the subset of w and n corresponding to M1,M2,M3,M4, we define diagonal matrices Pw = diag(−IM

), Pn = diag(0M

#### ,PM1

#### ,PM2

#### ,PM3

#### ,−IM

#### ,−IM

1

2

3

4

4

4

are identity matrices and 0M

), where IM

#### ,01M

#### ,02M

#### ,−P3M

#### ,0M

#### ,IM

#### ,IM

#### ,IM

1

2

3

1

2

3

4

4

4

are zero matrices corresponding to the subsets M1,M2,M3, and PM1

#### ,0M

#### ,0M

1

2

3

are diagonal binary matricies (i.e., selection matricies) corresponding to the three constraints (30-32) within the subset M4. To construct PM1

#### ,PM2

#### ,PM3

4

4

4

, we define a binary vector pmacOS, where a value of 1 indicates that the current device is running on macOS and a value of 0 indicates otherwise. The number of elements in pmacOS matches the number of devices in the set M4. Similarly, we define binary vectors pLinux, pAndroid, pmetal. Thus, we have PM1

#### ,PM2

#### ,PM3

4

4

4

= diag(pmacOS ⊙ (1 − pmetal)), PM2

= diag(pmacOS ⊙ pmetal), PM3

= pLinux + pAndroid.

4

4

4

To handle constraints (34-35), we define Pngpu as a similar diagonal binary matrix, with elements set to one for devices with CUDA or Metal support. Specifically, we let Pngpu = Pncuda + Pnmetal, where Pncuda = diag(0M

,PMcuda

,PMcuda

) and Pnmetal = diag(0M

,PMmetal

). Let the decision variables be wMT

#### ,0M

#### ,IM

#### ,0M

1

2

1

2

3

3

4

4

= [nm | m ∈ M4], w′T = [wT,wMT

= [wm | m ∈ M4], nTM

4

4

], n′T = [nT,nTM

], the RAM upper bound be

,wMT

,nTM

4

4

4

4





davailm − bciom | m ∈ M1 davailm,metal − bciom − cgpu | m ∈ M2 davailm + dswapoutm · IAndroid − bciom | m ∈ M3 −davailm + bciom | m ∈ M4 −davailm,metal + bciom + cgpu | m ∈ M4 −davailm − dswapoutm · IAndroid + bciom | m ∈ M4

1 Lb′

,

z =

 

 

and the VRAM/shared-memory upper bound be zgpu = [z1gpu,··· ,zMgpu], where

 

0, if Icuda = 0 and Imetal = 0, davailm,cuda − cgpu, if Icuda = 1, davailm,metal − cgpu, if Imetal = 1 and m ̸= 1, davailm,metal − cgpu − bo, if Imetal = 1 and m = 1.

1 Lb′ ·

zmgpu =



The problem model can then be reformatted as:

aT · w + bT · n + eT · c eT · w

+ κ, (37)

L ·

min

w,n

s.t. wm ∈ Z>0,nm ∈ Z≥0,nm ≤ wm ≤ L, (38) L − k(eT · w) = 0,k ∈ Z>0, (39) Pw · w′ + Pn · n′ + eT · w · z < 0, (40)

− Pngpu · zgpu · eT · w + Pngpu · n ≤ 0. (41) Table 6 summarizes the key symbols used in this paper.

- A.4 GENERALIZATION TO A NEW HETEROGENEOUS TESTBED

To show the generalizability of prima.cpp, we repeated the experiments of Table 4 on another testbed. This testbed includes: a host PC (5 GB RAM, 1080 TI GPU with 11 GB VRAM), a Mac Mini (10 GB UMA RAM), a laptop (23 GB RAM, 3060 GPU with 6 GB VRAM), and a Redmi phone (7 GB RAM). Here, all RAM/VRAM values refer to available memory rather than total capacity. We ran llama.cpp on the laptop (with a 3060 GPU), and exo on the host PC, Mac Mini, and laptop (since exo is not supported on the phone). Meanwhile, dllama and prima.cpp ran on all four devices. Their TPOT

Table 6: Summary of key symbols and their explanations.

Symbol Explanation

|M|Number of devices.<br><br>|
|---|---|
|wm<br><br>|Layer window size on device dm.|
|nm<br><br>|Number of GPU layers on device dm.|
|T|The optimization objective (i.e., TPOT).<br><br>|
|lm<br><br>|Total model layers processed by device dm.|
|lmgpu|Total GPU layers processed by device dm.|
|L|Total number of model layers.|
|W<br><br>|Total layer window size across all devices (W = Mm=1 wm).|
|hk, hv<br><br>|Number of attention heads for keys and values.|
|ek, ev|Embedding size per attention head.<br><br>|
|e<br><br>|Embedding size.|
|b, bi, bo<br><br>|Bytes of weight tensors per layer, and of input/output tensors.|
|nkv|Number of tokens stored in the KV cache.<br><br>|
|V<br><br>|Vocabulary size.|
|davailm|Available memory on device dm.<br><br>|
|ccpu, cgpu|Buffer sizes for CPU/GPU computations.|
|sdiskm<br><br>|Disk read throughput for device dm.|
|sdiskthreshold<br><br>|A threshold for disk read throughput. If the throughput is below this threshold, the disk is considered slow.|
|M1, M2, M3, M4|Set assignments, corresponding to cases 1-4.|
|a, b, c|Coefficient vectors for the objective function.|
|Pw, Pn|Diagonal binary selection matrices for constraints on w and n.<br><br>|
|Pngpu<br><br>|Diagonal binary matrix that indicates whether a device uses a GPU.|
|w′, n′<br><br>|Extended vectors for w and n.|
|z, zgpu|Vectors of RAM/VRAM upper bounds for constraints.|

Table 7: TPOT (ms/token) on the new testbed.

Model llama.cpp exo dllama prima.cpp

Llama 3-8B 27 OOM 875 27 Llama 3-14B 199 - - 67 Llama 1-30B 469 - - 308 Llama 3-45B 623 - - 328 Llama 3-60B 12762 - - 671 Llama 1-65B 20073 - - 703 Llama 3-70B 23834 OOM OOM 718

values are listed in Table 7, following a trend similar to Table 4. In this case, llama.cpp encounters a VRAM bottleneck at 14B, and prima.cpp matches (only at 8B) or achieves the fastest speeds across 8-70B. Additionally, due to the limited RAM of the host PC, exo ran out of memory while loading the weight files. This supplementary experiment supports the generalizability of prima.cpp.

- A.5 GENERALIZATION TO A HOMOGENEOUS TESTBED

We also compared llama.cpp, exo, dllama, and prima.cpp on a homogeneous, low-resource cluster. Since we do not have identical home devices, we used Docker to create four identical Linux containers on a server with an RTX 4090. Each container had 8 CPU cores, 8 GiB RAM (4 GiB available), 5 GiB VRAM, and 2 GB/s disk read throughput. Table 8 reports their TPOT results.

Prima.cpp achieves substantial speedups over llama.cpp in this low-resource testbed, delivering more than 100× improvement on the 14B model. Exo, however, crashes even on the 8B model due to its full-precision GPU backend, making it unsuitable for low-end devices. Although exo’s TPOT at 8B cannot be measured directly, we can infer its lower bound based on the performance of llama.cpp. For example, llama.cpp runs the full 8B (Q4K) model on 5 GiB of VRAM at 15 ms/token, implying that exo’s TPOT would be at least this high. Furthermore, prima.cpp matches the speed of llama.cpp at this scale, suggesting that it is at least as fast as exo. Finally, despite the near-zero communication

Table 8: TPOT (ms/token) on a homogeneous testbed.

Model llama.cpp exo dllama prima.cpp

Llama 3-8B 15 OOM 81 15 Llama 3-14B 2243 - - 21 Llama 1-30B 6870 - - 52 Llama 3-45B 10563 - - 195 Llama 3-60B 14652 - - 391 Llama 1-65B 15798 - - 502 Llama 3-70B 17590 OOM OOM 1128

delay from the Docker bridge, dllama remains slow, indicating that TP can still be inefficient even on homogeneous, high-bandwidth clusters.

- A.6 RUN PRIMA.CPP ON LLAMA 1&3, QWEN 2.5, QWQ AND DEEPSEEK R1

Fig. 7 plots TPOT and TTFT for Llama models as curves. The prima.cpp curves (solid and dashed diamonds) consistently lie at the bottom, confirming that prima.cpp achieves the lowest TPOT and TTFT across 8-70B. Disabling prefetching or Halda increases latency, but the performance drop is substantially larger when Halda is disabled. This suggests that Halda contributes more critically to the speedup than prefetching.

TPOT - llama.cpp

- 102

- 103

- 104

- 102

- 103

- 104 TTFT(ms)

TPOT - exo

TPOT - dllama

TPOT - prima.cpp

TPOT - prima.cpp (w/o halda)

TPOT - prima.cpp (w/o prefetch)

TPOT(ms/token)

| |
|---|

TTFT - llama.cpp

TTFT - exo

TTFT - dllama

TTFT - prima.cpp

Llama3-8BLlama3-14BLlama1-30BLlama3-45BLlama3-60BLlama1-65BLlama3-70B

Figure 7: TPOT and TTFT for Llama models across 8-70B.

Table 9: TPOT (ms/token) for Qwen 2.5, QwQ, and DeepSeek R1 models across 7-72B.

Model llama.cpp exo dllama prima.cpp

Qwen-2.5-7B 14 86 - 14 DeepSeek-R1-Distill-Qwen-7B 14 681 - 14 DeepSeek-R1-Distill-Llama-8B 14 771 435 14 Qwen-2.5-14B 23 31710 - 23 DeepSeek-R1-Distill-Qwen-14B 24 23475 - 24 Qwen-2.5-32B and QwQ-32B 224 OOM - 89 DeepSeek-R1-Distill-Qwen-32B 232 OOM - 93 DeepSeek-R1-Distill-Llama-70B 10978 OOM - 724 Qwen-2.5-72B 12227 OOM - 867

1 TPOT is lower because exo provides full-precision Qwen models, whereas the DeepSeek-distilled models are quantized to 3-bit.

- Table 9 extends the evaluation to more models, including Qwen 2.5, QwQ, and DeepSeek R1 (distilled versions) across 7-72B. The results are consistent with Table 4. For small models (<14B), Halda

places prima.cpp on D3, so TPOT and TTFT match those of llama.cpp, and 31× faster than dllama. For larger models, Halda distributes the workload across GPUs and CPUs in a smart way, enabling prima.cpp to maintain sub-second TPOT even at 70B, whereas llama.cpp can only offload locally, triggering disk offloading much earlier and causing TPOT to explode.

A special case arises for exo, which only provides an MLX backend for these models. Among the devices in Table 3, only Mac M1 (D1) can run it, so exo executes solely on D1. For small models, since D1 is less powerful than D3, exo is 6× slower than prima.cpp. At 14B, D1 runs out of memory, and TPOT spikes due to disk swapping. As the model size increases further, exo fails with OOM. This highlights that pinning model weights in memory can induce uncontrolled swapping, whose overhead is often much more severe than proactively managing disk offloading via mmap.

- A.7 SELECT DEVICES TO BUILD THE MOST POWERFUL CLUSTER

Existing systems require clusters with sufficient aggregated RAM/VRAM, often forcing users to add more devices to support larger models. However, assembling enough devices is difficult for households. This raises two questions: (a) Should we collect enough devices to meet the model’s needs? (b) Do more devices always lead to better performance? The answer is no. Fig. 8 illustrates how the TPOT of prima.cpp on Llama 3-70B (Q4K) is affected by the device set. For each device set, the layer partitioning across device-backend pairs is determined by Halda.

80

80

10000

D1-CPU D2-CPU D2-GPU

D3-CPU D3-GPU D4-CPU

D5-CPU D6-CPU TPOT

| |
|---|

| |
|---|

| |
|---|

70

8000

NumberofAssignedLayers

60

56

TPOT(ms/token)

50

6000

42

40

38 38 38

4000

30

25 25 25 25 24

23 23 23 23 23 23

20

16 16 16 16 16

2000

11 12 13 13

10

4 4 4

1 1

1

0

0

6 5 4 3 2 1 Number of Devices

Figure 8: Layer partitioning and TPOT over different device sets.

- For question (a), with only 3 devices (D2, D3, and D5), the aggregated RAM+VRAM is 38 GiB, which is insufficient to hold the 40 GiB Llama 3-70B (Q4K) model. However, thanks to the fast SSDs on D2 and D3, mmap can reload model layers quickly, and prima.cpp achieves the lowest latency. Thus, prima.cpp does not require memory sufficient to hold the entire model.
- For question (b), when we increase the number of devices to 6, the aggregated RAM+VRAM reaches 50 GiB, which is enough to hold the entire model. However, the TPOT is lower with just 3 devices, because the devices D4 and D6 have weak CPUs and slow disks, creating bottlenecks. This shows that more devices do not always result in faster inference.

This raises a new question: (c) If the user has a device with a weak CPU or a slow disk, should it be removed from the cluster? Intuitively, such a weak device would be a bottleneck. However, in cases of severe memory shortage, it does help. For example, D2 and D3 are devices with a GPU, a strong CPU, and a fast disk, while D5 is a weak device. As shown in Fig. 8, adding D5 reduced TPOT by roughly half because D3’s disk loading latency (which was heavily overloaded) dominated the impact of D5’s weaker CPU.

This raises more questions: if users have some weak devices, which ones should be removed? More generally: (d) Given a set of heterogeneous devices, how can we select a subset to build the bestperforming cluster? This is challenging due to the uncertain number of devices to be selected, the highly heterogeneous cluster, and the various factors like CPU, GPU, RAM, VRAM, disk, network,

and even OS that significantly affect inference speed. Fortunately, Halda offers an easy solution: include all available devices, then remove those with only one assigned layer, since Halda indicates that excluding them would improve speed. This procedure is made automatically in prima.cpp.

- A.8 MEMORY FOOTPRINT ON DEVICE-BACKEND PAIRS

Fig. 9 shows each device’s RAM and VRAM usage to illustrate why prima.cpp achieves faster speed and prevents OOM. As exo and dllama don’t support Llama 14B-65B and encounter OOM at 70B, the memory usage for 14B-70B in Figs. 9b and 9c is estimated based on system behavior and memory load at 8B. Fig. 9a was discussed in Section 4.1.

Figs. 9b and 9c show that exo and dllama consume high memory. Exo mixes multiple backends, using MLX on macOS for 4-bit computation and Tinygrad on Linux, where model weights are loaded in 16-bit on the CPU and decoded to 32-bit on the GPU. In our case, D1(8 GiB UMA RAM) and D2 (8 GiB VRAM) get the same number of model layers, yet D2-CPU uses 4× more RAM and D2-GPU 8× more VRAM than D1-GPU. This results in high memory usage on Linux devices, increasing the risk of OOM.

For dllama, it uses TP and Q40 quantization to distribute and compress memory usage, but lacks GPU support, so all memory load is on RAM, and inference speed is limited. It has similar memory usage across devices due to its uniform tensor splitting, which causes problems on low-memory devices. In Fig. 9c, for a 30B model, D2 and D3 have more available RAM, while D1 and D4 have less. To allocate enough memory, D1 and D4 must free more active pages or swap out application data, which can slow user apps or even crash the system. In such cases, OOM may be the safer outcome. Additionally, D3 (the head device) loads the entire model before slicing and distributing it, taking significant RAM and making it more prone to OOM.

In contrast, prima.cpp optimizes workload distribution with Halda and prevents OOMs with mmap. Although the solution to the problem Eqs. (1)-(5) is hard to understand, we can observe Halda’s preference from Fig. 9d: powerful GPUs > weak GPUs > powerful CPUs > fast disks. For example, at 8B-30B, Halda first fills D2-GPU and D3-GPU. At 45-65B, it fills D1-CPU to D4-CPU. Lastly, the remaining layers are placed on D2-CPU and D3-CPU because they have fast disks. This assignment prevents weak CPUs and slow disks from being used. Finally, only D2-CPU and D3-CPU experience RAM overload, but this does not cause OOM because the OS will free inactive mmapped pages instantly and prefetch model layers in advance. With fast disk reads, disk loading latency stays low, ensuring minimal TPOT, which is exactly the result of our optimization goal (Eq. 1).

Beyond the advanced workload distribution via Halda, prima.cpp also prevents memory waste. With mmap, it loads only the required model layers instead of the full model, eliminating the need for model slicing (which may cause OOMs like in dllama). Additionally, it supports model inference in Q4K format across heterogeneous platforms, eliminating the need to decode back to 16-bit or 32-bit, thereby further reducing RAM/VRAM usage.

- A.9 RUN PRIMA.CPP WITH SPECULATIVE DECODING

Prima.cpp can be further accelerated with speculative decoding. In prima.cpp, we add support for this technique. Since the draft model is small (0.5-3B), we run it as a standalone process on the head device and set the most powerful device as the head. The draft model predicts 5 tokens per step, which are then verified in batch by the larger target model. The testbed consists of 4 Linux devices, each with an 8-core CPU, 8 GiB RAM, and a sequential disk read throughput of 600 MB/s. Two nodes are equipped with a 4090 GPU, but each is limited to 11 GiB VRAM.

- Table 10 compares the TPOT of llama.cpp and prima.cpp with and without speculative decoding. With speculative decoding, prima.cpp delivers an additional 25-45% latency reduction across 14-70B models. For example, Qwen 2.5-32B speeds up from 18 to 26 tokens/s, and Llama 3.3-70B from 1.2 to 2.3 tokens/s. At this throughput, a 32B model meets the 20-50 tokens/s throughput commonly required by LLM agents, facilitating broader deployment of frontier LLM agents on home devices.

120

120

D3-CPU Bound D3-GPU Bound

D1-GPU D2-CPU D2-GPU D3-CPU D3-GPU

D1-GPU Bound D2-CPU Bound D2-GPU Bound D3-CPU Bound D3-GPU Bound

| |
|---|

D3-CPU D3-GPU

100

100

| |
|---|

| |
|---|

| |
|---|

MemoryUsage(GiB)

MemoryUsage(GiB)

80

80

60

60

40

40

20

20

0

0

Llama3-8BLlama3-14BLlama1-30BLlama3-45BLlama3-60BLlama1-65BLlama3-70B

Llama3-8BLlama3-14BLlama1-30BLlama3-70B

(a) llama.cpp

(b) exo

120

D1-CPU D2-CPU D3-CPU D4-CPU

D1-CPU Bound D2-CPU Bound D3-CPU Bound D4-CPU Bound

| |
|---|

100

| |
|---|

| |
|---|

MemoryUsage(GiB)

80

60

40

20

0

Llama3-8B Llama3-14B Llama1-30B Llama3-45B Llama3-60B Llama1-65B Llama3-70B

(c) dllama

120

D1-CPU D2-CPU D2-GPU D3-CPU

D3-GPU D4-CPU

D2-GPU Bound D3-CPU Bound D3-GPU Bound D4-CPU Bound

| |
|---|

| |
|---|

| |
|---|

100

D1-CPU Bound D2-CPU Bound

| |
|---|

| |
|---|

MemoryUsage(GiB)

80

60

40

20

0

Llama3-8B Llama3-14B Llama1-30B Llama3-45B Llama3-60B Llama1-65B Llama3-70B

(d) prima.cpp

Figure 9: Memory footprint on each device-backend pair.

Table 10: TPOT (ms/token) for llama.cpp and prima.cpp with and without speculative decoding.

Model llama.cpp prima.cpp prima.cpp (with speculative) Qwen-2.5-7B 20 ± 0 ms 20 ± 0 ms 18 ± 1 ms (draft 0.5B on GPU)

- Llama 3.2-8B 20 ± 0 ms 20 ± 0 ms 20 ± 1 ms (draft 1B on GPU) Qwen-2.5-14B 36 ± 1 ms 36 ± 0 ms 27 ± 1 ms (draft 1.5B on GPU) DeepSeek-R1-Distill-Qwen-14B 32 ± 1 ms 32 ± 0 ms 22 ± 1 ms (draft 1.5B on GPU) Qwen-2.5-32B 6551 ± 38 ms 55 ± 0 ms 38 ± 5 ms (draft 0.5B on GPU) DeepSeek-R1-Distill-Llama-70B 20118 ± 69 ms 859 ± 40 ms 593 ± 65 ms (draft 8B on CPU)
- Llama 3.3-70B 20083 ± 10 ms 803 ± 9 ms 442 ± 11 ms (draft 3B on GPU) Qwen-2.5-72B 21600 ± 80 ms 963 ± 18 ms 544 ± 19 ms (draft 3B on CPU)

- A.10 ABLATION STUDY OF HALDA WITH HEURISTIC SCHEDULER BASELINES

While prior work has studied layer partitioning across heterogeneous devices, these approaches often rely on strong and unrealistic assumptions that limit their use in real home clusters: (a) They only support workload scheduling within GPU clusters; (b) They require the cluster’s aggregated VRAM to meet the model’s needs; (c) They assume all devices are necessary. For (a) and (b), most households cannot afford an expensive machine with sufficient VRAM, let alone a GPU cluster. For (c), since home devices can vary widely, it could be better to drop weak devices than to keep them.

The novelty in Halda lies in supporting GPU/CPU-mixed clusters while relaxing memory requirements, and in explicitly accounting for disk latency and OS-specific memory behaviors. Moreover, Halda can automatically select the optimal subset of devices from a candidate pool to serve an inference engine that runs at maximum speed. We call this ability "device selection". No previous work has considered these features, but they are necessary in home clusters.

To better demonstrate Halda’s effectiveness and novelty, we conduct a supplementary experiment comparing it with two heuristic scheduler baselines:

- • MemSched: Partitions layers according to RAM/VRAM ratios. This method originates from exo and is also the default strategy in prima.cpp (w/o halda).
- • PerfSched: This method originates from Galaxy. First, partition layers by devices’ compute power, then migrate OOM layers to other devices. This can still hit OOM, so we add a greedy fallback: If OOM persists, offload to CPUs (in proportion to CPU compute power).

The testbed uses a Mac Mini (16GB RAM), a host PC (16GB RAM and 11GB 1080TI GPU), a laptop (32GB RAM and 6GB 3060 GPU), and a Redmi phone (16GB RAM). In this setup, aggregated VRAM is insufficient, but aggregated RAM+VRAM is sufficient, so OOM does not occur.

MemSched

2.5

PerfSched

TPOT(s/token)

2.0

Halda

1.5

1.0

0.5

0.0

8B 14B 30B 45B 60B 65B 70B Llama Models

Figure 10: TPOT of MemSched, PerfSched, and Halda on Llama models across 8-70B.

As shown in Fig. 10, on Llama 3-70B (Q4K), Halda runs 3.3× faster than MemSched and 1.6× faster than PerfSched because it learns to drop the 4th device for faster speed. We also tested models of smaller sizes, and Halda always outperforms heuristic schedulers. For 8-45B models, Halda is 1.1-1.5× faster than PerfSched, 4.5-8.8× faster than MemSched. For 60-70B models, 1.6-1.8x faster than PerfSched, 1.9-3.3× faster than MemSched. This demonstrates the effectiveness and novelty of the proposed Halda scheduler.

