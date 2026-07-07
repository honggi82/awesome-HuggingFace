## SIGMA: Differential Rescaling of Query, Key and Value for Efficient Language Models

# arXiv:2501.13629v2[cs.CL]10Feb2025

Zhenghao Lin♡1 Zihao Tang♡1 Xiao Liu♡1 Yeyun Gong♡1 Yi Cheng♢1 Qi Chen♢1 Hang Li♢1 Ying Xin♢1 Ziyue Yang♢1 Kailai Yang♣1 Yu Yan♣1 Xiao Liang♣1 Shuai Lu♣1 Yiming Huang♣1 Zheheng Luo♣1 Lei Qu♣1 Xuan Feng♣1 Yaoxiang Wang♣1 Yuqing Xia♣1 Feiyang Chen♣1 Yuting Jiang♣1 Yasen Hu♣1 Hao Ni♣1 Binyang Li♣1 Guoshuai Zhao♣1 Jui-Hao Chiang♣1 Zhongxin Guo♣1 Chen Lin♣1 Kun Kuang♣1 Wenjie Li♣1 Yelong Shen♣1 Jian Jiao♣1 Peng Cheng♠1 Mao Yang♠1

### Abstract

We introduce SIGMA, an efficient large language model specialized for the system domain, empowered DiffQKV attention, and pre-trained on selfcollected system domain data. Given the varing impacts on the model performance and efficienct indicators of Query (Q), Key (K), and Value (V), SIGMA use DiffQKV attention to optimize them differentially and significantly enhance inference efficiency. Specifically, we (1) differentially compress K and V, leveraging the model’s varying sensitivity to KV compression, as demonstrated in our extensive experiments, and (2) propose augmented Q to expand the dimension of the Q head, which improves the representation capacity of the model with minimal impacts on the inference speed. Rigorous theoretical and empirical analyses reveal that DiffQKV attention significantly enhances inference efficiency, achieving up to a 33.36% improvement over the conventional grouped-query attention (GQA) in longcontext scenarios. We pre-train SIGMA on 6T tokens from various sources, including 19.5B system domain data that we carefully collect and 1T tokens of synthesized and rewritten data. In general domains, SIGMA performs on par with state-of-the-art models. In the system domain, it excels on AIMICIUS, the first comprehensive system-domain benchmark we propose, surpassing GPT-4 by up to 52.5% across all tasks.

♡First Authors ♢Core Authors ♣Contributors ♠Leaders 1Microsoft SIGMA Team.

### 1. Introduction

In recent years, significant progress has been made in the development of large language models (LLMs), which have demonstrated remarkable performance across a wide range of domains (Bubeck et al., 2023; Jiang et al., 2023a; GLM et al., 2024; Dubey et al., 2024; Yang et al., 2024). Meanwhile, a novel research direction, known as the “system domain,” has emerged with promising potential to further accelerate AI development through automated optimization of AI infrastructure (Xiong et al., 2024; Shi et al., 2024; Hu et al., 2024b). This domain focuses on leveraging AI models to autonomously validate, evaluate, diagnose, and optimize the key components of AI infrastructure (e.g., hardware, configurations, cloud services, databases, and workloads). Despite its promise, however, this important area of research has yet to receive commensurate attention.

To bridge this gap, this paper introduces SIGMA, an efficient large language model specialized for the system domain, empowered by a novel architecture including DiffQKV attention and our carefully collected system domain data.

The DiffQKV attention substantially improves SIGMA’s inference efficiency by mitigating the bottleneck associated with KV cache (Pope et al., 2023).1 Despite significant efforts to address the KV cache issue (Ainslie et al., 2023; Kwon et al., 2023; Luohe et al., 2024; Zhang et al., 2024b), previous studies tend to treat the compression of K and V vectors uniformly and rarely take into account the optimization of Q. In contrast, DiffQKV attention differentially optimizes the Query (Q), Key (K), and Value (V) components in the attention mechanism with tailored strategies, based on their varying impacts on the model performance and

1KV cache is a common technique in the decoder-only Transformer architecture (Radford et al., 2019), which stores Key and Value vectors for future reuse in decoding. It can consume considerable GPU memory (Pope et al., 2023) and place substantial demands on memory bandwidth (Shazeer, 2019; Ribar et al., 2024).

efficiency indicators. Specifically, it involves two critical techniques: differentially compressed KV and augmented Q. Differentially compressed KV aggressively compresses K while lightly compressing V, grounded on our experimental findings that model performance is significantly more sensitive to compression in V vectors than in K, both in terms of dimension and the number of heads. Augmented Q adopts a higher dimension for the Q head compared to KV heads, as we discover that introducing extra parameters to Q can effectively boost the model performance. With minimal impacts on the inference speed, augmented Q can to some degree counteract the performance decline that inevitably results from KV compression. Rigorous theoretical and empirical analyses reveal that SIGMA’s DiffQKV significantly enhances efficiency, achieving up to a 33.36% improvement in inference speed over the conventional grouped-query attention (GQA) in long-context scenarios.

To equip SIGMA with the capability of addressing system domain tasks, we carefully identify 15 primary source categories from over 120 system-related websites, collecting a total of 19.5 billion data for system domain pre-training and fine-tuning. Moreover, we construct the AIMICIUS benchmark to facilitate the evaluation of system domain task performance. It includes four major tasks - CMDGen, Infrawise, Optiflow, and NL2KQL - based on Azure services, which assess the critical capabilities in the system domain, such as command generation, benchmark retrieval, topology optimization, and infrastructure issue analysis.

Building on the above commitments, we pre-train SIGMA on 6T tokens from various sources, with around 1T tokens from synthesized and rewritten data and 19.5B system domain data, all of which have undergone extensive quality screening. In general domains, SIGMA achieves comparable performance to other state-of-arts models. Besides, SIGMA demonstrates remarkable performance across all tasks in AIMICIUS, significantly outperforming GPT-4 with an absolute improvement up to 52.5%.

### 2. DiffQKV Attention

In the standard Multi-Head Attention (MHA) introduced by Vaswani et al. (2017), Q, K, and V components consistently employ an equal number of heads, with each head maintaining the same dimension. Our research explores a more generalized form of attention mechanism, termed DiffQKV attention, where QKV can possess distinct numbers of heads as well as different dimensions per head. In this section, we will first formally describe the DiffQKV attention process and then discuss the optimal configurations for it to effectively balance efficiency and model performance.

Formulation of DiffQKV Attention. Let the numbers of QKV heads as nhq,nhk,nhv, respectively, and the dimensions

of QKV heads as dhq,dhk,dhv. Formally, given the attention input ht, we begin by transforming it to:

qt = ht · WQ, kt = ht · WK, vt = ht · WV ,

where the dimensions of qt, kt, vt are [nhq,dhq], [nhk,dhk], [nhv,dhv] respectively. Then, we load the previous K vectors from the cache and concatenate it with kt, thereby obtaining Kt, which has a dimension of [t,nhk,dhk]. To prepare for the attention score computation, Kt and qt are sliced into a series of heads. Since the number of K heads and Q heads can be inconsistent, we employ a GroupSharing operation on Kt, which allows multiple Q heads to be jointly associated with the same K head (see Appendix, Figure 3):

q)] = GroupSharing(Kt,nhq,nhk), [q(t,1);q(t,2);..;q(t,nh

[K(t,1);K(t,2);..;K(t,nh

q)] = qt, where q(t,i) ∈ Rd

h

k represents the i-th Q head. K(t,i) is in the shape of [t,dhk]. The attention scores are computed as:

Attend(q(Tt,i),K(t,i)) √dk

α(t,i) = Softmaxi[

],

where α(t,i) ∈ Rt. In standard MHA, the Attend function computes the inner products of the Q heads and K heads. Nonetheless, as DiffQKV accommodates varying dimensions for the Q and K heads, an alternative implementation of the Attend function is necessitated. In our experiments, we implement this part by transforming the dimension of K to the same as V through another feed-forward layer. Next, we load previous V cache and compute the outputs as:

Vt = [Vt-1;vt], Vt-1 = LoadCacheV(&Vcache,α(t,i)), [V(t,1);V(t,2);..;V(t,nh

q)] = GroupSharing(Vt,nhq,nhv), o(t,i) =

t j=1

α(j,i)V(j,i), xt = WO[o(t,1);o(t,2);...;o(t,nh v)].

Here, the dimension of Vt is [t,nhv,vh], that of V(t,i) is [t,dhv], o(t,i) ∈ Rd

h

m

. Notably, the LoadCacheV function, used for retrieving previous V vectors, can differ from the strategy for loading K cache.

v and xt ∈ Rd

In order to find the optimal configurations of DiffQKV attention, we train the model with different settings of DiffQKV from scratch and assess their performance. In the following experiments, we adopt 100B tokens from FineWeb-Edu (Penedo et al., 2024) as the pre-training data and the model are scaled to approximately 1B parameters. We employ the following benchmarks to evaluate the model: HellaSwag (Zellers et al., 2019), OpenBookQA (Mihaylov et al., 2018), WinoGrande (Sakaguchi et al., 2021), ARC Challenge (Clark et al., 2018), PIQA (Bisk et al., 2020), SciQ (Welbl et al., 2017), BoolQ (Clark et al., 2019), LogiQA (Liu et al., 2020), and LAMBADA (Paperno et al., 2016). Our major observations are as below.

Table 1. Comparisons of model performance when reducing an equal number of K or V heads. The number of Q heads is 32 for all models (nhq = 32). Note that the reported performance scores in Table 1-4 are the average results across nine benchmarks. See Appendix B.3 for detailed results on each benchmark.

Model Performance MHA (nhk=nhv=32) 52.40

- -50% V Heads (nhv=16) 51.74 (↓0.66)

- -50% K Heads (nhk=16) 52.83 (↑0.43) GQA (nhk=nhv=16) 52.14

- -75% V Heads (nhv=4) 51.76 (↓0.38)

- -75% K Heads (nhk=4) 51.97 (↓0.17) GQA (nhk=nhv=4) 51.66

- -75% V Heads (nhv=1) 51.03 (↓0.63)

- -75% K Heads (nhk=1) 51.67 (↑0.01)

Table 2. The ablation studies of halving the K head dimension. The results indicate that this adjustment, while largely improving the inference efficiency by reducing the size of KV cache, does not significantly compromise performance. The number of Q heads is 32 for all models (nhq = 32).

Model Performance MHA (nhk=nhv=32) 52.40 w/ Half K Dim. 52.56 (↑0.16) GQA (nhk=nhv=16) 52.14 w/ Half K Dim. 52.06 (↓0.08) GQA (nhk=nhv=4) 51.66 w/ Half K Dim. 51.92 (↑0.26)

Table 3. Compare baseline model architectures with those having augmented Q. dhq refers to the intermediate Q head dimension. The number of Q heads is 32 for all models in the table (nhq = 32). For the baseline without AugQ, the intermediate dimension of Q head is dhq = 2048.

Model Performance MHA 52.40

+ AugQ (dhq=5632) 53.03 (↑0.63) GQA (nhk=nhv=16) 52.14

+ AugQ (dhq=3072) 53.38 (↑1.24) + AugQ (dhq=4096) 52.93 (↑0.79) + AugQ (dhq=5632) 53.07 (↑0.93)

###### GQA (nhk=nhv=4) 51.66

+ AugQ (dhq=5632) 53.13 (↑1.47)

- Observation 1: The model performance is more sensitive to the decrease in the number of V heads (nhv) than

in the number of K heads (nhk). Specifically, we reduce the number of heads of K and V in the DiffQKV attention,

respectively, and assess their impacts on model performance by comparing with the baseline of equal KV heads. As shown in Table 1, reducing K heads has a minor impact compared to reducing V heads in most cases. This differential impact is reasonable, considering the distinct roles of K and V within the attention mechanisms. The K heads primarily serve to compute attention matrices, which prove to have a remarkable sparsity exceeding 95% during inference (Zhang et al., 2024b), so a slight decrease in the parameters used for its calculation can still yield a precise approximation. In contrast, the V vectors, which directly influence the final attention output, demands a more nuanced approach.

- Observation 2: Reducing K head dimension to half of

V head dimension (dhk = dhv/2) can preserve the model performance compared to equal-dimension K and V

heads (dhk = dhv). We explore compressing K components by shrinking K head dimension. To compute the atten-

tion scores when the dimensions of K and V are different, we need to transform the dimension of K to the same as V through a trainable feed-forward layer. In essence, the reduction of K head dimension reduces KV cache at the cost of additional computation from a feed-forward layer. This trade-off is cost-effective as the primary bottleneck for LLM inference speed lies in memory consumption rather than computation (Shazeer, 2019; Ribar et al., 2024). To study the impact, we conduct compare several commonlyused attention settings, such as MHA and GQA, with their counterparts having only half of the K head dimension. Results shown in Table 2 demonstrate that the performance decrease from reducing K head dimension is negligible; in some cases, the model with a smaller K head dimension even outperforms the baseline setting prior to compression.

Table 4. Comparisons of the model performance when incorporating the augmented Q component (AugQ) with different sizes and enlarging the FFN module (AugF). The baseline method is GQA, with the FFN dimension being 5632 and nhk=nhv=16. ∆dF denotes the enlarged dimension for the FFN module, while dhq represents the intermediate Q head dimension (δ=3072).

Model Performance GQA 52.14

+ AugF (∆dF=δ) 53.26 (↑1.12) + AugQ (dhq=δ) 53.38 (↑1.24)

- + AugF (∆dF=2δ) 53.16 (↑1.02)

+ AugF (∆dF=δ) & AugQ (dhq=δ) 54.55 (↑2.41)

- + AugF (∆dF=3δ) 54.50 (↑2.36)

- + AugF (∆dF=2δ) & AugQ (dhq=δ) 54.67 (↑2.53)

+ AugF (∆dF=5δ) 55.08 (↑2.94)

- + AugF (∆dF=3δ) & AugQ (dhq=δ) 55.09 (↑2.95)

- Observation 3: Augmenting Q components can boost model performance. Augmenting with extra parameters is a straightforward way to enhance a model’s representational capacity and performance. Since Q vectors in self-attention do not need to be cached during inference, their impact on memory usage and data transfer duration is minimal. Thus, adding extra parameters to Q components is more cost-effective than doing so for KV components. To verify this, we experiment by varying Q’s parameter count. As demonstrated in Table 3, adding extra parameters to Q can consistently enhance the model’s performance to different extents, especially when K and V have higher compression

levels (nhv=4,nhk=4). Overall, a scaling factor of ×1.5 (i.e. dhq=3072) on the Q component appears optimal.

- Observation 4: With the same number of extra parameters, augmenting Q yields a greater performance boost than expanding the FFN module. Moreover, the performance gains from these two actions are independent.

That is, with a fixed augmented Q, further enlarging the FFN module still improves model performance. By adding the same number of parameters to the FFN and Q modules, respectively, we examine and compare their impacts on the model performance. Results in Table 4 show that augmenting Q consistently gives a more significant performance boost than adding the same number of parameters to the FFN module. Comparing the last two lines of Table 4, even when FFN’s additional parameters (2δ) are double those of Q (dhq = δ), the model performance is still slightly worse with FFN. This indicates that adding parameters to the Q component in the self-attention layer more effectively improves model performance than to the FFN module.

Summary: Optimal Configurations for DiffQKV Attention. The observations detailed above give rise to two primary guidelines for the optimal configurations of DiffQKV attention: 1) differentially compressed KV: Aggressive compression can be more pronouncedly applied to the K components as opposed to the V components, both in terms of the head number and head dimension; 2) augmented Q: introducing extra parameters to the Q components can enhance the representational capacity of the self-attention layer, while having a relatively small impact on inference efficiency. More experiments that combine different settings of DiffQKV are included in Appendix Table 13.

SIGMA Model Architecture. Building on the DiffQKV attention, we construct a pre-trained language model, named SIGMA. Specifically, we adopt two model scales with 1.5B parameters and 10B parameters, respectively (i.e. SIGMA1.5B and SIGMA-10B). They both adopt the aforementioned strategies of differentially compressed KV and augmented Q in the self-attention layer. The detailed configurations of the SIGMA architecture are included in Appendix B.5.

### 3. Efficiency Analysis

In this section, we analyze SIGMA-1.5B’s efficiency both theoretically and empirically, using FlashAttention2 (Dao, 2024) as the default attention mechanism.

#### 3.1. Theoretical Analysis

The efficiency gains of SIGMA can be largely attributed to the reduction in the number of key heads nhk. In SIGMA1.5B, nhk is reduced from 16 to 4, whereas nhv remains 16. This reduction directly impacts two critical efficiency indicators: KV Cache and Attention Computation.

KV Cache. Generally, key cache and value cache take dimensions (b,s,nh,dh), where b is the batch size, s is the sequence length, nh denotes the number of attention heads, and dh represents the dimension of each head. Since KV cache is responsible solely for storing and loading the key and value tensors, we model the cost of KV Cache

as being proportional to the total number of elements in the key and value caches. Therefore, it can be expressed as a linear function of the cache size as shown in Eq. (1), with α representing the proportional cost per element and β accounting for fixed overheads or unrelated losses.

L = α · [b · s · (nhk · dhk + nhv · dhv)] + β. (1) In SIGMA 1.5B, the key cache has the dimension of (b,s,4,64), whereas the value cache is (b,s,16,64). Compared to grouped query attention (GQA), since nhk is reduced, the total cost of KV cache operations will decrease accordingly. As the size of cache increases, the reduction rate r of the total cost of KV cache operations converges toward a theoretical value as shown in Eq. (2).

LGQA − LSigma LGQA

32 − 20 32

= 37.5%. (2)

=

r = lim

s→+∞

Attention Computation. Here, the cost reduction also originates from the reduction in key heads. The theoretical reduction rate r therefore remains 37.5%. However, while KV cache operations are primarily I/O-intensive, attention computation is significantly more computation-intensive. Due to the increased complexity of this module, the observed reduction in practice is likely to deviate more significantly from the theoretical rate compared to the KV cache.

Apart from aforementioned improvements, it is important to highlight that the reduction of nhk also decreases KV cache’s memory consumption. SIGMA hence allows for larger batch sizes, enhancing inference efficiency naturally.

#### 3.2. Implementation

While reducing nhk theoretically leads to significant efficiency improvements, practical implementation is hindered

by the tight integration of KV cache and attention operations in FlashAttention and existing LLM deployment frameworks (e.g., vLLM (Kwon et al., 2023) and TensorRT2). Here, we propose our current solutions and emphasize the need for broader DiffQKV support.

Differential Management of KV Cache. In SIGMA, since key and value have different number of heads, their cache sizes differ. Generally, it poses no issues for most implementations that store K and V cache separately. However, for frameworks that combine key and value cache into a whole matrix, it becomes challenging. To accommodate these frameworks, one workaround is to duplicate K to match the size of V (named KV Group Sharing). However, this approach is highly discouraged, as it negates the efficiency improvements entirely by degrading to GQA. Its official implementation can be found in Appendix E.

Flexible Attention Computation. The official FlashAttention supports attention computations only when nhk equals

2https://github.com/NVIDIA/TensorRT-LLM/

[Figure 1]

[Figure 2]

[Figure 3]

Prefix Split Combine Total

2k 1.17% 1.68% 1.39% 4k 25.33% 0.08% 18.02% 16k 26.30% 0.25% 23.35% 32k 27.21% -0.93% 25.31%

(d) KET’s relative improvement of each kernel

(a) KET of the split kernel. (b) KET of the combine kernel. (c) Total KET.

Figure 1. KET comparison of FlexHeadFA between Standard model(STD) and SIGMA.

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

(a) Output Length = 2k. (b) Output Length = 4k. (c) Output Length = 8k. (d) Absolute CEET Improvment.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

(e) Output Length = 16k. (f) Output Length = 32k. (g) Output Length = 64k. (h) Relative CEET Improvment. Figure 2. Comparison of total CEET cost between Standard model(STD) and SIGMA. The gray dashed line indicates where the inference costs of both models are equal. As the output length increases, this intersection point moves progressively earlier.

nhv, and nhq is an integer multiple of nhk and nhv, which could not be directly applied to SIGMA. To address this issue, we

introduce FlexHeadFA3, which leverages address probing to support attention computation with a flexible number of heads. Specifically, FlexHeadFA’s implementation follows that of FlashAttention2, consisting of two major GPU kernel operations flash_fwd_splitkv_kernel and flash_fwd_splitkv_combine_kernel. The split kernel handles primary attention computations by dividing key and value matrices into chunks, performing multiplications, and generating output chunks. FlexHeadFA separates address calculations for key and value heads, using the query head index idxq to determine the key and value head indices (idxk, idxv) via Eq. (3).

idxi = idxq ·

nhi nhq

,i ∈ {k,v}. (3)

This design enables efficient element retrieval for multiplication, removing the constraint of equal head numbers. The combine kernel then assembles these chunks into the final output matrix.

3https://shorturl.at/dpzaX

#### 3.3. Empirical Analysis

Experiment Setup. To demonstrate the efficiency gains achieved by reducing nhk, we compare SIGMA 1.5B with a hypothetical baseline model, referred to as the standard model (STD). To be specific, while SIGMA features unbalanced KV heads (nhk = 4 and nhv = 16) and includes an additional augmented Q (dhq = 3072), STD employs a standard Group Query Attention (GQA) mechanism, configured with balanced K and V heads (nhk = 16 and nhv = 16). To accurately measure efficiency improvements, we employ two metrics: Cuda Event Elapsed Time (CEET) and Kernel Execution Time (KET). CEET measures elapsed time of certain operations by placing checkpoints within the code. While it may be unstable at small absolute time scales, it is well-suited for large-scale experiments. In this test, we record the cost of storing and loading KV Cache, applying Attention Computation, and incorporating Augmented Q (only in SIGMA), treating their summation as the total cost of attention layers. Both SIGMA 1.5B and STD are evaluated with output and prefix lengths independently increasing in grid patterns of [2k, 4k, 8k, 16k, 32k, 64k] and [0, 2k, 4k, 16k, 32k, 64k], respectively. KET provides a precise time estimation by recording kernel execution times using

nsys. Here, we measure the cost of the combine kernel and the split kernel, treating their summation as the overall cost of applying FlexHeadFA. Although this method ensures high accuracy, it incurs significant memory overhead as the inference scale increases. Therefore, both models are evaluated with 10 output length and prefix length varying in a grid pattern: [2k, 4k, 16k, 32k]. All experiments are conducted on a NVIDIA H100 80G HBM3 GPU, with details in Appendix F.

KET Results. We present KET results in Figure 1 with details recorded in Table 15. The results align closely with our previous analyses. The split kernel, which directly operates on the key and value matrices, benefits naturally from the reduction in key heads in terms of memory load cost. Both its absolute cost and relative improvement increase sharply as the prefix length grows. In contrast, the combine kernel, responsible soly for combining output chunks, is unaffected by changes in the key matrix. Consequently, its improvement ratio consistently stays near zero. The relative improvements for both the split kernel and the total cost become increasingly significant with longer prefix lengths, and we anticipate the absolute improvement to approach 37.5% as the prefix length continues to expand.

CEET Results - Total Cost. Since STD and SIGMA share a largely similar design, the combined cost of above three modules is treated as the total cost (excluding Augmented Q for STD). As illustrated in Figure 2, SIGMA becomes increasingly efficient with the increase in output length and prefix length. Specifically, while SIGMA incurs higher inference costs than STD for shorter prefix lengths in Figure 2a, the cost crossover point shifts earlier as output length increases. For example, SIGMA surpasses STD only when prefix lengths exceed 16k tokens for 2k outputs, but it outperforms STD even without a prefix when generating 64k tokens, reducing total cost by up to 33.36% for 64k outputs (Figure 2g). Detailed cost analyses of the three modules can be found in Appendix G.

In summary, while SIGMA is slightly less efficient than the standard model in short-context scenarios (∼ 103 ms), it shows significant advantages in long-context scenarios (∼ 106 ms), which are common and critical in practical applications. This makes SIGMA promising for advancing test-time scale-up in LLM, enhancing their scalability and efficiency for more complex, extended inference processes.

### 4. System Domain Pre-training and AIMICIUS Benchmark

In our work, we pre-train SIGMA from scratch to be a specialized model tailored for system-related tasks, called SIGMA-SYSTEM, with our meticulously collected domain data. Specifically, our goal is to create LLMs that can au-

Table 5. SIGMA’s system domain pre-training data composition.

Data Type Sources Size # Tokens General System

CCF Ranking list 14.0 G 3.3 B arXiv 33.0 G 5.4 B

Design Capability Blogs & Forums 14.5 G 3.2 B Debug Capability Stack Overflow 38.9 G 7.6 B

tomatically diagnose AI infrastructure issues and profile AI workloads (e.g., being able to generate command lines to monitor GPU utilization). These features would allow the LLMs to oversee the training processes of neural models, even including their own, and support automated optimization. To achieve this, we compile system-specific pre-training and fine-tuning data and introduce AIMICIUS, a benchmark for system domain evaluation.

System Domain Data Collection. We identify 15 primary source categories from over 120 system-related websites to collect data. In particular, to enrich general system knowledge, we gather data from a diverse range of sources, including academic papers from arXiv and renowned conferences or journals. StackOverflow is the main source for enhancing debugging skills, while technical blogs and developer forums are crucial for refining system design capability. Additionally, we collect data from related websites to include knowledge on Azure VM, hardware abstraction, Linux commands, GitHub issues, and Stack Exchange. These sources make up the bulk of our pre-training data, supplemented by several other minor resources. As these data sources exhibit varied formats, we process each source individually to extract publicly available system data. To efficiently cleanse the substantial data, we utilize LLM labeling and various AI tools automatically, including category classification, quality control, and data format conversion. For instance, we employ GPT-3.5 to categorize the data items from StackOverflow, subsequently training a smaller model to classify the remaining data, thereby saving on costs. Ultimately, we gather approximately 19.5B tokens for pre-training. Table 5 presents detailed statistics of our collected data.

AIMICIUS Benchmark. Currently, AIMICIUS comprises four tasks: CMDGen, Infrawise, Optiflow, and NL2KQL, with most data sourced from Azure service. Details on evaluation and examples are provided in Appendices C and D.

• CMDGen focuses on command-line generation for the NVIDIA and AMD platforms. The target is to generate optimal commands to address specific GPUrelated challenges described in the prompts. The commands in CMDGen can be categorized into seven distinct subgroups: NCCL, Nvidia-smi, NVCC, RCCL, Rocm-smi, Superbench, and a general others category. The data in CMDGen are sourced from a variety of origins, ensuring diversity and realism. Some exam-

ples are curated from official documentation, paired with human-written or LLM-generated queries for context; others are directly extracted from Azure service logs or websites like StackOverflow to capture real-world usage patterns. It comprises a total of 2,838 instruction-tuning examples and 395 test cases, with 200 for the NVIDIA platform and 195 for the AMD platform.

- • Infrawise aims at retrieving benchmark results of a particular model in terms of its infrastructure-wise performance (e.g. the inference speed of GPT-3 on a single A100). Performing this task generally involves two critical steps: DCW Generation and Benchmark Result Retrieval. The dataset of Infrawise comprises 422 instruction-tuning samples and 911 end-to-end test cases.
- • Optiflow targets for optimizing network topology and data flow within a specified multi-GPU SKU and data size to minimize all-gather latency. It is further divided into two subtasks: (1) Plan Generation and (2) Plan Improvement. In Plan Generation, the LLM generates Python codes that represent the optimized network topology and data flow for the given multi-GPU configuration. In Plan Improvement, users provide the latency of their current design, and the LLM is tasked with refining the existing plan to deliver improved code with reduced latency. This task challenges the model’s ability to reason about complex hardware configurations and effectively deliver actionable, efficient solutions. Optiflow comprises a total of 8,000 instruction-tuning examples and 1,258 test cases. Among the instruction-tuning examples, 5,047 are derived from the Plan Generation subtask, while 2,953 focus on the Plan Improvement subtask. The test cases are structured in an end-to-end manner: LLM is first tasked with generating a plan based on the input, followed by refining the plan given the current latency.
- • NL2KQL is for converting the user instruction in the form of natural language into Kusto Query Language (KQL)4, which is specifically designed for querying and analyzing large datasets in Azure Data Explorer and other Microsoft services, such as Log Analytics and Application Insights. The NL2KQL dataset includes a total of 5,166 instruction-tuning examples and 43 test cases.

### 5. Performance Evaluations

Pre-training Settings. The pre-training data, totaling 6T tokens, consists of general domain data and domainspecific property data. For general domain data, we combine DCLM and FineWeb-EDU, remove duplicates to get General Dataset I (about 4T tokens). After quality filtering, the result is General Dataset II (1T tokens). Then, we select higher-scoring data with stricter rules for General Dataset III (about 200B tokens for the annealing phase of SIGMA’s

4https://shorturl.at/cvIOc

Table 6. Evaluation results on the AIMICIUS benchmark. The baselines include GPT-4, Deepseek-Coder-7b-Instruct-v1.5, Qwen2.5-Coder-7B-Instruct, and LLaMA3-8B-Instruct. All metrics are normalized to a scale of 0 to 100, with higher values indicating better performance. Bolded metrics represent the most critical evaluation criteria for each task. SIGMA-SYSTEM 10B is fine-tuned (SFT) using our proprietary dataset.

Task Metric (↑) GPT-4 DeepSeek Qwen Llama SIGMA

CMD Score 84.0 59.4 81.1 33.6 87.5 Output Score 61.0 12.9 38.3 12.0 80.9 Calibration Score 62.0 0.5 52.0 5.0 78.0 Exact Match 13.0 - - - 57.0 Success Ratio 21.0 0.5 15.5 3.5 74.0 Accuracy 25.0 0.5 15.5 3.5 74.5

CMDGen NVIDIA

CMD Score 73.0 55.0 77.2 31.4 88.9 Output Score 49.0 20.3 49.1 14.1 78.0 Calibration Score 43.0 3.1 51.3 2.6 79.3 Exact Match 14.0 - - - 53.9 Success Ratio 13.0 3.1 35.8 1.6 69.4 Accuracy 17.0 3.1 35.8 1.6 69.4

CMDGen AMD

Target 40.7 43.5 44.9 28.1 95.2 Baseline 34.1 43.6 42.7 34.5 92.9 Criterion 55.1 36.3 44.2 11.1 75.1 Workload 52.1 39.9 47.8 14.2 48.4 DCW 22.5 16.6 20.8 4.6 40.3 Bench. Recall 19.8 11.6 20.4 6.5 28.3 Bench. Acc. 18.7 11.6 12.6 6.5 28.3

Infrawise

Code Detected 95.8 82.2 100.0 98.2 100.0 Code Executable 50.3 18.6 28.3 51.2 85.9 Plan Valid 16.8 18.6 28.3 16.4 86.7 Plan Improved 0.5 - - 1.1 66.7

Optiflow

Syntax Acc. 100.0 81.4 100 83.7 100.0

Similarity 31.8 33.9 36.7 33.6 34.9 Cluster Score - - - - 43.0 Database Score 2.3 2.3 4.7 2.3 40.7 Table Score 4.7 4.7 4.7 4.7 17.4 Column Score 17.3 19.3 22.2 20.8 29.0

NL2KQL

pre-training). For math domain data, we use proof-pile2 and combine it with 280 billion math-related data from General Dataset I. For code domain data, following StarcoderV2’s filtering method, we select a 500B token dataset. We also have about 1T tokens of synthesized and rewritten pre-training data, which have passed quality screening and contain multi-domain content, for the later phase of SIGMA pre-training. More details can be found in Appendix H.1.

System Domain Performance. We pre-train SIGMASYSTEM-10B on our system domain pre-training data and fine-tuned with full-parameter updates on our SFT dataset, tailored individually for each task. We consider the following models as the compared baselines: GPT-4 (Bubeck

- et al., 2023), DeepSeek-Coder-7b-Instruct-v1.5 (DeepSeek) (Xin et al., 2024), Qwen-Coder-7B-Instruct (Qwen) (Yang
- et al., 2024), and LLaMA3-8B-Instruct (LLaMA) (Dubey et al., 2024), to compare them with SIGMA-SYSTEM on the complete AIMICIUS benchmark.

As shown in Table 6, SIGMA-SYSTEM showcases superior performance across all tasks in the AIMICIUS benchmark,

Table 7. Comparisons with baseline models on commonsense reasoning and text understanding tasks. Differences with original reports in the baseline models are due to our unified re-evaluations for fair comparisons.

Commonsense & Comprehension Continued

Model Params Avg.

Hella. ObQA Wino. ARC.e ARC.c PIQA SciQ Bool. Logi.

Pythia 1.0B 49.3 47.1 31.4 53.4 49.0 27.1 69.3 76.1 60.8 29.8 TinyLlama 1.1B 54.0 61.5 36.8 59.5 55.6 32.7 73.6 84.2 56.0 25.8 Bloom 1.1B 46.5 43.0 29.4 55.0 45.5 25.6 67.2 74.5 59.1 18.9 OLMo 1.2B 54.5 63.0 36.2 59.9 57.3 30.9 75.1 78.7 61.8 27.8 OPT 1.3B 51.2 53.7 33.2 59.8 51.0 29.5 72.4 76.7 57.7 26.9 CerebrasGPT 1.3B 46.5 38.4 29.0 52.1 45.8 25.3 66.8 73.0 59.4 29.2 Phi1 1.3B 39.6 30.4 25.0 49.9 34.6 23.5 56.0 64.5 45.2 27.3 Pythia 1.4B 51.7 52.0 33.2 57.3 53.9 28.3 70.9 79.3 63.3 27.5 DCLM 1.4B 62.8 71.6 42.6 66.2 71.6 43.5 77.7 90.7 71.4 29.8 StableLM2 1.6B 61.4 69.0 38.8 63.6 68.2 38.9 76.6 95.3 74.7 27.2 SmolLM 1.7B 61.0 65.7 42.0 60.8 73.5 46.3 76.1 89.6 66.0 28.7 Gemma 2.0B 62.2 71.4 40.0 64.6 72.3 41.8 78.2 91.5 69.3 30.4

SIGMA (Ours) 1.5B 61.6 67.3 40.4 67.5 72.3 43.9 77.8 94.0 63.0 28.4

substantially outperforming the baseline models. Specifically, compared to the second-best model, its absolute improvement is as significant as 49.5%, 33.6%, 9.6%, 65.6%, and 36% on the major metrics of the CMDGen NVIDIA, CMDGen AMD, Infrawise, Optiflow, and NL2KQL tasks, respectively. Its high scores in various metrics indicate its remarkable ability to generate high-quality commands, monitor infrastructure effectively, and perform well in other system-related tasks. Though the baseline models excel in general domains, they struggle in AIMICIUS, especially in CMDGen and Optiflow tasks, highlighting the importance of domain-specific advancements for LLMs.

Although our model surpasses the baselines in nearly all metrics, it is crucial to recognize that its absolute performance in specific tasks, such as Infrawise and NL2KQL, is still subpar. In Infrawise, SIGMA-SYSTEM performs exceptionally in terms of the Target, Baseline, Criterion scores. Nonetheless, its other metrics do not exceed 50%, with Benchmark Result Accuracy standing at 28.3%. The performance of other models on this task is even more disappointing. This indicates that there is room for further improvement, particularly in scenarios requiring more nuanced understanding and precise decision-making. As part of our future work, we plan to enrich the training dataset and refine the data generation policy to better capture the complexity and variability of these tasks, aiming to achieve enhanced performance across all evaluation metrics.

General Domain Performance. We pre-train a general domain model using the SIGMA-1.5B architecture to investigate its performance in more settings. We consider following benchmarks for evaluation of commonsense reasoning and text-understanding ability: HellaSwag (Hella.) (Zellers

- et al., 2019), OpenBookQA (ObQA) (Mihaylov et al., 2018), WinoGrande (Wino.) (Sakaguchi et al., 2021), ARC easy/Challenge (ARC.e/c) (Clark et al., 2018), PIQA (Bisk
- et al., 2020), SciQ (Welbl et al., 2017), BoolQ (Bool.) (Clark

et al., 2019), and LogiQA (Logi.) (Liu et al., 2020). We compare SIGMA against the following state-of-the-art models with similar scales: TinyLLaMA-1.1B (Zhang et al., 2024a), Pythia-(1.0B, 1.4B) (Biderman et al., 2023), OPT1.3B (Zhang et al., 2022), Bloom-1.1B (Muennighoff et al., 2022), Cerebras-GPT-1.1B (Dey et al., 2023), Phi1-

- 1.3B (Gunasekar et al., 2023), StableLM-2-1.6B (Bellagente et al., 2024), SmolLM-1.7B (Allal et al., 2024), DCLM-
- 1.4B (Li et al., 2024), OLMo-1.2B (Groeneveld et al., 2024), and Gemma-2B (Team et al., 2024). All results are obtained via zero-shot prompting.

As shown in Table 7, SIGMA-1.5B achieves an outstanding average performance of 61.6 on 9 benchmarks, significantly outperforming strong baseline models such as OLMo-1.2B and TinyLLaMA-1.1B on various reasoning and comprehension tasks. It also achieves comparable performance to state-of-the-art large language models such as Gemma2B, and DCLM-1.4B. Specifically, SIGMA achieves top-2 performance at WinoGrande, PIQA, ARC.e/c and SciQ, showing its broad and accurate grasp of both intuitive and logical commonsense during its pre-training on large-scale text sequences, which provides reliable knowledge backups for reasonable text completions and further task-specific tuning. Although it excels at common sense reasoning, SIGMA shows limited performance on BoolQ and LogiQA, reflecting a moderate level of reading comprehension on option-form questions. We observe a significant decrease in these benchmarks during the annealing stage when math problem-solving ability improves, showing a potential conflict between natural and formal language understanding for small-scale models. Additional evaluation results on problem-solving tasks are presented in Appendix H.3.

### 6. Conclusion

We introduce SIGMA, an efficient LLM specialized for the system domain. Its architecture features a novel attention module that called DiffQKV, which is equipped with augmented Q for performance improvement and differential KV compression for enhanced inference efficiency. Through theoretical and empirical analyses, we demonstrate the competitive efficiency of DiffQKV, with up to 33.36% speed improvement compared with Grouped-Query Attention(GQA) in long-context scenarios. Moreover, after pre-training on 6 trillion tokens, SIGMA demonstrates performance on par with the latest state-of-the-art models across general domains. On the AIMICIUS benchmark, the first comprehensive system domain benchmark that we propose, SIGMA substantially surpass all baseline models across all tasks, achieving an absolute improvement up to 52.5%.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Abdin, M., Jacobs, S. A., Awan, A. A., Aneja, J., Awadallah, A., Awadalla, H. H., Bach, N., Bahree, A., Bakhtiari, A., Behl, H. S., Benhaim, A., Bilenko, M., Bjorck, J., Bubeck, S., Cai, M., Mendes, C. C. T., Chen, W., Chaudhary, V., Chopra, P., Giorno, A. D., de Rosa, G., Dixon, M., Eldan, R., Iter, D., Goswami, A., Gunasekar, S., Haider, E., Hao, J., Hewett, R. J., Huynh, J., Javaheripi, M., Jin, X., Kauffmann, P., Karampatziakis, N., Kim, D., Kim, Y. J., Khademi, M., Kurilenko, L., Lee, J. R., Lee, Y. T., Li,

- Y., Liang, C., Liu, W., Lin, E., Lin, Z., Madan, P., Mitra,

- A., Modi, H., Nguyen, A., Norick, B., Patra, B., PerezBecker, D., Portet, T., Pryzant, R., Qin, H., Radmilac, M., Rosset, C., Roy, S., Saarikivi, O., Saied, A., Salim, A., Santacroce, M., Shah, S., Shang, N., Sharma, H., Song, X., Ruwase, O., Wang, X., Ward, R., Wang, G., Witte, P., Wyatt, M., Xu, C., Xu, J., Yadav, S., Yang, F., Yang,

- Z., Yu, D., Zhang, C.-Y., Zhang, C., Zhang, J., Zhang, L. L., Zhang, Y., Zhang, Y., and Zhou, X. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint, arXiv:2404.14219, 2024.

Aghajanyan, A., Yu, L., Conneau, A., Hsu, W.-N., Hambardzumyan, K., Zhang, S., Roller, S., Goyal, N., Levy, O., and Zettlemoyer, L. Scaling laws for generative mixed-modal language models. In International Conference on Machine Learning, pp. 265–279. PMLR, 2023.

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y.,

Lebron, F., and Sanghai, S. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 4895–4901, 2023.

Allal, L. B., Lozhkov, A., Bakouch, E., von Werra, L., and Wolf, T. Smollm - blazingly fast and remarkably powerful, 2024.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Azerbayev, Z., Schoelkopf, H., Paster, K., Santos, M. D., McAleer, S., Jiang, A. Q., Deng, J., Biderman, S., and Welleck, S. Llemma: An open language model for mathematics. arXiv preprint arXiv:2310.10631, 2023.

Bellagente, M., Tow, J., Mahan, D., Phung, D., Zhuravinskyi, M., Adithyan, R., Baicoianu, J., Brooks, B., Cooper, N., Datta, A., et al. Stable lm 2 1.6 b technical report. arXiv preprint arXiv:2402.17834, 2024.

Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020.

Brandon, W., Mishra, M., Nrusimha, A., Panda, R., and Kelly, J. R. Reducing transformer key-value cache size with cross-layer attention. arXiv preprint arXiv:2405.12981, 2024.

Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., R´e, C., and Mirhoseini, A. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S., et al. Sparks of artificial general intelligence: Early experiments with GPT-4. arXiv preprint arXiv:2303.12712, 2023.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Dai, D., Deng, C., Zhao, C., Xu, R., Gao, H., Chen, D., Li, J., Zeng, W., Yu, X., Wu, Y., et al. DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024.

Dao, T. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.

Dey, N., Gosal, G., Khachane, H., Marshall, W., Pathria, R., Tom, M., Hestness, J., et al. Cerebras-gpt: Open computeoptimal language models trained on the cerebras waferscale cluster. arXiv preprint arXiv:2304.03208, 2023.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle,

- A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan,

- A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Ge, S., Zhang, Y., Liu, L., Zhang, M., Han, J., and Gao, J. Model tells you what to discard: Adaptive KV cache compression for LLMs. In The Twelfth International Conference on Learning Representations, 2024.

GLM, T., Zeng, A., Xu, B., Wang, B., Zhang, C., Yin, D., Rojas, D., Feng, G., Zhao, H., Lai, H., et al. ChatGLM: A family of large language models from GLM-130B to GLM-4 all tools. arXiv preprint arXiv:2406.12793, 2024.

Groeneveld, D., Beltagy, I., Walsh, P., Bhagia, A., Kinney, R., Tafjord, O., Jha, A. H., Ivison, H., Magnusson, I., Wang, Y., et al. Olmo: Accelerating the science of language models. arXiv preprint arXiv:2402.00838, 2024.

Gunasekar, S., Zhang, Y., Aneja, J., Mendes, C. C. T., Del Giorno, A., Gopi, S., Javaheripi, M., Kauffmann, P., de Rosa, G., Saarikivi, O., et al. Textbooks are all you need. arXiv preprint arXiv:2306.11644, 2023.

Gupta, A., Dar, G., Goodman, S., Ciprut, D., and Berant, J. Memory-efficient transformers via top-k attention. arXiv preprint arXiv:2106.06899, 2021.

Han, C., Wang, Q., Peng, H., Xiong, W., Chen, Y., Ji, H., and Wang, S. LM-infinite: Zero-shot extreme length generalization for large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3991–4008. Association for Computational Linguistics, 2024.

He, J. and Zhai, J. FastDecode: High-throughput GPUefficient LLM serving using heterogeneous pipelines. arXiv preprint arXiv:2403.11421, 2024.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

Hu, S., Tu, Y., Han, X., He, C., Cui, G., Long, X., Zheng, Z., Fang, Y., Huang, Y., Zhao, W., et al. MiniCPM: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024a.

Hu, X., Xiong, T., Yi, B., Wei, Z., Xiao, R., Chen, Y., Ye, J., Tao, M., Zhou, X., Zhao, Z., Li, Y., Xu, S., Wang, S., Xu, X., Qiao, S., Kuang, K., Zeng, T., Wang, L., Li, J., Jiang, Y. E., Zhou, W., Wang, G., Yin, K., Zhao, Z., Yang, H., Wu, F., Zhang, S., and Wu, F. Os agents: A survey on mllm-based agents for general computing devices use. Preprints, December 2024b. doi: 10.20944/ preprints202412.2294.v1. URL https://doi.org/ 10.20944/preprints202412.2294.v1.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023a.

Jiang, H., Wu, Q., Luo, X., Li, D., Lin, C.-Y., Yang, Y., and Qiu, L. LongLLMLingua: Accelerating and enhancing llms in long context scenarios via prompt compression. arXiv e-prints, pp. arXiv–2310, 2023b.

Jin, H., Han, X., Yang, J., Jiang, Z., Liu, Z., Chang, C.Y., Chen, H., and Hu, X. Llm maybe longlm: Selfextend llm context window without tuning. In Forty-first International Conference on Machine Learning, 2024.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pp. 611–626, 2023.

Li, J., Fang, A., Smyrnis, G., Ivgi, M., Jordan, M., Gadre, S., Bansal, H., Guha, E., Keh, S., Arora, K., et al. Datacomplm: In search of the next generation of training sets for language models. arXiv preprint arXiv:2406.11794, 2024.

Li, R., Zi, Y., Muennighoff, N., Kocetkov, D., Mou, C., Marone, M., Akiki, C., Jia, L., Chim, J., Liu, Q., et al. StarCoder: may the source be with you! Transactions on Machine Learning Research, 2023.

Liu, J., Cui, L., Liu, H., Huang, D., Wang, Y., and Zhang, Y. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. arXiv preprint arXiv:2007.08124, 2020.

Liu, Z., Desai, A., Liao, F., Wang, W., Xie, V., Xu, Z., Kyrillidis, A., and Shrivastava, A. Scissorhands: Exploiting the persistence of importance hypothesis for LLM KV cache compression at test time. Advances in Neural Information Processing Systems, 36, 2024a.

Liu, Z., Zhao, C., Iandola, F., Lai, C., Tian, Y., Fedorov, I., Xiong, Y., Chang, E., Shi, Y., Krishnamoorthi, R., et al. MobileLLM: Optimizing sub-billion parameter language models for on-device use cases. arXiv preprint arXiv:2402.14905, 2024b.

Lu, Z., Li, X., Cai, D., Yi, R., Liu, F., Zhang, X., Lane, N. D., and Xu, M. Small language models: Survey, measurements, and insights. arXiv preprint arXiv:2409.15790, 2024.

Luohe, S., Zhang, H., Yao, Y., Li, Z., et al. Keep the cost down: A review on methods to optimize LLM’s KVcache consumption. In First Conference on Language Modeling, 2024.

Mehta, S., Sekhavat, M. H., Cao, Q., Horton, M., Jin, Y., Sun, C., Mirzadeh, I., Najibi, M., Belenko, D., Zatloukal, P., et al. OpenELM: An efficient language model family with open-source training and inference framework. arXiv preprint arXiv:2404.14619, 2024.

Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018.

Mu, J., Li, X., and Goodman, N. Learning to compress prompts with gist tokens. Advances in Neural Information Processing Systems, 36, 2024.

Muennighoff, N., Wang, T., Sutawika, L., Roberts, A., Biderman, S., Scao, T. L., Bari, M. S., Shen, S., Yong, Z.-X., Schoelkopf, H., et al. Crosslingual generalization through multitask finetuning. arXiv preprint arXiv:2211.01786, 2022.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, Q., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fern´andez, R. The lambada dataset: Word prediction requiring a broad discourse context. Cornell University arXiv,Cornell University - arXiv, Jun 2016.

Penedo, G., Kydl´ıˇcek, H., Lozhkov, A., Mitchell, M., Raffel, C., Von Werra, L., Wolf, T., et al. The fineweb datasets: Decanting the web for the finest text data at scale. arXiv preprint arXiv:2406.17557, 2024.

Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Heek, J., Xiao, K., Agrawal, S., and Dean, J. Efficiently scaling transformer inference. Proceedings of Machine Learning and Systems, 5:606–624, 2023.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are unsupervised multitask learners. 2019.

Ribar, L., Chelombiev, I., Hudlass-Galley, L., Blake, C., Luschi, C., and Orr, D. SparQ attention: Bandwidthefficient LLM inference. In Forty-first International Conference on Machine Learning, 2024.

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

Shazeer, N. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150, 2019.

Shi, H., Cheng, L., Wu, W., Wang, Y., Liu, X., Nie, S., Wang, W., Min, X., Men, C., and Lin, Y. Enhancing cluster resilience: LLM-agent based autonomous intelligent cluster diagnosis system and evaluation framework. arXiv e-prints, pp. arXiv–2411, 2024.

Sun, Y., Dong, L., Zhu, Y., Huang, S., Wang, W., Ma, S., Zhang, Q., Wang, J., and Wei, F. You only cache once: Decoder-decoder architectures for language models. arXiv preprint arXiv:2405.05254, 2024.

Suzgun, M., Scales, N., Sch¨arli, N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H., Zhou, D., , and Wei, J. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

Team, Gemma, Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju, S., Pathak, S., Sifre, L., Rivi`ere, M., Kale, M. S., Love, J., et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.

Vaswani, A., Shazeer, N. M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In Neural Information Processing Systems, 2017.

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., R´e, C., Barrett, C., et al. H2O: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36, 2024b.

Zhou, Z., Ning, X., Hong, K., Fu, T., Xu, J., Li, S., Lou, Y., Wang, L., Yuan, Z., Li, X., et al. A survey on efficient inference for large language models. arXiv preprint arXiv:2404.14294, 2024.

Welbl, J., Liu, N. F., and Gardner, M. Crowdsourcing multiple choice science questions. arXiv preprint arXiv:1707.06209, 2017.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, 2023.

Xin, H., Ren, Z., Song, J., Shao, Z., Zhao, W., Wang, H., Liu,

- B., Zhang, L., Lu, X., Du, Q., et al. Deepseek-prover-v1. 5: Harnessing proof assistant feedback for reinforcement learning and monte-carlo tree search. arXiv preprint arXiv:2408.08152, 2024.

Xiong, Y., Jiang, Y., Yang, Z., Qu, L., Zhao, G., Liu, S., Zhong, D., Pinzur, B., Zhang, J., Wang, Y., et al. {SuperBench}: Improving cloud {AI} infrastructure reliability with proactive validation. In USENIX Annual Technical Conference, pp. 835–850, 2024.

Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

Ye, L., Tao, Z., Huang, Y., and Li, Y. ChunkAttention: Efficient self-attention with prefix-aware KV cache and two-phase partition. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11608–11620. Association for Computational Linguistics, 2024.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Zhang, P., Zeng, G., Wang, T., and Lu, W. TinyLlama: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024a.

### A. Related Work

Over the past few years, Large Language Models (LLMs) have exerted a considerable impact across various domains (Bubeck et al., 2023; Jiang et al., 2023a; GLM et al., 2024; Dubey et al., 2024; Yang et al., 2024). Nevertheless, the formidable computational demands and memory requirements of LLM inference present considerable challenges for deployment in many resource-constrained scenarios. Thus, while the exploration into scaling up a larger model scale to achieve even more advanced level of intelligence is still ongoing (Kaplan et al., 2020; Aghajanyan et al., 2023; Brown et al., 2024), the development of smaller, more efficient language models has also garnered increasing research interest due to their significantly reduced inference costs and lower deployment requirements (Hu et al., 2024a; Zhang et al., 2024a; Abdin et al., 2024; Lu et al., 2024; Liu et al., 2024b; Team et al., 2024; Dai et al., 2024; Mehta et al., 2024). Meanwhile, a variety of methods have been explored to improve the inference efficiency of LLMs (Kwon et al., 2023; Ainslie et al., 2023; Xiao et al., 2023; Mu et al., 2024; Zhou et al., 2024). Significant efforts have been dedicated to addressing inference bottleneck associated with KV cache (Ainslie et al., 2023; Kwon et al., 2023; Luohe et al., 2024; Zhang et al., 2024b). KV cache is a common technique adopted by the decoder-only Transformer architecture (Vaswani et al., 2017), which stores the Key (K) and Value (V) vectors in the attention operation for future reuse at each decoding step to avoid recomputation. It can consume a substantial amount of memory that linearly increases with the sequence length (Pope et al., 2023). This consumption practically limits the context length that the model can handle. In addition, repeatedly loading KV cache also places substantial demands on memory bandwidth, which is recognized as the major bottleneck for LLM inference speed (Shazeer, 2019; Ribar et al., 2024). Notably, most prior studies tend to treat the compression of K and V vectors uniformly, both in terms of the optimization methodology and the compression ratio. For instance, Grouped-Query Attention (GQA) (Ainslie et al., 2023) reduces the number of K and V heads in the attention layer to the same extent by organizing Query (Q) heads into groups, with each group sharing a single K/V head. The same applies to methods based on KV cache eviction, which carefully select a fixed number of tokens and cache both their K and V vectors (Beltagy et al., 2020; Xiao et al., 2023; Han et al., 2024; Liu et al., 2024a; Zhang et al., 2024b; Ge et al., 2024; Ribar et al., 2024). However, the different roles K and V vectors play in the attention mechanism might allow for differential treatment in their efficiency optimization, an area that remains underexplored. Moreover, existing research rarely takes the optimization of Q into consideration when devising the optimization approach, which might also provide opportunities for further improvement in model performance and efficiency. In this study, we investigate more efficient model architectures for LLMs from the perspective of differentially adjusting the QKV components in the attention mechanism, and introduce SIGMA, a pre-trained language model featured by its efficient inference capabilities.

Apart from KV cache optimization, the studies on improving inference efficiency can be roughly divided into three categories: pre-training stage optimization, post-training stage optimization, and system-level optimization. Pre-training stage optimization approaches typically involve refinement on the model architecture (Shazeer, 2019; Ainslie et al., 2023; Sun et al., 2024; Brandon et al., 2024; Dai et al., 2024). Among them, Multi-Query Attention (MQA) (Shazeer, 2019) and Grouped-Query Attention (GQA) (Ainslie et al., 2023) are the two most widely-used methods. MQA is a variant of the standard Multi-Head Attention (MHA) mechanism, where all query heads share a single key head and value head. GQA further generalizes MQA by using an intermediate number of shared key and value heads. Post-training stage optimization is usually accomplished by selectively evicting part of the KV cache via token elimination (Beltagy et al., 2020; Xiao et al., 2023; Han et al., 2024; Liu et al., 2024a; Zhang et al., 2024b; Ge et al., 2024; Ribar et al., 2024). These approaches identify and eliminate non-essential tokens by employing criteria based on the attention scores at each decoding step. Another line of research (Mu et al., 2024; Jiang et al., 2023b) focuses on compressing prompts into a smaller number of “gisting” tokens, which can be cached and reused to enhance efficiency. The last category of approaches improves efficiency by designing systems specifically tailored to the characteristics of LLM inference (Kwon et al., 2023; Jin et al., 2024; Ye et al., 2024; He & Zhai, 2024). In particular, Kwon et al. (2023) proposed paged attention and introduced the vLLM framework, effectively mitigating the memory fragmentation issues associated with LLM inference.

### B. Detailed Illustration of DiffQKV Attention

The overview of DiffQKV attention is presented in Figure 3. In all experiments discussed in §2, we adopt 100B tokens from FineWeb-Edu (Penedo et al., 2024) as the pre-training data and the model are scaled to approximately 1B parameters, which consists of 22-layer transformer blocks with a hidden dimension of 2048. To evaluate the model’s performance on different experimental settings, we employed the following benchmarks: HellaSwag (Hella.) (Zellers et al., 2019), OpenBookQA (ObQA) (Mihaylov et al., 2018), WinoGrande (Wino.) (Sakaguchi et al., 2021), ARC Challenge (ARC.) (Clark et al., 2018), PIQA (Bisk et al., 2020), SciQ (Welbl et al., 2017), BoolQ (Bool.) (Clark et al., 2019), LogiQA (Logi.) (Liu et al., 2020),

and LAMBADA (Paperno et al., 2016) (LAMB.).

#### B.1. Comparison with Existing Attention Mechanisms.

The three widely used attention mechanisms, MHA, Multi-Query Attention (MQA), and Grouped-Query Attention (GQA), can be viewed as special forms of the above algorithm. As shown in Figure 3, these three methods all employ the same dimensions for Q, K, and V heads (dhq = dhk = dhv). In terms of the head number, MHA sets the same number of Q, K, and V heads (nhq = nhk = nhv), MQA only employs a single K head and V head (nhk = nhv = 1), and GQA features an intermediate number of K and V heads, yet still aligns their values (nhk = nhv).

###### Muti-Head Attention (MHA)

Mult-Query Attention (MQA)

Grouped-Query Attention (GQA)

DiffQKV Attention

Selective V Cache Fetching

(Only load a selection of cached V vectors based on the attention scores)

Values

Differentially Compressed KV

(K has fewer dimensions and number of heads)

Keys

Queries

Augmented Q

|Cached in Memory and Loaded for Calculation Cached in Memory but Unused at the Current Decoding Step|
|---|

- Figure 3. Overview of our proposed method for differential rescaling of QKV, compared alongside Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped Query Attention (GQA). Specifically, our method involves: (1) differentially compressed KV: applying more aggressive compression on the number of K heads and their dimensions than on the V components, which more significantly reduces the size of K cache. We can also optionally adopt selective V cache fetching for V compression; (2) augmented Q: adopting a higher dimension for the Q head compared to the KV heads.

- Table 8. The ablation studies of the model performance when only selectively loading the V vectors corresponding to the highest attention scores for approximate calculation. This operation significantly enhances inference efficiency by reducing memory usage. The number of Q heads is 32 for all models in the table (nhq = 32).

Commonsense & Comprehension Continued LM

Model Overall

Hella. ObQA Wino. ARC. PIQA SciQ Bool. Logi. LAMB.

MHA (nhk=nhv=32) 52.40 55.6 37.6 57.6 36.0 73.9 85.5 59.6 28.9 36.8 + Sel.V-top100 52.10 (↓0.30) 55.6 37.6 57.6 36.0 74.0 84.8 59.3 27.0 36.9

GQA (nhk=nhv=16) 52.14 55.1 39.6 56.3 35.4 71.9 85.0 61.4 27.8 36.8 + Sel.V-top100 52.08 (↓0.06) 55.2 39.6 56.3 35.4 71.8 84.4 61.6 27.8 36.7

GQA (nhk=nhv=4) 51.66 54.0 38.0 56.0 37.5 72.3 82.0 61.3 28.6 35.4 + Sel.V-top100 51.67 (↑0.01) 54.0 38.2 55.9 37.5 72.2 82.0 61.2 28.6 35.4

#### B.2. Selective V Cache Fetching

Apart from differentially compresssed KV and augmented Q, we can further optimize this part by loading V vectors selectively during inference. Due to the high sparsity exhibited by the attention scores (Xiao et al., 2023; Zhang et al., 2024b), it can still preserve the model performance with only a small number of V vectors to approximate the results, while largely reducing the memory usage. We term this strategy as selective V cache fetching (Xiao et al., 2023; Zhang et al., 2024b). Due to the high sparsity in attention scores, efficiency can be further enhanced by loading V vectors selectively during inference.

#### Specifically, we observe that using only a small proportion of V vectors, specifically those corresponding to the highest

- Table 9. Comparisons of model performance when reducing the same number of K heads versus V heads. The number of Q heads is 32

for all models (nhq = 32). The results show that compressing the number of K heads has a relatively smaller impact on the overall model performance.

Model Overall

Commonsense & Comprehension Continued LM

Hella. ObQA Wino. ARC. PIQA SciQ Bool. Logi. LAMB. MHA (nhk=nhv=32) 52.40 55.6 37.6 57.6 36.0 73.9 85.5 59.6 28.9 36.8

- -50% V Heads (nhv=16) 51.74 (↓0.66) 55.5 39.6 55.0 35.9 71.6 85.9 56.9 28.3 37.0

- -50% K Heads (nhk=16) 52.83 (↑0.43) 55.1 38.8 56.4 35.8 71.9 85.2 63.6 29.0 39.7 GQA (nhk=nhv=16) 52.14 55.1 39.6 56.3 35.4 71.9 85.0 61.4 27.8 36.8

- -75% V Heads (nhv=4) 51.76 (↓0.38) 54.0 38.2 55.6 34.8 72.7 85.0 60.3 29.9 35.3

- -75% K Heads (nhk=4) 51.97 (↓0.17) 54.6 37.8 57.1 35.1 72.3 84.1 62.5 28.3 36.0 GQA (nhk=nhv=4) 51.66 54.0 38.0 56.0 37.5 72.3 82.0 61.3 28.6 35.4

- -75% V Heads (nhv=1) 51.03 (↓0.63) 53.5 38.4 57.0 35.1 72.1 82.6 56.9 28.4 35.1

- -75% K Heads (nhk=1) 51.67 (↑0.01) 53.9 36.2 58.6 36.9 71.1 83.5 60.7 28.7 35.5

- Table 10. The ablation studies of halving the K head dimension. The results indicate that this adjustment, while largely improving the inference efficiency by reducing the size of KV cache, does not significantly compromise performance. The number of Q heads is 32 for all models (nhq = 32).

Model Overall

Commonsense & Comprehension Continued LM

Hella. ObQA Wino. ARC. PIQA SciQ Bool. Logi. LAMB.

MHA (nhk=nhv=32) 52.40 55.6 37.6 57.6 36.0 73.9 85.5 59.6 28.9 36.8 w/ Half K Dim. 52.56 (↑0.16) 55.2 39.4 56.9 36.9 72.7 84.1 63.3 27.8 36.8

GQA (nhk=nhv=16) 52.14 55.1 39.6 56.3 35.4 71.9 85.0 61.4 27.8 36.8 w/ Half K Dim. 52.06 (↓0.08) 54.3 39.8 56.9 36.7 72.0 83.9 59.5 29.2 36.2

GQA (nhk=nhv=4) 51.66 54.0 38.0 56.0 37.5 72.3 82.0 61.3 28.6 35.4 w/ Half K Dim. 51.92 (↑0.26) 53.4 39.2 56.6 35.6 72.5 84.0 62.3 28.9 34.8

- Table 11. Comparisons between the baseline model architectures and those incorporating augmented Q. dhq refers to the intermediate Q head dimension. The number of Q heads is 32 for all models in the table (nhq = 32). For the baseline without AugQ, the intermediate dimension of Q head is dhq = 2048.

Commonsense & Comprehension Continued LM

Model Overall

Hella. ObQA Wino. ARC. PIQA SciQ Bool. Logi. LAMB.

MHA 52.40 55.6 37.6 57.6 36.0 73.9 85.5 59.6 28.9 36.8 + AugQ (dhq=5632) 53.03 (↑0.63) 57.4 38.0 57.9 39.4 72.9 85.9 60.1 27.3 38.3 GQA (nhk=nhv=16) 52.14 55.1 39.6 56.3 35.4 71.9 85.0 61.4 27.8 36.8 + AugQ (dhq=3072) 53.38 (↑1.24) 56.6 40.2 56.1 40.0 73.2 87.3 61.0 28.3 37.7 + AugQ (dhq=4096) 52.93 (↑0.79) 56.7 40.8 56.9 37.1 73.6 83.5 61.7 28.0 38.1 + AugQ (dhq=5632) 53.07 (↑0.93) 57.3 39.8 57.3 36.4 74.2 83.6 61.4 28.7 39.0 GQA (nhk=nhv=4) 51.66 54.0 38.0 56.0 37.5 72.3 82.0 61.3 28.6 35.4 + AugQ (dhq=5632) 53.13 (↑1.47) 56.5 40.8 58.2 37.6 73.6 84.7 61.2 27.5 37.9

attention scores, in the attention output calculation can still preserve the model performance. As in previous work, topk V can improve the memory efficiency of the model (Gupta et al., 2021), inspiring us to adopt this strategy in this KV imbalance scenario. As shown in Table 8, even when we only use 100 V vectors corresponding to the top100 highest attention scores (denoted as “Sel.V-top100”), the model can still maintain a comparable effect to the original setting. These results indicate that we only need to retrieve a small proportion of V vectors from the cache during inference to achieve competitive performance. It significantly reduces data transfer, and thereby improves inference speed.

- Table 12. Comparisons of the model performance when incorporating the augmented Q component (AugQ) with different sizes and

enlarging the FFN module (AugF). The baseline method is GQA, with the FFN dimension being 5632 and nhk=nhv=16. ∆dF denotes the enlarged dimension for the FFN module, while dhq represents the intermediate Q head dimension (δ=3072).

Model Overall

Commonsense & Comprehension Continued LM

Hella. ObQA Wino. ARC. PIQA SciQ Bool. Logi. LAMB.

GQA 52.14 55.1 39.6 56.3 35.4 71.9 85.0 61.4 27.8 36.8 + AugF (∆dF=δ) 53.26 (↑1.12) 57.6 39.6 57.3 38.5 73.2 87.3 59.0 27.6 39.2 + AugQ (dhq=δ) 53.38 (↑1.24) 56.6 40.2 56.1 40.0 73.2 87.3 61.0 28.3 37.7

- + AugF (∆dF=2δ) 53.16 (↑1.02) 59.3 40.0 57.0 38.3 73.9 85.2 59.8 24.9 40.1

+ AugF (∆dF=δ) & AugQ (dhq=δ) 54.55 (↑2.41) 58.8 41.8 57.5 39.7 74.4 86.9 62.4 28.1 41.5

- + AugF (∆dF=3δ) 54.50 (↑2.36) 60.5 42.4 59.8 39.8 74.7 87.3 59.9 27.0 39.2

- + AugF (∆dF=2δ) & AugQ (dhq=δ) 54.67 (↑2.53) 60.5 41.6 57.4 39.9 74.7 87.3 59.6 27.3 43.8

+ AugF (∆dF=5δ) 55.08 (↑2.94) 62.4 41.4 57.3 41.3 75.3 88.3 60.6 26.6 42.5

- + AugF (∆dF=3δ) & AugQ (dhq=δ) 55.09 (↑2.95) 61.6 39.8 61.0 40.5 75.1 88.7 59.6 28.3 41.2

- Table 13. Combinations of three strategies for optimizing the self-attention architecture: augmented Q, compressing the number of K heads, and compressing K head dimension. ✓ and ✗ represent whether the corresponding strategy is used or not respectively. If the strategy is not used, the standard model setting is adopted (i.e. nhk=nhv=16, and Q is not augmented).

Commonsense & Comprehension Continued LM AugQ -75% K Heads Half K Dim

Model Config(nhk=nhv=16)

Overall

Hella. ObQA Wino. ARC. PIQA SciQ Bool. Logi. LAMB. (dhq=3072) (nhk=4) (dhk=dhv/2)

- ✗ ✗ ✗ 52.14 55.1 39.6 56.3 35.4 71.9 85.0 61.4 27.8 36.8

✓ ✓ ✗ 52.97 56.0 38.4 57.7 37.0 72.5 83.8 62.5 30.1 38.8 ✓ ✗ ✓ 52.72 55.9 39.4 59.1 37.5 73.1 84.3 60.6 27.0 37.4

- ✗ ✓ ✓ 51.74 54.3 37.4 57.5 36.3 72.9 85.5 60.5 26.3 35.1

✓ ✓ ✓ 52.61 55.8 40.2 54.9 38.5 74.0 84.6 61.1 26.9 37.6

#### B.3. Detailed Evaluation Results

Due to the space limitation of the main text, in Table 1-4, we only include the average results across nine benchmarks. Here, we report the detailed evaluation results on each benchmark in Tables 9–12.

#### B.4. Combination of Multiple Optimization Strategies

As shown in Table 13, by integrating three optimization strategies (i.e., -75% K Heads (nhk=4), Half K Dim(dhk= dhv/2), and AugQ (dhq=3072), we observe that the model’s performance surpasses that of the original model and also exhibits advantages in terms of inference speed and KV cache utilization.

Table 14. Key configurations and hyperparameters of SIGMA.

Parameter \ Scale 1.5B 10B

Layers 26 32 Hidden Dimension 2,048 4,096 FFN Dimension 6,144 14,336 Aug Q Dimension 3,072 6,144 Attention Heads 32 32 Key Heads 4 4 Value Heads 16 16 Peak Learning Rate 4.0e-4 1.5e-4 Activation Function SwiGLU SwiGLU Vocabulary Size 128,256 128,256 Positional Embeddings ROPE (θ =50,000) ROPE (θ =500,000)

#### B.5. SIGMA Model Architecture

Building on the DiffQKV attention, we construct a pretrained language model, named SIGMA. Specifically, we adopt two model scales with 1.5B parameters and 10B parameters, respectively (i.e. SIGMA-1.5B and SIGMA10B). For the sake of balancing the model performance and the cost of the KV cache, during the training of SIGMA-1.5B and SIGMA-10B, no dimension compression is applied to the K heads. Only the number of K heads was decreased. Concretely, the K head is set to 4, and the number of V heads,

which is half of the number of Q heads, is set to 16. For SIGMA-1.5B, we set dhq = 3072, and for SIGMA-10B, we set dhq = 6144, corresponding to 1.5 times the dimension of the hidden state, so as to extend the representational space of Q. We utilize the same vocabulary as Llama3 (Dubey et al., 2024), with a vocabulary size of 128k. For more detailed configurations of the SIGMA architecture, please refer to Table 14.

### C. Evaluation Procedure on AIMICIUS Benchmark

- C.1. CMDGen We consider the following evaluation metrics on the CMDGen task:

- • CMD Score computes the cosine similarity between the embeddings of the generated command and the ground-truth command. The embeddings are encoded with all-MiniLM-L6-v2.5
- • Output Score evaluates the cosine similarity between the execution results of the generated command and that of the ground-truth command. The embeddings are also obtained with all-MiniLM-L6-v2.
- • Calibration Score serves as an approximate measure of accuracy. For a given test sample, the calibration score is assigned as one if either the CMD Score or the Output Score exceeds a predefined threshold.
- • Exact Match checks if the output command exactly matches the ground-truth command.
- • Success Ratio measures if the execution result of the generated command is similar enough to that of the ground-truth command.
- • Accuracy serves as a comprehensive measure of the model’s overall performance. For a given test sample, a generated command is deemed accurate if it either exactly matches the ground-truth command (Exact Match = 1) or produces highly similar execution results to the ground-truth (Success Ratio = 1). We consider it as the primary metric on the CMDGen task.

#### C.2. Infrawise

DCW generation aims to produce a JSON object, referred to as a DCW, based on the user instructions. Specifically, DCW stands for:

- • Design, a kind of virtual machine (probably with a specific feature) or a GPU type. It defines the testbed to be evaluated and should be inferred based on the user’s intentions. The design has two sub-keys: baseline and target, implying the baseline design and target design user wants to retrieve. Commonly used values include NDv4, NDmv4, NDv4 MI200, NDv5, NCv3, A100, H100, MI200 and MI300x.

- • Workload: the benchmarks, models, algorithms, or performance measurement tools that the user intends to run on the design testbed.
- • Criterion: the evaluation principles or performance metrics the user intends to apply. It specifies how the effectiveness, quality, or performance between the two designs should be compared. Common criteria include evaluating both testbeds’ peak performance, assessing cost-effectiveness, or measuring results within a specified time duration.

Benchmark result retrieval entails two substeps for enhanced performance. In the first phase, a set of benchmark results (typically 10 samples) is retrieved based on initial filtering criteria. In the second phase, LLMs are instructed to select all the appropriate benchmark results from this set, guided by the generated DCW. If none of the retrieved results align with the DCW specifications, the final answer would be marked as “None.”

We adopt the following metrics to conduct evaluation on Infrawise:

- • Target, Baseline, Criterion, Workload: These four metrics evaluate individual components of the generated results. It equals one if the generated result (e.g., Target Design, Baseline Design, Criterion, Workload) exactly matches the ground truth.
- • DCW: This composite metric evaluates the correctness of the entire DCW. It equals one if all four components: Target, Baseline, Criterion, and Workload, are correct (i.e., each equals one individually).
- • Benchmark Result Recall: This metric measures the completeness of the retrieval phase. It equals one if all suitable benchmark results are retrieved without omission. 5https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

- • Benchmark Result Accuracy: The most critical metric. For a given test sample, its benchmark result accuracy is assigned as one only if all suitable benchmark results are retrieved and no irrelevant results are included.

- C.3. Optiflow The evaluation of Optiflow focuses on assessing the quality of the generated Python code using four key metrics, described

- as follows:

- • Code Detected: This metric verifies whether the output includes a Python code block. If a code block is present, the metric is set to one.
- • Code Executable: This metric evaluates the validity of the Python code. If the generated code can be executed without errors, the metric equals one.
- • Plan Valid: This metric assesses whether the generated Python code provides a valid plan based on the specified SKU. The metric equals one if the plan meets the given requirements.
- • Plan Improved: Each test case involves both Plan Generation and Plan Improvement tasks. The evaluator first measures the latency of the initial output provided by the LLM and then requests an improved plan. This metric equals one if the latency of the second output is reduced compared to the first.

#### C.4. NL2KQL

A typical KQL query consists of four key components: CLUSTER, DATABASE, TABLE, and COLUMN, taking the structure of:

cluster(CLUSTER).database(DATABASE).TABLE | where COLUMN

The evaluation of NL2KQL focuses on the exact match of each KQL component and the overall syntax accuracy of the generated queries. Specifically, the evaluation metrics include:

- • Syntax Accuracy, which measures whether the generated KQL query adheres to the correct syntax;
- • Similarity, which assesses the Jaccard similarity between the generated KQL query and the ground-truth query;
- • Cluster Score, Database Score, Table Score, Column Score, used to assess the correctness of individual components (CLUSTER, DATABASE, TABLE, and COLUMN) by checking for the recall rate with the corresponding elements in the ground-truth query.

### D. Data Examples of AIMICIOUS Benchmark

- D.1. CMDGen Examples

|CMDGen Example 1: Generate a Command to Use “nvidia-smi pmon”.|
|---|
|Question. How to use 'nvidia-smi pmon' to monitor GPU with ID 0, updating every 4 seconds for a total of 10 times, and<br><br>logging the output to '\/tmp\/nvidia-log.txt’?<br><br>Answer.<br><br>nvidia-smi pmon -i 0 -d 4 -c 10 --filename \tmp\nvidia-log.txt|

|CMDGen Example 2: Generate a Command to Use Superbenchmark.|
|---|
|Question. How do you generate the baseline of benchmarking results from a jsonline file according to rules defined in rule files, using the data file located at `/root/sb-workspace/sb_test/generate_baseline_results.jsonl`, the summary rule file located at `/root/sb-workspace/sb_test/generate_baseline_summary_rules.yaml`, the diagnosis rule file located at `/root/sb-workspace/sb_test/generate_baseline_diagnosis_rules.yaml`, enabling debug mode to increase logging verbosity, and specifying the output directory as `/root/sb-workspace/outputs`? Use full names for parameters whenever possible.<br><br>Answer.<br><br>sb result generate-baseline --data-file \root\sb-workspace\sb_test\generate_baseline _results.jsonl --summary-rule-file \root\sb-workspace\sb_test\generate_baseline_summ ary_rules.yaml --debug --decimal-place-value 5|

- D.2. NL2KQL Examples

|NL2KQL Example 1.|
|---|
|Natural Language.<br><br>Check which UFM device in previous 1d under the cluster prefixed with phx61.<br><br>KQL.<br><br>cluster("azphynet").database('azdhmds').SyslogData<br><br>| where TIMESTAMP > ago(1d) | where Device startswith "phx61” | where Device endswith "ufm”<br><br>| distinct Device, DeviceIp|

- D.3. Infrawise Examples

|Infrawise Example 1: DCW Generation.|
|---|
|User Intention. Would you please tell me how the ort inference performance is on A100?<br><br>DCW. {<br><br>"Design": {"Target": "A100", "Baseline": ""}, "Criterion": "default", "Workload": "ORT inference"<br><br>}|

|Infrawise Example 2: Benchmark Result Retrieval.|
|---|
|Question.<br><br>Find benchmark results for workloads related to ORT inference on hardware or sku most similar to the A100, and<br><br>only return the index in the json without any explanation. Perform the following steps:<br><br>Step 1. For each json item, identify the most similar hardware or sku to the A100 by comparing the 'hardware' and 'sku' fields.<br>Step 2. For all json items with an A100 in the 'hardware' or 'sku' field, to identify items whose "workload" field is VERY similar\/consistent to ORT inference. Note: dtoh and htod refer to GPU to CPU and CPU to GPU, respectively; 'nccl reduce performance' and 'nccl allreduce performance' are very different workloads;<br><br>**"matmul" and "CUBLAS functions' performance" are very different workloads**; FP64, FP32, FP16 and FP8<br><br>are very different workloads.<br><br>Step 3. (This is very important!!!) Examine the selected json items to ENSURE that whose 'hardware' field or 'sku' field is VERY consistent to A100 and 'workload' field is VERY similar to ORT inference. Otherwise, remove this item. If no json item is found, output None.<br><br><br>context:<br><br>{"hardware": "A100 GPU", "sku": "NDmv4", "workload": "InfiniBand<br><br>bandwidth(loopback)", "results": {"ib_write_bw_8388608": 25.0}, "index": 0}<br><br>{"hardware": "A100 GPU", "sku": "NDmv4", "workload": "AI ResNet model<br><br>training performance, resnet101", "results": {"fp16_train_throughput":<br><br>809.63, "fp32_train_throughput": 554.47}, "index": 1}<br><br>{"hardware": "MI300x GPU", "gpu firmware version": "bkc24.06", "sku":<br><br>"MI300x pv4", "workload": "GPT-175B Mimic", "results":<br><br>{"inference_batch_size_1_latency": 5.27, "inference_batch_size_64_latency": 7.88, "inference_batch_size_2048_latency": 72.6}, "index": 2} {"hardware": "A100 GPU", "sku": "NDmv4", "workload": "AI BERT model training performance, bert-large", "results": {"fp16_train_throughput": 158.35, "fp32_train_throughput": 108.76}, "index": 3} {"hardware": "A100 GPU", "sku": "NDmv4", "workload": "AI ResNet model training performance, resnet152", "results": {"fp16_train_throughput": 579.24, "fp32_train_throughput": 394.86}, "index": 4} {"hardware": "H100 GPU", "sku": "NDv5", "workload": "GPT-175B Mimic", "results": {"inference_batch_size_1_latency": 6.59, "inference_batch_size_64_latency": 7.56, "inference_batch_size_2048_latency": 50.44}, "index": 5} {"hardware": "MI300x GPU", "gpu firmware version": "bkc23.49", "sku": "MI300x pv1", "workload": "GPT-175B Mimic", "results":<br>{"inference_batch_size_1_latency": 6.42, "inference_batch_size_64_latency": 12.65, "inference_batch_size_2048_latency": 87.71}, "index": 6} {"hardware": "MI300x GPU", "gpu firmware version": "bkc23.49", "sku": "MI300x pv1", "workload": "InfiniBand bandwidth(loopback)", "results": {"ib_write_bw_8388608": 47.16}, "index": 7} {"hardware": "A100 GPU", "sku": "NDmv4", "workload": "AI BERT model training performance, bert-base", "results": {"fp16_train_throughput": 387.07, "fp32_train_throughput": 293.79}, "index": 8} {"hardware": "MI300x GPU", "gpu firmware version": "bkc23.49", "sku": "MI300x pv1", "workload": "AI bert model training performance, bert-base", "results": {"fp16_throughput": 538.66, "fp32_throughput": 279.0}, "index": 9}<br><br><br>Answer.<br><br>None.|

#### D.4. Optiflow Examples

|Optiflow Example 1: Generate a Plan for NDV5.|
|---|
|Latency adjacency matrix (in us).<br><br>| | GPU-0 | GPU-1 | GPU-2 | GPU-3 | GPU-4 | GPU-5 | GPU-6 | GPU-7 | NVSwitch |<br><br>| GPU-0 | 0 | inf | inf | inf | inf | inf | inf | inf | 0.2 |<br>| GPU-1 | inf | 0 | inf | inf | inf | inf | inf | inf | 0.2 |<br>| GPU-2 | inf | inf | 0 | inf | inf | inf | inf | inf | 0.2 |<br>| GPU-3 | inf | inf | inf | 0 | inf | inf | inf | inf | 0.2 |<br>| GPU-4 | inf | inf | inf | inf | 0 | inf | inf | inf | 0.2 |<br>| GPU-5 | inf | inf | inf | inf | inf | 0 | inf | inf | 0.2 |<br>| GPU-6 | inf | inf | inf | inf | inf | inf | 0 | inf | 0.2 |<br>| GPU-7 | inf | inf | inf | inf | inf | inf | inf | 0 | 0.2 | | NVSwitch | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0 |<br><br><br>Bandwidth adjacency matrix (in GB/s).<br><br>| | GPU-0 | GPU-1 | GPU-2 | GPU-3 | GPU-4 | GPU-5 | GPU-6 | GPU-7 | NVSwitch |<br><br>| GPU-0 | inf | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 450 |<br>| GPU-1 | 0 | inf | 0 | 0 | 0 | 0 | 0 | 0 | 450 |<br>| GPU-2 | 0 | 0 | inf | 0 | 0 | 0 | 0 | 0 | 450 |<br>| GPU-3 | 0 | 0 | 0 | inf | 0 | 0 | 0 | 0 | 450 |<br>| GPU-4 | 0 | 0 | 0 | 0 | inf | 0 | 0 | 0 | 450 |<br>| GPU-5 | 0 | 0 | 0 | 0 | 0 | inf | 0 | 0 | 450 |<br>| GPU-6 | 0 | 0 | 0 | 0 | 0 | 0 | inf | 0 | 450 |<br>| GPU-7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | inf | 450 | | NVSwitch | 450 | 450 | 450 | 450 | 450 | 450 | 450 | 450 | inf |<br><br><br>Initial Plan.<br><br>(Env Vars, Omitted)<br><br>def generate_plan(GPU_COUNT, CHUNKS_PER_GPU):<br><br>plan = []<br><br>(Generate Plan Logits, Omitted)<br><br>return plan<br><br>plan = generate_plan(GPU_COUNT, CHUNKS_PER_GPU)<br><br>(Print Plan, Omitted)<br><br>Answer.<br><br>(Comments, Omitted)<br><br>new_plan = []<br><br>for k in range(7): for i in range(8):<br><br>receiver = (i + k + 1) % 8<br><br>new_plan.append(f"D({i}, 0): G({i}) => G({receiver})")<br>new_plan.append(f"D({i}, 1): G({i}) => G({receiver})")<br><br><br>(Print Plan, Omitted)|

|Optiflow Example 2: Improve a Plan for NDV5.|
|---|
|Previous Plan.<br><br>(Omitted, See Optiflow Example 1)<br><br>Latency. 7.78 us<br><br>Answer.<br><br>(Comments, Omitted)<br><br>new_plan = []<br><br>for round in range(14): part = []<br><br>for i in range(8):<br><br>sender = i<br><br>receiver = (i + round) % 8<br><br>if sender != receiver: for j in range(2):<br><br>part.append(f"D({sender}, {j}): G({sender}) => G({receiver})")<br><br>new_plan.append(part)<br><br>(Print Plan, Omitted)|

- E. Official Implementation of KV Group Sharing Here, we release the official implementation of KV Group Sharing to ensure compatibility with mainstream LLM frameworks.

|def kv_group_sharing(self, key_states, value_states): """ This is for balance the number of KV heads to facilitate subsequent flash<br><br>attention calculations<br><br>""" batch, _, slen, head_dim = key_states.shape<br><br>if self.num_key_heads > self.num_value_heads: n_rep = int(self.num_key_heads / self.num_value_heads) value_states = value_states[:, :, None, :, :].expand(batch, self.<br><br>num_value_heads, n_rep, slen, head_dim) return key_states, value_states.reshape(batch, self.num_value_heads * n_rep, slen, head_dim)<br><br>else: n_rep = int(self.num_value_heads / self.num_key_heads) if n_rep == 1:<br><br>return key_states, value_states key_states = key_states[:, :, None, :, :].expand(batch, self.num_key_heads, n_rep, slen, head_dim) return key_states.reshape(batch, self.num_key_heads * n_rep, slen, head_dim), value_states<br><br>|
|---|

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16

You can use this code in the forward function like this:

- 1 def forward(

- 2 self,

- 3 hidden_states: torch.Tensor,

- 4 attention_mask: Optional[torch.LongTensor] = None,

- 5 position_ids: Optional[torch.LongTensor] = None,

- 6 past_key_value: Optional[Cache] = None,

- 7 output_attentions: bool = False,

- 8 use_cache: bool = False,

- 9 cache_position: Optional[torch.LongTensor] = None,

- 10 ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

- 11 if isinstance(past_key_value, StaticCache):

- 12 raise ValueError(

- 13 "‘static‘ cache implementation is not compatible with ‘ attn_implementation==flash_attention_2‘ "

- 14 "make sure to use ‘sdpa‘ in the mean time, and open an issue at https:// github.com/huggingface/transformers"

- 15 )

- 16 output_attentions = False

- 17 bsz, q_len, _ = hidden_states.size()

- 18

- 19 query_states = self.q_proj(hidden_states)

- 20

- 21 key_states = self.k_proj(hidden_states)

- 22

- 23 query_states = self.q_down_proj(self.act_fn(self.q_gate_proj(query_states)) * self.q_up_proj(query_states))

- 24

- 25 value_states = self.v_proj(hidden_states)

- 26

- 27 # Flash attention requires the input to have the shape

- 28 # batch_size x seq_length x head_dim x hidden_dim

- 29 # therefore we just need to keep the original shape

- 30

- 31 query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim). transpose(1, 2)

- 32 key_states = key_states.view(bsz, q_len, self.num_key_heads, self.head_dim). transpose(1, 2)

- 33 value_states = value_states.view(bsz, q_len, self.num_value_heads, self.head_dim)

.transpose(1, 2)

- 34

- 35 cos, sin = self.rotary_emb(value_states, position_ids)

- 36 query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

- 37

- 38 if past_key_value is not None:

- 39 # sin and cos are specific to RoPE models; cache_position needed for the static cache

- 40 cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

- 41 key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

- 42

- 43 key_states, value_states = self.kv_group_sharing(key_states, value_states)

- 44 query_states = query_states.transpose(1, 2)

- 45 key_states = key_states.transpose(1, 2)

- 46 value_states = value_states.transpose(1, 2)

- 47

- 48 (Code Omitted.)

- 49

- 50 ’’’

- 51 flash attention

- 52 ’’’

- 53 attn_output = self._flash_attention_forward(

- 54 query_states, key_states, value_states, attention_mask, q_len, dropout= dropout_rate

- 55 )

- 56

- 57 attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

- 58 attn_output = self.o_proj(attn_output)

- 59

- 60

- 61 if not output_attentions:

- 62 attn_weights = None

- 63

- 64 return attn_output, attn_weights, past_key_value

### F. Official Implementation of Efficiency Recording

Here, we offers an official implementation to use Cuda Event Elapsed Time to record the cost of attention computation in modeling sigma.py.

|global total_attention_cost start = torch.cuda.Event(enable_timing=True) end = torch.cuda.Event(enable_timing=True) start.record() attn_output = flash_attn_func(<br><br>query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal<br><br>) end.record() end.synchronize() latency = start.elapsed_time(end) total_attention_cost += latency<br><br>|
|---|

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11

##### To use Kernel Execution Time to record the cost, there is no need to edit the modeling python file, we can directly use a command line tool to implement this: nsys profile --output xxx --stats=true -t cuda python xxx.py.

### G. Detailed Efficiency Results

- Table 15. KET Results (ns) with the prefix length increase from 2k to 32k, keeping the output length as 10. Split represents the split kernel and Combine represents the combine kernel.

Prefix STD SIGMA 1.5B Relative Improvement Length Split Combine Total Cost Split Combine Total Cost Split Combine Total Cost

2k 2.53E+6 1.88E+6 4.41E+6 2.50E+6 1.85E+6 4.34E+6 1.17% 1.68% 1.39% 4k 4.68E+6 1.91E+6 6.59E+6 3.49E+6 1.91E+6 5.40E+6 25.33% 0.08% 18.02% 16k 1.52E+7 1.94E+6 1.72E+7 1.12E+7 1.94E+6 1.31E+7 26.30% 0.25% 23.35% 32k 2.75E+7 1.99E+6 2.95E+7 2.00E+7 2.01E+6 2.21E+7 27.21% -0.93% 25.31%

[Figure 12]

[Figure 13]

[Figure 14]

(a) Output Length = 2k. (b) Output Length = 4k. (c) Output Length = 8k.

[Figure 15]

[Figure 16]

[Figure 17]

(d) Output Length = 16k. (e) Output Length = 32k. (f) Output Length = 64k.

[Figure 18]

[Figure 19]

(g) SIGMA’s absolute CEET improvment vs STD. (h) SIGMA’s relative CEET improvment vs STD.

- Figure 4. CEET comparison of augmented Q between Standard model(STD) and SIGMA. From (a) to (f), the output length increases progressively from 2k to 64k tokens.

Here, we report detailed KET results and CEET results of three modules: KV Cache, Attention Computation, and Augmented Q, as shown in Figures 4–6 and Table 15.

[Figure 20]

[Figure 21]

[Figure 22]

(a) Output Length = 2k. (b) Output Length = 4k. (c) Output Length = 8k.

[Figure 23]

[Figure 24]

[Figure 25]

(d) Output Length = 16k. (e) Output Length = 32k. (f) Output Length = 64k.

[Figure 26]

[Figure 27]

(g) SIGMA’s absolute CEET improvment vs STD. (h) SIGMA’s relative CEET improvment vs STD.

- Figure 5. CEET comparison of KV cache between Standard model(STD) and SIGMA. From (a) to (f), the output length increases progressively from 2k to 64k tokens.

CEET Results - Augmented Q. The augmented Q module is not inherently included within the standard model(STD). In this experiment, we attach the augmented Q module to STD to demonstrate that the augmented Q module’s cost remains consistent regardless of variations in other parts of the model or different context lengths. This setup ensures the reliability of efficiency evaluations for other modules. The CEET results shown in Figure 4 can strongly testify to it. The relative improvement ratio in Figure 4h for Augmented Q offers clearer support to our hypothesis, as it consistently hovers near zero across nearly all settings. To provide a more rigorous validation, we perform a t-test to evaluate whether the relative improvement ratio of Augmented Q has an expectation of zero. The computed T-value is −0.214, and the corresponding P-value is 0.832, indicating no evidence to reject our hypothesis.

CEET Results - KV Cache. CEET results of KV cache can be found in Figure 5. CEET here primarily reflects the cost of loading and storing the KV cache. These operations are largely proportional to the size of the key and value matrices with minimal extraneous influencing factors. Therefore, the results for KV cache align most closely with our theoretical analysis.

In Figures 5a–5f, SIGMA demonstrates a significantly lower cost for KV cache operations. As the output length increases from Figure 5a to Figure 5e, the CEET gap between SIGMA and STD in KV cache becomes increasingly pronounced. As illustrated in Figure 5h, when the prefix and output lengths reach 64k, SIGMA achieves a speedup of 36.57% compared to STD, which is very close to the theoretically calculated improvement rate of 37.5%.

[Figure 28]

[Figure 29]

[Figure 30]

(a) Output Length = 2k. (b) Output Length = 4k. (c) Output Length = 8k.

[Figure 31]

[Figure 32]

[Figure 33]

(d) Output Length = 16k. (e) Output Length = 32k. (f) Output Length = 64k.

[Figure 34]

[Figure 35]

(g) SIGMA’s absolute CEET improvment vs STD. (h) SIGMA’s relative CEET improvment vs STD.

- Figure 6. CEET comparison of attention computation between Standard model(STD) and SIGMA. From (a) to (f), the output length increases progressively from 2k to 64k tokens.

CEET Results - Attention Computation. CEET results of the attention computation are demonstrated in Figure 6. We observe a steady increase in the relative improvement of Attention Computation. The ratio deviates more significantly from the theoretical value compared to the one of KV Cache. This observation aligns with our analysis in §3.1, as Attention Computation involves a greater number of operations that are not directly influenced by the number of key heads, e.g. flash fwd splitkv combine kernel. These operations will partially dilute the efficiency improvement we achieve. Besides, the relative improvement ratio here is also lower than the observed KET improvement shown in Figure 1. The underlying reason is the same: since CEET measures the end-to-end time of attention computation, it accounts for additional

overhead, such as context switching and other CPU operations, making the improvement less pronounced.

It is also important to note that when the prefix length increases and the output length is kept fixed, the relative improvement still increases. As shown in Figure 6h, SIGMA achieves up to a 15.4% reduction in the CEET of attention computation and becomes increasingly efficient as the prefix or output lengths increase. This is attributed not only to the reduction in the number of key heads but also to the great capability of flash attention kernels to handle long sequences efficiently.

### H. Additional Performance Evaluations

#### H.1. Detailed Pre-training Settings

The pre-training data includes general domain data and some domain-specific property data, amounting to total of 6 trillion tokens. For general domain data, we combine DCLM (Li et al., 2024) and FineWeb-EDU (Penedo et al., 2024) and then remove duplicates to obtain General Dataset I. The number of tokens is approximately 4 trillion. After further quality filtering, we take the filtered data as General Dataset II, with the number of tokens 1 trillion tokens. Finally, we select the data with higher scores and stricter rules as General Dataset III, which contains approximately 200 billion tokens that participate in the annealing phase of SIGMA’s pre-training. For data in the math domain, we use proof-pile-2 (Azerbayev et al., 2023) and combine it with 280 billion math-related data filtered from General Dataset I as the pre-training data for the math domain. Regarding data in the code domain, we refer to the filtering method of StarcoderV2 (Li et al., 2023) and select the dataset related to the code domain of 500 billion tokens. Moreover, we have approximately 1 trillion tokens of synthesized and rewritten pre-training data that have undergone quality screening and contain content from multiple domains, which participate in the later phase of SIGMA pre-training.

We conducted pre-training for SIGMA-1.5B utilizing 512×A100-40G GPUs and for SIGMA-10B using 256×H100-80G GPUs. Initially, we combined datasets from general, mathematical, and coding domains with General Dataset I, adhering to a distribution ratio of General:Math:Code = 8:1:1. The training was carried out with a maximum learning rate of 1.5e-4, starting from a batch size of 4 million tokens, which was incrementally scaled up to 16 million tokens, culminating in a total training volume of 3.5 trillion tokens.

In the subsequent phase, we adjusted the data mix to favor mathematical and coding content, employing the quality-filtered General Dataset II. The mixing ratio for this phase was set to General: Math: Code = 4:3:3, with a total training volume of 1.0 trillion tokens.

For the third phase, we introduced a blend of synthesized and rewritten pre-training data alongside the General Dataset II at a ratio of 6: 4. This phase encompassed a total training volume of 1 trillion tokens, during which the learning rate was reduced to 20% of its peak value, i.e., 3e-5.

Finally, in the annealing phase, we utilized General Dataset III, which was selected for its highest quality, along with meticulously chosen synthesized and rewritten pre-training data and the system domain data. The learning rate was gradually decreased to zero, concluding with a total training volume of 1 trillion tokens.

#### H.2. Additional Results in System Domain

To assess the quality of our carefully curated system domain data, we also continual pre-train several popular open-source LLMs with our 19.5B system domain pretrain data, including Mistral-7B (Jiang et al., 2023a) and Llama3-8B (Dubey et al., 2024). We further fine-tune them using a preliminary version of our SFT dataset on the CMDGen NVIDIA subtask of AIMICIUS. The results are presented in Table 16. Both models exhibit notable performance improvements after fine-tuning, with absolute improvements of 1.5 and 6.4 on the Accuracy metric, respectively.

#### H.3. Additional Results in General Domain

Problem-Solving Tasks. We also evaluate on three general problem-solving benchmarks MMLU (5-shot) (Hendrycks et al., 2020), MMLU-Pro (5-shot) (Wang et al., 2024), and BBH (3-shot) (Suzgun et al., 2022), two code generation benchmarks HumanEval (0-shot) (Chen et al., 2021) and MBPP (Austin et al., 2021), and two math problem datasets MATH (5-shot) (Hendrycks et al., 2021) and GSM8K (5-shot) (Cobbe et al., 2021). The evaluation results on various general, coding and math problem-solving benchmarks are shown in Table 17. According to the results, SIGMA-1.5B achieves an average of 27.1 on all tasks, which is comparable to strong baseline models such as Phi1 and Gemma-2B.

- Table 16. Performance of different models on CMDGen NVIDIA subtask in AIMICIUS. The postfix of “-S” indicates that the model has been SFTed using a preliminary version of our SFT dataset, while “-P” denotes that the model has been continually pre-trained on our system-domain pre-training dataset.

Model

CMD Score

Output Score

Calibration Score

Exact Match

Success Ratio

Accuracy

Mistral-7B-S 80.6 58.7 62.0 24.9 19.0 30.7 Mistral-7B-P-S 83.4 65.3 66.3 23.9 21.5 32.2 Llama3-8B-S 86.4 69.1 64.4 42.0 32.7 50.7 Llama3-8B-P-S 87.5 72.2 69.3 46.3 37.1 57.1

- Table 17. Comparisons with baseline models on general, coding, and math problem-solving tasks. Differences with original reports in the baseline models are due to our unified re-evaluations for fair comparisons.

General Coding Math

Model Params Avg.

MMLU MMLU-Pro BBH HumanEval MBPP MATH GSM8K Pythia 1.0B 10.76 26.1 11.1 24.6 5.5 4.0 2.1 1.9 TinyLlama 1.1B 11.13 26.7 11.3 25.4 - 10.8 1.9 1.8 Bloom 1.1B 5.20 26.5 2.1 6.5 - - 0.1 1.2 OLMo 1.2B 10.53 26.2 10.8 25.8 5.5 0.3 2.3 2.8 OPT 1.3B 8.95 24.7 10.8 22.8 - - 1.7 2.7 CerebrasGPT 1.3B 9.67 26.6 11.2 24.5 1.8 0.8 1.0 1.8 Phi1 1.3B 19.81 24.9 11.0 22.9 48.2 27.2 2.0 2.5 Pythia 1.4B 10.73 25.7 10.9 24.8 4.9 4.9 2.1 1.7 DCLM 1.4B 17.92 47.7 16.3 29.9 9.1 13.2 2.5 6.8 StableLM2 1.6B 17.88 38.1 8.7 26.8 7.3 17.7 5.2 21.3 SmolLM 1.7B 17.60 29.9 11.7 28.9 1.2 41.0 4.2 6.4 Gemma 2.0B 26.58 41.2 14.7 36.0 25.0 41.5 10.9 16.8 SIGMA (Ours) 1.5B 27.10 47.0 17.6 32.7 21.3 30.7 12.7 27.8

Specifically, our model reaches top-2 performances on all three general benchmarks: MMLU, MMLU-Pro, and BBH, showing its capability to apply world knowledge to various problem-solving scenarios. In the code domain, SIGMA achieves 21.3% and 30.7% pass rates on the HumanEval and MBPP benchmarks, outperforming Phi1-1.3B on MBPP, a model customized for code generation. However, in HumanEval, SIGMA exhibits inferior performance relative to Gemma-2B and Phi1-1.3B, which may be attributed to the relatively lower proportion of code in our pre-training corpus. In the math domain, SIGMA reaches 12.7% and 27.8% precision on the MATH and GSM8K benchmarks, surpassing competitive baseline models such as StableLM-1.6B and Gemma-2B by a large margin. Notably, SIGMA-1.5B maintains similar commonsense reasoning and general problem-solving capabilities to DCLM-1.4B, but greatly advances on all coding and math benchmarks. Despite some overlap in the pre-training corpus adopted by SIGMA and that by DCLM-1.4B, the results presented above further demonstrate the importance of mixing mathematical and coding data into the general corpus throughout the entire pre-training process, which contributes to a more balanced improvement of the model’s general, reasoning, and programming capabilities.

### I. Future Work

Despite its advancements, SIGMA still presents significant opportunities for improvement. For instance, further optimization on the architecture of SIGMA has not been fully explored. Key areas for investigation include the trade-off between Augmented Q and Feed-Forward Network (FFN) parameters, varied key-value (KV) heads compression across layers, and appropriate hyper-parameters for scale up. Additionally, the number of tasks currently included in AIMICIUS is still limited. It is essential to evaluate the model’s ability on a wider spectrum of system domain challenges. Besides, we discover that SIGMA’s performance is largely constrained by the quality of the synthetic data used for pre-training. We believe that model self-evolution and lifelong learning could be possible avenues to overcome this limitation. In future work, we plan to address these issues, unlock the vast potential of SIGMA within the system domain.

