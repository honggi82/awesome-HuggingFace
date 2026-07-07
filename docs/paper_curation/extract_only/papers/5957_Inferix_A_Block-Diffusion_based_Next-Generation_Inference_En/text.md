# arXiv:2511.20714v2[cs.CV]28Apr2026

[Figure 1]

[Figure 2]

[Figure 3]

## Inferix: A Block-Diffusion based Next-Generation Inference Engine for World Simulation

Inferix Team

World models serve as core simulators for fields such as agentic AI, embodied AI, and gaming, capable of generating long, physically realistic, and interactive high-quality videos. Moreover, scaling these models could unlock emergent capabilities in visual perception, understanding, and reasoning, paving the way for a new paradigm that moves beyond current LLM-centric vision foundation models. A key breakthrough empowering them is the semi-autoregressive (block-diffusion) decoding paradigm, which merges the strengths of diffusion and autoregressive methods by generating video tokens in blocks-applying diffusion within each block while conditioning on previous ones, resulting in more coherent and stable video sequences. Crucially, it overcomes limitations of standard video diffusion by reintroducing LLM-style KV Cache management, enabling efficient, variable-length, and high-quality generation.

Therefore, Inferix is specifically designed as a next-generation inference engine to enable immersive world synthesis through optimized semi-autoregressive decoding processes. This dedicated focus on world simulation distinctly sets it apart from systems engineered for high-concurrency scenarios (like vLLM or SGLang) and from classic video diffusion models (such as xDiTs). Inferix further enhances its offering with interactive video streaming and profiling, enabling real-time interaction and realistic simulation to accurately model world dynamics. Additionally, it supports efficient benchmarking through seamless integration of InterVBench, a new fine-grained evaluation benchmark tailored for minute-long video generation scenarios. We hope the community will work together to advance Inferix and foster world model exploration.

Date: April 30, 2026 Code: https://github.com/alibaba-damo-academy/Inferix

1 Introduction

World models are capable of generating interactive, long-form, and physically plausible video sequences. Most current video diffusion models [34] rely on the Diffusion Transformer (DiT) [28]—which uses bidirectional attention without KV caching. While this enables parallelized generation and controllability, decoding is inefficient and restricted to fixed lengths. In contrast, AR-based frameworks [35] support variable-length generation and KV Cache management, but their generation quality lags behind video diffusion, and decoding is not parallelizable. Importantly, block diffusion [13, 33] interpolates between AR and diffusion by reintroducing LLM-style KV Cache management, enabling efficient, variable-length, and high-quality generation, as shown in Figure 1.

The overall framework of Inferix is illustrated in Figure 2. The model generates a clean video block from noise via iterative denoising. Crucially, the attention mechanism at each step leverages a global KV Cache containing context from previously generated blocks. After a new block is generated, its KV information is used to update the cache, providing context for subsequent blocks. This generate-and-cache loop facilitates efficient, arbitrary-length video generation.

A new paradigm inevitably brings forth new infrastructure and fundamental research, just as the LLM era gave rise to vLLM [18] & SGLang [46], the Visual Diffusion era to xDiT [5] and FastVideo [32], and the Post-training era to OpenRLHF [11] and verl [29], etc. Now, the world model era also demands its own dedicated inference engine, and Inferix is purpose-built as a next-gen inference engine, empowering immersive world synthesis via optimized semi-autoregressive decoding paradigm.

- (1) Autoregressive
- (2) Diffusion
- (3) Block Diffusion

Arbitrary-length KV caching

[Figure 4]

[Figure 5]

Not Parallelizable

[Figure 6]

[Figure 7]

Noisy video token

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

| | |
|---|---|

𝑺𝒕𝒆𝒑𝟏 𝑺𝒕𝒆𝒑𝟐 𝑺𝒕𝒆𝒑𝟑

[Figure 13]

Clean video token

Fixed-length

No KV caching Parallelizable

[Figure 14]

[Figure 15]

[Figure 16]

|[Figure 17]|
|---|

|[Figure 18]<br><br>[Figure 19]|
|---|

|[Figure 20]<br><br>[Figure 21]|
|---|

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Video block

𝑻𝒊𝒎𝒆𝒔𝒕𝒆𝒑𝟏 𝑻𝒊𝒎𝒆𝒔𝒕𝒆𝒑𝟐 𝑻𝒊𝒎𝒆𝒔𝒕𝒆𝒑𝟑

Arbitrary-length

KV caching Parallelizable

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

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

𝑩𝒍𝒐𝒄𝒌𝟐 𝑩𝒍𝒐𝒄𝒌𝟏 𝑩𝒍𝒐𝒄𝒌𝟐 𝑩𝒍𝒐𝒄𝒌𝟏 𝑩𝒍𝒐𝒄𝒌𝟐

𝑩𝒍𝒐𝒄𝒌𝟑 𝑩𝒍𝒐𝒄𝒌𝟑 𝑩𝒍𝒐𝒄𝒌𝟑

𝑩𝒍𝒐𝒄𝒌𝟏

- Figure 1 Architecture comparison. AR vs. Diffusion vs. Block Diffusion (Semi-AR). Block Diffusion combines the strengths of both AR and Diffusion, enabling arbitrary-length generation, KV caching, and high parallelizability within each block.

Key features of Inferix are as follows:

- • Next-Generation Inference Paradigm: A block diffusion framework built for immersive world synthesis at scale.
- • Efficient Long Video Generation Benchmarking: Integrated with InterVBench, a fine-grained benchmark for minute-long videos with dedicated metrics to evaluate long-range coherence.
- • Video Streaming: Basic video streaming capabilities for generated content, with both RTMP and WebRTC supported as streaming protocols.
- • Continuous Prompt Support: Enable dynamic narrative control with different prompts for different video segments.
- • Advanced KV Cache Management: Intelligent memory management for persistent world simulation.
- • Distributed World Synthesis: Support multiple parallelism for large-scale immersive environment generation.
- • Built-in Profiling: Performance monitoring and analysis capabilities with enhanced diffusion model profiling.

2 Challenges in Inference of World Simulation

During the inference of world simulation, the world models need to generate long-form video sequences. Moreover, for current world models and video generation models, their model size is pretty large. The large model size and long-form video sequences bring unprecedented pressure to storage and computing.

For storage, the usage of KV Caches is the main bottleneck. In world simulation, the KV Caches of former blocks need to be stored as the context for the generation of current and future blocks, which is important to ease the drifting and forgetting problem [43] when generating long video sequences. However, these KV Caches will consume a large amount of GPU memory. Therefore, how to make KV Cache management efficient is important for the inference of world simulation. Some advanced techniques that have been studied in LLM inference need to be brought to the inference of world simulation, such as PageAttention [18], offload [30, 19], KV Cache compression [26, 21], and so on.

For computation, the large model size and extremely lone video sequences increase the amount of computation greatly. For example, it will consume about 6,800 seconds when generating a 5-second video with Wan2.1 14B in a single NVIDIA H20. For the inference of world simulation, the computation will be much more

Block DiT Pipeline

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

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

###### … …

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

𝒙𝒕 𝒙𝒕 𝟏

Noisy Block Clean Block

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Attention Kernel

KVCache Video Streaming

###### KV Selection

[Figure 90]

……

……

[Figure 91]

Attention Kernel

𝑪𝑷 = 𝟒

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

𝑸 𝑲

𝑽

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

|[Figure 105]|[Figure 106]| |[Figure 107]| |[Figure 108]|
|---|---|---|---|---|---|

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

|𝑲|[Figure 113]| |
|---|---|---|

|𝑽|[Figure 114]|[Figure 115]|
|---|---|---|

𝑸

|𝑲𝑲<br><br>[Figure 116]|[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]|[Figure 120]<br><br>[Figure 121]|[Figure 122]<br><br>[Figure 123]|
|---|---|---|---|

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

|[Figure 135]| |[Figure 136]<br><br>[Figure 137]|
|---|---|---|
| | | |

| |[Figure 138]<br><br>[Figure 139]|[Figure 140]|
|---|---|---|

KVCache Manager

[Figure 141]

[Figure 142]

[Figure 143]

Value

[Figure 144]

[Figure 145]

[Figure 146]

Key

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

|[Figure 164]<br><br>𝑽 𝑽<br><br>[Figure 165]|[Figure 166]|[Figure 167]<br><br>[Figure 168]|[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]|
|---|---|---|---|

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Attention

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

…

…

[Figure 183]

[Figure 184]

Attention

play

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Attention Kernel

[Figure 191]

[Figure 192]

[Figure 193]

Attention Kernel

𝑸 𝑲

𝑽

[Figure 194]

DAX Quantization

LVBench

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

Cache Key & Value tokens

|𝑲| |[Figure 200]|
|---|---|---|

|𝑽| |[Figure 201]|
|---|---|---|

𝑸

[Figure 202]

[Figure 203]

|[Figure 204]<br><br>[Figure 205]|
|---|
|[Figure 206]<br><br>[Figure 207]|
|[Figure 208]<br><br>[Figure 209]|
|[Figure 210]<br><br>[Figure 211]|

|[Figure 212]<br><br>[Figure 213]|
|---|
|[Figure 214]<br><br>[Figure 215]|
|[Figure 216]<br><br>[Figure 217]|
|[Figure 218]<br><br>[Figure 219]|

[Figure 220]

|[Figure 221]| |[Figure 222]<br><br>[Figure 223]|
|---|---|---|
| | | |

| |[Figure 224]<br><br>[Figure 225]|[Figure 226]|
|---|---|---|

| | |[Figure 227]<br><br>[Figure 228]<br><br>[Figure 229]|[Figure 230]| | |
|---|---|---|---|---|---|

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

Quant.

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

Attention

[Figure 240]

[Figure 241]

Attention

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

- Figure 2 Framework of Inferix. To enhance the efficiency of block diffusion models, INFERIX provides a set of interconnected components: efficient parallel strategies, block-wise KV Cache management, DAX [1] quantization, real-time video streaming, and fine-grained video evaluation.

heavier due to the longer context. Therefore, it’s significant to accelerate the computation of world simulation to make it accessible. There are several methods that can be taken to achieve this: quantization to utilize low-bit computation [45, 20], sparse attention [39, 42], decreasing the denoising steps [40, 8], leveraging the redundancy during inference [24, 44], utilizing distributed computation [6, 7], and so on.

- 3 Framework Design

- 3.1 Parallelism

To accelerate the inference process and minimize per-GPU memory footprint, Inferix employs a suite of parallelism techniques tailored for long sequence models. These include Ulysses-style sequence parallelism [16], which partitions independent attention heads across multiple GPUs to relieve memory pressure while preserving computational efficiency, and Ring Attention [25, 38], which enables scalable attention computation over long sequences by distributing attention operations in a ring topology. Depending on the selected attention mechanism, ring attention can either pass queries or pass keys and values, leading to different performance profiles. Inferix selects the most suitable parallelism strategy based on model architecture, network topology, and communication overhead. This adaptive approach ensures optimal resource utilization and performance across deployment scenarios.

- 3.2 KV Management

Block-Diffusion-based models leverage KV caches to accelerate the generation process. To support KV cache access from various kinds of models, Inferix provides a unified KV management interface backed by block-wise KV memory management. To maintain scalability in the face of future-time models requiring both sliding-window access patterns and selective global KV context dependency, the KV management system preserves the extensibility of flexible KV fetching methods including both range-based chunked access and index-based selective fetch. Latent store used in Multi-latent Attention (MLA) [23] and offloading to main memory for GPU memory optimization are also supported to keep the KV management future-proof.

###### Table 1 Overview of the datasets used for constructing InterVBench.

Dataset Video Number Object Classes

##### DanceTrack 66 Humans (66, 100%) GOT-10k 272 Humans (177, 65%) Animals (54, 20%) Environment (41, 15%) HD-VILA-100M 117 Humans (47, 40%) Animals (35, 30%) Environment (35, 30%) ShareGPT4V 545 Humans (381, 70%) Animals (82, 15%) Environment (82, 15%)

InterVBench 1000 Humans (671, 67%) Animals (171, 17%) Environment (158, 16%)

- 3.3 Models and Pipelines

The Inferix framework is designed to support a variety of block diffusion models, currently MAGI-1 [33], CausVid [41], and Self Forcing [13] as examples. These models differ in their foundations: CausVid and Self Forcing are built upon Wan2.1, a 5-second full-attention base diffusion video model, while MAGI-1 is trained from scratch with a distinct infrastructure. To efficiently accommodate this diversity, Inferix first abstracts their shared computational patterns into a generalized inference pipeline. Building on this abstraction, we then design and integrate several key components, such as a sophisticated KV Manager and a suite of parallel strategies, to significantly boost inference performance. Users are welcome to integrate their own models with these abstractions and interfaces.

- 3.4 System Profiling

Inferix provides a built-in performance profiling mechanism that enables end-to-end visibility into resource utilization during inference. The profiler includes three key characteristics:

Near zero overhead. The full profile only incurs minimal overhead of less than 5%, compared with no profiling. Highly customizable. In addition to GPU usage and system-wide metrics, Inferix allows users to add custom metrics during inference. Users can define custom metrics via lightweight hooks or callbacks that execute inline with inference, enabling domain-specific measurements.

Easy to use. The profiler exposes both Python decorator and context manager. The Python decorator enables declarative profiling of individual functions, while the context manager supports block-level instrumentation for broader code regions with almost no code change.

- 3.5 Video Streaming

When generating long videos or executing world simulation, it is important to enable dynamic narrative control with different signals for different video chunks. These signals include prompts, motions, inputs from peripherals and so on. For example, when inference with CausVid, Inferix supports generating a long video whose different video chunks are controlled by different prompts specified by users. If a different prompt is given when generating a new video chunk, Inferix will clear the cross-attention cache to eliminate the influence brought by the former prompt.

### 4 InterVBench

- 4.1 Dataset

To address the challenge of generating minute-long videos, we construct InterVBench, a large-scale benchmark comprising 1,000 long-form videos collected from diverse open-source sources. As summarized in Table 1, we select high-resolution videos exceeding 50 seconds from DanceTrack [31], GOT-10k [12], HD-VILA-100M [37], and ShareGPT4V [3].

To ensure comprehensive temporal coverage and linguistic diversity, we employ GPT-4o as a data engine to generate detailed captions every 2–3 seconds. The prompting pipeline and examples are included in Subsection 4.3. To guarantee annotation quality, we adopt a rigorous human-in-the-loop validation framework across all stages: (1) data sourcing, where annotators filter out low-quality or unsuitable clips; (2) chunk segmentation, where human reviewers ensure temporal coherence and eliminate transition artifacts; and (3) caption verification, where annotators refine automatically generated descriptions for semantic accuracy and temporal alignment. Each validation stage involves at least two independent reviewers to maintain inter-rater reliability. Finally, the curated dataset is divided into an 80/20 train–evaluation split.

- 4.2 Metrics

Evaluating long-form video generation requires assessing both spatial fidelity and temporal stability. Prior studies [22, 27] introduce drift penalties to quantify degradation over time, focusing on identity consistency [10] and perceptual robustness [36]. Inspired by the Mean Absolute Percentage Error (MAPE) and Weighted MAPE [17, 4], we propose a unified metric, Video Drift Error (VDE), which measures relative quality changes across the temporal axis.

Building upon VDE, we design five complementary metrics for long-horizon video evaluation: (1) VDE-Clarity, assessing temporal drift in image sharpness; (2) VDE-Motion, quantifying smoothness of motion dynamics; (3) VDE-Aesthetic, capturing consistency of visual appeal; (4) VDE-Background, measuring spatial stability of scene layouts; and (5) VDE-Subject, detecting identity drift in primary subjects. Lower scores in each indicate stronger temporal consistency. Following prior benchmarks [9, 2], we also integrate five complementary quality dimensions from VBench [15]: Subject Consistency ↑, Background Consistency ↑, Motion Smoothness ↑, Aesthetic Quality ↑, and Image Quality ↑. Together, these metrics form a comprehensive protocol for evaluating long video generation models.

- 4.3 Prompts for InterVBench’s Data Engine

Role. Act as a professional video content analyst. Describe a given video frame in English. Context. The previous frame was described as: "{previous_description}". Use this as context to ensure temporal coherence. Instruction. Write a single, descriptive paragraph that:

- • Identifies the main subject, their specific actions, and expressions.
- • Describes the environment and background, including setting and lighting.
- • Highlights the cinematic quality, such as composition, color palette, and atmosphere (e.g., tense, serene, spectacular).

Constraints. Output must be one coherent paragraph, written in natural language prose, without bullet points or numbered lists. Return. The paragraph description of the current frame.

### 5 Development Roadmap

- • Support more complex KV Management, with flexible block-sparse attention
- • Support finetuning a pretrained video generation model (Diffusion to Semi-AR) & distill models into few steps [14, 40]
- • Support high-concurrency deployment
- • Support more complex distributed inference
- • Improve video streaming usage and performance
- • Support more advanced real-time, interactive streaming capabilities

- 6 Conclusion

We develop Inferix, a block-diffusion based next-generation inference engine for world simulation, which integrates some important features and a new benchmark for long video generation. The inference engine takes the differences between block-diffusion generation and former generation paradigms as the starting point, which intends to make the researches in world model and long video generation more convenient. Inferix unifies the inference interfaces of different block-diffusion models and apply several efficient inference techniques, which are important to improve ease of use. The InterVBench integrated in Inferix aims to evaluate the quality of long video generation precisely and efficiently, which is also valuable for the development of world model.

For future works, more efficient inference techniques specific to block-diffusion generation will be taken into considerations, which includes sparse attention, feature cache, step distillation and so on. We hope Inferix will become a useful tool for this.

## Appendix

A Contributions and Acknowledgments

We are a joint team from Zhejiang University & Hong Kong University of Science and Technology & Alibaba DAMO Academy & Alibaba TRE. All current contributors of Inferix are listed in alphabetical order by their last names.

We warmly welcome everyone to join our virtual team and together harness the collective power of community. Contributors: Tianyu Feng Yizeng Han Jiahao He Yuanyu He Xi Lin Teng Liu Hanfeng Lu Jiasheng Tang Wei Wang Zhiyuan Wang Jichao Wu Mingyang Yang Yinghao Yu Zeyu Zhang Bohan Zhuang

References

- [1] Dax: Diffusion accelerated execution. https://github.com/RiseAI-Sys/DAX, 2025.
- [2] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025.
- [3] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer, 2024.
- [4] Arnaud De Myttenaere, Boris Golden, Bénédicte Le Grand, and Fabrice Rossi. Mean absolute percentage error for regression models. Neurocomputing, 192:38–48, 2016.
- [5] Jiarui Fang, Jinzhe Pan, Xibo Sun, Aoyu Li, and Jiannan Wang. xdit: an inference engine for diffusion transformers (dits) with massive parallelism. arXiv preprint arXiv:2411.01738, 2024.
- [6] Jiarui Fang, Jinzhe Pan, Jiannan Wang, Aoyu Li, and Xibo Sun. Pipefusion: Patch-level pipeline parallelism for diffusion transformers inference. In Advances in Neural Information Processing Systems, 2025.
- [7] Jiarui Fang and Shangchun Zhao. Usp: A unified sequence parallelism approach for long context generative ai, 2024.
- [8] Youping Gu, Xiaolong Li, Yuhao Hu, Minqi Chen, and Bohan Zhuang. Blade: Block-sparse attention meets step distillation for efficient video generation. arXiv preprint arXiv:2508.10774, 2025.
- [9] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589, 2025.
- [10] Wenkang Han, Wang Lin, Yiyun Zhou, Qi Liu, Shulei Wang, Chang Yao, and Jingyuan Chen. Show and polish: reference-guided identity preservation in face video restoration. arXiv preprint arXiv:2507.10293, 2025.
- [11] Jian Hu, Xibin Wu, Wei Shen, Jason Klein Liu, Zilin Zhu, Weixun Wang, Songlin Jiang, Haoran Wang, Hao Chen, Bin Chen, Weikai Fang, Xianyu, Yu Cao, Haotian Xu, and Yiming Liu. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework, 2025.
- [12] Lianghua Huang, Xin Zhao, and Kaiqi Huang. Got-10k: A large high-diversity benchmark for generic object tracking in the wild. IEEE transactions on pattern analysis and machine intelligence, 43(5):1562– 1577, 2019.
- [13] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.
- [14] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion, 2025.
- [15] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [16] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Reza Yazdani Aminadabi, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. System optimizations for enabling training of extreme long sequence transformer models. In Proceedings of the 43rd ACM Symposium on Principles of Distributed Computing, PODC ’24, page 121–130, New York, NY, USA, 2024. Association for Computing Machinery.
- [17] Sungil Kim and Heeyoung Kim. A new metric of absolute percentage error for intermittent demand forecasts. International Journal of Forecasting, 32(3):669–679, 2016.

- [18] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
- [19] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. Infinigen: Efficient generative inference of large language models with dynamic kv cache management, 2024.
- [20] Muyang Li*, Yujun Lin*, Zhekai Zhang*, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by low-rank components for 4-bit diffusion models. In The Thirteenth International Conference on Learning Representations, 2025.
- [21] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation, 2024.
- [22] Zhuoling Li, Hossein Rahmani, Qiuhong Ke, and Jun Liu. Longdiff: Training-free long video generation in one go. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17789–17798, 2025.
- [23] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [24] Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model, 2025.
- [25] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ringattention with blockwise transformers for near-infinite context. In The Twelfth International Conference on Learning Representations, 2024.
- [26] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. Kivi: A tuning-free asymmetric 2bit quantization for kv cache. arXiv preprint arXiv:2402.02750, 2024.
- [27] Yu Lu, Yuanzhi Liang, Linchao Zhu, and Yi Yang. Freelong: Training-free long video generation with spectralblend temporal attention. Advances in Neural Information Processing Systems, 37:131434–131455, 2024.
- [28] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [29] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys ’25, page 1279–1297. ACM, March 2025.
- [30] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. Flexgen: High-throughput generative inference of large language models with a single gpu, 2023.
- [31] Peize Sun, Jinkun Cao, Yi Jiang, Zehuan Yuan, Song Bai, Kris Kitani, and Ping Luo. Dancetrack: Multi-object tracking in uniform appearance and diverse motion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20993–21002, 2022.
- [32] The FastVideo Team. Fastvideo: A unified framework for accelerated video generation, April 2024.
- [33] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.
- [34] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [35] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024.
- [36] Qi Xie, Yongjia Ma, Donglin Di, Xuehao Gao, and Xun Yang. Moca: Identity-preserving text-to-video generation via mixture of cross attention. arXiv preprint arXiv:2508.03034, 2025.
- [37] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5036–5045, 2022.
- [38] Amy Yang, Jingyi Yang, Aya Ibrahim, Xinfeng Xie, Bangsheng Tang, Grigory Sizov, Jeremy Reizenstein, Jongsoo Park, and Jianyu Huang. Context parallelism for scalable million-token inference, 2025.
- [39] Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. In Advances in Neural Information Processing Systems, 2025.
- [40] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis, 2024.
- [41] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. 2025.
- [42] Jintao Zhang, Chendong Xiang, Haofeng Huang, Haocheng Xi, Jun Zhu, Jianfei Chen, et al. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. In International Conference on Machine Learning, 2025.
- [43] Lvmin Zhang, Shengqu Cai, Muyang Li, Gordon Wetzstein, and Maneesh Agrawala. Frame context packing and drift prevention in next-frame-prediction video diffusion models, 2025.
- [44] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention, 2025.
- [45] Tianchen Zhao, Tongcheng Fang, Enshu Liu, Wan Rui, Widyadewi Soedarmadji, Shiyao Li, Zinan Lin, Guohao Dai, Shengen Yan, Huazhong Yang, Xuefei Ning, and Yu Wang. Vidit-q: Efficient and accurate quantization of diffusion transformers for image and video generation, 2024.
- [46] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. Sglang: Efficient execution of structured language model programs, 2024.

