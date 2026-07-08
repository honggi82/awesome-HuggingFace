arXiv:2507.04723v1[cs.CL]7Jul2025

# LOOM-Scope: a comprehensive and efficient LOng-cOntext Model evaluation framework

Zecheng Tang1,2, Haitian Wang1,2, Quantong Qiu1,2, Baibei Ji1,2 Ruoxi Sun1,2, Keyan Zhou1,2, Juntao Li1,2*, Min Zhang1 1Soochow University, China 2Key Laboratory of Data Intelligence and Advanced Computing, Soochow University zctang@stu.suda.edu.cn {ljt,minzhang}@suda.edu.cn

## Abstract

Long-context processing has become a fundamental capability for large language models (LLMs). To assess model’s long-context performance, numerous long-context evaluation benchmarks have been proposed. However, variations in evaluation settings across these benchmarks lead to inconsistent results, making it difficult to draw reliable comparisons. Besides, the high computational cost of long-context evaluation poses a significant barrier for the community to conduct comprehensive assessments of longcontext models. In this paper, we propose LOOM-Scope, a comprehensive and efficient framework for long-context evaluation. LOOM-Scope standardizes evaluation settings across diverse benchmarks, supports deployment of efficient long-context inference acceleration methods, and introduces a holistic yet light-weight benchmark suite to evaluate models comprehensively.1

## 1 Introduction

The ability to process long context is an essential capability for large language models (Bertsch et al., 2024; Meta, 2025), enabling them to address previously challenging areas like longcontext reasoning (Kuratov et al., 2024) and

* Corresponding author. 1Homepage: https://loomscope.github.io

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>Terminal WebUI<br><br>[Figure 21]<br><br>Interface|
|---|

|[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>Benchmark<br><br>[Figure 28]<br><br>[Figure 29]<br><br>RawDataSet<br><br>DataBase<br><br>··· ProcessedData<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]|
|---|

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

###### Results

|[Figure 41]<br><br>Deployment<br><br>Model Accleration<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>Acceleration<br><br>[Figure 47]<br><br>[Figure 48]<br><br>RAG<br><br>[Figure 49]<br><br>[Figure 50]<br><br>Server<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>Architecture<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]|
|---|

[Figure 64]

[Figure 65]

|Evaluator<br><br>[Figure 66]<br><br>Evaluator<br><br>|F1 Score Accuracy ROUGE LLM ······<br><br>|
|---|
<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]|
|---|

Figure 1: Workflow of LOOM-Scope framework.

unlock unbounded external knowledge through the long context input (Ray, 2025). Alongside advancements in long-context language models (LCLMs), recent research in the long-context processing field has increasingly focused on two directions: (1) establishing various benchmarks to evaluate model performance across diverse long-context tasks (Liu et al., 2025) and (2) improving efficiency for long-context inference (Zhou et al., 2024).

Yet, with the increasing number of bench-

marks, inconsistent evaluation results across benchmarks create substantial obstacles in assessing and selecting appropriate LCLMs for different usage. For example, when evaluating the model’s long-context reasoning capability on different benchmarks, LongBench V2 (Bai et al., 2024a) indicates that GLM-9B (GLM et al., 2024) slightly outperforms Llama-3.18B (Meta, 2024), while LongReason (Ling et al., 2025) shows Llama-3.1-8B significantly surpasses GLM-9B. Furthermore, as benchmarks increasingly incorporate longer context distributions and expand their domain coverage, the evaluation requires significant computational resources, e.g., simply evaluating Llama-3.1-8B on RULER (Hsieh et al., 2024) benchmark even costs over 100 H20 (92GB) GPU hours, posing a serious challenge for the community.

To mitigate the aforementioned issues, we introduce LOOM-Scope, a comprehensive and efficient framework for LOng-cOntext Model evaluation. As shown in Figure 1, LOOM-Scope supports two usage modes: via terminal and WebUI, and consists of three key modules: the BENCHMARK module, the DEPLOYMENT module, and the EVALUATOR module. Within each module, the evaluation environment is fully customized to minimize confounding factors across benchmarks, such as instruction prompts and inference hyperparameters, enabling fair and comparable assessments among benchmarks. In addition, to enhance inference efficiency while maintaining model performance, LOOMScope supports three representative long-context optimization techniques: retrieval-augmented generation (RAG) (Liu, 2022), key-value cache optimization (Li et al., 2024a), and trainingfree sparse attention (Zhang et al., 2025a). Our framework is also compatible with efficient inference frameworks such as vLLM (Kwon et al., 2023) and SGLang (Zheng et al., 2024).

Our platform supports 22 long-context benchmarks and more than 140 long-context tasks. To demonstrate the effectiveness of our platform, we construct a comprehensive evaluation set for experiments, LOOMBENCH, by up-sampling from 12 mainstream open-source long-context benchmarks. With LOOM-Scope, this evaluation can be completed very efficiently: for models of 8B scale, a full capability assessment covering 6 different competencies on LOOMBENCH requires only approximately 50 H20 GPU hours or 140 RTX 3090 (24GB) GPU hours, significantly less than the computational cost demanded by most single-capability longcontext benchmarks.

## 2 Framework Design

As shown in Figure 2, the LOOM-Scope platform consists of three key modules, including BENCHMARK module, DEPLOYMENT module, and EVALUATOR module.

### 2.1 BENCHMARK Module

The BENCHMARK module allows for automatic benchmark data detection, downloading, raw data pre-processing, data structure converting. To support efficient distributed inference, the BENCHMARK module automatically allocates an appropriate number of samples to each GPU, ensuring distributed inference balance. Currently, LOOM-Scope supports 22 widely-used long-context benchmarks, covering over 140 sub-tasks, 8k∼2M context range, and spanning 6 major LCLM capabilities: General, Faithfulness, Reasoning, Retrieval, Generation, and Specialization. Details of supporting benchmarks are shown in Appendix B. To ensure fair evaluation and eliminate performance discrepancies caused by prompt variations across benchmarks, BENCHMARK module allows users to define a instruction template applicable to each task.

[Figure 77]

[Figure 78]

[Figure 79]

###### Benchmark

###### Evaluator

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Reasoning

###### Faithfulness

###### General

###### Generative

###### Discriminative

CountingStars

LEval LooGLE RULER LongBench

LongIns LVEval

L-CiteEval LongCite ······

F1 Score BERT Pass@k ROUGE

F1 Score Accuracy Exam ROUGE

LongBench -V2

BABILong ······

BAMBOO

······

Recall Precision

METEOR LLM

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

LLM

······

HUMAN

······

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

###### Specialization

###### Retrieval

###### Generation

NIAH NThread InfiniteBench

LIBRA CLongEval

[Figure 104]

HelloBench LongWriter ······

Augmentation

LongHealth

NoLiMa

LongSafety

[Figure 105]

[Figure 106]

[Figure 107]

Acceleration

LongICLBench

······

LOCOMO

······

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Layer-wise

Sparse Attention

Token Eviction

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Quantization

Token Merge

Hybrid

Model

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

###### Architecture

Server

###### Retrieval-Augmented Generation

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

BM25& LlamaIndex

RWKV Mamba LinearAttention

Contriever OpenAI Haystack LangChain

HF_Models VLLM SGLang API

Transformer

Figure 2: Overview of LOOM-Scope, where the workflow of three modules can be refereed to Figure 1.

### 2.2 DEPLOYMENT Module

API, ensuring flexible integration and scalability for long-context processing.

Following data processing in the BENCHMARK module, the DEPLOYMENT module handles the deployment and inference of models. It supports diverse model architectures (MODEL sub-module) and advanced inference optimization techniques (AUGMENTATION sub-module), including inference servers and augmentation methods. Lists of models and optimization strategies are shown in Appendix B (Table 2).

AUGMENTATION Sub-module To enhance the long-context inference, the AUGMENTATION sub-module supports various inference optimization techniques such as Sparse Attention (Lou et al., 2024), KV-cache optimization (Goel et al., 2025; Hooper et al., 2024a), and RAG-based augmentation (Leng et al., 2024) methods. Specifically, Sparse Attention (Lai et al., 2024; Xu et al., 2025; Jiang et al., 2024) accelerates inference by selectively focusing on relevant tokens, reducing computational overhead. KV-cache optimization (Li et al., 2024b; Liu et al., 2024b; Kang et al., 2024) significantly reduces memory usage and enhances inference speed. RAG-based augmentation (Robertson et al., 2009; Liu, 2022) leverages retrieval mechanisms to enhance model external knowledge, improving performance on tasks requiring exten-

MODEL Sub-module This sub-module aims to support diverse server infrastructures and model architectures, including RWKV (Peng et al., 2023), Mamba (Gu and Dao, 2023), Linear-Attention (Yang and Zhang, 2024), and Transformer-based models. It also supports standardized deployment interfaces, including HF_Models (Wolf et al., 2020), VLLM (Kwon et al., 2023), SGLang (Zheng et al., 2024), and

sive context. This sub-module maintains consistency with the MODEL sub-module, while supporting custom user-defined methods for comprehensive evaluation.

### 2.3 EVALUATOR Module

After obtaining all the model predictions, the EVALUATOR module serves as a comprehensive assessment. It integrates both discriminative and generative evaluation metrics to provide a holistic view of the model’s capabilities in longcontext processing. Discriminative metrics, including F1 Score, Accuracy, Exam, Recall, Precision, and LLM-specific evaluations, are integrated to assess the model’s understanding ability. To assess the model’s generation capability, metrics such as BERT, ROUGE, Pass@k, METEOR, and human evaluations are employed to gauge the quality, coherence, and relevance of the generated text. The module ensures that the outputs of the DEPLOYMENT module are rigorously tested against these diverse metrics. Statistics of the evaluation metrics for each benchmark are shown in Appendix B (Table 3).

### 2.4 Workflow of LOOM-Scope

As shown in Figure 1, the above three modules are finally integrated into the following evaluation workflow: (1) verifies the availability of and pre-process the specified benchmarks (BENCHMARK module), (2) deploys and run LCLMs with specified sever and augmentation methods (DEPLOYMENT module), and (3) conducts evaluation after model prediction (EVALUATOR module).

## 3 Usage and Experiments

LOOM-Scope is accessible via both the command line and the local WebUI interface. For efficient evaluation of LCLM’s capabilities,

LOOM-Scope Usage Command Line loom-scope.run \

--model_path {model path} \

--cfg_path {benchmark config} \

--template {template config} \

--device {device id} \

--gp_num {data parallel size} \

--server {server config} \

--acceleration {augmentation config} \

--eval \

--save_tag {save name} Local Web Interface python WebUI/app.py # will open a gradio

Figure 3: Illustration of two low-code ways to start customized evaluation with LOOM-Scope.

LOOM-Scope also contains an integrated benchmark LOOMBENCH, which is up-sampled and reorganized from 12 different available benchmarks, allowing for comprehensive evaluation of an 8B LCLM within 6 hours. In Appendix B.2, we show the task distribution of LOOMBENCH in Figure 8 and the statistics of each sub-task as well as the evaluation methods.

### 3.1 LOOM-Scope Usage

LOOM-Scope is primarily controlled via userdefined configuration files, enabling a high degree of customization. As illustrated in Fig-

- ure 3, the platform supports evaluation through two low-code interfaces: a command-line interface and a local WebUI, together requiring no more than 11 lines of code in total. For the command-line interface, detailed hyperparameter documentation is available on the official repository2, allowing users to create custom configuration files to fully control the evaluation workflow. The WebUI interface, shown in Fig-
- ure 4, offers an intuitive alternative for users 2https://github.com/LCM-Lab/LOOM-Scope

[Figure 145]

#### Figure 4: Partial snapshot of the local deployment WebUI of LOOM-Scope.

Faithfulness

41

34

General

Specialization

60

59

51

52

78

32 39

91

Reasoning

Generation

56

67

Retrieval

Qwen3-14B

THUDM/glm-4-9b-chat

Qwen3-30B-A3B

Qwen3-8B

Llama-3.1-8B-Instruct

Phi-3-mini-128k-instruct

c4ai-command-r7b-12-2024

Figure 5: Capability radar chart of partial LCLMs.

who prefer a graphical interface. 3.2 Experimental Results

We evaluate on LOOMBENCH with three settings: (1) naive LCLM with HF_Models server, (2) RAG, and (3) inference acceleration methods. For RAG, we adopt both the BM25 and StreamingRAG algorithms with a retrieval chunk size of 16K and evaluate them on four tasks sampling from LOOMBENCH. For inference acceleration methods, experiments are conducted on 128K-length tasks to demonstrate the efficiency.

80

60

###### Score

40

20

0

Faithfulness General Reasoning Retrieval

Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct + BM25 Llama-3.1-8B-Instruct + Self-Route

Qwen3-8B Qwen3-8B + BM25 Qwen3-8B + Self-Route

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 6: RAG results for partial models. Full evaluation results are shown in Appendix C (Figure 10).

Naive HF_Model Results We show the partial results in Figure 5, where it can be observed that the Qwen-3 series models (Yang et al., 2025) exhibit comprehensive long-context capabilities, while other models, e.g., Phi-3 (Abdin et al., 2024), demonstrate relatively strong understanding abilities but struggle with longform generation performance. More evaluation results are shown in Appendix C.

RAG Results We experiment with RAG on two settings: rule-based method (BM25) and model-based method (Self-Route (Li et al., 2024c)). As shown in Figure 6, we observe that the rule-based RAG method underperforms

Model Architecture

Interface Type

Augmentation Method

Custom Benchmark

Benchmark† Num

Framework

Inference Engine

OpenCompass (Contributors, 2023) 15 Transformer Command - VLLM / LMDeploy / API EvalHardness (Gao et al., 2024) 6 Transformer / Mamba Command - VLLM / SGLang / API UltraEval (He et al., 2024) 5 Transformer Command - VLLM / API TAIL (Gu et al., 2024) 1 - Command - VLLM / API TAIL

Transformer / RNN-series / Linear Attn

Sparse Attn / KV Cache / etc.

Command / WebUI

LOOM-Scope (Ours) 22

VLLM / SGLang / API LOOMBENCH

Table 1: Comparison with other frameworks. † denotes that we only count long-context benchmarks.

[Figure 146]

12x

Figure 7: Time cost of acceleration methods.

compared to directly using LCLMs. In contrast, prediction based on the model-based method (Self-Route) can improve the performance. More RAG evaluation results are shown in Appendix C (Figure 10).

Acceleration Method Results We optimized and processed selected acceleration methods, ensuring that all can process the 128K-length context on a single 40GB A100 GPU with Llama3.1-8B-instruct. After optimization, we evaluated RULER using the LOOM-Scope framework, sampling 15 data instances per subtask under each method’s official configuration: the Native Transformer (FlashAttention implementation) used batch size 1, while all acceleration methods used batch size 8. The timing results for the methods tested on 40GB A100 and H20

GPUs are shown in Figure 7. The remaining results are shown in Appendix C (Figure 11).

- 3.3 Comparison with Existing Frameworks We compare LOOM-Scope with other evaluation frameworks in Table 1. We find that LOOM-Scope offers the most comprehensive support for long-context evaluation in terms of benchmark coverage, model architecture compatibility, server deployment, and an integrated comprehensive evaluation benchmark. Notably, LOOM-Scope is the only existing platform that incorporates long-context inference acceleration methods, greatly improving its extensibility and evaluation efficiency.
- 4 Conclusion

To ensure fairness and efficiency in long-context evaluation, we propose LOOM-Scope, a comprehensive and efficient evaluation framework for long-context models, context lengths ranging from 8K to 2M tokens. It supports all mainstream model architectures, provides two userfriendly interfaces (command-line and WebUI), and accommodates various augmentation strategies, including both inference acceleration and retrieval-augmented generation (RAG) methods. Experimental results using LOOM-Scope and LOOMBENCH demonstrate the efficiency and comprehensiveness of our platform.

## Limitation and Future Work

We list two main limitations and their corresponding future work for LOOM-Scope.

Supporting More Benchmarks Although our evaluation framework is designed to be modular and extensible, it currently does not include all existing public benchmarks due to the need for customized data formatting and integration. Incorporating a broader range of benchmarks to support more flexible and diverse evaluation is one of our key future directions.

Supporting More Modalities In addition, the current version of our framework is limited to text evaluation. In future work, we plan to extend support to more modalities relevant to longcontext scenarios, such as video (Tang et al., 2025), along with corresponding inference acceleration techniques. These enhancements aim to make our framework more comprehensive and widely applicable across multi-modal longcontext tasks.

## Ethics Statement and System License

The development and deployment of LOOMScope are guided by ethical principles, with a strong commitment to transparency, reproducibility, and accessibility. We aim to ensure that the framework is used responsibly and contributes positively to the broader research community. To this end, we publicly release the source code and datasets under the Apache License 2.03 , which permits open-source use, modification, and commercial deployment, including usage within industrial and enterprise settings.

3https://www.apache.org/licenses/LICENSE-2. 0

## Broader Impact Statement

Transparency and Accountability All datasets used in LOOM-Scope are clearly labeled with their sources and are licensed under permissive open-source agreements, e.g., Apache 2.0, allowing for modification and redistribution. Our benchmark suite, LOOMBENCH, builds on these datasets and is fully documented and reproducible. We also provide testing scripts and demo inputs to ensure that all reported results can be independently validated, fostering trust and transparency in long-context evaluation.

Support for Low-Resource Settings LOOMScope is the first framework to support multiple inference acceleration methods under a unified environment, making it particularly suitable for low-resource settings. Users can efficiently evaluate long-context performance and compare trade-offs across various acceleration techniques without the need for large-scale infrastructure.

User Accessibility To broaden accessibility, LOOM-Scope provides a locally deployable WebUI, enabling researchers and practitioners, regardless of their programming proficiency, to conduct long-context evaluations with minimal setup. This user-centered design lowers the barrier to entry for non-expert users and facilitates practical adoption.

Contribution to the Long-Context Community We believe that LOOM-Scope will significantly advance research in long-context modeling, an increasingly important direction in the development of large language models (LLMs). By offering a unified, extensible, and reproducible evaluation platform, we hope to support the community in building more capable, robust, and transparent LCLMs.

## References

Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, and 1 others. 2024. Phi3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508.

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, and 1 others. 2024a. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. arXiv preprint arXiv:2412.15204.

Yushi Bai, Jiajie Zhang, Xin Lv, Linzhi Zheng, Siqi Zhu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024b. Longwriter: Unleashing 10,000+ word generation from long context llms. arXiv preprint arXiv:2408.07055.

Amanda Bertsch, Maor Ivgi, Uri Alon, Jonathan Berant, Matthew R Gormley, and Graham Neubig. 2024. In-context learning with long-context models: An in-depth exploration. arXiv preprint arXiv:2405.00200.

OpenCompass Contributors. 2023. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/ opencompass.

Harry Dong, Xinyu Yang, Zhenyu Zhang, Zhangyang Wang, Yuejie Chi, and Beidi Chen. 2024. Get more with less: synthesizing recurrence with kv cache compression for efficient llm inference. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain

Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2024. The language model evaluation harness.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2023. Retrievalaugmented generation for large language models: A survey. arXiv e-prints, pages arXiv–2312.

Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. 2023. Model tells you what to discard: Adaptive kv cache compression for llms. In The Twelfth International Conference on Learning Representations, volume abs/2310.01801.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, and 1 others. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793.

Raghavv Goel, Junyoung Park, Mukul Gagrani, Dalton Jones, Matthew Morse, Harper Langston, Mingu Lee, and Chris Lott. 2025. Caote: Kv caching through attention output error based token eviction. arXiv preprint arXiv:2504.14051.

Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.

Gefei Gu, Yilun Zhao, Ruoxi Ning, Yanan Zheng, and Arman Cohan. 2024. TAIL: A toolkit for automatic and realistic long-context large language model evaluation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 198–208, Miami, Florida, USA. Association for Computational Linguistics.

Chaoqun He, Renjie Luo, Shengding Hu, Yuanqian Zhao, Jie Zhou, Hanghao Wu, Jiajie Zhang, Xu Han, Zhiyuan Liu, and Maosong Sun. 2024. Ultraeval: A lightweight platform for flexible and comprehensive evaluation for llms. Preprint, arXiv:2404.07584.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Yakun S Shao, Kurt Keutzer, and Amir Gholami. 2024a. Kvquant: Towards 10 million context length llm inference with kv cache quantization. Advances in Neural Information Processing Systems, 37:1270– 1303.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. 2024b. Kvquant: Towards 10 million context length llm inference with kv cache quantization. In Advances in Neural Information Processing Systems, volume 37, pages 1270–1303. Curran Associates, Inc.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. 2024. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654.

Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua

- Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024. MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. In The Thirtyeighth Annual Conference on Neural Information Processing Systems.
- Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, and Tuo Zhao. 2024. GEAR: An efficient error reduction framework for KV cache compression in LLM inference. In Proceedings of The 4th NeurIPS Efficient Natural Language and Speech Processing Workshop, volume 262 of Proceedings of Machine Learning Research, pages 305–321. PMLR.

Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, and Mikhail Burtsev. 2024. Babilong: Testing the limits of llms with long context reasoning-in-ahaystack. arXiv preprint arXiv:2406.10149.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language

model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, and Xun Zhou. 2024. Flexprefill: A contextaware sparse attention mechanism for efficient long-sequence inference. In The Thirteenth International Conference on Learning Representations.

Jinhyuk Lee, Anthony Chen, Zhuyun Dai, Dheeru Dua, Devendra Singh Sachan, Michael Boratko, Yi Luan, Sébastien M. R. Arnold, Vincent Perot, Siddharth Dalmia, Hexiang Hu, Xudong Lin, Panupong Pasupat, Aida Amini, Jeremy R. Cole, Sebastian Riedel, Iftekhar Naim, Ming-Wei Chang, and Kelvin Guu. 2024. Can long-context language models subsume retrieval, rag, sql, and more? Preprint, arXiv:2406.13121.

Quinn Leng, Jacob Portes, Sam Havens, Matei Zaharia, and Michael Carbin. 2024. Long context rag performance of large language models. arXiv preprint arXiv:2411.03538.

Haoyang Li, Yiming Li, Anxin Tian, Tianhao Tang, Zhanchao Xu, Xuejia Chen, Nicole Hu, Wei Dong, Qing Li, and Lei Chen. 2024a. A survey on large language model acceleration based on kv cache management. arXiv preprint arXiv:2412.19442.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024b. Snapkv: Llm knows what you are looking for before generation. In Advances in Neural Information Processing Systems, volume 37, pages 22947–22970. Curran Associates, Inc.

Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky. 2024c. Retrieval augmented generation or long-context llms? a comprehensive study and hybrid approach. arXiv preprint arXiv:2407.16833.

Zhan Ling, Kang Liu, Kai Yan, Yifan Yang, Weijian Lin, Ting-Han Fan, Lingfeng Shen, Zhengyin Du, and Jiecao Chen. 2025. Longreason: A synthetic long-context reasoning benchmark via context expansion. arXiv preprint arXiv:2501.15089.

Jerry Liu. 2022. LlamaIndex.

Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Wang, Chenchen Zhang, Ge Zhang, Jiebin Zhang, and 1 others. 2025. A comprehensive survey on long context language modeling. arXiv preprint arXiv:2503.17407.

Xiang Liu, Peijie Dong, Xuming Hu, and Xiaowen Chu. 2024a. Longgenbench: Long-context generation benchmark. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 865–883.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen (Henry) Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. 2024b. Kivi: a tuning-free asymmetric 2bit quantization for kv cache. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Chao Lou, Zixia Jia, Zilong Zheng, and Kewei Tu. 2024. Sparser is faster and less is more: Efficient sparse attention for long-range transformers. arXiv preprint arXiv:2406.16747.

- AI Meta. 2024. Introducing llama 3.1: Our most capable models to date, 2024. URL https://ai. meta. com/blog/meta-llama-3-1/. New models including flagship 405B parameter model, along with upgraded 8B and 70B models featuring 128K context length and multilingual capabilities.
- AI Meta. 2025. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. https://ai. meta. com/blog/llama-4-multimodalintelligence/, checked on, 4(7):2025.

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, and 1 others. 2023. Rwkv: Reinventing rnns for the transformer era. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 14048–14077.

Ziran Qin, Yuchen Cao, Mingbao Lin, Wen Hu, Shixuan Fan, Ke Cheng, Weiyao Lin, and Jianguo Li. 2025. CAKE: Cascading and adaptive KV cache eviction with layer preferences. In The Thirteenth International Conference on Learning Representations.

Partha Pratim Ray. 2025. A survey on model context protocol: Architecture, state-of-the-art, challenges and future directions. Authorea Preprints.

Jonathan Roberts, Kai Han, and Samuel Albanie. 2025. Needle threading: Can llms follow threads through near-million-scale haystacks? In The Thirteenth International Conference on Learning Representations.

Stephen Robertson, Hugo Zaragoza, and 1 others. 2009. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389.

Mingyang Song, Mao Zheng, and Xuan Luo. 2024. Counting-stars: A multi-evidence, position-aware, and scalable benchmark for evaluating longcontext large language models. Preprint.

Yi Su, Yuechi Zhou, Quantong Qiu, Juntao Li, Qingrong Xia, Ping Li, Xinyu Duan, Zhefeng Wang, and Min Zhang. 2025. Accurate kv cache quantization with outlier tokens tracing. In The 63rd Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics.

Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, and 1 others. 2025. Video understanding with large language models: A survey. IEEE Transactions on Circuits and Systems for Video Technology.

Minzheng Wang, Longze Chen, Cheng Fu, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan Xu, Lei Zhang, Run Luo, and 1 others. 2024a. Leave no document behind: Benchmarking longcontext llms with extended multi-doc qa. arXiv preprint arXiv:2406.17419.

Weiyun Wang, Shuibo Zhang, Yiming Ren, Yuchen Duan, Tiantong Li, Shuo Liu, Mengkang Hu, Zhe Chen, Kaipeng Zhang, Lewei Lu, and 1 others. 2024b. Needle in a multimodal haystack. Advances in Neural Information Processing Systems, 37:20540–20565.

Zheng Wang, Boxiao Jin, Zhongzhi Yu, and Minjia Zhang. 2024c. Model tells you where to merge: Adaptive kv cache merging for llms on long-context tasks. ArXiv, abs/2407.08454.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, and Song Han. 2025. Xattention: Block sparse attention with antidiagonal scoring. arXiv preprint arXiv:2503.16428.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Songlin Yang and Yu Zhang. 2024. Fla: A tritonbased library for hardware-efficient implementations of linear attention mechanism.

Dingyu Yao, Bowen Shen, Zheng Lin, Wei Liu, Jian Luan, Bin Wang, and Weiping Wang. 2025. Tailorkv: A hybrid framework for long-context inference via tailored kv cache optimization. Preprint, arXiv:2505.19586.

Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, and Danqi Chen. 2024. Helmet: How to evaluate long-context language models effectively and thoroughly. arXiv preprint arXiv:2410.02694.

Yifei Yu, Qian-Wen Zhang, Lingfeng Qiao, Di Yin, Fang Li, Jie Wang, Zengxi Chen, Suncong Zheng, Xiaolong Liang, and Xing Sun. 2025. Sequentialniah: A needle-in-a-haystack benchmark for extracting sequential needles from long contexts. arXiv preprint arXiv:2504.04713.

Yijiong Yu, Ma Xiufa, Fang Jianwei, Zhi Xu, Su Guangyao, Wang Jiancheng, Yongfeng Huang, Zhixiao Qi, Wei Wang, Weifeng Liu, and 1

others. 2024. Hyper-multi-step: The truth behind difficult long-context tasks. arXiv preprint arXiv:2410.04422.

Tao Yuan, Xuefei Ning, Dong Zhou, Zhijie Yang, Shiyao Li, Minghui Zhuang, Zheyue Tan, Zhuyu Yao, Dahua Lin, Boxun Li, and 1 others. 2024. Lv-eval: A balanced long-context benchmark with 5 length levels up to 256k. arXiv preprint arXiv:2402.05136.

Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, and Jianfei Chen. 2025a. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. In International Conference on Learning Representations (ICLR).

Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. 2025b. Spargeattn: Accurate sparse attention accelerating any model inference. In International Conference on Machine Learning (ICML).

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Hao, Xu Han, Zhen Thai, Shuo Wang, Zhiyuan Liu, and 1 others. 2024. Infinite-bench: Extending long context evaluation beyond 100k tokens. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15262–15277.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. 2023. H2o: heavy-hitter oracle for efficient generative inference of large language models. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, and 1 others. 2024. Sglang: Efficient execution of structured language model programs. Advances in Neural Information Processing Systems, 37:62557–62583.

Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming Xu, Shiyao Li, Yuming Lou, Luning Wang,

#### Zhihang Yuan, Xiuhong Li, and 1 others. 2024. A survey on efficient inference for large language models. arXiv preprint arXiv:2404.14294.

## A Related work

- A.1 Long-context Augmentation Methods

To alleviate computational inefficiencies of transformer-based models during long-context inference, recent advances have introduced various augmentation methods, such as RAG (Lee et al., 2024; Gao et al., 2023; Liu, 2022), keyvalue cache optimization (Li et al., 2024a), and inference-time sparse attention (Jiang et al., 2024; Lai et al., 2024; Xu et al., 2025; Zhang et al., 2025b). Among these methods, key-value cache optimization has emerged as the most commonly adopted approach, which can be further categorized into selective eviction (Zhang

- et al., 2023; Ge et al., 2023; Li et al., 2024b; Qin et al., 2025), quantization (Liu et al., 2024b; Hooper et al., 2024b; Su et al., 2025; Yao et al., 2025), and approximation (Kang et al., 2024; Wang et al., 2024c; Dong et al., 2024) methods. Notably, LOOM-Scope already supports most mainstream long-context augmentation methods currently in use.

A.2 Long-context Evaluation

Existing long-context benchmarks can generally be classified into 3 categories according to the specific capabilities they assess: long-context understanding (Bai et al., 2023; Zhang et al., 2024; Roberts et al., 2025), long-form reasoning (Kuratov et al., 2024; Bai et al., 2024a; Song et al., 2024), and long-form generation (Bai et al., 2024b). Furthermore, the task formats themselves can be further divided into realworld tasks (e.g., multi-document QA (Wang

- et al., 2024a; Yuan et al., 2024), article generation (Liu et al., 2024a)) and synthetic tasks (e.g., needle-in-a-haystack retrieval (Yu et al., 2025; Wang et al., 2024b), synthetic multi-hop reasoning (Yu et al., 2024)). Currently, there are more than 150 benchmarks dedicated to long-

context evaluation (Liu et al., 2025). However, conflicting evaluation results exist across benchmarks (Yen et al., 2024), and disparate evaluation settings, such as model prompts and execution environments, make those benchmarks cumbersome to use and difficult to compare. In this paper, we introduce LOOM-Scope, a comprehensive and efficient framework for long-context evaluation, offering a unified environment that enables plug-and-play benchmarking and supports highly user-customizable modules for flexible and convenient evaluation.

## B Benchmark Overview

We provides a systematic overview of the supported benchmarks, categorizing them by capability and summarizing their key metrics and data volumes in Table 3.

### B.1 Supporting Benchmarks

LOOM-Scope supports 22 widely used longcontext benchmarks in Table 4, covering more than 140 subtasks, the 8k-2M context range, and spanning six main LCLM capabilities:

- • Faithfulness: Focus on citation accuracy and information attribution
- • General: Broad evaluation across multiple domains and task types
- • Retrieve: Needle-in-haystack and information location tasks
- • Generation: Long-form content creation capabilities
- • Reasoning: Complex multi-hop reasoning and logical inference
- • Specialization: Domain-specific (medical, legal) and language-specific benchmarks

Type Implementation Description

HF_Models HuggingFace optimized inference VLLM PagedAttention for long sequences SGLang RadixAttention for fast execution

Local Models

Server

Cloud Models API API-based scalable inference service

RWKV Attention-free recurrent architecture Mamba Selective state space model

RNN-based

Architecture

Linear-Attention GLA Gated linear attention

QWEN Pretrained LLM proposed by Alibaba Cloud Deepseek Pretrained LLM proposed by Deepseek Mistral Frontier AI LLMs, Assistants, Agents, Services GLM Open Multilingual Multimodal Chat LMs

Transformer-based

H2O Attention-based selection StreamingLLM Retain first few tokens SnapKV Attention Pooling before selection L2Compress L2 Norm is better than attention as a metric

Token Eviction

PyramidKV Layer-wise budget allocation CakeKV Layer-specific preference score

Layer-wise

Augmentation

Quantization KIVI Asymmetric 2-bit Quantization

Hybrid ThinK Thinner Key Cache by Query-Driven Pruning Token Merge CaM Cache Merging for LLMs Inference

FlexPrefill A context-aware sparse attention mechanism XAttention Block sparse attention with antidiagonal scoring

Sparse Attention

BM25 Classic sparse retrieval algorithm LlamaIndex Framework for efficient RAG indexing OpenAI Powers generation/embeddings in RAG

RAG

Table 2: Deployment Module Details

Category Benchmark Metric Data Volumn Data Length Faithfulness

L-CiteEval ROUGE,ACC,Recall 3220 0 to 52K LongCite F1,Recall,Precision,ACC,LLM 1000 0 to 82K

LEval ACC,LLM,ROUGE,F1 8645 3K to 200K LooGLE ROUGE,BLEU,LLM 1617 12K to 282K RULER ACC,F1,Recall - 4K to 128K

General

LongBench ACC,F1 8418 0 to 64K BAMBOO ACC,F1,Recall,Pass@1 2904 0 to 16K

Counting-Stars F1,Pass@N 128 4K to 128K LongIns F1 20767 0 to 16K LVEval ROUGE,F1 8645 8K to 418K

Reasoning

LongBench V2 ACC 503 8K to 4M

babilong ACC 4500 0 to 128K Ada_LEval ACC 13200 0 to 141K

NIAH ACC - -

NThread ACC,LLM 10860 1K to 616K InfiniteBench ROUGE,F1 3946 26k to 5M

Retrieval

NoLiMa ACC - 1K to 32K Generation LongWriter LLM 228 0 to 1K

LIBRA F1,EM 13071 1K to 142K CLongEval F1,ROUGE,ACC 7263 1K to 128K LongHealth ACC 6000 8K to 16K LongSafety F1,LLM 6172 3K to 22K

Specialization

Table 3: Benchmarks Overview.The hyphen ("-") represents infinite length.

##### Total Benchmarks: 22 Total Tasks: 149 Languages: EN, ZH, RU Context Length: 8K-2M tokens

###### Category Benchmark Tasks Key Capabilities Domains

L_CiteEval

11 Citation evaluation, multi-hop QA, dialogue simulation News, Government, Dialogue

Faith.

LongCite 1 Long document citation accuracy Academic

BAMBOO 10 Paper QA, hallucination detection, meeting/show prediction Academic, Business LongBench

16 Multilingual QA, summarization, passage retrieval, code Multi-domain LEval

General

20 Financial QA, legal contracts, scientific papers, TV shows Finance, Legal, Science RULER

5 Variable tracking, multi-query NIAH Synthetic

LooGLE 2 Long dependency QA, summarization General

NIAH

1 Classic needle-in-haystack retrieval Synthetic NThread 6 Multi-needle variants with CoT and distractors Synthetic

Retrieve

InfiniteBench

12 Code debug, math calc, long dialogue, book QA (EN/ZH) Code, Math, Literature

NoLiMa 5 Conditional needles, multi-thread tracking Synthetic

LongWriter

Generation

3 Long-form writing in English and multilingual Creative Writing

Counting-Stars

- 4 Reasoning and searching in EN/ZH Logic

babilong

- 5 Multi-hop reasoning tasks (qa1-qa5) Synthetic

LongIns 2 GIST extraction, LIST processing Instruction Following LVEval

Reasoning

11 Mixed-up tasks, fact recall, reading comprehension Multi-domain LongBench_v2

1 Deep understanding with 8K-2M context Complex Reasoning

Ada-LEval 2 Stack selection, text sorting Algorithmic

LIBRA

21 Russian language tasks across all categories Russian Multi-domain

CLongEval 7 Memory, summarization, table query, key retrieval Chinese-focused

Special.

LongHealth 3 Medical document understanding Healthcare

LongSafety 1 Safety evaluation in long contexts AI Safety

- Table 4: Comprehensive Overview of Long Context Evaluation Benchmarks.LOOMBENCH is indicated by the orange-highlighted entries

[Figure 147]

LOOMBENCH using the LOOM-Scope framework, with results presented in Figure 5, Figure 9, and Table 5.

Faithfulness

41

34

General

Specialization

60

59

51

52

78

32 39

91

Reasoning

Generation

56

67

Retrieval

Phi-4-mini-instruct

GLM-4-9B

Qwen3-4B

Llama-3.1-Nemotron-Nano-8B-v1

Qwen2.5-7B-Instruct

Llama-3.1-Nemotron-Nano-4B-v1.1

Mistral-Nemo-Instruct-2407

Figure 8: Task distribution of LOOMBENCH.

### B.2 Statistic of LOOMBENCH

LOOMBENCH is a composite benchmark constructed from 12 existing datasets, as indicated by the orange-highlighted entries in Table 4. These selected datasets ensure a balanced evaluation across six core long-context capabilities, providing comprehensive coverage of the benchmark’s design objectives. The visualization of task distribution is also shown in Figure 8.

C Full Experimental Results

Table 5 presents the full evaluation of all models in LOOMBENCH, including detailed performance metrics for LCLMS. Table 7 provides comparative results of the augmentation method. These results collectively demonstrate the benchmark’s comprehensive assessment of long-context capabilities.

### C.1 Vanilla Model Performance

Model Performance We evaluated the six core capabilities of 14 mainstream LCLMs in

Figure 9: Capability radar chart of remain LCLMs.

Evaluation Latency We used the LOOMScope to evaluate both the native benchmark and LOOMBENCH. The Evaluation Latency presented in Table 6 clearly demonstrates that LOOMBENCH, which serves as a comprehensive but lightweight benchmarking tool, enables efficient performance assessment.

### C.2 Augmentation Methods

To systematically investigate the efficacy of long-context augmentation techniques, we evaluated retrieval-augmented generation and inference acceleration methods across mainstream methods using LOOM-Scope. Figure 10 presents comprehensive comparisons of RAG performance. Figure 11 further quantifies the latency reductions for Llama-3.1-8B-Instruct under various acceleration strategies, demonstrating up to 12× speedup with acceleration methods. Table 7 and Table 8 consolidate a detailed result of the Performance and Prediction Time.

Faithfulness General Reasoning Retrieve Generation Specialization LCite LEval RULER LongB BABI Count LVE LB2 NIAH InfB LongW LIBRA

Rank Model Avg

Qwen Series

- 1 Qwen3-14B 51.54 35.64 43.84 74.94 45.47 59.15 56.41 21.26 29.85 100 10.24 85.75 55.87

- 2 Qwen3-30B-A3B 51.18 37.96 40.61 78.32 43.24 60.31 48.96 22.82 28.42 100 14.14 83.24 56.09

- 6 Qwen3-8B 44.71 33.18 41.15 67.68 38.62 55.28 52.32 15.15 27.25 64.00 8.06 81.99 51.78

- 9 Qwen3-4B 43.10 24.55 39.03 70.29 39.32 55.01 42.06 18.24 32.52 62.00 13.05 74.25 46.92

- 10 Qwen2.5-7B-Instruct 42.01 29.12 44.63 72.02 40.85 55.89 38.25 14.94 27.33 64.18 13.97 52.75 50.23

Meta/Llama Series 3 Llama-3.1-8B-Instruct 46.94 25.79 39.70 86.79 37.94 57.42 37.68 25.66 30.40 91.00 33.64 45.96 51.24

- 13 Nemotron-Nano-8B-v1 24.47 14.11 34.32 42.51 27.19 28.78 11.72 6.57 12.67 43.73 0.47 38.99 32.54

- 14 Nemotron-Nano-4B-v1.1 21.05 10.11 25.88 38.85 19.94 22.67 7.48 6.69 22.67 28.38 7.43 45.68 16.81

Microsoft Phi Series

7 Phi-3-mini-128k-instruct 44.67 32.96 39.87 78.62 38.31 53.56 31.04 39.87 24.02 90.00 35.14 33.73 38.86 8 Phi-4-mini-instruct 43.83 24.20 40.18 76.70 42.69 53.56 13.31 30.93 31.33 92.61 27.87 41.27 51.28

GLM Series

5 GLM-4-9B-chat 44.89 30.66 46.42 85.25 45.24 55.00 36.84 23.33 32.00 65.27 20.35 43.90 54.42 12 GLM-4-9B 36.80 21.59 45.70 55.96 38.41 46.33 21.51 17.18 24.00 47.15 3.11 74.89 45.76

Other Models 4 c4ai-command-r7b-12-2024 45.39 24.73 42.68 77.41 37.16 47.44 35.00 35.66 33.33 92.43 20.09 51.69 47.00

- 11 Mistral-Nemo-Instruct-2407 38.37 24.91 42.47 60.60 39.75 53.67 21.12 21.61 21.34 60.41 16.98 48.30 49.25

- Table 5: High-performance Long-context Language Models on LOOMBench: Comprehensive evaluation across 12 benchmarks measuring reasoning, retrieval, generation, and comprehension capabilities. LCite: L_CiteEval; LongB: LongBench; BABI: BABILong; Count: Counting-Stars; LVE: LVEval; LB2: LongBench_v2; InfB: InfiniteBench; LongW: LongWriter

Benchmark Sum

Faithfulness General Reasoning Retrieve Generation Specialization LCite LEval RULER LongB BABI Count LVE LB2 NIAH InfB LongW LIBRA

NVIDIA GeForce RTX 3090 LOOMBENCH 22:42:34 2:20:39 2:17:58 1:59:01 1:40:47 2:19:20 1:07:38 2:35:32 1:19:12 0:50:40 1:49:10 1:47:06 2:35:31

Native 216:44:36 10:49:25 6:09:10 35:24:45 8:28:28 31:8:45 0:39:04 37:12:43 3:25:30 4:31:48 45:33:55 9:04:27 1:34:02 NVIDIA 40GB A100

LOOMBENCH 8:38:06 0:25:53 1:10:46 0:59:17 0:14:57 0:41:49 0:37:15 0:50:53 0:48:50 0:24:28 0:39:12 1:06:59 0:22:07

Native 110:56:06 7:35:33 3:58:46 16:40:31 4:00:00 12:09:37 0:19:38 21:01:12 1:37:19 1:59:01 23:40:31 2:28:56 0:43:07 NVIDIA H20

LOOMBENCH 5:40:40 0:29:11 0:32:59 0:31:57 0:14:19 0:30:24 0:20:49 0:31:49 0:26:01 0:34:08 0:24:24 0:33:15 0:31:24 Native 96:14:11 6:45:38 3:41:35 20:10:37 6:25:36 16:37:10 0:17:58 23:42:26 2:05:23 2:50:34 26:01:28 1:29:10 0:48:51

- Table 6: The Evaluation Latency for the Native Benchmark and LOOMBENCH.In LOOMBENCH, the naive Transformer service running on a single 40GB A100 GPU encountered Out-of-Memory (OOM) errors on certain datasets. To address this, we employed a dual-GPU inference strategy for those datasets, which resulted in slightly longer processing times. LCite: L_CiteEval; LongB: LongBench; BABI: BABILong; Count: Counting-Stars; LVE: LVEval; LB2: LongBench_v2; InfB: InfiniteBench; LongW: LongWriter

RULER KIVI H2O StreamingLLM L2Norm CaM CakeKV PyramidKV FlexPrefill SnapKV ThinK Transformers VLLM SGLang

4k 94.83 46.13 27.79 36.42 67.39 91.88 77.42 93.33 81.54 81.83 94.83 89.54 91.09 8k 94.29 34.42 22.21 27.75 59.60 82.42 71.25 87.92 73.29 72.88 94.42 90.74 89.83

NVIDIARTX3090

16k 91.08 21.63 11.46 27.58 48.53 78.96 68.75 83.38 67.88 67.25 92.50 84.09 86.40 32k 89.50 19.67 10.79 22.17 43.42 79.54 64.33 83.38 66.29 66.21 92.75 79.83 74.58 64k - 10.54 7.67 18.25 33.25 62.88 57.29 65.04 55.58 56.92 83.71 82.32 80.98

128k - 6.38 9.79 12.17 28.10 53.17 48.29 46.58 47.25 48.71 - 36.55 67.07 Prediction Time 1:29:58 5:11:21 5:07:27 5:08:31 50:31:13 5:13:01 5:14:12 3:58:17 5:08:31 5:18:08 3:11:15 1:49:41 1:41:35

4k 95.46 46.38 28.83 33.96 67.29 89.67 77.00 93.58 79.75 79.33 95.04 90.71 90.76 8k 94.42 31.96 21.58 26.83 59.67 82.46 70.79 89.04 72.67 71.92 94.46 91.63 92.43

NVIDIA40GBA100

16k 90.04 23.00 10.63 27.92 48.53 77.29 67.54 84.83 66.63 65.25 92.50 91.75 85.67 32k 89.00 19.86 10.79 20.83 43.21 76.42 65.25 79.63 63.88 66.46 92.08 86.25 77.46 64k 90.25 10.50 7.67 17.25 33.35 65.79 55.54 62.50 54.75 55.75 83.71 78.83 82.11

128k 67.92 6.17 10.63 12.17 28.03 55.21 49.54 49.42 50.79 51.00 75.21 17.50 69.01

- Prediction Time 0:32:57 0:27:00 0:26:35 0:27:26 - 0:28:26 0:26:39 0:15:58 0:26:44 0:35:46 3:23:10 0:27:20 0:20:27

NVIDIAH20

4k 93.79 54.33 28.00 34.21 66.29 89.67 77.13 92.63 82.22 80.83 94.00 91.75 94.29 8k 94.33 41.25 21.58 26.75 59.67 82.63 71.50 87.88 70.46 72.33 94.46 88.79 88.83

16k 90.71 24.17 10.63 27.00 48.33 77.21 65.33 84.54 64.57 65.92 92.50 84.58 84.88 32k 88.08 19.42 10.79 22.58 43.42 76.42 66.21 80.71 69.27 66.46 91.88 79.50 78.47 64k 80.91 12.33 6.83 16.58 34.25 65.71 58.25 62.96 57.73 56.25 83.71 77.79 81.63

128k 69.13 10.92 10.63 12.25 28.13 54.21 47.58 48.08 45.94 47.25 75.21 57.88 65.92

- Prediction Time 0:33:44 0:33:08 0:32:35 00:33:48 - 0:32:37 0:32:41 0:13:10 0:32:41 0:34:25 3:57:44 0:38:56 0:23:18

#### Table 7: Performance comparison of various acceleration methods for the Llama-3.1-8B-Instruct model across different hardware configurations and input scales, showcasing reasoning times and accuracy metrics. The batch size per GPU is 8."-" represent Out of Memory.

80

60

###### Score

40

20

0

Faithfulness General Reasoning Retrieval

Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct + BM25 Llama-3.1-8B-Instruct + Self-Route GLM-4-9B GLM-4-9B + BM25 GLM-4-9B + Self-Route

Phi-4-mini-instruct Phi-4-mini-instruct + BM25 Phi-4-mini-instruct + Self-Route Qwen3-14B Qwen3-14B + BM25 Qwen3-14B + Self-Route

Qwen3-8B Qwen3-8B + BM25 Qwen3-8B + Self-Route

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 10: Complete RAG test results for mainstream models.

36min NVIDIA H20 NVIDIA 40G A100

ThinK

- 33min

27min

- 34min

| |
|---|

SnapKV

16min

FlexPrefill

13min

27min

PyramidKV

33min

- 27min

33min

- 28min

CakeKV

L2Norm

- 33min

27min

- 34min

StreamingLLM

27min

H2O

33min

33min

KIVI

34min

27min

VLLM

39min

203min

Transformers

238min

0 50 100 150 200 250

Testing Time (minutes)

Figure 11: Complete augmentation methods test results for Llama-3.1-8B-Instruct models.

XAttention

4k 93.45 8k 91.00

NVIDIA40GBA100

16k 89.94 32k 84.07 64k 78.66

128k 34.35 Prediction Time 0:16:34

- Table 8: Performance of XAttention acceleration methods for the Llama-3.1-8B-Instruct model across different input scales in RULER, showcasing reasoning times and accuracy metrics.XAttention only optimized the A100.

