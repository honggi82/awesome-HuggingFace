## LibMoE: A Library for Comprehensive Research Mixture of Experts in Large Language Models

# arXiv:2411.00918v4[cs.CL]10Feb2026

Nam V. Nguyen†,∗ Thong T. Doan†,∗ Luong Tran† Van Nguyen† Quang Pham‡,⋄

† FPT Software AI Center ‡ Independent Researcher ⋄ Correspondence: quangg2012@gmail.com

#### Abstract

Mixture of experts (MoE) architectures have become a cornerstone for scaling up and are a key component in most large language models such as GPT-OSS, DeepSeek-V3, Llama-4, and Gemini-2.5. However, systematic research on MoE remains severely constrained by the prohibitive computational costs of training and evaluation, restricting large-scale studies accessible to most researchers. We introduce LibMoE, a unified framework for reproducible, efficient, and extensible MoE research that supports both pretraining and sparse-upcycling regimes. Beyond unified implementations, the framework provides transparent analytical tools for probing routing and expert dynamics. Leveraging this foundation, we conduct a comprehensive analysis along three dimensions: (i) routing dynamics, covering expert selection patterns, routing stability and optimality, and how routing entropy reveals task specialization and expert diversity; (ii) the effect of lightweight initialization on load balancing, demonstrating how subtle changes in router initialization shape early expert utilization; and (iii) training regime differences, revealing how sparse upcycling and full pretraining exhibit distinct routing patterns and stability profiles. By lowering the barrier to entry and standardizing evaluation, along with our comprehensive analysis, LibMoE broadens access to MoE research and establishes a reliable benchmark to guide future innovations. GitHub: https://github.com/Fsoft-AIC/LibMoE.

#### 1 Introduction

Recent years have witnessed a dramatic surge in the adoption and success of deep learning models, fueled by their scalability with massive data and compute resources and their state-of-the-art performance across diverse domains. Yet, this rapid scaling has introduced significant challenges in computational efficiency. A prominent strategy to address these challenges is the sparse mixture of experts (SMoE) architecture, which expands model capacity without a proportional increase in computation.

First introduced by Jacob et al. (Jacobs et al., 1991), SMoE has since achieved remarkable success across multiple fields, including large language models (LLMs) (Shazeer et al., 2017b; Fedus et al., 2022; Comanici et al., 2025; Liu et al., 2024b; Jiang et al., 2024), multimodal learning (Yun et al., 2024; Lin et al., 2024a; Han et al., 2024a; Meta AI, 2025), and computer vision (Riquelme et al., 2021; Fan et al., 2022; Han et al., 2024b). By activating only a sparse subset of parameters per input, SMoE substantially improves training efficiency, enabling models with hundreds of billions of parameters to be trained at manageable computational cost

*Equal contribution.

(Fedus et al., 2022). Interestingly, SMoE algorithms often achieved superior performances compared to their dense counterpart (Shazeer et al., 2017b), which has gathered an increased interest in the community to develop more advanced algorithms and better theories to understand their behaviour (Muennighoff et al., 2024; Xue et al., 2024; Kang et al., 2025; Zoph et al., 2022; Nguyen et al., 2025b; 2024c). Consequently, SMoE has become a cornerstone methodology driving the development of next-generation intelligent systems.

However, despite rapid algorithmic advances, progress in MoE research remains fundamentally hindered by the lack of standardized frameworks. Large-scale MoE experiments are typically feasible only for groups with access to massive computational resources training OLMoE-1B-7B, for example, required 256 H100 GPUs (Muennighoff et al., 2024), and MOMA relied on 256 A100 GPUs (Lin et al., 2024b). In contrast, the majority of researchers are confined to small-scale studies (Nielsen et al., 2025; Han et al., 2024a; Teo & Nguyen, 2024; Csordás et al., 2023; 2024; Nguyen et al., 2025a; Teo & Nguyen, 2025) or even synthetic benchmarks (Nguyen et al., 2024c;b; Yan et al., 2024), resulting in fragmented and often incomparable results. This persistent discrepancy severely undermines the potential of MoE architectures, whose true strengths emerge only in large-scale training regimes. As a result, even the most prominent deployed models such as Phi3 (Abdin et al., 2024), Skywork-MoE (Wei et al., 2024), and OSS-GPT (OpenAI, 2025) tend to rely on conventional SMoE designs (Fedus et al., 2022), with recent advances in routing and expert selection algorithms seeing little adoption in practice (Do et al., 2023; Zhou et al., 2022; Dai et al., 2022b; Wang et al., 2024d). These challenges underscore a critical need for a unified, reproducible framework that lowers the barrier to meaningful SMoE research under realistic resource constraints and enables rigorous, systematic evaluation of both established and emerging methods.

To address these challenges, we present LibMoE a unified framework that brings together state-of-the-art SMoE algorithms, standardized training pipelines, and transparent analytical tools for rigorous investigation of routing dynamics. Our primary objectives are twofold: (1) to provide an accessible and streamlined toolkit that enables meaningful SMoE research on large language models (LLMs) under realistic resource constraints; and (2) to enable comprehensive, reproducible analysis of SMoE routing mechanisms, yielding deeper insights into expert behaviors. With LibMoE, we aim to accelerate progress and foster a more collaborative, open-source SMoE research community.

Designing the LibMoE Framework. To address persistent challenges in accessible and reproducible MoE research, we introduce LibMoE a unified framework for systematic evaluation of SMoE methods across both full pretraining and sparse-upcycling regimes (Komatsuzaki et al., 2022). LibMoE accommodates a broad spectrum of model sizes and algorithmic variants, supporting rigorous comparisons from small-scale language models (0.15B and 0.68B parameters) to large-scale vision–language models (5.67B parameters). The framework integrates seven recent state-of-the-art SMoE algorithms and provides standardized pipelines for both pretraining and evaluation, enabling consistent benchmarking and cross-domain analysis. All MoE algorithm experiments are conducted under realistic resource constraints, with training times ranging from

6 to 44 hours on 4× H100 GPUs (see Table 11 for details). While our results reveal that current SMoE algorithms yield only marginal performance improvements under these settings, LibMoE substantially lowers the barrier to entry, making advanced and reproducible experimentation feasible even for research groups with limited computational resources.

Comprehensive Analysis of Mixture-of-Experts Dynamics. Beyond providing unified implementations, LibMoE is designed as a transparent and extensible infrastructure for systematically analyzing the behavior of SMoE models. The framework exposes modular instrumentation for monitoring routing decisions, expert utilization, load balancing, and inter-expert interactions, enabling fine-grained inspection of routing behavior throughout both training and inference. Using this analytical capability, we conduct a large-scale, cross-domain empirical study of seven recent SMoE algorithms across multiple training regimes, including small-scale pretraining (0.15B and 679M parameters) and sparse upcycling at the 5.67B scale (Komatsuzaki

- et al., 2022). Our analysis uncovers consistent and previously under-characterized differences along three dimensions: (i) routing dynamics, reflected in expert selection stability, entropy evolution, and specialization patterns; (ii) the impact of lightweight router initialization on early-stage load balancing; and (iii) systematic divergences in routing behavior between full pretraining and sparse upcycling regimes. Together, these results demonstrate how design choices in routing and initialization influence training stability and expert utilization, and establish a reproducible analytical basis for principled comparison of SMoE algorithms.

#### 2 Related Works

##### 2.1 Mixture of Experts

Originally introduced by Jacobs et al. (1991), the MoE framework was first formulated as an ensemble method that combines multiple specialized models through an adaptive gating network. Early variants, including Jacobs’ gating network and the Hierarchical MoE (HME) (Jordan & Jacobs, 1994), improved learning algorithms for multiclass classification (Chen et al., 1999), and achieved success in domains such as speech recognition (Gales & Airey, 2006). Eigen et al. (2013) extends the MoE to a layer in neural network, which consists of a set of experts (neural networks) and a trainable gate. These early efforts collectively shaped the theoretical and empirical foundation of MoE, setting the stage for the large-scale sparse MoE architectures that gained prominence after 2017. In 2017, Shazeer et al. (2017b) introduced the SMoE model, which incorporated MoE layers into long short-term memory (LSTM) networks to dramatically scale model capacity without increasing computational cost. Building on this idea, GShard (Lepikhin et al., 2020) extended MoE to Transformers by replacing the feed-forward network (FFN) layers in the T5 (Raffel et al., 2020) architecture with MoE layers. Switch Transformers (Fedus et al., 2022) further simplified the routing mechanism and scaled training to 1.6 trillion parameters, setting a milestone for large language models (LLMs). Since then, substantial progress has been made in advancing sparsely-gated MoE architectures for LLM development, including novel routing techniques (Chi et al., 2022; Roller et al., 2021; Zuo et al., 2021; Zhong et al., 2024; Muqeeth et al., 2023; Wu et al., 2024b; Wang et al., 2024d; Nielsen et al., 2025; Nguyen et al., 2025c; Csordás et al., 2023), stability-oriented strategies (Zoph et al., 2022; Dai et al., 2022b), fine-grained expert segmentation (Park et al., 2024; Dai et al., 2024; He, 2024), shared experts (Rajbhandari et al., 2022; Liu et al., 2024a;c; Dai et al., 2024; Team, 2024), special-purpose experts (Jin et al., 2024; Yan et al., 2025), gradient update modification (Panda et al., 2025; Yang et al., 2024b), and sparse upcycling (Komatsuzaki et al., 2022; Nakamura et al., 2025; Chen et al., 2025; Hui et al., 2024); with extensions also demonstrated in multimodal learning (Shen et al., 2023; Han et al., 2024a; Li et al., 2024b; Lin et al., 2024a; Wu et al., 2024c; Yun et al., 2024). Despite the encouraging progress, a clear discrepancy remains in current MoE research. In particular, despite a proliferation of advanced algorithms and theoretical analyses, the most capable and widely deployed LLMs (Agarwal et al., 2025; Comanici et al., 2025; Liu et al., 2024a;c; Dai et al., 2024; xAI, 2024; Yang et al., 2024a; Jiang et al., 2024; Meta AI, 2025; He et al., 2024; Abdin et al., 2024; Databricks, 2024; Wu et al., 2024a) still rely on SMoE variants closely aligned with the original formulation (Shazeer et al., 2017b; Fedus et al., 2022). This indicates that our current theoretical understanding remains largely confined to academic settings and has yet to meaningfully influence real-world deployments precisely the domain where SMoE could be most impactful. We attribute this gap primarily to the high entry barrier posed by massive datasets and computational requirements, which place large-scale SMoE experimentation out of reach for most research groups. Therefore, we develop LibMoE to enable training and evaluation settings that closely reflect real-world practice, including both early-stage pretraining and late-stage sparse upcycling under limited resources. This design allows for fair, comprehensive, and large-scale benchmarking of SMoE algorithms, while making state-of-the-art methods more accessible to the broader research community.

##### 2.2 Analyzing and Understanding Mixture-of-Experts Models

Since the introduction of the sparsely-gated MoE layer (Shazeer et al., 2017a), research has explored both the promise of conditional capacity and the challenge of expert imbalance. This has driven the development of stability-oriented methods (e.g., StableMoE (Dai et al., 2022a)), novel routing designs (e.g., ExpertChoice (Zhou et al., 2022)), and system-level optimizations (e.g., MegaBlocks (Gale et al., 2022)). Differentiable routing mechanisms, including DSelect-k (Hazimeh et al., 2021), BASE layers (Lewis et al., 2021), and ReMoE (Wang et al., 2024c), have further increased flexibility, while AdaMoE (Zeng et al., 2024) and loss-free balancing (Wang et al., 2024a) have enhanced efficiency and adaptability. Beyond algorithmic innovation, a growing body of theoretical and empirical work has examined MoE dynamics. For example, Zhao et al. (2024) derived sparsity-aware generalization bounds, while Fan et al. (2024) systematically ablated routing granularity and expert count, revealing that token-level routers tend to capture syntactic rather than semantic patterns. Other studies have characterized router behavior (e.g., Nguyen et al. (2024a)) and highlighted linguistic specialization (e.g., Antoine et al. (2024)), collectively connecting routing design to generalization,

specialization, and stability. Efforts to democratize MoE research have yielded open-source frameworks such as OpenMoE (Xue et al., 2024), OLMoE (Muennighoff et al., 2024), and FLAME-MoE (Kang et al., 2025), which provide models, training frameworks, and diagnostic tools. However, the adoption of these resources is often limited by substantial computational requirements, restricting reproducibility and practical experimentation. Moreover, most prior studies focus on behaviors specific to their own MoE variants, rather than providing systematic cross-method comparisons. In contrast, our work aims to provide a unified and comprehensive empirical analysis of leading MoE algorithms under a standardized evaluation protocol. By systematically comparing a diverse set of methods and probing the effects of key design parameters, we offer deeper insights into the factors driving performance and specialization in modern MoE architectures.

##### 2.3 Mixture of Experts Toolkits

Several open-source toolkits, including FastMoE (He et al., 2021), OpenMoE+t5x (Xue et al., 2024), and Tutel (Hwang et al., 2023), support the implementation of SMoE algorithms. However, these frameworks present notable limitations for contemporary research. Tutel and OpenMoE+t5x are primarily designed for large-scale pretraining on hundreds of GPUs, restricting accessibility for groups with limited resources. FastMoE, while more lightweight, lacks support for recent LLM architectures and advanced distributed training libraries such as DeepSpeed (Lian et al., 2024). In contrast, LibMoE is specifically designed to lower these barriers: researchers can train models with as few as 1B tokens for vision–language tasks and 6B tokens for language modeling representing a 1,000× reduction in data requirements compared to OpenMoE while benefiting from support for modern LLMs and modular distributed training.

#### 3 Designing LibMoE

##### 3.1 Preliminary: Mixture of Experts

The standard SMoE layer (Shazeer et al., 2017b) consists of a router R(·,Wr), parameterized by Wr, and a set of N experts {g(·,We

for i ∈ [N]. Given an input token x, the router computes an affinity score vector over all experts as: sR = σ TopK−∞(x⊤Wr) , where σ is a scoring function, typically implemented as a softmax or sigmoid. The operator TopK−∞ retains the top-K values and sets the remaining entries to negative infinity (−∞), enforcing sparsity. The SMoE output is then calculated as a weighted sum of expert outputs, modulated by the affinity scores:

)}Ni=1, each with parameters We

i

i

N

yˆ =

siR · g(x;We

). (1)

i

i=1

In practice, K is often chosen such that K < N to reduce computational cost while preserving performance.

##### 3.2 Vision-Language Model: LLaVA Architecture

A key challenge in training SMoE models is obtaining a massive dataset and a large amount of compute. Thus, beyond the traditional pre-training setting, we propose to incorporate SMoE training into any existing dense LLM checkpoints via the Sparse Upcycling technique (Komatsuzaki et al., 2022), which duplicates the original model to create experts and continue training them on a downstream dataset as a normal SMoE. Consequently, we can bypass the expensive pre-training step and evaluate SMoE algorithms with the most advanced public LLMs.

Training pipeline We adopt a vision–language pretraining task, which represents a challenging multimodal learning setting while requiring a relatively modest amount of data to initiate training (approximately 1.4B tokens). Following the CUMO framework (Li et al., 2024b), we upcycle the LLaVA model (Liu

- et al., 2023a), which consists of a pretrained visual encoder, a pretrained large language model (LLM), and a randomly initialized vision–language MLP connector. Our training procedure follows a two-stage paradigm, comprising dense training and sparse MoE training. During dense training, we adopt a two-step strategy consisting of pre-training and pre-finetuning. In the pre-training stage, only the vision–language

[Figure 1]

[Figure 2]

Trainable parameter Non-trainable parameter

[Figure 3]

[Figure 4]

[Figure 5]

Large Language Model

Large Language Model

Large Language Model

Sparse Upcycling

[Figure 6]

[Figure 7]

[Figure 8]

MLP

MLP

MLP - MoE

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Vision Encoder Word Embedding

Vision Encoder Word Embedding

VE - MoE Word Embedding

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

(I) (II) (III)

Dense Training MoE Training

- Figure 1: Overview of the LibMoE-VLM architecture and training process. In the first stage of Dense Training, only the MLP is trained to improve alignment. In the second stage, all parameters are trained. During SMoE Training, the feed-forward networks (FFNs) of the Vision Encoder (VE) and MLP Connector are used to initialize the experts within the SMoE framework, and all parameters continue to be trained.

MLP connector is optimized, while both the visual encoder and the LLM remain frozen. This stage aligns visual representations with the LLM embedding space and establishes a stable multimodal interface without perturbing the pretrained backbones. Subsequently, in the pre-finetuning stage, all model parameters are unfrozen and jointly optimized using high-quality image–caption data. After dense training, we perform visual instruction tuning by converting the model into an SMoE architecture via sparse upcycling, as illustrated in Figure 1. Following prior findings (Li et al., 2024b), sparse upcycling is restricted to the MLP connector and visual encoder, since upcycling a dense LLM has been shown to be less effective than directly employing an SMoE LLM. Importantly, the dense training stage is agnostic to SMoE-specific design choices, allowing the resulting checkpoints to be reused across different SMoE algorithms. In our largest-scale experimental setting using 4×H100 GPUs, the complete training pipeline requires slightly over 54 hours in total, with the sparse MoE (visual instruction tuning) stage accounting for approximately 17 hours on the LLaVA-665K dataset or 35 hours on the OneVision (1.2M) dataset. Notably, the dense training stage (approximately 19 hours) is performed only once and reused when comparing different MoE algorithms, which significantly reduces the marginal cost of benchmarking. These results demonstrate that the proposed training setup is computationally accessible. Full specifications of model scales, datasets, and hyperparameters are provided in Appendices A.1 and B.1.

##### 3.3 Pre-training LLM: Decoder-only Switch Transformer

A key design of our language modeling experiment is to establish a standardized yet flexible setup that enables fair comparison across different MoE algorithms while maintaining computational efficiency. We adopt a decoder-only Transformer architecture in the style of Switch Transformer (Fedus et al., 2022), where the feedforward sublayer is replaced by sparsely activated MoE modules. This design supports scaling from small (0.15B) to large (0.68B) models and provides a unified backbone for evaluating MoE variants. To ensure consistency, we keep the architecture and optimization pipeline fixed across experiments, varying only the routing strategies.

Training pipeline. Our experimental setup builds upon the framework of Csordás et al. (2023), with adjustments tailored to our model configurations and minor refinements to the baseline hyperparameters. Our codebase supports all major forms of parallelism, including data, tensor, pipeline, and expert parallelism, ensuring scalability across model sizes. MoE variants are implemented by modularly replacing the MoE component, while keeping other training components identical. This design ensures consistency and reproducibility across experiments, allowing us to isolate the impact of routing. Full specifications of model scales, datasets, and hyperparameters are provided in Appendix A.2 and Appendix B.2.

##### 3.4 Comprehensive Evaluation of SMoE Algorithms

Beyond model scaling, rigorous evaluation plays a crucial role in identifying SMoE algorithms that translate to real-world impact. Traditional machine learning studies typically follow the training–validation–testing paradigm, where separate models are trained and evaluated on individual benchmarks. Many existing SMoE works (Chi et al., 2022; Do et al., 2023) adopt this framework, training a dedicated model per benchmark. In contrast, large language models (LLMs) have introduced a paradigm shift, wherein a single model trained on massive corpora can generalize across benchmarks without explicit fine-tuning a setting commonly referred to as zero-shot evaluation. To align SMoE development with this emerging paradigm and real-world deployment scenarios, LibMoE is designed to support zero-shot evaluation across both language and vision-language domains. Specifically, for vision-language modeling, we extend the LMMS-Eval framework (Zhang et al., 2024a) to evaluate the final checkpoints of various SMoE algorithms. We carefully select 11 widely used benchmarks from LMMS-Eval and report their performance, while also providing a LibMoE model loader to enable future researchers to explore nearly 100 supported benchmarks with minimal effort. In the language modeling setting, we incorporate nine widely recognized zero-shot benchmarks, including HellaSwag(Zellers et al., 2019) and ARC-Challenge(Clark et al., 2018), which serve as canonical examples of standard evaluation datasets. These benchmarks are frequently used in the assessment of state-of-the-art models such as GPT4(OpenAI et al., 2023) and DeepSeek-V3(Liu et al., 2024c), making them highly representative of real-world evaluation practices. By adopting these standardized tasks, our evaluation aligns closely with established protocols in the LLM community.

Overall, LibMoE is designed to facilitate comprehensive, zero-shot evaluation of SMoE algorithms under realistic conditions, bridging the gap between research experimentation and deployment-oriented assessment.

##### 3.5 Design Principles

LibMoE is designed with a strong emphasis on modularity, extensibility, and practical usability to accelerate research on SMoE algorithms. At its core, LibMoE consists of three major modules. The MoE module provides a unified abstraction for routing strategies, supporting customizable designs such as gating mechanisms, load balancing losses, and expert selection logic. The training module handles optimization workflows across both LLM and VLM tasks, allowing researchers to integrate pretrained backbones, apply sparse upcycling or pretraining from scratch, and utilize distributed strategies such as data parallelism and model sharding. Finally, the evaluation module enables seamless plug-and-play evaluation, supporting a wide range of benchmarks and task formats through standardized APIs.

In addition to providing baseline implementations of leading SMoE algorithms, LibMoE emphasizes ease of experimentation: researchers can rapidly prototype new routing designs or modify existing pipelines with minimal code changes. Its modular infrastructure allows each component routing, training, and evaluation to be extended or swapped independently. Moreover, by abstracting common functionality across tasks and model types, LibMoE lowers the engineering burden typically associated with SMoE experimentation, especially in resource-constrained settings.

#### 4 Training LibMoE

##### 4.1 SMoE Algorithms

To demonstrate that LibMoE is well-suited for SMoE research, we implement and benchmark seven stateof-the-art SMoE algorithms. First, we consider methods that modify or improve the router network. These include the original SMoE (Fedus et al., 2022), which uses a softmax router; the sigmoid-gated SMoE (Csordás

- et al., 2023), which has been shown to achieve faster convergence; and XMoE (Chi et al., 2022), which mitigates representation collapse and enhances routing capacity through input dimension reduction and cosine-normalized routing. Second, we incorporate recent designs that utilize shared expert mechanisms, such as DeepSeek-V2 (Liu et al., 2024a) with softmax routing and DeepSeek-V3 (Liu et al., 2024c) with a sigmoid router. Inspired by these designs, we implement SharedE-V2 and SharedE-V3, which adopt the shared-expert principles of DeepSeek-V2 and DeepSeek-V3, respectively. Lastly, we include MoE++(Jin

- et al., 2024), which augments standard SMoE with zero-computation experts and pathway-aware routing to improve efficiency and throughput, and TC-MoE(Yan et al., 2025), which introduces a ternary expert choice space and reward-based routing to reduce redundancy while enhancing model expressiveness. These diverse algorithms allow us to evaluate LibMoE’s compatibility with a wide range of routing strategies and architectural innovations.

- Table 1: Performance of SMoE variants with 6 versus 3 active experts on a ViT backbone (5.6 B parameters). Bold values mark the best score in each column; ↓/↑ indicate that lower/higher values are preferable, respectively.

MoE Method

AI2D

Text VQA

GQA

MM Bench

Hallusion Bench

Math Vista

MMMU MMStar Pope MME

MME RW

OCR Bench

AVG Acc ↑

AVG Rank↓

LLAVA + OneVision / 1M2 samples

SMoE 69.56 43.93 61.51 71.31 46.90 37.90 41.56 41.23 86.28 63.33 27.83 37.50 52.40 5.96 XMoE 69.72 43.93 61.52 72.25 47.42 38.50 42.11 43.99 86.61 63.81 29.18 39.40 53.20 3.33 σ-MoE 69.79 44.69 61.70 71.74 47.00 38.40 43.11 42.08 86.69 63.80 29.70 38.40 53.09 3.33 SharedE-V2 70.56 45.04 61.34 71.13 47.11 39.50 42.78 42.73 86.53 63.93 29.55 38.70 53.24 3.33 SharedE-V3 71.92 44.59 61.93 72.59 46.37 38.20 42.33 43.30 86.78 64.61 29.29 39.40 53.44 2.46 TC-MOE 70.08 43.75 61.89 71.05 45.74 38.10 41.89 43.64 86.76 63.05 31.84 38.30 53.01 4.50 MOE++ 70.13 43.37 61.52 71.39 46.16 38.60 40.78 43.24 86.60 63.26 28.19 37.50 52.56 5.08

LLAVA / 665K samples

SMoE 65.52 41.51 61.62 72.25 41.75 29.50 41.67 42.24 87.13 61.05 32.15 31.70 50.67 3.42 XMoE 65.84 41.96 61.61 72.16 41.85 31.60 41.67 40.32 86.64 60.95 32.73 33.20 50.88 3.08 σ-MoE 65.52 42.08 61.74 71.22 40.80 30.30 41.22 41.99 86.64 61.44 32.31 32.50 50.65 3.71 SharedE-V2 64.77 41.96 61.27 71.74 41.85 30.90 43.33 41.56 86.82 60.52 32.88 31.40 50.75 3.67 SharedE-V3 65.58 42.06 61.26 72.42 41.43 30.60 42.44 41.75 86.81 60.93 31.47 32.60 50.86 3.33 TC-MOE 65.50 40.70 61.19 71.22 42.06 29.30 41.22 41.10 86.53 60.30 31.89 33.00 50.34 5.42 MOE++ 65.03 41.44 60.61 71.74 42.69 30.30 43.00 40.32 86.58 60.11 31.32 31.20 50.36 5.38

- Table 2: Performance of SMoE variants with 66 total experts and 8 active experts in the language pretraining setting, evaluated on small-scale (0.15B) and large-scale (0.68B) models. PPL denotes perplexity for language modeling. Bold values denote the best result in each column; ↓/↑ indicate that lower/higher values are preferable, respectively.

LAM BADA

Hella Swag

ARCEasy

Common SenseQA

AVG Acc ↑

AVG Rank ↓

MoE Method PPL ↓

BLiMP CBT

PIQA

RACE SIQA

SMoE 13.63 25.27 77.71 84.18 29.43 57.94 32.68 30.11 35.62 24.65 44.18 4.50 XMoE 13.98 24.57 76.53 84.12 29.34 58.27 32.26 29.69 35.47 24.49 43.86 6.45 σ-MoE 13.61 25.43 77.38 84.23 29.13 58.92 32.73 31.05 34.90 24.90 44.30 4.30

Small Model (0.15B)

SharedE-V2 13.49 25.29 77.37 84.33 29.38 60.17 33.83 31.02 35.57 24.98 44.66 2.80 SharedE-V3 13.42 25.49 77.20 84.40 29.38 59.14 32.52 30.60 35.57 25.47 44.42 3.20

TC-MoE 13.51 25.60 76.91 84.68 29.27 59.03 33.02 30.63 36.03 26.37 44.62 2.90

MoE++ 13.54 25.45 77.23 84.83 29.28 58.49 33.49 30.11 35.62 24.49 44.33 3.85

SMoE 9.51 37.13 80.47 89.83 37.49 64.36 38.22 33.03 37.41 26.54 49.39 5.15 XMoE 9.66 35.25 80.38 89.35 37.19 64.20 38.99 32.95 37.77 28.34 49.38 5.75 σ-MoE 9.46 37.56 81.08 89.57 37.52 64.91 39.15 32.68 37.67 28.50 49.85 3.60

Large Model (0.68B)

SharedE-V2 9.52 37.11 80.98 89.93 37.14 64.36 38.06 33.17 36.95 27.35 49.45 5.25 SharedE-V3 9.49 36.88 81.28 89.65 37.32 65.72 38.86 33.12 38.59 28.09 49.95 3.60

TC-MoE 9.38 37.87 81.21 90.19 37.95 64.47 39.28 33.77 37.92 27.85 50.06 2.35

MoE++ 9.38 38.80 80.88 89.77 37.70 64.64 39.37 34.02 37.97 28.34 50.16 2.30

##### 4.2 Performance Comparison

Across both vision–language modeling and language modeling settings, we observe a consistent trend: performance differences among contemporary SMoE variants remain relatively modest under matched training conditions. In the VLM sparse upcycling regime, no single method consistently outperforms others, with only marginal variations observed across benchmarks. Similarly, in language model pretraining, while recent MoE designs exhibit incremental improvements, these gains do not amount to a decisive advantage over the standard SMoE formulation. Collectively, these results reflect current practical deployments of frontier models, where the original SMoE design is often preferred for its simplicity, stability, and scalability, as opposed to more complex routing mechanisms that yield limited empirical benefits under comparable training conditions.

#### 5 Analysis

Beyond empirical performance, we analyze how different MoE architectures route, specialize, and balance experts during training. We focus on expert selection dynamics, capacity utilization, and the effects of initialization, architecture, and task domain. Our analysis covers both from-scratch pretraining (0.15B LLM on 6.85B tokens) and sparse upcycling (VLM fine-tuned on LLaVA-665K), providing a unified view of routing behavior across regimes.

-MoE SharedE-V2 SharedE-V3 SMoE++ SMoE TC-MoE X-MoE

###### MME

MathVista-Mini

Validation

HellaSwag

0.80

0.7

0.7

0.375

0.75

ExpertChangeRate

ExpertChangeRate

ExpertChangeRate

ExpertChangeRate

0.6

0.6

0.350

0.70

0.325

0.65

0.5

0.5

0.60

0.300

0.4

0.4

0.55

0.275

0.3

0.50

0.3

0.250

20-40 40-60 60-80 80-100

20-40 40-60 60-80 80-100

10-2020-3030-4040-5050-6060-7070-8080-9090-100

10-2020-3030-4040-5050-6060-7070-8080-9090-100

Data Percentage

Data Percentage

Data Percentage

Data Percentage

Figure 2: Impact of Training Data Percentage on Expert Selection.

- a) How Stable Is Expert Routing Throughout Training in MoE Algorithms? This experiment investigates how training-data scale influences expert-selection dynamics across SMoE algorithms. We quantify routing stability using the Expert Change Rate (ECR). For a fixed evaluation set D, we compare consecutive checkpoints and, for each token at each MoE layer, record the selected expert. We define ECR(D) as the fraction of token-to-expert assignments on D (aggregated across MoE layers) for which the selected expert differs between the two checkpoints. Lower ECR corresponds to more stable routing, where the router makes fewer reassignment decisions for identical inputs, whereas higher ECR indicates more volatile routing, with experts being reassigned more frequently over the course of training.

- Figure 2 reports ECR throughout training for both vision-language models (MME, MathVista; sparse upcycling) and language models (Validation, HellaSwag; from-scratch pretraining). Across all settings, we observe that SMoE algorithms exhibit a consistently decreasing ECR as training progresses. Among the evaluated methods, XMoE shows stronger late-stage convergence on VLM tasks. We attribute this to its cosinenormalized routing and the smaller learning rate, which makes this setup more efficient. In contrast, during language-model pretraining, XMoE exhibits a relatively high ECR (peaking around the 40% training mark), consistent with the degraded performance reported in Table 2. We hypothesize that this degradation arises from an interaction between cosine-normalized routing and the pretraining learning rate. By bounding the router logits (up to a temperature/scale), cosine/L2 normalization can reduce the effective logit margin and make softmax gating more sensitive to the chosen scale and step size (Agarwala et al., 2020). Moreover, since normalization choices can affect gradient scaling, an overly large learning rate can exacerbate optimization instability and even trigger divergence accompanied by growing internal activations/norms (Xiong et al., 2020; Rybakov et al., 2024).

-MoE

SharedE-V3

SMoE

X-MoE

SharedE-V2

SMoE++

TC-MoE

SharedE-Upcyc V2

Validation

HellaSwag

0.80

0.375

0.75

ExpertChangeRate

ExpertChangeRate

0.350

0.70

0.325

0.65

0.300

0.60

0.275

0.55

0.250

0.50

0.225

0.45

10-2020-3030-4040-5050-6060-7070-8080-9090-100

10-2020-3030-4040-5050-6060-7070-8080-9090-100

Data Percentage

Data Percentage

Figure 3: Effect of upcycled shared experts trained on prior tasks on routing behavior, measured by expert change rate during language model pretraining.

Furthermore, we observe a divergent trend in the behavior of shared expert architectures namely, SharedE-V2 and SharedE-V3 across different tasks. In vision language models (VLMs), both variants exhibit significantly higher routing stability compared to other methods, likely because the router effectively selects among only K−S non-shared experts (with S shared experts always active), which reduces the degrees of freedom in

expert assignment. However, this advantage disappears during language model pretraining. We hypothesize that this divergence arises from the distinct initialization schemes and functional roles of shared experts. In the VLM sparse upcycling setup, SMoE layers are initialized from a pretrained dense model, where shared experts already encode general purpose visual linguistic representations. These pretrained experts act as stable anchors, enabling the router to focus its capacity on distributing tokens among non-shared experts, thereby reducing volatility in expert selection. In contrast, language model pretraining starts from random initialization. Here, both the router and shared experts must co-adapt from scratch. As the shared experts’ representations evolve over time, their influence on the final output changes accordingly. This compels the router to continuously revise its decisions not just for load balancing but also to adapt to the shifting behavior of shared experts in order to minimize the task loss. The result is a tight coupling between unstable shared experts and routing dynamics, which leads to higher expert change rates and delayed convergence.

To validate this hypothesis, we conduct a controlled experiment (Figure 3) by weakening the shared expert’s influence: we upcycle only the shared expert from a pretrained checkpoint while randomly initializing the rest of the model, referred to as SharedE-Upcyc V2. The results show that when the shared expert is strong and stable, the router can focus on allocating inputs to task-specific experts leading to a lower expert change rate compared to other baselines. Interestingly, this stabilizing effect diminishes as training data increases. In large-scale regimes (e.g., Table 1, trained on over 1M samples), SharedE-V2 and SharedE-V3 not only preserve routing stability but also achieve top performance. This confirms that shared expert architectures are particularly beneficial in data-rich scenarios, where their pre-initialized knowledge complements the router’s optimization.

Taken together, these findings highlight that routing stability is shaped by the training regime. In fromscratch language-model pretraining, both the router and experts must co-adapt under random initialization, so early-stage stability is critical to prevent prolonged assignment churn. In contrast, VLM sparse upcycling starts from a pretrained dense checkpoint, where experts can serve as stable general-purpose anchors; consequently, routing tends to evolve more smoothly and robustness becomes more important than rapid early convergence. Therefore, effective sparse MoE designs should account for both regimes: fast stabilization during from-scratch pretraining and sustained robustness during upcycling.

VLM: Drop Top-1 vs Top-1&2

LM: Drop Top-1 vs Top-1&2

10

2.3

0

-0.9

-1.7

-2.0

-2.6

-3.1 -2.8

-3.9 -3.6

TotalScore

TotalScore

-3.7

10

-16.5

-20.6

-84.9

100

-90.9

100

-102.4

-112.4

-114.5 -117.8

-127.9

Drop Top-1

Drop Top-1&2

-139.7 -141.7 -136.1 -131.4 -129.8

-139.7 -134.8

| |
|---|

| |
|---|

-230.5

-238.4

SMoE XMoE -MoE SharedE-V2SharedE-V3 TC-MoE MoE++

SMoE XMoE -MoE SharedE-V2SharedE-V3 TC-MoE MoE++

Figure 4: Performance of SMoE variants under routing perturbations, where the top-1 expert is replaced by the top-(K+1) expert in vision–language and language modeling tasks.

- b) Are the Selected Experts Truly Optimal in MoE Algorithms? To assess whether SMoE algorithms fully exploit their expert capacity, we conduct a diagnostic routing perturbation experiment across seven architectures. Specifically, we intentionally degrade the routing decision by replacing the top-1 expert (i.e., the expert with the highest affinity score) with the top-(K+1) expert, and quantify the resulting performance change using ∆Total Score (see Appendix E.1 for details). Here, a higher ∆Total Score indicates improved performance relative to the original routing configuration. This controlled perturbation allows us to systematically measure each model’s sensitivity to suboptimal expert assignments and to identify potential inefficiencies in expert utilization. Figure 4 reports the cumulative accuracy drop across benchmarks under two settings: replacing only the top-1 expert (DropTop1) and replacing both the top-1 and top-2 experts (DropTop1&2). As expected, most SMoE variants exhibit performance degradation, confirming that precise expert selection is critical to maintaining model quality. Overall, in the language modeling (pretraining) setting, all methods suffer large and relatively uniform performance drops, reflecting a consistent reliance on precise expert routing across architectures. In contrast, vision-language models exhibit greater divergence.

Notably, SMoE shows a counterintuitive a +2.3 gain in ∆Total Score under the DropTop1 perturbation suggesting that its original routing configuration may have failed to effectively utilize its expert capacity. This exposes a potential inefficiency in the router, where suboptimal expert choices actually yield better outcomes. Conversely, TC-MoE suffers the most severe degradation, highlighting its strong dependence on highly optimized, task-aligned routing decisions.

In summary, this experiment highlights that the impact of routing quality is fundamentally intertwined with architectural design across SMoE variants. Architectures such as TC-MoE and MoE++ exhibit pronounced performance degradation under routing perturbations, indicating that their gains rely heavily on precise and task-aligned expert selection. This sensitivity reveals a strong coupling between routing robustness and the effectiveness with which the architecture exploits its representational capacity. However, the results also demonstrate that routing robustness alone is insufficient to guarantee the highest absolute performance. Each SMoE variant is constrained by an inherent architectural capacity ceiling, beyond which improvements in routing precision yield diminishing returns. Consequently, optimal performance depends not only on selecting the “right” experts, but on whether the routing mechanism is capable of fully leveraging the architectural capacity available. These findings suggest that routing behavior should be interpreted in conjunction with architectural design, rather than as an isolated indicator of model quality.

- c) How Does Normalized Expert-Allocation Entropy Reveal Domain Specialization Across SMoE Variants? In Figure 5, we examine the behavior of the expert selection mechanism by measuring how frequently each expert is activated across subtasks in the MME benchmark, using the Expert Allocation Entropy (EAE) metric (see Appendix E.2). This analysis enables us to assess the degree of specialization exhibited by individual experts across different domains.

-MoE

SMoE

SharedE-V2

SharedE-V3

0.9882

0.9888

0.9843

0.9874

txt_trans

txt_trans

num_calc

num_calc

0.9898

0.9909

0.9889

0.9880

common_sense

OCR

code_reason

code_reason

0.9904

0.9911

0.9892

0.9905

common_sense

txt_trans

num_calc

num_calc

0.9915

0.9918

0.9904

0.9905

common_sense

common_sense

artwork

posters

0.9918

0.9926

0.9911

0.9915

posters

OCR

artwork

posters

0.9918

0.9927

0.9919

0.9916

txt_trans

artwork

posters

artwork

Domains

0.9923

0.9928

0.9925

0.9920

code_reason

celebs

OCR

celebs

0.9931

0.9932

0.9932

0.9922

celebs

count

color

color

0.9932

0.9933

0.9939

0.9932

count

code_reason

count

OCR

0.9935

0.9937

0.9939

0.9939

scene

scene

position

count

0.9936

0.9939

0.9939

0.9939

position

color

scene

position

0.9937

0.9950

0.9945

0.9943

color

existence

existence

scene

0.9946

0.9950

0.9945

0.9944

landmark

landmark

landmark

landmark

0.9951

0.9951

0.9952

0.9947

existence

position

celebs

existence

0.980 0.985 0.990 0.995 1.000 1.005

0.980 0.985 0.990 0.995 1.000 1.005

0.975 0.980 0.985 0.990 0.995 1.000 1.005

0.980 0.985 0.990 0.995 1.000

Expert Allocation Entropy (norm.)

Expert Allocation Entropy (norm.)

Expert Allocation Entropy (norm.)

Expert Allocation Entropy (norm.)

XMoE

TCMoE

MoE++

0.7862

0.6825

0.9206

num_calc

num_calc

num_calc

0.8058

0.6894

0.9207

celebs

code_reason

artwork

0.8096

0.6991

0.9212

txt_trans

common_sense

celebs

0.8262

0.7045

0.9213

txt_trans

code_reason

posters

0.8312

0.7089

0.9216

common_sense

common_sense

OCR

0.8374

0.7129

0.9219

txt_trans

landmark

artwork

Domains

0.8391

0.7143

0.9232

OCR

landmark

position

0.8413

0.7167

0.9236

existence

OCR

color

0.8454

0.7170

0.9237

scene

count

count

0.8464

0.7183

0.9242

color

color

scene

0.8474

0.7193

0.9245

count

existence

landmark

0.8494

0.7198

0.9250

artwork

scene

celebs

0.8497

0.7255

0.9250

position

position

existence

0.8503

0.7257

0.9257

posters

posters

code_reason

0.78 0.79 0.80 0.81 0.82 0.83 0.84 0.85 0.86

0.68 0.69 0.70 0.71 0.72 0.73

0.915 0.920 0.925 0.930 0.935

Expert Allocation Entropy (norm.)

Expert Allocation Entropy (norm.)

Expert Allocation Entropy (norm.)

- Figure 5: Normalized expert-allocation entropy across domains for seven SMoE variants.

We observe that reasoning-centric subtasks such as text_translation, code_reasoning, and numerical_calculation induce more distinctive expert activation patterns compared to simpler tasks. These subtasks often lead to sharper expert allocation boundaries, reflecting a higher degree of domainspecific specialization in the routing behavior. Among the evaluated methods, TC-MoE and XMoE demonstrate the strongest specialization tendencies. TC-MoE, in particular, exhibits a highly concentrated expert selection pattern, with up to 64% of routing focused on a small subset of experts. While this suggests an aggressive specialization mechanism, it also raises concerns about over-specialization, as many experts remain underutilized, potentially limiting the model’s robustness and generalization capacity. In contrast, XMoE achieves a more favorable trade-off. It maintains a lower overall entropy, indicating specialization, while also preserving better load balance across tasks. Notably, XMoE exhibits a wide EAE variation of approximately

- 6% across subtasks (from EAE = 0.78 to EAE = 0.85), suggesting that it dynamically adapts expert routing depending on task complexity. For complex tasks, XMoE tends to engage a more selective subset of experts,

whereas for simpler tasks, it distributes computation more broadly, enabling better generalization while still retaining task sensitivity.

In comparison, methods such as SMoE, σ-MoE, MoE++, SharedE-V2, and SharedE-V3 maintain a nearly uniform expert distribution across all subtasks, with EAE variations consistently below 1%. This consistent balance suggests that these routers prioritize even expert utilization over task-specific specialization. However, such uniformity may also reflect a lack of task-sensitive expert discovery, where the model fails to identify and leverage domain-relevant experts effectively. As a result, their capacity to capture fine-grained, specialized knowledge remains limited.

Overall, these findings reveal a spectrum of expert allocation behaviors from highly specialized but potentially imbalanced (e.g., TC-MoE), to adaptive and task-aware (e.g., XMoE), to balanced yet under-specialized (e.g., SMoE family) highlighting the importance of routing strategies that balance specialization with generalization.

| |Task Group| |
|---|---|---|
| |Coarse-Grained<br><br>Fine-Grained| |
| |OCR<br><br>Reasoning| |

ExpertWeightAllocationEntropy

0.98

0.97

0.96

0.95

0.94

0.93

0.92

SMoE -MoE SharedE-V2 SharedE-V3 XMoE TCMoE MoE++

SMoE Variant

- Figure 6: Normalized entropy of expert weight allocation across different routing strategies.

- d) How Sparse MoE Methods Allocate Weights Across Experts and Tasks? Beyond analyzing how expert selection varies across subtasks, we also investigate how the router distributes the weight mass of sR among the selected experts a critical factor that influences the model’s output behavior.

In Figure 6, we report the normalized entropy of expert-weight allocation (EWA), calculated as the Shannon entropy of the softmax router weights over selected experts, and normalized with the maximum entropy of a vector with K elements and averaged over samples within each task group OCR, Coarse-Grained, and Reasoning to reveal trends in how confidently or evenly each routing strategy distributes weight across experts, which EWA larger than weights more uniform. Notably, XMoE exhibits the most desirable profile across all three axes of evaluation. First, it achieves the highest EWA, indicating that the router distributes weight relatively evenly among the selected experts. Second, the entropy remains remarkably consistent across all task groups, with nearly identical scores for Coarse-Grained, Fine-Grained, OCR, and Reasoning. This reflects a routing strategy balance weights to experts reduce appear some expert nominate and each expert contribution to output is nearly equal. At the other extreme, TC-MoE records the lowest EWA across all tasks, with minimal inter-task variation. This indicates that the router consistently concentrates weight on a small subset of experts, regardless of task type. While such focused routing may encourage sharper specialization, it also limits expert diversity, wastes model capacity, and increases vulnerability to under-performing experts. Other methods, including SMoE, σ-MoE, MoE++, and SharedE-V2/V3, fall into an intermediate regime. They consistently exhibit higher EWA on reasoning tasks than on simpler tasks, indicating that the router distributes weights more broadly across experts when handling complex problems. In contrast, for simpler tasks, routing decisions tend to concentrate weight on a smaller subset of experts.

In summary, the EWA analysis reveals key differences in how effectively each routing strategy leverages its experts. We observe 3 main behaviors: (i) methods with uniformly high EWA can fully exploit expert capacity across diverse conditions without over-concentration or instability, while (ii) methods with taskdependent variation show that reasoning tasks benefit from more uniform expert utilization than other task types. (iii) excessive specialization toward a small subset of experts limits the router’s ability to exploit complementary expertise, leaving much of the model capacity underutilized. Taken together, the results

demonstrate that different MoE methods encode markedly different routing weight allocation patterns, each with unique trade-offs in specialization and capacity utilization.

SMoE -MoE X-MoE MoE++ SharedE-V2 SharedE-V3 TCMoE

###### Validation

###### HellaSwag

###### MMStar

MMMU_VAL

0.4

0.04

RouterMargin

0.020

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

0.3

0.3

0.03

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.015

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

0.2

| |
|---|

0.02

0.2

| |
|---|

0.010

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

0.1

| |
|---|

0.01

0.1

| |
|---|

| |
|---|

0.005

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

0.00

0.000

0.0

0.0

| |
|---|

3 6 9 12 15

3 6 9 12 15

0 4 8 12 16 20 24

0 4 8 12 16 20 24

Layer_ids

Figure 7: Layer-wise router margin for different MoE methods, router margin is defined by computing the difference between the top-1 and top-2 routing scores. Left: router margin of pre-training LLM on Validation set and Hellaswag; Right: router margin of vision encoder in vision-language model in MMStar and MMMU Validation

- e) Does Router Confidence Grow with Depth in MoE Networks? Router margin measures the dominance of the top-ranked expert in a Mixture-of-Experts (MoE) routing decision. It is defined as the difference between the top-1 and top-2 routing scores, with a larger margin indicating a stronger preference for the top-1 expert (see Appendix E.3). Figure 7 shows the layer-wise router margin across all MoE methods, revealing a consistent trend of increasing margin with model depth in both language modeling and vision–language tasks. This indicates that expert dominance becomes more pronounced in deeper layers. In particular, the router margin is close to zero in layer 0 for most methods, reflecting early training stages where routing is uncertain and experts receive balanced weights. As depth increases, the rising margin signals a transition from generalist processing toward more specialized expert allocation, with the exception of the final layer in vision–language models, where the pattern deviates.

3 6 9 12 15

0.00

0.01

0.02

0.03

0.04

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

| |
|---|

Validation

3 6 9 12 15

0.000

0.005

0.010

0.015

0.020

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

HellaSwag

Layer_ids

RouterMargin

SMoE

-MoE

X-MoE

MoE++

SharedE-V2 SharedE-V3

TCMoE

SharedE-Upcyc V2

Figure 8: Effect of upcycled shared experts trained on prior tasks on routing behavior, measured by exrouter margin during language model pretraining.

Notably, SharedE-V2 and SharedE-V3 exhibit the steepest increase in router margin in the language modeling setting, suggesting a more decisive routing strategy and potentially stronger expert specialization. In contrast, this trend does not carry over to VLM sparse upcycling, where their router margins are comparable to other methods. We hypothesize that this discrepancy stems from the sparse upcycling regime. To examine this, we conduct an additional experiment analogous to Section 5a. As shown in Figure 8, when the shared expert is upcycled, the router is relieved from allocating inputs to general-purpose experts and can instead emphasize task-specific experts, resulting in a lower router margin than in models trained from scratch. This behavior is consistent with the routing patterns discussed in Section 5.

- f) Do Experts Exhibit Similarity in Sparse Upcycling? In sparse upcycling, experts are initialized by duplicating dense MLP layers, raising the question of whether they remain similar or progressively specialize during training. To examine this, Figure 9 reports the layer-wise cosine similarity of expert output weights, where values close to zero indicate greater divergence between expert outputs. We find that most MoE variants exhibit low similarity, indicating that experts diverge and specialize over time despite identical initialization. Similarity is slightly higher at the input and output layers than in intermediate layers, with the effect most pronounced in MoE++, which employs both copy- and zero-experts and thus encourages

[Figure 21]

[Figure 22]

SMoE

0.060

XMoE

0.055

-MoE

MoEMethod

0.050

- SharedE-V2

- SharedE-V3

0.045

0.040

TC-MoE

0.035

MoE++

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

0

1

2

3

4

5

6

7

8

9

mm_projector

Layer ID

Figure 9: Layer-wise expert similarity across MoE methods.

greater diversity in middle layers. Overall, excessive expert similarity does not persist, supporting the validity of comparing MoE methods under the sparse upcycling regime.

- g) How efficient is it to initialize a weight router network for load balancing? Prior works have explored various load-balancing techniques to mitigate the expert imbalance problem in MoE models (Fedus et al., 2022; Wei et al., 2024; Wang et al., 2024b). However, aggressive balancing may lead to routing collapse (Shazeer et al., 2017b). Recent studies, such as DeepSeekV3 (Liu et al., 2024c), argue that overly strong load-balancing loss can degrade model performance. To address this, they employed auxiliary-loss-free strategies (Wang et al., 2024b) and reduced the coefficient of the standard balancing loss (Fedus et al., 2022). However, these approaches require introducing additional parameters.

Interestingly, in our study, we observe that the initialization standard deviation (std) of the router network weights alone can influence balance dynamics even without modifying the loss function. As shown in Figure 10, we initialize the SMoE router network with different std values (0.02, 0.04, and 0.06), and train for 1000 steps using the standard balance loss (Fedus et al., 2022). The results clearly show that a smaller initialization std leads to better load balance under the same softmax routing configuration. This suggests that subtle changes in the initial logit distribution can enhance routing diversity, offering an alternative or complementary axis to tune alongside both the auxiliary loss coefficient and the initialization std.

SMoE = 0.02 SMoE = 0.04 SMoE = 0.06

3.0

2.5

BalanceLoss

2.0

1.5

1.0

0 200 400 600 800 1000

Global Step

Figure 10: Impact of initialization standard deviation on balance loss dynamics in SMoE models.

#### 6 Summary of Key Results

Using LibMoE, we observe that the performance gap among current MoE algorithms is relatively small, as shown in Table 1 and Table 2, suggesting that future MoE designs must deliver more substantial performance improvements to justify their added architectural complexity. However, beyond similar end-task accuracy, our behavioral analysis reveals meaningful differences across algorithms in three key dimensions.

First, in terms of routing dynamics, we find that expert routing stability varies significantly: some algorithms converge quickly, while others remain volatile for longer, as measured by expert change rates. We also show that routing optimality remains a challenge especially in VLM settings where models like vanilla SMoE often select suboptimal experts. Furthermore, routing entropy reveals task dependent specialization patterns: on reasoning tasks, routers tend to assign uniform weights across a narrow subset of experts, while perception

tasks engage a broader pool of experts but distribute weights more unevenly. In addition, expert selection margins increase in deeper layers, suggesting a natural progression from generalist to specialist routing as depth increases.

Second, in terms of load balancing, we find that a simple reduction in the router’s initialization standard deviation leads to significantly better expert utilization and balance in early training. This improvement requires no changes to the loss function or architecture, highlighting initialization scale as a lightweight yet highly effective control knob in MoE training.

Third, our analysis of training regimes reveals distinct differences in both routing stability and expert optimality between pretraining from scratch and sparse upcycling. In terms of routing stability, we find that during pretraining, routers must co-adapt with evolving expert representations especially shared experts which introduces high volatility in early training stages. In contrast, in sparse upcycling (particularly in VLMs), the shared experts are already pretrained and fixed, allowing the router to focus purely on expert selection. This results in significantly more stable and interpretable routing behavior. These dynamics are clearly demonstrated in our SharedE-V2 experiments. In terms of expert optimality, we analyze performance sensitivity via DropTop-1 and DropTop1&2 perturbation. In the pretraining setting, all algorithms exhibit large and relatively uniform performance drops when the top expert is masked, suggesting a shared reliance on fragile routing decisions regardless of algorithm. However, in sparse upcycling, the performance drop patterns differ substantially between algorithms, indicating that routing precision and its dependence on the router design is more algorithm specific in this regime. This highlights that sparse upcycling better exposes differences in routing optimality across methods, while pretraining tends to homogenize their behavior under stress.

Finally, we find that expert representations remain diverse across all algorithms, even without strong contrastive supervision. This reinforces the practicality of sparse upcycling as a compute-efficient and behaviorally valid alternative to full pretraining. Altogether, LibMoE enables researchers to systematically evaluate MoE algorithms across both early-stage (unstable pretraining) and late-stage (sparse upcycling) regimes, offering actionable insights that go beyond standard performance metrics.

#### 7 Conclusion

In this work, we presented LibMoE, a unified and extensible framework that enables reproducible research on Mixture-of-Experts models across language and vision–language domains. By integrating seven recent algorithms within standardized pipelines and offering analytical tools for routing, specialization, and load balancing, LibMoE lowers the barrier for experimentation under resource constraints while providing fair and systematic comparisons. Particularly, all experiments in this study were conducted on a cluster of 4×H100 GPUs, with the longest experiment taking just under 54 hours for VLMs (of which only 35 hours were spent on MoE training) and 43 hours for language model pretraining. Our comprehensive empirical study reveals that no single method universally dominates, with performance and stability shaped by task type, initialization, and expert design, underscoring the need for flexible and transparent evaluation. We hope that by bridging the gap between algorithmic innovation and practical deployment, LibMoE will not only accelerate MoE research but also provide the community with a principled foundation for building the next generation of scalable, efficient, and interpretable large models.

#### References

Marib Abdin et al. Phi-3 technical report: A highly capable language model series with long context. arXiv preprint arXiv:2404.14219, 2024. Includes the Phi-3.5-MoE variant.

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

Atish Agarwala, Jeffrey Pennington, Yann Dauphin, and Samuel S. Schoenholz. Temperature check: theory and practice for training models with softmax-cross-entropy losses. ArXiv, abs/2010.07344, 2020. URL https://api.semanticscholar.org/CorpusID:222380046.

Abhinav Agarwalla, Abhay Gupta, Alexandre Marques, Shubhra Pandit, Michael Goin, Eldar Kurtic, Kevin Leong, Tuan Nguyen, Mahmoud Salem, Dan Alistarh, et al. Enabling high-sparsity foundational llama models with efficient pretraining and deployment. arXiv preprint arXiv:2405.03594, 2024.

Elie Antoine, Frédéric Béchet, and Philippe Langlais. Part-of-speech sensitivity of routers in mixture of experts models. arXiv preprint arXiv:2412.16971, 2024.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, 2020.

Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. arXiv preprint arXiv:2402.11684, 2024a.

Ke Chen, Lei Xu, and Huisheng Chi. Improved learning algorithms for mixture of experts in multiclass classification. Neural networks, 12(9):1229–1252, 1999.

Lin Chen, Jinsong Li, Xiao wen Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? ArXiv, abs/2403.20330, 2024b. URL https://api.semanticscholar.org/CorpusID:268793433.

Shengzhuang Chen, Ying Wei, and Jonathan Richard Schwarz. Automatic expert discovery in llm upcycling via sparse interpolated mixture-of-experts. arXiv preprint arXiv:2506.12597, 2025.

Zewen Chi, Li Dong, Shaohan Huang, Damai Dai, Shuming Ma, Barun Patra, Saksham Singhal, Payal Bajaj, Xia Song, Xian-Ling Mao, et al. On the representation collapse of sparse mixture of experts. Advances in Neural Information Processing Systems, 35:34600–34613, 2022.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Together Computer. Redpajama: An open source recipe to reproduce llama training dataset, April 2023. URL https://github.com/togethercomputer/RedPajama-Data.

Róbert Csordás, Kazuki Irie, and Jürgen Schmidhuber. Approximating two-layer feedforward networks for efficient transformers. arXiv preprint arXiv:2310.10837, 2023.

Róbert Csordás, Kazuki Irie, Jürgen Schmidhuber, Christopher Potts, and Christopher D Manning. Moeut: Mixture-of-experts universal transformers. Advances in Neural Information Processing Systems, 37:28589– 28614, 2024.

Damai Dai, Li Dong, Shuming Ma, Bo Zheng, Zhifang Sui, Baobao Chang, and Furu Wei. Stablemoe: Stable

- routing strategy for mixture of experts. arXiv preprint arXiv:2204.08396, 2022a.

Damai Dai, Li Dong, Shuming Ma, Bo Zheng, Zhifang Sui, Baobao Chang, and Furu Wei. Stablemoe: Stable

- routing strategy for mixture of experts. arXiv preprint arXiv:2204.08396, 2022b.

Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024.

Databricks. Dbrx: A fine-grained mixture-of-experts open llm. Hugging Face model card and technical blog,

2024. URL https://huggingface.co/databricks/dbrx-instruct. 132B total, 36B active, 16 experts choose 4, 12T tokens, 32K context.

Nolan Dey, Daria Soboleva, Faisal Al-Khateeb, Bowen Yang, Ribhu Pathria, Hemant Khachane, Shaheer Muhammad, Zhiming, Chen, Robert Myers, Jacob Robert Steeves, Natalia Vassilieva, Marvin Tom, and Joel Hestness. Btlm-3b-8k: 7b parameter performance in a 3b parameter model, 2023. URL https: //arxiv.org/abs/2309.11568.

Giang Do, Khiem Le, Quang Pham, Trungtin Nguyen, Thanh-Nam Doan, Bint T Nguyen, Chenghao Liu, Savitha Ramasamy, Xiaoli Li, and Steven Hoi. Hyperrouter: Towards efficient training and inference of sparse mixture of experts. arXiv preprint arXiv:2312.07035, 2023.

David Eigen, Marc’Aurelio Ranzato, and Ilya Sutskever. Learning factored representations in a deep mixture of experts. arXiv preprint arXiv:1312.4314, 2013.

Dongyang Fan, Bettina Messmer, and Martin Jaggi. Towards an empirical understanding of moe design choices. arXiv preprint arXiv:2402.13089, 2024.

Zhiwen Fan, Rishov Sarkar, Ziyu Jiang, Tianlong Chen, Kai Zou, Yu Cheng, Cong Hao, Zhangyang Wang, et al. M3vit: Mixture-of-experts vision transformer for efficient multi-task learning with model-accelerator co-design. Advances in Neural Information Processing Systems, 35:28441–28457, 2022.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. ArXiv, abs/2306.13394, 2023. URL https://api.semanticscholar. org/CorpusID:259243928.

Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. Megablocks: Efficient sparse training with mixture-of-experts. arXiv preprint arXiv:2211.15841, 2022.

Mark JF Gales and SS Airey. Product of gaussians for speech recognition. Computer Speech & Language, 20(1):22–40, 2006.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xiyang Wu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In arXiv preprint, 2023. URL https://api.semanticscholar.org/CorpusID:265499116.

Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats L Richter, Quentin Anthony, Eugene Belilovsky, Irina Rish, and Timothée Lesort. Continual pre-training of large language models: How to (re) warm your model? arXiv preprint arXiv:2308.04014, 2023.

Xing Han, Huy Nguyen, Carl Harris, Nhat Ho, and Suchi Saria. Fusemoe: Mixture-of-experts transformers for fleximodal fusion. arXiv preprint arXiv:2402.03226, 2024a.

Xumeng Han, Longhui Wei, Zhiyang Dou, Zipeng Wang, Chenhui Qiang, Xin He, Yingfei Sun, Zhenjun

- Han, and Qi Tian. Vimoe: An empirical study of designing vision mixture-of-experts. arXiv preprint arXiv:2410.15732, 2024b.

Hussein Hazimeh, Zhe Zhao, Aakanksha Chowdhery, Maheswaran Sathiamoorthy, Yihua Chen, Rahul Mazumder, Lichan Hong, and Ed H. Chi. Dselect-k: Differentiable selection in the mixture of experts with applications to multi-task learning. arXiv preprint arXiv:2106.03760, 2021.

Ethan He, Abhinav Khattar, Ryan Prenger, Vijay Korthikanti, Zijie Yan, Tong Liu, Shiqing Fan, Ashwath Aithal, Mohammad Shoeybi, and Bryan Catanzaro. Upcycling large language models into mixture of experts. arXiv preprint arXiv:2410.07524, 2024.

Jiaao He, Jiezhong Qiu, Aohan Zeng, Zhilin Yang, Jidong Zhai, and Jie Tang. Fastmoe: A fast mixture-of-

expert training system. arXiv preprint arXiv:2103.13262, 2021. Xu Owen He. Mixture of a million experts. arXiv preprint arXiv:2407.04153, 2024. Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston. The goldilocks principle: Reading children’s

books with explicit memory representations. arXiv preprint arXiv:1511.02301, 2015. Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering, 2019. URL https://arxiv.org/abs/1902.09506. Tingfeng Hui, Zhenyu Zhang, Shuohuan Wang, Yu Sun, Hua Wu, and Sen Su. Upcycling instruction tuning from dense to mixture-of-experts via parameter merging. arXiv preprint arXiv:2410.01610, 2024.

Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, et al. Tutel: Adaptive mixture-of-experts at scale. Proceedings of Machine Learning and Systems, 5:269–287, 2023.

Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79–87, 1991.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Peng Jin, Bo Zhu, Li Yuan, and Shuicheng Yan. Moe++: Accelerating mixture-of-experts methods with zero-computation experts. arXiv preprint arXiv:2410.07348, 2024.

Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. Neural computation, 6(2):181–214, 1994.

- Hao Kang, Zichun Yu, and Chenyan Xiong. Flame-moe: A transparent end-to-end research platform for mixture-of-experts language models. arXiv preprint arXiv:2505.20225, 2025.

Aniruddha Kembhavi, Michael Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. ArXiv, abs/1603.07396, 2016. URL https://api.semanticscholar. org/CorpusID:2682274.

Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. Sparse upcycling: Training mixture-of-experts from dense checkpoints. arXiv preprint arXiv:2212.05055, 2022.

Taku Kudo and John Richardson. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226, 2018.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 785–794, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1082. URL https://aclanthology.org/D17-1082.

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020.

Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. Base layers: Simplifying training of large, sparse models. arXiv preprint arXiv:2103.16716, 2021.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a.

Jiachen Li, Xinyao Wang, Sijie Zhu, Chia-Wen Kuo, Lu Xu, Fan Chen, Jitesh Jain, Humphrey Shi, and Longyin Wen. Cumo: Scaling multimodal llm with co-upcycled mixture-of-experts. arXiv preprint arXiv:2405.05949, 2024b.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji rong Wen. Evaluating object hallucination in large vision-language models. In Conference on Empirical Methods in Natural Language Processing, 2023. URL https://api.semanticscholar.org/CorpusID:258740697.

Xinyu Lian, Sam Ade Jacobs, Lev Kurilenko, Masahiro Tanaka, Stas Bekman, Olatunji Ruwase, and Minjia Zhang. Universal checkpointing: Efficient and flexible checkpointing for large scale distributed training. ArXiv, abs/2406.18820, 2024. URL https://api.semanticscholar.org/CorpusID:270764954.

Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Jinfa Huang, Junwu Zhang, Yatian Pang, Munan Ning, et al. Moe-llava: Mixture of experts for large vision-language models. arXiv preprint arXiv:2401.15947, 2024a.

Xi Victoria Lin, Akshat Shrivastava, Liang Luo, Srinivasan Iyer, Mike Lewis, Gargi Ghosh, Luke Zettlemoyer, and Armen Aghajanyan. Moma: Efficient early-fusion pre-training with mixture of modality-aware experts. arXiv preprint arXiv:2407.21770, 2024b.

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024a.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng,

- Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024b.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng,

- Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024c.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023a. Yuanzhan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi

Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? ArXiv, abs/2307.06281, 2023b. URL https://api.semanticscholar.org/CorpusID: 259837088.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024d.

Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In Neural Information Processing Systems, 2019. URL https://api.semanticscholar.org/CorpusID:199453025.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chun yue Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, 2023. URL https: //api.semanticscholar.org/CorpusID:264491155.

Meta AI. The llama 4 herd: Turning llama 4 into ai’s universal multimodal intelligence. https://ai. meta.com/blog/llama-4-multimodal-intelligence/, 2025. Official blog post describing Llama 4’s MoE design.

Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, et al. Olmoe: Open mixture-of-experts language models. arXiv preprint arXiv:2409.02060, 2024.

Mohammed Muqeeth, Haokun Liu, and Colin Raffel. Soft merging of experts with adaptive routing. arXiv preprint arXiv:2306.03745, 2023.

Taishi Nakamura, Takuya Akiba, Kazuki Fujii, Yusuke Oda, Rio Yokota, and Jun Suzuki. Drop-upcycling: Training sparse mixture of experts with partial re-initialization. arXiv preprint arXiv:2502.19261, 2025.

Dung V Nguyen, Minh H Nguyen, Luc Q Nguyen, Rachel SY Teo, Tan M Nguyen, and Linh Duy Tran. Camex: Curvature-aware merging of experts. arXiv preprint arXiv:2502.18821, 2025a.

Huy Nguyen, Pedram Akbarian, Trang Pham, Trang Nguyen, Shujian Zhang, and Nhat Ho. Statistical advantages of perturbing cosine router in mixture of experts. arXiv preprint arXiv:2405.14131, 2024a.

Huy Nguyen, Nhat Ho, and Alessandro Rinaldo. On least squares estimation in softmax gating mixture of experts. arXiv preprint arXiv:2402.02952, 2024b.

Huy Nguyen, Nhat Ho, and Alessandro Rinaldo. Sigmoid gating is more sample efficient than softmax gating in mixture of experts. Advances in Neural Information Processing Systems, 37:118357–118388, 2024c.

Huy Nguyen, Thong T Doan, Quang Pham, Nghi DQ Bui, Nhat Ho, and Alessandro Rinaldo. On deepseekmoe: Statistical benefits of shared experts and normalized sigmoid gating. arXiv preprint arXiv:2505.10860, 2025b.

Nam V Nguyen, Huy Nguyen, Quang Pham, Van Nguyen, Savitha Ramasamy, and Nhat Ho. Competesmoe– statistically guaranteed mixture of experts training via competition. arXiv preprint arXiv:2505.13380, 2025c.

Stefan K Nielsen, Rachel SY Teo, Laziz U Abdullaev, and Tan M Nguyen. Tight clusters make specialized experts. arXiv preprint arXiv:2502.15315, 2025.

OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/2508.10925.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo,

Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2023.

Ashwinee Panda, Vatsal Baherwani, Zain Sarwar, Benjamin Therien, Supriyo Chakraborty, and Tom Goldstein. Dense backpropagation improves training for sparse mixture-of-experts. arXiv preprint arXiv:2504.12463, 2025.

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context. arXiv preprint arXiv:1606.06031, 2016.

Jungwoo Park, Young Jin Ahn, Kee-Eung Kim, and Jaewoo Kang. Monet: Mixture of monosemantic experts for transformers. arXiv preprint arXiv:2412.04139, 2024.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale. In International conference on machine learning, pp. 18332–18346. PMLR, 2022.

Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems, 34:8583–8595, 2021.

Stephen Roller, Sainbayar Sukhbaatar, Jason Weston, et al. Hash layers for large sparse models. Advances in Neural Information Processing Systems, 34:17555–17566, 2021.

Oleg Rybakov, Mike Chrzanowski, Peter Dykas, Jinze Xue, and Ben Lanir. Methods of improving llm training stability. ArXiv, abs/2410.16682, 2024. URL https://api.semanticscholar.org/CorpusID:273507181.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint

- arXiv:1701.06538, 2017a.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint

- arXiv:1701.06538, 2017b.

Sheng Shen, Zhewei Yao, Chunyuan Li, Trevor Darrell, Kurt Keutzer, and Yuxiong He. Scaling visionlanguage models with sparse mixture of experts. arXiv preprint arXiv:2303.07226, 2023.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8309–8318, 2019. URL https://api.semanticscholar.org/ CorpusID:85553602.

Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. https://cerebras.ai/blog/ slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama, June 2023. URL https://huggingface.co/datasets/cerebras/SlimPajama-627B.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937, 2018.

Qwen Team. Qwen1.5-moe: Matching 7b model performance with 1/3 activated parameters", February 2024. URL https://qwenlm.github.io/blog/qwen-moe/.

Rachel S Teo and Tan M Nguyen. Momentumsmoe: Integrating momentum into sparse mixture of experts. Advances in Neural Information Processing Systems, 37:28965–29000, 2024.

Rachel SY Teo and Tan M Nguyen. Molex: Mixture of layer experts for finetuning with sparse upcycling. arXiv preprint arXiv:2503.11144, 2025.

Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. Auxiliary-loss-free load balancing strategy for mixture-of-experts. arXiv preprint arXiv:2408.15664, 2024a.

Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. Auxiliary-loss-free load balancing strategy for mixture-of-experts, 2024b.

Ziteng Wang, Jun Zhu, and Jianfei Chen. Remoe: Fully differentiable mixture-of-experts with relu routing. arXiv preprint arXiv:2412.14711, 2024c.

Ziteng Wang, Jun Zhu, and Jianfei Chen. Remoe: Fully differentiable mixture-of-experts with relu routing. arXiv preprint arXiv:2412.14711, 2024d.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R Bowman. Blimp: The benchmark of linguistic minimal pairs for english. Transactions of the Association for Computational Linguistics, 8:377–392, 2020.

Tianwen Wei, Bo Zhu, Liang Zhao, Cheng Cheng, Biye Li, Weiwei Lü, Peng Cheng, Jianhao Zhang, Xiaoyu Zhang, Liang Zeng, Xiaokun Wang, Yutuan Ma, Rui Hu, Shuicheng Yan, Han Fang, and Yahui Zhou. Skywork-moe: A deep dive into training techniques for mixture-of-experts language models, 2024.

Shaohua Wu, Jiangang Luo, Xi Chen, Lingjun Li, Xudong Zhao, Tong Yu, Chao Wang, Yue Wang, Fei Wang, Weixu Qiao, Houbo He, Zeru Zhang, Zeyu Sun, Junxiong Mao, and Chong Shen. Yuan 2.0-m32: Mixture of experts with attention router. arXiv preprint arXiv:2405.17976, 2024a.

Shaohua Wu, Jiangang Luo, Xi Chen, Lingjun Li, Xudong Zhao, Tong Yu, Chao Wang, Yue Wang, Fei Wang, Weixu Qiao, et al. Yuan 2.0-m32: Mixture of experts with attention router. arXiv preprint arXiv:2405.17976, 2024b.

Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024c.

xAI. Open release of grok-1. https://x.ai/news/grok-os, 2024. 314B-parameter Mixture-of-Experts model; open weights and architecture.

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On layer normalization in the transformer architecture. ArXiv, abs/2002.04745, 2020. URL https://api.semanticscholar.org/CorpusID:211082816.

Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang You. Openmoe: An early effort on open mixture-of-experts language models. arXiv preprint arXiv:2402.01739, 2024.

Fanqi Yan, Huy Nguyen, Dung Le, Pedram Akbarian, and Nhat Ho. Understanding expert structures on minimax parameter estimation in contaminated mixture of experts. arXiv preprint arXiv:2410.12258,

- 2024.

Shen Yan, Xingyan Bin, Sijun Zhang, Yisen Wang, and Zhouchen Lin. Tc-moe: Augmenting mixture of experts with ternary expert choice. In The Thirteenth International Conference on Learning Representations,

- 2025.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Ke-Yang Chen, Kexin Yang, Mei Li, Min Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yunyang Wan, Yunfei Chu, Zeyu Cui, Zhenru Zhang, and Zhi-Wei Fan. Qwen2 technical report. ArXiv, abs/2407.10671, 2024a. URL https://api.semanticscholar.org/CorpusID:271212307.

Longrong Yang, Dong Shen, Chaoxiang Cai, Fan Yang, Tingting Gao, Di Zhang, and Xi Li. Solving token gradient conflict in mixture-of-experts for large vision-language model. arXiv preprint arXiv:2406.19905, 2024b.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. ArXiv, abs/2311.16502, 2023. URL https://api.semanticscholar.org/CorpusID:265466525.

Sukwon Yun, Inyoung Choi, Jie Peng, Yangfan Wu, Jingxuan Bao, Qiyiwen Zhang, Jiayi Xin, Qi Long, and Tianlong Chen. Flex-moe: Modeling arbitrary modality combination via the flexible mixture-of-experts. Advances in Neural Information Processing Systems, 37:98782–98805, 2024.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Zihao Zeng, Yibo Miao, Hongcheng Gao, Hao Zhang, and Zhijie Deng. Adamoe: Token-adaptive routing with null experts for mixture-of-experts language models. arXiv preprint arXiv:2406.13233, 2024.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11975– 11986, 2023.

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024a. URL https://arxiv.org/abs/2407.12772.

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024b.

Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge highresolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257, 2024c.

Jinze Zhao, Peihao Wang, and Zhangyang Wang. Generalization error analysis for sparse mixture-of-experts: A preliminary study. arXiv preprint arXiv:2403.17404, 2024.

Zexuan Zhong, Mengzhou Xia, Danqi Chen, and Mike Lewis. Lory: Fully differentiable mixture-of-experts for autoregressive language model pre-training. arXiv preprint arXiv:2405.03133, 2024.

Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew Dai, Zhifeng Chen, Quoc Le, and James Laudon. Mixture-of-experts with expert choice routing. arXiv preprint arXiv:2202.09368, 2022.

Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. St-moe: Designing stable and transferable sparse expert models. arXiv preprint arXiv:2202.08906, 2022.

Simiao Zuo, Xiaodong Liu, Jian Jiao, Young Jin Kim, Hany Hassan, Ruofei Zhang, Tuo Zhao, and Jianfeng Gao. Taming sparsely activated transformer with stochastic experts. arXiv preprint arXiv:2110.04260, 2021.

### Supplement to “LibMoE: A Library for Comprehensive Research Mixture of Experts in Large Language Models”

#### A Experiment Settings

##### A.1 Vision Language Model

###### General (26.6%)

Image Textualization (Filt.) (24.9 K) TallyQA (Cauldron/LLaVA) (24.7 K) CLEVR (Cauldron/LLaVA) (17.5 K) ShareGPT 4o (14.3 K) ShareGPT 4V (COCO) (12.5 K) ALLaVA Instr. (LAION 4V) (12.5 K) LLaVA Wild 39K (9.8 K) ShareGPT 4V (LLaVA) (7.5 K) ALLaVA Instr. (V FLAN 4V) (5.0 K) LLaVAR (GPT 4) (4.9 K) ST VQA (Cauldron/LLaVA) (4.3 K) AOK VQA (Cauldron/LLaVA) (4.1 K) LLaVA Wild 12K (3.8 K) Visual7W (Cauldron/LLaVA) (3.6 K) VisText (Cauldron) (2.5 K) Hateful Memes (Cauldron/LLaVA) (2.1 K) ScienceQA (Cauldron/LLaVA) (1.2 K) VSR (Cauldron/LLaVA) (538) InterGPS (Cauldron/LLaVA) (318) VQARAD (Cauldron/LLaVA) (77)

###### Doc/Chart/Screen (28.3%)

GeneralOCR17.6%

DVQA (Cauldron/LLaVA) (50.0 K) FigureQA (Cauldron/LLaVA) (25.0 K) UReader Cap (22.9 K) RoBUT WikiSQL (Cauldron) (18.7 K) RoBUT WTQ (Cauldron/LLaVA) (9.6 K) TQA (Cauldron/LLaVA) (6.8 K) Chart2Text (Cauldron) (6.7 K) ChartQA (Cauldron/LLaVA) (4.6 K) UReader IE (4.3 K) Screen2Words (Cauldron) (3.9 K) AI2D (InternVL) (3.1 K) LRV Normal (Filt.) (2.6 K) RoBUT SQA (Cauldron) (2.1 K) AI2D (GPT 4V) (1.2 K) Infographic VQA (1.1 K) VisualMRC (Cauldron) (755) HiTab (Cauldron/LLaVA) (623) AI2D (Cauldron/LLaVA) (607) Infographic VQA (LLaVA) (528) Infographic (GPT 4V) (495) LRV Chart (444) Diagram2Text (Cauldron) (73)

General 26.6%

One-Vision 0.6 M

###### Math/Reasoning (27.5%)

27.5%Math/Reasoning

MAVIS RuleGeo (25.0 K) MAVIS MetaGen (21.8 K) Geo170K QA (17.0 K) Geo170K Align (15.1 K) Raven (Cauldron) (10.5 K) MapQA (Cauldron/LLaVA) (9.4 K) PMC VQA (MathV360K) (9.0 K) MathQA (7.5 K) IconQA (Cauldron/LLaVA) (6.8 K) IconQA (MathV360K) (5.6 K) TabMWP (MathV360K) (5.6 K) FigureQA (MathV360K) (4.4 K) GeoQA+ (MathV360K) (4.3 K) UniGeo (MathV360K) (3.0 K) TQA (MathV360K) (2.5 K) Geometry3K (MathV360K) (2.4 K) GeoMVerse (Cauldron) (2.3 K) Super CLEVR (MathV360K) (2.2 K) VizWiz (MathV360K) (1.7 K) VQA AS (MathV360K) (1.5 K) CLEVR Math (MathV360K) (1.3 K) MapQA (MathV360K) (1.3 K) VQA RAD (MathV360K) (531) Geo3K (522) GEOS (MathV360K) (124)

Doc/Chart/Screen 28.3%

###### General OCR (17.6%)

K 12 Printing (64.2 K) HME 100K (18.6 K) TextOCR (GPT 4V) (6.3 K) TextCaps (5.5 K) Rendered Text (Cauldron) (2.5 K) WebSight (Cauldron) (2.5 K) Chrome Writing (2.2 K) IAM (Cauldron) (1.4 K) IIIT 5K (497)

- Figure 11: Visualization of dataset subsets from the OneVision dataset in LibMoE.

General60.8%

23.8 %Math/Reasoning

GeneralOCR15.3%

LLaVA-665K 0.7 M

General (60.8 %)

LLaVA (157.7 K) VQA-v2 (82.8 K) A-OKVQA (66.2 K) RefCOCO (48.4 K) SG40k (40.7 K) OKVQA (9.0 K)

Math/Reasoning (23.8 %)

VG (86.4 K) GQA (72.1 K)

General OCR (15.3 %)

OCRVQA (80.0 K) TextCaps (22.0 K)

- Figure 12: Visualization of dataset subsets from the LLAVA-665K dataset in LibMoE.

Datasets. We adopt the vision-language pretraining task (Lu et al., 2019) and follow the CUMO framework (Li et al., 2024b) to upcycle the LLaVA model (Liu et al., 2023a), enabling systematic evaluation of various SMoE algorithms. In the initial dense training stage, we first initialize the MLP connector using the LLaVA-558K dataset (Liu et al., 2023a), then jointly train all three model components (image encoder, language model, and connector) on the ALLaVA dataset (Chen et al., 2024a). This yields a dense checkpoint that serves as the starting point for sparse upcycling.

In the subsequent SMoE training stage, we consider two training settings:

- • LLaVA-665K (Liu et al., 2023a), a standard and widely used benchmark in the community;
- • A hybrid dataset with 1M2 samples, constructed by combining LLaVA-665K with OneVision (Li et al., 2024a), where we uniformly sample 25% from each OneVision sub-benchmark to ensure broad domain coverage.

We visualize the data distribution across categories for both datasets in Figure 12 and Figure 11. All SMoE algorithms are trained using the same datasets and configurations to ensure fair benchmarking. For further details on dataset construction and training stage objectives, we refer readers to Liu et al. (2023a); Li et al. (2024b).

Evaluation Benchmarks and Metrics LibMoE employs a set of popular benchmarks for vision-language models, including AI2D (Kembhavi et al., 2016), Text VQA Validation (Singh et al., 2019), GQA (Hudson & Manning, 2019), Hallusion Benchmark (Guan et al., 2023), MathVista (Lu et al., 2023), MMBenchEN (Liu et al., 2023b), MME (Fu et al., 2023), MMMU Validation(Yue et al., 2023), MMStar (Chen et al., 2024b), POPE (Li et al., 2023), MME-Real World(Zhang et al., 2024c) and OCR Bench(Liu et al., 2024d). We carefully choose these benchmark to assert the model across several vision-language capabilities such as perception, reasoning, OCR, instruction following, and more. Beyond the performance on standard benchmarks, we analyze the algorithms holistically by evaluating their generalization throughout training, expert selection behaviors, and expert specialization, which we will report in Section 5. For benchmarks requiring GPT-based evaluation, such as MathVista and HallusionBench, we use GPT-4o-mini,version 2024-07-18.

Model Architecture. We adopt Phi-3.5 Mini (Abdin et al., 2024) as the language model and SigLIPSO400M (Zhai et al., 2023) as the vision encoder two widely used backbones in recent multimodal systems. During the VIT stage, we upcycle the dense MLP blocks in both the vision encoder and connector into sparse MoE layers, each composed of NE=6 experts with top-K=3 routing.

Additionally, Table 3 reports the parameter breakdown across key components. The full model contains approximately 5.67 billion parameters, including a 3.82B LLM backbone, a 1.75B vision encoder equipped with MoE layers, and 99M parameters dedicated to expert MLPs. To ensure fair comparison across routing methods, all models are configured under the same total parameter budget.

Table 3: Parameter counts for each major component. Total model size is 5.67B.

Component Parameters Vision Encoder - MoE 1.75B MLP Connector - MoE 0.099B LLM 3.82B Total 5.67B

##### A.2 Language Model Pretrain

Dataset. We conduct our experiments on the widely-used SlimPajama dataset (Soboleva et al., 2023), a high-quality corpus curated from RedPajama (Computer, 2023), specifically designed for training large language models (LLMs). SlimPajama has become a standard choice for open LLM research and has been employed in the development of several influential models, including TinyLlama (Zhang et al., 2024b) and BTLM (Dey et al., 2023), as well as in a range of recent empirical studies (Agarwalla et al., 2024; Gupta et al., 2023). Its adoption across diverse works underscores its value for benchmarking and advancing LLM research.

Tokenizer and Model Architecture. We adopt the SentencePiece tokenizer (Kudo & Richardson, 2018) with byte-pair encoding (BPE), which provides a balance between subword granularity and vocabulary

efficiency. generally follow Switch Transformer (Fedus et al., 2022) for our model architecture, adopting a standard Transformer backbone with sparsely activated Mixture-of-Experts (MoE) layers. Each MoE layer replaces the conventional feedforward sublayer and comprises N expert networks (FFNs) and a router mechanism. Table 6 summarizes the comprehensive set of hyperparameters and configurations for both scales and different model variants evaluated in our experiments.

Load Balancing Loss and Router Z-Loss. We adopt the standard load balancing loss from Switch Transformer (Fedus et al., 2022) to penalize imbalances in routing decisions, thereby encouraging more uniform expert utilization. This auxiliary term has been shown to mitigate expert under-utilization and promote balanced load distribution, which improves both convergence and generalization in MoE models. In contrast, we do not incorporate the router z-loss in our experiments (Zoph et al., 2022). Following common practice, we set the weight of the load balancing loss to α = 0.01. Additional experimental details and design considerations for the auxiliary load balancing loss are provided in Appendix D.4.

Evaluation pipeline. We evaluate our implemented models with the Perplexity score (PPL) and zero-shot performance with nine different downstream tasks: LAMBADA (Paperno et al., 2016), BLiMP (Warstadt et al., 2020), Children’s Book Test (Hill et al., 2015), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), ARC-Easy (Clark et al., 2018), RACE (Lai et al., 2017), SIQA (Sap et al., 2019) and CommonSenseQA (Talmor et al., 2018). For LAMBADA, we use the detokenized version from OpenAI, and we evaluate the top-1 accuracy of the last word (it can span multiple tokens; here we use greedy decoding). For CBT, BLiMP, and RACE, we measure the accuracy of each task and report the average accuracy of the tasks.

#### B Hyperparameter Setting

##### B.1 Vision-Language Model

- Table 4: Hyperparameter settings across the three training stages of Phi-3.5 Mini. MoE routing is only applied during the final VIT phase.

Hyperparameter PT PFT VIT Learning rate 1e-3 2e-6 4e-6 Learning rate schedule Cosine Cosine Cosine Batch size per GPU 64 6 4 GPUs 4×H100 4×H100 4×H100 ZeRO optimization ZeRO-2 ZeRO-2 ZeRO-3 Optimizer AdamW AdamW AdamW MLP parameters Trained Trained Trained Vision encoder Frozen Trained Trained Language model Frozen Trained Trained MoE blocks No No Yes Balance loss coefficient 0.0 0.0 0.01 Z-loss coefficient 0.0 0.0 0.001 Maximum tokens 2048 2048 2048

Hyperparameter Settings and Training Stages. Table 4 summarizes the key hyperparameter configurations across the three sequential training stages of Phi-3.5 Mini: Pretraining (PT), Pre-FineTuning (PFT), and Visual Instruction Tuning (VIT). All stages are trained on 4×H100 GPUs with consistent token lengths, optimizers, and learning rate schedules to ensure comparability. The Mixture-of-Experts (MoE) routing mechanism is activated only during the VIT stage, where multimodal specialization is required. Following the initialization strategy used in the official GPT-2 implementation,1 router parameters are sampled from a normal distribution N(0,0.02) using a fixed random seed of 42 to ensure reproducibility. All expert parameters are trained during VIT, while only the router is initialized from scratch. This design yields a controlled and fair evaluation protocol across different sparse routing strategies.

- Table 5: MoE configuration details across different methods, where N denotes the total number of experts, K the number of routed (active) experts per token, and Ns the number of shared experts.

MoE Method N K Ns

SMoE 6 3 0 SharedE 6 2 1 MoE++ 8 3 0 TC-MoE 15 3 0

- MoE Architecture Configurations. Table 5 summarizes the MoE configurations used across different methods, where N denotes the total number of experts, K the number of routed (active) experts per token,

1https://github.com/openai/gpt-2

and Ns the number of shared experts. For standard routing-based methods, including SMoE, σ-MoE, and XMoE, we adopt a common configuration with N = 6 experts and top-K = 3 routing. For shared-expert architectures, we follow the VLM setting with one shared expert (Ns = 1) and reduce the number of routed experts to K = 2 to maintain comparable computational cost. For MoE++, we follow Jin et al. (2024) and augment the expert set by introducing zero-computation experts, resulting in a total of N = 8 experts while keeping K = 3. For TC-MoE, we follow Yan et al. (2025), where the effective expert pool is expanded via ternary compositions, yielding a total of 2N + K experts; in our configuration, this corresponds to N = 15 with K = 3.

##### B.2 Language Modeling

- Table 6: Comprehensive Model Configurations for Pre-train LLM. SMoE refers to settings applied for Vanilla SMoE, σ-MoE and XMoE, whereas SharedE corresponds to configurations used for SharedE-V2 and SharedEV3 models.

Scale Model # params

# act. params

# trained tokens dmodel H dhead N K Ns

Expert dim

Small

SMoE

0.15B 36M 6.5B 512 12 82

66 8 0

128

SharedE 66 6 2 MoE++ 66 + 8 8 0 TC-MoE 66 * 2 + 8 8 0

Large

SMoE

0.68B 131M 26.2B 1024 16 128

66 8 0

256

SharedE 66 6 2 MoE++ 66 + 8 8 0 TC-MoE 66 * 2 + 8 8 0

Hyperparameter Settings and Training Stages. Table 6 summarizes the key hyperparameters, covering model dimensionality, number of attention heads, expert counts, and routing strategies. The small-scale setting processes 6.5B tokens with model dim dmodel = 512 and number of attention heads H = 12, while the large-scale setting extends to 26.2B tokens with dmodel = 1024 and H = 16. Each variant differs in expert number (N), routing capacity (K), and warmup strategies, with expert dimensions set to 128 for small and 256 for large models. These standardized yet diverse configurations ensure a balanced comparison across MoE algorithms while reflecting realistic large-scale training regimes.

- Table 7 details the training hyperparameters, where both scales are optimized with AdamW using a cosine learning rate schedule, gradient clipping, and mixed-precision training on 4×H100 GPUs. While the smallscale models are trained for 100k steps on 6.5B tokens, large-scale models extend to 400k steps on 26.2B tokens, with warm-up and stronger gradient clipping applied to ensure stability. Together, these settings provide a consistent yet scalable foundation for benchmarking diverse MoE algorithms.

- MoE Architecture Configurations. Table 6 also specifies the MoE architecture configurations used in the language modeling pre-training setting, where N denotes the total number of experts, K the number of

routed (active) experts per token, and Ns the number of shared experts. For standard routing-based methods (SMoE, σ-MoE, and XMoE), we adopt a common configuration with N = 66 experts and top-K = 8 routing. For shared-expert variants (SharedE-V2/V3), we set Ns = 2 shared experts and reduce routed capacity to K = 6 to maintain comparable compute. For MoE++, following Jin et al. (2024), we augment the expert pool by adding zero-computation experts, resulting in N = 66 + 8 while keeping K = 8. For TC-MoE, following Yan et al. (2025), the effective expert pool is expanded via ternary compositions, yielding 2N + K experts; under our configuration, this corresponds to N = 66 × 2 + 8 with K = 8. Across both model scales, we keep the expert dimension fixed (128 for 0.15B and 256 for 0.68B) to ensure a controlled comparison across MoE designs.

- Table 7: Training hyperparameter settings for LibMoE across two model scales (0.15B and 0.68B parameters) in the language modeling task.

Hyperparameter 0.15B 0.68B Learning rate 2.5e-4 2.5e-4 LR schedule Cosine Cosine Nwarmup 0 4000

Min LR multiplier 0.1 0.1 Optimizer AdamW AdamW Weight decay 0.01 0.01 Gradient clip (κ) 0.1 0.25 Dropout No No Batch size (per device) 16 16 Total batch size 64 64 Sequence length 1024 1024 Training steps 100k 400k Validation ratio 0.5% 0.5% Precision AMP (fp16) AMP (fp16) GPUs 4 × H100 4 × H100

#### C Comparison Between Dense and SMoE Models

SMoE Dense

###### Training Loss

###### Validation Loss

###### PPL

###### Avg Acc

25

3.25

6

44

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

158M679M

20

3.00

42

4

Training Steps (k) Performance

2.75

15

40

2

0 50000 100000

25 50 75 100

25 50 75 100

25 50 75 100

50

6

2.75

15

4

45

2.50

10

2

0 200000 400000

200 400

200 400

200 400

- Figure 13: Training benchmark curves comparing Dense and SMoE models during language model pretraining (0.67B parameters).

SMoE SMoE SG X-MoE MoE++ SharedE-V2 SharedE-V3 TCMoE

RouterSaturation(%)

###### Top-1

###### Top-8

###### Top-1

Top-8

95

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

90

95

80

| |
|---|

| |
|---|

90

| |
|---|

| |
|---|

| |
|---|

80

| |
|---|

| |
|---|

| |
|---|

| |
|---|

90

| |
|---|

| |
|---|

| |
|---|

| |
|---|

60

85

| |
|---|

| |
|---|

| |
|---|

70

| |
|---|

| |
|---|

| |
|---|

| |
|---|

85

| |
|---|

| |
|---|

80

| |
|---|

60

40

| |
|---|

| |
|---|

| |
|---|

20 40 60 80

20 40 60 80

20 40 60 80

20 40 60 80

Validation Heallaswag

Training Steps (k)

- Figure 14: Router Saturation across methods during training on the language modeling task. We present both Top-1 and Top-8 routing results for small (0.15B) and large (0.68B) models, illustrating the progression of router convergence across model scales and expert selection strategies.

- Table 8: Performance comparison between dense and SMoE models for language model pre-training, evaluated on a small-scale model (0.15B parameters) and a large-scale model (0.68B parameters). PPL denotes perplexity; lower values indicate better performance.

Hella Swag

ARCEasy

Common SenseQA

MoE Method PPL ↓ LAMBADA BLiMP CBT

PIQA

RACE SIQA

AVG ↑ Small Model (0.15B)

Dense (36M) 17.04 19.74 73.48 81.03 27.56 55.82 32.09 29.36 35.72 25.14 42.22 SMoE 13.63 25.27 77.71 84.18 29.43 57.94 32.68 30.11 35.62 24.65 44.18

Dense (131M) 11.51 31.00 77.43 87.45 31.90 61.37 35.14 31.58 36.90 27.03 46.64 SMoE 9.51 37.13 80.47 89.83 37.49 64.36 38.22 33.03 37.41 26.54 49.39

Large Model (0.68B)

Figure 13 and Table 8 presents a comparative analysis between dense baselines and SMoE models in the language model pre-training setting. Across all reported metrics, SMoE models consistently achieve stronger performance during the early stages of training, indicating faster optimization and more efficient utilization of model capacity compared to dense counterparts.

#### D Additional Analysis

##### D.1 Router Saturation

Router Saturation - first introduced in OLMoE (Muennighoff et al., 2024) - quantifies the proportion of overlapping activated experts between an intermediate checkpoint at training step t and the final checkpoint. This metric serves as an indicator of the router’s convergence dynamics throughout training. Higher router saturation values indicate greater alignment in expert selection, signifying that the router’s decisions are becoming increasingly consistent with its final checkpoint. Consequently, router saturation provides insight into the convergence of expert assignments of routing strategies during training. The formal definition and formula are defined in the Appendix E.4.

As shown in Figure 14, router saturation for all evaluated methods rises sharply during training, with most surpassing 60%, and even reaching over 85% on the HellaSwag benchmark, within the first 10% of training under top-8 selection. In contrast, top-1 selection exhibits slightly slower convergence, indicating a more gradual stabilization of expert assignments. Notably, XMoE stands out as an outlier, converging more slowly than other variants—a trend consistent with its relatively lower performance reported in Table 2. Overall, this early stabilization behavior aligns with prior work (Muennighoff et al., 2024; Xue et al., 2024; Nguyen et al., 2025b; Kang et al., 2025), suggesting a general tendency for MoE routers to converge rapidly toward stable expert assignments. Such early convergence is also consistent with the learning strategy advocated by Stable MoE (Dai et al., 2022b), which aims to mitigate fluctuations in expert allocation during training.

- Table 9: Performance variation under different router temperatures across MoE algorithms. We evaluate the impact of router temperature (τ) on the behavior of expert cooperation and competition in both small (0.15B) and large (0.68B) models, using two language understanding tasks (BLiMP and HellaSwag).

τ Task SMoE SMoE SG XMoE SharedE-V2 SharedE-V3 TC-MoE MoE++ AVG ∆

BLIMP 77.71% 76.75% 76.53% 77.37% 77.20% 76.91% 77.23%

1.0

HellaSwag 29.43% 29.15% 29.34% 29.38% 29.38% 29.27% 29.28% 10.0

Small Model (0.15B)

BLIMP 69.64%↓8.07 68.52%↓8.24 76.23%↓0.30 56.87%↓20.5 55.41%↓21.8 63.73%↓13.2 67.98%↓9.25 ↓11.6% HellaSwag 29.55%↑0.12 29.41%↑0.26 29.31%↓0.03 28.75%↓0.63 28.79%↓0.59 29.42%↑0.15 29.40%↑0.12 ↓0.09%

BLIMP 63.89%↓13.8 51.85%↓24.9 76.47%↓0.06 70.30%↓7.07 69.65%↓7.56 63.57%↓13.3 64.14%↓13.1 ↓11.4% HellaSwag 27.82%↓1.60 25.03%↓4.11 28.69%↓0.65 28.91%↓0.47 29.22%↓0.16 27.41%↓1.85 27.66%↓1.61 ↓1.49%

0.1

BLIMP 80.47% 81.08% 80.38% 80.98% 81.28% 81.21% 80.88%

1.0

HellaSwag 37.49% 37.52% 37.19% 37.14% 37.32% 37.95% 37.70% 10.0

Large Model (0.68B)

BLIMP 73.96%↓6.51 76.56%↓4.52 80.52%↑0.13 75.05%↓5.93 76.45%↓4.83 66.34%↓14.9 73.37%↓7.51 ↓6.29% HellaSwag 34.37%↓3.12 35.30%↓2.22 36.72%↓0.48 33.44%↓3.70 34.92%↓2.40 32.12%↓5.84 33.98%↓3.72 ↓3.07%

BLIMP 67.94%↓12.5 71.24%↓9.84 79.13%↓1.25 71.69%↓9.29 74.78%↓6.50 66.33%↓14.9 68.03%↓12.9 ↓9.59% HellaSwag 32.57%↓4.92 35.42%↓2.10 36.44%↓0.76 33.72%↓3.42 35.18%↓2.14 32.52%↓5.43 33.17%↓4.53 ↓3.33%

0.1

##### D.2 Cooperation or Competition

To characterize expert behavior when contributing to the creation of outcomes, we analyze the performance change that occurs when we adjust the router’s temperature. The high router’s temperature leads to the expert weights being more peaky, indicating more competition between experts. On the other hand, low temperature makes experts’ weights more uniform, indicating cooperation between experts. To be more specific, with the logit of expert k is sk, our router has been adjusted as:

 

exp(sk/τ) NE j=1 exp(sj/τ)

, if softmax router σ(sk/τ), if sigmoid router

g(sk) =

(2)



- Table 9 summarizes the effects of varying the router’s temperature parameter on model performance for both small (0.15B) and large (0.68B) model sizes across two representative language understanding tasks. We observe that deviating from the original temperature used during training generally results in performance degradation across both small and large model scales. Notably, applying a low temperature (τ = 0.1) induces only marginal performance loss, and in several cases - such as with MoE++ and XMoE - yields slight improvements. In contrast, setting a high temperature (τ = 10.0) consistently leads to substantial performance drops for all evaluated models, indicating that increased competition among experts impairs the model’s ability to integrate expert knowledge effectively. Interestingly, XMoE demonstrates greater robustness to temperature changes compared to other variants.

Overall, these findings suggest that most existing MoE architectures inherently favor cooperative expert behavior, with performance being more resilient to enhanced cooperation (lower temperature) than to heightened competition (higher temperature). We leave a deeper investigation of the underlying mechanisms driving this cooperation-competition trade-off to future work.

##### D.3 Expert Co-Activation Over Time

To further examine expert interaction dynamics, we analyze the evolution of expert co-activation (ECA) patterns throughout training. Following Muennighoff et al. (2024), ECA measures how frequently two experts are activated together, normalized by the total number of activations of one expert (see Appendix E.5 for the formal definition). Higher ECA values indicate persistent co-utilization between expert pairs.

- Figure 15 shows the ECA matrices for Layer 4 and Layer 12 of a small (0.15B) Vanilla SMoE model at multiple training checkpoints. Across training, the identity of the most strongly co-activated expert pairs remains largely unchanged, with only modest fluctuations in co-activation strength. This consistency suggests

[Figure 23]

21 39 59 47 58 34

Layer4Layer12

- 16 55 38 54 49 27

- 6

- 7

31

10K

[Figure 24]

39 16 18 58 59 47 49 34 55 27 54 38 6 7 31

39 16 18

- 58

- 59 47 49 34 55 27 54 38

- 6

- 7

31

40K

[Figure 25]

34 20 42 26 49 55 59 18 47 6 58 45 54 7 31

34 20 42 26 49 55 59 18 47

- 6

58 45 54

- 7

31

60K

[Figure 26]

42 34 20 18 6 26 49 59 55 58 47 45 7 54 31

42 34 20 18

- 6

26 49 59 55 58 47 45

- 7

54 31

80K

[Figure 27]

20 34 42 18 55 49 6 26 58 59 47 45 7 54 31

20 34 42 18 55 49

- 6

26

- 58

- 59 47 45

- 7

54 31

100K

[Figure 28]

43 47 24 54 49 53 30 21 38 55 5 17 39 57 33

43 47 24

- 54 49 53 30 21

- 38

55

5 17

- 39 57 33

21 39 59 47 58 34 16 55 38 54 49 27 6 7 31

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

- 42

- 43 49 24 47

2 43 49

22 49 43

22 43 49

6 47

6 47 42 53 26

6 47

- 38

- 39 21

5 42 26 33 53 55 30 17 39 57

5 53 42 26 33 55 30 39 17 57

5 17 30 53 57 55 33

5 39 55 33 30 17 57

42 43 49 24 47 38 39 21 5 17 30 53 57 55 33

2 43 49 6 47 5 42 26 33 53 55 30 17 39 57

22 49 43 6 47 42 53 26 5 39 55 33 30 17 57

22 43 49 6 47 5 53 42 26 33 55 30 39 17 57

Training Steps

- Figure 15: Expert co-activation (ECA) matrices for a Vanilla SMoE (0.15B) on the language modeling task, shown at Layers 4 and 12 across training checkpoints (10K–100K steps) on the validation set.

that expert collaboration structures emerge early in training and remain stable thereafter, indicating limited reorganization of expert interactions during later optimization.

We observe similar stability in expert co-activation patterns across other SMoE variants, including XMoE, SMoE-Sigmoid, DeepSeek-V2, DeepSeek-V3, TCMoE, and MoE++. Corresponding ECA visualizations are provided in Figure 20, Figure 21, Figure 22, Figure 25, Figure 24, and Figure 23. Collectively, these results indicate that, despite differences in routing mechanisms and architectural refinements, expert co-activation relationships in SMoE-style models are remarkably stable over training.

D.4 Load Balancing Loss and Router Z-loss Experiment

0 50000 100000

- 3

- 4

- 5

Training Loss

25 50 75 100

2.7

2.8

2.9

3.0

3.1

Validation Loss

25 50 75 100

14

16

18

20

PPL

25 50 75 100

- 40

- 41

- 42

- 43

Avg Acc

Training Steps (k)

Performance

SMoE SMoE No Load Balancing

- Figure 16: Benchmark curves during training in language modeling tasks for models with 0.15B parameters with and without load balancing loss.

- Table 10: Performance comparisons in the impact of load balancing loss and router z-loss of small model (0.15B parameters) in language modeling task.

LAM BADA

Hella Swag

ARCChallenge

Common SenseQA

Average SMoE (0.01 lb) 13.63 25.27% 77.71% 84.18% 29.43% 57.94% 21.20% 30.11% 35.62% 24.65% 42.90% SMoE (no-lb) 13.69 24.90% 76.81% 84.13% 29.38% 57.51% 21.63% 30.26% 35.67% 25.63% 42.88% SMoE (0.01 lb + 0.001 z-loss) 13.62 25.49% 77.55% 84.23% 29.11% 58.71% 21.89% 29.63% 35.21% 23.91% 42.86%

MoE Method PPL ↓

BLiMP CBT

PIQA

RACE SIQA

- Figure 16 and Figure 17 illustrate the effect of the load balancing loss on both training dynamics and expert assignment in Mixture-of-Experts models.The results demonstrate that incorporating load balancing loss

###### Layer 0

###### Layer 5

###### Layer 10

###### Layer 15

10

WithLBNoLB

%oftokensassigned

5

5

5

0

0

0

0

20 40 60

20 40 60

20 40 60

20 40 60

10

5

5

5

0

0

0

0

20 40 60

20 40 60

20 40 60

20 40 60

Expert_Ids

- Figure 17: Expert selection ratio of SMoE model with and without load balancing loss in language modeling tasks (0.15B parameters model size).

contributes to more stable training and improved overall performance. Specifically, Figure 17 highlights how the load balancing loss promotes a more uniform distribution of tokens across experts, mitigating the risk of expert under-utilization.

Beyond load balancing, (Zoph et al., 2022) introduced the router z-loss as an additional regularization strategy to further stabilize MoE training. To assess the practical impact of these auxiliary losses, we report in Table 10 a comparison of model performance under different loss configurations. Our results indicate that applying the router z-loss does not yield a performance improvement; in fact, it slightly degrades accuracy relative to using only the load balancing loss. Therefore, we opt not to include the router z-loss in our final experimental setup.

#### E Definitions and Formulations of Analysis Metrics

##### E.1 Total Score Change under Routing Perturbations: DropTop1 vs. DropTop1&2

Let D = {Di}|D|i=1 denote the set of evaluation benchmarks. For a given MoE model, let Mi ∈ R be its original evaluation score (e.g., accuracy; higher is better) on benchmark Di. After applying a routing perturbation π ∈ {DropTop1, DropTop1&2} at inference time, we obtain the perturbed score Mˆiπ ∈ R.

We define the aggregate performance change (reported as ∆ Total Score in Figure 4) as:

|D|

∆TotalScoreπ =

i=1

M ˆiπ − Mi . (3)

Negative values indicate that the routing perturbation leads to a net performance degradation relative to the original routing configuration, whereas positive values indicate that the perturbed routing achieves higher aggregate performance than the original model. In the latter case, the improvement suggests that the original routing decision may have been suboptimal, and that alternative expert assignments can better exploit the model’s representational capacity.

##### E.2 Expert Entropy Allocation

We investigate the behaviors of the expert selection mechanism by exploring how often each expert is selected in the MME benchmark. To this end, we analyze the frequency of the expert selection across different subtasks to gain insights into the specialization behavior of each expert. Given an MoE algorithm

with N experts and L layers, the selection frequency of each expert i at a given layer l is denoted as freq(il) (i = 1,2,...,N and l = 1,2,...,L). Note that this selection frequency is counted across all samples in

the benchmark, in this case we choose to be MME. Then, the entropy H(l) at each layer l is calculated by integrating the probability of selecting expert i into Shannon’s entropy formula as follows:

(l) i

freq(il) N j=1 freq(jl)

log2 freq

− Ni=1

N j=1 freq(jl)

EAE(l) =

, (4)

log2(N)

where:

- • freq(il): The number of times expert i is selected at layer l.
- • N: The total number of experts in the MoE algorithms.
- • Nj=1 freq(jl): The total number of expert selections at layer l.
- • EAE(l): The entropy value at layer l, measuring the uncertainty or diversity in expert selections.

With EAE, we can measure the frequency of expert selections across all layers HEAE. Importantly, high EAE Indicates a balanced expert utilization across all layers, where the model tends to distribute selections evenly among all experts. In contrast, low EAE suggests a concentrated usage of a few experts in most layers, indicating specialization or a preference for certain experts.

##### E.3 Router Margin

Router Margin is a metric that quantifies the dominance of the highest-scoring expert in a Mixture-of-Experts (MoE) routing decision. It is defined as the difference between the top-1 and top-2 gating scores:

Where:

1 N

Router Margin(l) =

N

top1(xi) − top2(xi),

i=1

- • RouterMargin(l): Router margin at layer l
- • N: The total number of tokens in the dataset.
- • xi: i-th input token
- • top1(xi): top-1 routing score of input xi
- • top2(xi): top-2 routing score of input xi

Router Margin provides a quantitative measure of how decisively the router selects the top expert relative to alternatives. A larger margin indicates that the router strongly favors the top-1 expert, reflecting more confident and specialized routing, whereas a smaller margin implies greater ambiguity and potential overlap among experts. This metric is thus a valuable diagnostic tool for understanding the dynamics of expert dominance and the evolution of routing confidence during training.

##### E.4 Router Saturation

In formal terms, router saturation is the proportion of expert activations at some intermediary checkpoint at time t that matches the expert IDs activated at some final checkpoint T over the same dataset:

1 N

RouterSaturation(t) =

Ei(t) ∩ Ei(T) k

N

, (5)

i=1

Where:

- • N: The total number of tokens in the dataset.
- • k: The number of top-k experts activated per input token.
- • Ei(t): The set of k experts activated for the i-th token at the t-th checkpoint.
- • Ei(T): The set of k experts activated for the i-th token at the final checkpoint T.
- • Ei(t) ∩ Ei(T) : The number of common experts activated for the i-th token between the t-th and final checkpoints T.

Router saturation provides a quantitative measure of how early the routing decisions converge during training. A saturation value of 100% indicates that the router at an intermediate checkpoint routes to the same set of experts as at the final checkpoint. High saturation values at early checkpoints reflect early convergence in expert selection, indicating that the router has rapidly settled into a stable assignment pattern. In contrast, low saturation values suggest ongoing exploration or adaptation in expert allocations, signaling that the routing mechanism is still undergoing significant adjustments.

##### E.5 Experts Co-activation

We define expert co-activation as the proportion of times two specific experts, Ei and Ej, are simultaneously activated out of the total number of activations of one of those experts:

where:

NE

Expert co-activation(Ei,Ej) =

, (6)

i,Ej NE

i

- • Ei: The first expert.
- • Ej: The second expert.
- • NE

i,Ej: The number of times experts Ei and Ej are activated together.

- • NE

: The total number of times expert Ei is activated.

i

A co-activation of 100% indicates that if Ei is activated, Ej is also always activated. A value of 0% indicates that the experts never co-occur. If multiple expert pairs have high co-activation, it may suggest that these experts could be merged, benefiting less from keeping them separate. In a distributed setup, we could place highly co-activated experts on the same device to reduce communication costs during model inference.

#### F Training Time and Resource Allocation

- Table 11 reports the training time and GPU resource consumption across all experimental configurations. In the VLM setting, SharedE-V2 and SharedE-V3 consistently demonstrate superior training efficiency, completing training on the OneVision dataset approximately two hours faster than all competing methods. This speedup stems from the use of shared experts, which do not require per-token routing probability computation. As a result, the router only performs inference over the remaining non-shared experts—i.e., N minus the number of shared experts thereby reducing routing overhead and improving overall training efficiency. Importantly, this reduction in runtime does not come at the expense of model quality. As shown in Table 1, both variants remain highly competitive and, in several cases, achieve the best overall performance on OneVision. For language model pretraining, we adopt the CVMM Triton kernel Csordás et al. (2024), which significantly accelerates sparse MoE computation. Consequently, the pretraining runtime

in this setting differs substantially from that of the VLM experiments and is dominated by kernel-level optimizations rather than architectural design choices such as SharedE-V2 or SharedE-V3. For this reason, we recommend interpreting training-time comparisons primarily within the VLM setting, where runtime differences more directly reflect methodological design decisions and thus provide a more informative and meaningful comparison.

Table 11: Training Time and GPU Resource Allocation across all Experimental Settings.

Training Time (hours)

Resource Allocation

Model

Pre-Training 2h35m 4xH100 Pre-FineTuning 16h 4xH100

VLM

SMoE 17h19m 4xH100 XMoE 17h59m 4xH100 σ-MoE 17h32m 4xH100

Visual Instruction Tuning LLAVA-665K

SharedE-V2 16h29m 4xH100 SharedE-V3 16h47m 4xH100

TC-MoE 18h01m 4xH100 MoE++ 18h50m 4xH100

SMoE 33h01m 4xH100 XMoE 34h27m 4xH100 σ-MoE 33h33m 4xH100

Visual Instruction Tuning OneVision / 1M2 samples

SharedE-V2 31h19m 4xH100 SharedE-V3 31h39m 4xH100

TC-MoE 33h47m 4xH100 MoE++ 35h40m 4xH100

Dense 5h47m 4xH100 SMoE 6h15m 4xH100 XMoE 6h21m 4xH100 σ-MoE 6h12m 4xH100

0.15B parametes

SharedE-V2 6h25m 4xH100 SharedE-V3 6h23m 4xH100

TC-MoE 6h10m 4xH100 MoE++ 6h17m 4xH100

Language Modeling

Dense 41h30m 4xH100 SMoE 42h11m 4xH100 XMoE 42h54m 4xH100 σ-MoE 41h57m 4xH100

0.68B parametes

SharedE-V2 43h01m 4xH100 SharedE-V3 43h09m 4xH100

TC-MoE 41h38m 4xH100 MoE++ 42h22m 4xH100

#### G Training Benchmark Curves

-MoE

Dense

SharedE-V2 SharedE-V3

MoE++ TC-MoE

SMoE

XMoE

###### Training Loss

###### Balancing Loss

###### Validation Loss

- 3

- 4

- 5

- 6

- 7

3.2

0.15

3.0

0.10

0.05

2.8

0.00

20 40 60 80 100

20 40 60 80 100

20 40 60 80 100

###### LAMBADA

###### BLiMP

###### CBT

78

85.0

25

| |
|---|

| |
|---|

| |
|---|

76

82.5

| |
|---|

| |
|---|

| |
|---|

| |
|---|

20

80.0

74

| |
|---|

| |
|---|

77.5

72

15

Performance

75.0

20 40 60 80 100

20 40 60 80 100

20 40 60 80 100

###### HellaSwag

###### PIQA

###### ARC-Easy

- 30

- 31

- 32

- 33

- 34

60

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

- 27

- 28

- 29

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

58

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

56

| |
|---|

| |
|---|

| |
|---|

| |
|---|

54

| |
|---|

20 40 60 80 100

20 40 60 80 100

20 40 60 80 100

###### RACE

###### SIQA

CommonSenseQA

- 35

- 36

- 37

- 29

- 30

- 31

- 23

- 24

- 25

- 26

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

20 40 60 80 100

20 40 60 80 100

20 40 60 80 100

Training Steps (k)

- Figure 18: Benchmark curves during training in language modeling tasks for models with 0.15B parameters.

-MoE

Dense

SharedE-V2 SharedE-V3

MoE++ TC-MoE

SMoE

XMoE

###### Training Loss

###### Balancing Loss

###### Validation Loss

8

2.8

0.2

6

2.6

0.1

4

2.4

2

0.0

80 160 240 320 400

80 160 240 320 400

80 160 240 320 400

###### LAMBADA

###### BLiMP

###### CBT

40

90

| |
|---|

| |
|---|

80

| |
|---|

| |
|---|

| |
|---|

88

35

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

78

86

| |
|---|

30

| |
|---|

| |
|---|

| |
|---|

84

76

25

| |
|---|

Performance

82

| |
|---|

20

74

80 160 240 320 400

80 160 240 320 400

80 160 240 320 400

###### HellaSwag

###### PIQA

###### ARC-Easy

40

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

37.5

| |
|---|

65.0

| |
|---|

| |
|---|

38

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

35.0

| |
|---|

| |
|---|

62.5

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

36

| |
|---|

| |
|---|

| |
|---|

| |
|---|

32.5

60.0

| |
|---|

| |
|---|

| |
|---|

34

| |
|---|

30.0

57.5

32

27.5

80 160 240 320 400

80 160 240 320 400

80 160 240 320 400

###### RACE

###### SIQA

CommonSenseQA

- 35

- 36

- 37

- 38

- 39

34

28

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

32

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

26

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

30

24

| |
|---|

80 160 240 320 400

80 160 240 320 400

80 160 240 320 400

Training Steps (k)

- Figure 19: Benchmark curves during training in language modeling tasks for models with 0.68B parameters.

[Figure 33]

27 19 62 13

Layer4Layer12

1

- 9

50 39 58 32 54 23 44 36 61

[Figure 34]

44 62 38 13 50 27 58 9 32 23 54 39 43 61 36

44 62

- 38 13 50 27 58

9 32 23 54

- 39 43 61 36

[Figure 35]

38 13 44 62 27 50 39 58 23 32 9 54 43 61 36

- 38 13 44 62 27 50

- 39 58 23 32

9 54 43 61 36

[Figure 36]

38 44 13 27 62 50 58 32 54 39 23 43 9 61 36

- 38 44 13 27 62 50 58 32 54

- 39 23 43

9 61 36

[Figure 37]

44 21 13 27 62 50 58 23 32 54 39 43 9 61 36

44 21 13 27 62 50 58 23 32 54 39 43

9 61 36

[Figure 38]

1 54 48 21 16 10 35 11 53 49 60 31 9 34 8

1 54

- 48 21 16

10 35 11 53

- 49 60 31

27 19 62 13 1 9 50 39 58 32 54 23 44 36 61

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

50 63 60 49 31

6

6 65 49

- 10 50 49 65 31 63 60

9 21 16

- 34

11 53

- 35 8

- 10 65 63 49 31 21

9 53 16

- 34 60

11

- 35 8

- 10 31 63 60 21

9

- 34 16

11 53

- 35 8

- 10 65

9 21

- 34 16

11 53

- 35 8

9 34

8

50 63 60 49 31 10 65 9 21 34 16 11 53 35 8

6 10 65 63 49 31 21 9 53 16 34 60 11 35 8

6 65 49 10 31 63 60 21 9 34 16 11 53 35 8

10 50 49 65 31 63 60 9 21 16 34 11 53 35 8

Training Steps

Figure 20: Expert Co-Activation across training XMoE model in a language modeling task.

[Figure 43]

34 38

9 46

Layer4Layer12

- 40 8 5 2

18

- 27 21 50

0 54 48

10K

[Figure 44]

52 15 40 38 5 21 46 8 18 27 50 0 54 2 48

52 15 40 38

5 21 46

8 18 27 50

0 54

2 48

40K

[Figure 45]

62 38 52 5 46 21 15 18 8 27 50 54 0 2 48

62 38 52

5 46 21 15 18

8 27 50 54

0 2

48

60K

[Figure 46]

62 52 38 5 15 21 46 8 18 27 50 0 54 2 48

62 52 38

5 15 21 46

8 18 27 50

0 54

2 48

80K

[Figure 47]

30 52 38 5 21 15 46 18 27 8 50 0 54 2 48

30 52 38

5 21 15 46 18 27

8 50

0 54

2 48

100K

[Figure 48]

18 12 34 41 28 54 5 55 20 33 38 48 14 26 58

18 12 34 41

- 28

34 38 9 46 40 8 5 2 18 27 21 50 0 54 48

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

34 18 20 28

2

21 18 20

21 18

- 4

18

- 54 38

5

- 55 33 41 48 28

- 4

- 54 38

5

- 55 33 48 41 28 14

- 54 38

- 55 5

5

- 54 5

- 55 20 33 38 48 14 26 58

- 54 38 33

- 55 41 48 27 14 26 58

33 48 28 41 14

- 26 14 58

- 27 21 18 20 54 38 55 5 33 48 28 41 14 26 58 27

- 26 58

- 27 21 18 4 54 38 5 55 33 48 41 28 14 26 58 27

- 26 58

- 27

34 18 20 28 5 54 38 33 55 41 48 27 14 26 58

2 4 18 54 38 5 55 33 41 48 28 26 14 58 27

Training Steps

Figure 21: Expert Co-Activation across training σ-MoE model in a language modeling task.

[Figure 53]

- 55 19

- 5

- 53

- 54 49 61

- 44 23 15

- 10

6

- 36 13

56

10K

[Figure 54]

43 7 44 22 23 6 15 61 49 54 10 53 13 36 56

- 43 7

- 44

- 22

- 23 6

15 61 49 54 10 53 13 36 56

40K

[Figure 55]

6 52 17 0 23 22 61 15 54 10 49 53 36 56 13

6

- 52 17

0 23 22 61 15 54 10 49

- 53 36 56 13

60K

[Figure 56]

43 22 17 7 6 23 10 15 49 53 61 54 36 56 13

43

- 22 17

7 6

- 23 10 15 49

- 53 61

- 54 36 56 13

80K

[Figure 57]

7 22 52 6 17 15 49 23 53 61 54 10 56 13 36

7

- 22

- 52 6

17 15 49 23

- 53 61

- 54 10 56 13 36

100K

[Figure 58]

45 3 52 13 29 37 43 30 61 22 26 11 39 55 21

45

3 52 13

- 29

37 43

- 30 61 22 26

- 11 39

- 55 21

Training Steps Layer4Layer12

55 19 5 53 54 49 61 44 23 15 10 6 36 13 56

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

- 43 57 30 52

- 44 13 26

1

30

46 31 26 13 20 10

- 45 41

- 56 9

13 55 46

- 57 26 12

- 11 10 13 26

- 54 56

- 55

- 12 19 41

- 55 54 41

- 56

- 57 24

9 56

- 3

48 21 24

- 4

4 48 24 29

4 24 48 29

4 48 29

29

43 57 30 52 44 13 26 9 56 3 48 21 24 4 29

1 45 41 56 9 13 55 46 57 26 12 4 48 24 29

30 11 10 13 26 54 56 55 12 19 41 4 24 48 29

46 31 26 13 20 10 55 54 41 56 57 24 4 48 29

Figure 22: Expert Co-Activation across training SharedE-V2 model in a language modeling task.

[Figure 63]

23 38

6

- 20

- 25 13

1 44 63

3

- 51 53 56 61

- 10 43 18 19 25 23 0 36 63 44 1 6 53 10 56 61

[Figure 64]

- 43

- 18

- 19 25 23

- 0

36 63 44

- 1 6

53 10 56 61

[Figure 65]

13 18 53 25 19 26 44 63 23 51 36 6 56 61 10

13

- 18 53

- 25

19

- 26 44 63 23 51 36

6 56 61 10

[Figure 66]

53 18 13 44 25 19 3 26 23 63 6 36 61 56 10

53

- 18 13 44

- 25

19 3

- 26 23 63

6 36 61 56 10

[Figure 67]

26 13 18 44 25 53 6 23 63 51 19 36 56 61 10

26 13

- 18 44 25 53

6 23 63 51

- 19 36 56 61 10

[Figure 68]

43 19 60 11 6 47 10 41 44 26 52 29 21 55 48

- 43 19 60

11 6

- 47 10 41

44 26 52 29 21 55

- 48 21 5 44 19 25 11 30 61 41 12 48 6 55 52 8

Layer4Layer12

23 38 6 20 25 13 1 44 63 3 51 53 56 61 10

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

21

20 12 46

- 48

- 60 47 44 55 46

- 61 8 1

- 49

25

- 5

44 19 25

- 11 30 61 41

- 12 48

- 6

- 60 12 38 20

7 6

41

- 61 27

- 60 30 11

- 61 48

- 55 8

44

- 56 41 52

- 5

41

- 6

8 5

55 52

46 52 56

52 27

8

6

20 12 46 60 30 11 61 48 55 8 44 56 41 52 6

48 60 47 44 55 46 61 8 1 49 5 41 6 52 27

25 60 12 38 20 7 6 41 61 27 8 5 46 52 56

Training Steps

Figure 23: Expert Co-Activation across training SharedE-V3 model in a language modeling task.

[Figure 73]

95 30

- 22

111 117

93 52 70

- 23 7

Layer4Layer12

64

- 16

126

- 24 39

10K

[Figure 74]

95 8 20 66 9311723 52 7 7012664 16 24 39

95

8 20 66 93

117 23 52

7 70

126 64 16 24 39

40K

[Figure 75]

5611793 9511166 23 52 7 7012664 16 24 39

56

117 93 95

111 66 23 52

7 70

126 64 16 24 39

60K

[Figure 76]

11193 95 4911766 23 7 52 7012664 16 24 39

111 93 95 49

117 66 23

7 52 70

126 64 16 24 39

80K

[Figure 77]

5611149 93 6611723 7 52 70 6412616 24 39

56

111 49 93 66

117 23

7 52 70 64

126 16 24 39

100K

[Figure 78]

47 18 8 17 49 44 11 32 26 2 56 25 33 23 52

47 18

8 17 49 44 11

- 32 26

2 56 25

- 33 23 52

95 30 2211111793 52 70 23 7 64 1612624 39

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

60 45 49 17 44 11

35

130 45 35 49 11

17 62 35 49 11

- 51 45 49 11 44

8 26

- 32 56

2

- 33

- 52 25 23

8

- 32 8

44 26 56

2

- 33 25 52 23

8

- 32 44 26 56

2

- 33 25 52 23

- 32 26 56

2

- 33 25 52 23

60 45 49 17 44 11 8 32 26 56 2 33 25 52 23

35 51 45 49 11 44 8 26 32 56 2 33 52 25 23

13045 35 49 11 8 32 44 26 56 2 33 25 52 23

17 62 35 49 11 32 8 44 26 56 2 33 25 52 23

Training Steps

Figure 24: Expert Co-Activation across training TC-MoE model in a language modeling task.

[Figure 83]

61 49

4 2

Training Steps Layer4Layer12

16 18 40

0 29 54

- 6

24 12 32 10

10K

[Figure 84]

71 14 44 18 26 21 0 40 29 24 54 6 32 12 10

71 14 44 18 26 21

0 40 29 24 54

6 32 12 10

40K

[Figure 85]

62 26 49 21 44 18 0 40 29 24 54 32 12 6 10

62 26 49 21 44 18

0 40 29 24 54 32 12

6 10

60K

[Figure 86]

38 44 26 18 49 21 0 29 40 24 6 54 32 12 10

38 44 26 18 49 21

0 29 40 24

6 54 32 12 10

80K

[Figure 87]

38 26 39 21 49 18 0 29 40 24 54 6 32 12 10

- 38 26

- 39 21 49 18

0 29

- 40 24 54

6 32 12 10

100K

[Figure 88]

7 68 54 21 36 23 51 46 26 42 38 18 10 34 70

- 7

61 49 4 2 16 18 40 0 29 54 6 24 12 32 10

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

54 21 31 46

68 31 51 42 46 44 36 38 23 26 10

42 31 51 46 36 38

- 25 23 31 51 42 38 36 46

3

- 26 7

68 54 21 36 23 51 46 26 42 38 18 10 34 70

7 36 44 38

3 23 44 26

- 25 51

- 26 34 10 18 70

7 10 34 70 18

7 70 34 18

10 70 34 18

54 21 31 46 7 36 44 38 25 51 26 34 10 18 70

68 31 51 42 46 44 36 38 23 26 10 7 70 34 18

42 31 51 46 36 38 3 23 44 26 7 10 34 70 18

25 23 31 51 42 38 36 46 3 26 7 10 70 34 18

Figure 25: Expert Co-Activation across training MoE++ model in a language modeling task.

