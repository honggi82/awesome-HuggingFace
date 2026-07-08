arXiv:2412.06845v6[cs.CL]4Jul2025

# 7B Fully Open Source Moxin-LLM/VLM – From Pretraining to GRPO-based Reinforcement Learning Enhancement

Pu Zhao1, Xuan Shen1, Zhenglun Kong2, Yixin Shen3, Sung-En Chang1, Arash Akbari1, Timothy Rupprecht1, Lei Lu1, Enfu Nan1, Changdi Yang1, Yumei He4, Weiyan Shi1, Xingchen Xu5, Yu Huang6, Wei Jiang7, Wei Wang7, Yue Chen7, Yong He7, Yanzhi Wang1,8

1Northeastern University, 2Harvard University, 3Cornell University, 4Tulane University, 5University of Washington, 6Roboraction.ai, 7Futurewei Technologies, 8AIBAO LLC

## Abstract

Recently, Large Language Models (LLMs) have undergone a significant transformation, marked by a rapid rise in both their popularity and capabilities. Leading this evolution are proprietary LLMs like GPT-4 and GPT-o1, which have captured widespread attention in the AI community due to their remarkable performance and versatility. Simultaneously, open-source LLMs, such as LLaMA and Mistral, have made great contributions to the ever-increasing popularity of LLMs due to the ease to customize and deploy the models across diverse applications. Although open-source LLMs present unprecedented opportunities for innovation and research, the commercialization of LLMs has raised concerns about transparency, reproducibility, and safety. Many open-source LLMs fail to meet fundamental transparency requirements by withholding essential components like training code and data, and some use restrictive licenses whilst claiming to be “open-source,” which may hinder further innovations on LLMs. To mitigate this issue, we introduce Moxin 7B, a fully open-source LLM developed in accordance with the Model Openness Framework (MOF), a ranked classification system that evaluates AI models based on model completeness and openness, adhering to principles of open science, open source, open data, and open access. Our model achieves the highest MOF classification level of “open science” through the comprehensive release of pre-training code and configurations, training and fine-tuning datasets, and intermediate and final checkpoints, aiming to make continuous commitments to fully open-source LLMs. The pre-training cost is about $160,000. After pre-training and obtaining the base model, we finetune the Moxin Base model with SOTA post-training framework and instruction data to obtain Moxin Instruct model. To improve the reasoning capability, we further finetune our Instruct model with chain-of-thought data distilled from DeepSeek R1, and then use Group Relative Policy Optimization (GRPO), an efficient and effective reinforcement learning algorithm following DeepSeek R1, to finetune our model, leading to the Moxin Reasoning model. Moreover, based on an open-source vision language model (VLM) framework, we develop and release our VLM with our Moxin model as the LLM backbone. Experiments show that our models achieve superior performance in various evaluations such as zero-shot evaluation, few-shot evaluation, and CoT evaluation, compared with other base models and instruct models. The outstanding performance of our Reasoning model demonstrates the effectiveness of RL for small LLMs such as 7B models. Our VLM outperforms other VLM models or LLM backbones. Besides our open pretraining with released base model, data, code, etc., in our post-training including the instruction finetuning, CoT finetuning,

and VLM finetuning, we adopt open-source training frameworks with available data, code, and configurations. We release our instruct model, reasoning model, and VLM model, along with the available data and code to derive these models.

Homepage with all codes: https://github.com/moxin-org/Moxin-LLM Base model: https://huggingface.co/moxin-org/Moxin-7B-LLM Instruct model: https://huggingface.co/moxin-org/Moxin-7B-Instruct Reasoning model: https://huggingface.co/moxin-org/Moxin-7B-Reasoning VLM model: https://huggingface.co/moxin-org/Moxin-7B-VLM

## 1 Introduction

The field of natural language processing has witnessed the most exciting discoveries of the last ten years with the emergence of large language models (LLMs). At the forefront of this evolution are LLMs such as GPT-4 [1], Claude [2], and Gemini [3], which have captured the attention of the AI community due to their performance and versatility. Meanwhile, the recent emergence of openly accessible yet highly capable LLMs such as LLaMA [4], Falcon [5], and Mistral [6] allow researchers and practitioners to easily obtain, customize, and deploy LLMs in more various environments and for more diverse use cases. The trends have people eagerly asking about what’s next and some suggest “a general intelligence” is right around the corner.

Despite the growing influence and accessibility of open-source LLMs, a notable challenge has emerged: many model producers restrict visibility and access to their training, fine-tuning, and evaluation processes, including crucial components such as their training code and data [7]. Some model producers even use restrictive licenses whilst claiming to be “open-source.” This practice creates barriers for the broader AI research community to study, replicate, and innovate upon advanced LLMs. In parallel, it prevents businesses from fully leveraging open-source models for innovative industrial applications, as its commercialization has raised concerns about transparency, reproducibility, and safety.

To unlock the full potential of LLMs and open innovation, we must democratize this research by putting models into the hands of more researchers and making the datasets the models train on fully open-source. This requires moving beyond the simple sharing of model weights to embrace complete transparency in training, datasets, and implementation detail, which is crucial for fostering a more inclusive and collaborative research environment that can sustain a healthy open-source ecosystem [8].

To achieve this goal, we introduce Moxin 7B, a fully open-source LLM developed by complying with the Model Openness Framework (MOF) introduced by [9]. The MOF provides a systematic ranking classification system to rate AI models based on their completeness and openness, incorporating the principles of open science, open source, open data, and open access. By promoting transparency and reproducibility, the MOF serves as a crucial tool to combat “openwashing” and to establish completeness and openness as primary criteria alongside the core tenets of responsible AI. The wide adoption of MOF and open-source state-of-the-art models will cultivate a more open AI ecosystem, benefiting research and innovation.

Our open-source LLM has released pre-training code and configurations, training and fine-tuning data, and intermediate and final checkpoints, aiming to make continuous commitments to fully open-source LLMs. Our model achieves the highest MOF classification level of “open science”. It is noteworthy that this commitment to openness has not compromised performance: our base model achieves superior performance in zero-shot evaluation compared to popular 7B models and performs competitively in few-shot evaluation.

We also finetune the Moxin Base model with SOTA post-training framework and instruction data to obtain Moxin Instruct model. To improve the reasoning capability, we further finetune our model with chain-of-thought data distilled from DeepSeek R1, and then use Group Relative Policy Optimization, an efficient and effective reinforcement learning algorithm following DeepSeek R1, to finetune our model, leading to the Moxin Reasoning model. Moreover, we develop and release our vision language model (VLM) based on our Moxin model as the LLM backbone with an open-source VLM framework. Experiments show that our models achieve superior performance in various evaluations

such as zero-shot evaluation, few-shot evaluation, and CoT evaluation, compared with other instruct and reasoning models. The outstanding performance of our Reasoning model demonstrates the effectiveness of RL for small LLMs such as 7B models. Our VLM outperforms other VLM models or LLM backbones.

Besides our open pretraining with released base model, data, code, etc., in our post-training including the instruction finetuning, CoT finetuning, and VLM finetuning, we adopt open-source training frameworks with available data, code, and configurations. We release our instruct model, reasoning model, and VLM model, along with the available data and code to derive these models. Our homepage is https://github.com/moxin-org/Moxin-LLM. We summarize our contributions below.

- • For the pre-training, we release the Moxin-7B-Base model, together with all training code, data, and checkpoints, aiming to make continuous commitments to fully open-source LLMs. Our base model performs competitively compared with SOTA pre-trained base models, with a moderate training cost about $160,000.
- • To further improve the model performance, we post-train our base model with SFT and DPO methods, based on the open-source Tülu 3 framework. Two versions of instruction models are developed with two different open-source datasets (Tülu 3 and Infinity Instruct), respectively. Both models are available and the one finetuned with the Tülu 3 dataset is released as Moxin-7B-Instruct, with superior performance in different evaluations.
- • To improve the reasoning capabilities of the model with CoT, we further finetune our instruct model with SFT based on reasoning data from Open-Thoughts and OpenR1-Math-220k, and then reinforcement learning (GRPO following DeepSeek R1). We develop two versions of reasoning models with two different open-source RL frameworks (DeepScaleR and AReaL) and their corresponding data, respectively. Both models are available and the one finetuned with DeepScaleR is released as Moxin-7B-Reasoning, with superior performance in CoT evaluations. We demonstrate that RL such as GRPO can work effectively for small LLMs.
- • Moreover, we develop and release our VLM based on our Moxin model as the LLM backbone. Based on the open-source VLM framework Prismatic VLMs, we train our VLM on fully open-source datasets with Moxin as the LLM backbone and DINOv2 & SigLIP as the visual backbone. Our experiments demonstrate that our VLM outperforms other VLM models or LLM backbones.

We summarize our open-source releases below.

- • Pre-training code, data, and Moxin Base model.
- • Post-training code, data, and Moxin Instruct model.
- • RL code, data and Moxin Reasoning model.
- • Training code, data and Moxin VLM model.

## 2 Related Work

#### 2.1 Models, Tokenizers, and Training

Models. State-of-the-art large language models (LLMs) typically comprise a substantial number of parameters, often approaching or exceeding 100 billion [4, 1, 3]. To facilitate broader accessibility, smaller models with fewer than 20 billion parameters, and even those around 7 billion parameters, have been developed [10, 11, 4, 6]. In addition, efficiency-enhancing techniques, such as pruning [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 18], quantization [25, 26, 23, 27, 28, 29, 30], token reduction [31, 32, 33, 34, 35, 36, 37] or implementing MAMBA-based architectures in Jamba, have been employed to optimize performance [38, 39].

Tokenizers. Tokenizers are essential to convert raw data into a suitable format for model processing. Many contemporary models employ Byte-Pair Encoding (BPE)[40], with OpenAI’s tiktoken tokenizer[41] being a notable implementation. However, for languages that handle tokens differently from Romance languages, alternatives such as SentencePiece [42] are utilized, as seen in XLNet [43]. Hugging Face offers an excellent summary of state-of-the-art tokenizers with practical examples [44].

Moreover, tokenization extends beyond text modalities; many foundational models now include multimodal capabilities, processing documents, audio, images, and even videos [45, 46, 47, 48].

Training. To enhance the performance of smaller models beyond their inherent limitations, various training strategies can be employed. A notable example is the application of Mixture of Experts (MoE) training, which has achieved significant success in models such as Mixtral [49].

#### 2.2 Data curation methods

Researchers commonly collect large datasets for training language models (LMs)[50] by performing web crawls. However, these datasets often contain undesirable content, necessitating data curation to improve their quality. To enhance model performance[51, 52, 50, 53], several data curation techniques are widely employed. These include filtering by language [54, 55, 56], heuristic-based filtering [51, 57, 58], quality filtering [59, 60, 61], data deduplication [62, 63], and data mixing [64, 65, 66].

#### 2.3 Open-source datasets

As the scale of LMs has increased in recent years [4, 67, 68, 1], the community has correspondingly curated larger datasets to support their training. Early datasets include the C4 dataset, containing 160 billion tokens, and The Pile [58], which comprises 300 billion tokens. More recently, even larger datasets have been introduced: RefinedWeb [51] with 600 billion tokens, Dolma [69] with 3 trillion tokens, FineWeb [70] with 15 trillion tokens, and RedPajama-v2 [71] containing 30 trillion tokens. In addition to these general-purpose datasets, large domain-specific datasets have also been developed. For instance, StackV2 [72], a code-focused dataset, includes 900 billion tokens, and FineWeb-Edu [70], a high-quality filtered educational text dataset, contains 1.3 trillion tokens.

Besides model pre-training, post-training has become a vital step in refining behaviors and unlocking new capabilities in LLMs. Post-training includes multiple techniques such as instruction tuning and reinforcement learning from human feedback. The sophistication and complexity of post-training approaches have continued to increase, with multiple rounds of training, model merging, leveraging synthetic data, and various training algorithms and objectives. Multiple post training datasets are released, such as Tülu 3 [73] and Infinity Instruct [74]. Tülu 3 [73] creates new datasets & new training procedures, and introduces new methods for training directly on verifiable problems with reinforcement learning. It includes both SFT training datasets and reinforcement learning datasets, together with the SFT, DPO, and PPO training recipes. Infinity Instruct [74] aims to develop a large-scale, high-quality instruction dataset with millions of high-quality instruction samples.

Furthermore, as the breakthrough of DeepSeek R1 [75], a line of SFT datasets and reinforcement learning datasets have been developed to enhance the reasoning capabilities of LLMs. Openthoughts [76] and OpenR1-Math-220k [77] datasets are obtained by feeding questions to DeepSeek R1 and collecting the reasoning responses from R1. Openthoughts [76] contains open synthetic reasoning dataset with 114k high-quality examples covering math, science, code, and puzzles. OpenR1-Math220k [77] is a large-scale dataset for mathematical reasoning, with 220k math problems and their corresponding solutions generated by DeepSeek R1. Besides, multiple datasets are developed and released for the reinforcement learning of LLMs following DeepSeek R1, such as DeepScaleR [78] and AReaL [79, 80].

#### 2.4 Open-source LLMs

Large language models (LLMs) have rapidly evolved into a diverse ecosystem that spans closedsource, open-weight, and fully open-sourced paradigms. On one end of the spectrum, closed-source models such as GPT-4 [1], and Gemini [81] have set high performance standards but remain accessible primarily through API services, limiting insight into their underlying architectures and training methodologies. In contrast, open-weight LLMs—while sharing their final model architectures and weights—often leave training data and many training details undisclosed. This category includes influential models Llama [82], Mistral [6], Gemma [67], Qwen [10], DeepSeek [75, 83], Baichuan [84], Phi [85], etc. Pushing the frontier further, fully open-sourced LLMs have begun to provide not only complete model weights and architectures but also the training code and datasets necessary for reproducibility. Exemplars of this fully open paradigm include Pythia [86], GPT-NeoX [87], OpenLLaMA, StarCoder [72], OLMo [88], Amber and Crystal from LLM360 [89], etc. Collectively,

this vibrant landscape highlights an ongoing shift toward more accessible and fully reproducible LLMs, fostering collaboration and innovation across both academia and industry.

## 3 Model Pre-Training

In this section, we provide an in-depth discussion of the model training process, covering the architectural design, data curation and preprocessing, training configurations, fine-tuning for alignment, and techniques for handling long-context sequences. We detail both our methodological choices and the technical challenges addressed during the development of Moxin 7B.

#### 3.1 Model Architecture

We extend the Mistral model architecture [6] to balance high performance with efficient inference. The original Mistral 7B model demonstrates superior performance compared to multiple 7B language models and even outperforms larger models on various evaluation benchmarks. Notably, it surpasses the LLaMA 34B model [90] in tasks such as mathematics and code generation. Key architectural enhancements include:

- • Depth Extension: While the original Mistral model comprises 32 blocks, our extended model uses 36 transformer blocks. This additional depth was empirically validated to improve both learning capacity and performance on complex downstream tasks.
- • Layer Normalization and Initialization: Pre-layer normalization is applied to stabilize the training process, accompanied by a custom initialization scheme to mitigate gradient vanishing/exploding issues in deep transformer networks.
- • Regularization Techniques: Dropout with a rate of 0.1 is employed across attention and feed-forward layers, and label smoothing is used during fine-tuning to further regularize the model.
- • Mixed-Precision and Activation Checkpointing: Mixed-precision training (FP16) is leveraged to double throughput and reduce memory consumption. Activation checkpointing is used to trade additional computation for a lower memory footprint, enabling the training of deeper networks on the available hardware.

The choice of hyperparameters, as listed in Table 1, was informed by preliminary ablation studies and a grid search over the parameter space.

#### 3.2 Long-Context

To process extensive sequences efficiently, we integrate our long-context handling framework into the training and inference pipelines:

- • Grouped-Query Attention (GQA) [91]: We employ GQA to partition query heads into groups, where each group shares a single key and value head. This approach serves as an interpolation between multi-query attention (MQA) and multi-head attention (MHA), balancing computational speed with representational richness. GQA reduces memory requirements during decoding, enabling larger batch sizes and improved throughput.
- • Sliding Window Attention (SWA) [92]: SWA enables efficient handling of long sequences by partitioning attention into a fixed-size sliding window. This not only limits computational overhead but also preserves the ability to model dependencies beyond the immediate context window. Specifically, using a window size W = 4096 allows the final layer to achieve a theoretical attention span exceeding 14,000 tokens.
- • Rolling Buffer Cache [6]: To further mitigate memory usage during inference, a rolling buffer cache with a fixed attention span is employed. For timestep i, keys and values are stored at position i mod W, ensuring that once the sequence length exceeds W, older context tokens are overwritten rather than expanding the cache indefinitely. On sequences of 32K tokens, this approach reduces cache memory usage by approximately 8× without degrading model performance.

With the above techniques, our model can support 32K context length with fast inference and low memory cost.

#### 3.3 Training Data

Data are fundamental to the pre-training of LLMs. Preparing such training data requires careful consideration of multiple challenges, including handling sensitive information, ensuring comprehensive knowledge coverage, and achieving higher efficiency with improved data quality.

In this section, we detail the processes of preparing textual data from general domains and coding data related to programming languages.

- 3.3.1 Text Data We use a mix of data from SlimPajama [93] and DCLM-BASELINE [64] as our text training data.

SlimPajama Curation and Deduplication: During the training of LLaMA, it was demonstrated that the performance of a 7B model continues to improve even after being trained on more than 1T tokens [82]. Given the outstanding performance of LLaMA, its data collection methodology was rapidly replicated, leading to the release of RedPajama, an open-source dataset containing 1.2 trillion tokens [94].

Table 1: Parameter setting.

|Parameter<br><br>|Value|
|---|---|
|n_layers dim head_dim hidden_dim n_heads n_kv_heads<br><br>|36 4096 128 14336 32 8|

However, subsequent analyses reveal a significant limitation: some corpora within RedPajama contain a large percentage of duplicate content. The deduplication guidelines in RedPajama operate only within individual data sources, leaving inter-source duplicates largely unaddressed. To improve data quality and training efficiency, SlimPajama1 was developed as a refined iteration of RedPajama, offering a cleaned and extensively deduplicated version [93].

SlimPajama implements a rigorous two-stage preprocessing pipeline to enhance data quality. In the first stage, short and low-quality documents are removed from RedPajama. Specifically, documents that have fewer than 200 characters after removing punctuation, space symbols, newlines, and tabs are filtered out, as these documents typically contain only metadata and lack useful information. As a result of this step, 1.86% of RedPajama documents are eliminated.

The second step involves removing duplicate data, as deduplication enhances training efficiency and reduces memorization, thereby decreasing the likelihood of generating text solely by recalling training data [51, 95, 96, 62, 97]. To perform deduplication, document signatures are created using pre-processed, lower-cased 13-grams. Subsequently, MinHashLSH [98] is employed to identify and eliminate duplicates based on a Jaccard similarity threshold of 0.8. Deduplication is performed both within and across data sources. Overall, by pruning 49.6% of the bytes from the RedPajama dataset, the 627B-token SlimPajama dataset is obtained.

DCLM-BASELINE Processing: Additionally, we utilize the DCLM-BASELINE [64] dataset2, which is derived from CommonCrawl, a web-crawled dataset [99]. The construction of DCLMBASELINE involves several steps. First, resiliparse is employed to extract text from CommonCrawl. Second, deduplication is performed using MinHash [100] within a suffix array pipeline [101, 62] and near-duplicate Bloom filtering, which enhances the exact document and paragraph deduplication scheme [69]. Third, recent studies [102, 69, 103] demonstrate that utilizing learnable models as quality filters leads to downstream performance improvements. Consequently, DCLM-BASELINE applies a fastText OH-2.5 combined with an ELI5 classifier score to retain the top 10% of documents.

- 3.3.2 Coding Data

Programming is crucial for LLMs to support various downstream tasks, such as code completion from natural language descriptions, documentation generation for individual functions, and autocompletion of code snippets. Furthermore, as code is generally better structured and organized than

- 1https://huggingface.co/datasets/cerebras/SlimPajama-627B
- 2https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0

natural language, training on code data may improve the LLM reasoning capabilities [88]. Therefore, We use part of the-stack-dedup [104] dataset3 during the pretraining.

The Stack comprises more than 6TB of permissively-licensed source code files across 358 programming languages [104]. This carefully curated resource was designed to enhance the code generation capabilities of LLMs. It facilitates the synthesis of programs by code-generating AI systems from both natural language descriptions and existing code snippets.

To construct the Stack dataset, 220.92 million active GitHub repositories were collected from event archives published between 2015 and 2022 on GHArchive. Of these repositories, only 137.36 million were publicly accessible on GitHub, resulting in 51.76 billion downloaded files. After initial filtering, 5.28 billion unique files were identified, with an uncompressed size of 92.36 TB.

To ensure data quality, near-deduplication was implemented within the preprocessing pipeline in addition to exact deduplication. Specifically, MinHash with 256 permutations was computed for all documents, and Locality Sensitive Hashing was employed to identify clusters of duplicates. Within these clusters, Jaccard similarities were calculated to detect near-duplicates using a similarity threshold of 0.85. Approximately 40% of permissively licensed files were identified as (near)duplicates and subsequently removed.

#### 3.3.3 Capability Enhancement

LLMs are expected to demonstrate capabilities such as reasoning, mathematical problem-solving, and knowledge memorizing. However, a significant challenge lies in that, in the pre-training process, high-quality capability-related data is sparsely distributed in the entire corpus, and thereby it is difficult for models to be proficient at these above-mentioned capabilities. Previous research, such as work on Qwen [10], GLM-130B [105], Nemotron-4 [106], has tried to incorporate instruction-based or high-quality data during the pre-training stage to enhance these abilities. In our study, we collect open-source data from HuggingFace, primarily utilizing the training datasets of various evaluation benchmarks such as MMLU [107] and HellaSwag [108]. These data are used experimentally to investigate the relationship between high-quality, capability-focused training data and model performance.

##### 3.4 Training Configuration The pre-training of Moxin 7B spans over 2 trillion tokens and is executed in three distinct phases:

- 1. Base Pre-training: In this initial phase, the model is trained on data with a fixed context length of 2,000 tokens. This stage establishes foundational language modeling abilities, capturing syntax and basic semantic patterns.
- 2. Extended Context Pre-training: In the second phase, the context length is increased to 4,000 tokens, allowing the model to capture longer dependencies and richer context.
- 3. Capability Enhancement: The final phase incorporates the capability-specific enhancement data described earlier. We provide ablation results that compare the performance with only Phases 1 and 2 versus all three phases, thereby quantifying the benefits of the capabilityenhancement data.

We use Colossal-AI [109] as our training framework with acceleration techniques including optimized parallelism, memory optimization, and heterogeneous training methods. 1) Optimized Parallelism: We combine model, data, and pipeline parallelism to scale training across GPUs. This hybrid parallelism strategy minimizes communication overhead and ensures efficient resource utilization. 2) Memory Optimization: Techniques such as ZeRO redundancy elimination, gradient checkpointing, and mixed-precision training (FP16) are employed to maximize the effective batch size and reduce memory consumption. 3) Heterogeneous Training Methods: the dynamic scheduling and asynchronous communication protocols are leveraged to further boost training throughput. These methods are provided through user-friendly APIs, requiring minimal code modifications. The training cost is around $160,000.

During training, AdamW [110] with β1 = 0.9, β2 = 0.95, ϵ = 1e−8 and weight decay = 0.1 is used to optimize the model. We use the cosine learning rate decay, and the learning rate decays to 10% of

3https://huggingface.co/datasets/bigcode/the-stack-dedup

its maximum. Learning Rate is set to 2e−6. Our micro-batch size and gradient accumulation steps were tuned to balance between memory constraints and convergence speed.

## 4 Model Post-Training

Following the pre-training phase, we fine-tune the model into a helpful and harmless AI assistant. We adopt the open-source Tülu 3 dataset and framework [73] for our model post-training. Tülu 3 is a family of open state-of-the-art post-trained models, alongside all of the data, data mixes, recipes, code, infrastructure, and evaluation framework, which pushes the boundaries of research in post-training. To close the performance gap between open and closed fine-tuning recipes, Tülu 3 creates new datasets & new training procedures, and introduces new methods for training directly on verifiable problems with reinforcement learning. The success of Tülu 3 is rooted in careful data curation, rigorous experimentation, innovative methodologies, and improved training infrastructure, integrating partial details from proprietary methods with novel techniques and established academic research.

For our post-training, with our base model, we follow Tülu 3 to perform supervised finetuning (SFT) and then Direct Preference Optimization (DPO). Specifically, we use the Tülu 3 SFT Mixture dataset from Tülu 3 to train our base model with the SFT training method for two epochs and obtain our SFT model, following the default training configuration of the Tülu 3 8B SFT model [73]. To promote diversity, the Tülu 3 SFT Mixture dataset contains 939,344 samples from multiple datasets such as CoCoNot [111], FLAN v2 [112], WildChat GPT-4 [113], etc., together with some data samples collected and released by the Tülu 3 team. It especially consider enhancing several capabilities that can power common use cases and the specific needs, by including following datasets: OpenMathInstruct [114] and NuminaMath [115] for mathematical reasoning, Evol-CodeAlpaca for coding, a subset of Daring-Anteater [116] for precise instruction following, SciRIFF [117] for scientific literature understanding, and TableGPT [118] for processing table-related tasks. It tracks the provenance of dataset and their subsets to verify their licenses and remove those that have issues. With thorough experimentation of Tülu 3, the final SFT data and training hyperparameters are determined to enhance target core skills without significantly impacting the performance of others, guided by the Tülu 3 evaluation framework. We use the Tülu 3 SFT Mixture dataset to train our base model with SFT for 2 epochs with a learning rate 5e-06 and a batch size of 128.

Next, we continue to train our SFT model on the Tülu 3 8B Preference Mixture dataset from Tülu 3 with the DPO training method to obtain our DPO model, following the same training configuration of the Tülu 3 8B DPO model [73]. The data creation pipeline for the Tülu 3 8B Preference Mixture dataset consists of three stages: prompt selection, response generation from a pool of models, and preference annotation with LLM-as-a-judge to create (preferred, rejected) pairs. The preference mixes data come from different prompt sources, such as SFT data, WildChat and Persona IF. It includes prompts seen during SFT training but also new, unseen prompts. To find a better training algorithm and hyper-parameters, Tülu 3 revisits the hyperparameter and algorithm choices alongside the preference datasets. It ablates both algorithm and hyperparameter choices using an early SFT checkpoint and the UltraFeedback dataset. It explores using DPO, SimPO, and length-normalized DPO, leading to the final training configuration with all hyper-parameters. We use the Tülu 3 8B Preference Mixture dataset to finetune our SFT model with DPO for 1 epochs with a learning rate 5e-07.

We also adopt the Infinity Instruct dataset [74] to train another version of our instruct model based on our base model. Infinity Instruct aims to develop a large-scale, high-quality instruction dataset. To construct a ten-million high-quality instruction dataset, it collects a large amount of open-source data

- as seed and iterate the dataset using two strategies: instruction selection and instruction evolution. It includes the Foundational Dataset and the Chat Dataset. The Foundational Dataset contains millions of instruction selected from open-source dataset to improve the performance of model on challenging downstream tasks (e.g., code, math). The Chat Dataset contains about 1M instructions evolved from a small subset of high-quality seed data, to further improve the instruction-following ability of model in real conversation scenarios.

## 5 Chain-of-Thought Enhancement

In the context of reasoning capabilities, OpenAI’s o1 series models firstly introduce inference-time scaling by increasing the length of Chain-of-Thought (CoT) reasoning process. This approach has achieved significant improvements in various reasoning tasks, such as mathematics, coding, and scientific reasoning. However, the challenge of effective test-time scaling remains an open question for the research community. DeepSeek R1 [75] takes the first step towards improving language model reasoning capabilities using pure reinforcement learning (RL). The goal is to explore the potential of LLMs to develop reasoning capabilities without any supervised data, focusing on their self-evolution through a pure RL process. After thousands of RL steps, DeepSeek-R1-Zero exhibits super performance on reasoning benchmarks.

To enhance the CoT capabilities of our model, we adopt RL techniques similar to DeepSeek R1 [75]. We first use high quality reasoning data to SFT our instruct model. The reasoning data mainly includes Openthoughts [76] and OpenR1-Math-220k [77]. These datasets are obtained by feeding questions to DeepSeek R1 and collecting the reasoning responses from R1. Openthoughts [76] contains open synthetic reasoning dataset with 114k high-quality examples covering math, science, code, and puzzles. It generates reasoning traces from DeepSeek-R1 and verifies correctness to construct the final dataset. OpenR1-Math-220k [77] is a large-scale dataset for mathematical reasoning. It consists of 220k math problems with two to four reasoning traces generated by DeepSeek R1 for problems from NuminaMath 1.5. To build OpenR1-Math-220k, it generates two answers for 400k problems using DeepSeek R1 with 512 H100 GPUs. Then it applies Math Verify to only retain problems with

- at least one correct answer, and leverages Llama3.3-70B-Instruct as a judge for 12% of the samples to retrieve more correct examples, leading to a filtered dataset containing 220k problems with at least one reasoning trace with a correct answer for each problem. We finetune our instruct model on these datasets with SFT based on the open-instruct finetuning framework, following the default training configuration of the Tülu 3 8B SFT model [73], with 2 epochs finetuning following a learning rate 5e-06 and a batch size of 128.

Next after SFT, we further adopt the RL techniques in DeepSeek R1, i.e., GRPO to finetune our model with RL. We adopt the DeepScaleR [78] as our RL training framework. DeepScaleR is an open-source project to fully democratize RL for LLMs and reproduce DeepSeek R1 and OpenAI O1/O3 at scale on real tasks. It opens source all training scripts (including hyperparameters), models, dataset, and logs. It is based on veRL [119], Volcano Engine Reinforcement Learning for LLM, a flexible, efficient and production-ready RL training framework designed for LLMs. veRL is flexible and easy to use with easy extension of diverse RL algorithms (combining the strengths of single-controller and multi-controller paradigms to enable flexible representation and efficient execution of complex Post-Training dataflows), seamless integration of existing LLM infra with modular APIs (by decoupling computation and data dependencies, and enabling seamless integration with existing LLM frameworks, such as PyTorch FSDP, Megatron-LM and vLLM), and flexible device mapping (supporting various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes). veRL is fast with state-of-the-art throughput and efficient actor model resharding with 3D-HybridEngine.

DeepScaleR releases DeepScaleR-1.5B-Preview, a 1.5B model, trained on 40K high-quality math problems with 3,800 A100 hours, outperforming OpenAI’s o1-preview on multiple competition-level math benchmarks (such as surpassing O1-Preview and achieves 43.1% Pass@1 on AIME). It is achieved by iteratively scaling Deepseek’s GRPO algorithm from 8K->16K->24K context length for thinking. It releases the training recipe, hyperparameters, data and underlying systems. For the training dataset, it compiled AIME problems and AMC problems prior to 2023, along with questions from the Omni-MATH and Still datasets, which feature problems from various national and international math competitions. The data processing pipeline consists of three key steps: extracting answers, removing redundant questions, and filtering ungradable questions. After deduplication and filtering, the final training dataset consists of approximately 40,000 unique problem-answer pairs.

Different from the original DeepScaleR to train a 1.5B LLM with RL, we adopt the DeepScaleR framework to finetune our 7B DPO model with RL for improving CoT capabilities. As advocated in Deepseek-R1, we employ an Outcome Reward Model (ORM) as opposed to a Process Reward Model (PRM) to avoid reward hacking. The reward function returns: 1 if the LLM’s answer passes basic LaTeX/Sympy checks, and 0 if the LLM’s answer is incorrect or formatted incorrectly. We evaluate our COT reasoning model on competition-level mathematics benchmarks, including AMC

2023, MATH-500, Minerva Math, and OlympiadBench. We report the Pass@1 accuracy. We use Qwen Math’s codebase for evaluation following [120, 121].

Besides DeepScaleR, we adopt another framework, AReal [80, 79] to finetune our DPO model and obtain another version of our COT model. AReaL (Ant Reasoning RL) [80, 79] is an open-source and efficient reinforcement learning system developed at the RL Lab, Ant Research. AReaL inherits and adapts the Open-Source Project ReaLHF [80] for training Large Reasoning Models (LRMs). Its RL training dataset consists of 40k high-quality mathematical reasoning tasks released by DeepScaleR. It assigns rewards of +5 for correct responses and -5 for incorrect responses. The correctness of responses is evaluated by comparing the responses with the answers using Sympy, following the evaluation protocol of Qwen. A simplified variant of PPO is adopted as the RL algorithm with a major change to PPO by eliminating the critic to reduce computation resources. We follow the default configuration of AReal (with necessary modifications due to our specific environment) to train our 7B DPO model for COT reasoning capabilities.

## 6 Vision Language Model based on Moxin

Vision language models (VLMs) [122, 123, 124, 125], which leverage LLMs, have achieved significant breakthroughs, enabling sophisticated vision-language dialogues and interactions. To develop VLM, we adopt the Prismatic VLMs framework [126] to train our VLM based on our Moxin model. Specifically, for the visual backbone with image processing and visual representations, we adopt DINOv2 [127] and SigLIP [128], and fuse their features to provide significant gains. For the LLM part, we adopt our Moxin-7B-Base as the LLM backbone.

Following the hypotheses in [129] and similar work, the DINOv2 features provide features that capture low-level spatial properties of an image, augmenting the higher-level “semantic” properties captured by vision-language contrastive models. Typically, the visual module such as (CLIP and SigLIP) trained with the vision-language contrastive objective leads to much better performance. Furthermore, SigLIP contains internet-sourced images from multiple sources (e.g., sketches, diagrams, animated graphics, etc.) which are not in ImageNet or in the DINOv2 pretraining data. DINOv2 and SigLIP are good complements for the image processing and visual representations.

For the model architecture, we adopt the general architecture used by many recent VLMs, such as LLaVa, Qwen-VL, and PaLI-3 [130, 122, 131]. These architectures use a (pretrained) visual backbone to map an input image to a sequence of patch features that are then projected individually into the embedding space of an LM. Specifically, a VLM takes as input an image and text prompt tokens with arbitrary sequence length. These inputs are then fed to the following components: 1) a visual representation backbone, 2) a vision-language projector, and 3) a language model.

We select the pretraining dataset that are fully open-source, and have been used in prior works. Specifically, we use the LLaVa v1.5 data mixture, which consists of two subsets used for a multi-stage training pipeline. The first subset consists of a 558K sample mixture of examples sourced from various captioning datasets (e.g., Conceptual Captions, LAION [132]), while the second consists of 665K multimodal instruct tuning examples comprised of synthetic data generated in [133], as well as examples from existing vision-language training sets (e.g., GQA, TextCaps, [134]), and notably, a sample of language-only data from ShareGPT [135].

During the training of our VLM, we adopt a single-stage approach that both the projection and LM are trained while the visual representation modules are frozen. We train for two epochs.

## 7 Evaluation

We conducted comprehensive performance comparisons against leading language models of comparable scale, including Mistral-7B [6], LLaMA 2-7B [82], Gemma-7B [67], and Qwen v2-7B [11]. These models were selected based on their demonstrated excellence within the 7B or 8B category and represent diverse development approaches from various research organizations worldwide. To ensure a robust evaluation, we re-run all benchmarks with the same evaluation pipeline for fair comparisons. Specifically, we use lm-evaluation-harness [136], opencompass [137], Qwen-Math [120], and OLMES [73] for evaluation.

Lm-evaluation-harness provides a unified framework to test generative language models on a large number of different evaluation tasks. It supports over 60 standard academic benchmarks for LLMs, with hundreds of subtasks and variants implemented. This framework is versatile as it extends to models implemented through various architectures, including transformers (including quantization via AutoGPTQ [138]), GPT-NeoX [87], and Megatron-DeepSpeed [139], all unified through a flexible, tokenization-agnostic interface. The framework is reliable, as evidenced by serving as the backend for HuggingFace’s popular Open LLM Leaderboard and being utilized by dozens of organizations, including NVIDIA, Cohere, BigScience, BigCode, Nous Research, and Mosaic ML.

To complement, we also employed openCompass. This framework performs an in-depth and holistic assessment of large language models structured around eight fundamental dimensions of language model capabilities: language comprehension, knowledge precision, logical deduction, creative ideation, mathematical problem-solving, programming proficiency, extended text analysis, and intelligent agent engagement.

- 7.1 Evaluation Tasks We evaluate the model performance on various tasks below.

- • AI2 Reasoning Challenge (ARC) [140] - a set of genuine grade-school level, multiple-choice science questions, assembled to encourage research in advanced question-answering. The dataset is partitioned into a Challenge Set (ARC-C) and an Easy Set (ARC-E), where the former contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm.
- • HellaSwag [108] - a test of commonsense natural language inference, which is easy for humans ( 95%) but challenging for SOTA models. It consists of 70,000 multiple-choice questions. Each question presents a scenario followed by four possible outcomes, asking the model to select the most reasonable conclusion.
- • MMLU [141] - a test to measure a text model’s multitask accuracy. The test covers 57 tasks, including elementary mathematics, US history, computer science, law, etc.
- • Winogrande [142] - an adversarial and difficult Winograd benchmark at scale, for commonsense reasoning. It contains 44,000 multiple-choice questions with two options each. It requires the model to choose the appropriate entity word for the pronoun in the descriptive text based on the scenario.
- • PIQA [143] - the task of physical commonsense reasoning and a corresponding benchmark dataset Physical Interaction: Question Answering (PIQA). Physical commonsense knowledge is a major challenge on the road to true AI-completeness, including robots that interact with the world and understand natural language. PIQA focuses on everyday situations with a preference for atypical solutions.

- 7.2 Pre-Training Evaluation

We name the initial model as Moxin-7B-Original, which presents the foundation model before fine-tuning on the training data of the evaluation datasets. After subsequent partial fine-tuning of Moxin-7B-Original on the training data of the evaluation datasets, we developed Moxin-7B-Enhanced, enabling direct assessment of how targeted fine-tuning affects model performance. We release our Moxin-7B-Enhanced model as Moxin-7B-Base model.

#### 7.2.1 Zero-Shot Evaluation

We report the result of base models for zero-shot evaluation in Table 2. The tasks are listed below. After training with the training data of evaluation tasks, our Moxin-7B-Enhanced can achieve superior performance compared with state-of-the-art (SOTA) baselines. This significant increase from the base model demonstrates the effectiveness of our fine-tuning approach. The improved performance is particularly notable on complex reasoning tasks like PIQA, where the score increased from 78.07% to 82.24%, matching or exceeding several leading models. Consequently, our models emerge as an excellent candidate for real-world applications.

- • AI2 Reasoning Challenge (0-shot)

- • AI2 Reasoning Easy (0-shot)
- • HellaSwag (0-shot)
- • PIQA (0-shot)
- • Winogrande (0-shot)

Table 2: Performance comparison for various models in zero-shot evaluation.

|Models|HellaSwag WinoGrade PIQA ARC-E ARC-C<br><br>|Ave|
|---|---|---|
|Mistral - 7B LLaMA 2 - 7B LLaMA 2 - 13B LLaMA 3.1 - 8B Gemma - 7b Qwen v2 - 7B Internlm2.5 - 7b Baichuan2 - 7B Yi-1.5-9B DeepSeek - 7B<br><br>|80.39 73.4 82.15 78.28 52.22<br><br>75.99 69.06 79.11 74.54 46.42<br><br>79.37 72.22 80.52 77.4 49.06 78.92 74.19 81.12 81.06 53.67<br><br>80.45 73.72 80.9 79.97 54.1<br><br><br>78.9 72.38 79.98 74.71 50.09<br><br>79.14 77.9 80.52 76.16 51.37 72.25 67.17 77.26 72.98 42.15 77.86 73.01 80.74 79.04 55.03<br><br><br>76.13 69.77 79.76 71.04 44.8<br><br><br>|73.29 69.02 71.71 73.79 73.83 71.21 73.02 66.36 73.14 68.3|
|Moxin - 7B - Original Moxin - 7B - Enhanced|72.06 66.31 78.07 71.47 48.15 80.03 75.17 82.24 81.12 58.64<br><br>|67.21 75.44|

- 7.2.2 Few-Shot Evaluation

Table 3 presents our zero-shot evaluation results across multiple benchmark tasks. The tasks and their few-show settings are listed below. Thanks to its rigorous and high-quality training corpus, our model demonstrates a remarkable competitive edge in tasks that involve language understanding and knowledge application. Our Moxin-7B-Original achieves superior performance than LLaMA2-7B in this scenario. After training with the training data of evaluation tasks, our Moxin-7B-Enhanced can achieve competitive performance compared with SOTA baselines.

Consequently, our models emerge as an excellent choice for a multitude of real-world applications where the reliance on robust language comprehension and extensive knowledge is paramount.

- • AI2 Reasoning Challenge (25-shot)
- • HellaSwag (10-shot)
- • MMLU (5-shot)
- • Winogrande (5-shot)

### Table 3: Performance comparison for various models in few-shot evaluation.

|Model<br><br>|ARC-C Hellaswag MMLU WinoGrade<br><br>|Ave|
|---|---|---|
|Mistral - 7B LLaMA 3.1 - 8B LLaMA 3 - 8B LLaMA 2 - 7B Qwen 2 - 7B Gemma - 7B Internlm2.5 - 7B Baichuan2 - 7B Yi-1.5-9B|57.59 83.25 62.42 78.77<br><br>54.61 81.95 65.16 77.35<br><br>55.46 82.09 65.29 77.82<br><br>49.74 78.94 45.89 74.27<br><br>57.68 80.76 70.42 77.43<br><br>56.48 82.31 63.02 78.3 54.78 79.7 68.17 80.9 47.87 73.89 54.13 70.8<br><br>58.36 80.36 69.54 77.53<br><br><br><br><br>|70.51 69.77 70.17 62.21 71.57 70.03 70.89 61.67 71.48|
|Moxin - 7B - Original Moxin - 7B - Enhanced<br><br>|53.75 75.46 59.43 70.32 59.47 83.08 60.97 78.69|64.74 70.55<br><br>|

#### 7.3 Post-Training Evaluation

In our post-training, with Tülu 3, we further fine-tune our base model with SFT and then DPO, leading to our Moxin-7B-SFT model and Moxin-7B-DPO model, respectively. We also adopt the Infinity Instruct dataset [74] to train another version of our instruct model based on our base model, resulting in Moxin-7B-DPO-II. We demonstrate the performance in Table 4 for zero-shot evaluation and Table 5 for few show evaluation. We also use the OLMES framework from Tulu 3 [73] to evaluate our model across a range of tasks and show the results in Table 6. We can observe that our Moxin-7B-DPO model can achieve comparable performance with other SOTA instruct models. We release the Moxin-7B-DPO model as Moxin-7B-Instruct model.

- Table 4: Performance comparison for various models in zero-shot evaluation.

|Models<br><br>|HellaSwag WinoGrade PIQA ARC-E ARC-C<br><br>|Ave|
|---|---|---|
|Mistral 8B Instruct Llama3.1 8B Instruct Qwen2.5 7B Instruct<br><br>|79.08 73.56 82.26 79.88 56.57<br><br>79.21 74.19 80.79 79.71 55.03<br>80.5 71.03 80.47 81.31 55.12<br>|74.27 73.79 73.69<br><br>|
|Moxin - 7B - II Moxin - 7B - SFT Moxin - 7B - DPO<br><br>|79.32 72.93 81.56 80.43 56.91 81.44 73.09 81.07 79.8 54.67<br><br>85.7 73.24 81.56 81.1 58.02<br><br>|74.23 74.01 75.92|

- Table 5: Performance comparison for various models in few-shot evaluation.

|Model<br><br>|ARC-C Hellaswag MMLU WinoGrade<br><br>|Ave|
|---|---|---|
|Mistral 8B Instruct Llama3.1 8B Instruct Qwen2.5 7B Instruct|62.63 80.61 64.16 79.08 60.32 80 68.18 77.27 66.72 81.54 71.3 74.59<br><br>|71.62 71.44 73.54|
|Moxin - 7B - II Moxin - 7B - SFT Moxin - 7B - DPO<br><br>|61.35 82.1 62.95 77.98 60.11 83.43 60.56 77.56 64.76 87.19 58.36 76.32|71.095 70.42 71.66<br><br>|

Table 6: Performance comparison for various models in olmes evaluation.

|Models/Datasets<br><br>|GSM8K MATH Humaneval<br><br>Humaneval plus<br><br>MMLU PopQA BBH TruthfulQA|Ave|
|---|---|---|
|Qwen2.5 7B Instruct Gemma2 9B Instruct<br><br>|83.8 14.8 93.1 89.7 76.6 18.1 21.7 63.1 79.7 29.8 71.7 67 74.6 28.3 2.5 61.4|57.61 51.88<br><br>|
|Moxin - 7B - II Moxin - 7B - DPO<br><br>|71.04 21 78.21 72.35 63.27 27.98 44.33 56.22 81.19 36.42 82.86 77.18 60.85 23.85 57.44 55.27|54.42 59.38<br><br>|

#### 7.4 CoT Evaluation

Based on our instruct model, we further finetune the model with reasoning data based on SFT and then train using reinforcement learning with GRPO following DeepSeek R1. Starting from the same model after SFT with the OpenThoughts and Open-R1-Math-220k datasets, we use two different RL frameworks, i.e., DeepScaleR [78] and AReal [79], to develop two versions of our reasoning model, respectively, resulting in Moxin-7B-RL-DeepScaleR and Moxin-7B-RL-AReal. We evaluate our reasoning models on competition-level mathematics benchmarks, including AMC 2023, MATH500, Minerva Math, and OlympiadBench. We report the Pass@1 accuracy. We use Qwen Math’s codebase for evaluation following [120, 121] and show the comparison in Table 7. We can observe that the Moxin-7B-RL-DeepScaleR model can achieve outstanding performance compared with baselines, demonstrating the effectiveness of RL for small LLMs such as 7B models. Our Reasoning model trained with DeepScaleR performs better than that trained with AReal. Our reasoning model Moxin-7B-RL-DeepScaleR trained with DeepScaleR [78] is released as Moxin-7B-Reasoning model.

Table 7: Performance comparison for various models on reasoning evaluation.

|Models/Datasets<br><br>|MATH 500 AMC Minerva Math OlympiadBench<br><br>|Ave|
|---|---|---|
|Qwen2.5-Math-7B-Base Qwen2.5-Math-7B-Base + 8K MATH SFT Llama-3.1-70B-Instruct<br><br>|52.4 52.5 12.9 16.4 54.6 22.5 32.7 19.6 64.6 30.1 35.3 31.9<br><br>|33.55 32.35 40.48|
|Moxin-7B-RL-AReal Moxin-7B-RL-DeepScaleR<br><br>|68.6 50 16.9 31.7 68 57.5 16.9 30.4|41.8 43.2<br><br>|

#### 7.5 VLM Evaluation

We adopt the evaluation suit in Prismatic VLMs [126] to evaluate the VLM performance. It focuses on evaluations with well-defined metrics, spanning the following three areas:

Open-Ended Visual Question Answering. We evaluate on VizWiz [144] and GQA [134]. VizWiz assess general visual reasoning; VizWiz also contains a series of unanswerable questions. GQA evaluates spatial reasoning.

Localization. Part of the pretraining data mixture contains examples of predicting normalized bounding box coordinates given referring expressions in language. As such, we evaluate bounding box prediction accuracy on RefCOCO, RefCOCO+, and RefCOCOg [145], and on OCID-Ref [146]. RefCOCO focuses on short descriptions with spatial anchors, RefCOCO+ on strictly appearance based descriptions, and RefCOCOg on long, rich descriptions; OCID-Ref is a robotics dataset probing out-of-distribution generalization, with a focus on localizing objects in clutter.

Challenge Sets (Closed-Set Prediction). We evaluate on Visual Spatial Reasoning [130], TallyQA [147], and POPE [148]. VSR consists of challenging True/False questions about individual spatial relationships in diverse scenes; this is an especially challenging task, with most existing models failing to outperform the majority class baseline. TallyQA consists of questions that assess a VLM’s ability to count objects described in language, with expressions that range in complexity. POPE consists of targeted Yes/No questions that assess a VLM’s propensity to hallucinate.

We use the validation sets for all benchmarks except GQA (where use the recommended the test-dev split), VSR (where we use the zero-shot test split), and POPE (where there is only a single evaluation split).

The comparisons with other VLMs are demonstrated in Table 8. We compare with LLaVa v1.5 7B. Furthermore, we adopt the same VLM framework and change the LLM backbone to other LLMs such as Llama2 7B and Mistral 7B. Then we train these VLMs and compare with our VLM based on Moxin-7B. We can observe that our model outperforms all other VLM baselines.

Table 8: Performance comparison for various VLM models.

| |GQA VizWiz RefCOCO+ OCID-Ref VSR POPE TallyQA|Ave.<br><br>|
|---|---|---|
|LLaVa v1.5 7B (Base) Llama-2 Chat 7B Mistral v0.1 7B Mistral Instruct v0.1 7B Llama-2 7B<br><br>|61.58 54.25 49.47 35.07 51.47 86.57 62.06<br><br>62.11 56.39 58.5 46.3 61.8 86.8 58.1<br><br>63.3 55.32 65.1 48.8 58.5 87.1 61.7 62.71 54.35 64.9 48 57.8 87.5 64.5 62.44 55.98 59.47 43.89 63.67 86.74 59.22<br><br><br>|57.21 61.43 62.83 62.82 61.63|
|Ours<br><br>|64.88 54.08 71.3 48.4 60.8 87.3 66|64.68|

#### 7.6 Moxin Family

We demonstrate a list of our models in the paper in Table 9. Multiple models are developed in our paper, and we show their names in our releases.

Table 9: Our developed models and their names in our releases.

|Developed Models<br><br>|Names in Releases|
|---|---|
|Moxin-7B-Enhanced|Moxin-7B-LLM<br><br>|
|Moxin-7B-SFT| |
|Moxin-7B-DPO|Moxin-7B-Instruct<br><br>|
|Moxin-7B-DPO-II| |
|Moxin-7B-RL-DeepScaleR<br><br>|Moxin-7B-Reasoning|
|Moxin-7B-RL-AReal| |
|Moxin-7B-VLM|Moxin-7B-VLM|

- 7.7 Generation Examples

We demonstrate some text generation examples from our Moxin Chat 7B and Llama2 Chat 7B in Table 10 and 11. Moxin Chat 7B is developed by finetune the Moxin base model on the Tulu2 dataset. The response of Moxin Chat 7B is more accurate and concise.

| |Example 1|
|---|---|
|Moxin Chat 7B|Can you explain the concept of regularization in machine learning? Regularization is a technique used in machine learning to prevent overfitting by introducing a penalty term in the loss function. The idea behind regularization is to encourage simpler models that are easier to interpret and generalize better to new data. There are two main types of regularization:<br><br>1. L1 regularization: also known as Lasso regularization, this method adds a penalty term to the loss function that is proportional to the absolute value of the coefficients in the model. This encourages sparsity, meaning that some coefficients may be exactly zero, which simplifies the model and makes it easier to interpret.<br>2. L2 regularization: also known as Ridge regularization, this method adds a penalty term to the loss function that is proportional to the square of the coefficients in the model. This encourages shrinkage, meaning that the coefficients are pulled closer to zero, which simplifies the model and reduces the variance of the model estimates.<br><br><br>Regularization can be used in various machine learning models, including linear regression, logistic regression, and neural networks. It is often used in conjunction with other techniques such as crossvalidation to find the optimal hyperparameters that balance the bias-variance tradeoff and prevent overfitting.<br><br>|
|Llama2 Chat 7B|Can you explain the concept of regularization in machine learning? Regularization is a technique used in machine learning to prevent overfitting. It involves adding a penalty term to the loss function to discourage large values of the model’s weights. There are several types of regularization, including:<br><br>1. L1 regularization (Lasso): This adds a penalty term to the loss function that is proportional to the absolute value of the model’s weights. This encourages the model to have smaller weights, which can help prevent overfitting.<br>2. L2 regularization (Ridge): This adds a penalty term to the loss function that is proportional to the square of the model’s weights. This also encourages the model to have smaller weights, but it can lead to some weights being set to zero, which can result in a simpler model.<br>3. Dropout regularization: This is a type of regularization that is applied during training. It involves randomly setting a fraction of the model’s neurons to zero during each training iteration. This helps to prevent overfitting by making the model less reliant on any single neuron. 4. Early stopping: This is a type of regularization that is applied during training. It involves monitoring the validation loss during training and stopping the training process when the validation loss stops improving. This helps to prevent overfitting by stopping the training process before the model has a chance to fit the noise in the training data.<br><br><br>Regularization can be applied to different layers of a neural network, such as the input layer, hidden layers, or output layer. It can also be applied to different parts of a model, such as the weights, biases, or activation functions. The main idea behind regularization is to add a penalty term to the loss function that discourages large values of the model’s weights. This helps to prevent overfitting by encouraging the model to find a simpler solution that generalizes better to new data.<br><br>Regularization can be used in conjunction with other techniques, such as cross-validation and grid search, to improve the performance of a machine learning model. In summary, regularization is a technique used in machine learning to prevent overfitting by adding a penalty term to the loss function. There are several types of regularization, including L1, L2, dropout, and early stopping, and it can be applied to different layers and parts of a model. Regularization helps to find a simpler solution that generalizes better to new data and can be used in conjunction with other techniques to improve the performance of a machine learning model.|

##### Table 10: Generation Examples.

| |Example 2|
|---|---|
|Moxin Chat 7B<br><br>|Can you explain the concept of normalization in machine learning? Normalization is a technique used to scale the data before using it for training a machine learning model. It is used to make the data more uniform, and to prevent it from being skewed by large or small values. There are several ways to normalize data, including min-max scaling, z-score normalization, and logarithmic scaling. In min-max scaling, the values are scaled to a range between 0 and 1, by dividing them by the maximum value in the dataset. In z-score normalization, the values are scaled to have a mean of 0 and a standard deviation of 1, by subtracting the mean from each value and dividing by the standard deviation. In logarithmic scaling, the values are transformed using the logarithm function, which can be useful for normalizing data with a wide range of values. Normalization is an important step in preparing data for machine learning, as it can help to improve the performance of the model by making the data more uniform and easier to analyze. It can also help to prevent overfitting and improve generalization.|
|Llama2 Chat 7B<br><br>|Can you explain the concept of normalization in machine learning? Normalization is a technique used in machine learning to transform input data into a range of values that are suitable for use in a model. The goal of normalization is to ensure that the data is on the same scale, so that the model can learn meaningful relationships between the features. There are several types of normalization techniques, including:<br><br>1. Min-max normalization: This involves scaling the data so that it falls within a specific range, usually between 0 and 1.<br>2. Standardization: This involves subtracting the mean and dividing by the standard deviation of the data, so that the data has a mean of 0 and a standard deviation of 1.<br>3. Z-score normalization: This involves subtracting the mean and dividing by the standard deviation of the data, so that the data has a mean of 0 and a standard deviation of 1.<br>4. Log transformation: This involves taking the logarithm of the data, which can help to reduce the effect of extreme values in the data.<br>5. Square root transformation: This involves taking the square root of the data, which can help to reduce the effect of outliers in the data.<br><br><br>Normalization is an important preprocessing step in machine learning, as it can help to improve the performance of the model by reducing the effect of irrelevant features and improving the interpretability of the results.<br><br>Here are some key points to keep in mind when normalizing data for machine learning:<br><br>1. Normalization can improve the performance of the model by reducing the effect of irrelevant features and improving the interpretability of the results.<br>2. There are several types of normalization techniques, including min-max normalization, standardization, z-score normalization, log transformation, and square root transformation.<br>3. The choice of normalization technique will depend on the nature of the data and the specific requirements of the model.<br>4. Normalization can be applied to both continuous and categorical data.<br>5. It is important to normalize the data before training the model, as normalization can improve the convergence of the model and reduce the risk of overfitting.<br>6. Normalization can also be used to improve the interpretability of the results by reducing the effect of irrelevant features and improving the scalability of the model.<br>7. It is important to carefully evaluate the performance of the normalized data and consider the trade-offs between different normalization techniques.<br>8. Normalization can be applied to different features of the data, such as the input features, the target variable, or the output features.<br>9. Normalization can be used in combination with other preprocessing techniques, such as feature scaling, feature selection, and data augmentation.<br>10. Normalization is a powerful tool for improving the performance of machine learning models, but it is important to carefully evaluate the performance of the normalized data and consider the trade-offs between different normalization techniques.<br>|

- Table 11: Generation Examples.

## 8 Conclusion

The field of Large Language Models has witnessed a significant shift toward open-source development, fostering innovation within the AI community. However, a critical challenge emerges: many purportedly open-source models withhold essential components necessary for full understanding and reproducibility, creating barriers that limit both academic advancement and commercial adoption. This not only hampers scientific progress, but also prevents businesses from fully leveraging these models for innovative applications, ultimately diminishing potential societal benefits and economic value creation. To address these limitations, we introduce Moxin 7B, a fully open-source language model developed in accordance with the Model Openness Framework (MOF), providing comprehensive access to pre-training code, configurations, training and fine-tuning datasets, and all intermediate checkpoints. We further finetune the Moxin Base model with open-source post-training frameworks and instruction/CoT data to obtain Moxin Instruct and Moxin Reasoning models. Experiments show that our models achieve superior performance in various evaluations such as zero-shot, few-shot, and

CoT evaluations. We wish to see more work that establishes new standards for reproducible research in language model development, fostering a more inclusive and economically vibrant AI ecosystem.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Anthropic. The claude 3 model family: Opus, sonnet, haiku. https: //www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/ Model_Card_Claude_3.pdf.
- [3] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [4] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [5] Thomas Prest, Pierre-Alain Fouque, Jeffrey Hoffstein, Paul Kirchner, Vadim Lyubashevsky, Thomas Pornin, Thomas Ricosset, Gregor Seiler, William Whyte, and Zhenfei Zhang. Falcon. Post-Quantum Cryptography Project of NIST, 2020.
- [6] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
- [7] Rishi Bommasani, Kevin Klyman, Shayne Longpre, Sayash Kapoor, Nestor Maslej, Betty Xiong, Daniel Zhang, and Percy Liang. The foundation model transparency index. arXiv preprint arXiv:2310.12941, 2023.
- [8] Sayash Kapoor, Rishi Bommasani, Kevin Klyman, Shayne Longpre, Ashwin Ramaswami, Peter Cihon, Aspen Hopkins, Kevin Bankston, Stella Biderman, Miranda Bogen, et al. On the societal impact of open foundation models. arXiv preprint arXiv:2403.07918, 2024.
- [9] Matt White, Ibrahim Haddad, Cailean Osborne, Ahmed Abdelmonsef, Sachin Varghese, et al. The model openness framework: Promoting completeness and openness for reproducibility, transparency and usability in ai. arXiv preprint arXiv:2403.13784, 2024.
- [10] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [11] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.
- [12] Yuqi Li, Xingyou Lin, Kai Zhang, Chuanguang Yang, Zhongliang Guo, Jianping Gou, and Yanli Li. Fedkd-hybrid: Federated hybrid knowledge distillation for lithography hotspot detection. arXiv preprint arXiv:2501.04066, 2025.
- [13] Yihua Zhang, Yuguang Yao, Parikshit Ram, et al. Advancing model pruning via bi-level optimization. NeurIPS, 2022.
- [14] Yao Lu, Yutao Zhu, Yuqi Li, Dongwei Xu, Yun Lin, Qi Xuan, and Xiaoniu Yang. A generic layer pruning method for signal modulation recognition deep learning models. IEEE TCCN, 2024.
- [15] Jiaming Chu, Yanzhuo Xiang, Yuqi Li, Chuanguang Yang, Zhulin An, and Yongjun Xu. Cross-layer graph knowledge distillation for image recognition. In ICASSP, 2025.
- [16] Zhenglun Kong, Haoyu Ma, Geng Yuan, Mengshu Sun, Yanyue Xie, Peiyan Dong, Xin Meng, Xuan Shen, Hao Tang, Minghai Qin, et al. Peeling the onion: Hierarchical reduction of data redundancy for efficient vision transformer training. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 8360–8368, 2023.
- [17] Pu Zhao, Fei Sun, Xuan Shen, Pinrui Yu, Zhenglun Kong, Yanzhi Wang, and Xue Lin. Pruning foundation models for high accuracy without retraining. In Findings of EMNLP 2024, pages 9681–9694. ACL, November 2024.

- [18] Xuan Shen, Hangyu Zheng, Yifan Gong, Zhenglun Kong, Changdi Yang, Zheng Zhan, Yushu Wu, Xue Lin, Yanzhi Wang, Pu Zhao, and Wei Niu. Sparse learning for state space models on mobile. In The Thirteenth International Conference on Learning Representations, 2025.
- [19] Xuan Shen, Zhao Song, Yufa Zhou, Bo Chen, Jing Liu, Ruiyi Zhang, Ryan A Rossi, Hao Tan, Tong Yu, Xiang Chen, et al. Numerical pruning for efficient autoregressive models. In AAAI, 2025.
- [20] Xuan Shen, Zhao Song, Yufa Zhou, Bo Chen, Yanyu Li, Yifan Gong, Kai Zhang, Hao Tan, Jason Kuen, Henghui Ding, et al. Lazydit: Lazy learning for the acceleration of diffusion transformers. In AAAI, 2025.
- [21] Jun Liu, Zhenglun Kong, Pu Zhao, Changdi Yang, Hao Tang, Xuan Shen, Geng Yuan, Wei Niu, Wenbin Zhang, Xue Lin, Dong Huang, and Yanzhi Wang. Toward adaptive large language models structured pruning via hybrid-grained weight importance assessment. In AAAI, 2025.
- [22] Xuan Shen, Pu Zhao, Yifan Gong, Zhenglun Kong, Zheng Zhan, Yushu Wu, Ming Lin, Chao Wu, Xue Lin, and Yanzhi Wang. Search for efficient large language models. In NeurIPS, 2024.
- [23] Changdi Yang, Pu Zhao, Yanyu Li, et al. Pruning parameterization with bi-level optimization for efficient semantic segmentation on the edge. In CVPR, 2023.
- [24] Yuqi Li, Qingqing Long, Yihang Zhou, Ning Cao, Shuai Liu, Fang Zheng, Zhihong Zhu, Zhiyuan Ning, Meng Xiao, Xuezhi Wang, et al. Comae: Comprehensive attribute exploration for zero-shot hashing. arXiv preprint arXiv:2402.16424, 2024.
- [25] Xuan Shen, Weize Ma, Jing Liu, et al. Quartdepth: Post-training quantization for real-time depth estimation on the edge. In CVPR, 2025.
- [26] Zheng Zhan, Yushu Wu, Yifan Gong, et al. Fast and memory-efficient video diffusion using streamlined inference. In NeurIPS, 2024.
- [27] Yushu Wu, Yifan Gong, Pu Zhao, et al. Compiler-aware neural architecture search for onmobile real-time super-resolution. In ECCV, pages 92–111. Springer, 2022.
- [28] Zheng Zhan, Yifan Gong, Pu Zhao, Geng Yuan, et al. Achieving on-mobile real-time superresolution with neural architecture and pruning search. In ICCV, pages 4821–4831, 2021.
- [29] Yanyu Li, Changdi Yang, Pu Zhao, et al. Towards real-time segmentation on the edge. AAAI’23/IAAI’23/EAAI’23, 2023.
- [30] Yanyu Li, Pu Zhao, Geng Yuan, Xue Lin, Yanzhi Wang, and Xin Chen. Pruning-as-search: Efficient neural architecture search via channel pruning and structural reparameterization. arXiv preprint arXiv:2206.01198, 2022.
- [31] Zheng Zhan, Zhenglun Kong, Yifan Gong, Yushu Wu, Zichong Meng, Hangyu Zheng, Xuan Shen, Stratis Ioannidis, Wei Niu, Pu Zhao, and Yanzhi Wang. Exploring token pruning in vision state space models. In NeurIPS, 2024.
- [32] Zheng Zhan, Yushu Wu, Zhenglun Kong, Changdi Yang, Yifan Gong, Xuan Shen, Xue Lin, Pu Zhao, and Yanzhi Wang. Rethinking token reduction for state space models. In EMNLP, pages 1686–1697, Miami, Florida, USA, nov 2024. ACL.
- [33] Zhenglun Kong, Peiyan Dong, Xiaolong Ma, Xin Meng, Wei Niu, Mengshu Sun, Xuan Shen, Geng Yuan, Bin Ren, Hao Tang, et al. Spvit: Enabling faster vision transformers via latency-aware soft token pruning. In European conference on computer vision, pages 620–640. Springer, 2022.
- [34] Xuan Shen, Yizhou Wang, Xiangxi Shi, Yanzhi Wang, Pu Zhao, and Jiuxiang Gu. Efficient reasoning with hidden thinking. arXiv preprint arXiv:2501.19201, 2025.
- [35] Zhenglun Kong, Zheng Zhan, Shiyue Hou, Yifan Gong, Xin Meng, Pengwei Sui, Peiyan Dong, Xuan Shen, Zifeng Wang, Pu Zhao, et al. Enabling flexible multi-llm integration for scalable knowledge aggregation. arXiv preprint arXiv:2505.23844, 2025.
- [36] Xuan Shen, Weize Ma, Yufa Zhou, Enhao Tang, Yanyue Xie, Zhengang Li, Yifan Gong, Quanyi Wang, Henghui Ding, Yiwei Wang, et al. Fastcar: Cache attentive replay for fast auto-regressive video generation on the edge. arXiv preprint arXiv:2505.14709, 2025.
- [37] Lin Zhao, Yushu Wu, Xinru Jiang, Jianyang Gu, Yanzhi Wang, Xiaolin Xu, Pu Zhao, and Xue Lin. Taming diffusion for dataset distillation with high representativeness. arXiv preprint arXiv:2505.18399, 2025.

- [38] Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, et al. Jamba: A hybrid transformer-mamba language model. arXiv preprint arXiv:2403.19887, 2024.
- [39] Jamba Team, Barak Lenz, Alan Arazi, Amir Bergman, Avshalom Manevich, Barak Peleg, Ben Aviram, Chen Almagor, Clara Fridman, Dan Padnos, et al. Jamba-1.5: Hybrid transformermamba models at scale. arXiv preprint arXiv:2408.12570, 2024.
- [40] Rico Sennrich. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.
- [41] OpenAI Team. tiktoken, 2022.
- [42] T Kudo. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226, 2018.
- [43] Zhilin Yang. Xlnet: Generalized autoregressive pretraining for language understanding. arXiv preprint arXiv:1906.08237, 2019.
- [44] Hugging Face Team. Summary of the tokenizers. 2024.
- [45] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [46] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023.
- [47] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.
- [48] Duzhen Zhang, Yahan Yu, Chenxing Li, Jiahua Dong, Dan Su, Chenhui Chu, and Dong Yu. Mm-llms: Recent advances in multimodal large language models. arXiv preprint arXiv:2401.13601, 2024.
- [49] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.
- [50] Ben Mann, N Ryder, M Subbiah, J Kaplan, P Dhariwal, A Neelakantan, P Shyam, G Sastry, A Askell, S Agarwal, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 1, 2020.
- [51] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116, 2023.
- [52] Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021.
- [53] Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave. Ccnet: Extracting high quality monolingual datasets from web crawl data. arXiv preprint arXiv:1911.00359, 2019.
- [54] L Xue. mt5: A massively multilingual pre-trained text-to-text transformer. arXiv preprint arXiv:2010.11934, 2020.
- [55] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.
- [56] Alexis Conneau and Guillaume Lample. Cross-lingual language model pretraining. Advances in neural information processing systems, 32, 2019.
- [57] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

- [58] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.
- [59] Noveen Sachdeva, Benjamin Coleman, Wang-Cheng Kang, Jianmo Ni, Lichan Hong, Ed H Chi, James Caverlee, Julian McAuley, and Derek Zhiyuan Cheng. How to train data-efficient llms. arXiv preprint arXiv:2402.09668, 2024.
- [60] Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barret Zoph, Denny Zhou, Jason Wei, Kevin Robinson, David Mimno, et al. A pretrainer’s guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity. arXiv preprint arXiv:2305.13169, 2023.
- [61] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-of-experts. In International Conference on Machine Learning, pages 5547–5569. PMLR, 2022.
- [62] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. arXiv preprint arXiv:2107.06499, 2021.
- [63] Amit Agarwal, Hema Swetha Koppula, Krishna P Leela, Krishna Prasad Chitrapura, Sachin Garg, Pavan Kumar GM, Chittaranjan Haty, Anirban Roy, and Amit Sasturkar. Url normalization for de-duplication of web pages. In Proceedings of the 18th ACM conference on information and knowledge management, pages 1987–1990, 2009.
- [64] Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, et al. Datacomp-lm: In search of the next generation of training sets for language models. arXiv preprint arXiv:2406.11794, 2024.
- [65] Alon Albalak, Liangming Pan, Colin Raffel, and William Yang Wang. Efficient online data mixing for language model pre-training. In R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models, 2023.
- [66] Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Zhengzhong Liu, Hongyi Wang, Bowen Tan, Joel Hestness, Natalia Vassilieva, Daria Soboleva, et al. Slimpajama-dc: Understanding data combinations for llm training. arXiv preprint arXiv:2309.10818, 2023.
- [67] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.
- [68] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1– 113, 2023.
- [69] Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al. Dolma: An open corpus of three trillion tokens for language model pretraining research. arXiv preprint arXiv:2402.00159, 2024.
- [70] Guilherme Penedo, Hynek Kydlíˇcek, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. arXiv preprint arXiv:2406.17557, 2024.
- [71] Malte Ostendorff, Pedro Ortiz Suarez, Lucas Fonseca Lage, and Georg Rehm. Llm-datasets: An open framework for pretraining datasets of large language models.
- [72] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173, 2024.
- [73] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. T\" ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124,

- 2024.

- [74] BAAI Team. Infinity Instruct. https://huggingface.co/datasets/BAAI/Infinity-Instruct, January

- 2025.

- [75] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [76] OpenThoughts Team. Open Thoughts. https://open-thoughts.ai, January 2025.
- [77] OpenR1 Team. OpenR1-Math-220k. https://huggingface.co/datasets/open-r1/OpenR1-Math220k, January 2025.
- [78] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Previewwith-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2, 2025. Notion Blog.
- [79] Ant Research RL Lab. Areal: Ant reasoning rl. https://github.com/inclusionAI/AReaL, 2025.
- [80] Zhiyu Mei, Wei Fu, Kaiwei Li, Guangju Wang, Huanchen Zhang, and Yi Wu. Realhf: Optimized rlhf training for large language models through parameter reallocation. arXiv preprint arXiv:2406.14088, 2024.
- [81] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, et al. Gemini: A family of highly capable multimodal models, 2024.
- [82] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [83] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [84] Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023.
- [85] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.
- [86] Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR, 2023.
- [87] Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. Gpt-neox-20b: An open-source autoregressive language model. arXiv preprint arXiv:2204.06745, 2022.
- [88] Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. Olmo: Accelerating the science of language models. arXiv preprint arXiv:2402.00838, 2024.
- [89] Zhengzhong Liu, Aurick Qiao, Willie Neiswanger, Hongyi Wang, Bowen Tan, Tianhua Tao, Junbo Li, Yuqi Wang, Suqi Sun, Omkar Pangarkar, et al. Llm360: Towards fully transparent open-source llms. arXiv preprint arXiv:2312.06550, 2023.
- [90] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.
- [91] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

- [92] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.
- [93] Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. https://cerebras.ai/blog/slimpajama-a-627b-token-cleaned-anddeduplicated-version-of-redpajama, 2023.
- [94] Maurice Weber, Daniel Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, et al. Redpajama: an open dataset for training large language models. arXiv preprint arXiv:2411.12372, 2024.
- [95] Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, and Ari S Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication. arXiv preprint arXiv:2303.09540, 2023.
- [96] Large-scale near-deduplication behind bigcode, 2023.
- [97] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751, 2019.
- [98] J Leskovec, A Rajaraman, and JD Ullman. Mining of massive datasets, cambridge university press, cambridge, 2014.
- [99] Jay M Patel and Jay M Patel. Introduction to common crawl datasets. Getting structured data from the internet: running web crawlers/scrapers on a big data production scale, pages 277–324, 2020.
- [100] Andrei Z Broder. On the resemblance and containment of documents. In Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No. 97TB100171), pages 21–29. IEEE, 1997.
- [101] Fineweb, 2024.
- [102] David Brandfonbrener, Hanlin Zhang, Andreas Kirsch, Jonathan Richard Schwarz, and Sham Kakade. Color-filter: Conditional loss reduction filtering for targeted language model pretraining. arXiv preprint arXiv:2406.10670, 2024.
- [103] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023.
- [104] Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, et al. The stack: 3 tb of permissively licensed source code. arXiv preprint arXiv:2211.15533, 2022.
- [105] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, and Jie Tang. GLM-130b: An open bilingual pre-trained model. In The Eleventh International Conference on Learning Representations, 2023.
- [106] Jupinder Parmar, Shrimai Prabhumoye, Joseph Jennings, Mostofa Patwary, Sandeep Subramanian, Dan Su, Chen Zhu, Deepak Narayanan, Aastha Jhunjhunwala, Ayush Dattagupta, et al. Nemotron-4 15b technical report. arXiv preprint arXiv:2402.16819, 2024.
- [107] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021.
- [108] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.
- [109] Shenggui Li, Hongxin Liu, Zhengda Bian, Jiarui Fang, Haichen Huang, Yuliang Liu, Boxiang Wang, and Yang You. Colossal-ai: A unified deep learning system for large-scale parallel training. In Proceedings of the 52nd International Conference on Parallel Processing, pages 766–775, 2023.
- [110] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [111] Faeze Brahman, Sachin Kumar, Vidhisha Balachandran, Pradeep Dasigi, Valentina Pyatkin, Abhilasha Ravichander, Sarah Wiegreffe, Nouha Dziri, Khyathi Chandu, Jack Hessel, Yulia Tsvetkov, Noah A. Smith, Yejin Choi, and Hannaneh Hajishirzi. The Art of Saying No: Contextual Noncompliance in Language Models. 2024.

- [112] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations.
- [113] Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatGPT interaction logs in the wild. In The Twelfth International Conference on Learning Representations, 2024.
- [114] Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman. Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data. arXiv preprint arXiv:2410.01560, 2024.
- [115] Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [https: //huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/projectnumina/aimo-progress-prize/blob/main/report/numina_dataset.pdf), 2024.
- [116] Zhilin Wang, Yi Dong, Olivier Delalleau, Jiaqi Zeng, Gerald Shen, Daniel Egert, Jimmy J. Zhang, Makesh Narsimhan Sreedhar, and Oleksii Kuchaiev. Helpsteer2: Open-source dataset for training top-performing reward models, 2024.
- [117] David Wadden, Kejian Shi, Jacob Morrison, Aakanksha Naik, Shruti Singh, Nitzan Barzilay, Kyle Lo, Tom Hope, Luca Soldaini, Shannon Zejiang Shen, et al. Sciriff: A resource to enhance language model instruction-following over scientific literature. arXiv preprint arXiv:2406.07835, 2024.
- [118] Peng Li, Yeye He, Dror Yashar, Weiwei Cui, Song Ge, Haidong Zhang, Danielle Rifinski Fainman, Dongmei Zhang, and Surajit Chaudhuri. Table-gpt: Table-tuned gpt for diverse table tasks. arXiv preprint arXiv:2310.09263, 2023.
- [119] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv:2409.19256, 2024.
- [120] hkust-nlp Team. Simple Reinforcement Learning for Reasoning. https://github.com/hkustnlp/simpleRL-reason, January 2025.
- [121] Weihao Zeng, Yuzhen Huang, Wei Liu, Keqing He, Qian Liu, Zejun Ma, and Junxian He. 7b model and 8k examples: Emerging reasoning with reinforcement learning is both effective and efficient. https://hkust-nlp.notion.site/simplerl-reason, 2025. Notion Blog.
- [122] J Bai, S Bai, S Yang, S Wang, S Tan, P Wang, J Lin, C Zhou, and J Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arxiv 2023. arXiv preprint

- arXiv:2308.12966.

[123] Pan Zhang, Xiaoyi Dong, Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Haodong Duan, Songyang Zhang, Shuangrui Ding, et al. Internlm-xcomposer: A visionlanguage large model for advanced text-image comprehension and composition. arXiv preprint

- arXiv:2309.15112, 2023.

- [124] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [125] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.
- [126] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. In International Conference on Machine Learning (ICML), 2024.
- [127] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [128] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

- [129] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19729–19739, 2023.
- [130] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [131] Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, et al. Pali-3 vision language models: Smaller, faster, stronger. arXiv preprint arXiv:2310.09199, 2023.
- [132] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018.
- [133] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.
- [134] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.
- [135] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer, 2024.
- [136] LM Evaluation Harness Team. Lm evaluation harness, 2024. Accessed: Summer 2024.
- [137] Open Compass Team. Open compass, 2024. Accessed: Summer 2024.
- [138] AutoGPTQ Team. Autogptq: An user-friendly llms quantization package, 2024. Accessed: Spring 2024.
- [139] Shuaiwen Leon Song, Bonnie Kruft, Minjia Zhang, Conglong Li, Shiyang Chen, Chengming Zhang, Masahiro Tanaka, Xiaoxia Wu, Jeff Rasley, Ammar Ahmad Awan, et al. Deepspeed4science initiative: Enabling large-scale scientific discovery through sophisticated ai system technologies. arXiv preprint arXiv:2310.04610, 2023.
- [140] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.
- [141] Teun van der Weij, Felix Hofstätter, Ollie Jaffe, Samuel F Brown, and Francis Rhys Ward. Ai sandbagging: Language models can strategically underperform on evaluations. arXiv preprint arXiv:2406.07358, 2024.
- [142] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.
- [143] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language, 2019.
- [144] Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, et al. Vizwiz: nearly real-time answers to visual questions. In Proceedings of the 23nd annual ACM symposium on User interface software and technology, pages 333–342, 2010.
- [145] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014.
- [146] Ke-Jyun Wang, Yun-Hsuan Liu, Hung-Ting Su, Jen-Wei Wang, Yu-Siang Wang, Winston H Hsu, and Wen-Chin Chen. Ocid-ref: A 3d robotic dataset with embodied language for clutter scene grounding. arXiv preprint arXiv:2103.07679, 2021.
- [147] Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 8076–8084, 2019.

##### [148] Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, and Yu Qiao. Query-relevant images jailbreak large multi-modal models. arXiv preprint arXiv:2311.17600, 7:14, 2023.

