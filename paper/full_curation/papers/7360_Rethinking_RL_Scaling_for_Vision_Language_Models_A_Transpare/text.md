# arXiv:2504.02587v2[cs.LG]4Apr2025

## Rethinking RL Scaling for Vision Language Models: A Transparent, From-Scratch Framework and Comprehensive Evaluation Scheme

##### Yan Ma3,5, Steffi Chern5, Xuyang Shen2, Yiran Zhong2∗, Pengfei Liu1,4,5∗

1Shanghai Jiao Tong University (SJTU) 2Minimax 3Fudan University 4SII 5Generative Artificial Intelligence Lab (GAIR)

### Abstract

Reinforcement learning (RL) has recently shown strong potential in improving the reasoning capabilities of large language models and is now being actively extended to vision-language models (VLMs). However, existing RL applications in VLMs often rely on heavily engineered frameworks that hinder reproducibility and accessibility, while lacking standardized evaluation protocols, making it difficult to compare results or interpret training dynamics. This work introduces a transparent, from-scratch framework for RL in VLMs, offering a minimal yet functional four-step pipeline validated across multiple models and datasets. In addition, a standardized evaluation scheme is proposed to assess training dynamics and reflective behaviors. Extensive experiments on visual reasoning tasks uncover key empirical findings: response length is sensitive to random seeds, reflection correlates with output length, and RL consistently outperforms supervised fine-tuning (SFT) in generalization—even with high-quality data. These findings, together with the proposed framework, aim to establish a reproducible baseline and support broader engagement in RL-based VLM research. Code is public and available at: https://github.com/GAIR-NLP/MAYE.

### 1 Introduction

Reinforcement learning (RL) has recently demonstrated remarkable success in enhancing reasoning capabilities of LLMs, particularly on tasks with verifiable answers such as mathematical problem solving (Deepseek, 2025; Chen et al., 2025b). Inspired by this progress, growing efforts have extended RL to VLMs, aiming to replicate the so-called “R1 moment” (Wang et al., 2025; Qwen, 2025). These studies have primarily concentrated on enhancing performance and pushing the state-of-the-art. However, many of these works rely heavily on highly engineered and encapsulated codebases, such as TRL (von Werra et al., 2020), OpenRLHF (Hu et al., 2024), and verl (Sheng et al., 2024), making it difficult for newcomers to understand, replicate, or modify the underlying processes. This has led to a gap in the field, particularly for researchers who are not already deeply familiar with both RL and VLMs. As a result, the learning curve for those entering this area remains steep.

We address this gap by introducing a reproducible standard framework for RL in VLMs, which serves as a transparent and accessible foundation for training RL-based VLMs. Unlike prior works that rely on complex, pre-packaged RL libraries, the proposed framework is implemented entirely from scratch, using only standard libraries such as Transformers (Wolf et al., 2020), FSDP2 (Zhao et al., 2023) for distributed training, and vLLM (Kwon et al., 2023) for inference. This minimal yet functional implementation allows for a clearer understanding of the RL training process and ensures that the core logic is fully transparent, enabling easy customization and experimentation.

∗ Corresponding authors. Email: zhongyiran@gmail.com,pengfei@sjtu.edu.cn.

By building the framework from the ground up, this work provides a solid foundation for further improvements and extensions in RL for VLMs. It also serves as a crucial resource for beginners, offering a simplified entry point to understanding how RL can be applied to VLMs. This framework, while not aiming to be the most performant or highly optimized, acts as an essential entry into the mechanism of RL in VLMs, much like OpenAI’s SpinningUp (Achiam, 2018) for RL, providing significant value to the research community. It can be used both as a base for future RL innovations and as an educational tool for fostering broader engagement with RL-based VLM research.

Besides, while the proposed framework addresses the need for a reproducible RL training process, the evaluation of RL remains a challenging task. Currently, there is no unified or standardized approach to assess RL training in the context of LLMs/VLMs, leaving a significant gap in the field. To address this, a comprehensive evaluation scheme is introduced, offering a structured framework for assessing RL training effectiveness. Unlike instructiontuning (Zhang et al., 2023) or DPO (Rafailov et al., 2023), where a single performance score is often deemed sufficient, RL training involves dynamic, fluctuating performance that is sensitive to several factors such as initialization and random seed variation (Henderson et al., 2018; Andrychowicz et al., 2020). Reporting a single final score can overfit to incidental fluctuations, compromising the reproducibility and generalization of results. The proposed evaluation scheme, detailed in Sec. 4, emphasizes capturing the training dynamics across multiple stages. Key performance metrics include accuracy curves under different generation settings, as well as behavioral indicators such as response length and reflection ratio. By incorporating fine-grained reflective behavior metrics, the scheme ensures a more nuanced and transparent evaluation of RL’s effectiveness.

Based on the proposed framework, RL experiments are conducted on multiple VLMs across diverse visual reasoning datasets. Each experiment is independently repeated to account for training variance and ensure reproducibility—consistent with best practices in the RL community (Colas et al., 2018; Agarwal et al., 2021). By applying the evaluation scheme, several notable findings emerge: response length is highly sensitive to random seeds; reflective behaviors strongly correlate with length dynamics; and RL consistently demonstrates superior generalization compared to SFT, even when the latter is trained with high-quality supervision. These findings are detailed in Sec. 5.

In this work, three core contributions are made: 1) A reproducible and from-scratch RL framework for VLMs. A transparent four-step pipeline is implemented without relying on existing RL toolkits, validated across multiple VLMs and datasets. 2) A standardized evaluation scheme tailored for RL training. The scheme captures training dynamics and reflective behavior, offering robust and reproducible benchmarks for future studies. 3) Empirical insights into length, reflection, and generalization. Analysis reveals the coupling between reflection and response length, and highlights RL’s superior generalization over SFT, even with high-quality supervision.

### 2 Preparation

This section outlines the foundational setup required for RL in VLMs. It includes four parts: data, algorithm, reward function, and model. Together, these elements define the training context and ensure that the subsequent RL process proceeds under a coherent and reproducible configuration.

Data serves as the foundation for training and evaluation. Rule-based RL has demonstrated strong effectiveness in text-based reasoning tasks where answers can be explicitly verified (Deepseek, 2025; Chen et al., 2025b). In this report, we continue to focus on verifiable mathematical reasoning problems to construct training and evaluation queries. To account for the varying granularity of information provided by these two modalities, we categorize visual mathematical reasoning into two subtypes: text-dominant and visiondominant, as illustrated in Fig. 1. In the text-dominant setting, most of the necessary information is in the text, while the image provides additional support. In contrast, the vision-dominant setting requires extracting key information directly from the image.

Table 1: Dataset Statistics, † means that samples are from the MathVerse benchmark.

Dataset Name Training Set Size Validation Set Size Test Set Size Task Data Source

mm_math5k 5000 100 100† Text-dominant THU-KEG/MM_Math geometry3k 2101 300 601 Vision-dominant hiyouga/geometry3k

[Figure 1]

[Figure 2]

For text-dominant tasks, we use the mm_math5k dataset (Sun et al., 2024), while for vision-dominant tasks, we use the geometry3k dataset (Zheng et al., 2025). The partitioning of training, validation, and test sets for both datasets is detailed in Tab. 1. To assess the out-of-distribution generalization of RL in VLMs, we construct the test set for mm_math5k using 100 problems sampled from MathVerse (Zhang et al., 2024). Additionally, to prevent reward hacking, all problems are designed as numerical computation tasks, ensuring that RL-based models focus on reasoning rather than exploiting spurious correlations in reward signals (Kimi et al., 2025).

text-dominant

[Figure 3]

As shown in the figure, OB is the bisector of

AOC, COD = 1/3 BOD, COD = 17°. What is the measure of AOD?

vision-dominant

[Figure 4]

Find

Figure 1: Text-dominant tasks rely on text with visual support; visiondominant tasks rely on visuals with textual support.

Algorithm selection plays a crucial role in RL for VLMs. Policy-based RL, particularly methods that discard value functions, has become the mainstream approach. Among them, Group Relative Policy Optimization (GRPO) (Shao et al., 2024) has been the most widely used in recent research. In this report, we explore an alternative approach, Reinforce++ (Hu, 2025), to investigate its potential as another option for RL in VLMs and assess its effectiveness in VLM training. Following Xie et al. (2025), we also incorporate a KL divergence penalty between the policy and the reference model, which introduces an additional loss term. The modified update objective is given by:

LCLIP(θ) = E[q∼P(q),oq∼π

θold(o|q)] (1) 1 |oq|

|oq|

πθ(oq,t|q, oq,<t) πθold(oq,t|q, oq,<t)

πθ(oq,t|q, oq,<t) πθold(oq,t|q, oq,<t)

Aˆt, clip

,1 − ϵ,1 + ϵ A ˆt − βlossDKL [πθ∥πref]

∑

min

t=1

 

 

|oq|

Where as Aˆt =

γk−t

∑

I(oq,t = [EOS])r(q, oq) Rule-based reward

−βrew DKL πθ(oq,t|q, oq,<t)∥πref(oq,t|q, oq,<t)





k=t

Token-level KL reward

P(q) represents the distribution of queries, and oq denotes the sequence of response tokens. ϵ constrains the probability ratio ππθ(at|st)

θold(at|st) within [1 − ϵ,1 + ϵ]. Aˆt represents the estimated advantage for token t, which plays a crucial role in determining the direction of parameter updates. The discount factor γ ∈ [0,1] is fixed to 1 in our experiments. The identity function I(oq,t = [EOS]) evaluates to 1 when the <EOS> token is reached, and 0 otherwise. DKL follows the k3 formulation (Schulman, 2025), which provides an unbiased estimation. Additionally, βrew is the coefficient for the KL reward, while βloss is the coefficient for KL penalty loss. It is important to note that in the subsequent experiments, we only applied the KL penalty loss while discarding the KL reward by setting βrew to 0. Modifications to the algorithm remain consistent across all experiments.

Reward Function serves as a rule-based signal for guiding the RL training process. A correct final answer receives a reward of +1; otherwise, 0. A secondary language reward penalizes responses containing non-English characters to discourage multilingual drift. Format rewards are deliberately omitted to avoid constraining the model’s output patterns during learning (Zeng et al., 2025).

Model capability determines whether its cognitive abilities, such as verification and reflection, can be effectively activated. We choose Qwen-VL series for two key reasons. First, based on the findings of (Gandhi et al., 2025) and the prevailing choices in the research

[Figure 5]

[Figure 6]

Text Data Vision Data

Step II: Response Collection

Step I: Data Flow

[Figure 7]

[Figure 8]

Processor

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Query Token ids

Response Token ids

Response Text

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Text Input Vision Input

[Figure 13]

[Figure 14]

Step III: Trajectory Collection

[Figure 15]

[Figure 16]

Ref Model Policy Model

Language Reward

Accuracy Reward

(Old) Policy Ref Logprobs Logprobs

Query Response Token ids

Attention Mask

Position ids

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Token-Level KL Reward

chunking

[Figure 25]

[Figure 26]

Scores

[Figure 27]

[Figure 28]

Policy Model

[Figure 29]

(Updated) Policy Logprobs

[Figure 30]

Step IV: Policy Update

[Figure 31]

Reward

Advantage xN

[Figure 32]

[Figure 33]

Ratio

KL-Divergence Loss

[Figure 34]

[Figure 35]

Policy Loss

Figure 2: Overview of MAYE framework. The process is divided into four steps. Each step integrates various components, including text and vision data, policy models, and reward signals.

community, these models have demonstrated strong potential for test-time scaling. Second, they are natively integrated into Transformers (Wolf et al., 2020), making them highly accessible and convenient to use. Therefore, we select Qwen2/2.5-VL-Instruct (Wang et al., 2024; Bai et al., 2025) as our backbone models.

### 3 MAYE Framework: A Transparent, From-Scratch RL Framework for VLM

This section presents the MAYE framework, a transparent, from-scratch RL training pipeline for VLMs, designed as a reproducible and standardized baseline. Rather than introducing yet another training system, the framework distills RL into four components—data flow, response collection, trajectory generation, and policy update—each made explicit and modular.

Setup From a high-level perspective, Hydra (Yadan, 2019) is used to manage experiment configurations, Transformers (Wolf et al., 2020) for modeling VLMs, FSDP2 (Zhao et al., 2023) for distributed training, and vLLM (Kwon et al., 2023) for collecting responses for multimodal queries. Training and inference are conducted on separate GPU devices. Before training begins, the system loads configurations from a YAML file and then initializes the policy and reference models, dataloaders for the training, validation, and test sets, the optimizer, training parameters, the learning rate scheduler, and the vLLM engine.

It is worth noting that VLMs typically consist of a ViT encoder, an MLP connector, and a LLM backend. Thus, selecting which components to freeze or train is crucial. Based on preliminary experiments, training the connector and ViT on several thousands samples does not yield significant performance improvements but slows down training speed. Since the lower layers of the LLM also participate in processing visual inputs (Zhu et al., 2025), there is no concern that the model’s visual capabilities will be left untuned. Therefore, all experiments solely train the LLM backend.

The RL process involves a variety of parameters and configurations, some of which are easily confused due to overlapping terminology. In particular, commonly used terms such as batch, epoch, and step may refer to different concepts depending on context. Tab. 4 provides a concise reference to clarify these definitions. A complete list of training and hyperparameters

is provided in Appx. A. For rollout inference, vLLM (Kwon et al., 2023) is used to accelerate sampling. To keep the implementation simple, we do not introduce Ray (Moritz et al., 2017) for managing training or inference task scheduling. After completing these setup steps, the subsequent implementation follows a four-step iterative process.

- Step I: Data Flow Under a multimodal setting, each query contains both vision and text data. As shown in the top-left of Fig. 2, the query batch is first processed by a processor provided by Transformers. This step converts raw data into model-compatible inputs, consisting of both textual and visual modalities. The textual input includes token ids sequences—where image slots are padded using special tokens such as <image_pad>—along with the corresponding attention masks. The visual input is transformed into pixel values and auxiliary features. Additionally, the query token ids from text input will be used to concatenate with the generated response tokens in Step II.
- Step II: Response Collection This step (top-right of Fig. 2) involves collecting responses to queries, which can be accelerated using the inference engine. First, the sharded parameters are gathered on the CPU and synchronized to the inference engine. Then, the processed inputs from all training GPUs are gathered to the inference device, collecting a response for each query, including both response text and token ids. After inference, the responses are broadcast back to their corresponding GPUs. Since response lengths vary, padding is applied to ensure an aligned length.
- Step III: Trajectory Generation A trajectory can be considered as an essential input for model learning. It is fundamentally a namedtuple that contains both the components required for loss computation and the metrics that need to be recorded.

The center of Fig. 2 illustrates how text_input is updated: the token ids of queries and responses are concatenated, and the corresponding attention_masks and position_ids are recalculated accordingly. These updated inputs are then stored in the trajectory, as they are required to recompute log probabilities during Step IV. Meanwhile, as illustrated in the middle-left of Fig. 2, the (updated) text input and vision input are forwarded through both the policy and reference models to compute log probabilities (logprobs), with the batch being chunked to prevent out-of-memory. It is important to note that only the logprobs of the response are retained, as RL is a post-training procedure. Meanwhile, the center of Fig. 2 depicts how the token ids of queries and responses are concatenated, from which the corresponding attention_masks and position_ids are derived and stored in the trajectory, as they are needed to recompute the logprobs of the updated policy model during Step IV. Another crucial target is calculating multiple rule-based rewards based on the response texts. These rule-based rewards, along with their summed scores, are also stored in the trajectory. Finally, response length, an important factor in evaluating reasoning capability (Deepseek, 2025), is recorded in the trajectory. See Sec. 4 for detailed evaluation metrics.

- Step IV: Policy Update Once trajectories required for updates are prepared, the first is to estimate the token-level KL divergence between current policy and reference model, scaled

by a coefficient βrew as the KL reward. The summed scores, which are then appended to the last valid position (i.e., <EOS>) of the KL reward as total rewards. Next, following the iterative formula in Eq. 1, total rewards are accumulated token by token in a recursive manner to estimate advantages. The policy logprobs are updated during each parameter update. These probabilities are calculated in chunks, with the chunk size potentially differing from that used in Step III. Consequently, the vision input must be re-collected and re-processed, which is key to ensuring the correct flow of visual data throughout the pipeline. The updated policy logprobs, along with the old logprobs stored in trajectories, are used to compute the clipped ratio for policy loss calculation, as shown in Eq. 1. Besides, the KL divergence between the current policy and reference model is then estimated and weighted by a coefficient βloss to compute the KL loss. Finally, the total loss is computed using Eq. 1, and policy parameters are updated. In total, updates are performed N = (batch_size // ppo_batch_size) × ppo_epochs times. At this point, a single iteration of VLM-RL training is completed. The process is then repeated across all four parts while observing key metrics and evaluating performance.

[Figure 36]

Figure 3: Overview of evaluation metrics.

### 4 MAYE Scheme: Tracking Training Dynamics in RL for LLMs/VLMs

Reliable evaluation has long been a challenge in RL research (Agarwal et al., 2021). Despite the growth of RL-based post-training for LLMs/VLMs, a unified and standardized evaluation scheme remains lacking. Here outlines the evaluation scheme used in the experiments, as shown in Fig. 3. It categorizes evaluation metrics into three aspects: Train Set Metrics, Validation/Test Set Metrics, and Reflection Metrics, aiming to establish a more rigorous and reliable assessment scheme.

General settings In RL evaluation, learning curves are commonly used to visualize training dynamics, with the y-axis representing key metrics such as cumulative rewards or accuracy. The x-axis often represents two types of steps: generation steps and gradient steps, with generation steps being preferred for clearer sample efficiency measurement and allow for fairer comparisons, as response generation typically takes longer than gradient updates. Here, for accuracy learning curves, we advocate using epochs as the x-axis label for improved interpretability, facilitating comparisons akin to those in SFT, where progress is tracked over dataset passes.

Additionally, due to the inherent fragility of RL algorithms (Henderson et al., 2018; Andrychowicz et al., 2020), factors such as different random seeds and initialization states can significantly impact training outcomes (Colas et al., 2018). In traditional RL research, multiple runs (e.g., five, ten, or even dozens) are typically conducted, with the mean and error bars reported in learning curves to ensure statistical reliability. In the context of LLMs/VLMs training, to balance computational cost and result stability, the mean learning curve from three independent runs should be reported.

##### 4.1 Training Set Metrics

Accuracy curves Training set accuracy reflects the correctness and effectiveness of both the algorithm and data preparation. Accuracy is recorded cumulatively per batch and logged per epoch. The main purpose is to illustrate training dynamics, while true performance should be assessed on the validation and test sets. A typical training accuracy curve initially rises and then stabilizes. The stabilization phase, or bottleneck period, indicates convergence and helps decide when to halt training. Ideally, evaluation should include accuracy up to the bottleneck period for a comprehensive understanding of training dynamics.

Response length It reflects the model’s output pattern, including its level of detail and reasoning depth, can be shaped by RL training. Empirical results ( Sec. 5.2) show that as responses become longer, models exhibit more reflective behaviors, contributing to improved generalization (Chu et al., 2025). Hence, response length serves as a crucial metric for monitoring the training process.

##### 4.2 Validation & Test Set Metrics

Accuracy curves Evaluation on the validation and test sets is critical for accurately assessing the model’s capability and generalization. Therefore, accurate accuracy measurements are essential, with online evaluation for small datasets and offline evaluation for larger ones.

Three sets of inference parameters are used to provide a comprehensive view of the model’s performance: 1) pass@8, temperature=1.0, top_p=1.0; 2) pass@1, temperature=0.6, top_p=1.0;

###### 3) pass@1, temperature=0.01, top_p=0.001. The first set evaluates the model’s upper bound, while the second and third assess true performance, with the second preventing endless repetitions or incoherent outputs (DeepSeek, 2025), and the third following the VLM benchmark setting (Bai et al., 2023). In practice, longer CoT models benefit from setting 2), while shorter response models are better reflected by setting 3). These three settings ensure a balanced assessment of the model, highlighting both its maximum potential and true capabilities.

Accuracy tabs In addition to using curves to dynamically visualize and compare performance, static numerical tables are required to provide a clear summary of performance changes. Since accuracy fluctuates throughout the training process, both the mean and maximum accuracy over all epochs are reported. These values are averaged across multiple runs to ensure statistical reliability.

##### 4.3 Reflection Metrics

Words count Reflective behavior (or "aha moments") in models signals the effectiveness of RL training. However, the challenge lies in designing a mechanism to observe changes in this behavior over time. Tracking the frequency of reflective words directly measures the model’s reflective reasoning, revealing patterns in self-correction and problem-solving strategies. A curated list of 15 reflective words: [‘“re-check”, “re-evaluate”, “re-examine”, “re-think”, “recheck”, “reevaluate”, “reexamine”, “reevaluation”, “rethink”, “check again”, “think again”, “try again”, “verify”, “wait”, “yet”] is tracked by counting their frequency during each generation_steps, as inspired by Luo et al. (2025) and Xie et al. (2025).

Ratio curves Simply tracking word frequency is insufficient; it is also essential to observe how the proportion of reflective behavior changes and whether it contributes to accuracy improvement. To achieve this, five ratio metrics are designed, and the corresponding formulas are provided in Tab. 2, where N is the number of responses per batch, Nref is the number of responses with reflection words, N+ is the number of correct responses per batch, and Nref+ is the number of correct responses with reflection words. These metrics quantify different aspects of reflection: the overall proportion of reflective responses, their distribution among correct and incorrect answers, and the accuracy differences between responses with and without reflection.

|Ratio Name|Formula<br><br>|
|---|---|
|reflection_ratio reflection_ratio_in_correct_answers reflection_ratio_in_incorrect_answers correct_ratio_in_reflection_texts correct_ratio_in_no_reflection_texts|Nref N Nref+ N+ Nref −Nref+ N−N+ Nref+ Nref N+−Nref+ N−Nref<br><br>|

Table 2: Definition of reflection ratios.

### 5 Experiment

This section presents an evaluation of RL for VLMs, focusing on training and generalization aspects. First, the correctness of the proposed framework is validated by evaluating performance across different VLMs and datasets, including mm_math5k (Sun et al., 2024) and geometry3k (Lu et al., 2021). Performance improvements on validation and test sets are measured, as discussed in Sec. 5.3. Second, key RL training metrics are analyzed according to the scheme in Sec. 4, covering epoch-wise accuracy and insights into the relationship between response length, reflection word ratio, and aha moments. Finally, RL’s generalization ability is assessed, especially in comparison to SFT on high-quality data (see Sec. 5.5).

25

|train|ing_accu|racy| |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

|ref|lection_ra|tio|
|---|---|---|
| | | |
| | | |
| | | |

|ref|lection_ra|tio|
|---|---|---|
| | | |
| | | |
| | | |
| | | |

1000

reflection word: verify

training_accuracy

reflection word: recheck

0.40

0.07

0.25

0.03

20

800

0.38

Accuracy

Accuracy

0.06

0.20

15

Count

Count

600

Ratio

Ratio

0.36

0.02

0.05

10

400

0.34

0.15

0.01

5

0.32

200

0.04

0.10

0.30

0

0

0 50 100 150

0 50 100 150

0 10 20 30

0 50 100 150

0 10 20 30

0 50 100 150

Epochs

Generation Steps

Epochs

Generation Steps

Generation Steps

Generation Steps

450

|resp|onse_len|gth|
|---|---|---|
| | | |
| | | |
| | | |

|correct_rat|io_in_refle|ction_texts|
|---|---|---|
| | | |
| | | |
| | | |
| | | |

|resp|onse_len|gth|
|---|---|---|
| | | |
| | | |
| | | |

|correct_rat|io_in_reflec|tion_texts|
|---|---|---|
| | | |
| | | |
| | | |
| | | |

|reflection|word: re-|evaluate|
|---|---|---|
| | | |
| | | |
| | | |

reflection word: re-examine

0.3

60

0.4

60

400

650

0.2

Length

Length

Count

Count

0.3

Ratio

Ratio

40

40

350

600

0.1

0.2

20

20

300

0.0

0.1

550

0

0

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

Generation Steps

Generation Steps

Generation Steps

Generation Steps

Generation Steps

Generation Steps

(a) Qwen2-VL-Instruct-7B@mm_math5k

###### (b) Qwen2.5-VL-Instruct-7B@mm_math5k

0.07

|reflectio|n_ratio|
|---|---|
| | |
| | |
| | |
| | |
| | |

|reflection wor|d: re-examine|
|---|---|
| | |
| | |
| | |

| | |
|---|---|
|reflectio|n_ratio|
| | |
| | |
| | |
| | |

|reflection w|ord: verify|
|---|---|
| | |
| | |
| | |
| | |
| | |

training_accuracy

training_accuracy

25

0.30

0.40

60

0.020

0.06

20

0.38

0.25

Accuracy

Accuracy

0.015

Count

Count

40

Ratio

Ratio

15

0.05

0.36

0.20

0.010

10

0.04

20

0.34

0.005

0.15

5

0.03

0.000

0.32

0.10

0

0

0 50 100

0 50 100

0 20 40

0 50 100

0 20 40

0 50 100

Epochs

Generation Steps

Epochs

Generation Steps

Generation Steps

Generation Steps

150

50

275

| | |
|---|---|
|response|_length|
| | |
| | |
| | |
| | |

|correct_ratio_in_|reflection_texts|
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
|reflection wor|d: re-evaluate|
| | |
| | |
| | |
| | |

|response|_length|
|---|---|
| | |
| | |
| | |
| | |
| | |

|correct_ratio_in_|reflection_texts|
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
|reflection word|: re-evaluate|
| | |
| | |

400

0.4

0.3

40

250

380

100

0.3

0.2

Length

Length

30

Count

Count

Ratio

Ratio

360

225

0.2

0.1

20

340

200

50

10

320

0.0

0.1

175

0

0

0 50 100

0 50 100

0 50 100

0 50 100

0 50 100

0 50 100

Generation Steps

Generation Steps

Generation Steps

Generation Steps

Generation Steps

Generation Steps

(c) Qwen2-VL-Instruct-7B@geometry3k

(d) Qwen2.5-VL-Instruct-7B@geometry3k

- Figure 4: Training set metrics across models and datasets. Red curves show training accuracy (per epoch) and response length (per generation step). Blue curves depict key reflection ratios from Sec. 4, and green curves illustrate the usage trends of the two most frequent and dynamic reflection words per experiment. Shaded regions represent standard deviation across three runs.

##### 5.1 Setup

Settings In this work, only the LLM backend of VLM is trained, with the ViT encoder and connector frozen. For answer pattern extraction, the model is instructed to reason step by step, and the final answer is enclosed in \boxed. Only accuracy and language rewards are applied, omitting format and token-level KL rewards. Format reward is easily learned and may limit exploration space Zeng et al. (2025). Token-level KL rewards are excluded to avoid reference model influence on advantage estimation, as recommended in Xie et al. (2025). All experiments are conducted independently three times to ensure robustness, with the average of each evaluation metric reported across runs.

Parameters The learning rate is set to 5.0 × 10−6 with a warmup and cosine decay scheduler. Batch_size is 128, and forward_batch_size is 16. Training is conducted for 1 ppo_epochs and batch is divided into 32 minibatches, resulting in 32 off-policy updates per batch. Generation settings include temperature and top_p both set to 1.0 and max length 2048 tokens. All experiments are run on 8×H800 GPUs, with 7 allocated for training and 1 for inference. The total batch size for response collection is 896. The same hyperparameter settings are shared across experiments. mm_math5k is trained for 30 epochs, corresponding to 150 generation steps, while geometry3k is trained for 50 epochs, resulting in 100 generation steps.

##### 5.2 Training Set Results and Analysis

Fig. 4 presents key training metrics across four experimental settings. The red lines represent the epoch-wise accuracy on the training set (top-left) and the response length trend over generation steps (bottom-left). Training accuracy consistently increases, indicating that RL optimization is functioning as expected. Response length serves as a useful diagnostic signal, reflecting the model’s generation pattern and output richness. Its variation is influenced

valid pass@8|temp=1.0|top_p=1.0

valid pass@1|temp=0.6|top_p=1.0

valid pass@1|temp=0.01|top_p=0.001

valid pass@8|temp=1.0|top_p=1.0

valid pass@1|temp=0.6|top_p=1.0

valid pass@1|temp=0.01|top_p=0.001

|SFT Vanilla| | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

0.55

RL

RL SFT Vanilla

0.70

0.30

0.45

0.30

0.45

0.50

0.40

Accuracy

Accuracy

0.40

0.25

0.25

0.65

0.35

0.45

0.35

0.20

0.20

0.30

0.40

0.30

0.60

0.15

0.25

0.15

0.35

0.25

0.20

0.10

0 10 20 30

0 10 20 30

0 10 20 30

0 10 20 30

0 10 20 30

0 10 20 30

test pass@8|temp=1.0|top_p=1.0

test pass@1|temp=0.6|top_p=1.0

test pass@1|temp=0.01|top_p=0.001

test pass@8|temp=1.0|top_p=1.0

test pass@1|temp=0.6|top_p=1.0

test pass@1|temp=0.01|top_p=0.001

0.35

0.20

0.40

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

0.20

0.60

0.40

0.35

0.30

0.55

Accuracy

Accuracy

0.15

0.30

0.35

0.50

0.25

0.15

0.25

0.45

0.20

0.30

0.10

0.20

0.40

0.10

0.15

0.15

0.35

0.25

0.10

0.05

0.10

0.30

0.05

0 10 20 30

0 10 20 30

0 10 20 30

0 10 20 30

0 10 20 30

0 10 20 30

Epoch

Epoch

Epoch

Epoch

Epoch

Epoch

(a) Qwen2-VL-Instruct-7B@mm_math5k

###### (b) Qwen2.5-VL-Instruct-7B@mm_math5k

valid pass@8|temp=1.0|top_p=1.0

valid pass@1|temp=0.6|top_p=1.0

valid pass@1|temp=0.01|top_p=0.001

valid pass@8|temp=1.0|top_p=1.0

valid pass@1|temp=0.6|top_p=1.0

valid pass@1|temp=0.01|top_p=0.001

0.38

|RL Vanilla| | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

|RL Vanilla| | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

0.30

0.38

0.64

0.30

0.55

0.36

Accuracy

Accuracy

0.36

0.62

0.25

0.34

0.25

0.50

0.60

0.34

0.32

0.20

0.58

0.45

0.20

0.32

0.30

0.56

0.15

0 20 40

0 20 40

0 20 40

0 20 40

0 20 40

0 20 40

test pass@8|temp=1.0|top_p=1.0

test pass@1|temp=0.6|top_p=1.0

test pass@1|temp=0.01|top_p=0.001

test pass@8|temp=1.0|top_p=1.0

test pass@1|temp=0.6|top_p=1.0

test pass@1|temp=0.01|top_p=0.001

0.72

0.35

0.35

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

0.44

0.44

0.60

0.30

Accuracy

Accuracy

0.70

0.42

0.30

0.55

0.42

0.25

0.40

0.68

0.50

0.40

0.25

0.38

0.20

0.45

0.66

0 20 40

0 20 40

0 20 40

0 20 40

0 20 40

0 20 40

Epoch

Epoch

Epoch

Epoch

Epoch

Epoch

(c) Qwen2-VL-Instruct-7B@geometry3k

(d) Qwen2.5-VL-Instruct-7B@geometry3k

- Figure 5: Validation and test accuracy curves across training epochs for different VLMs and datasets. Red lines denote RL, blue lines denote SFT (see Sec. 5.5), and green indicate untrained (Vanilla) performance. All curves are averaged over 3 runs, with shaded areas indicating standard deviation.

by model architecture (see Figs. 4a and 4b), data distribution (see Figs. 4b and 4d), and even random seed (see the widening shaded area in late training stages). Notably, a steady increase in response length is observed in Qwen2.5-VL-Instruct-7B trained on mm_math5k, suggesting that the model adopts a more elaborate reasoning style as training progresses.

##### 5.3 Reflection Metrics and Analysis

Fig. 4 presents key statistics on reflective behavior during training. The blue curves show reflection_ratio and correct_ratio_in_reflection_texts, which capture how often reflection appears and whether it aids in correct reasoning. A full overview of all five ratios is in Fig. 6. The green curves show two representative reflection words per experiment, selected based on frequency and variation. Full trends are in Figs. 7 and 8. Qwen2.5-VL consistently shows higher reflection and correct-in-reflection ratios than Qwen2-VL, suggesting reflective reasoning may be embedded in its pretraining corpus. Still, reflection remains a minority behavior, and performance gains are primarily driven by improvements in non-reflective reasoning. A key analytical focus is the relationship between response length, reflection_ratio, and specific reflection words. Across all experiments, reflection ratio strongly correlates with response length, suggesting reflection contributes significantly to output length variation. However, length and reflection variation do not always track accuracy. In (a) and (c), length decreases while accuracy improves; in (b), reflection ratio rises but correct reflection ratio remains stable (20–30%). In Qwen2-VL, verify spikes early then fluctuates; in Qwen2.5-VL, richer expressions like re-evaluate and re-examine rise steadily, suggesting stylistic and behavioral differences. In summary, while reflection and length reveal aspects of reasoning, performance remains the ultimate indicator.

“Aha Moments” An "aha moment" refers to the model’s ability to identify and correct its own reasoning errors during rollout (Deepseek, 2025). As illustrated in Appx. D, examples

Validation Set Test Set Vanilla SFT (Mean) RL (Mean) SFT (Max) RL (Max) Vanilla SFT (Mean) RL (Mean) SFT (Max) RL (Max)

Generation Config

Dataset Model

pass@8 temp=1.0 36.0 46.8 48.5 53.0 55.7 29.0 25.2 35.2 32.0 42.7 pass@1 temp=0.6 16.0 19.9 23.4 29.0 31.0 11.0 8.7 14.1 12.0 19.7 pass@1 temp=0.01 16.0 18.8 24.1 25.0 31.0 11.0 9.6 14.1 16.0 20.7

Qwen2-VLInstruct-7B

mm_math5k

pass@8 temp=1.0 65.0 60.6 64.8 66.0 70.0 54.0 35.2 54.6 44.0 61.8 pass@1 temp=0.6 36.0 30.8 39.8 39.0 47.2 28.0 13.3 29.0 20.0 35.2 pass@1 temp=0.01 35.0 31.2 40.3 37.0 46.2 28.0 15.0 31.8 20.0 38.3

Qwen2.5-VLInstruct-7B

pass@8 temp=1.0 41.7 - 52.5 - 57.0 46.6 - 58.0 - 62.1 pass@1 temp=0.6 18.3 - 25.2 - 30.3 20.3 - 30.3 - 34.3 pass@1 temp=0.01 18.7 - 26.7 - 31.1 23.1 - 31.1 - 34.1

Qwen2-VLInstruct-7B

geometry3k

pass@8 temp=1.0 63.0 - 60.8 - 64.4 69.4 - 69.3 - 71.8 pass@1 temp=0.6 31.3 - 33.4 - 37.9 38.6 - 41.5 - 44.6 pass@1 temp=0.01 35.3 - 34.6 - 37.7 40.6 - 42.0 - 45.0

Qwen2.5-VLInstruct-7B

Table 3: Mean and maximum accuracy on validation & test sets averaged across 3 runs. RL consistently outperforms the untrained (Vanilla) baseline across all settings. Cell colors indicate relative improvement: deeper red denotes larger gains over Vanilla, while green indicates degradation.

are provided in which different VLMs generate reflective reasoning chains that successfully lead to correct answers. It is important to note that instances of such behavior can already be observed in base models (Liu et al., 2025a). RL training amplifies this behavior, enhancing it rather than creating it from scratch. Even after reflection, minor perceptual errors may persist, indicating that RL could further enhance perceptual grounding to improve overall model capacity. While capturing “aha moments” is valuable, the main focus should be on improvements in validation and test accuracy, as discussed in the next section.

##### 5.4 Validation & Test set Results and Analysis

Fig. 4 shows the accuracy dynamics, with red curves for RL-trained VLMs, blue for SFT (discussed in Sec. 5.5), and green dashed lines for the untrained (Vanilla) model. Each curve shows the mean over 3 independent runs, with shaded regions indicating standard deviation. Tab. 3 summarizes the mean and maximum accuracy for all epochs on the validation and test sets across different generation settings. Color intensity reflects improvement relative to Vanilla: darker red indicates higher gains, while green represents underperformance.

Notable performance improvements are observed on both validation and test sets. RL consistently yields significant gains across all generation settings. On mm_math5k, RL achieves a 1.35× average increase in accuracy, peaking at 1.76×. Similarly, on geometry3k, RL brings an average gain of 1.36×, with a maximum of 1.51×. Even for Qwen2.5-VL-Instruct7B, already among the strongest VLMs of its size, RL continues to enhance generalization, improving pass@1 test accuracy on mm_math5k by 3.5%, with a peak gain of 10%. For geometry3k, RL improves by 1.4%, up to 4.8%. These results demonstrate that RL can effectively enhance both in-distribution and out-of-distribution performance of strong vision-language models, even when baseline capabilities are already very high.

##### 5.5 Generalization on visual mathematical tasks: RL versus SFT

Since the mm_math dataset (Sun et al., 2024) provides CoT solutions from textbooks, these high-quality responses can serve as supervision signals. A key objective is to compare the generalization ability of RL and SFT, a topic of ongoing debate in the research community (Chu et al., 2025; Ye et al., 2025). SFT is performed on Qwen2/2.5-VL-Instruct-7B for the same number of epochs as RL, using the mm_math5k dataset with golden CoT solutions. The learning rate follows a warm-up cosine decay schedule with an initial value of 1 × 10−5, and the batch size is set to 16. Performance is evaluated on the validation and test sets after each epoch, as shown in Fig. 5.

Our findings are summarized as follows: 1) RL outperforms SFT across all configurations and models, with the gap widening as training progresses. 2) On the test set (OOD queries), SFT occasionally underperforms the untrained baseline, indicating overfitting to the training

distribution. In contrast, RL achieves higher accuracy than both SFT and the baseline, demonstrating stronger generalization.

In summary, the advantages of RL for VLMs are threefold: 1) It does not require high-quality responses, often scarce in multimodal scenarios (Guo et al., 2024). 2) Queries can be reused multiple times, improving sample efficiency. 3) RL maintains strong generalization in vision mathematical tasks, while SFT is limited by poor out-of-distribution performance.

- 6 Related Work

Recent efforts in RL for VLMs focus on enhancing reasoning for visual mathematics (Meng et al., 2025; Huang et al., 2025; Peng et al., 2025; Chen et al., 2025a) and extending RL to broader visual tasks such as grounding, detection, and classification (Liu et al., 2025b; Shen et al., 2025). While these works advance the frontier, this report addresses two foundational gaps: 1) the absence of a concise framework outlining RL training for VLMs, and 2) the lack of a structured evaluation framework tailored for RL training. Unlike feature-rich RL toolkits like TRL (von Werra et al., 2020), verl (Sheng et al., 2024), and OpenRLHF (Hu et al.,

- 2024), which prioritize performance and complexity, our framework offers a minimalist, from-scratch implementation focused on transparency and ease of customization, without competing on performance. Evaluation practices for RL-based LLM/VLM training are still under-standardized, making comparison difficult. This report introduces a unified evaluation scheme with metrics covering both performance and behavioral aspects of RL training. A concurrent effort, SimpleRL-Zoo (Zeng et al., 2025), also highlights the importance of robust evaluation in LLMs under zero-settings. Compared to this, this work offers finer-grained analysis of reflective behavior and more comprehensive tracking of accuracy dynamics.

7 Conclusion and Future Work

This work introduces a minimalist and reproducible RL framework for VLMs, built entirely from scratch, alongside a standardized evaluation scheme for tracking performance dynamics and reflective behaviors. Empirical findings offer significant insights into the interplay between reflection, response length, and generalization, showing RL’s superior performance over SFT. In future work, the framework will be further refined for improved usability, simplicity, and extensibility. Leveraging its modular and extensible design, we plan to explore its application to emerging architectures, such as VLMs with linear attention (MiniMax et al.,

- 2025), and even extend RL scaling to fully autoregressive image generation settings (OpenAI, 2025). Meanwhile, the evaluation scheme will be continuously enhanced to provide deeper and more comprehensive insights into model behavior across these diverse scenarios.

References

Joshua Achiam. Spinning Up in Deep Reinforcement Learning, 2018. URL https://github. com/openai/spinningup.

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G Bellemare. Deep reinforcement learning at the edge of the statistical precipice. Advances in Neural Information Processing Systems, 2021.

Marcin Andrychowicz, Anton Raichuk, Piotr Stan´czyk, Manu Orsini, Sertan Girgin, Raphael Marinier, Léonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, et al. What matters in on-policy reinforcement learning? a large-scale empirical study. arXiv preprint arXiv:2006.05990, 2020.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision-language models with less than $3. https://github.com/ Deep-Agent/R1-V, 2025a.

Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wangxiang Che. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models. arXiv preprint arXiv:2503.09567, 2025b.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma. SFT memorizes, RL generalizes: A comparative study of foundation model post-training. CoRR, abs/2501.17161, 2025. doi: 10.48550/ ARXIV.2501.17161. URL https://doi.org/10.48550/arXiv.2501.17161.

Cédric Colas, Olivier Sigaud, and Pierre-Yves Oudeyer. How many random seeds? statistical power analysis in deep reinforcement learning experiments. arXiv preprint arXiv:1806.08295, 2018.

DeepSeek. Deepseek-r1, 2025. URL https://github.com/deepseek-ai/DeepSeek-R1. Deepseek. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement

learning. CoRR, abs/2501.12948, 2025. doi: 10.48550/ARXIV.2501.12948. URL https: //doi.org/10.48550/arXiv.2501.12948.

Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.

Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhu Chen, and Xiang Yue. Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale. CoRR, abs/2412.05237, 2024. doi: 10.48550/ARXIV.2412.05237. URL https://doi.org/10.48550/arXiv.2412.05237.

Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. Deep reinforcement learning that matters. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

Jian Hu. REINFORCE++: A simple and efficient approach for aligning large language models. CoRR, abs/2501.03262, 2025. doi: 10.48550/ARXIV.2501.03262. URL https: //doi.org/10.48550/arXiv.2501.03262.

Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. arXiv preprint arXiv:2405.11143, 2024.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

Team Kimi, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei,

Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599, 2025. doi: 10.48550/ARXIV.2501.12599. URL https://doi.org/10.48550/arXiv.2501.12599.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025a.

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025b.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and SongChun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pp. 6774–6786. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.ACL-LONG.528. URL https://doi.org/10.18653/v1/2021.acl-long.528.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. Notion Blog.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

MiniMax, Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo, Da Chen, Dong Li, Enwei Jiao, Gengxin Li, Guojun Zhang, Haohai Sun, Houze Dong, Jiadai Zhu, Jiaqi Zhuang, Jiayuan Song, Jin Zhu, Jingtao Han, Jingyang Li, Junbin Xie, Junhao Xu, Junjie Yan, Kaishun Zhang, Kecheng Xiao, Kexi Kang, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Zheng, Linbo Chai, Long Xing, Meizhi Ju, Mingyuan Chi, Mozhi Zhang, Peikai Huang, Pengcheng Niu, Pengfei Li, Pengyu Zhao, Qi Yang, Qidi Xu, Qiexiang Wang, Qin Wang, Qiuhui Li, Ruitao Leng, Shengmin Shi, Shuqi Yu, Sichen Li, Songquan Zhu, Tao Huang, Tianrun Liang, Weigao Sun, Weixuan Sun, Weiyu Cheng, Wenkai Li, Xiangjun Song, Xiao Su, Xiaodong Han, Xinjie Zhang, Xinzhu Hou, Xu Min, Xun Zou, Xuyang Shen, Yan Gong, Yingjie Zhu, Yipeng Zhou, Yiran Zhong, Yongyi Hu, Yuanxiang Fan, Yue Yu, Yufeng Yang, Yuhao Li, Yunan Huang, Yunji Li, Yunpeng Huang, Yunzhi Xu, Yuxin Mao, Zehan Li, Zekang Li, Zewei Tao, Zewen Ying, Zhaoyang Cong, Zhen Qin, Zhenhua Fan, Zhihang Yu, Zhuo Jiang, and Zijia Wu. Minimax-01: Scaling foundation models with lightning attention, 2025. URL https://arxiv.org/abs/2501.08313.

Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, William Paul, Michael I. Jordan, and Ion Stoica. Ray: A distributed framework for emerging AI applications. CoRR, abs/1712.05889, 2017. URL http://arxiv.org/abs/ 1712.05889.

OpenAI. Introducing 4o image generation, 2025. URL https://openai.com/index/ introducing-4o-image-generation/.

Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.

Qwen. Qvq-max: Think with evidence, 2025. URL https://qwenlm.github.io/blog/ qvq-max-preview/.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/ 2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html.

John Schulman. Approximating kl divergence. http://joschu.net/blog/kl-approx.html, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024. doi: 10.48550/ARXIV.2402.03300. URL https://doi.org/10.48550/arXiv.2402.03300.

Haozhan Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, and Tiancheng Zhao. Vlm-r1: A stable and generalizable r1-style large vision-language model. https: //github.com/om-ai-lab/VLM-R1, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Kai Sun, Yushi Bai, Ji Qi, Lei Hou, and Juan-Zi Li. MM-MATH: advancing multimodal math evaluation with process evaluation and fine-grained classification. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12-16, 2024, pp. 1358–1375. Association for Computational Linguistics, 2024. URL https://aclanthology.org/2024. findings-emnlp.73.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. Trl: Transformer reinforcement learning. https://github.com/huggingface/trl, 2020.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Yaoting Wang, Shengqiong Wu, Yuecheng Zhang, William Wang, Ziwei Liu, Jiebo Luo, and Hao Fei. Multimodal chain-of-thought reasoning: A comprehensive survey. arXiv preprint arXiv:2503.12605, 2025.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45, Online, October 2020. Association for Computational Linguistics. URL https://www.aclweb.org/ anthology/2020.emnlp-demos.6.

Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning, 2025. URL https://arxiv.org/abs/2502.14768.

Omry Yadan. Hydra - a framework for elegantly configuring complex applications. Github,

2019. URL https://github.com/facebookresearch/hydra.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. LIMO: less is more for reasoning. CoRR, abs/2502.03387, 2025. doi: 10.48550/ARXIV.2502.03387. URL https://doi.org/10.48550/arXiv.2502.03387.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild, 2025. URL https://arxiv.org/abs/2503.18892.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, Peng Gao, and Hongsheng Li. MATHVERSE: does your multi-modal LLM truly see the diagrams in visual math problems? In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (eds.), Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part VIII, volume 15066 of Lecture Notes in Computer Science, pp. 169–186. Springer, 2024. doi: 10.1007/978-3-031-73242-3\_10. URL https:

//doi.org/10.1007/978-3-031-73242-3_10.

Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, et al. Instruction tuning for large language models: A survey. arXiv preprint arXiv:2308.10792, 2023.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. Easyr1: An efficient, scalable, multi-modality rl training framework. https: //github.com/hiyouga/EasyR1, 2025.

Didi Zhu, Yibing Song, Tao Shen, Ziyu Zhao, Jinluan Yang, Min Zhang, and Chao Wu. REMEDY: Recipe merging dynamics in large vision-language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=iX7eHHE5Tx.

### A Hyper-Parameters

- • General training setup: These parameters control the core training loop, including the number of epochs and batch size. batch_size=128; epochs=30(geometry3k), 50(mm_math5k).
- • Model component training configuration: Specifies which parts of the model are trainable. train_vit=False; train_connector=False; train_llm=True
- • Optimization and numerical precision: Sets gradient clipping and computation precision to ensure training stability and efficiency. clip_grad_norm=1.0; dtype=bfloat16
- • PPO-related parameters: Define how policy optimization is performed, including the number of PPO passes, clipping thresholds, and reward normalization. ppo_epochs=1; forward_batch_size=16; ppo_batch_size=4; ppo_backward_batch_size=4; gradient_accumulation_steps=1, epsilon=0.2, gamma=1.0
- • Reward shaping and regularization: These parameters control KL Loss penalties and KL reward modifications to balance exploration and stability. kl_loss_coeff=0.001, kl_reward_coeff=0.0
- • vLLM Inference and sampling configuration: Controls how outputs are generated during training, including sequence length and sampling strategy. max_tokens=2048; top_p=1.0; temperature=1.0; gpu_memory_utilization=0.8

Term Definition batch_size Number of queries per GPU for response collection.

forward_batch_size Number of query-responses processed in a single forward pass to obtain logits. Due to GPU memory constraints, only a subset of the sampled responses can be forwarded at a time.

ppo_batch_size Size of mini-batches into which the sampled queryresponses of batch_size are divided. It allows for a degree of off-policy updates, facilitated by PPO-clip loss.

ppo_backward_batch_size Number of query-responses processed per backward pass. This value is computed as ppo_batch_size // gradient_accumulation_steps.

epochs Number of iterations over RL queries, which is consistent with the concept of data epochs in SFT.

ppo_epochs The number of times a batch of query-response pairs is reused. A higher number of updates indicates a greater degree of off-policy learning. generation_steps Number of generating iterations, where each call to

llm.generate increments the count by one. gradient_steps Number of gradient backward steps, incremented by one with each call to loss.backward.

Table 4: Definitions of Batch and Step-related Terms

### B Reflection Ratios

response_length

reflection_ratio_in_correct_answers

correct_ratio_in_reflection_texts

response_length

reflection_ratio_in_correct_answers

correct_ratio_in_reflection_texts

450

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

0.03

0.3

0.06

0.4

400

650

0.02

0.2

0.3

0.04

350

600

0.01

0.1

0.2

300

0.02

0.0

0.00

0.1

550

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

reflection_ratio

reflection_ratio_in_incorrect_answers

correct_ratio_in_no_reflection_texts

reflection_ratio

reflection_ratio_in_incorrect_answers

correct_ratio_in_no_reflection_texts

0.10

0.45

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

0.30

0.07

0.03

0.03

0.25

0.40

0.08

0.06

0.20

0.02

0.02

0.35

0.06

0.05

0.15

0.01

0.01

0.04

0.30

0.10

0.04

0 50 100 150 Generation Steps

0 50 100 150 Generation Steps

0 50 100 150 Generation Steps

0 50 100 150 Generation Steps

0 50 100 150 Generation Steps

0 50 100 150 Generation Steps

(a) Qwen2-VL-Instruct-7B@mm_math5k

###### (b) Qwen2.5-VL-Instruct-7B@mm_math5k

response_length

reflection_ratio_in_correct_answers

correct_ratio_in_reflection_texts

response_length

reflection_ratio_in_correct_answers

correct_ratio_in_reflection_texts

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

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

275

400

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

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

0.4

0.05

0.020

0.3

380

250

0.04

0.015

0.3

0.2

360

225

0.03

0.010

0.2

0.1

340

0.005

0.02

200

320

0.1

0.000

0.0

0.01

175

0.005

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

reflection_ratio

reflection_ratio_in_incorrect_answers

correct_ratio_in_no_reflection_texts

reflection_ratio

reflection_ratio_in_incorrect_answers

correct_ratio_in_no_reflection_texts

0.07

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
| | | | |

0.425

0.03

0.08

0.30

0.020

0.06

0.400

0.25

0.015

0.02

0.375

0.05

0.06

0.20

0.010

0.350

0.01

0.04

0.005

0.15

0.04

0.325

0.03

0.000

0.00

0.10

0.300

0 25 50 75 100 Generation Steps

0 25 50 75 100 Generation Steps

0 25 50 75 100 Generation Steps

0 25 50 75 100 Generation Steps

0 25 50 75 100 Generation Steps

0 25 50 75 100 Generation Steps

(c) Qwen2-VL-Instruct-7B@geometry3k

(d) Qwen2.5-VL-Instruct-7B@geometry3k

Figure 6: Reflection Ratios

### C Reflection Word Counts

- 0

- 1

- 2

- 3

- 4

AverageCount

re-check

0 50 100 150

0

5

10

15

20

25

30

re-evaluate

0 50 100 150

0

10

20

30

40

50

60

70

re-examine

0 50 100 150

0.0

0.2

0.4

0.6

0.8

re-think

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
|50 100 check_again<br><br>| | |

0 150

0

5

10

15

20

25

recheck

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

0 50 100 150

0

- 5

0 50 100 150

reevaluate

reexamine

reevaluation

rethink

14

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

25

- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7

1.50

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

12

1.25

20

AverageCount

10

1.00

8

15

0.75

6

10

0.50

4

0.25

2

0.00

0

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

think_again

try_again

verify

wait

yet

400

1000

0.05

25

2.5

0.04

800

300

AverageCount

20

2.0

0.03

600

15

1.5

200

0.02

400

10

1.0

0.01

100

200

5

0.5

0.00

0.0

0

0

0

0.01

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

Generation Steps

Generation Steps

Generation Steps

Generation Steps

Generation Steps

###### (a) Qwen2-VL-Instruct-7B@mm_math5k

re-check

re-evaluate

re-examine

re-think

recheck

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

70

25

0.05

1.50

1.50

60

0.04

20

1.25

1.25

AverageCount

50

0.03

1.00

1.00

15

40

0.02

0.75

0.75

30

10

0.01

0.50

0.50

20

5

0.00

0.25

0.25

10

0.00

0.00

0

0

0.01

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

reevaluate

reexamine

reevaluation

rethink

check_again

0.8

0.8

0.8

0.8

1.50

1.25

AverageCount

0.6

0.6

0.6

0.6

1.00

0.4

0.4

0.4

0.4

0.75

0.50

0.2

0.2

0.2

0.2

0.25

0.0

0.0

0.0

0.0

0.00

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

think_again

try_again

verify

wait

yet

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

40

0.8

0.05

0.05

- 0

- 1

- 2

- 3

- 4

0.04

0.04

AverageCount

0.6

30

0.03

0.03

0.4

20

0.02

0.02

0.01

0.01

0.2

10

0.00

0.00

0.0

0

0.01

0.01

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

0 50 100 150

Generation Steps

Generation Steps

Generation Steps

Generation Steps

Generation Steps

(b) Qwen2.5-VL-Instruct-7B@mm_math5k

Figure 7: Reflection Counts

re-check

re-evaluate

re-examine

re-think

recheck

70

50

| | | | |
|---|---|---|---|
| | | | |
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

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.8

15.0

12

60

12.5

40

10

AverageCount

0.6

50

10.0

8

30

40

0.4

7.5

6

30

20

5.0

4

20

0.2

10

2.5

2

10

0.0

0

0.0

0

0

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

reevaluate

reexamine

reevaluation

rethink

check_again

2.5

20

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
| | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

- 0
- 1
- 2
- 3
- 4

14

20

2.0

12

15

AverageCount

10

15

1.5

8

10

10

1.0

6

4

5

5

0.5

2

0.0

0

0

0

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

think_again

try_again

verify

wait

yet

2.5

40

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

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

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.05

120

800

2.0

0.04

100

AverageCount

30

600

0.03

80

1.5

20

0.02

60

400

1.0

0.01

40

10

200

0.5

0.00

20

0.0

0

0

0

0.01

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

Generation Steps

Generation Steps

Generation Steps

Generation Steps

Generation Steps

###### (a) Qwen2-VL-Instruct-7B@geometry3k

re-check

re-evaluate

re-examine

re-think

recheck

50

- 0
- 1
- 2
- 3
- 4
- 5

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
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| |
|---|
| |
| |
| |
| |
| |
| |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.05

140

1.50

40

0.04

120

1.25

AverageCount

100

0.03

1.00

30

80

0.02

0.75

20

60

0.01

0.50

40

10

0.00

0.25

20

0.00

0

0

0.01

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

reevaluate

reexamine

reevaluation

rethink

check_again

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
|25 50 75 verify| | | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

2.0

0.8

0.8

0.05

3.0

0.04

2.5

AverageCount

1.5

0.6

0.6

0.03

2.0

0.4

0.4

1.0

0.02

1.5

0.01

1.0

0.2

0.2

0.5

0.00

0.5

0.0

0.0

0.0

0.0

0.01

0 25 50 75 100

0 25 50 75 100

0 100

0 25 50 75 100

0 25 50 75 100

think_again

try_again

wait

yet

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| |
|---|
| |
| |
| |
| |
| |
| |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.05

0.05

0.05

25

12

0.04

0.04

0.04

10

20

AverageCount

0.03

0.03

0.03

8

15

0.02

0.02

0.02

6

10

0.01

0.01

0.01

4

5

0.00

0.00

0.00

2

0

0

0.01

0.01

0.01

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

Generation Steps

Generation Steps

Generation Steps

Generation Steps

Generation Steps

(b) Qwen2.5-VL-Instruct-7B@geometry3k

Figure 8: Reflection Counts

### D “Aha Moments”

##### Qwen2.5-VL-Instruct 7B@mm_math5k

[Figure 37]

Question: As shown in Figure (1), given point O is the intersection point of lines AB and CD, ∠BOD = 32◦, OE bisects ∠AOD, ∠EOF = 90◦, find the degree measure of ∠COF.

Response: To solve the problem, we need to determine the measure of ∠COF given the information in the diagram and the conditions provided. First, let’s analyze the given information: - ∠BOD = 32◦ - OE bisects ∠AOD ∠EOF = 90◦. Since OE bisects ∠AOD, it means that ∠AOE = ∠DOE. Let’s denote the measure of ∠AOE and ∠DOE as x. Therefore, we have:

∠AOD = ∠AOE + ∠DOE = x + x = 2x

Next, we know that the angles around point O on a straight line sum up to 180◦. Since ∠AOD and ∠BOD are supplementary, we can write:

∠AOD + ∠BOD = 180◦ Substituting the known values:

2x + 32◦ = 180◦ Solving for x:

2x = 180◦ − 32◦ 2x = 148◦ x = 74◦

Thus, ∠AOE = 74◦ and ∠DOE = 74◦. Since ∠EOF = 90◦, we can find ∠COF by subtracting ∠EOF from ∠AOE:

∠COF = ∠AOE − ∠EOF ∠COF = 74◦ − 90◦ ∠COF = −16◦

However, since angles cannot be negative, we need to re-evaluate the problem. The correct interpretation is that ∠COF is the complementary angle to ∠EOF within the context of the problem, which is:

∠COF = 180◦ − ∠EOF − ∠DOE ∠COF = 180◦ − 90◦ − 74◦ ∠COF = 16◦ Thus, the correct measure of ∠COF is:

|16|
|---|

##### Qwen2-VL-Instruct 7B@mm_math5k

[Figure 38]

Question: As shown in the figure, it is known that △ABC is an equilateral triangle, BD is the median of △ABC. Extend BC to E so that CE = CD, and connect DE. What is the measure of ∠BDE?

##### Response: Since

△ABC is an equilateral triangle, we have

AB = BC = AC and ∠ABC = ∠BAC = ∠BCA = 60◦. Since

BD

is the median of △ABC, we have AD = DC. Therefore,

△ABD and

△DCE

are congruent by SAS congruence. This means that ∠ABD = ∠DCE. Since ∠ABC = ∠BAC = 60◦, we have ∠BDE = ∠ABC − ∠ABD = ∠BAC − ∠DCE = 60◦ − 60◦ = 0◦. However, this is not possible since

∠BDE

is an angle. We made a mistake in our reasoning. Let’s try again. Since

△ABD and

△DCE

are congruent, we have ∠ABD = ∠DCE. Therefore, ∠BDE = ∠ABD + ∠BAC = 60◦ + 60◦ = 120◦. So the measure of

∠BDE is

|120◦|
|---|

.

##### Qwen2.5-VL-Instruct 7B@geometry3k

[Figure 39]

Question: In △ABC, CD is an altitude and ∠ACB = 90◦ If AD = 12 and BD = 3, find AC to the nearest tenth.

Response: To solve the problem, we need to find the length of AC in △ABC where CD is an altitude and ∠ACB = 90◦. We are given that AD = 12 and BD = 3. Since D is the foot of the perpendicular from C to AB, CD is perpendicular to AB, making △ACD and △BCD right triangles. We can use the Pythagorean theorem in △ACD and △BCD to find the length of AC. In △ACD, we have:

AC2 = AD2 + CD2 AC2 = 122 + CD2 AC2 = 144 + CD2

In △BCD, we have:

BC2 = BD2 + CD2 BC2 = 32 + CD2 BC2 = 9 + CD2

Since AC = AB, we can set the two expressions for AC2 equal to each other:

144 + CD2 = 9 + CD2 Subtract CD2 from both sides:

144 = 9 This is a contradiction, so we need to re-evaluate the problem. Instead, we use the fact that AC2 = AD · AB because CD is the altitude to the hypotenuse of a right triangle, and the Pythagorean theorem in △ACD and △BCD confirms this relationship. Calculate AB:

- AB = AD + BD = 12 + 3 = 15

Now, using the relationship AC2 = AD · AB: AC2 = 12 · 15 = 180

- AC =

#### √180 =

√36 · 5 = 6√5 ≈ 13.4 Thus, the length of AC to the nearest tenth is:

|13.4|
|---|

Qwen2-VL-Instruct 7B@geometry3k

[Figure 40]

Question: Find sin A.

Response: To find sin A, we first need to determine the length of side AC. We can use the Pythagorean theorem in right triangle ABC, where AB = 25, BC = 5, and AC = 5√26.

AB2 + BC2 = AC2 252 + 52 = (5√26)2 625 + 25 = 250 650 = 250

This equation is not correct, indicating that there might be a mistake in the given lengths or angles. Let’s re-examine the given information and the calculations. Given that AB = 25, BC = 5, and AC = 5√26, we can use the sine function to find sin A. The sine of an angle in a right triangle is the ratio of the length of the side opposite the angle to the length of the hypotenuse.

BC AC

sin A =

5 5√26 sin A =

sin A =

1 √26 sin A =

√26 26

|√26 26<br><br>|
|---|

.

Therefore, the value of sin A is

