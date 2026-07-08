# arXiv:2408.07410v1[cs.CL]14Aug2024

## AQUILA2 TECHNICAL REPORT

Bo-Wen Zhang, Liangdong Wang, Jijie Li, Shuhao Gu, Xinya Wu, Zhengduo Zhang, Boyan Gao, Yulong Ao, Guang Liu∗

Language Foundation Model & Software Team BAAI

### ABSTRACT

This paper introduces the Aquila2 series, which comprises a wide range of bilingual models with parameter sizes of 7, 34, and 70 billion. These models are trained based on an innovative framework named HeuriMentor (HM), which offers real-time insights into model convergence and enhances the training process and data management. The HM System, comprising the Adaptive Training Engine (ATE), Training State Monitor (TSM), and Data Management Unit (DMU), allows for precise monitoring of the model’s training progress and enables efficient optimization of data distribution, thereby enhancing training effectiveness. Extensive evaluations show that the Aquila2 model series performs comparably well on both English and Chinese benchmarks. Specifically, Aquila2-34B demonstrates only a slight decrease in performance when quantized to Int4. Furthermore, we have made our training code2 and model weights3 publicly available to support ongoing research and the development of applications.

Keywords Large Language Model · Efficient Training · Data Centric Framework

### 1 Introduction

Large Language Models (LLMs) exhibit remarkable competence in a wide range of downstream tasks and are catalyzing a fundamental shift in research paradigms [1, 2, 3]. Data play a crucial role in the model training process. Lately, there has been significant focus on investigating the effects of different training data combinations, such as OPT [4], Bloom [5], Palm [6], and LLaMA [7]. These models are typically trained on static datasets over long periods. For instance, LLaMA 65B underwent a 21-day training session in 2048 A100 GPU with 80GB of RAM [8]. Nonetheless, conventional training methods frequently struggle to adapt to variations in data composition or the integration of new data. Due to the resource-intensive nature of each training iteration, enhancing training approaches is crucial for the effective training of LLMs.

This paper introduces the Aquila2 series, which comprises bilingual models ranging in parameter sizes from 7 to 70 billion. The HeuriMentor (HM) Framework is developed to improve the training efficiency of the Aquila series models. The HM System consists of the Adaptive Training Engine (ATE), the Training State Monitor (TSM), and the Data Management Unit (DMU). By integrating these components, the system allows for better monitoring of the model’s training progress and enables efficient adjustments to the data distribution for optimizing training effectiveness. The Adaptive Training Engine (ATE) is designed to train models by continuously updating a mixture of data that includes the most recent data sources, thus enhancing model performance on subsequent tasks. ATE allows for the flexible modification of cluster sizes during training, such as moving from a 12xA100 40G cluster to a 16xA800 80G cluster. Moreover, ATE facilitates training on diverse devices. The Data Management Unit (DMU) is in charge of gathering and organizing data from the Internet and collaborators to use for model training. Initially, the DMU collects information from web pages and PDFs, then proceeds with thorough deduplication and quality filtering. Moreover, the DMU oversees the data combination recipe for each training cycle. We have explored various combinations of these diverse data sources, detailing the outcomes of these experiments to offer empirical insights and key learnings. The Training

∗*Project lead, the corresponding author, liuguang@baai.ac.cn

- 2https://github.com/FlagOpen/FlagScale
- 3https://github.com/FlagAI-Open/Aquila2

Model Layers Hidden FFN Head GQA MaxLen LR Batch Size Aquila2-7B 32 4096 11008 32 - 2048 2e-4 1728 Aquila2-34B 60 6144 24576 48 8 4096 1.5e-4 1024 Aquila2-70B-expr 80 8192 28672 64 8 4096 1.5e-4 1032

Table 1: The training configurations for the Aquila2 series models.

State Monitor (TSM) is designed to evaluate the development of models trained by the Adaptive Training Engine in real time. By tracking metrics such as loss, downstream performance, and model weight changes throughout the training process, we can seamlessly incorporate new data or adjust data combinations as needed. This approach sets up a continuous learning feedback loop that enables the system to improve its performance by leveraging insights from past results.

We conducted a thorough evaluation of the Aquila2 Series using various benchmarks. The extensive experimental findings indicate that Aquila2-34B consistently outperforms the baseline models in terms of average scores. This suggests that the Aquila2-34B model shows improved performance with fewer training tokens compared to LLaMA-270B and other bilingual models tested across 21 diverse datasets. Importantly, we noticed only minimal performance degradation in Aquila2-34B when subjected to 4-bit quantization. Besides, we collected instructional data to train the chat version of the Aquila2 models. The evaluation of AquilaChat2-34B using a bilingual benchmark, which included subjective and objective assessments, consistently illustrates the superior performance of AquilaChat2-34B over LLaMA-2-70B and its corresponding chat models.

### 2 Aquila2 series

Tokenizer. To identify the optimal vocabulary size, preliminary experiments were carried out with varying vocabulary sizes and language proportions. Based on the results, we decided to set Aquila2’s vocabulary size at 100,000. Simultaneously, we adopted Byte Pair Encoding (BPE) [9] to extract the vocabulary with huggingface transformers [10] tool. The training data corpus consists of an equal proportion of English and Chinese, sourced from WudaoCorpus [11] and Pile [12] for Chinese and English data, respectively.

Group Query Attention. We adopt the Grouped Query Attention (GQA) [13] mechanism, which demonstrates an enhanced level of efficiency in comparison to traditional multi-head attention during the inference process. Previous work [8] illustrates that pre-trained GQA models produce quality levels that closely match those of conventional multi-head attention models while maintaining processing speeds that are comparable to those of the MHA models.

Position Embedding. In the architecture of Aquila2, we deploy Rotary Position Embedding (RoPE) [14], a widely recognized and adopted methodology within the paradigm of large language model architectures. The incorporation of RoPE, which ingeniously amalgamates the benefits of both traditional relative and absolute position encoding, aims to enhance the potency of the model by efficiently capturing the underlying spatiotemporal patterns in sequence data.

Other details of the model structure are listed in Table 1.

### 3 HeuriMentor Framework

The HeuriMentor framework focuses on training models effectively with dynamically changing data mixtures. In Figure 1, two main scenarios are illustrated: first, when a new data source is acquired, and second, when the goal is to improve the model’s performance for downstream tasks.

#### 3.1 Adaptive Training Engine

The Adaptive Training Engine (ATE) is designed to train models using an updated mix of data from the latest sources. This training approach aims to enhance the performance of models on subsequent tasks. To ease the need for computational resources, ATE accommodates various model structures and parameter scales, with the training strategy optimized across multiple chips. ATE also allows for flexible adjustment of cluster sizes during training. Importantly, ATE supports training on heterogeneous devices (Aquila2-70B-expr is trained on heterogeneous devices).

In the training of Aquila2-7B, we employ data parallelism and a distributed optimizer to enhance efficiency [15, 16]. For Aquila2-34B, to address its significant GPU memory demands without impacting GPU utilization, we incorporate data and tensor parallelism, coupled with a distributed optimizer and integrated with 1F1B pipeline and sequence parallelism [17, 18, 19]. Remarkably, Aquila2-70B-Expr, trained under a heterogeneous strategy, is an experimental version that

[Figure 1]

Figure 1: The HeuriMentor Framework structure.

operates on a mixed cluster of A100 40G and A800 80G GPUs, employing specifically optimized strategies for the heterogeneous devices. Moreover, Aquila2 utilizes FlashAttion-2 [20] to further increase computational efficiency. This approach results in roughly 666 tokens/sec/GPU during Aquila2-34B’s training.

Mixed precision. We utilize bfloat16 in model training for numerical stability, supplemented by float32 for precisioncritical operations, encompassing word and Rotary position embeddings, attention softmax, and gradient all-reduce.

Hyperparameters. In the preliminary training phase encompassing the initial 8B tokens, the batch size is progressively elevated from 32 to 1024. The AdamW optimizer is employed with the hyperparameters set as follows: β1 = 0.9, β2 = 0.95, eps = 10−8, complemented by a weight decay of 0.1. Our approach incorporates a cosine learning rate schedule and imposes gradient clipping at 1.0. The learning rate undergoes an initial escalation to 1.5 × 10−4 within the first 2B tokens, then decays to 1.5 × 10−5. Dropout techniques were not applied during the training process.

Training hardware. The Aquila2-7B model is pre-trained on 11x8 NVIDIA A800-80GB GPUs, while the Aquila2-34B model is pre-trained on 64x8 NVIDIA A100-40GB GPUs. Both clusters feature an interconnect equipped with 2x200 Gbps InfiniBand. The A800 cluster has an intra-node GPU connection of 400 GB/s, whereas the A100 cluster offers 600 GB/s.

#### 3.2 Training State Monitor (TSM)

Our Training State Monitor is designed to analyze the status of models trained by the Adaptive Training Engine dynamically. During training, we track metrics such as loss, downstream performance, and model weights to monitor the model’s status. This allows us to purposefully incorporate new data or adjust the data mixture as required. This approach introduces a dynamic learning feedback loop, enabling the system to improve its performance by incorporating insights from prior results. If the subsequent Language Model (LLM) fails to surpass its predecessor, adjustments can be made to the training procedures or the data used for training to achieve optimal efficiency. Furthermore, this framework provides a versatile tool that effectively enhances the performance of LLMs across a wide range of tasks, especially those requiring intensive knowledge.

In the initial phase, our Aquila2 models are fine-tuned using a well-researched data distribution method [21, 8]. This stage involves exposing the models to approximately 200-300 billion tokens, which facilitates the model to generate predictions based on a minimal collection of exemplars.

After training the models, the next step is to monitor their performance. This involves recording and assessing any changes in efficiency, detecting patterns, and highlighting deviations. By closely monitoring the models, potential issues can be detected and corrected early.

In the final phase, the data distribution is adjusted based on empirical evidence. Some possible adjustments include increasing the sampling of under-represented classes, readjusting the data distribution, or introducing data changes to elicit diverse responses from the model. The goal of this iterative redistribution process is to maximize the overall performance of the model. Later sections will provide more details on refining this method and explaining its potential applications in a wider context.

[Figure 2]

Figure 2: Training loss for Aquila2-34B and Aquila2-70B-expr Models.

#### 3.2.1 Training loss

The training loss curves of the Aquila2-34B and Aquila2-70B-expr models, as shown in Figure 2, reveal key insights into the impact of strategic data adjustments on model performance. Initially, both models exhibit a steady decline in loss, which gradually slows as the training progresses. This flattening of the loss curve is indicative of the models reaching a phase where further improvements become more challenging, likely due to the models approaching a local minimum or the limits of what can be learned from the current dataset.

Recognizing this trend, adjustments were made to the training process, such as introducing new data, modifying the learning rate, or other forms of data augmentation. These adjustments are reflected in the sharp drops in the loss curves following the plateaus. These interventions effectively reinvigorate the training process, leading to a marked improvement in training efficiency. This is evident from the more pronounced and rapid decreases in loss after each adjustment.

The Aquila2-70B-expr model, in particular, shows a consistently lower loss compared to the Aquila2-34B model, demonstrating its superior capacity to learn and generalize from the data. This can be attributed to its larger parameter space, which allows it to model more complex relationships within the data. The observed patterns highlight the importance of strategically timing these interventions to overcome stagnation in training and to enhance the overall performance of the models.

In the early training stages, we observed unexpected spikes in the training loss. This phenomenon occurred more frequently in Aquila2-70B than in Aquila2-34B and was attributed to a low-quality, noisy dataset in certain language topics. To stabilize training without rolling back to previous checkpoints, we meticulously cleaned the dataset before feeding it into the model. We monitored the norm of the model weights, assessing their performance on downstream tasks to ensure smooth training. As a result, the frequency of these spikes significantly decreased.

A sharp decrease in training loss is observed in Aquila2-34B after approximately 1500 tokens. However, this decrease does not directly indicate an improvement in model performance, as there are no immediate differences in the downstream tasks. Instead, we attribute this to the characteristics of the training data, specifically the knowledgespecific dataset used at that point in the training process.

[Figure 3]

[Figure 4]

(a) (b)

- Figure 3: Performance of Aquila-34B (a) and Aquila-70B-expr (b) on downstream tasks during training. We use different colors to distinguish between different data stages of the training loss. We apply the score of HELM and LM evaluation as the main metrics. Details of evaluation are covered in section 4.1.

[Figure 5]

(a) (b)

[Figure 6]

- Figure 4: Convergence evidence from the perspective of the weight. Different colors of rectangles represent different data stages, corresponding to the stages(K6, K61&K62, K63, and K64) in Aquila2-34B, and the stages(K6, K61, K63, and K65) in Aquila-70B.

#### 3.2.2 Downstream performance

In Figure 3, we observe the training performance of two large language models, Aquila-34B (a) and Aquila-70B-expr (b), as represented by their loss and accuracy metrics over consumed tokens. The loss curves are initially marked by distinct stages of training, represented by different colors, which indicate variations in the data employed at each stage.

For Aquila-34B (a), the loss decreases rapidly during the initial stages, indicating effective learning. However, as training progresses and the model stabilizes, the rate of loss reduction becomes more gradual. At this point, adjustments in the data (as indicated by the color changes) are introduced to address the plateau in performance. This intervention leads to a marked improvement in training efficiency, as evidenced by a sharper decline in the loss and corresponding improvements in accuracy during subsequent stages.

Similarly, in Aquila-70B-expr (b), we observe a comparable trend where the loss reduction slows down after initial rapid learning. The introduction of data adjustments (denoted by color shifts) again leads to significant gains in model performance. The accuracy, which corresponds inversely to the loss, also demonstrates notable improvements post-adjustment, reflecting the model’s enhanced ability to generalize from the modified training data.

#### 3.2.3 Weight Trajectory

Figure. 4 presents the convergence behavior of different weight matrices across training stages for both Aquila-34B (a) and Aquila-70B (b) models. The x-axis represents the number of tokens processed, while the y-axis indicates the mean distance between weights across a billion tokens. The colored rectangles demarcate distinct training stages. Detailed parameter distributions are shown in Appendix.7.8

This section explores the effects of the different batches of pretraining datasets on the convergence of the Aquila2 model. We evaluated this by monitoring the trajectory of model parameter distribution during training, which is visually depicted in Figure. 4 using the standard deviation of each parameter across transformer layers. For this analysis, N is the number of transformer layers and M is the number of checkpoints, with the parameter for the layer k at the

checkpoint i referred to as layerik. We focus on nine essential parameters of layerik,p, where p refers to specific features of the model. The standard deviation of layerik,p is signified as stdk,pi . Given that the average value of each parameter is fixed at 0, our primary metric is the standard deviation as it reflects weight alterations. stdpi represents the standard deviation of parameter p across all Transformer layers at the ith checkpoint.

To evaluate the weight change magnitude between two checkpoints, we calculate the distance between different checkpoints’ trajectories. We consider two successive trajectories, i and i + 1, and calculate the corresponding representations. We use the Fréchet distance [22] to measure the distance between the two trajectories, dist(p,i,i + 1). The Fréchet distance considers the spatial and sequential variance of the curves, making it ideal for capturing differences between trajectories.

This distance is normalized using the difference in training tokens between the trajectories to account for bias and provide a relative measure of change per 1B tokens. Thus, dist(p,i,i + 1) is computed as follows:

F_dist(stdpi ,stdpi+1) (Tokeni+1 − Tokeni)

dist(p,i,i + 1) =

,

where Tokeni denotes the token count for the ith checkpoint. Fig. 4 suggests the following preliminary conclusions:

- • Data Adjustments and Convergence. Despite variations in data stages, the convergence patterns of different weight matrices exhibit remarkable stability. This suggests that the proposed training methodology is robust to changes in data distribution and that data adjustments have a relatively minor impact on the overall convergence behavior of the model.
- • Training Efficiency. The convergence curves demonstrate a clear trend of decreasing mean distance between weights over time, indicating that the models converge rapidly. This is particularly evident in the later stages of training, where the curves flatten out, suggesting that the models have reached a stable state. The accelerated convergence rate can be attributed to the optimized training algorithm and hardware acceleration, leading to significant improvements in training efficiency.
- • Weight Matrix Behavior. The weight matrices associated with self-attention, MLP, and normalization layers exhibit distinct convergence patterns. While the self-attention layers converge relatively quickly, the MLP layers often require more training iterations to stabilize. This observation aligns with previous findings in the literature and highlights the complex interactions between different components of the model.

#### 3.3 Data Management Unit (DMU)

Our pretraining dataset primarily consists of a blend of internet content (Web), encyclopedic information (Wiki), electronic books (textbook), literary works (paper), knowledge-intensive data (Knowledge) from academically supervised data sets or some domain-specific knowledge and coding materials. We have explored various combinations of these diverse data sources, detailing the outcomes of these experiments to offer empirical insights and key learnings. Our goal is to offer valuable perspectives and serve as a source of inspiration for future research in this domain. The appendix. 7.11 contains an in-depth description of the experiments conducted to assess the effects of different data mixture ratios.

Batched knowledge datasets(K6-K65) By the experiences and lessons learned from the preliminary attempts, the final pretraining dataset is made up of three phases. The detailed data sources and processing in these datasets are shown in Table 2.

Table 2: Data sources and processing in dataset K6-K65, : switching to new batches with the same data source, : filtering data with a language model, : sampling data with a target data distribution, : introducing new datasets with the same type, for example, new academic datasets

|Domain<br><br>|K6|K61<br><br>|K62<br><br>|K63<br><br>|K64|K65|
|---|---|---|---|---|---|---|
|Web|Wudaocorpora, Falcon refineweb<br><br>|+<br><br>|+<br><br>|+ +|+ +<br><br>|+ +|
|Wiki|Chinese Wiki, Pile wiki<br><br>|+|+<br><br>|+ +|+ +<br><br>|+ +|
|Paper<br><br>|Chinese Paper, Redpajama arxiv<br><br>|+<br><br>|+|+ +<br><br>|+ +<br><br>|+ +|
|Textbook<br><br>|Redpajama book|Textbook, Teaching assistants, Masterpieces,|+<br><br>|+ +|+ +<br><br>|+ +|
|Code<br><br>|Starcoder Data<br><br>||+|+<br><br>|+<br><br>|Tiny-codes, CodeExercises|
|Knowledge<br><br>|Chinese QA forum, Pile stackexchange<br><br>|Flan 2022, COIG-PC, Academic Datasets<br><br>|Instruction tuninig datasets, +|+<br><br>|QA data, +|+|

- • Language modeling phase (K6) In the first phase, the model engages in language modeling on relatively high-quality data, allowing the model parameters to converge within a reasonably optimal range. Empirically, this data set should exhibit diversity and primarily originate from a significant source, such as web data, to ensure model convergence. Furthermore, segments of relatively high quality are chosen from homogeneous data rather than random sampling. This approach is grounded in curriculum learning, where learning begins with relatively straightforward samples before progressively introducing more challenging ones. This might help prevent the model from getting stuck in local optima and accelerate the convergence process.
- • Knowledge learning phase (K61-K63) In the knowledge learning phase, the goal of the model is to gradually strengthen its acquisition of knowledge content, building upon the foundation established in the language learning phase. To prevent excessive fluctuations in training loss, it is necessary to progressively reduce the proportion of internet-sourced data and increase the proportion of knowledge-intensive data. Additionally, due to factors like cost, non-internet data may be acquired in batches. Consequently, multiple discrete datasets need to be constructed. We developed three datasets: K61, K62, and K63, and adjusted their proportions gradually.
- • Target-oriented phase (K64, K65) The final stage of the model seeks to combine knowledge with various language tasks, enabling it to carry out specific tasks, similar to task-specific fine-tuning. Task-specific data (e.g. academic supervised datasets) are included. The FLAN work [23] inspires augmenting knowledge-oriented data. Knowledge-oriented data are converted into task-specific data (QA pairs with zero-shot, few-shot, option, and no-options templates). We developed K64 for Aquila2-34B and K65 for Aquila2-70B. It is noteworthy that the 70B model opted for a different dataset K65 stemmed from observations of the trial training. The 70B model, due to budgetary constraints, did not process a sufficient number of tokens in the initial two phases. So the drastic changes in data proportions adversely affected the performance. Consequently, a compromise solution was implemented.

In the Aquila2-34B model, the token distributions across the three phases are as follows: 523B for the first phase, 1053B for the second phase, and 298B for the third phase. Similarly, for the Aquila2-70B model, these distributions are 203B, 719B, and 347B respectively. The varying proportions of data sourced from different origins at each stage are comprehensively depicted in Figure. 5.

The content of pretraining data can potentially reveal the fundamental causes of downstream issues such as biases, illusions, and values. Therefore, data security is of paramount importance [8]. While constructing the pretraining dataset, we initially performed source analysis and selection from the Chinese web corpus WuDaoCorpora Text. Only a few thousand trusted sources were chosen as candidates for the data. Subsequently, we defined types of risky data (such as content involving illegal activities, personal privacy data, harmful advice, and advertising/marketing materials) and

[Figure 7]

Figure 5: The proportions of different domains in K6-K65

established a manual annotation process. A security model combining classification and retrieval was trained to filter the pretraining data. Around 3% of the data were further filtered on trusted sources.

The pretraining data was meticulously deduplicated to enhance data usage efficiency. The Aquila-34B model was trained with a total of approximately 1.8 trillion bilingual Chinese and English tokens. Compared with other open-source models such as LLaMA2-70B (2 trillion tokens), Qwen-14B (3 trillion tokens) and InternLM-20B (2.3 trillion tokens), Aquila achieved comparable results with significantly less data. Additionally, we adopted a phased approach to data preparation and training due to the lengthy data collection and training process. We adjusted the proportions of data from different domains in different batches, ensuring that the language model could better capture robust language understanding capabilities in the early stages of training. As training progressed, knowledge-based data continued to increase, allowing the model to capture richer knowledge[23].

Aquila2’s training was conducted on the FlagScale framework [24]. Detailed training configuration can be found in the Appendix 7.10.

### 4 Model evaluation

In this section, we primarily exhibit the performance of our model in comparison with other models across various datasets. Initially, we juxtapose the outcomes on commonly utilized datasets, followed by a more granular analysis of the model’s capabilities in different dimensions based on its specific abilities.

#### 4.1 Overall results

We have curated a set of open-source bilingual (Chinese-English) models, all released before December 2023, that resemble our model for comparative analysis. The primary models considered for comparison are as follows.

- • Baichuan2. [25] Baichuan2 is a series of large-scale multilingual language models, with versions containing 7 billion and 13 billion parameters. They were trained from scratch on a massive dataset of 2.6 trillion tokens.
- • Qwen. [26] Qwen is introduced as a comprehensive series of large language models (LLMs) with varying parameter counts. It includes base pre-trained models and Qwen-Chat models, which are fine-tuned with human alignment techniques for chat applications. The models in the Qwen series, such as Qwen-7B and Qwen-14B, are known for their superior performance across many downstream tasks.
- • LLaMA2. [27] LLaMA2 is a collection of pre-trained and fine-tuned large language models (LLMs) ranging from 7 billion to 70 billion parameters. The fine-tuned versions, called Llama 2-Chat, are optimized for dialogue use cases. It employs an auto-regressive language model architecture and is known for outperforming open-source chat models on most tested benchmarks
- • InternLM. [28] InternLM is a multilingual foundational language model with a massive 104 billion parameters, pre-trained on a large corpus consisting of 1.6 trillion tokens. The model employs a multi-phase progressive training process and is fine-tuned to align with human preferences. InternLM offers variations with different architectures and parameter sizes, such as InternLM-7B and InternLM-20B, with the latter having a deeper architecture of 60 layers compared to conventional models.

Baichuan2-13B Baichuan2-7B InternLM-7B Llama-2-7B

Aquila2-34B Aquila2-70B Aquila2-7B Qwen-14B LLama2-70B InternLM-20B

Qwen-7B

Model

Mean 72.20 66.51 58.66 67.71 67.98 65.99 61.78 61.93 57.21 52.48 46.61 En Mean 68.63 64.35 59.57 64.01 67.15 65.55 60.55 59.12 56.18 52.03 53.57 Zh Mean 76.56 69.14 57.54 72.24 69.29 66.52 63.29 65.36 58.47 53.04 38.10

WMT22 (en-zh) (5-shot) 61.66 59.93 59.98 61.04 - 56.90 60.09 60.50 55.91 53.25 36.39 CLUEWSC(5-shot) 85.93 80.3 49.20 85.60 85.00 84.70 63.30 75.60 63.80 53.90 51.50 winograde (0-shot) 72.85 70.09 67.50 67.40 78.00 75.10 68.59 70.30 68.43 68.19 67.09

HellaSwag (10-shot) 82.51 81.35 71.50 84.00 87.30 82.10 78.37 76.10 73.80 74.35 78.60

OpenBookQA (0-shot) 44.20 43.00 44.80 43.60 48.80 42.60 44.40 43.60 39.40 39.80 40.80 PIQA (0-shot) 74.92 74.16 77.90 81.07 82.80 80.60 78.18 78.80 77.42 73.94 76.82 ARC-e(0-shot) 79.38 74.66 71.50 70.41 81.00 81.10 73.90 73.90 72.90 69.99 53.54

BUSTM(5-shot) 81.18 74.50 74.00 73.20 71.20 59.40 63.30 70.50 67.30 54.70 51.70 BoolQ (0-shot) 88.84 86.30 77.60 86.67 83.70 82.10 68.01 79.10 73.00 43.88 71.13

TruthfulQA(0-shot) 46.99 38.42 44.80 49.50 44.80 51.90 47.85 39.80 37.55 39.34 38.80 RAFT(5-shot) 72.50 69.10 63.20 68.30 75.20 75.20 75.40 71.40 65.50 61.30 68.60 ChID(5-shot) 89.10 85.50 71.40 84.70 66.00 72.40 66.20 74.00 53.00 51.40 14.40

EPRSTMT(5-shot) 90.49 92.80 89.80 91.20 89.30 91.20 91.50 86.60 88.10 88.10 59.00 TNEWS(5-shot) 58.90 40.80 18.30 53.80 51.70 51.20 47.00 44.60 36.80 24.00 17.60 OCNLI(5-shot) 79.92 81.50 57.00 55.00 57.60 62.90 51.00 43.30 37.10 36.90 34.20

SEM-Chinese(5-shot) 79.50 74.29 47.63 77.50 67.20 72.90 61.08 64.90 55.83 51.48 32.45

MMLU (5-shot) 73.74 53.90 58.10 65.80 69.50 61.80 60.16 56.90 54.60 51.20 46.90 CMMLU (5-shot) 73.10 47.42 58.75 70.50 - 59.00 64.20 62.00 57.07 51.80 31.38

CSL(5-shot) 69.19 66.70 48.80 52.60 54.60 51.00 52.90 49.50 48.40 48.70 48.50 HumanEval (0-shot) 39.02 35.40 21.40 32.30 29.90 25.60 22.12 17.10 18.29 13.40 12.80

Table 3: Overall results of the Base model.

Mean Mean Mean En Mean Zh Mean

Model

(Sub. & Obj.) (Obj.) (Sub.) (Obj. & Sub.) (Obj. & Sub.) AquilaChat2-34B 76.34 76.38 75.80 69.92 79.19

Baichuan2-13B-chat 63.27 64.14 58.02 59.09 65.58

AquilaChat2-7B 63.21 64.02 58.34 59.38 65.04 LLaMA2-70B-chat 61.25 61.59 57.20 65.13 58.83 InternLM-7B-chat 60.84 62.40 51.48 53.95 66.48 Baichuan2-7B-chat 57.99 58.01 57.87 55.54 58.28

ChatGLM2-6B 34.13 32.46 54.17 39.89 28.75

Table 4: Overall results of the Chat model. "Obj." denotes objective and "Sub." denotes subjective. Detailed results can be found in the appendix.

Herein, we chiefly contrast the capabilities in five aspects, namely Language, Reasoning, Understanding, Examination, and Code Generation, on prevalent datasets about these domains. The specific datasets utilized are as follows:

- • Language. WMT, CLUEWSC [29]
- • Reasoning. GSM8K, winogrande [30], HellaSwag [31], OpenBookQA [32], PIQA [33], ARC-e [34], BUSTM [35]
- • Understanding. BoolQ [36], TruthfulQA [37], RAFT, ChID [35], EPRSTMT [35],TNEWS [29], OCNLI[29], SEM-Chinese
- • Examination. MMLU [38], C-Eval [39], CMMLU [40], CSL [29]
- • code genration. HumanEval [41]

To make a more fair and impartial comparison in the data sets above, we used primarily two open source thirdparty frameworks for evaluation: lm-evaluation-harness [42] and HELM [43]. Specifically, Winograde, HellaSwag, OpenBookQA, PIQA, ARC-e, BoolQ, TruthfulQA, and MMLU were evaluated using the lm-evaluation-harness framework, while CLUEWSC, BUSTM, RAFT, ChID, EPRSTMT, TNEWS, OCNLI, SEM-Chinese, and CSL were assessed using the HELM framework. For the remaining datasets, evaluations were carried out using the default evaluation method recommended by the dataset.

The detailed exposition of base model results in Table 3 divulges a rich narrative concerning the comparative performance across various models on different tasks and datasets. our Aquila2-34B model demonstrates a strong performance with the highest mean score of 68.09, indicating its robustness and generalization capabilities across a myriad of NLP tasks. In addition, Aquila2-34B scores 68.63 on average on English tasks and also dominates Chinese language tasks with a score of 76.56.

The differential performance across various tasks unveils the unique strengths of each model. For instance, LLaMA270B shines in the Winograde, HellaSwag, and OpenBookQA tasks, suggesting its adeptness in handling reasoning and understanding-centric challenges. In contrast, the ARC-e task highlights a close competition between LLaMA2-70B and InternLM-20B, both of which hover around a score of 81, indicating their prowess in addressing science-related queries.

In the realm of bilingual understanding, as demonstrated in the BUSTM task, Aquila2-34B emerges as a strong performer with the highest score of 81.18. This prowess underscores the model’s capability in bilingual sentence matching, which is an indispensable facet of cross-lingual NLP tasks.

The HumanEval task, albeit challenging for all models as displayed by the relatively lower scores, unveils a notable lead for Aquila2-34B with a score of 39.02. However, this lead suggests a potential edge in mimicking human-like understanding, a frontier that remains a formidable challenge in the NLP domain.

Lastly, a close contest is observed in tasks like TNEWS, and C-Eval, indicating a balanced challenge posed by these tasks to the evaluated models. This competitive landscape underscores the importance of a thorough and diversified evaluation to glean comprehensive insights into the models’ capabilities, thereby paving the way for future advancements in NLP.

### 5 Conclusion and Future Work

This research introduces Aquila2, a series of bilingual models that leverage the innovative HeuriMentor framework to optimize training efficiency and performance. HeuriMentor dynamically adjusts data distribution during training, resulting in faster convergence and improved model quality. Aquila2-34B, in particular, surpasses LLaMA-2-70B-expr and other benchmarks across 21 diverse datasets, demonstrating the efficacy of the HeuriMentor approach. Notably, Aquila2-34B maintains strong performance even under 4-bit quantization. By open-sourcing code, weights, and data, this work promotes further research in bilingual models. Future work will explore Mixture-of-Experts and data quality enhancements.

### 6 Limitation

Our evaluation process identified the inadvertent inclusion of test set data from GSM8K in the pre-training dataset. This data contamination potentially compromises the validity of Aquila2’s zero-shot and few-shot results on the GSM8K benchmark. We therefore recommend excluding these specific results from further comparisons.

To mitigate this issue, we have conducted a comprehensive re-evaluation using alternative benchmarks such as WTM22 (en-zh) and CLUEWSC. Detailed results of this re-evaluation can be found on the project’s GitHub repository:https: //github.com/FlagAI-Open/Aquila2.

This incident underscores the critical role of robust data validation procedures in machine learning research. Such practices ensure research integrity and the reliability of resulting AI models.

### References

- [1] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.
- [2] OpenAI. Introducing chatgpt, 2022.
- [3] OpenAI. Gpt-4 technical report, 2023.
- [4] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. ArXiv preprint, abs/2205.01068, 2022.
- [5] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, and et al. BLOOM: A 176b-parameter open-access multilingual language model. ArXiv preprint, abs/2211.05100, 2022.
- [6] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. ArXiv preprint, abs/2204.02311, 2022.
- [7] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. ArXiv preprint, abs/2302.13971, 2023.
- [8] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. ArXiv preprint, abs/2307.09288, 2023.
- [9] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany, 2016. Association for Computational Linguistics.
- [10] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. Huggingface’s transformers: State-of-the-art natural language processing. ArXiv preprint, abs/1910.03771, 2019.
- [11] Sha Yuan, Hanyu Zhao, Zhengxiao Du, Ming Ding, Xiao Liu, Yukuo Cen, Xu Zou, Zhilin Yang, and Jie Tang. Wudaocorpora: A super large-scale chinese corpora for pre-training language models. AI Open, 2:65–68, 2021.
- [12] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The Pile: An 800GB dataset of diverse text for language modeling. ArXiv preprint, abs/2101.00027, 2021.
- [13] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. ArXiv preprint, abs/2305.13245, 2023.

- [14] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. ArXiv preprint, abs/2104.09864, 2021.
- [15] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. ZeRO: Memory Optimization Towards Training A Trillion Parameter Models. arXiv:1910.02054 [cs, stat], October 2019. arXiv: 1910.02054.
- [16] Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, and Soumith Chintala. PyTorch Distributed: Experiences on Accelerating Data Parallel Training. arXiv:2006.15704 [cs], June 2020. arXiv: 2006.15704.
- [17] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv:1909.08053 [cs], March 2020. arXiv: 1909.08053.
- [18] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. Efficient large-scale language model training on GPU clusters using megatron-LM. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC ’21, pages 1–15, New York, NY, USA, 2021. Association for Computing Machinery.
- [19] Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing Activation Recomputation in Large Transformer Models, May 2022. arXiv:2205.05198 [cs].
- [20] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. ArXiv preprint, abs/2307.08691, 2023.
- [21] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. ArXiv preprint, abs/2203.15556, 2022.
- [22] Maurice Fréchet. Sur quelques points du calcul fonctionnel. Rendiconti del Circolo Matematico di Palermo (1884-1940), 22:1–72, 1906.
- [23] Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. ArXiv preprint, abs/2301.13688, 2023.
- [24] FlagOpen. Flagscale.
- [25] Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai, Guosheng Dong, Haizhou Zhao, Hang Xu, Haoze Sun, Hongda Zhang, Hui Liu, Jiaming Ji, Jian Xie, Juntao Dai, Kun Fang, Lei Su, Liang Song, Lifeng Liu, Liyun Ru, Luyao Ma, Mang Wang, Mickel Liu, MingAn Lin, Nuolan Nie, Peidong Guo, Ruiyang Sun, Tao Zhang, Tianpeng Li, Tianyu Li, Wei Cheng, Weipeng Chen, Xiangrong Zeng, Xiaochuan Wang, Xiaoxi Chen, Xin Men, Xin Yu, Xuehai Pan, Yanjun Shen, Yiding Wang, Yiyu Li, Youxin Jiang, Yuchen Gao, Yupeng Zhang, Zenan Zhou, and Zhiying Wu. Baichuan 2: Open large-scale language models. ArXiv preprint, abs/2309.10305, 2023.
- [26] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. ArXiv preprint, abs/2309.16609, 2023.
- [27] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. ArXiv preprint, abs/2307.09288, 2023.

- [28] InternLM-Team. Internlm: A multilingual language model with progressively enhanced capabilities. https: //github.com/InternLM/InternLM-techreport, 2023.
- [29] Liang Xu, Hai Hu, Xuanwei Zhang, Lu Li, Chenjie Cao, Yudong Li, Yechen Xu, Kai Sun, Dian Yu, Cong Yu, Yin Tian, Qianqian Dong, Weitang Liu, Bo Shi, Yiming Cui, Junyi Li, Jun Zeng, Rongzhao Wang, Weijian Xie, Yanting Li, Yina Patterson, Zuoyu Tian, Yiwen Zhang, He Zhou, Shaoweihua Liu, Zhe Zhao, Qipeng Zhao, Cong Yue, Xinrui Zhang, Zhengliang Yang, Kyle Richardson, and Zhenzhong Lan. CLUE: A Chinese language understanding evaluation benchmark. In Proceedings of the 28th International Conference on Computational Linguistics, pages 4762–4772, Barcelona, Spain (Online), 2020. International Committee on Computational Linguistics.
- [30] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The ThirtySecond Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 8732–8740. AAAI Press, 2020.
- [31] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, Florence, Italy, 2019. Association for Computational Linguistics.
- [32] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, Brussels, Belgium, 2018. Association for Computational Linguistics.
- [33] Yonatan Bisk, Rowan Zellers, Ronan LeBras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning about physical commonsense in natural language. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 7432–7439. AAAI Press, 2020.
- [34] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv preprint, abs/1803.05457, 2018.
- [35] Liang Xu, Xiaojing Lu, Chenyang Yuan, Xuanwei Zhang, Huilin Xu, Hu Yuan, Guoao Wei, Xiang Pan, Xin Tian, Libo Qin, and Hai Hu. Fewclue: A chinese few-shot learning evaluation benchmark. ArXiv preprint, abs/2107.07498, 2021.
- [36] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, Minneapolis, Minnesota, 2019. Association for Computational Linguistics.
- [37] Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland, 2022. Association for Computational Linguistics.
- [38] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.
- [39] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. ArXiv preprint, abs/2305.08322, 2023.
- [40] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. ArXiv preprint, abs/2306.09212, 2023.
- [41] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. ArXiv preprint, abs/2107.03374, 2021.
- [42] Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation. 2021.

- [43] Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, et al. Holistic evaluation of language models. ArXiv preprint, abs/2211.09110, 2022.
- [44] Jason Weston, Antoine Bordes, Sumit Chopra, and Tomás Mikolov. Towards ai-complete question answering: A set of prerequisite toy tasks. In Yoshua Bengio and Yann LeCun, editors, 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016.
- [45] Koustuv Sinha, Shagun Sodhani, Jin Dong, Joelle Pineau, and William L. Hamilton. CLUTRR: A diagnostic benchmark for inductive reasoning from text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4506–4515, Hong Kong, China, 2019. Association for Computational Linguistics.
- [46] Bhavana Dalvi, Peter Jansen, Oyvind Tafjord, Zhengnan Xie, Hannah Smith, Leighanna Pipatanangkura, and Peter Clark. Explaining answers with entailment trees. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7358–7370, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics.
- [47] Chandra Bhagavatula, Ronan Le Bras, Chaitanya Malaviya, Keisuke Sakaguchi, Ari Holtzman, Hannah Rashkin, Doug Downey, Wen-tau Yih, and Yejin Choi. Abductive commonsense reasoning. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020.
- [48] Li Du, Xiao Ding, Kai Xiong, Ting Liu, and Bing Qin. e-CARE: a new dataset for exploring explainable causal reasoning. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 432–446, Dublin, Ireland, 2022. Association for Computational Linguistics.
- [49] Jessica López Espejel, El Hassane Ettifouri, Mahaman Sanoussi Yahaya Alassan, El Mehdi Chouham, and Walid Dahhane. Gpt-3.5, gpt-4, or bard? evaluating llms reasoning ability in zero-shot setting and performance boosting through prompts, 2023.
- [50] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models, 2023.
- [51] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning, 2023.
- [52] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.
- [53] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. ArXiv preprint, abs/2310.03744, 2023.
- [54] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. ArXiv preprint, abs/2308.12966, 2023.
- [55] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. ArXiv preprint, abs/2311.03079, 2023.
- [56] Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
- [57] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. ArXiv preprint, abs/2305.10355, 2023.
- [58] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017.
- [59] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities, 2023.
- [60] Zheqi He, Xinya Wu, Pengfei Zhou, Richeng Xuan, Guang Liu, Xi Yang, Qiannan Zhu, and Hua Huang. Cmmu: A benchmark for chinese multi-modal multi-type question understanding and reasoning, 2024.
- [61] Holger Schwenk, Guillaume Wenzek, Sergey Edunov, Edouard Grave, Armand Joulin, and Angela Fan. CCMatrix: Mining billions of high-quality parallel sentences on the web. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6490–6500, Online, 2021. Association for Computational Linguistics.

- [62] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. Starcoder: may the source be with you! 2023.
- [63] Together Computer. Redpajama: An open source recipe to reproduce llama training dataset, 2023.

### 7 Appendix

#### 7.1 Alignment Evaluation

The elucidation of chat model results in Table 4 unveils a compelling narrative on the comparative efficacy of diverse chat models across various tasks and datasets. The vanguard of performance is the AquilaChat2-34B-v1.2 model, which shows a superior mean score of 76.34 (both subjective and objective), 76.38 (objective), and 75.80 (subjective). This robust performance underscores its commendable generalization capabilities across disparate NLP tasks and its proficiency in both objective and subjective evaluations.

Closer scrutiny reveals nuanced performance dynamics across different linguistic domains. In tasks associated with the English language, AquilaChat2-34B-v1.2 secures the leading position with a mean score of 69.92, while in the Chinese language tasks, it further extends its lead, achieving a score of 79.19. This dual linguistic prowess highlights its versatility and aptitude in handling multilingual chat-oriented tasks.

The differential performance across various tasks unveils the unique competencies of each model. For example, the BoolQ task evidences a close competition between AquilaChat2-34B and Llama-2-70B chat, both scoring in the high 80s, underscoring their ability to handle boolean questions adeptly. In contrast, in the RAFT task, Baichuan2-13B chat outweighs others with a score of 74.24, indicating a potential strength in reasoning and factual accuracy.

In a more granular task-based analysis, the EPRSTMT task witnesses almost all models performing admirably, with scores hovering around the high 1980s and low 1990s, reflecting a shared competency in this particular task. However, the AquilaChat2-34B-v1.1 model, with a score of 91.64, still manages to establish a lead.

Lastly, the SEM-Chinese task presents a deviation from the prevailing trend, where Baichuan2-13B-chat takes the lead with a score of 65.69, suggesting a possible area of specialization for this model.

The tableau of results thus furnished underscores the importance of a diverse and comprehensive evaluation regime to better understand the strengths, weaknesses, and areas of specialization inherent in each model, thereby propelling further advancements in the domain of conversational AI.

For both the base model and the supervised fine-tuned model, we prepend a special token ’[CLS]’ at the beginning of the input to conduct objective analysis. For the subjective analysis of the supervised fine-tuned model, we utilize the following template:

A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human’s questions.###Human: {Input}###Assistant:

All tests are performed on a single A800 GPU card. The setting of hyperparameters during testing is shown in Table 5:

|Hyperparameter<br><br>|Objective|Subjective<br><br>|
|---|---|---|
|Top_k Top_p Temperature Max new tokens Seed|1 0.95<br><br>0.9<br><br>1<br><br><br>123<br><br>|15<br><br>0.9<br>1<br><br><br>512 123|

Table 5: The hyperparameters used for objective and subjective analysis.

|Model<br><br>|D Teasy Thard|Dextend Teasy Thard<br><br>|Avg.(%)<br><br>|
|---|---|---|---|
|LLaMA2-70B InternLM-20B Aquila2-34B Aquila2-70B<br><br>|46.84 45.26 51.84 47.63 57.89 53.95 57.89 58.95|63.42 52.11 65.26 51.32 70.36 56.05 70.26 56.58<br><br>|51.99 54.01 59.56 60.92<br><br>|

Table 6: Accuracy of different models on test sets Teasy and Thard.

Inductive Deductive Abductive Causal

Model Name

Average bAbI-task16 CLUTRR bAbI-task15 EntailmentBank αNLI E-Care

Llama2-7B-Chat 33.3 13.3 50.0 80.0 46.7 60.0 47.2 Baichuan2-7B-Chat 40.0 26.7 43.3 73.3 53.3 50.0 47.8

Qwen-7B-Chat 20.0 10.0 66.7 86.7 56.7 56.7 49.5 Qwen-14B-Chat 26.7 10.0 63.3 86.7 63.3 56.7 51.1

Baichuan2-13B-Chat 33.3 10.0 66.7 80.0 66.7 63.3 53.3 Internlm-Chat-20B 46.7 13.3 43.3 80.0 70.0 70.0 53.9

GPT3.5-turbo 36.7 3.3 93.3 86.7 53.3 50.0 53.9 GPT3.5-text-davinci-002 46.7 6.7 86.7 83.3 63.3 46.7 55.6

Llama-2-70B-Chat 63.3 20.0 53.3 80.0 66.7 60.0 57.2 GPT4 93.3 36.7 100 90.0 83.3 83.3 81.1

AquilaChat2-7B 63.0 13.0 40.0 60.0 53.0 67.0 49.0 AquilaChat2-34B 73.3 30.0 86.7 73.3 80.0 76.7 70.0 AquilaChat2-70B 86.7 26.7 90.0 83.3 86.7 76.7 75.0

Table 7: The evaluation results of our models(Aquila2-7B, Aquila2-34B and Aquila2-70B) on the Integrated Reasoning Dataset.

#### 7.2 Generalization Evaluation

To evaluate the generalization performance of the models, we design a series of experiments on Aquila2-34B, Aquila270B, LLaMA2-70B and InternLM20B. First, we use a generic instruction tuning dataset D to train the models and evaluate them on test set Teasy across various capabilities such as language, reasoning and computation. At the same time, we create a training dataset tailored to the task types in Teasy. This data set is then fused with D to form the combined training set Dextend to evaluate the learning capacity in the domain. To further evaluate the generalization capability, we retain dimensions such as language, reasoning, and computation in Teasy and adjust task difficulty to create a new test set Thard. For example, the mathematical task type on Teasy is to solve a linear equation of one variable, while on Thard will be adjusted to solve a linear equation of two variables.

The models are trained separately on training sets Deasy and Dhard, and than test on Teasy and Thard. The results are shown in Table 6. Through these experiments, we could evaluate the generalizability of the models across different

training data and test data of varying difficulty. For example, experiments in D with Teasy and Thard allow us to analyze the ability of the models to generalize from out-of-domain data to in-domain tasks. Experiments on Dextend with Teasy and Thard provide insight into the learning capability of the models in the domain when answering similar questions or solving more difficult questions after training on simpler ones. As shown in Table 6, Aquila2-34B has a 7.57% higher average accuracy and Aquila2-70B has a 8.93% higher average accuracy than LLaMA2-70B. Moreover, Aquila2-70B shows the highest average accuracy in the experiments.

#### 7.3 Reasoning Capacity Evaluation

The proficiency of large-scale language models(LLMs) in natural language reasoning serves as a pivotal capability for the realization of Artificial General Intelligence (AGI). Logical reasoning, causal reasoning, and commonsense reasoning are three popular categories in natural language reasoning evaluation. Notably, the evaluation of commonsense reasoning primarily assesses the model’s capacity for knowledge retention rather than its pure reasoning prowess. Consequently, our comprehensive assessment focused extensively on logical reasoning and causal reasoning, encompassing inductive, deductive, and abductive reasoning, including six test datasets: bAbI-task16 [44], CLUTRR [45], bAbI-task15 [44], EntailmentBank [46], αNLI [47] and E-Care [48]. We name the meticulously curated test data from [49] as "Integrated Reasoning Dataset(IRD)". Using the IRD data set, we subjected our AquilaChat2-34B model to stringent subjective evaluations, meticulously examining its performance against its leading industry counterparts in five publicly available

| |GQA POPE VQAv2 MM-Vet|
|---|---|
|MiniGPT-4[50] Otter[51] InstructBLIP[52] LLava-1.5-7b[53]|30.80 - - 22.10 38.10 - - 24.60 49.20 - - 26.20 61.90 86.89 78.50 30.50<br><br>|
|Aquila2VL-7B<br><br>|61.28 87.05 76.88 29.40|

Table 8: The accuracy of comparing Aquila2VL-7B with other models on different datasets.

| |Val Test<br><br>|MCQ MRQ FBQ|
|---|---|---|
|LLava-1.5-13b GPT-4V|11.36 11.96 30.19 30.91<br><br>|12.29 00.79 12.32 29.70 22.44 33.21|
|Aquila2VL-34B<br><br>|41.37 40.79|50.15 38.98 26.35|

Table 9: The accuracy of comparing Aquila2VL-34B with other models on different question types. MCQ means multiple-choice question, MRQ means multiple-response question and FBQ means fill-in-the-blank question.

tasks. Remarkably, our AquilaChat2-34B and AquilaChat2-70B demonstrate exceptional proficiency, positioning themselves as a close runner-up to GPT4.

This notable improvement in reasoning ability can be attributed to two key factors: 1. Enhanced model capacity: Models with larger parameter quantities exhibit greater potential in reasoning tasks, showcasing the significance of scale in achieving superior performance. 2. Augmented training data: The incorporation of an increased proportion of Code training data has proven instrumental in enhancing the model’s reasoning capabilities. This enriched dataset contributes to a deeper understanding of complex logical structures and strengthens the model’s reasoning prowess.

#### 7.4 Multi-modal capability adaptation

Broadening the scope to encompass multimodal tasks emerges as a significant application trajectory for extensive language models [54, 53, 55]. To verify the ability of the Aquila2 to acclimate to downstream tasks, we follow the experimental setting reported in LLaVA [53] to train multi-modal chat models, namely Aquila2VL-7B and Aquila2VL34B. To evaluate the multimodal understanding ability of Aquila2VL, we evaluate GQA[56], POPE[57], VQAv2[58] and MM-Vet[59]. As shown in Table 8, Aquila2VL-7B achieves comparable results to LLava-1.5-7b, even outperforming POPE, achieving an accuracy of 78.50%. To further explore the multimodal ability of Aquila2VL in Chinese, we evaluate CMMU[60], a benchmark for multimodal and multi-type question understanding and reasoning in Chinese. The evaluation results are shown in Table 9. Remarkably, Aquila2VL-34B exhibits commendable against LLaVA-1.5-13B and GPT-4V, getting an accuracy of 41.37% and 40.79% on the validation and testing sets, respectively.

#### 7.5 Tokenizer

We randomly extracted multilingual sentences from CCMatrix [61], extracted different codes from Starcoder data [62], and tested and compared the average tokenized sequence length of our tokenizer and other tokenizers. The results are shown in Tab.10 and 11, respectively.

[Figure 8]

Figure 6: The training loss of 1.3B model with different Chinese and English data ratios (zh:en).

Ar De En Es Fr Hi It Ja Ko Pt Zh Ru Th Average

Chatglm2(64K) 74.3 36.2 19.8 35.1 30.1 74.8 36.7 27.0 35.4 34.3 15.8 102.1 140.5 38.1 Baichuan(64K) 74.3 39.6 20.9 37.6 32.6 69.6 39.3 27.4 34.4 36.7 17.9 116.9 137.0 39.1 Baichuan2(125K) 69.5 36.7 19.7 35.4 31.0 66.3 37.0 25.4 33.0 34.6 15.2 106.6 135.0 36.7 Aquila(100K) 58.5 34.7 19.1 32.2 28.6 66.1 34.4 25.5 49.8 31.7 15.9 107.2 154.9 36.0 Qwen(256k) 31.2 29.7 19.1 28.2 25.3 63.7 31.9 18.1 23.2 27.6 15.9 77.2 78.2 28.5

- Table 10: The average tokenized sequence length for different languages.

Tokenizer Shell Tex Java Python Js SQL C Cpp Average Ziya(40K) 386.8 758.9 679.0 708.3 543.0 510.3 739.7 878.3 650.5 Chatglm2(64K) 402.7 780.0 691.4 716.1 555.6 529.1 744.5 889.1 663.6 Baichuan(64K) 431.8 835.2 723.0 761.7 586.9 567.9 787.4 943.5 704.7 Aquila2(100K) 353.6 684.4 594.8 623.9 473.7 448.5 644.6 767.8 573.9 Qwen(256k) 303.4 674.3 488.9 526.0 411.3 392.9 538.1 647.0 497.7

- Table 11: The average tokenized sequence length for different codes.

#### 7.6 Data Recipe for Languages

Aquila2 is a bilingual language model designed to accommodate both Chinese and English languages. Before commencing formal training, assessing the influence of varying ratios of Chinese and English data on the model’s training performance is necessary. To achieve this goal, we trained different 1.3B models with different proportions of Chinese and English data and observed changes in training loss. The results are shown in Figure 6. We found that generally speaking, the higher the proportion of Chinese in the training data, the greater the overall loss of model training. This may to some extent prove that the difficulty of learning Chinese corpus is greater than that of English. In our investigation, we observed that, for our dataset, maintaining a token ratio of approximately 1:2 or 1:3 between the Chinese and English training corpora led to lower training loss. Consequently, during formal training, we establish and adjust the proportion of Chinese and English corpora based on both this finding and the performance metrics of downstream tasks.

#### 7.7 Case Study

We provide some generation examples of our AquilaChat2-34B-v1.2 model, which are shown as in Figure 7, 8, 9, 10, 11, 12 and 13.

#### 7.8 Convergence Observation via Weights

We plot the trajectories of the standard deviations of each parameter, namely self_attn.k_proj (WK), self_attn.q_proj (WQ), self_attn.v_proj (WV ), and self_attn.o_proj (WO), for each layer, shown in Fig. 14, 15, 16, 17, 18, 19, 20, 21. The y-axis represents the standard deviation, while the x-axis represents the index of each layer. Each trajectory serves as a representation of the corresponding parameter in a checkpoint. Each subplot represents a training stage, namely K6, K61&K62, K63, K64 for Aquila2-34B, and K6, K61, K63, K65 for Aquila-70B.

[Figure 9]

- Figure 7: Question answer example.

[Figure 10]

- Figure 8: Code generation example.

[Figure 11]

Figure 9: Mathematical reasoning example.

[Figure 12]

Figure 10: Translation example.

[Figure 13]

Figure 11: Knowledge explanation example.

Configuration Key Value

- adam_beta1 0.9
- adam_beta2 0.95 adam_eps 1e-08 add_bias_linear False apply_layernorm_rms True apply_query_key_layer_scaling True apply_residual_connection_post_layernorm False attention_dropout 0.0 attention_softmax_in_fp32 True bf16 True bias_dropout_fusion True data_parallel_size 32 distributed_backend nccl embedding_weights_in_fp32 True ffn_hidden_size 24576 global_batch_size 1024 gradient_accumulation_fusion True group_query_attention True hidden_dim_multiplier 1.3 hidden_dropout 0.0 hidden_size 6144 init_method_std 0.0165 layernorm_epsilon 1e-05 layernorm_init_weight 0.3 lr 0.00015 lr_decay_style cosine lr_warmup_samples 500000 make_vocab_size_divisible_by 64 max_position_embeddings 4096

- micro_batch_size 1 min_lr 1.5e-05 num_attention_heads 64 num_layers 60 num_query_groups 8 optimizer adam pipeline_model_parallel_size 4 position_embedding_type rope rampup_batch_size 32, 32, 2000000 rotary_percent 1.0 rotary_position_embeddings_in_fp32 True save_interval 1000 seed 42 seq_length 4096 sequence_parallel True split 1 swiglu True tensor_model_parallel_size 4 tokenizer_type AquilaTokenizer use_distributed_optimizer True use_flash_attn True vocab_size 100008 weight_decay 0.1 weight_decay_incr_style constant

- Table 12: Full configuration of Aquila2-34B pretraining

Configuration Key Value adam_beta1 0.9 adam_beta2 0.95 adam_eps 1e-08 add_bias_linear False apply_layernorm_rms True apply_query_key_layer_scaling True apply_residual_connection_post_layernorm False attention_dropout 0.0 attention_softmax_in_fp32 True bf16 True bias_dropout_fusion True data_parallel_size 24 distributed_backend nccl embedding_weights_in_fp32 True ffn_hidden_size 28672 global_batch_size 1056 gradient_accumulation_fusion True group_query_attention True hidden_dim_multiplier 1.3 hidden_dropout 0.0 hidden_size 8192 init_method_std 0.0149 layernorm_epsilon 1e-05 layernorm_init_weight 0.25 lr 0.00015 lr_decay_style cosine lr_warmup_samples 500000 make_vocab_size_divisible_by 64 max_position_embeddings 4096

- micro_batch_size 2 min_lr 1.5e-05 num_attention_heads 64 num_layers 80 num_query_groups 8 optimizer adam pipeline_model_parallel_size 4 position_embedding_type rope rampup_batch_size 48, 48, 2000000 rotary_percent 1.0 rotary_position_embeddings_in_fp32 True save_interval 500 seed 42 seq_length 4096 sequence_parallel True split 1 swiglu True tensor_model_parallel_size 8 tokenizer_type AquilaTokenizer use_distributed_optimizer True use_flash_attn True vocab_size 100008 weight_decay 0.1 weight_decay_incr_style constant

- Table 13: Full configuration of Aquila2-70B pretraining

[Figure 14]

Figure 12: Advice consultation example.

[Figure 15]

Figure 13: Role play example.

#### 7.9 Detailed results of the Chat model.

As illustrated in Fig.14, AquilaChat2-34B-v1.2 and AquilaChat2-34B-v1.1 achieve the highest scores in most tasks. Moreover, AquilaChat2-34B-v1.2 achieves the highest average scores in both Chinese and English tasks.

#### 7.10 Training parameters

The parameters used for training Aquila2-34B and Aquila2-70B can be found in Tab.12 and Tab.13, which include configuration information such as model structure, learning rate, optimizer, and so on.

#### 7.11 Exploring Data Formulation Strategies in Specific Domains

- • v1 and v2 datasets We refer to the sampling weights of various sources used in the GPT-3 work. A 2-3 epoch oversampling approach is employed for high-quality data like wiki and QA, while high-quality long textual data from textbooks and paper was oversampled for 1.5 epochs. For curated web data collections such as OpenWebText, the data were oversampled for 1.5 epochs, while other web data and code data were sampled for 1 epoch. Furthermore, in terms of language distribution, a balanced ratio of English to Chinese text to code was maintained at 60%:30%:10%. The validation process was performed on the 7B model training. During the training process, challenges emerged due to significant variations in the data sources and disparate loss values during data fusion, leading to slow convergence of the model.
- • v3 dataset To address these issues, we considered the actual volume of data from each source. Web data was designated as the primary data source. Sampling weights for the data were determined based on the proportions of data obtained from various sources. This approach was inspired by the works of GPT-3 [1] and Llama [7]. Specifically, the web data represented 82%, with code, encyclopedia, and textbook data contributing 4.5% each, literature 2.5%, and questions and answers data 2%. This refined methodology ensured a more effective and efficient training process, allowing improved convergence and overall model performance.
- • knowledge-oriented datasets (k1-k5) Upon completion of training the 7B model on the v3 dataset (approximately 1000B tokens), we evaluated the model’s performance with several validation datasets sampled from the

[Figure 16]

[Figure 17]

(a) K6 (b) K61&K62

[Figure 18]

[Figure 19]

(c) K63 (d) K64

- Figure 14: Standard variance of the weight matrix WK across different layers in Aquila-34B. Different tokens in the graph represent the total amount of training data tokens used.

[Figure 20]

[Figure 21]

(a) K6 (b) K61&K62

[Figure 22]

[Figure 23]

(c) K63 (d) K64

- Figure 15: Standard variance of the weight matrix WQ across different layers in Aquila-34B. Different tokens in the graph represent the total amount of training data tokens used.

[Figure 24]

[Figure 25]

(a) K6 (b) K61&K62

[Figure 26]

[Figure 27]

(c) K63 (d) K64

- Figure 16: Standard variance of the weight matrix WV across different layers in Aquila-34B. Different tokens in the graph represent the total amount of training data tokens used.

[Figure 28]

[Figure 29]

(a) K6 (b) K61&K62

[Figure 30]

[Figure 31]

(c) K63 (d) K64

- Figure 17: Standard variance of the weight matrix WO across different layers in Aquila-34B. Different tokens in the graph represent the total amount of training data tokens used.

[Figure 32]

[Figure 33]

[Figure 34]

(a) K6 (b) K63 (c) K65

- Figure 18: Standard variance of the weight matrix WK across different layers in Aquila-70B. Different tokens in the graph represent the total amount of training data tokens used.

[Figure 35]

(a) K6 (b) K63 (c) K65

[Figure 36]

[Figure 37]

- Figure 19: Standard variance of the weight matrix WQ across different layers in Aquila-70B. Different tokens in the graph represent the total amount of training data tokens used.

[Figure 38]

(a) K6 (b) K63 (c) K65

[Figure 39]

[Figure 40]

- Figure 20: Standard variance of the weight matrix WV across different layers in Aquila-70B. Different tokens in the graph represent the total amount of training data tokens used.

[Figure 41]

[Figure 42]

[Figure 43]

(a) K6 (b) K63 (c) K65

- Figure 21: Standard variance of the weight matrix WO across different layers in Aquila-70B. Different tokens in the graph represent the total amount of training data tokens used.

AquilaChat2-34B-v1.2AquilaChat2-34B-v1.1 Baichuan2-13B-chat

AquilaChat2-7B Llama2-70B-chat InternLM-7B-chat Baichuan2-7B-chat

ChatGLM2-6B

Model

Mean 76.34 74.49 63.27 63.21 61.25 60.84 57.99 34.13 En Mean 69.92 69.19 59.09 59.38 65.13 53.95 55.54 39.89 Zh Mean 79.19 77.43 65.58 65.04 58.83 66.48 58.28 28.75

CLUEWSC(5-shot) 87.60 87.32 78.26 56.86 79.07 75.27 68.14 50.70 BUSTM(5-shot) 82.50 84.44 75.37 72.80 62.97 76.13 63.08 -

BoolQ (5-shot) 86.80 88.00 74.90 78.10 88.57 76.77 68.47 73.60

TruthfulQA(5-shot) 64.80 70.49 50.56 66.21 53.82 33.28 46.33 27.52 RAFT(5-shot) 66.40 70.68 74.24 65.23 70.91 59.85 66.82 40.23 ChID(5-shot) 88.20 87.78 61.93 72.80 36.93 68.43 52.28 8.15

EPRSTMT(5-shot) 90.00 91.64 88.85 90.98 89.73 89.95 81.20 52.79 TNEWS(5-shot) 55.50 63.11 47.68 55.82 45.04 43.58 39.48 4.37 OCNLI(5-shot) 83.60 78.96 50.48 65.46 50.62 60.82 51.15 29.86

SEM-Chinese(5-shot) 73.08 55.74 65.69 44.17 50.84 59.29 56.54 35.86 MMLU(5-shot) 61.70 51.19 52.96 38.39 55.14 47.02 48.31 18.20

IRD 58.30 65.60 42.80 49.00 57.20 52.80 47.80 -

CSL(5-shot) 76.40 73.08 48.76 61.41 55.42 58.39 54.36 48.27 CLCC-H-v2.0 75.80 74.84 73.24 67.68 - 50.16 67.95 54.17

Table 14: Detailed results of the Chat model.

same distribution to training datasets. Through the validation loss curve, we find that many knowledge-intensive datasets are not adequately learned. Therefore, we extracted pivotal subsets and collected richer, knowledgeenriched data to construct specialized knowledge-oriented datasets for continuous pretraining. Specifically, K1 dataset contains bilingual wiki datasets, bilingual paper datasets, English textbook datasets, and code-text pair datasets. In the k2 dataset, we introduce several new Chinese sources like question-and-answer forum data, technical blog data, newspaper data, and headline news data. The k3 dataset incorporates open-source literature data (such as redpajama [63]) and medical question-answering data. Additionally, the k4 dataset includes mathematical computation data and higher-quality open-source code data (from starcoder [62]). K5 further integrates a wider range of open-source high-quality Chinese data and data translated from sources (such as tigerbotdata https://huggingface.co/datasets/TigerResearch/pretrain_zh, ccmatrix [61], etc.). During the training of the k1, k2, and k3 data sets, it was noted that the use of only knowledge-oriented data

led to highly unstable model losses. To address this, in the k4 and k5 datasets, we supplemented knowledgeoriented data with the corresponding web data to ensure the stability of the model. Through experimentation with the k1 to k5 datasets, we validated the efficacy of high-quality data sources and determined the final configuration of the pretraining data set.

