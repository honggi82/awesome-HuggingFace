# Leveraging Open Knowledge for Advancing Task Expertise in Large Language Models

Yuncheng Yang, Yulei Qin, Tong Wu, Zihan Xu, Gang Li, Pengcheng Guo, Hang Shao, Yuchen Shi, Ke Li, Xing Sun, Jie Yang, Yun Gu

## arXiv:2408.15915v2[cs.CV]7Sep2024

Abstract—The cultivation of expertise for large language models (LLMs) to solve tasks of specific areas often requires special-purpose tuning with calibrated behaviors on the expected stable outputs. To avoid huge cost brought by manual preparation of instruction datasets and training resources up to hundreds of hours, the exploitation of open knowledge including a wealth of low rank adaptation (LoRA) models and instruction datasets serves as a good starting point. However, existing methods on model and data selection focus on the performance of general-purpose capabilities while neglecting the knowledge gap exposed in domain-specific deployment. In the present study, we propose to bridge such gap by introducing few human-annotated samples (i.e., K-shot) for advancing task expertise of LLMs with open knowledge. Specifically, we develop an efficient and scalable pipeline to cost-efficiently produce task experts where K-shot data intervene in selecting the most promising expert candidates and the task-relevant instructions. A mixture-of-expert (MoE) system is built to make the best use of individual-yet-complementary knowledge between multiple experts. We unveil the two keys to the success of a MoE system, 1) the abidance by K-shot, and 2) the insistence on diversity. For the former, we ensure that models that truly possess problem-solving abilities on K-shot are selected rather than those blind guessers. Besides, during data selection, instructions that share task-relevant contexts with K-shot are prioritized. For the latter, we highlight the diversity of constituting experts and that of the fine-tuning instructions throughout the model and data selection process. Extensive experimental results confirm the superiority of our approach over existing methods on utilization of open knowledge across various tasks. Our codes will be available at https://github.com/Yaphabates/Rocket.

Index Terms—Mixture of Experts, Low Rank Adaptation, Model Selection, Data Selection

✦

### 1 INTRODUCTION

Recent few years have witnessed significant development of large language models (LLMs) across a broad spectrum of tasks and domains. The open-source community offers an array of competitive pre-trained/foundation models with general-purpose language understanding and generation capabilities [7], [36], [88], allowing for the follow-up special-purpose tuning customized for specific tasks and domains.

To match specialist capabilities, one common exploration to boost pre-trained base models within specialty areas is instruction tuning on expert-curated contents [57], [71], [113]. However, such a training-intensive approach poses challenges to the manual collection and annotation of instruction-response pairs, which impedes agile development and deployment. On the other hand, extravagant computing resources are required to grasp the missing domain knowledge from scratch, making it difficult to appropriately calibrate for the expected responses.

Fortunately, a vast amount of fine-tuned and aligned models [96], together with instruction datasets across various domains and tasks [62], [93], [102], are available online. Such models often exist in the form of LoRA [32] adapters, which are derived

Y. Yang is with Institute of Image Processing and Pattern Recognition, Institute of Medical Robotics, Shanghai Jiao Tong University, Shanghai 200240, China, and Tencent YouTu Lab, Shanghai 200233, China (Email: yaphabates@sjtu.edu.cn). Y. Qin, T. Wu, Z. Xu, G. Li, P. Guo, H. Shao, Y. Shi, K. Li, and X. Sun are with Tencent YouTu Lab, Shanghai 200233, China (Email: yuleiqin@tencent.com). J. Yang and Y. Gu are with Institute of Image Processing and Pattern Recognition, Institute of Medical Robotics, Shanghai Jiao Tong University, Shanghai 200240, China (Email: yungu@ieee.org). Y. Yang and Y. Qin contributed equally. Manuscript received Aug 25, 2024. Corresponding author: Yun Gu.

from few strong base models. Given any task of interest, one can choose pertinent models from the existing LoRA bank or library as a good starting point. Accordingly, researchers have developed various approaches to utilize and select the publicly released LoRA models [33], [63], [67] either in an off-line or an on-line manner for expeditious task adaptation. Meanwhile, great efforts have been put into the selection of high-quality open instruction datasets [75] for improving the instruction following capabilities, where the most challenging [15], [18], [48], [103] and influential [80], [100], [101], [111] datapoints are preferred.

Despite the tremendous progress made in model and data selection, most existing methods target at lifting LLMs for better alignment with human preference in general domains and common tasks. It remains an under-explored problem to advance specific task expertise of LLMs by fully exploiting the open knowledge such as public LoRA models and datasets. To fulfill such taskoriented knowledge augmentation, in this paper, we propose to bridge the gap between general and “vertical” domain knowledge by resorting to a few human-verified instruction samples (i.e., Kshot) from the task of interest. Such K-shot data play a steering role in guiding the selection of the most appropriate candidate models and beneficial datapoints to solve the task at hand.

Problem Definition Given a small amount of labeled, realworld instructions from the task of interest (K-shot), we aim at developing a universally effective and easily scalable pipeline that leverages publicly available models and datasets to advance the task expertise of state-of-the-art (SOTA) LLMs.

In this context, we encounter three primary challenges:

Challenge 1 Given a collection of LLMs including one foundation model and its fine-tuned LoRA variants, how can we take full advantage of K-shot to efficiently pinpoint the models with

[Figure 1]

[Figure 2]

[Figure 3]

K-shot Data from Task of Interest

Open-Source Datasets

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

... ...

[Figure 13]

[Figure 14]

Step3 Data Selection

[Figure 15]

[Figure 16]

Open-Source Models

Similarity-First Diversity-Aware

###### Base Models Task Adapters

Perplexity Performance

Diversity

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

K-shot & Training Data Augmentation

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

LLaMA

[Figure 29]

[Figure 30]

Step2 MoE Initializa tion

Step4 Continue FineTuning

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Step1 Model Selection

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Mistral

Task

[Figure 43]

Expert Expert Candidates

Mixture-of-Expert System

Fig. 1. Given few annotated data from any task of interest (K-shot), we aim to advance LLMs in task expertise by leveraging open-source models and datasets. We propose an efficient and scalable pipeline to fully exploit the steering role of K-shot throughout model and data selection. Highly promising experts are first selected from the model bank by comprehensive consideration of their perplexity and performance on the K-shot and intra-group diversity. These experts are initialized as one MoE system. Subsequently, we perform data augmentation by selecting diverse open instructions that resemble K-shot the most. Finally, we fine-tune the MoE system with both K-shot and the augmented data, which not only improves token-wise cooperation between experts but also integrates broad knowledge into the system. The ultimate task expert benefits from the complementary skills and knowledge of constituting experts.

the highest potential for solving tasks of interest?

uncovers K-shot for model selection where both the reasoning perplexity and the performance in accuracy provide insights into the generalized empirical risk of the model on examples from tasks of interest. We investigate and highlight the role of chainof-thought (CoT) [94] reformulation of answers with rationales in improving the interpretability and robustness of perplexity as the expertise indicator. To address Challenge 2, we develop a similarity-first, diversity-aware data selection strategy for augmentation of task expertise, where K-shot intervene in retrieving the most similar open-source instructions in the embedding space. To bridge the knowledge gap between general- and special-purpose tasks without easily overfitting K-shot, we once again integrate diversity into the selection philosophy where semantic duplicates are removed to ensure a more dispersed distribution of datapoints. To resolve Challenge 3, we construct a mixture-of-experts (MoE) system and pay extra attention to the composition choice of experts at initialization and the token-wise coordination between experts in fine-tuning. For the former, we are the first study to demonstrate the critical role of a diversified selection of expert candidates in building a successful MoE system with open-source models. Such diversity effectively expands the knowledge boundary beyond a single model. For the latter, traditional methods either use linear model merging [34], [37], [108] or detached gating mechanism [63], [67], [85], neglecting the potential conflicts or sub-optimal allocation among experts. Their lack of interpretability further worsens the generalization [104]. In contrast, our MoE system is fully optimized on K-shot data augmentation for mitigating inter-expert conflicts and getting familiar with relevant

- Challenge 2 Given abundant instructions from open-source

datasets, how can we identify the ones that share similar task or domain contexts with K-shot to inject supplementary knowledge into LLMs without causing overfitting?

- Challenge 3 If multiple prospective LLMs are proved to be

valid, how can we build an adaptive token-wise gating system to harness their individual-yet-complementary knowledge with improved cooperation between LLMs over K-shot?

To address these challenges, we propose the following task augmentation pipeline (see Fig. 1). Initially, for experimental control of comparability and reproducibility, we maintain our own LoRA bank by collecting thirty-eight representative open-source datasets and preparing a pool of LoRA models. It ensures the availability and diversity of instruction-tuned models on the basis of LLaMA [88] and Mistral [36] pre-trained models. Such a bank can be easily generalized to scenarios where all publicly released models are considered in practice. To tackle Challenge 1, we propose to select models that truly understand the task of interest rather than guessing randomly. We incorporate K-shot data to develop a selection mechanism featured by three key indicators:

- 1) reasoning perplexity: the uncertainty of a LLM in modeling the reasoning process towards its ultimate answer to each instruction;
- 2) exact match accuracy: the evaluation of the generated responses of the model on K-shot determined by the accuracy metric;
- 3) group diversity: the degree of distinction between multiple selected candidates measured in parameter versatility. To our best knowledge, the proposed method is the pioneering work that

tasks. In summary, our contributions are four-fold:

- • We propose a novel K-shot learning setting in forging LLMs with task expertise, where few human-verified instructions from tasks of interest are available for improving real-world applications in a cost-efficient way.
- • We propose an effective and scalable pipeline that makes the best use of K-shot in leveraging open knowledge, where the most promising candidates and the most beneficial instructions are respectively selected from the public LoRA bank and the open-source datasets.
- • We emphasize diversity throughout the proposed model and data selection strategies, which lays a solid foundation for constructing a versatile MoE system and acquiring task-relevant skills without overfitting.
- • We demonstrate the advantages of a MoE system maximized by proper initialization and fine-tuning with enhanced domain knowledge and expert harmonization.

- 2 RELATED WORKS

- 2.1 Efficient Fine-tuning of Parameters

Parameter-efficient fine-tuning methods (PEFTs) are a set of techniques that diverge from traditional full-parameter fine-tuning. In PEFTs, only a certain subset of a model’s parameters are adjusted to better suit specific tasks of interest.

- 2.1.1 LoRA Low-rank adaptation (LoRA) [32] and its variants [25], [54], [60], [112] use low-rank matrices to approximate additive weights during training. These methods are beneficial as they do not necessitate additional computational resources during inference, where the updated weights can be integrated into the model without any complications.
- 2.1.2 Prompt Tuning Prompt-based methods incorporate randomly initialized soft tokens to the input, usually as a prefix, and train their embeddings while maintaining the LLM’s weights constant, as suggested in [44], [49], [120]. While these methods perform competitively, they do entail a substantial computational load during inference.
- 2.1.3 Adapters Adapter methods involve the training of extra modules (for instance, fully connected layers) on top of the frozen pre-trained model [72], [119]. Unfortunately, these adapters are not effortlessly integrated into the original architecture, thereby reducing inference efficiency.

- 2.2 Mixture-of-Expert Models

The MoE technique [38], [84] combines several specialized expert models to efficiently scale up models for improved generalization. The principle behind MoE is that each expert is adept at handling a specific region of the input distributional space, and their combined decision-making outperforms any individual expert. Recent studies [29], [45], [50], [56], [99] focus on utilizing the MoE as a PEFT technique. Other methods [20], [33], [64], [67], [116] underscore the use of existing LoRA experts for convenient assembly, where the fine-tuning of their parameters is saved. Specifically, both [35] and [10] route to one expert by comparing

the input query with the averaged embeddings of the datasets used to train each individual expert. Such reliance on the fine-tuning datasets restricts the practicability of these methods. [28], [52], [58] all investigate external models such as GPT4 [3] and reward model [7] for estimation of the routing policy, which increases their cost of deployment and causes isolation between the routing model and the candidate experts. In spite of the development of merging and routing techniques, few efforts [53], [69], [85] have been made to combine cross-domain experts to address multi-task problems.

Very recently, PHATGOOSE [63] proposes a post-hoc adaptive token-wise gating mechanism to recycle from a collection of specialized experts. It aims at improving the zero-shot generalization of the pre-trained base model by constructing a routing system and performing dynamic token-by-token assignment. Despite the shared MoE principal, PHATGOOSE differs from the proposed method in three key aspects. First, the problem setting of PHATGOOSE is fundamentally different from ours in that they are not targeted at improving specific tasks of interest. There exists no model and data selection procedures in PHATGOOSE for acquiring task-relevant skills. Second, PHATGOOSE assumes that the contributors of the LoRA models provide additional gate modules that are implemented as sigmoid layers in front of each LoRA module. However, it is almost impossible to enforce the same gate training pipeline on the entire open-source community. In practice, one can only find the released LoRA modules from the repositories of Huggingface and Github without gating vectors. Third, our MoE system adapts to different tasks of interest by polishing the routing weights and the constituting experts simultaneously, while PHATGOOSE keeps the LoRA modules and gating vectors fixed with lower flexibility. Arrow [67] is another contemporary method which maintains the LoRA library and proposes a routing mechanism to select the most input-relevant LoRAs. Although Arrow is featured by its training-free routing, it does not consider the potential conflicts when the inputs are not representative enough of the given task. The statistics of the LoRA parameters (e.g., singular vectors) are directly used as the proxy for the routing weights, which is prone be biased towards the original datasets used to train LoRA modules. Furthermore, they do not introduce the K-shot datapoints from downstream tasks for calibrating the expert behavior, and neglect the exploitation of open-source datasets for optimizing the collaboration between experts.

###### 2.3 Data Selection for Efficient Tuning

Existing methods on data selection for pre-training and instruction tuning of LLMs can be categorized into: 1) quality-based, 2) diversity-based, and 3) importance-based [75].

2.3.1 Quality

[59] explores perplexity, L2-Norm error, and memorization as quality scores to rank and prune pre-training corpora. [48] presents a novel self-guided approach to autonomously select high-quality samples from open-source datasets using the instruction-following difficulty (IFD) metric. [18] employs the GPT3.5 to score instruction-response pairs and filters out samples with scores below a threshold. [15] introduces instruct mining to automatically select high-quality instruction-following data for LLM fine-tuning. [27] proposes a quality evaluation model to extract high-quality subsets from the original dataset and designs an algorithm to further select a seed instruction dataset with extensive coverage.

- 2.3.2 Diversity

Geometry-based sampling methods are the most intuitive and widely-used ones for maximizing diversity of the selected samples [40], [83], [118]. In particular, k-center greedy [81] is favored in diversity sampling on massive pre-training and instructiontuning corpus [11], [16], [27], [98]. It performs iterative and greedy selection on data that exhibit the most dissimilarity with the already selected set in the embedding space.

- 2.3.3 Importance

Mixture-of-Expert Model SFT Base Model (Low Reasoning Perplexity, Medium Performance) Base Model (High Reasoning Perplexity, High Performance)

′SFTModel(LowReasoningPerplexity,MediumPerformance) ′SFTModel(HighReasoningPerplexity,HighPerformance)

′

′

Base Model (High Reasoning Perplexity, Low Performance)

′ ′ ′

Performance

′ ′ ′

′

∆𝑨𝒄𝒄 +7.9%

∆𝑨𝒄𝒄 +4.6%

′

Two kinds of gradient-based methods on importance estimation have been developed: 1) gradient matching [8], [111], [115], i.e., the gradients of the entire set being approximated by the weighted gradients of the selected set, and 2) gradient influence [9], [13], [73], i.e., the influence of each training datapoint on the testing set being measured by upweighted gradient multiplication. For importance-oriented sampling, [101] adopts importance resampling to select a subset from a large-scale unlabeled dataset that shares similar distributions with the target set.

Reasoning Perplexity

Fig. 2. The performance of task-specific fine-tuning versus the reasoning perplexity of models in the bank. Preliminary experiments demonstrate that models of lower performance are not always in lack of domainspecific knowledge. Instead, their inability to follow instructions on the expected output format (e.g., answer choice) causes parsing failure during post-processing on the generated responses, which diminishes their performance. To avoid such biased, partial measurement merely by the metric such as exact-match accuracy, we propose to use the perplexity over the CoT rationales of answers as a superior, complementary proxy for model assessment. Accordingly, we evaluate if the model possesses the task-specific knowledge by computing its perplexity score of modeling the reasoning process. Models that achieve lower reasoning perplexity are considered competent and tend to achieve greater improvement after fine-tuning than those with higher reasoning perplexity.

Although these methods strike a balance between data quality and quantity, they fail to incorporate the data selection pipeline into the task-oriented model development. On the contrary, the proposed method focuses on improving the downstream performance given limited K-shot real-word datapoints under the context of expert fusion. Correspondingly, we simultaneously consider the quality and importance where the resemblance of an open-source instruction to the K-shot set is prioritized during selection. In addition, we treasure the diversity of the selected dataset as its variety helps polish the coordination between experts in token-wise routing. To the best of our knowledge, the proposed method is the pioneer that integrates comprehensive data selection into the advancement of a MoE system for task expertise, where the concepts of quality, diversity, and importance play a critical role throughout the selection of both expert and data candidates.

- 3.1.2 Data Preprocessing

All raw datasets are processed into the following format: {”instruction”: ⟨instruction⟩, ”input”: ⟨input (can be empty)⟩, ”output”: ⟨output⟩}. For certain CoT datasets, we directly follow [76] to use the template from FLAN [57] and transform the original data into the data format above.

- 3.1.3 LoRA Fine-Tuning

- 3 METHODOLOGY 3.1 LoRA Bank Construction

For each dataset, we fine-tune pre-trained base models and derive their LoRA weights. These LoRA models are used for subsequent model selection. We follow [91] to set the hyper-parameters for optimization: a batch size of 2, gradient accumulation steps of 16, and an initial learning rate of 5e-5. Uniformly, all models underwent training for three epochs to ensure that each dataset is fully mastered regardless of the task difficulty. A cosine decaying schedule is adopted for adjusting the learning rate over iterations. The LoRA modules are applied on the linear embedding layers of the Query and Value matrices in self-attention. For all LoRAs, the rank was set to 16. To reduce training overhead, sequences with length exceeding 1024 were truncated.

The open-source community has witnessed a significant increase in the number of LoRA models and high-quality datasets. To ensure experimental reproducibility, rationality, and comparability, we have selected thirty-eight representative and widely-used instruction datasets from the Huggingface [96] to construct a rich and reliable LoRA bank.

- 3.1.1 Data Sources Specifically, the data to construct our LoRA Bank are summarized

- as follows: ARC [22], Winogrande [1], GSM8K [23], PiQA [12], CommonSenseQA [86], RACE [42], MBPP [6], MathQA [5], Esnli [14], ECQA [4], CREAK [66], GPT4Tools [105], AQuA [51], QASC [39], QED [43], StrategyQA [30], SensemakingQA [90], Toolformer [79], HellaSwag [109], SiQA [78], BoolQ [21], Dolly [31], WizardLM [102], WebGPT [65], Lima [117], Code-Alpaca [92], ThoughtSource [68], CAMEL [46]. We choose these datasets for two reasons: 1) they provide a comprehensive coverage of mainstream tasks and domains, and 2) their quality is confirmed through massive downloads worldwide and positive feedback comments from researchers.

#### 3.2 K-shot Guided Expert Model Selection

In a bank of LoRA models, it becomes essential to appropriately select the most relevant task expert candidates [106], [107]. However, few studies have explored model selection options for language models, especially in the context of K-shot settings. Furthermore, K-shot labeled data are typically sparse and uniform in the input embedding space, which means that the evaluation results of models on such limited data cannot reflect their true expertise level.

In this section, we propose to select highly-potential models that would perform well on any task of interest before conducting the task-specific fine-tuning. We hypothesize that two following perspectives should be prioritized during model selection: 1) Does the model possess the necessary knowledge for the tasks of interest? and 2) Can the model adequately follow the instructions under the tasks of interest? Accordingly, we first consider two indicators in estimation of the empirical risk on Kshot samples: 1) the specific evaluation metric for performance measurement, and the perplexity of a LLM in modeling the answers in an auto-regressive manner. Specifically, exact match accuracy is adopted in the present study as the default evaluation metric on the generated responses. If the answer choice precisely matches the model’s output, it is considered correct; otherwise, it is deemed incorrect. However, the empirical risk merely computed by performance is prone to be affected by poor instruction following capabilities of LLMs, where models that produce correct but unparsable answers are severely underestimated. The post-processing techniques cannot handle all the corner cases of answers that are formatted unexpectedly. On the other hand, models that randomly guess the answer choices in QA tasks might be over-estimated. Such misjudgement originates from the low informative evaluation metric that fails to comprehensively assess whether the model can understand and handle the given task. Therefore, it calls upon the perplexity of the model on Kshot sequences as a straightforward complement. To reduce biased measurement merely sourcing from one indicator, we take into consideration of three aspects: 1) perplexity, 2) performance, and

- 3) diversity.

- 3.2.1 Perplexity

The perplexity of a LLM on auto-regressive modeling of the answers serve as an effective indicator of the model’s capability. Given a model m parameterized by θm, an input sequence x, and its expected output sequence y, the perplexity of language modeling is defined with cross-entropy:

|y|

PPL(x,y,θm) = exp(−

i=1

log P(y(i)|x,y(<i);θm)), (1)

where P(y(i)|x,y(<i)) is the predicted probability of the i-th token y(i) of y given the sequences x and y(<i).

However, as explained in the drawbacks of exact match on multiple-choice problems, the perplexity computed solely on the ground-truth answer options in many QA tasks still suffers from inaccurate estimation of model capabilities. To address this issue, we utilized an advanced open model [61] to expand the answers with CoT rationales. Subsequently, we calculated the reasoning perplexity of LoRA models by considering both the predicted answer choices and the rationales behind them:

|yˆ|

PPL(x,y,θˆ m) = exp(−

log P(ˆy(i)|x,yˆ(<i);θm)), (2) yˆ = Φ(x,y), (3)

i=1

where Φ(x,y) represents the CoT expansion process given both the input x and the output y. For such CoT rationales, we target

- at multiple-choice datasets whose outputs are only answer options (e.g., A and B). We use the CoT-formatted answers rewritten by the WizardLM2-8x22B [61] for reasoning explanations (see Fig. 3

Prompt Template to Generate CoT Rationales

Read the question and answer. According to the answer, present your thought about the solution.

- - Please think step by step and provide the answer first then with its explanation.
- - Your explanation should not exceed three sentences.

Question: {Question} Answer: {Answer}.

Fig. 3. Expansion of the CoT rationales on K-shot instructions.

for detailed prompts). Such rationales improve the transparency and interpretability of the decision-making process, which benefits accurate model selection by estimating the uncertainty of any LLM on the rationales under the context of instruction. Besides, answers with reasoning process in advance are more effective in fine-tuning models, where the justified rationales well calibrate the response towards the ultimate correct answers. As illustrated in Fig. 2, models with higher reasoning perplexity might achieve better performance before task-specific tuning, but their performance gains brought by fine-tuning are not as strong as models with lower reasoning perplexity.

For the K-shot DK = {(x1,y1),(x2,y2),...,(xK,yK)}, |DK| = K, the total reasoning perplexity is defined as:

PPLR(m,DK) =

PPL(xi,Φ(xi,yi),θm). (4)

(xi,yi)∈DK

- 3.2.2 Performance The evaluation metric measured on K-shot directly reflects a model’s ability to solve the corresponding task in an “end-to-end” way. We denote such evaluation results as the performance, which is defined by any metric calculated by comparing the generated responses y˜ and the ground-truth answers y. In line with the task requirement, a post-processing step might be involved to extract formatted answers from responses. It can be implemented through certain function f(·), i.e., y˜′ = f(˜y). Common post-processing operations include threshold-based truncation and probability-tocategory mapping. Out of simplicity, the performance of a model

m on the dataset DK can be defined as the accuracy of matching the post-processed y˜′ with the ground-truth y:

Acc(m,DK,f) =

1 N (x

i,yi)∈DK

(˜yi′ = yi),

y˜i′ = f(˜yi),

(5)

where (·) is the indicator function that equals to 1 when the condition holds true and 0 otherwise. y˜i is the predicted response by θm given the input xi.

- 3.2.3 Diversity To fully harness the task-related knowledge inherent in models from the LoRA Bank, our objective is to select all candidate models that are likely to solve the downstream tasks. To expand the knowledge base, it is essential to “deduplicate” the selected models so that each candidate model contributes to the accumulation of skills. Therefore, diversity is ensured so that different selected experts possess distinct abilities. Contrary to existing

Algorithm 1 K-shot Guided Expert Model Selection Input: A model bank B, number of candidate models M, number

studies [41], [47], [85] that primarily focus on combining experts without considering their relationships, we highlight the concept of group diversity for retrieving expert candidates. It refers to the variety of the selected models in a group. A high intra-group diversity lays a solid foundation for exploitation of task-relevant yet complementary knowledge. Given a group of N expert models BN = {m1,m2,...,mN}, the group diversity ΩB

of selected experts N, K-shot data DK, and post-processing function f(·)

Output: Selected experts BE

- 1: Compute the reasoning perplexity PPLR(m,DK) and the performance by accuracy Acc(m,DK,f) for ∀m ∈ B
- 2: for all m ∈ B do
- 3: Extract the ranking RP(m) ← rank(PPLR(m,DK)) by sorting from smallest to largest
- 4: Extract the ranking RA(m) ← rank(Acc(m,DK,f)) by sorting from largest to smallest
- 5: end for
- 6: Find the top M models BM with the smallest sum of rankings in both perplexity and performance

BM∗ ← arg minB

M⊂B,|BM|=M m∈BM(RP(m) + RA(m))

- 7: Calculate the group diversity ΩB

N for all N-tuples ∀BN ⊂ BM,|BN| = N

- 8: for all BN ⊂ BM do
- 9: Extract the ranking RD(BN) ← rank(ΩB

N

) by sorting from largest to smallest

- 10: end for
- 11: Find the N-tuple BN ⊂ BM,|BN| = N with the smallest sum of rankings in perplexity, performance, and group similarity

BN∗ ← arg minB

N⊂BM∗ ,|BN|=N m∈BN(RP(m) + RA(m)) + RD(BN)

- 12: return BE = BN

N is defined as the inverse of the sum of cosine similarities between the parameters of each model pair E(mi) and E(mj):

E(mi) · E(mj) ∥E(mi)∥∥E(mj)∥

2 N(N − 1) m

)−1,

ΩB

= (

N

i∈BN mj̸=mi

(6) where E(mi) denotes the flattened matrices of layers in the model mi. The cosine similarity between the paired mi and mj is computed matrix by matrix and averaged over all layers.

- 3.2.4 Selection Mechanism

The overall pipeline of model selection is illustrated in Fig. 4 and detailed in Alg. 1. One important principle behind our pipeline design is to comprehensively employ indicators from various aspects, which reduces the bias by partial measurement.

We first calculate the reasoning perplexity and the performance in accuracy for each model in the bank B based on Eq. 4 and Eq. 5. We sort models by perplexity and performance respectively from smallest to largest and from largest to smallest. The rankings by perplexity and performance are respectively denoted as RP and RA. Then, we select M candidate models BM∗ from the LoRA Bank B with minimum sum of rankings RP and RA. In this way, models in lack of task-relevant knowledge and skills are filtered out in advance to greatly reduce the computation overhead on diversity measurement. Subsequently, we calculate the group diversity ΩB

Specifically, we train the router in such a manner that the model autonomously allocates different tokens to appropriate experts. Given an input x, the output of a N-expert MoE system at the l-th layer gl(x) is computed as the weighted sum of the outputs from each expert:

N on all combinations of N-tuples BN from BM∗ , and sort these tuples by intra-group diversity from largest to smallest for the ranking RD. Finally, we sum up all the rankings of RP, RA, and RD and choose the top N models BN∗ ⊂ BM,|BN∗ | = N for initializing our MoE system BE.

N

gl(x) =

Gli(x) · gil(x), (8)

Compared with other methods, our selection approach is more robust to variation of K-shot across tasks due to its all-sided evaluation in perplexity, performance, and diversity. Besides, it consumes fewer computational resources for two reasons: 1) Lowquality and impotent models are eliminated before the exhaustive computation of intra-group diversity. 2) Each model performs evaluation only once on K-shot data.

i=1

where Gli(x) represents the i-th element of the gating vector Gl(x) ∈ RN, controlling the contribution of the i-th expert.

gil(x) is the output from the i-th expert at the l-th layer. A simple yet efficient implementation of the gating network Gl(·)

is a single fully-connected layer Wgl, where the gating vector is obtained by matrix multiplication between Wgl and gil(x). Only the top-k activated experts are selected with their gating weights re-normalized via softmax function:

###### 3.3 Mixture-of-Experts Initialization

Upon selecting the most promising experts, our objective is to efficiently utilize their potentials at respective domains. The linear arithmetic composition offers a straightforward approach to benefit from our LoRA bank [20], [33], [110]:

##### Gl(x) = softmax top-k gil(x) · Wgl , (9)

where top-k(·) keeps the value of the largest k elements unchanged and returns the other elements as −∞. We fine-tune all LoRA experts and routers simultaneously (see Fig. 5).

N

Wˆ = W +

wi∆Wi, (7)

i=1

#### 3.4 K-shot Guided Sim-Div Data Selection

where W indicates the original parameter of a pre-trained model and ∆Wi denotes the i-th LoRA variant with Ni=1 wi = 1. Nevertheless, such a naive method of assigning weights to different LoRAs does not promise a dynamic and flexible routing mechanism, restricting its quick adaptation to any task of interest. A more reasonable strategy would be a mixture of expertise.

In situations where few annotated data are available under the task of interest, it is prone to overfitting if our MoE system is directly fine-tuned on these datapoints. A common approach to address this issue is data augmentation [26], [82], [95], [114], which mitigates the risk of degeneration via lexical or semantic augmentation

Performance Evaluation on K-shot Data Model*LoRA

Model*LoRA

A:

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

（ … ）

- 1
- 2
- 3

[Figure 50]

- 1
- 2
- 3

- Model 1: B. Pour it into a jar
- Model 2: A. Pour it onto a plate

RankingbyPerformance&

RankingbyPerformance&

[Figure 51]

N-tuple 1 Group Diversity=0.114

Perplexity&Diversity

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

（ … ）

Perplexity

Q: When boiling butter, when it's ready, you can:

[Figure 59]

[Figure 60]

- A. Pour it onto a plate
- B. Pour it into a jar

N-tuple 2 Group Diversity=0.004

[Figure 61]

…

…

###### …

CoT: Pouring it into a jar is a suitable option because … The Answer is: B. Pour it into a jar.

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

（ … ）

[Figure 66]

[Figure 67]

|B|

N

N-tuple 𝑪𝑴𝑵 Group Diversity=0.251

[Figure 68]

- Model 1: Perplexity=3.998
- Model 2: Perplexity=2.787

[Figure 69]

𝑀 Candidates 𝑁 Models

Bank 𝐵 Reasoning Perplexity on K-shot Data

Experts 𝐵

In-tuple Diversity

- Fig. 4. The overall pipeline of our K-shot guided model selection strategy. A comprehensive assessment in terms of perplexity, performance, and diversity is conducted on each model for expert selection. Given K-shot data, we evaluate a model’s performance via exact match accuracy on the directly inferred results. The reasoning perplexity is obtained by computing the perplexity on auto-regressive modeling of the CoT rationales towards answers of K-shot. The top-M ranked candidate models are first selected to save computation of the subsequent intra-group diversity, where every N-tuple out of the M candidates are involved. The N models that share the lowest similarity in parameters (i.e., the largest group diversity) contribute to the initialization of a MoE system.

|Feed Forward Network<br><br>Add & Norm …<br><br>[Figure 70]<br><br>[Figure 71]<br><br>Selected Unselected<br><br>[Figure 72]<br><br>Expert 0 Expert 1 Expert 2 Expert N<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>Router<br><br>[Figure 77]<br><br>…<br><br>[Figure 78]<br><br>[Figure 79]| |
|---|---|
| | |

× 𝐿

Multi-Head Attention

Add & Norm

Hidden States

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Learnable Frozen

…

[Figure 84]

Selected Unselected

[Figure 85]

Expert 0 Expert 1 Expert 2 Expert N

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Router

[Figure 96]

…

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

- Fig. 5. The architecture of our MoE system. It is implemented with LoRA modules, where the selected models from the LoRA bank are trained with an additional router to learn to assign different tokens to the responsible experts. Each token is routed to the top-k activated experts with their representations multiplied by the corresponding routing weights for normalization.

similar distribution with samples in the downstream tasks ought to be prioritized. Accordingly, we propose the similarity-first and diversity-aware principle to guide the data selection process.

- 3.4.1 Data Encoding

In the first step, we perform encoding of the raw instruction texts for their projection into the embedding space. Given the K-shot DK and a set of S open-source samples DS = {(x1,y1),(x2,y2),...,(xS,yS)},|DS| = S, we aim to find the most relevant datapoints in DS based on their similarity with samples in DK. Given a pre-trained encoding model h parameterized by θh, we obtain the representation of each sample ui with both input and output via:

ui = h([xi,yi];θh), (10)

where [·,·] denotes the concatenation operation in python syntax. The embeddings of samples (xi,yi) from DK and DS are respectively denoted as UK and US. In practice, we employ the pre-trained BGE model [17] as h, which is fine-tuned on the XLM-RoBERTa [24] with self-distillation on corpus of multiple languages, tasks, and granularities. Specifically for extraction of ui, we use the prefix prompt of query_instruction_for_retrieval: “Represent the following sentence for similar task retrieval: ⟨[xi,yi]⟩”.

- 3.4.2 Similarity-First

on the original text inputs. Such manually designed techniques tend to work on traditional discriminative models by imposing perturbations to the decision boundary. However, for generative LLMs, vanilla data augmentation methods do not inherently inject any new knowledge into the model, nor do they fundamentally improve the diversity of the maintained instruction dataset.

We calculate the cross-dataset similarity between DK and DS to select a subset DC ⊂ DS that resembles DK the most. First, we define a distance d(·,·) to measure the similarity between two encoded samples ui and uj. Without losing generality, we demonstrate an intuitive implementation of similarity metric via cosine distance:

In the light of this statement, we propose to leverage opensource data for task-oriented augmentation. It has three advantages including: 1) high cost-efficiency of utilizing the massive and free open-source datasets, 2) prevention of overfitting by introducing diverse and beneficial instructions, and 3) improvement of tokenwise collaboration between experts via acquiring novel knowledge. When it comes to the selection of specific data instances, a new question is raised: What type of open-source data should be favored to improve the performance on our task of interest? In the present study, we assume that datapoints that share a

ui · uj ∥ui∥∥uj∥

, (11)

sim(ui,uj) = 1 − dcos(ui,uj) =

where each pair of ui ∈ UK and uj ∈ US constructs the entry Aij of the similarity matrix A ∈ RK×S:

##### Aij = sim(ui,uj). (12)

Algorithm 2 K-shot Guided Sim-Div Data Selection Input: A set of open-source instruction samples DS, K-shot data

###### Embedding Space

[Figure 109]

[Figure 110]

K-shot Data Open-Source Data

Open-Source Dataset

Q: Give three tips for staying healthy. A: 1. Eat a balanced and nutritious diet: Make sure …

[Figure 111]

DK, encoding model h, pairwise similarity metric sim(·,·), data budget C, and similarity threshold τ

Pretrained Encoder

Output: Augmented dataset DA

[Figure 112]

K-Shot Data

- 1: Encode each (xi,yi) ∈ DS and (xj,yj) ∈ DK by h respectively for the embedding sets of US and UK
- 2: Compute the cross-dataset similarity matrix between DK and DS as A ∈ RK×S, where Aij = sim(ui,uj),ui ∈ UK,uj ∈ US
- 3: Find the top C datapoints in DS that share the highest similarity with samples in K-shot DK

DC∗ = arg maxD

C⊂DS,|DC|=C (xj,yj)∈DC(maxi Aij)

- 4: Compute the intra-dataset similarity matrix within DC as I ∈ RC×C, where Iij = sim(ui,uj),ui ∈ UC,uj ∈ UC,UC = {ui|(xi,yi) ∈ DC}
- 5: for all (xi,yi) ∈ DC,(xj,yj) ∈ DC,i ̸= j do
- 6: if Iij > τ then
- 7: if maxr Ari > maxr Arj then
- 8: DC ← DC \ {(xj,yj)}
- 9: else
- 10: DC ← DC \ {(xi,yi)}
- 11: end if
- 12: end if
- 13: end for
- 14: return DA = DC

Q: Who is likely to use a comb? A: A. Medicine cabinet…

Diversity-Based Data Selection

Similarity-Based Data Selection

B. Barber…

- Fig. 6. The overall pipeline of our K-shot guided sim-div data selection. We extract the embeddings of both K-shot and open-source data through pre-trained models. In the embedding space, samples that are close to K-shot are identified and prioritized for task augmentation. Simultaneously, to ensure data diversity, we remove batches of data with near-duplicate semantics.

Then, we pinpoint top C samples in DS that share the most similarity with DK by maximizing along the rows of A:

DC∗ = arg max

Aij). (13)

(max

i

DC⊂DS,|DC|=C (xj,yj)∈DC

The reasons why DC is first selected out of DS are two-fold: 1) It ensures that the samples which enjoy a high level of similarity with DK are prioritized while those dissimilar ones would not be unexpectedly introduced into the candidate set during the subsequent selection. 2) It reduces the computation overhead of diversity measurement since the entire set DS shrinks into a much smaller candidate set DC with C ≪ S.

4 EXPERIMENTS

- 3.4.3 Diversity-Aware

4.1 Experiments Setup

We remove duplicates in DC to improve its overall diversity for the selected dataset DA. A greater level of diversity not only improves the token-wise cooperation between experts on broader domains and topics but also reduces the overfitting of the MoE system on K-shot datapoints. Specifically, we remove a subset of data with excessively high semantic similarity. We use the same distance metric defined in Eq. 11 and compute the intra-dataset similarity matrix I. The pairwise similarity is measured on both ui ∈ UC and uj ∈ UC, where UC = {ui|(xi,yi) ∈ DC}:

In this section, we validate the effectiveness of our proposed method through a series of experiments on various tasks of interest. To begin with, we use six popular open-source datasets, namely ARC-Challenge, ARC-Easy [22], PiQA [12], BoolQ [21], MBPP [6], and GSM8K [23], to serve as our evaluation sets under tasks of interest. These datasets cover a diverse range of fields, including examination, knowledge question-answer (QA), common sense reasoning, mathematical problems, and code generation. For each dataset, we randomly sample K labeled instruction-response pairs from its official training set as K-shot. In consideration of the response format, we employ a post-processing approach customized for each task to standardize the outputs for fair comparison. For examination and QA datasets in the LoRA bank whose ground-truth answers are multiple-choices, we implement the rationale expansion and train the corresponding LoRA models on the CoT-formatted answers for improved task comprehension. Details about the datasets and the performance of each LoRA model can be found in the supplementary materials.

##### Iij = sim(ui,uj). (14)

We follow the SemDeDup [2], [87] to perform semantic deduplication by thresholding with τ. If the similarity between any two instructions exceeds τ, we discard the one whose similarity with K-shot DK is lower. By this means we maintain the overall diversity of the final selected dataset DA. The entire process of data selection is elaborated in Alg. 2.

To enhance the performance of our MoE system, we employ data augmentation by introducing additional datasets for data selection and MoE fine-tuning (see Secs. 3.4 and 3.5). Specifically, CommonSenseQA [86] and SiQA [78] are involved for multiplechoice QA tasks (ARC-Challenge, ARC-Easy, PiQA, and BoolQ). We choose CommonSenseQA and SiQA in data augmentation for two reasons: 1) they share the same output formats with the downstream tasks in the testing sets; 2) they span a wide variety of common knowledge and social situations. While for code generation (MBPP) and mathematical reasoning (GSM8K) tasks, we choose WizardLM [102] because the dataset itself claims to be effective in improving LLMs in solving coding and maths

###### 3.5 Mixture-of-Experts Fine-Tuning

We combine the augmented dataset DA and the K-shot dataset DK for optimizing both the routing weights and the experts of our MoE system θMoE, where the cross-entropy loss is employed to supervise the language modeling on the outputs of the last L-th MoE layer (Eq. 8):

|y|

log P(y(i)|x,y(<i);θMoE), (15)

L(x,y,θMoE) = −

i=1

where P(y(i)|x,y(<i);θMoE) = softmax(gL([x,y(<i)]))y

(i).

TABLE 1 Comparison with baselines and SOTA methods on six downstream tasks. The DK, DA, and DT respectively denote the K-shot data, the selected open-source data for augmentation, and the entire training set of the downstream tasks.

Model Method MoE K-shot External ARC-c ARC-e BoolQ PiQA GSM8K MBPP Avg.

Base Model × × × 33.90% 49.91% 47.86% 53.05% 16.68% 18.20% 36.60% Random × DK × 40.11% 58.42% 60.12% 55.47% 18.53% 20.00% 42.11% Random† × DK DA 43.72% 61.19% 61.89% 54.29% 20.92% 22.40% 44.07% Expert LoRA × DT DT 44.07% 62.61% 63.39% 56.58% 22.88% 22.40% 45.32% Source Best × × × 49.49% 69.31% 65.81% 61.53% 22.88% 24.60% 48.94% Source Best SFT × DK DA 55.13% 72.14% 68.91% 64.32% 23.82% 24.40% 51.45% MixLoRA [45] ✓ DK × 41.24% 57.92% 60.56% 54.79% 18.53% 20.20% 42.21% MixLoRA† [45] ✓ DK DA 52.88% 65.69% 60.51% 56.20% 20.54% 22.00% 46.30% LoRAHub [33] ✓ DK × 43.01% 57.43% 62.91% 53.12% 19.34% 23.80% 43.27% LoRAHub† [33] ✓ DK DA 43.38% 60.30% 61.65% 53.48% 21.53% 22.80% 43.86% SMEAR [64] ✓ DK DA 54.98% 69.85% 66.32% 64.32% 23.98% 22.20% 50.28% PEMs [110] ✓ DK DA 50.12% 69.43% 67.12% 63.21% 19.83% 24.00% 48.95% Poly [74] ✓ DK DA 53.55% 70.37% 66.29% 63.54% 22.06% 22.20% 49.67% π-tuning [97] ✓ DK DA 55.78% 64.48% 67.31% 59.82% 24.18% 18.60% 48.36% Arrow [67] ✓ DK DA 56.94% 70.37% 66.94% 63.22% 22.59% 24.00% 50.68% PHATGOOSE [63] ✓ DK DA 46.44% 49.41% 58.22% 56.25% 21.53% 22.40% 42.38% Ours ✓ DK DA 57.76% 73.60% 69.45% 65.13% 24.83% 24.20% 52.50%

LLaMA2-7B

Base Model × × × 60.68% 73.54% 55.96% 57.67% 45.56% 36.00% 54.90% Random × DK × 68.81% 81.31% 69.30% 66.97% 47.54% 36.80% 61.79% Random† × DK DA 76.27% 87.65% 75.44% 63.16% 48.74% 40.20% 65.24% Expert LoRA × DT DT 77.97% 90.30% 80.12% 68.12% 49.73% 37.60% 67.31% Source Best × × × 80.00% 90.30% 80.12% 71.76% 51.55% 40.60% 69.06% Source Best SFT × DK DA 78.43% 88.76% 83.29% 76.01% 52.43% 40.00% 69.82% MixLoRA [45] ✓ DK × 69.15% 83.42% 74.53% 67.93% 48.67% 37.80% 63.58% MixLoRA† [45] ✓ DK DA 75.93% 81.83% 84.52% 71.76% 48.74% 38.80% 66.93% LoRAHub [33] ✓ DK × 69.21% 84.14% 80.24% 65.31% 47.94% 39.60% 64.41% LoRAHub† [33] ✓ DK DA 70.84% 85.00% 78.53% 67.84% 47.30% 39.80% 64.89% SMEAR [64] ✓ DK DA 79.32% 90.12% 85.27% 74.29% 52.84% 37.80% 69.94% PEMs [110] ✓ DK DA 79.84% 89.87% 87.41% 74.07% 48.43% 39.60% 69.87% Poly [74] ✓ DK DA 77.62% 87.47% 86.81% 73.11% 51.47% 40.20% 69.45% π-tuning [97] ✓ DK DA 76.87% 87.80% 81.67% 74.25% 46.47% 38.80% 67.64% Arrow [67] ✓ DK DA 81.01% 89.59% 86.81% 77.85% 52.91% 41.00% 71.53% PHATGOOSE [63] ✓ DK DA 71.18% 85.53% 79.11% 70.62% 46.92% 38.80% 65.36% Ours ✓ DK DA 81.43% 92.29% 89.71% 78.89% 52.91% 41.40% 72.77%

Mistral-7B

† Results are obtained by additionally introducing the same data augmentation technique DA as ours for performance comparison.

problems. The selection of open-source data for augmentation adheres to the principle that no data leakage occurs with respect to the testing sets. The availability and quality of these datasets are guaranteed according to the feedback from developers in the community. In total, a mixture of K-shot data from downstream tasks and the selected data from open-source datasets are utilized for fine-tuning our MoE system.

###### 4.2 Implementation Details

For preparation of 38 models in our LoRA bank, both LLaMA27B (Base) [88] and Mistral-7B (Base) [36] are investigated in the present study. We empirically set the number of candidate models M = 8, the number of chosen experts N = 4, and the number of selected experts k = 2. By default, we set K = 50 for K-shot data. Due to the randomness of sampling K-shot as the seed instructions, we perform sampling three times and report the averaged experimental results with three different sets of Kshot to ensure the reliability of testing. For data augmentation, the total number of open-source samples S is respectively 43K and 143K for the combined CommonSenseQA and SiQA, and the WizardLM. By default, the data budget C is set to 1K and the similarity threshold is τ = 0.9. During experiments, we set the hyper-parameters following [91] for training the MoE system: a batch size of 2, gradient accumulation steps of 16, and a learning rate of 5e-5. We fine-tune all MoE models for five epochs and their

convergence is guaranteed. We do not observe further benefits with longer training. For all LoRA models incorporated in the LoRA Bank, the rank was set to 16. To optimize memory utilization, the employed training setting was the Deepspeed [77] zero stage

- 3. All models were trained with PyTorch [70] Transformers [96] (version 4.36.1) on NVIDIA V100 GPUs with Float16 mode.

Explanations on all the symbols and the settings of all hyperparameters can be found in the supplementary materials.

- 4.3 Baselines

We compare the proposed method with five vanilla baselines: 1) the pre-trained base (Base Model), 2) the randomly selected model from the LoRA bank fine-tuned only on K-shot data (Random) 3) the base model respectively fine-tuned on the entire training set DT of each task of interest (Expert LoRA), 4) the expert in the LoRA bank that achieves the best evaluation results respectively on the testing set of each downstream task (Source Best), and 5) each source best expert fine-tuned on the same augmented open datasets with ours (Source Best SFT). It is noted that baselines of Random, Expert LoRA, Source Best, and Source Best SFT are optimized task-by-task, and therefore their results on each individual task are reported respectively. Note that for the Random method, random sampling is performed three times and results are averaged over three fine-tuned models.

Furthermore, we compare with several SOTA methods including: 1) training a MoE system with randomly initialized LoRA

models and routers (MixLoRA) [45], 2) combining pre-trained LoRA models and fine-tuning only the routers with K-shot data (LoRAHub) [33], 3) performing model merging and fine-tuning on LoRA models (PEMs) [110], 4) optimizing the routing parameters by minimizing the language modeling loss on K-shot data and augmentend data (Poly) [74], 5) retrieving and constructing task experts via the fisher information matrix (FIM) embeddings (πtuning) [97], 6) dynamically selecting the most appropriate experts via singular vectors (Arrow) [67], and 7) post-hoc routing of experts with pre-trained gating vectors (PHATGOOSE) [63]. All these methods unearth available LoRA models for maximizing the generalization of LLMs either on unseen domains and tasks or under task-specific supervised scenarios. Out of comparability, we implemented SOTA methods with their officially released codes. We adopted the same hyper-parameter settings with ours for optimization of these methods except LoRAHub, where its default hyper-parameters are kept unchanged. For methods that did not originally bring in K-shot and external open source datasets under investigation, we conduct extra comparison by modifying them to fit the same setting with ours where both K-shot and data augmentation are seamlessly integrated into their pipelines.

###### GSM8K

###### ARC-c

[Figure 113]

[Figure 114]

ExpertIndex

ExpertIndex

Layers

Layers

- Fig. 7. The activation patterns of experts across layers. The distribution of the routing policy is relatively sparse on both the tasks of GSM8K and ARC-c.

models merely by performance in accuracy (RA), and 4) selection of top-ranked models by perplexity and performance without considering intra-group diversity (RD). The selected N experts by each method are used to initialize a MoE system and fine-tuned following the same data augmentation pipeline on DK ∪ DA. For the Random method, we also perform sampling three times for initialization and fine-tuning of three MoE systems. Their averaged results are reported here.

- 4.5.2 The selection of promising models by comprehensive consideration of performance, reasoning perplexity, and group diversity outperforms the vanilla, simple ones that solely emphasize one aspect Firstly, we found that the performance of LoRA models randomly selected from the LoRA bank for tasks of interest is incompetent. This suggests that only the most relevant set of models can enhance the performance of specific tasks. As shown in Table 2, our proposed model selection strategy outperforms the approaches that solely rely on K-shot performance and reasoning perplexity. Moreover, using a single model with either the lowest perplexity or the maximized performance for identical initialization of our MoE system (also known as upcycling) results in lower performance. Compared with a group of experts with diversity concerns, the MoE system initialized by the same experts only achieve similar results with Source Best SFT, indicating that highly similar models are detrimental to the training of MoE.

[Figure 115]

[Figure 116]

Reasoning Perplexity Vanilla Perplexity

Accuracy

Accuracy

Fig. 8. Comparison between the model selection strategies with the vanilla perplexity on answer choices and the reasoning perplexity on CoT rationales of the ARC-c dataset. A high negative correlation is observed between the performance (exact match accuracy) and the reasoning perplexity. On the contrary, it is difficult to accurately identify promising candidate experts from the vanilla perplexity due to its weak association with performance.

- 4.5.3 The reasoning perplexity outperforms the vanilla perplexity in model selection We compared the reasoning perplexity and the vanilla perplexity (without CoT rationales) for model selection. As shown in Fig. 8,

###### 4.4 Results on Tasks of Interest

- 4.4.1 Comparison with Baselines

The experimental results can be found in Table 1. The comparison between LoRA and MixLoRA is straightforward, as it highlights the importance of our model selection. The rational use of opensource knowledge for specific tasks proves to be superior to the single LoRA and the vanilla mixing policy in MixLoRA. LoRAHub is a scheme that also maintains an open-source knowledge base. PEMs is a strategy that directly merges model parameters, which is intuitive but lacks rationality. In addition, these methods rarely consider the impact of the diversity between experts on the overall system. Our method outperforms existing approaches, showcasing its effectiveness.

- 4.4.2 Visualization of Experts The activation patterns of individual experts are illustrated in

- Fig. 7. For each layer, we calculate the average activation rate of each expert. Specifically, for each token in a sequence, we only assign “1” to the chosen experts and “0” to the remaining ones. Then, we calculate the proportions of tokens that are routed through each expert, where the number of tokens per expert is normalized over that of all experts. Such a proportion of expert activation reflects the learning preference and the effectiveness of the MoE system. We observe that on both the GSM8K and the ARC-c datasets, the distribution of activation is relatively sparse across layers and each expert is nearly uniformly activated over the entire datasets. Such activation patterns confirm that the MoE system does not collapse equivalently into a single model, where the routing mechanism responds to tokens adaptively and each expert contributes to the overall MoE system.

- 4.5 Ablation Study on Model Selection

- 4.5.1 Baselines

In terms of our model selection strategy, four baselines are introduced for comparison: 1) random selection of models from the LoRA Bank (Random), 2) selection of top-ranked models merely by reasoning perplexity (RP), 3) selection of top-ranked

Ablation study on the model selection strategy for initialization of the MoE system. The RP , RA, and RD respectively represent the ranking by reasoning perplexity, the ranking by performance in accuracy, and the ranking by group diversity.

Perplexity

Performance

Group Diversity

Model Method MoE

ARC-c ARC-e BoolQ PiQA GSM8K MBPP Avg.

Base Model × × × × 33.90% 49.91% 47.86% 53.05% 16.68% 18.20% 36.60% Random ✓ × × × 47.81% 63.42% 64.19% 55.26% 17.87% 22.40% 45.16%

✓ RP × × 54.32% 68.54% 69.71% 63.39% 23.71% 24.00% 50.61% ✓ × RA × 55.16% 69.32% 66.15% 62.32% 23.71% 24.00% 50.11% ✓ RP RA × 54.64% 71.94% 67.99% 65.32% 23.49% 23.40% 51.13% ✓ RP RA RD 57.76% 73.60% 69.45% 65.13% 24.83% 24.20% 52.50%

LLaMA2-7B

Ours

Base Model × × × × 60.68% 73.54% 55.96% 57.67% 45.56% 36.00% 54.90% Random ✓ × × × 76.51% 84.41% 85.03% 70.99% 47.83% 39.00% 67.29%

✓ RP × × 79.32% 89.31% 86.91% 75.14% 49.37% 40.40% 70.08% ✓ × RA × 80.36% 91.12% 83.33% 76.19% 49.26% 40.20% 70.08% ✓ RP RA × 78.64% 88.12% 84.11% 77.21% 52.65% 40.20% 70.16% ✓ RP RA RD 81.43% 92.29% 89.71% 78.89% 52.91% 41.40% 72.77%

Mistral-7B

Ours

TABLE 3 Ablation study on the data selection strategy for augmentation in fine-tuning the MoE system.

Model Method Similarity Diversity ARC-c ARC-e BoolQ PiQA GSM8K MBPP Avg.

K-shot Only × × 50.41% 69.41% 66.76% 62.91% 23.00% 23.60% 49.35% Random × × 52.13% 68.67% 67.01% 59.21% 23.42% 23.80% 49.04%

Cosine × 54.58% 72.49% 68.52% 66.48% 20.70% 23.80% 51.10% Cosine ✓ 57.76% 73.60% 69.45% 65.13% 24.83% 24.21% 52.50% Convex ✓ 57.28% 70.54% 69.09% 66.63% 24.18% 24.00% 51.95% KDE (γ=2) ✓ 60.67% 75.66% 67.51% 59.08% 22.74% 23.80% 51.58% KDE (γ=10) ✓ 57.28% 70.01% 67.84% 66.45% 23.50% 24.00% 51.51% Cosine Repr Filter [55] 56.94% 73.36% 70.70% 65.23% 24.03% 24.20% 52.41%

LLaMA2-7B

Ours

K-shot Only × × 79.32% 88.67% 79.23% 73.21% 46.67% 40.00% 67.85% Random × × 78.42% 84.56% 83.12% 71.21% 48.96% 38.80% 67.51%

Cosine × 82.13% 90.30% 90.01% 75.73% 51.55% 42.40% 72.02% Cosine ✓ 81.43% 92.29% 89.71% 78.89% 52.91% 41.40% 72.77% Convex ✓ 82.02% 91.45% 91.43% 79.53% 52.84% 41.40% 73.11% KDE (γ=2) ✓ 83.45% 92.34% 88.53% 77.92% 53.43% 41.40% 72.85% KDE (γ=10) ✓ 80.43% 92.74% 90.55% 79.43% 53.92% 42.00% 73.18% Cosine Repr Filter [55] 83.72% 90.30% 88.59% 77.63% 52.38% 41.60% 72.37%

Mistral-7B

Ours

with the expanded CoT process, the reasoning perplexity of candidate models has a higher correlation with their performance on testing sets. Consequently, model selection by the reasoning perplexity is more robust than that by the vanilla perplexity.

###### 4.6 Ablation Study on Data Selection

- 4.6.1 Baselines

To validate our data selection strategy, we compare with MoE systems fine-tuned on: 1) K-shot data only, 2) randomly selected data from DS (Random), 3) data selected by cosine similarity without considering diversity, 4) data selected by convex hull with the proposed diversity control, 5) data selected by kernel density estimation (KDE) [19] with the proposed diversity control, 6) data selected by cosine similarity with Repr Filter diversity control [55]. All methods were performed on the same MoE system but using different DA. Except K-shot MoE, we set the data budget C to be 1K for all methods. Due to diversity control, Repr Filter and the proposed method discard duplicates and reduce the size of DC for a slightly smaller DA. To mathematically describe the convex hull-based selection, we first define the convex hull UKC of the given K-shot data DK with their embeddings UK:

UKC = {u|u =

K

n

λiui,ui ∈ UK,λi ≥ 0,

i=1

i=1

λi = 1}, (16)

where λi are the coefficients that determine the convex combination of samples in UK. Then, the convex-hall-based data selection is performed by randomly sampling a subset DC ⊂ DS,|DC| = C whose embeddings are located inside UKC, namely uj ∈ UKC,uj ∈ UC holds true for ∀(xj,yj) ∈ DC. The kernel density estimation (KDE) is a non-parametric geometric technique to estimate the probability density function of a random variable. Given the K-shot DK and their embeddings UK, the estimated density probability of any sample u in the embedding space is presented by:

K

u − ui γ

1 K · γ

,ui ∈ UK, (17)

Ker

p(u) =

i=1

where Ker(·) denotes the Gaussian kernel function for distance measurement and γ is the bandwidth controlling the smoothness of the density function. For KDE-based data selection, we sample a subset DC ⊂ DS,|DC| = C according to the probability p(uj),uj ∈ US. Samples that are closer to K-shot data on average are of higher likelihood to be selected.

4.6.2 The open-source data augmentation by similarityfirst and diversity-aware selection further improves the MoE system

We investigated the performance of fine-tuning under different data augmentation scenarios. As shown in Table 3, data augmen-

[Figure 117]

[Figure 118]

Ablation on Diversity (Mistral-7B) Ablation on Diversity (LLaMA2-7B)

[Figure 119]

[Figure 120]

Accuracy

Accuracy

Cosine

Convex Hull

Data Budget C Data Budget C

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Ablation on Quantity (Mistral-7B) Ablation on Quantity (LLaMA2-7B)

Accuracy

Accuracy

Data Budget C Data Budget C

KDE (𝛾=2)

KDE (𝛾 =10)

Fig. 9. As the data budget C increases, the performance exhibits a trend of initially rising and then declining. It indicates that the augmentation by truly relevant data improves performance. However, the dominance of irrelevant, redundant samples degrades performance as they dilute the contributions of K-shot similar samples. Diversity also plays an important role in balancing distributions, which alleviates overfitting in fine-tuning the MoE system.

Fig. 10. T-SNE visualization of different similarity-based selection methods including cosine distance, convex hull, and KDE. Blue dots, green dots, and orange dots respectively represent the entire open-source datapoints, the K-shot data, and the selected C samples for augmentation.

tation based on cosine similarity alone effectively enhances the performance of our MoE on tasks of interest. However, as shown in Fig. 9, with the increasing amount of data DA, the performance exhibits a pattern of an initial growth followed by a decline. Such a trend indicates that at the beginning, the involvement of more taskrelevant open data exerts an instant positive effect on performance, which not only brings in knowledge but also stabilizes MoE optimization. Accompanied by the enlarged dataset, an excessive amount of irrelevant data begin to dominate. They dilute the contribution of K-shot related datapoints and easily cause the system’s forgetting of task-specific skills and knowledge.

Additionally, we examined the role of diversity in data selection. Fig. 9 reveals that filtering out data with high semantic similarity leads to a noticeable improvement in the model’s performance. Especially when the number of data budget C increases, deduplication becomes an indispensable step to mitigate overfitting. We set different thresholds τ of 1, 0.9, 0.85, and 0.8 for controlling the diversity of the selected DA, which respectively corresponds to the proportions of 100%, 90%, 50%, and 10% of the candidates DC. It is observed that without diversity control (τ = 1), the exact match accuracy decreases with respect to our default setting τ = 0.9, confirming the validity of the proposed diversity control. However, it might go from one extreme to another if a strict threshold (τ = 0.8) is applied to remove any two samples with certain similarity. We believe the excessive removal of semantically similar datapoints is detrimental to the generalization of the fine-tuned model for two reasons: 1) the task-relevant datapoints are too scarce to effectively calibrate responses and optimize expert cooperation; 2) the MoE system easily overfits the K-shot samples. Therefore, it is critical to choose an appropriate threshold that only removes semantic duplicates for diversified data augmentation without aggressively discarding similar-yetbeneficial samples.

- 4.6.3 The implementation of similarity-based selection by cosine distance can be replaced with other competitive geometric sampling techniques

Besides, we conducted a comparison between different sampling schemes, including the vanilla selection by minimizing cosine distance (Eq. 13), random sampling within the convex hull enclosed by K-shot data (Eq. 16), and sampling based on the estimated density probability (Eq. 17). Given the same budget C, we present a t-SNE [89] visualization of the selected datapoints to highlight the differences between these methods (see Fig. 10).

Table 3 reports that the convex hull-based sampling achieves relatively good results in reasoning tasks such as BoolQ and PiQA. KDE behaves more robustly as it tend to select samples encompassing K-shot as cluster centers. Especially on math and coding tasks, the KDE-based sampling improves the initial MoE the most because it simultaneously introduces samples around K-shot and those extrapolated along boundaries, effectively expanding the knowledge. Furthermore, we notice that the bandwidth γ affects the rate of attenuation in density estimation. A larger γ results in a sparser distribution of the selected data, allowing greater diversity.

4.7 Ablation Study on K, N, and k

- 4.7.1 The marginal benefits of increasing K on performance decrease

We experimented from K = 1 to K = 50 (see Table 4) and found that our approach already achieves highly competitive results even under K = 5. It demonstrates that the proposed method requires only a small amount of human-verified data to quickly produce expert models with strong professional capabilities. The continual increase of K from 5 to 50 further improves the performance of the overall MoE system. However, the gains are diminishing, suggesting the trade-off between the efforts of collecting more K-shots and their ultimate profits should be considered.

Ablation study on the K of K-shot.

Model Method ARC-c ARC-e BoolQ PiQA GSM8K MBPP Avg.

K = 1 50.68% 65.08% 62.14% 61.53% 17.29% 20.40% 46.19% K = 5 57.00% 72.64% 68.41% 64.10% 22.98% 23.80% 51.49% K = 20 57.49% 73.39% 69.01% 64.82% 24.57% 24.00% 52.20% K = 50 57.76% 73.60% 69.45% 65.13% 24.83% 24.20% 52.50%

LLaMA2-7B

K = 1 76.27% 89.24% 83.73% 75.77% 45.56% 37.80% 68.06% K = 5 80.72% 90.98% 89.12% 78.31% 51.92% 40.80% 71.98% K = 20 81.02% 91.27% 89.53% 78.75% 52.85% 41.20% 72.43% K = 50 81.43% 92.29% 89.71% 78.89% 52.91% 41.40% 72.77%

Mistral-7B

TABLE 5 Error rates of the MoE system and its four experts (Mistral-7B) on the mixed cases of the ARC-c, where at least one expert succeeds and one expert fails on each sample.

Evaluation Results on Experts Mixed Sample Performance

[Figure 125]

[Figure 126]

Count

|Model|Expert 1 Expert 2 Expert 3 Expert 4 MoE<br><br>|
|---|---|
|Error|55.0% 62.5% 37.5% 45.0% 32.5%|

All Correct All Incorrect Mixed Case Index

As confirmed in Fig. 7, the activation of routers is adequately dispersed across layers. It suggests that the optimization of routing weights converges even with 1K data. Moreover, we believe that the quantity of learnable parameters and routing weights in a MoE system should be positively correlated with the volume of data. For larger language models (e.g., 70B and 110B), the data budget C should exceed 1K and potentially reaches up to 10K. In our preliminary experiments, we also find that the complexity of the downstream tasks is worthy of consideration. For simpler tasks such as multiple-choice QA, the appropriate data volume hovers around 0.5K-1K. On the contrary, for challenging tasks such as math and coding problems, 5K-10K data are encouraged. The perplexity of each task by measuring the IFD score [48] of Kshot might be indicative in setting the optimal C.

Fig. 11. Evaluation results of each expert (Mistral-7B) in a fine-tuned MoE system on the ARC-c dataset. All Correct and All Incorrect respectively represent the number of cases where all experts succeed and fail. Mixed represents the number of cases where at least one expert perform correctly and at least one incorrectly. For each testing case, correct and incorrect experts are respectively highlighted in green and red.

- 4.7.2 The dissimilarity between expert candidates is the key to maintaining a task-oriented knowledgesupplementary MoE system We explored the diversity among expert models for building a successful MoE system. As shown in Fig. 11, we divide the testing cases of the ARC-c into all correct, all incorrect, and mixed, where the constituting experts respectively all succeed, all fail, and perform in between. For all correct cases, all experts are capable of independently solving these cases without external assistance. While for all incorrect cases, case-by-case knowledge injection becomes a requisite to master the missing preliminaries. For the mixed cases, it is the judgement by the routing that resolves disagreement between experts and makes the best use of model ensemble for improvement. Table 5 reports the error rate of each expert and the MoE system on the ARC-c. It can be observed that each expert is adept at different testing cases, exhibiting a varied distribution across experts. Besides, the error rate of the overall MoE system is much lower than that of each expert, confirming the advantages of knowledge expansion. During model selection, the heterogeneity of experts measured by their parameters guarantees the success of a constructed MoE, where the knowledge of each expert supplements to the others.
- 4.7.3 The amount of data points for fine-tuning the MoE system should be optimized for the task of interest

TABLE 6 Ablation study on the number of experts N. Results on all downstream tasks are averaged.

Number of Experts N 2 4 6 8

Model

LLaMA2-7B 52.19% 52.50% 52.43% 50.49% Mistral-7B 70.45% 72.77% 72.19% 69.84%

4.7.4 The number of experts does not need to be large for high capacity

In Table 6, we observe that a small number (e.g., four) of highquality experts can yield better performance in a MoE system. One reason behind the degraded performance with an increasing number of experts is that the training of a MoE system with numerous experts requires more open-source datasets during optimization. Under the same K-shot settings with data augmentation, the optimization of an eight-expert MoE system is more challenging. It becomes more difficult for the routing itself to converge with fewer data points.

The amount of data required to effectively train a MoE system, especially the token-wise routing mechanism, remains an intriguing question to discuss. In the present study, we utilized a total of 1K data for fine-tuning the MoE system by default. Compared with a single LoRA model, the MoE system yields an increase of +1.05% on LLaMA and +2.95% on Mistral under the same data volume.

TABLE 7 Ablation study on the number of selected experts k in a 4-expert MoE. Results on all downstream tasks are averaged.

PEFT methods. We leave the issue of applicability beyond LoRA modules to be addressed in the future work.

Number of the Selected Experts k 1 2 3 4

Model

LLaMA2-7B 49.93% 52.50% 51.19% 52.89% Mistral-7B 71.82% 72.77% 73.12% 71.64%

- 4.7.5 The optimal number of selected experts is not consistent across model architectures We initialized the MoE system with four experts, and fine-tuned four MoE models respectively with top-1, top-2, top-3, and all selected experts during routing. Table 7 confirms that when only one expert is selected, the optimization of the MoE system is sub-optimal. With an increase in the number of selected experts k, the performance gradually improves. However, we notice that the optimal number of selected experts is different between the LLaMA and Mistral models. Meticulous efforts on choosing the number of selected experts might be needed to achieve the best performance of the given MoE system.

5 DISCUSSION

- 5.1 Non-use of Meta Info from the Datasets and Models

In the present study, we do not rely on metadata (e.g., descriptions) to select datasets and models for the following three reasons. First, the descriptions of various open-source models and datasets are often insufficiently detailed. A thorough understanding of the dataset composition or model characteristics demands delving into the semantics of datapoints and the representations of the model layers. Second, each dataset may consist of subsets spanning a wide range of domains and tasks. Identifying the most relevant data points necessitates data selection in the embedding space for retaining only samples that contribute to mastering task-dependent skills. Third, it is challenging to keep track of all the constantly updating metadata on the internet due to privacy and copyrights issues. On the other hand, it is more feasible to only access datasets and models as long as they are available online.

###### 5.2 Applicability and Availability

The proposed comprehensive, scalable pipeline begins with the collection and preparation of datasets and models, followed by the selection of expert candidates and data augmentation for downstream tasks, and culminates at the construction and optimization of a MoE system for the optimized task expertise. Note that a majority of open-source LLMs are variants of the LLaMA and Mistral families that undergo instruction tuning and preference alignment in different ways. A simple search on Huggingface retrieves hundreds of models of the identical size and architecture, profit from the contributions of the open-source community to the development of LLMs. Therefore, the availability of a large model bank is ensured in practical scenarios. It is noted that in the present study, LoRA modules, namely the offsets to the parameters of the base model, are investigated as the main existing forms of model parameters. However, there exist numerous open models that are developed with other PEFT techniques such as adapters. Our approach necessitates the computation of inter-model similarity between weights, which may not be compatible across different

### 6 CONCLUSION

In this study, we developed an efficient and scalable pipeline to fully utilize K-shot data from downstream tasks of interest for augmentation of existing LLMs in task expertise. The Kshot samples play an important role in both model selection and data augmentation. Specifically, we comprehensively assess the capability of each expert candidate on K-shot data by simultaneously measuring its performance in exact match accuracy and its reasoning perplexity via auto-regressive language modeling. We bring in diversity for extending the knowledge boundary enclosed by the selected experts, which lays a solid foundation for initializing a flexible MoE system. For optimizing the token-wise collaboration between experts, we propose to fine-tune the MoE with a data augmentation technique. Under such circumstance, a similarity-first, diversity-aware data selection method is developed where the K-shot data steer the selection of task-relevant samples in the embedding space. Diversity is also highlighted by removing the semantic duplicates for alleviating overfitting under data scarce scenarios. Experimental results demonstrate that the proposed method outperforms existing approaches in producing a MoE system of specific task expertise. Extensive ablation studies confirm the validity of the proposed selection methods in pinpointing the most promising models and appropriate data with K-shot, demonstrating a cost-efficient pipeline to excavate open knowledge for customized skill consolidation.

### REFERENCES

- [1] Winogrande: An adversarial winograd schema challenge at scale. 2019.
- [2] Amro Abbas, Kushal Tirumala, D´aniel Simig, Surya Ganguli, and Ari S Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication. arXiv preprint arXiv:2303.09540, 2023.
- [3] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [4] Shourya Aggarwal, Divyanshu Mandowara, Vishwajeet Agrawal, Dinesh Khandelwal, Parag Singla, and Dinesh Garg. Explanations for commonsenseqa: New dataset and models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3050–3065, 2021.
- [5] Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. arXiv preprint arXiv:1905.13319, 2019.
- [6] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.
- [7] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [8] Lukas Balles, Giovanni Zappella, and C´edric Archambeau. Gradientmatching coresets for rehearsal-based continual learning. arXiv preprint arXiv:2203.14544, 2022.
- [9] Samyadeep Basu, Philip Pope, and Soheil Feizi. Influence functions in deep learning are fragile. arXiv preprint arXiv:2006.14651, 2020.
- [10] Joshua Belofsky. Token-level adaptation of lora adapters for downstream task generalization. In Proceedings of the 2023 6th Artificial Intelligence and Cloud Computing Conference, pages 168–172, 2023.
- [11] Gantavya Bhatt, Yifang Chen, Arnav M Das, Jifan Zhang, Sang T Truong, Stephen Mussmann, Yinglun Zhu, Jeffrey Bilmes, Simon S Du, Kevin Jamieson, et al. An experimental design framework for label-efficient supervised finetuning of large language models. arXiv preprint arXiv:2401.06692, 2024.

- [12] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.
- [13] Jonathan Brophy, Zayd Hammoudeh, and Daniel Lowd. Adapting and evaluating influence-estimation methods for gradient-boosted decision trees. Journal of Machine Learning Research, 24(154):1–48, 2023.
- [14] Oana-Maria Camburu, Tim Rockt¨aschel, Thomas Lukasiewicz, and Phil Blunsom. e-snli: Natural language inference with natural language explanations. Advances in Neural Information Processing Systems, 31, 2018.
- [15] Yihan Cao, Yanbin Kang, Chi Wang, and Lichao Sun. Instruction mining: When data mining meets large language model finetuning. arXiv preprint arXiv, 2307, 2023.
- [16] Hao Chen, Yiming Zhang, Qi Zhang, Hantao Yang, Xiaomeng Hu, Xuetao Ma, Yifan Yanggong, and Junbo Zhao. Maybe only 0.5% data is needed: A preliminary exploration of low training data instruction tuning. arXiv preprint arXiv:2305.09246, 2023.
- [17] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 2024.
- [18] Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, et al. Alpagasus: Training a better alpaca with fewer data. arXiv preprint arXiv:2307.08701, 2023.
- [19] Yen-Chi Chen. A tutorial on kernel density estimation and recent advances. Biostatistics & Epidemiology, 1(1):161–187, 2017.
- [20] Alexandra Chronopoulou, Matthew E Peters, Alexander Fraser, and Jesse Dodge. Adaptersoup: Weight averaging to improve generalization of pretrained language models. arXiv preprint arXiv:2302.07027, 2023.
- [21] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.
- [22] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.
- [23] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [24] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzm´an, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. CoRR, abs/1911.02116, 2019.
- [25] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. Advances in Neural Information Processing Systems, 36, 2024.
- [26] Kaustubh D Dhole, Varun Gangal, Sebastian Gehrmann, Aadesh Gupta, Zhenhao Li, Saad Mahamood, Abinaya Mahendiran, Simon Mille, Ashish Shrivastava, Samson Tan, et al. Nl-augmenter: A framework for task-sensitive natural language augmentation. arXiv preprint arXiv:2112.02721, 2021.
- [27] Qianlong Du, Chengqing Zong, and Jiajun Zhang. Mods: Modeloriented data selection for instruction tuning. arXiv preprint arXiv:2311.15653, 2023.
- [28] Jon Durbin. Airoboros. https://github.com/jondurbin/airoboros, 2024.
- [29] Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Yu Han, and Hao Wang. Mixture-of-loras: An efficient multitask tuning for large language models. arXiv preprint arXiv:2403.03432, 2024.
- [30] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346–361, 2021.
- [31] M Hayes, A Mathur, J Xie, J Wan, S Shah, A Ghodsi, P Wendell, M Zaharia, and R Xin. Free dolly: Introducing the world’s first truly open instruction-tuned llm. 2023.
- [32] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.
- [33] Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. Lorahub: Efficient cross-task generalization via dynamic lora composition. arXiv preprint arXiv:2307.13269, 2023.

- [34] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. arXiv preprint arXiv:2212.04089, 2022.
- [35] Joel Jang, Seungone Kim, Seonghyeon Ye, Doyoung Kim, Lajanugen Logeswaran, Moontae Lee, Kyungjae Lee, and Minjoon Seo. Exploring the benefits of training expert language models over instruction tuning. In International Conference on Machine Learning, pages 14702–14729. PMLR, 2023.
- [36] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
- [37] Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng. Dataless knowledge fusion by merging weights of language models. arXiv preprint arXiv:2212.09849, 2022.
- [38] Junmo Kang, Leonid Karlinsky, Hongyin Luo, Zhen Wang, Jacob Hansen, James Glass, David Cox, Rameswar Panda, Rogerio Feris, and Alan Ritter. Self-moe: Towards compositional large language models with self-specialized experts. arXiv preprint arXiv:2406.12034, 2024.
- [39] Tushar Khot, Peter Clark, Michal Guerquin, Peter Jansen, and Ashish Sabharwal. Qasc: A dataset for question answering via sentence composition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8082–8090, 2020.
- [40] John Kirchenbauer, Garrett Honke, Gowthami Somepalli, Jonas Geiping, Daphne Ippolito, Katherine Lee, Tom Goldstein, and David Andre. Lmd3: Language model data density dependence. arXiv preprint arXiv:2405.06331, 2024.
- [41] Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. Sparse upcycling: Training mixture-ofexperts from dense checkpoints. arXiv preprint arXiv:2212.05055, 2022.
- [42] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension dataset from examinations. arXiv preprint arXiv:1704.04683, 2017.
- [43] Matthew Lamm, Jennimaria Palomaki, Chris Alberti, Daniel Andor, Eunsol Choi, Livio Baldini Soares, and Michael Collins. Qed: A framework and dataset for explanations in question answering. Transactions of the Association for computational Linguistics, 9:790–806, 2021.
- [44] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.
- [45] Dengchun Li, Yingzi Ma, Naizheng Wang, Zhiyuan Cheng, Lei Duan, Jie Zuo, Cal Yang, and Mingjie Tang. Mixlora: Enhancing large language models fine-tuning with lora based mixture of experts. arXiv preprint arXiv:2404.15159, 2024.
- [46] Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for” mind” exploration of large language model society. Advances in Neural Information Processing Systems, 36:51991–52008, 2023.
- [47] Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A Smith, and Luke Zettlemoyer. Branch-train-merge: Embarrassingly parallel training of expert language models. arXiv preprint arXiv:2208.03306, 2022.
- [48] Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, and Jing Xiao. From quantity to quality: Boosting llm performance with self-guided data selection for instruction tuning. arXiv preprint arXiv:2308.12032, 2023.
- [49] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021.
- [50] Yunxin Li, Shenyuan Jiang, Baotian Hu, Longyue Wang, Wanqi Zhong, Wenhan Luo, Lin Ma, and Min Zhang. Uni-moe: Scaling unified multimodal llms with mixture of experts. arXiv preprint arXiv:2405.11273, 2024.
- [51] Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. arXiv preprint arXiv:1705.04146, 2017.
- [52] Jerry Liu. LlamaIndex, 11 2022.
- [53] Qidong Liu, Xian Wu, Xiangyu Zhao, Yuanshao Zhu, Derong Xu, Feng Tian, and Yefeng Zheng. Moelora: An moe-based parameter efficient fine-tuning method for multi-task medical applications. arXiv preprint arXiv:2310.18339, 2023.
- [54] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation. arXiv preprint arXiv:2402.09353, 2024.

- [55] Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning. In The Twelfth International Conference on Learning Representations, 2024.
- [56] Zefang Liu and Jiahua Luo. Adamole: Fine-tuning large language models with adaptive mixture of low-rank adaptation experts. arXiv preprint arXiv:2405.00361, 2024.
- [57] Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. In International Conference on Machine Learning, pages 22631–22648. PMLR, 2023.
- [58] Keming Lu, Hongyi Yuan, Runji Lin, Junyang Lin, Zheng Yuan, Chang Zhou, and Jingren Zhou. Routing to the expert: Efficient reward-guided ensemble of large language models. arXiv preprint arXiv:2311.08692, 2023.
- [59] Max Marion, Ahmet Ust¨¨ un, Luiza Pozzobon, Alex Wang, Marzieh Fadaee, and Sara Hooker. When less is more: Investigating data pruning for pretraining llms at scale. arXiv preprint arXiv:2309.04564, 2023.
- [60] Fanxu Meng, Zhaohui Wang, and Muhan Zhang. Pissa: Principal singular values and singular vectors adaptation of large language models. arXiv preprint arXiv:2404.02948, 2024.
- [61] Microsoft. Wizardlm-2 8x22b, 2024.
- [62] Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. Orca: Progressive learning from complex explanation traces of gpt-4. arXiv preprint arXiv:2306.02707, 2023.
- [63] Mohammed Muqeeth, Haokun Liu, Yufan Liu, and Colin Raffel. Learning to route among specialized experts for zero-shot generalization. arXiv preprint arXiv:2402.05859, 2024.
- [64] Mohammed Muqeeth, Haokun Liu, and Colin Raffel. Soft merging of experts with adaptive routing. arXiv preprint arXiv:2306.03745, 2023.
- [65] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.
- [66] Yasumasa Onoe, Michael JQ Zhang, Eunsol Choi, and Greg Durrett. Creak: A dataset for commonsense reasoning over entity knowledge. arXiv preprint arXiv:2109.01653, 2021.
- [67] Oleksiy Ostapenko, Zhan Su, Edoardo Maria Ponti, Laurent Charlin, Nicolas Le Roux, Matheus Pereira, Lucas Caccia, and Alessandro Sordoni. Towards modular llms by building and reusing a library of loras. arXiv preprint arXiv:2405.11157, 2024.
- [68] Simon Ott, Konstantin Hebenstreit, Valentin Li´evin, Christoffer Egeberg Hother, Milad Moradi, Maximilian Mayrhauser, Robert Praas, Ole Winther, and Matthias Samwald. Thoughtsource: A central hub for large language model reasoning data. Scientific Data, 10(1):528, 2023.
- [69] Lucas Page-Caccia, Edoardo Maria Ponti, Zhan Su, Matheus Pereira, Nicolas Le Roux, and Alessandro Sordoni. Multi-head adapter routing for cross-task generalization. Advances in Neural Information Processing Systems, 36, 2024.
- [70] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.
- [71] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277, 2023.
- [72] Jonas Pfeiffer, Andreas R¨uckl´e, Clifton Poth, Aishwarya Kamath, Ivan Vuli´c, Sebastian Ruder, Kyunghyun Cho, and Iryna Gurevych. Adapterhub: A framework for adapting transformers. arXiv preprint arXiv:2007.07779, 2020.
- [73] Agustin Picard, Lucas Hervier, Thomas Fel, and David Vigouroux. Influenciæ: A library for tracing the influence back to the data-points. In World Conference on Explainable Artificial Intelligence, pages 193–

204. Springer, 2024.

- [74] Edoardo Maria Ponti, Alessandro Sordoni, Yoshua Bengio, and Siva Reddy. Combining parameter-efficient modules for task-level generalisation. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 687–702, 2023.
- [75] Yulei Qin, Yuncheng Yang, Pengcheng Guo, Gang Li, Hang Shao, Yuchen Shi, Zihan Xu, Yun Gu, Ke Li, and Xing Sun. Unleashing the power of data tsunami: A comprehensive survey on data assessment and selection for instruction tuning of language models. 2024.

- [76] Zheng Lin Qingyi Si. Alpaca-cot: An instruction fine-tuning platform with instruction data collection and unified large language models interface. https://github.com/PhoebusSi/alpaca-CoT, 2023.
- [77] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3505–3506, 2020.
- [78] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.
- [79] Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36, 2024.
- [80] Stephanie Schoch, Ritwick Mishra, and Yangfeng Ji. Data selection for fine-tuning large language models using transferred shapley values. arXiv preprint arXiv:2306.10165, 2023.
- [81] Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. arXiv preprint arXiv:1708.00489, 2017.
- [82] Rico Sennrich, Barry Haddow, and Alexandra Birch. Improving neural machine translation models with monolingual data. arXiv preprint arXiv:1511.06709, 2015.
- [83] Samarth Sinha, Han Zhang, Anirudh Goyal, Yoshua Bengio, Hugo Larochelle, and Augustus Odena. Small-gan: Speeding up gan training using core-sets. In International Conference on Machine Learning, pages 9005–9015. PMLR, 2020.
- [84] Zhan Su, Fengran Mo, Prayag Tiwari, Benyou Wang, Jian-Yun Nie, and Jakob Grue Simonsen. Mixture of experts using tensor products. arXiv preprint arXiv:2405.16671, 2024.
- [85] Sainbayar Sukhbaatar, Olga Golovneva, Vasu Sharma, Hu Xu, Xi Victoria Lin, Baptiste Rozi`ere, Jacob Kahn, Daniel Li, Wen-tau Yih, Jason Weston, et al. Branch-train-mix: Mixing expert llms into a mixture-ofexperts llm. arXiv preprint arXiv:2403.07816, 2024.
- [86] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937, 2018.
- [87] Kushal Tirumala, Daniel Simig, Armen Aghajanyan, and Ari Morcos. D4: Improving llm pretraining via document de-duplication and diversification. Advances in Neural Information Processing Systems, 36, 2024.
- [88] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [89] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.
- [90] Cunxiang Wang, Shuailong Liang, Yue Zhang, Xiaonan Li, and Tian Gao. Does it make sense? and why? a pilot study for sense making and explanation. arXiv preprint arXiv:1906.00363, 2019.
- [91] Eric J. Wang. Alpaca-lora. https://github.com/tloen/alpaca-lora, 2024.
- [92] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560, 2022.
- [93] Zengzhi Wang, Rui Xia, and Pengfei Liu. Generative ai for math: Part i–mathpile: A billion-token-scale pretraining corpus for math. arXiv preprint arXiv:2312.17120, 2023.
- [94] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [95] Jason Wei and Kai Zou. Eda: Easy data augmentation techniques for boosting performance on text classification tasks. arXiv preprint arXiv:1901.11196, 2019.
- [96] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, et al. Huggingface’s transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771, 2019.
- [97] Chengyue Wu, Teng Wang, Yixiao Ge, Zeyu Lu, Ruisong Zhou, Ying Shan, and Ping Luo. π-tuning: Transferring multimodal foundation models with optimal multi-task interpolation. In International Conference on Machine Learning, pages 37713–37727. PMLR, 2023.
- [98] Shengguang Wu, Keming Lu, Benfeng Xu, Junyang Lin, Qi Su, and Chang Zhou. Self-evolved diverse data sampling for efficient instruction tuning. arXiv preprint arXiv:2311.08182, 2023.

- [99] Xun Wu, Shaohan Huang, and Furu Wei. Mole: Mixture of lora experts. In The Twelfth International Conference on Learning Representations, 2023.
- [100] Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. Less: Selecting influential data for targeted instruction tuning. arXiv preprint arXiv:2402.04333, 2024.
- [101] Sang Michael Xie, Shibani Santurkar, Tengyu Ma, and Percy S Liang. Data selection for language models via importance resampling. Advances in Neural Information Processing Systems, 36:34201–34227, 2023.
- [102] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023.
- [103] Yang Xu, Yongqiang Yao, Yufan Huang, Mengnan Qi, Maoquan Wang, Bin Gu, and Neel Sundaresan. Rethinking the instruction quality: Lift is what you need. arXiv preprint arXiv:2312.11508, 2023.
- [104] Prateek Yadav, Derek Tam, Leshem Choshen, Colin A Raffel, and Mohit Bansal. Ties-merging: Resolving interference when merging models. Advances in Neural Information Processing Systems, 36, 2024.
- [105] Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, and Ying Shan. Gpt4tools: Teaching large language model to use tools via self-instruction. Advances in Neural Information Processing Systems, 36, 2024.
- [106] Yuncheng Yang, Meng Wei, Junjun He, Jie Yang, Jin Ye, and Yun Gu. Pick the best pre-trained model: Towards transferability estimation for medical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 674–683. Springer, 2023.
- [107] Kaichao You, Yong Liu, Ziyang Zhang, Jianmin Wang, Michael I Jordan, and Mingsheng Long. Ranking and tuning pre-trained models: a new paradigm for exploiting model hubs. Journal of Machine Learning Research, 23(209):1–47, 2022.
- [108] Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch. arXiv preprint arXiv:2311.03099, 2023.
- [109] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.
- [110] Jinghan Zhang, Junteng Liu, Junxian He, et al. Composing parameterefficient modules with arithmetic operation. Advances in Neural Information Processing Systems, 36:12589–12610, 2023.
- [111] Jipeng Zhang, Yaxuan Qin, Renjie Pi, Weizhong Zhang, Rui Pan, and Tong Zhang. Tagcos: Task-agnostic gradient clustered coreset selection for instruction tuning data. arXiv preprint arXiv:2407.15235, 2024.
- [112] Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adaptive budget allocation for parameter-efficient fine-tuning. In The Eleventh International Conference on Learning Representations, 2023.
- [113] Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, et al. Instruction tuning for large language models: A survey. arXiv preprint arXiv:2308.10792, 2023.
- [114] Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28, 2015.
- [115] Bo Zhao and Hakan Bilen. Dataset condensation with distribution matching. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 6514–6523, 2023.
- [116] Ziyu Zhao, Leilei Gan, Guoyin Wang, Yuwei Hu, Tao Shen, Hongxia Yang, Kun Kuang, and Fei Wu. Retrieval-augmented mixture of lora experts for uploadable machine learning. arXiv preprint arXiv:2406.16989, 2024.
- [117] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36, 2024.
- [118] Daquan Zhou, Kai Wang, Jianyang Gu, Xiangyu Peng, Dongze Lian, Yifan Zhang, Yang You, and Jiashi Feng. Dataset quantization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17205–17216, 2023.
- [119] Han Zhou, Xingchen Wan, Ivan Vuli´c, and Anna Korhonen. Autopeft: Automatic configuration search for parameter-efficient finetuning. Transactions of the Association for Computational Linguistics, 12:525–542, 2024.
- [120] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In Proceedings of

the IEEE/CVF conference on computer vision and pattern recognition, pages 16816–16825, 2022.

APPENDIX

- .1 Datasets

We present the statistics of publicly available datasets used in the present study for constructing our LoRA bank in Table 8.

- .2 Evaluation Results of Models in the Bank

The evaluation results of LoRA models in the bank are reported in Tables 9 and 10 respectively for LLaMA2 [88] and Mistral [36] families.

Each model is fine-tuned on one high-quality open dataset under the same experimental settings. Results confirm that all these LoRA models exhibit different advantages on the downstream tasks of interest, implying that each model possesses a unique skill acquired from the fine-tuned dataset. However, we also observe that certain models are highly competitive nearly on all tasks (e.g., RACE for the LLaMA2 and ARC-e for the Mistral). Such phenomenon may be ascribed to the fact that these two datasets emphasize the reasoning and reading comprehension capabilities of LLMs, where the acquired skills appear more general in solving various tasks. Besides, we notice that the two model families (LLaMA vs. Mistral) behave inconsistently in the evaluation results even under the same fine-tuning settings. It suggests that different pre-training corpus are collected and memorized for language modeling, and thereafter the two base pre-trained models already differ in the parameterized knowledge before instruction fine-tuning. Consequently, when optimizing any LLM for the task-of-interest, there exists no golden rule of selecting one generally-applicable, ultimately-superior instruction dataset which would work all the time. In practice, one should dynamically select potential models following the proposed pipeline instead of empirically sampling from the bank. Last but not least, it is noted that the diversity of the candidate models in the bank has to be guaranteed to ensure that each model owns knowledge that can supplement to the other one. Such bank diversity allows a generalized skill-set to be utilized via the constructed MoE system for solving tasks of interest.

- .3 Mathmatical Notations

In this section, we present the description of all math notations in the manuscript (see Table 11).

- .4 Implementation Details

The detailed settings of all hyper-parameters for optimization of the LoRA models and the MoE system are listed in Table 12.

TABLE 8 Statistics of Datasets (Part 1).

###### Dataset Task Train Val Test Example

ARC-c Exam 1.1K 299 1.1K {”instruction”: ”Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat? A. dry palms\nB. wet palms\nC. palms covered with oil\nD. palms covered with lotion\nAnswer:”, ”input”: ””, ”output”: ”A. dry palms”}

ARC-c CoT-formatted

Exam 1.1K 299 1.1K {”instruction”: ”Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat? A. dry palms\nB. wet palms\nC. palms covered with oil\nD. palms covered with lotion\nAnswer:”, ”input”: ””, ”output”: ”Rubbing two surfaces together generates heat due to friction. In the case of warming hands, the effectiveness of this process depends on the coefficient of friction between the surfaces. Dry palms are likely to produce the most heat when rubbed together because they have a higher coefficient of friction compared to wet palms, which would slip more easily and thus generate less heat. Palms covered with oil or lotion would also have reduced friction due to the lubricating effect of these substances, leading to less heat generation upon rubbing. Therefore, dry palms are the most effective choice for generating heat through friction. The answer is A. dry palms”}

ARC-e Exam 2.2K 570 2.3K {”instruction”: ”Question: Which factor will most likely cause a person to develop a fever? A. a leg muscle relaxing after exercise\nB. a bacterial population in the bloodstream\nC. several viral particles on the skin\nD. carbohydrates being digested in the stomach\nAnswer:”, ”input”: ””, ”output”: ”B. a bacterial population in the bloodstream”}

Coreference Resolution

WinoGrande

9.2K 1267 – {”instruction”: ”Which of the following is a good sentence:\nA. John moved the couch from the garage to the backyard to create space. The garage is small.\nB. John moved the couch from the garage to the backyard to create space. The backyard is small.\nAnswer:”, ”input”: ””, ”output”: ”A. John moved the couch from the garage to the backyard to create space. The garage is small.”}

GSM8K Math 7.4K 0 1.3K {”instruction”: ”Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?”, ”input”: ””, ”output”: ”Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n”}

Commonsense Reasoning

PIQA

16.1K 1838 3K {”instruction”: ”Question: When boiling butter, when it’s ready, you can \nA. Pour it onto a plate\nB. Pour it into a jar\nAnswer:”, ”input”: ””, ”output”: ”B. Pour it into a jar”}

Commonsense Reasoning

CommonSense QA

9.7K 1221 1.1K {”instruction”: ”Who is likely to use a comb?A.medicine cabinet\n B.trashcan\n

C.suitcase\n D.pocket\n E.barber shop”, ”input”: ””, ”output”: ”E.barber shop”} RACE

Reading Comprehension

87.8K 4887 4.9K {”instruction”: ”Mrs Green lives in a small village. Her husband is dead, but she has one son. His name is Jack. He is twenty-one. He worked in a shop in the village and lived with his mother, but then he finds a job in a faraway town named Green Sea. He goes and lives there. Mrs Green is not happy about it.\nOne day Mrs Green is angry. She gets on a train and goes to her son’s house in Green Sea. Then she says to him, \”Jack, why do you never telephone me?\”\nJack smiles, \”But, Mother, you don’t have a telephone.\”\n\”No,\” she answers, \”I don’t have, but you have one!\”\nGreen Sea is the small village.A.in\n B.next to\n C.near\n D.far from”, ”input”: ””, ”output”: ”D.far from”}

MBPP Coding 374 90 500 {”instruction”: ”Write a python function to find the first repeated character in a given string.”, ”input”: ””, ”output”: ”def first repeated char(str1):\r\n for index,c in enumerate(str1):\r\n if str1[:index+1].count(c) >1:\r\n return c \r\n return \”None\””}

TABLE 8 Statistics of Datasets (Part 2).

###### Dataset Task Train Val Test Example

Mathqa Math 29.4K – – {”instruction”: ”6 workers should finish a job in 8 days . after 3 days came 4 workers join them . how many days m do they need to finish the same job ?\na ) 3 , b ) 4 , c ) 5 , d ) 6 , e ) 7”, ”input”: ””, ”output”: ”\”let rate of one worker be r = >( 6 * r ) * 8 = 1 ( rate * time = work ) = >r = 1 / 48 = >work remaining after 3 days 1 - ( 3 * 6 ) / 48 = 30 / 48 after 4 ppl joined in ( ( 6 + 4 ) * time ) / 48 = 30 / 48 time m = 3 days to finish the task imo a\””}

Esnli CoT 36K – – {”instruction”: ”Please answer the following question by reasoning step-by-step. Premise: \”Man scaling wall with fire in hand.\”\\nHypothesis: \”A man holding fire in his hand is trying to escape by scaling a wall.\”\\nDo we know that the hypothesis entailed by the premise?. Step-by-step reasoning:”, ”input”: ””, ”output”: ”Just because a man is scaling a wall doesn’t imply he is trying to escape. So the answer is it is not possible to tell.”}

ECQA CoT 7.1K – – {”instruction”: ”What might a person see at the scene of a brutal killing?\\nOptions:\\n- bloody mess\\n- pleasure\\n- being imprisoned\\n- feeling of guilt\\n- cake\nLet’s think carefully first. Step-by-step reasoning process:”, ”input”: ””, ”output”: ”Bloody mess is covered or stained with blood. A person might see a bloody mess at the scene of a brutal killing. So the final answer is bloody mess.”}

CREAK CoT 6.9K – – {”instruction”: ”Claim: \”Only people named Floyd wearing pink are allowed to attend Pink Floyd concerts.\”\\nIs the claim above correct, and can it be verified by human common sense and without a web search?\\nOptions:\\n- yes\\n- no Step-by-step reasoning process:”, ”input”: ””, ”output”: ”The rock group would not be as popular is they had such requirements for their concerts. Therefore, the final answer is no.”}

GPT4Tools ToolUse 71.4K – – {”instruction”: ”GPT4Tools can handle various text and visual tasks, such as answering questions and providing in-depth explanations and discussions. It generates human-like text and uses tools to indirectly understand images. When referring to images, GPT4Tools follows strict file name rules. To complete visual tasks, GPT4Tools uses tools and stays loyal to observation outputs. Users can provide new images to GPT4Tools with a description, but tools must be used for subsequent tasks.\nTOOLS:\n——\n\nGPT4Tools has access to the following tools:\n\n>Detect the Give Object: useful when you only want to detect or find out given objects in the pictureThe input to this tool should be a comma separated string of two, representing the image path, the text description of the object to be found\n>Pose Detection On Image: useful when you want to detect the human pose of the image. like: generate human poses of this image, or generate a pose image from this image. The input to this tool should be a string, representing the image path\n>Remove Something From The Photo: useful when you want to remove and object or something from the photo from its description or location. The input to this tool should be a comma separated string of two, representing the image path and the object need to be removed.\n\nTo use a tool, please use the following format:\n\n“‘\nThought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [Detect the Give Object, Pose Detection On Image, Remove Something From The Photo]\nAction Input: the input to the action\nObservation: the result of the action\n“‘\n\nWhen you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n\n“‘\nThought: Do I need to use a tool? No\nAI: [your response here]\n“‘\n\nFollow file name rules and do not fake non-existent file names. Remember to provide the image file name loyally from the last tool observation.\n\nPrevious conversation:\n\nHuman: Provide an image named jmblbwhh.png. Description: A woman sitting next to buildings made with gingerbread. Understand the image using tools.\nAI: Received.\n\nNew input: Detect the gingerbread houses:\nGPT4Tools needs to use tools to observe images, not directly imagine them. Thoughts and observations in the conversation are only visible to GPT4Tools. When answering human questions, repeat important information. Let’s think step by step.\nThought: Do I need to use a tool? Yes\nAction: Detect the Give Object\nAction Input: jmblbwhh.png, gingerbread houses\nObservation: [output from detection tool]\n”,”input”: ””,”output”: ”Thought: Do I need to use a tool? No\nAI: [output from detection tool]\n\n”}

AQuA CoT 2.7K – – {”instruction”: ”Question: Rs. 5600 is divided into three parts A, B and C. How much A is more than C if their ratio is 1/7:1/7:1/14?\\nOptions:\\n(A) 300\\n(B) 992\\n(C) 1120\\n(D) 552\\n(E) 312 Let’s think first. Some random reasoning:”, ”input”: ””, ”output”: ”1/7:1/7:1/14 = 2:2:1\\n1/5*5600 = 1120\\n2240-1120 = 1120 The final answer: (C).”}

QASC CoT 1K – – {”instruction”: ”What is the process by which living things give rise to offspring?\\nOptions:\\n(A) DNA\\n- (B) photosynthesis\\n- (C) bird\\n- (D) sex\\n- (E) subtraction\\n- (F) gametes\\n(G) eggs\\n- (H) ovum Let’s solve this problem gradually.\n”, ”input”: ””, ”output”: ”Reproduction is the process by which living things give rise to offspring. Sex equals reproduction. Sex is the process by which living things give rise to offspring. So the final answer is (D).”}

TABLE 8 Statistics of Datasets (Part 3).

###### Dataset Task Train Val Test Example

QED CoT 5.1K – – {”instruction”: ”Answer the following Q with stream of consciousness. Passage: Webbed toes is the common name for syndactyly affecting the feet. It is characterised by the fusion of two or more digits of the feet. This is normal in many birds, such as ducks; amphibians, such as frogs; and mammals, such as kangaroos. In humans it is considered unusual, occurring in approximately one in 2,000 to 2,500 live births.\\n\\nQuestion: Based on this passage, what is the medical term for webbed toes?”, ”input”: ””, ”output”: ”The relevant information is: Webbed toes is the common name for syndactyly affecting the feet. Therefore, the final answer is syndactyly affecting the feet.”}

StrategyQA CoT 2K – – {”instruction”: ”Answer the following question by reasoning step-by-step. Are more people today related to Genghis Khan than Julius Caesar?”, ”input”: ””, ”output”: ”Julius Caesar had three children. Genghis Khan had sixteen children. Modern geneticists have determined thatout of every 200 men today has DNA that can be traced to Genghis Khan. Therefore, the final answer is yes.”}

SensemakingQA CoT 6K – – {”instruction”: ”Of the following two sentences, which one is against common sense?\\nOptions:\\n- Sentence A: \”He poured orange juice on his cereal.\”\\nSentence B: \”He poured milk on his cereal.\”\\n Let’s be accurate as possible. So think first.”, ”input”: ””, ”output”: ”Orange juice does not taste good on cereal.\nThe final answer: Sentence A.”}, {”instruction”: ”Let’s think step by step! Which of the following two sentences does not make sense?\\nOptions:\\n- Sentence A: \”He drinks apple.\”\\n- Sentence B: \”He drinks milk.\”\\n\n”, ”input”: ””, ”output”: ”An apple is a whole food and unable to be drunk without being juiced. The answer: Sentence A.”

Toolformer (th=0.6) ToolUse 2.4K – – {”instruction”: ”toolformer: enabled\ntoolformer access: wikipedia\nA wrapper around Wikipedia. Useful for when you need to display general information about people, places, companies, historical events, or other subjects found in an encyclopedia, displays a snippet summary of the topic. Input should be a search query.\nwikipedia(query)\nProvide a summary of a famous historical event based on the input event given”, ”input”: ”French Revolution”, ”output”: ”Summary of the French Revolution: wikipedia(’French Revolution’)”}

Commonsense Reasoning

HellaSwag

39.9K – 10K {”instruction”: ”[header] How to reset a vizio remote [title] Remove your remote’s batteries. [step] They’re typically found in a slot that’s either on the bottom or in the back of the remote. [title] Press and hold the remote’s power button.\nQuestion: Which ending makes the most sense?A. [step] This allows the device to power off. [title] Press and hold the power button on the remote while you wait to hear a response.B. [step] It’s located near the bottom of the remote, right in the. [title] Release the power button when you’re sure it’s covered.C. [step] It’s usually at the top of the remote’s surface. [title] Release the power button after five seconds.D. [step] Hold down the power button as close to the batteries as possible. This will power up your device and allow the lower battery to refill with power.”, ”input”: ””, ”output”: ”C. [step] It’s usually at the top of the remote’s surface. [title] Release the power button after five seconds.”}

Commonsense Reasoning

SiQA

33.4K – 1.9K {”instruction”: ”Addison put Kendall’s cost-cutting ideas into action.\nWhat will happen to Addison?\nA. kendall will be thanked\nB. get promoted\nC. lose their job”, ”input”: ””, ”output”: ”B. get promoted”}

Knowledge Question Answering

BoolQ

2.4K 277 3K {”instruction”: ”Good Samaritan laws offer legal protection to people who give reasonable assistance to those who are, or who they believe to be, injured, ill, in peril, or otherwise incapacitated. The protection is intended to reduce bystanders’ hesitation to assist, for fear of being sued or prosecuted for unintentional injury or wrongful death. An example of such a law in common-law areas of Canada: a good Samaritan doctrine is a legal principle that prevents a rescuer who has voluntarily helped a victim in distress from being successfully sued for wrongdoing. Its purpose is to keep people from being reluctant to help a stranger in need for fear of legal repercussions should they make some mistake in treatment. By contrast, a duty to rescue law requires people to offer assistance and holds those who fail to do so liable.\nQuestion: do good samaritan laws protect those who help at an accident\nA. Yes\nB. No\nAnswer:”, ”input”: ””, ”output”: ”A. Yes”}

TABLE 8 Statistics of Datasets (Part 4).

###### Dataset Task Train Val Test Example

Dolly CoT 15K – – {”instruction”: ”Which American president spoke Dutch as their first language?”,”input”: ”Van Buren received a basic education at the village schoolhouse, and briefly studied Latin at the Kinderhook Academy and at Washington Seminary in Claverack. Van Buren was raised speaking primarily Dutch and learned English while attending school; he is the only president of the United States whose first language was not English. Also during his childhood, Van Buren learned at his father’s inn how to interact with people from varied ethnic, income, and societal groups, which he used to his advantage as a political organizer. His formal education ended in 1796, when he began reading law at the office of Peter Silvester and his son Francis.”,”output”: ”President Van Buren spoke Dutch as his first language. He is the only president of the United States of America who spoke English as a second language.”}

WizardLM General 143K – – {”instruction”: ”As an online platform teacher named Aimee, you possess impeccable credentials which include a Bachelor of Science degree in Industrial and Labor Relations from Cornell University, expertise in the English language, and intermediate proficiency in both Chinese and Spanish. Additionally, your professional experience as a STEAM teacher at UN Women in Singapore has honed your skills in teaching children from the ages of 6-11 and working with students from all levels of education. Your exceptional teaching abilities in spoken English and pronunciation paired with your personal strengths of being informed, patient, and engaging make you an ideal teacher for students seeking to improve their English language skills. Can you provide a short, concise, and unique English self-introduction in bullet point form that would attract students to enroll in your course?”,”output”: ”Sure, here are some bullet points for your English self-introduction:\n\n- Hi there! My name is Aimee and I’m thrilled to be your online English teacher!\n- With a Bachelor’s degree from Cornell University and experience teaching STEAM at UN Women in Singapore, I know how to help students learn effectively.\n- I’m passionate about language and have intermediate proficiency in both Chinese and Spanish, so I understand the challenges of learning a new language.\n- In my classes, I focus on improving spoken English and pronunciation, as well as building vocabulary and grammar skills.\n- But learning English doesn’t have to be boring! I’m an engaging and patient teacher who loves to have fun and make learning enjoyable.\n- So, whether you’re a beginner or an advanced student, I’m confident that I can help you improve your English skills and achieve your goals. Let’s get started!”,”input”: ””}

ShareGPT General 330K – – {”instruction”: ”Summarize the main ideas of Jeff Walker’s Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...”,”input”: ””,”output”: ”Here are the main ideas of Jeff Walker’s Product Launch Formula that can be applied by a growth marketing agency for their clients:\n\n1. Identify the target audience and their needs: Understand the ideal customer for the product or service, and create a messaging that resonates with them.\n2. Pre-launch: Build anticipation and excitement for the launch by creating buzz, gathering testimonials and case studies, and using social media to create awareness.\n3. Launch: Use a well-crafted launch sequence to maximize sales and conversions. This can include offering bonuses, creating scarcity, and using a deadline to create urgency.\n4. Post-launch: Follow up with customers, gather feedback, and continue to provide value to keep them engaged and loyal.\n5. Create a product suite: Once the initial product is successful, expand the offering by creating additional products or services that address other needs of the same audience.\n6. Continual optimization: Continually monitor and optimize the launch process and product suite to improve results.\n7. Build a community: Use the launch process to build a community of customers who are passionate about the product and can help spread the word.\n8. Use automation: Use technology and automation to streamline the launch process and improve efficiency.”}

GPT-Teacher-Code Coding 4.5K – – {”instruction”: ”Design a C++ function to calculate the factorial of a non-negative integer.”,”input”: ”5”,”output”: ”#include <iostream>\n\nunsigned long long factorial(int n) {\n if (n <= 1) {\n return 1;\n }\n return n * factorial(n - 1);\n}\n\nint main() {\n int num = 5;\n std::cout <<’Factorial of ’ <<num <<’ is ’ <<factorial(num) <<std::endl;\n return 0;\n}”}

TABLE 8 Statistics of Datasets (Part 5).

###### Dataset Task Train Val Test Example

GPT-Teacher-Roleplay Coding 3.1K – – {”instruction”: ”Imagine you are a detective trying to solve a murder mystery in a fictional Victorian-era town. Describe the scene of the crime, the main suspects, and your plan to investigate the case.”, ”input”: ””, ”output”: ”The scene of the crime is a dimly lit study inside a grand, Victorian-era mansion. There’s a large oak desk at the center of the room, and the deceased, Lord Barrington, lies slumped over it with a gunshot wound to the chest. A shattered windowpane hints at the possible point of entry for the assailant. The three main suspects are: 1) Lady Barrington, the victim’s wife, who had increasing disputes with her husband; 2) Randall, the family’s loyal butler, who recently discovered he was going to be replaced; and 3) Edgar, a distant cousin, who expected a large inheritance if Lord Barrington passed away. My plan to investigate the case includes: interviewing each suspect about their whereabouts during the crime; examining the broken window for any fingerprints or signs of forced entry; and analyzing the murder weapon, a pistol, to trace its ownership and look for any additional evidence like partial fingerprints or other traces left by the culprit.”}

GPT-Teacher-Instruct General 17K – – {”instruction”: ”Provide an overview of the process of human evolution.”, ”input”: ””, ”output”: ”Human evolution began millions of years ago with the emergence of common ancestors of humans, chimpanzees, and gorillas. The process involved a series of distinct species branching off and evolving separately, with some leading to modern humans, Homo sapiens. Key milestones in human evolution include the appearance of the first hominids (human-like primates), the development of bipedalism (walking upright), and the growth of larger brains. Over time, hominids developed more advanced tools, communication abilities, and social structures. The timeline of human evolution includes several genera such as Ardipithecus, Australopithecus, Paranthropus, and Homo. Noteworthy species include Ardipithecus ramidus, Australopithecus afarensis ’Lucy,’ Homo habilis, Homo erectus, Homo neanderthalensis (Neanderthals), and finally, Homo sapiens. The process of human evolution was influenced by factors such as migration, climate change, and genetic mutations, leading to the diverse populations and cultures that exist today.”}

WebGPT CoT 14K – – {”instruction”: ”What causes that horrible screeching feedback sound when a microphone gets too close to a speaker?\n\n”,”input”: ”[TITLE] Why does my microphone make a loud screeching sound? (www.scienceabc.com)\n[CONTEXT] A typical audio system (like a Public Address system or the audio system used by bands) consists of three essential components: a microphone, a speaker and an amplifier. Wherever all three of these components are present, there it the potential for feedback. The reason is simple; feedback occurs when the microphone picks up the output sound from the speakers and then sends it back to the speakers to re-amplify it, kick-starting an endless loop unless interfered with externally.\n[TITLE] Why does my microphone make a loud screeching sound? (www.scienceabc.com)\n[CONTEXT] However, if the sound discharged from the speakers somehow reaches back to the mic (which ideally shouldn’t happen), the process discussed above kick-starts again, i.e., the mic transmits sound to the amplifier, which then transmits to the speaker, and back to the mic. . . and then this goes on and on. The result is that you hear a high-pitched squeal, which gets louder and louder (due to the reinforced amplitude as a result of multiple rounds of amplification) until it is corrected.\n[TITLE] What Causes Microphone Feedback and How to Stop It Virtuoso Central (virtuosocentral.com)\n[CONTEXT] Audio feedback happens when a sound coming out of a loudspeaker is picked up by the microphone reentering the sound system again and again in an always increasing loop. The result is a loud rumble or a high pitch ringing noise.\n”,”output”: ”Audio feedback occurs when the microphone picks up the output sound from the speakers and then sends it back to the speakers to re-amplify it, kick-starting an endless loop unless interfered with externally. [1] If the sound discharged from the speakers somehow reaches back to the mic, the process discussed above kickstarts again, i.e., the mic transmits sound to the amplifier, which then transmits to the speaker, and back to the mic. . . and then this goes on and on. The result is that you hear a high-pitched squeal, which gets louder and louder (due to the reinforced amplitude as a result of multiple rounds of amplification) until it is corrected. [2]”}

TABLE 8 Statistics of Datasets (Part 6).

###### Dataset Task Train Val Test Example

Lima CoT 1K – – {”instruction”: ”Can brain cells move? By movement I mean long distance migration (preferably within the brain only).”,”input”: ””,”output”: ”The question is relatively broad and one should take into account that the brain not only consists of neurons, but also glial cells (supportive cells) and pre-mitotic neuronal stem cells. Furthermore, as critical fellow-scientists have indicated, developmental stage is very important, as the developing embryonic brain is very different from the adult brain.\nHowever, after sifting through various publications, the answer to the question is actually remarkably simple: Yes, brain cells migrate.\nIn the adult brain glial cells migrate in the brain (Kl\u00e4mbt, 2009). Glial cells are involved in a myriad of functions, but a notable example of migrating glial cells are the oligodendrocytes that migrate relative long distances to find their target axons onto which they wrap themselves to form the insulating myelin sheath (Tsai and Miller, 2002).\nNeuronal stem cells migrate over long distances in response to injury (Imitola et al., 2004) and they migrate from specific stem-cell locations (e.g., hippocampus and subventricular zone) to other regions (Clarke, 2003).\nPost-mitotic, but non-differentiated neurons have been shown to migrate in the adult brain in fish (Scott et al., 2012), and in mammals and non-human primates as well (Sawada et al., 2011).\nNot surprisingly, glial cells, stem cells and neurons also migrate during embryonic development. Most notably, post-mitotic neurons destined to fulfill peripheral functions have to migrate over relatively long distances from the neural crest to their target locations (Neuroscience, 2nd ed, Neuronal Migration).”}

Code-Alpaca Coding 20K – – {”instruction”: ”Create a Python program to accept a string from the user and print out the number of vowels in the string.”,”input”: ”Hello World”,”output”: ”def countVowels(str): \n\tvowel = 0\n\tfor i in str: \n\t\tif i.lower() in [’a’, ’e’, ’i’, ’o’, ’u’]: \n\t\t\tvowel += 1\n\treturn vowel \n\n# Driver code \nstr = \”Hello World\”\nprint(\”Number of vowels are:\”, countVowels(str))”}

Thought Source CoT 343K – – {”instruction”: ”Two friends plan to walk along a 43-km trail, starting at opposite ends of the trail at the same time. If Friend P’s rate is 15% faster than Friend Q’s, how many kilometers will Friend P have walked when they pass each other?”,”input”: ”choices: 21,21.5,22,22.5,23”,”output”: ”If Q complete x kilometers, then P completes 1.15x kilometers.\nx + 1.15x = 43\n2.15x=43\nx = 43/2.15 = 20\nThen P will have have walked 1.15*20=23 km.\nThe answer is E.the answer is 23”}

CAMEL-AI-Society Society 107K – – {”instruction”: ”Create a function that can read in a PDF file.”,”input”: ”File path to a PDF file.”,”output”: ”“‘\nimport PyPDF2\n\ndef read pdf(file path):\n with open(file path, ’rb’) as pdf file:\n pdf reader = PyPDF2.PdfFileReader(pdf file)\n text = ”\n for page in range(pdf reader.getNumPages()):\n text += pdf reader.getPage(page).extractText()\n return text\n“‘\n\nThis function takes in a file path to a PDF file and returns the text content of the PDF file. It uses the PyPDF2 library to read in the PDF file and extract the text from each page. The extracted text is then concatenated and returned as a single string.”}

CAMEL-Code Coding 367K 0 0 {”Instruction”: ”Implement a method to extract text from a PDF file using Apache PDFBox library.”,”Input”: ”Path to the PDF file.”,”output”:”Solution: Here’s a method that uses Apache PDFBox library to extract text from a PDF file:\n\n“‘\nimport java.io.IOException;\nimport org.apache.pdfbox.pdmodel.PDDocument;\nimport org.apache.pdfbox.text.PDFTextStripper;\n\npublic String extractTextFromPDF(String filePath) throws IOException {\n PDDocument document = null;\n try {\n document = PDDocument.load(new File(filePath));\n PDFTextStripper stripper = new PDFTextStripper();\n return stripper.getText(document);\n } finally {\n if (document != null) {\n document.close();\n }\n }\n}\n“‘\n\nThis method loads the PDF document from the specified file path, creates a PDFTextStripper object to extract the text, and returns the extracted text as a String. It also ensures that the document is properly closed after use. Next request.”}

CAMEL-Math Math 46K 0 0 {”Instruction”: ”Solve for x: 5x + 3 = 28”,”Input”: ””,”output”:”To solve for x, we need to isolate x on one side of the equation. Step 1: Subtract 3 from both sides of the equation. 5x + 3 - 3 = 28 - 3 5x = 25 Step 2: Divide both sides of the equation by 5. 5x / 5 = 25 / 5 x = 5 So, x = 5.”}

TABLE 8 Statistics of Datasets (Part 7).

###### Dataset Task Train Val Test Example

CAMEL-Physics Physics 20K 0 0 {”Instruction”: ”What is the energy difference between the n=4 and n=2 energy levels of a hydrogen atom, and what is the frequency of the photon emitted when an electron transitions from the n=4 to n=2 energy level?”,”Input”: ””,”output”:”To find the energy difference between the n=4 and n=2 energy levels of a hydrogen atom, we can use the formula for the energy of an electron in a hydrogen atom: E n = -13.6 eV / nˆ2 where E n is the energy of the electron in the nth energy level and n is the principal quantum number. First, let’s find the energy of the electron in the n=4 energy level: E 4 = -13.6 eV / 4ˆ2 = -13.6 eV / 16 = -0.85 eV Now, let’s find the energy of the electron in the n=2 energy level: E 2 = -13.6 eV / 2ˆ2 = -13.6 eV / 4 = -3.4 eV The energy difference between the n=4 and n=2 energy levels is: ∆E = E 4 - E 2 = -0.85 eV - (-3.4 eV) = 2.55 eV Now, we need to find the frequency of the photon emitted when an electron transitions from the n=4 to n=2 energy level. First, we need to convert the energy difference from electron volts (eV) to joules (J): 1 eV = 1.60218 × 10ˆ(-19) J ∆E = 2.55 eV × 1.60218 × 10ˆ(-19) J/eV = 4.08556 × 10ˆ(-19) J Next, we can use the Planck’s equation to find the frequency of the photon: E = h × f where E is the energy of the photon, h is the Planck’s constant (6.626 × 10ˆ(-34) Js), and f is the frequency of the photon. Rearranging the equation to solve for the frequency: f = E / h = (4.08556 × 10ˆ(-19) J) / (6.626 × 10ˆ(-34) Js) = 6.165 × 10ˆ14 Hz So, the frequency of the photon emitted when an electron transitions from the n=4 to n=2 energy level is approximately 6.165 × 10ˆ14 Hz.”}

CAMEL-Chemistry Chemistry 20K 0 0 {”Instruction”: ”What is the IUPAC name for the organic compound with the molecular formula C6H12O2?”,”Input”: ””,”output”:”There can be several isomers with the molecular formula C6H12O2, so it is not possible to provide a specific IUPAC name without more information about the structure of the compound. If you can provide the structure or any additional details, I would be happy to help you determine the IUPAC name.”}

CAMEL-Biology Biology 20K 0 0 {”Instruction”: ”What are the common techniques used in identifying a new species, and how can scientists accurately categorize it within the existing taxonomy system?”,”Input”: ””,”output”:”Identifying a new species and categorizing it within the existing taxonomy system involves several common techniques. These techniques can be broadly classified into morphological, molecular, and ecological approaches. Here’s an overview of these techniques: 1. Morphological analysis: This is the traditional method of identifying a new species based on its physical characteristics. Scientists examine the external and internal features of an organism, such as its size, shape, color, and structure. They also study the anatomy and physiology of the organism, including its skeletal, muscular, and reproductive systems. By comparing these features with those of known species, scientists can determine if the organism represents a new species. 2. Molecular analysis: With advances in molecular biology, scientists can now use genetic information to identify and classify new species. This involves analyzing the DNA or RNA sequences of the organism and comparing them with the sequences of known species. Techniques such as DNA barcoding, which compares a short, standardized region of the genome, can help determine if the organism is a new species. Additionally, phylogenetic analysis can be used to understand the evolutionary relationships between the new species and other known species, helping to place it within the existing taxonomy system. 3. Ecological analysis: Studying the ecological niche of an organism can also provide valuable information for species identification and classification. This involves examining the organism’s habitat, its interactions with other species, and its role in the ecosystem. By understanding the ecological context of the new species, scientists can better determine its taxonomic position and how it relates to other known species. 4. Behavioral analysis: Observing the behavior of an organism can also help in identifying a new species. This includes studying its feeding habits, mating rituals, and social interactions. Behavioral differences between closely related species can sometimes be more apparent than morphological or molecular differences. 5. Biogeography: The geographical distribution of an organism can provide clues about its evolutionary history and help in its classification. Studying the biogeography of a new species can reveal patterns of speciation and help determine its taxonomic position. Once a new species has been identified using these techniques, scientists can accurately categorize it within the existing taxonomy system by following the principles of biological classification. This involves placing the new species within a hierarchical structure of taxonomic ranks, such as kingdom, phylum, class, order, family, genus, and species. The new species is then given a scientific name, usually consisting of a genus name and a species epithet, following the rules of binomial nomenclature. In conclusion, identifying a new species and categorizing it within the existing taxonomy system involves a combination of morphological, molecular, ecological, behavioral, and biogeographical analyses. By using these techniques, scientists can accurately classify the new species and better understand its evolutionary relationships with other organisms.”}

TABLE 9 Evaluation results of all LLaMA2 models in the LoRA bank.

Model ARC-c ARC-e GSM8K MBPP BoolQ PiQA Average Base 33.90% 49.91% 16.68% 18.20% 47.86% 53.05% 29.67% ARC-c 44.07% 59.44% 15.54% 23.80% 43.33% 56.42% 35.71% ARC-e 47.12% 62.61% 16.98% 24.60% 48.13% 58.11% 37.83% WinoGrande 26.61% 53.26% 16.98% 23.60% 62.14% 56.09% 30.11% GSM8K 20.81% 14.58% 22.88% 21.20% 18.53% 52.61% 19.87% PiQA 37.97% 53.26% 17.51% 22.40% 42.51% 56.58% 32.79% CommonSenseQA 37.29% 49.38% 16.38% 23.20% 46.18% 51.80% 31.56% RACE 49.49% 69.31% 17.29% 21.40% 65.81% 55.55% 39.37% MBPP 31.19% 46.56% 16.22% 22.40% 43.18% 52.77% 29.09% MathQA 34.92% 49.03% 15.09% 21.80% 58.07% 53.65% 30.21% Esnli 20.34% 27.34% 13.57% 21.40% 36.88% 44.45% 20.66% ECQA 13.22% 13.58% 15.16% 21.80% 0.64% 35.15% 15.94% CREAK 30.17% 34.22% 15.54% 21.20% 22.35% 56.53% 25.28% GPT4Tools 25.76% 37.04% 17.29% 20.80% 31.87% 46.84% 25.22% AQuA 39.32% 53.79% 15.31% 21.20% 64.77% 56.91% 32.41% QASC 35.25% 42.33% 15.47% 21.60% 27.28% 55.66% 28.66% QED 10.51% 14.99% 14.56% 21.60% 5.20% 35.91% 15.42% StrategyQA 35.25% 45.50% 15.24% 20.40% 43.70% 56.37% 29.10% SensemakingQA 19.66% 20.46% 15.16% 20.00% 34.95% 48.31% 18.82% Toolformer 32.54% 36.86% 18.35% 19.00% 35.66% 51.80% 26.69% HellaSwag 35.59% 50.79% 18.50% 23.20% 59.76% 45.27% 32.02% SiQA 36.95% 49.74% 13.72% 18.20% 59.42% 56.53% 29.65% BoolQ 36.61% 54.32% 9.48% 24.20% 63.39% 45.43% 31.15% Dolly 31.19% 40.21% 17.21% 22.40% 15.84% 53.97% 27.75% WizardLM 38.98% 49.38% 16.38% 16.00% 21.77% 50.98% 30.19% ShareGPT 41.36% 55.38% 15.62% 18.20% 34.59% 53.97% 32.64% GPT-Teacher-Code 34.58% 46.56% 18.50% 21.40% 46.88% 54.73% 30.26% GPT-Teacher-Roleplay 35.25% 46.91% 17.51% 23.80% 40.64% 53.32% 30.87% GPT-Teacher-Instruct 35.43% 48.93% 17.01% 21.20% 42.02% 55.28% 30.64% WebGPT 32.88% 44.09% 14.40% 19.80% 30.89% 49.46% 27.79% Lima 30.52% 39.51% 16.00% 21.20% 15.78% 50.65% 26.81% Code-Alpaca 33.56% 44.80% 16.07% 21.80% 33.03% 51.69% 29.06% ThoughtSource 35.59% 51.32% 20.85% 21.20% 54.98% 61.53% 32.24% CAMEL-AI-Society 31.86% 41.09% 16.53% 19.40% 25.99% 43.04% 27.22% CAMEL-Code 26.10% 23.99% 1.67% 1.00% 29.42% 40.86% 13.19% CAMEL-Math 33.90% 48.32% 17.44% 22.80% 45.29% 50.00% 30.62% CAMEL-Physics 26.78% 39.33% 15.24% 24.40% 32.08% 49.73% 26.44% CAMEL-Chemistry 34.58% 40.74% 17.66% 24.20% 26.73% 48.37% 29.30% CAMEL-Biology 31.53% 41.62% 17.59% 22.60% 31.93% 50.44% 28.34%

TABLE 10 Evaluation results of all Mistral models in the LoRA bank.

Model ARC-c ARC-e GSM8K MBPP BoolQ PiQA Average Base 60.68% 73.54% 45.56% 36.00% 55.96% 57.67% 54.90% ARC-c 77.97% 88.36% 47.23% 40.20% 67.55% 64.85% 64.36% ARC-e 80.00% 90.30% 51.55% 40.60% 59.39% 60.34% 63.70% WinoGrande 61.69% 78.84% 47.54% 39.80% 69.30% 54.79% 58.66% GSM8K 49.83% 63.84% 49.73% 36.80% 11.19% 58.27% 44.94% PiQA 76.27% 87.83% 47.99% 40.60% 78.78% 68.12% 66.60% CommonSenseQA 75.25% 81.31% 49.20% 40.40% 61.68% 65.02% 62.14% RACE 74.92% 86.42% 48.67% 39.00% 74.53% 71.76% 65.88% MBPP 65.08% 60.34% 44.88% 37.60% 48.56% 57.02% 52.25% MathQA 73.90% 83.42% 43.52% 35.40% 76.79% 63.60% 62.77% Esnli 43.05% 55.37% 31.92% 34.60% 0.49% 49.56% 35.83% ECQA 60.00% 65.26% 36.92% 36.00% 5.78% 58.71% 43.78% CREAK 68.81% 71.43% 42.99% 38.20% 6.06% 66.97% 49.08% GPT4Tools 69.15% 75.13% 46.70% 38.00% 60.49% 58.49% 57.99% AQuA 68.74% 77.78% 41.32% 38.00% 45.93% 64.15% 55.99% QASC 72.20% 78.66% 39.65% 35.80% 16.33% 68.66% 51.88% QED 71.86% 66.78% 39.27% 37.20% 50.55% 66.76% 55.40% StrategyQA 72.88% 80.78% 43.06% 38.40% 64.56% 65.56% 60.87% SensemakingQA 38.31% 46.91% 38.82% 35.20% 1.83% 56.42% 36.25% Toolformer 69.49% 76.01% 45.64% 38.00% 54.07% 59.58% 57.13% HellaSwag 77.63% 88.01% 47.23% 40.20% 67.37% 67.79% 64.71% SiQA 74.24% 81.83% 47.99% 38.20% 54.01% 65.56% 60.31% BoolQ 73.22% 87.13% 46.78% 39.20% 80.12% 61.04% 64.58% Dolly 73.56% 83.07% 46.47% 38.80% 48.99% 66.38% 59.55% WizardLM 73.90% 79.37% 46.63% 36.60% 70.67% 63.60% 61.80% ShareGPT 69.83% 78.66% 45.64% 37.00% 68.50% 68.61% 61.37% GPT-Teacher-Code 66.78% 76.37% 47.01% 40.20% 57.22% 59.25% 57.81% GPT-Teacher-Roleplay 66.78% 74.07% 49.43% 37.60% 67.06% 63.22% 59.69% GPT-Teacher-Instruct 72.54% 79.72% 47.16% 37.60% 78.20% 64.74% 63.33% WebGPT 63.39% 76.37% 43.21% 36.40% 52.91% 59.09% 55.23% Lima 69.83% 77.43% 45.34% 37.80% 30.52% 56.64% 52.93% Code-Alpaca 71.53% 79.89% 45.72% 40.00% 44.56% 62.40% 57.35% ThoughtSource 77.97% 83.77% 45.49% 37.00% 78.38% 70.95% 65.59% CAMEL-AI-Society 68.81% 77.07% 44.66% 38.00% 57.58% 63.49% 58.27% CAMEL-Code 75.59% 79.54% 45.94% 37.80% 46.67% 65.23% 58.46% CAMEL-Math 77.97% 84.30% 46.02% 38.80% 58.65% 60.01% 60.96% CAMEL-Physics 71.86% 79.01% 46.78% 38.20% 51.01% 58.60% 57.58% CAMEL-Chemistry 70.51% 80.42% 46.70% 38.40% 54.89% 60.72% 58.61% CAMEL-Biology 63.73% 73.90% 47.16% 37.00% 52.02% 56.86% 55.11%

TABLE 11 List of symbols.

Symbol Description

- x an input sequence (i.e., instruction) consisting of multiple tokens x(i), i = 1, 2, ..., |x|
- y an output sequence (i.e., response) consisting of multiple tokens y(i), i = 1, 2, ..., |y|

- x(i) the i-th token of x
- y(i) the i-th token of y

y(<i) the sequence of y(j), j < i, i.e., [y(1), y(2), ..., y(i−1)] m a LLM model from the LoRA bank θm the parameters of m P(y(i)|x, y(<i); θm) the model m’s probability of the output token y(i) given the input x and the preceding output tokens y(<i) yˆ the expanded output of y with CoT rationales L(x, y, θMoE) the cross-entropy language modeling loss of the MoE system on y given x PPL(x, y, θm) the perplexity of the model m parameterized by θm on the sequences y given x Φ(·, ·) the CoT expansion process implemented with prompts K the number of human-annotated instructions from tasks of interest DK K-shot data from the task of interest: {(x1, y1), (x2, y2), . . . , (xK, yK)} PPL(x, y, θˆ m) the reasoning perplexity of the model m parameterized by θm on the CoT-formatted yˆ given x PPLR the total reasoning perplexity of the model m on DK f(·) the post-processing function for standardization of model responses Acc(m, DK, f) the performance of the model m on DK in exact match accuracy with a post-processing function f(·) y˜ the auto-regressively generated sequence given x y˜′ the post-processed response of y˜

(·) the indicator function

- B the model bank consists of multiple models {m1, m2, ..., m|B|}

- M the number of candidate models

- BM a set of candidate models BM ⊂ B E(mi) the flattened matrices of all layers of mi

N the number of experts

- BN a set of chosen experts BN ⊂ BM

ΩBN the intra-group diversity of BN rank(·) the ranking of models by certain indices

RL(m) the rank of the model m by L(m, DK) from smallest to largest RP (m) the rank of the model m by Acc(m, DK, f) from largest to smallest RD(BN) the rank of the N-tuple BN by intra-group diversity ΩBN from the largest to the smallest BE the chosen experts for initialization of a MoE system wi the weight of the i-th LoRA module for linear composition of model merging W the original parameter of a pre-trained model ∆Wi the i-th LoRA variant with Ni=1 wi = 1 gl(x) the output of a N-expert MoE system at the l-th layer given input x Gl(x) the gating vector at the l-th layer with Gl(x) ∈ RN given x Gli(x) the i-th element of the gating vector Gl(x) gil(x) the output from the i-th expert at the l-th layer Wgl the matrix of a single fully-connected layer for the gating network k the number of the selected experts in a MoE system top-k(·) the operation that returns the largest k elements unchanged and the other elements as −∞

DS a set of S open-source samples: {(x1, y1), (x2, y2), . . . , (xS, yS)} h a pre-trained encoding model

θh the parameters of the pre-trained model h ui the embedding of both input and output [xi, yi] UK the embeddings of samples in DK US the embeddings of samples in DS DC the candidate open-source instruction set DC ⊂ DS

- C the budget for the candidate dataset DC d(·, ·) the distance function sim(·, ·) the similarity metric

Aij the entry of the cross-dataset similarity matrix between each pair of ui ∈ UK and uj ∈ US, A ∈ RK×S Iij the entry of the intra-dataset similarity matrix between ui ∈ UC and uj ∈ UC, UC = {ui|(xi, yi) ∈ DC} τ the threshold for semantic deduplication DA the chosen open-source dataset for data augmentation DT the official training set from the downstream task of interest (only involved in ablation studies) UKC the convex hull of the embeddings UK from the K-shot data DK λi the coefficients that determine the convex combination of samples in UK Ker(·) the kernel function for distance measurement γ the bandwidth controlling the smoothness of the density function p(u) the estimated density probability of a sample u in the embedding space

TABLE 12 List of hyper-parameters for training settings.

Settings LoRA Models (Single) MoE System Optimizer AdamW Optimizer momentum 0.9 Optimizer weight decay 1 × 10−4 Batch size 2 Gradient accumulation steps 16 Training scheduler Cosine decay with linear warm-up Learning rate 5e-5 Transformers Version 4.36.1 Location of PEFT q proj, v proj Cutoff length 1024 Deepspeed Stage 3 Training epochs in total 3 5

