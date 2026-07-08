## O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning

Haotian Luo1 Li Shen1 Haiying He2 Yibo Wang3 Shiwei Liu4 Wei Li5 Naiqiang Tan5 Xiaochun Cao1 Dacheng Tao6

# arXiv:2501.12570v2[cs.CL]29Jan2025

### Abstract

Recently, long-thought reasoning LLMs, such as OpenAI’s O1, adopt extended reasoning processes similar to how humans ponder over complex problems. This reasoning paradigm significantly enhances the model’s problem-solving abilities and achieves promising results. However, long-thought reasoning process leads to a substantial increase in inference time. A pressing challenge is reducing the inference overhead of long-thought LLMs while ensuring accuracy. In this paper, we identify that long-thought reasoning models struggle to effectively allocate token budgets based on problem difficulty and reasoning redundancies. To address this, we propose Length-Harmonizing Fine-Tuning (O1-Pruner), aiming at minimizing reasoning overhead while maintaining accuracy. This effective fine-tuning method first estimates the LLM’s baseline performance through pre-sampling and then uses RL-style fine-tuning to encourage the model to generate shorter reasoning processes under accuracy constraints. This allows the model to achieve efficient reasoning with lower redundancy while maintaining accuracy. Experiments on various mathematical reasoning benchmarks show that O1-Pruner not only significantly reduces inference overhead but also achieves higher accuracy, providing a novel and promising solution to this challenge. Our code is coming soon at https://github.com/StarDewXXX/O1-Pruner

### 1. Introduction

Reasoning represents a fundamental capability of large language models (LLMs), serving as a cornerstone in the advancement of artificial intelligence research (Huang & Chang, 2023). Recently OpenAI’s O1(OpenAI, 2024)

1Shenzhen Campus of Sun Yat-sen University 2China Agriculture University 3Tsinghua University 4University of Oxford 5Didichuxing Co. Ltd 6Nanyang Technological University. Correspondence to: Li Shen <mathshenli@gmail.com>.

have introduced long-thought reasoning models that mimic human-like problem-solving processes. In addition to O1, researchers have also developed models that inference with a similar long-thought reasoning pattern, such as DeepseekR1 (DeepSeek, 2024), QwQ (Qwen, 2024) and Marco-

- o1(Zhao et al., 2024). These models leverage a long chain-
- of-thought framework, enabling them to tackle complex problems by iteratively identifying and correcting errors, simplifying intricate steps, and exploring alternative strategies when initial approaches prove inadequate. Furthermore, Mulberry (Yao et al., 2024) has demonstrated that O1-Like reasoning can also play a significant role in multimodal reasoning. This reasoning paradigm significantly enhances the problem-solving capabilities of large language models (LLMs) by allowing them to approach complex tasks in a more systematic and human-like manner, demonstrating an ability to handle problems that would otherwise be challenging or intractable for conventional LLMs.

While long-thought reasoning enhances reasoning capabilities and improves accuracy, it is accompanied by longer output sequences, which result in increased computational overhead. A critical challenge lies in developing mechanisms that enable LLMs to dynamically adjust the length and complexity of their reasoning processes in accordance with the difficulty of the problems they encounter.

In this paper, we first revisit the long-thought reasoning processes. we observe that the reasoning processes in longthought reasoning LLMs often exhibit significant redundancies, which leads to inefficient use of computational resources. This inefficiency not only increases inference costs but also highlights a fundamental limitation in the models’ ability to adapt their reasoning depth to suit the demands of diverse tasks. Building on this analysis, we formulate an optimization objective aimed at minimizing reasoning overhead while maintaining accuracy as a constraint. Our approach introduces a Length-Harmonizing Reward, which explicitly rewards shorter solutions while penalizing accuracy degradation. By embedding this reward into a RL-based framework, we enable the model to optimize for efficiency without compromising performance. Moreover, our method incorporates an off-policy training strategy inspired by Proximal Policy Optimization (PPO), which aimed at reducing training complexity while main-

taining robustness.

Our experiments are conducted using open-source longthought reasoning LLMs, and we compare our approach against several competing methods like SFT and DPO (Rafailov et al., 2024). Through extensive experiments, we demonstrate the efficiency of our proposed methods. Additionally, we perform further studies on the influence of hyperparameters and dataset difficulty on our approach, in order to gain deeper insights into the characteristics and behavior of this novel framework.

In conclusion, our contributions can be outlined as follows:

- • We design a simple experiment and identify a critical issue in the reasoning process of long-thought models, referred to as length disharmony, which leads to redundant inference overhead.
- • We formulate an optimization problem aimed at improving model inference efficiency while maintaining accuracy, and based on this, we propose LengthHarmonizing Fine-Tuning (O1-Pruner) approach.
- • Through extensive experiments, we demonstrate the effectiveness of O1-Pruner and conduct in-depth analyses, to provide insights and inspiration for future research in this area.

### 2. Related Work

Inference-time Scaling. Inference-time scaling refers to the ability of large language models (LLMs) to improve their outputs by utilizing additional computation during inference time. Recent studies (Snell et al., 2024) have explored how scaling inference-time computation can enhance the performance of LLMs on challenging prompts. This approach draws parallels to human reasoning, where additional cognitive effort is often allocated to complex problems. In addition to increasing the number of candidate solutions or searching different steps, OpenAI’s O1 inference (OpenAI, 2024) demonstrates that extending the length of the solution generated during reasoning can also significantly enhance the model’s performance.

LLM Alignment. LLM alignment (Shen et al., 2023) constitutes a technical process aimed at guaranteeing that the responses generated by large language models are not only precise and logically consistent but also secure, morally sound, and aligned with the expectations of both developers and users. Ensuring that these expansive language models are in harmony with human preference is crucial for leveraging their immense capabilities in a manner that is both reliable and conscientious. Common methodologies employed in LLM alignment include Supervised Fine-Tuning (Zhou et al., 2023), Reinforcement Learning from Human

Feedback (RLHF) (Ouyang et al., 2022), and Direct Preference Optimization (DPO), among others. The discourse on long thought reasoning optimization presented in this paper can be regarded as an extended setting of LLM alignment, where human preferences are inclined towards shorter outputs (faster inference) and enhanced reasoning accuracy.

CoT Compression. Chain-of-Thought (CoT) (Wei et al.,

- 2023) and its variations (ToT, (Yao et al., 2023), GoT (Besta et al., 2024)) are powerful techniques for improving the reasoning capabilities of LLMs. Although CoT is highly effective, it introduces additional computational overhead. Consequently, several studies have attempted to address this issue. For example, (Han et al., 2024a) introduced a token-budget-aware reasoning framework for large language models (LLMs), which dynamically allocates token budgets according to the complexity of different problems and leverages these budgets to guide the reasoning process. C3oT (Kang et al., 2024) employs GPT-4 as a compressor to retain critical information during the reasoning process, thereby reducing reasoning redundancy. Furthermore, several approaches try to utilize continuous representations to mitigate the computational overhead associated with Chainof-Thought (CoT). For example, CCoT (Cheng & Durme,
- 2024) reduces reasoning overhead by generating contentful and continuous contemplation tokens of variable sequence lengths. COCONUT (Hao et al., 2024) train LLMs to reason with fewer thinking tokens during inference in a continuous latent space. However, unlike traditional approaches that focus on compressing normal Chain-of-Thought (CoT), our method centers on long thought reasoning and reduces redundancy in such reasoning by optimizing the reasoning paths instead of compressing each reasoning step.

Some concurrent works, such as (Chen et al., 2024), have identified the issue of overthinking in O1 reasoning and employs SimPO(Meng et al., 2024) for optimization, which is based on the view of preference learning. And (Team et al.,

- 2025) propose long2short RL, using long-CoT techniques to improve short-CoT models. However, in this paper we analyze the long-thought model from a different perspective of length distribution. Moreover, we establish an optimization problem and propose a RL-based method to optimize the model, which provides a different and novel perspective for subsequent research.

### 3. Revisiting the “Length Disharmony” in Long Thought Reasoning

We employ the term “Length Disharmony” to characterize the phenomenon of inefficiency in the reasoning process of long-thought reasoning, when the model generates responses of varying lengths, among which the shorter responses possess sufficiently high accuracy, thereby rendering the longer responses a superfluous expenditure of com-

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

- Figure 1. Accuracy-Length Relationship at Instance level. The relationship between length and accuracy varies significantly across problems, with peak accuracy occurring at short, medium, or long intervals. Notably, high accuracy often persists in shorter intervals.

putational resources. Besides, due to the quadratic complexity of the Transformer architecture, this will significantly leads to an increase in inference time.

In this section, we have devised a simple experiment to substantiate the disharmony inherent in long thought reasoning. We randomly selected 64 problems from the MATH (Hendrycks et al., 2021) test set (For QwQ-32B, we filtered out hard samples first). For each problem, we generated 512 solutions using both the Marco-o1 and the QwQ-32B models through Top-P sampling (Holtzman et al., 2020). For each problem, we categorize all candidate solutions into 4 intervals based on their lengths and subsequently compute the accuracy rate for each interval.

Accuracy-Length Relationship at Instance Level. From the data we collected, we can ascertain the relationship between accuracy and length at the instance level, which is shown in Figure 1. It is evident that there exists a markedly inconsistent relationship between length and accuracy across different problems. The highest accuracy may manifest within the shortest, intermediate, or longest length intervals. Specifically, we observe that relatively high accuracy is preserved even within shorter-length intervals.

Accuracy-Length Relationship at Distribution Level. Furthermore, by calculating the average accuracy across all problems within different intervals, we have derived the relationship between accuracy and length at the distribution level, which is shown in Table 1. At the distribution level, our analysis reveals a consistent trend where shorter response lengths are associated with higher average accuracy rates. This observation can be explained by the premise that a shorter response length typically signifies the model’s

ability to identify the optimal solution path more efficiently, consequently requiring fewer iterative processes of reflection and backtracking.

Therefore, we can conclude that long-thought models exhibit a phenomenon of length disharmony during reasoning, which leads to redundant computational overhead in the inference phase. This reasoning redundancy can be mitigated, as high accuracy is still maintained even at shorter lengths. From this perspective, we propose Length-Harmonizing Fine-Tuning (O1-Pruner) to optimize long-thought reasoning, enabling it to maintain high accuracy while reducing inference redundancy.

Table 1. Accuracy-Length Relationship at Distribution Level. A larger interval number indicates a longer solution length. The average accuracy is higher when the solution length is short.

Model Interval 1 Interval 2 Interval 3 Interval 4

Marco 81.1 80.2 78.8 75.3 QwQ 44.9 49.9 45.9 45.3

### 4. Methodology

In this section, we elaborate on our proposed LengthHarmonizing Fine-Tuning (O1-Pruner) in detail and provide a simple and intuitive mathematical analysis elucidating how our method works for optimize long thought of reasoning.

#### 4.1. Problem Setup

We consider a LLM parameterized by θ and denoted as πθ. In the context of math problem solving, the LLM accepts a sequence x = [x1,...,xn], commonly termed as

Q:What is the least…

Q:What is the least…

A:Let M be the least po sitive multiple of 30 tha t can be written with on ly the digits 0 and 2…

| | | |
|---|---|---|
| |𝜋𝜃| |

| | | |
|---|---|---|
| |𝜋𝜃| |

A:Let M be the least posi tive multiple of 30 that c an be written with only the digits 0 and 2…

update

- Time:1 min Accuracy:77.5%

[Figure 7]

- Time:2 min Accuracy:73.8%

O1-Pruner

[Figure 8]

calculate reward

Q:What is the least…

Q:What is the least…

𝜋𝑟𝑒𝑓

𝜋𝑟𝑒𝑓

A:In order to find the smallest positive integer multiple of 30 that can be written using only the digits 0 and 2, we first…

A:In order to find the smallest positive integer multiple of 30 that can be written using only the digits 0 and 2, we first…

|max 𝐸𝑥∼𝐷,𝑦∼πθ 𝑦 𝑥<br><br>𝐸𝑦′∼π<br><br>𝑟𝑒𝑓 𝑦′ 𝑥 𝐿 𝑦′ 𝐿 𝑦<br><br>− 1 + λ 𝐴 𝑥,𝑦 − 𝐸𝑦′∼π<br><br>𝑟𝑒𝑓 𝑦′ 𝑥 𝐴 𝑥,𝑦′<br><br>|
|---|

(a) Training (b) Inference

- Figure 2. Length-Harmonizing Fine-Tuning. During the training phase, for each problem, we sample multiple times from the reference model. Subsequently, we sample from the model to be optimized and compute the reward based on the reference samples, followed by a RL-style fine-tuning. During the inference phase, the model optimized through O1-Pruner demonstrates a significant improvement in inference speed, along with a noticeable enhancement in accuracy.

the problem, and then generate a corresponding solution y = [y1,...,ym]. Hence, the solution y is construed as a sample drawn from the conditional probability distribution πθ(·|x). The conditional probability distribution πθ(y|x) can be decomposed as follows:

πθ(y|x) =

m

πθ(yj|x,y<j). (1)

j=1

Firstly, we review the process of supervised fine-tuning (SFT). SFT is the primary method to adapt a pre-trained LLM for downstream tasks with a relatively smaller supervised dataset of labeled examples compared to the data of pre-training stage. In this paper, we focus on the task of mathematic problem solving where the problem-solution pairs denoted as (x,y), are drawn from a specified SFT dataset D. Thus the training objective of SFT under this setting can be formulated as:

E(x,y)∼D log πθ(y | x) . (2)

max

πθ

#### 4.2. Length-Harmonizing Fine-Tuning (O1-Pruner)

To start with, let’s assume that πθ is a LLM that can solve math problems with long thought with redundancy and disharmony. we hypothesize that the reasoning paths represented by output thought of language model πθ contain

redundancies and lack proper coordination. To address this, we propose an optimization objective that ensures no degradation in accuracy while tackling the issue from two perspectives. First, at the overall level, we aim to shorten the reasoning paths. Second, we encourage the model to output shorter answers for simpler problems, while for more complex problems, we guide the model to learn the correct reasoning paths, which, according to the inference scaling law, typically involve longer reasoning sequences. Given a problem x, we define L(y) as the length (counted by token) of the solution y. Considering a reference model πref, we reduce the solution length of the policy model relative to that of the reference model, which can be formulated as:

L(y′) L(y) − 1 . (3)

maxEx∼D Ey∼π

θ(y|x),y′∼πref(y|x)

We subtract a constant 1 from the optimization objective to ensure that the initial expected value of the optimization is zero. We then define an accuracy function A(x,y,answer), which takes the problem, solution, and the real answer as inputs, and returns 0 or 1 to indicate whether the solution is incorrect or correct. For the sake of simplicity in the notation, we omit the real answer, denoting the function as A(x,y). We aim to ensure that the model’s accuracy does not decrease, or even improves, during the process of optimizing for length. Thus, we derive the following

constraint condition:

θ(y|x)A(x,y) ≥ Ex∼D,y′∼πref(y′|x)A(x,y′).

Ex∼D,y∼π

(4) Therefore, we can establish our optimization objective as:

L(y′) L(y) − 1 (5)

maxEx∼D Ey∼π

θ(y|x),y′∼πref(y|x)

θ(y|x)A(x,y) ≥ Ex∼D,y′∼πref(y′|x)A(x,y′).

s.t. Ex∼D,y∼π

To solve this constrained optimization problem, we incorporate constraint into the objective function as a penalty term. Specifically, the constraint on accuracy is added to the objective with a penalty weight λ ≥ 0:

L(y′) L(y) − 1

maxEx∼D,y∼π

θ(y|x),y′∼πref(y|x)

+λ(A(x,y) − A(x,y′)). (6)

By reorganizing the terms related with reference model πref, we have:

Ey′∼πref(y′|x)L(y′) L(y) − 1+

maxEx∼D,y∼π

θ(y|x)

λ(A(x,y) − Ey′∼πref(y′|x)A(x,y′)). (7)

In practice, we approximate the expectation terms related with πref by sampling. For each x, we sample for K times from πref(·|x) and calculate the mean value:

K

1 K

L¯ref(x) =

L(yi′), yi′ ∼ πref(· | x); (8)

i=1

K

1 K

A¯ref(x) =

A(x,yi′), yi′ ∼ πref(· | x); (9)

i=1

This approach is widely employed in Policy Gradient with Baseline. Furthermore, a recently proposed method GRPO (Shao et al., 2024) adopts a similar technique to reduce training overhead. Based on this technique, our objective can be approximated as:

L¯ref(x) L(y) − 1

maxEx∼D,y∼π

θ(y|x)

+λ(A(x,y) − A¯ref(x)). (10)

Since both L(y) and A(x,y) are not differentiable, we solving this optimization with policy gradient approach, which is shown to have strong performance despite its simplicity. Furthermore, it is worth noting that during the optimization process, frequent sampling from the current distribution πθ is required during training, which significantly increases the complexity of the training procedure. Considering that off-policy training can bring remarkable effectiveness with

pre-collected data, we adopt an off-policy training approach by directly sampling from the πref instead of πθ. Besides, since our reward is derived by assessing the merit of a sample within the distribution relative to the expected outcome, our reward can be regarded as an approximate advantage function. Consequently, we employ a PPO-style loss (Schulman et al., 2017) to optimize the objective function, which helps for our off-policy training strategy. Defining the Length-Harmonizing Reward RLH(x,y) = L¯refL(y()x) −1 + λ(A(x,y) − A¯ref(x)), the loss function of off-policyversion Length-Harmonizing Fine-Tuning is:

LLH(θ;x,y) = −Ex∼D,y∼π

ref(y|x) min(r(θ)RLH(x,y),

clip(r(θ),1 − ϵ,1 + ϵ)RLH(x,y)) , (11)

θ(y|x)

where r(θ) = π

πref(y|x). clip() is the clipping function.

This allows us to prepare the required data at the beginning of training, thereby greatly simplifying the training workflow. Our experiments show that this off-policy approach still enables our method to achieve outstanding performance, significantly surpassing other baselines.

#### 4.3. Understanding the Loss Function

To intuitively understand how our loss function works, we begin by analyzing the RLH term. Evidently, RLH comprises two distinct components, namely the length reward term L¯(x,πL(yref) ) − 1 and the accuracy reward term λ(A(x,y)−A¯(x,πref)). Obviously, the length reward term will reward shorter outputs. When the sequence length are consistent with expected output length of reference model, the length reward is 0; however, when the output is longer, the length reward becomes negative. The accuracy reward term is essential for balancing length and accuracy. For a problem x with a relatively high accuracy expectation, solving it correctly does not yield a significant accuracy reward. As a result, the model tends to explore shorter solutions. For more challenging problems, solving them correctly yields a higher accuracy reward, indicating that we do not want the model to prioritize shortening the output. Instead, we aim for the model to focus on generating a correct solution. On this basis, if the correct solution is relatively short, the model will receive an additional length reward. To the end, we summarize the training procedure of our proposed O1-Pruner in Algorithm 1.

### 5. Experiments

In this section, we conduct extensive experiments to verify the efficacy of the proposed O1-Pruner.

Table 2. Main Experiment Results. We present the performance of two selected models optimized through different methods across three mathematical reasoning datasets. It can be observed that the models trained with O1-Pruner achieve the best trade off between accuracy and length in comparison with other approaches.

MATH GSM8K GaoKao AVERAGE Acc Length AES Acc Length AES Acc Length AES Acc Length AES

Model

Marco-o1-7B (full fine-tune) Baseline 73.8 1156 0 89.2 530 0 57.1 1112 0 73.4 932 0 Fast-solving Prompt 71.0 1113 0.15 81.7 447 0.41 57.1 1062 0.04 69.9 874 0.20 SFT 73.6 1076 0.08 89.9 497 0.09 56.3 1066 0.08 73.3 880 0.08 DPO 71.8 761 0.42 88.6 410 0.25 56.6 780 0.32 72.3 650 0.33 O1-Pruner 77.5 657 0.58 91.4 343 0.43 61.6 664 0.64 76.8 554 0.55

QwQ-32B-Preview (freeze fine-tune last 48 layers) Baseline 90.6 2191 0 95.1 777 0 79.0 2183 0 88.2 1717 0 Fast-solving Prompt 90.2 1763 0.21 95.8 561 0.30 78.4 1911 0.15 88.1 1411 0.22 SFT 90.4 2031 0.08 95.7 717 0.10 79.5 2112 0.05 88.5 1620 0.08 DPO 91.7 1999 0.12 95.3 704 0.10 79.7 2021 0.10 88.9 1575 0.11 O1-Pruner 91.0 1385 0.38 96.5 534 0.36 80.3 1446 0.39 89.3 1121 0.38

#### Algorithm 1 O1-Pruner

for our experiment, we only use the problem-answer pairs. For training, we selected 5,000 problems from the MATH Trainset. For Marco-o1-7B, we generated 16 solutions for each problem; for QwQ-32B-Preview, we generated 12 solutions for each problem. The dataset utilized for testing encompasses the test sets of MATH, GSM8k (Cobbe et al., 2021), and GaoKao (mathematical) (Zhang et al., 2024), comprising a diverse range of mathematical problems with varying levels of difficulty.

- 1: Input: LLM πθ, Dataset D = {(xi,ai)}i∈[N]
- 2: Initialize: πref = πθ
- 3: for i = 1 to N do
- 4: sampling K solutions y1i, ...,yKi from πref(·|xi)
- 5: calculating L¯ref(xi) = K1 Kk=1 L(yki )

- 6: calculating A¯ref(xi) = K1 Kk=1 A(xi,yki )

- 7: randomly select m (m ≤ K) solutions from y1i, ...,yKi
- 8: Update θ = arg min θ

m j=1 LLH(θ;xi,yji)

- 9: end for
- 10: Output: Updated LLM πθ

Baselines. To validate the superiority of our method for long thought reasoning optimization tasks, we have selected the three representatively comparative methods. (i) FastSolving Prompt: The Fast-Solving Prompt is a prompting technique wherein we instruct the model within the prompt to solve the given problem as swiftly as possible, aiming to achieve the desired reduction in reasoning length. (ii) SFT: For the SFT method, we curated the training dataset by selecting the two shortest correct solutions for each problem, ensuring that the model is exposed to examples that embody both accuracy and conciseness. These solutions were then used to train the model following the standard SFT pipeline. (iii) DPO: For the implementation of DPO, we meticulously selected two of the shortest correct solutions to serve as the chosen samples, which exemplify efficiency and precision in problem-solving. Conversely, to represent the reject sample, we opted for the longest solution available.

#### 5.1. Experiment Setup

Long-thought Models. The long thought models we chosen for our experiment are Marco-o1-7B and QwQ-32BPreview, which have demonstrated excellent performance on a wide range of math problem-solving tasks. For Marcoo1-7B, we utilize full-parameter fine-tuning; however, for the larger-scale QwQ-32B-Preview, our computational resources are not able to support full-parameter training. As a result, we adopt Parameter-Efficient Fine-Tuning (Han et al., 2024b). After evaluating both LoRA (Hu et al., 2021) and Freeze Fine-Tune, we observed that Freeze Fine-Tune yields much better performance. Therefore, we selected this fine-tuning approach for our experiments.

Evaluation Metric. We employ the following average accuracy, average length and Accuracy-Efficiency Score (AES) as key metrics to assess whether the model achieves a desirable balance between reasoning accuracy and length:

Dataset. The dataset used for training is MATH. It comprises approximately 10k math problem of high school level accompanied with both ground truth solution and ground truth answer. Since the ground truth solution is not need

• Accuracy Accuracy reflects whether the model cor-

rectly solves the problem. It is measured as the proportion of problems for which the model’s generated solution is correct. A higher accuracy indicates better problem-solving capability.

- • Length Length denotes the number of tokens in the generated solution. It serves as a proxy for the computational cost of generating solutions, where a shorter length implies greater efficiency.
- • AES We define a novel metric called AccuracyEfficiency Score (AES), to evaluate the trade off between improving accuracy and reducing computational cost. It is calculated by weighting and summing the model’s solution length and accuracy. Defin-

ing ∆Length = Length

baseline−Lengthmodel Lengthbaseline and ∆Acc =

Accmodel−Accbaseline

Accbaseline , the AES is calculated by:

α · ∆Length + β · |∆Acc|, if ∆Acc ≥ 0 α · ∆Length − γ · |∆Acc|, if ∆Acc < 0

AES =

where α > 0, β > 0, and γ > 0. AES evaluates the trade-off between improving accuracy and reducing computational cost. And we emphasize the penalization of accuracy degradation by setting γ > β. We set the default values as α = 1, β = 3, γ = 5.

#### 5.2. Experimental Results

Table 2 demonstrates the performance of various methods across different evaluation metrics. The proposed O1Pruner consistently achieves superior performance in balancing reasoning accuracy and efficiency compared to baseline and competing methods. Notably, it exhibits the best trade-off between accuracy and reasoning length across all datasets, as further supported by its significantly higher Accuracy-Efficiency Score (AES) values. Across both models, Marco-o1-7B and QwQ-32B-Preview, O1-Pruner outperforms other methods in average length of generated solutions, with a noticeable improvement on accuracy. For instance, in the Marco-o1-7B experiments, O1-Pruner achieves an average accuracy of 76.8%, accompanied by a 40.5% reduction in solution length compared to the baseline. Similarly, for QwQ-32B-Preview, O1-Pruner yields an average accuracy of 89.3%, with a 34.7% reduction in solution length. These improvements highlight the robustness of O1-Pruner in enhancing computational efficiency without sacrificing accuracy.

The Fast-Solving Prompt method, while achieving a moderate reduction in solution length, compromises accuracy in most cases. This trade-off is evident from its lower AES values compared to O1-Pruner, indicating that the reduction in reasoning length often comes at the cost of problem-solving

Table 3. Ablation experiments on λ. Overall, the model’s accuracy and solution length increase with the penalty coefficient λ. A larger λ implies that the model places greater emphasis on variations in accuracy, thereby partially weakening the optimization for sequence length. λ = 2 achieves an optimal balance between accuracy and efficiency.

λ Accavg Lengthavg AESavg Marco-o1-7B

- 0 74.8 527 0.49
- 1 76.0 532 0.54
- 2 76.8 554 0.55 5 76.3 656 0.45

performance. On the other hand, SFT provides a better balance than the Fast-Solving Prompt, but its improvements in reasoning length remain marginal, with limited gains in AES. The DPO method achieves a reasonable balance between accuracy and length, but it falls short of the performance achieved by O1-Pruner. Besides, the average accuracy decreases notably on Marco-o1-7B.

#### 5.3. Inference Time-Cost Analysis

In this subsection, we take the MATH test set as an example to explore the time overhead during the model inference phase. We utilize one A800 GPU and the VLLM (Kwon et al., 2023) library for inference, recording the average inference time. For the Marco-o1 model, we employ one A800 GPU, while for the QwQ-32B-Preview model, we use four A800 GPUs. As illustrated in Figure 3, the inference time results reveal notable differences across methods and models: For the Marco-o1-7B model, the baseline approach demonstrates an inference time of approximately 2 minutes, while the Fast-Solving Prompt and SFT methods achieve slightly shorter times. Both the DPO and O1-Pruner methods exhibit significantly reduced inference times, with O1Pruner achieving the shortest duration, slightly exceeding 1 minute. For the larger model QwQ-32B-Preview, the overall inference time is considerably higher. The Baseline approach records the longest inference time, approaching 6 minutes, while the DPO and SFT methods achieve slightly shorter durations. Notably, the Fast-Solving Prompt reduces the inference time to around 5 minutes, likely due to the strong instruction-following capabilities of large models. Once again, O1-Pruner demonstrates the shortest duration, achieving an inference time of approximately 4 minutes.

In summary, O1-Pruner represents a significant advancement in optimizing long-thought reasoning for math problem-solving tasks for both smaller and larger language models, achieving the best balance between accuracy and efficiency while minimizing computational overhead.

|0 0.5 1 1.5 2 2.5<br><br>InferenceTime(minute)<br><br>Marco-o1-7B<br><br>Baseline Fast-Solving Prompt SFT DPO O1-Pruner<br><br>|0 1 2 3 4 5 6 7<br><br>InferenceTime(minute)<br><br>QwQ-32B-Preview<br><br>Baseline Fast-Solving Prompt SFT DPO O1-Pruner<br><br>|
|---|---|

InferenceTime(minute)

InferenceTime(minute)

- Figure 3. Comparison of inference time-cost on MATH among different models and methods. O1-Pruner achieves the shortest inference times (slightly over 1 minute for Marco-o1-7B and 4 minutes for QwQ-32B-Preview), demonstrating its effectiveness in accelerating long-thought model inference for both small and large long thought models.

5.4. Ablation Study

Ablation on Hyper-parameter Sensitivity. In this part, we evaluate the hyperparameter sensitively of constraint coefficient λ. We select several different values of λ (λ = 0,1,2,5) and evaluate the model accordingly. For the sake of brevity, we only report the average metrics across three datasets. It can be observed that, overall, the model’s accuracy increases as the penalty coefficient lambda rises, while the required inference length also grows. In our experiments, for Marco-o1-7b, setting λ = 2 achieves a favorable trade-off between accuracy and efficiency.

[Figure 9]

- Figure 4. Performance on MATH Test-set When Trained on Problems of Different Difficulty Levels. Models trained on more challenging datasets tend to generate longer solutions, while learning to solve harder problems enhances model accuracy. In contrast, for less challenging datasets, shorter solutions are produced without a corresponding accuracy improvement.

mance and characteristics of O1-Pruner across datasets of varying difficulty levels. Due to limited computational resources, we exclusively selected Marco-o1 for experimentation. Utilizing the data constructed from the MATH dataset as mentioned in prior experiments (comprising 5k problems * 16 solutions), we partition the dataset into three subsets of differing difficulty based on the model’s average accuracy. In Figure 4, We observe that models trained on more challenging datasets tend to generate longer solutions, as these datasets typically contain problems requiring more complex solutions. At the same time, by learning the correct solutions of harder problems, the models improve their problem-solving capabilities and ultimately achieve higher accuracy. In contrast, for the least challenging datasets, although the generated solution lengths are reduced, there is no improvement in accuracy. These experimental results suggest that while our approach demonstrates significant effectiveness in optimizing long-thought reasoning, it remains highly influenced by the nature of the training data.

### 6. Conclusion

In this paper, we conducted simple experiments to validate the phenomenon of length disharmony in long-thought models during reasoning, which leads to redundant computational overhead in the inference phase. To address this issue, we formulated it as an optimization problem and proposed Length Harmonizing Fine-Tuning (O1-Pruner) as a solution to optimize the model. Extensive experiments demonstrate that O1-Pruner significantly reduces the length of the solutions generated by the model while achieves a modest improvement in accuracy, thereby substantially enabling more efficient reasoning. Additionally, we performed an in-depth analysis, including experiments on key hyperparameter and datasets of varying difficulty, to better understand the characteristics of O1-Pruner.

Ablation on Difficulty Levels. We investigate the perfor-

### References

Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Podstawski, M., Gianinazzi, L., Gajda, J., Lehmann, T., Niewiadomski, H., Nyczyk, P., and Hoefler, T. Graph of thoughts: Solving elaborate problems with large language models. Proceedings of the AAAI Conference on Artificial Intelligence, 38(16):17682–17690, March 2024. ISSN 2159-5399. doi: 10.1609/aaai.v38i16. 29720. URL http://dx.doi.org/10.1609/ aaai.v38i16.29720.

Chen, X., Xu, J., Liang, T., He, Z., Pang, J., Yu, D., Song, L., Liu, Q., Zhou, M., Zhang, Z., Wang, R., Tu, Z., Mi, H., and Yu, D. Do not think that much for 2+3=? on the overthinking of o1-like llms, 2024. URL https:

//arxiv.org/abs/2412.21187.

Cheng, J. and Durme, B. V. Compressed chain of thought: Efficient reasoning through dense representations, 2024. URL https://arxiv.org/abs/2412.13171.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems, 2021. URL https://arxiv.

org/abs/2110.14168.

DeepSeek. Deepseek-r1-lite-preview: Unleashing supercharged reasoning power. https://api-docs. deepseek.com/news/news1120, 2024. Accessed: 2024-12-29.

Han, T., Wang, Z., Fang, C., Zhao, S., Ma, S., and Chen, Z. Token-budget-aware llm reasoning, 2024a. URL https: //arxiv.org/abs/2412.18547.

- Han, Z., Gao, C., Liu, J., Zhang, J., and Zhang, S. Q. Parameter-efficient fine-tuning for large models: A comprehensive survey, 2024b. URL https://arxiv. org/abs/2403.14608.
- Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., and Tian, Y. Training large language models to reason in a continuous latent space, 2024. URL https:// arxiv.org/abs/2412.06769.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/2103.03874.

Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. The curious case of neural text degeneration, 2020. URL https://arxiv.org/abs/1904.09751.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Low-rank adaptation of

large language models, 2021. URL https://arxiv. org/abs/2106.09685.

Huang, J. and Chang, K. C.-C. Towards reasoning in large language models: A survey. In Rogers, A., Boyd-Graber, J., and Okazaki, N. (eds.), Findings of the Association for Computational Linguistics: ACL 2023, pp. 1049–1065, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl. 67. URL https://aclanthology.org/2023.

findings-acl.67/.

Kang, Y., Sun, X., Chen, L., and Zou, W. C3ot: Generating shorter chain-of-thought without compromising effectiveness, 2024. URL https://arxiv.org/abs/ 2412.11664.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention, 2023. URL https:// arxiv.org/abs/2309.06180.

Meng, Y., Xia, M., and Chen, D. Simpo: Simple preference optimization with a reference-free reward, 2024. URL https://arxiv.org/abs/2405.14734.

OpenAI. Learning to reason with llms. https://openai.com/index/ learning-to-reason-with-llms/, 2024. [Accessed 19-09-2024].

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., and Lowe, R. Training language models to follow instructions with human feedback, 2022. URL https:

//arxiv.org/abs/2203.02155.

Qwen. Qwq: Reflect deeply on the boundaries of the unknown, November 2024. URL https://qwenlm. github.io/blog/qwq-32b-preview/.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct preference optimization: Your language model is secretly a reward model, 2024. URL https://arxiv.org/abs/2305.18290.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/ 1707.06347.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Shen, T., Jin, R., Huang, Y., Liu, C., Dong, W., Guo, Z., Wu, X., Liu, Y., and Xiong, D. Large language model alignment: A survey, 2023. URL https://arxiv. org/abs/2309.15025.

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm testtime compute optimally can be more effective than scaling model parameters, 2024. URL https://arxiv. org/abs/2408.03314.

Team, K., Du, A., Gao, B., and et al. Kimi k1.5: Scaling reinforcement learning with llms, 2025. URL https: //arxiv.org/abs/2501.12599.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., and Zhou, D. Chain-ofthought prompting elicits reasoning in large language models, 2023. URL https://arxiv.org/abs/ 2201.11903.

Yao, H., Huang, J., Wu, W., Zhang, J., Wang, Y., Liu, S., Wang, Y., Song, Y., Feng, H., Shen, L., et al. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. arXiv preprint arXiv:2412.18319, 2024.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models, 2023. URL https://arxiv.org/abs/2305.10601.

Zhang, X., Li, C., Zong, Y., Ying, Z., He, L., and Qiu, X. Evaluating the performance of large language models on gaokao benchmark, 2024. URL https://arxiv. org/abs/2305.12474.

Zhao, Y., Yin, H., Zeng, B., Wang, H., Shi, T., Lyu, C., Wang, L., Luo, W., and Zhang, K. Marco-o1: Towards open reasoning models for open-ended solutions, 2024. URL https://arxiv.org/abs/2411.14405.

Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L., and Levy, O. Lima: Less is more for alignment, 2023. URL https://arxiv.org/abs/ 2305.11206.

