arXiv:2505.19815v1  [cs.CL]  26 May 2025
2025-6-12
Deciphering Trajectory-Aided LLM Reasoning:
An Optimization Perspective
Junnan Liuℵ, Hongwei Liuℵ, Linchen Xiaoℵ, Shudong Liuℵ, Taolin Zhangℵ, Zihan Maℵ,
Songyang Zhangℵ,† and Kai Chenℵ,†
ℵShanghai AI Laboratory
We propose a novel framework for comprehending the reasoning capabilities of large language models (LLMs)
through the perspective of meta-learning. By conceptualizing reasoning trajectories as pseudo-gradient descent
updates to the LLM’s parameters, we identify parallels between LLM reasoning and various meta-learning
paradigms. We formalize the training process for reasoning tasks as a meta-learning setup, with each question
treated as an individual task, and reasoning trajectories serving as the inner loop optimization for adapting
model parameters. Once trained on a diverse set of questions, the LLM develops fundamental reasoning
capabilities that can generalize to previously unseen questions. Extensive empirical evaluations substantiate the
strong connection between LLM reasoning and meta-learning, exploring several issues of significant interest
from a meta-learning standpoint. Our work not only enhances the understanding of LLM reasoning but also
provides practical insights for improving these models through established meta-learning techniques.
1. Introduction
Recent advancements in large language models (LLMs) [28, 32, 71, 117] have significantly improved their
capacity to perform complex reasoning tasks. Current LLMs often utilize chain-of-thought (CoT) reasoning,
i.e., intermediate reasoning trajectories [18, 110], to facilitate systematic problem-solving through coherent,
step-by-step logical deductions. Among them, state-of-the-art LLMs, such as OpenAI-o1 [72], DeepSeek-R1 [27],
Kimi-k1.5 [97], QwQ [101], and Gemini-2.5-Pro [26], exhibit exceptional proficiency in addressing intricate
mathematical and programming challenges. These models employ long reasoning trajectories, characterized
by an iterative and detailed process of exploration and reflection, to enhance test-time scaling capabilities
[56, 88, 102]. This iterative approach has driven significant progress in complex reasoning while motivating the
studies to illuminate the potential of supervised fine-tuning (SFT) and reinforcement learning (RL) methods to
refine the learning and application of extended reasoning processes [66, 77].
Despite significant advancements, comprehending and interpreting how LLMs achieve prominent rea-
soning capabilities through reasoning trajectories remains crucial for further enhancement and generaliza-
tion [33, 48]. The opaque nature of LLMs’ internal mechanisms hinders efforts to comprehend their operations
[91]. Recent studies [19, 37, 58, 62] have explored the representational power of reasoning trajectories,
showing that LLMs equipped with these trajectories can solve complex problems. Other research [36, 45]
demonstrates that reasoning trajectories can effectively describe complex learning algorithms. Nevertheless,
there is a notable gap in research exploring the fundamental role of reasoning trajectories in LLM reason-
ing and connecting diverse training approaches to enhance these capabilities. To address this, we propose
RaML (Reasoning as Meta-Learning), a methodology that analyzes LLM reasoning through a meta-learning
perspective [4, 34, 42, 82, 86]. We conceptualize reasoning trajectories as pseudo-gradient descent updates to
model parameters, leveraging established meta-learning methodologies, such as Model-Agnostic Meta-Learning
(MAML) [34] and Learn to Optimize (L2O) [4], to enhance both the understanding and optimization of LLM
reasoning.
To be more specific, RaML frames the training regimen for reasoning tasks as a meta-learning framework,
wherein each question constitutes a distinct task, reasoning trajectories serve as inner-loop optimization for
parameter adaptation, and answers act as the query set to optimize LLMs. In the context of RaML, the training
† Corresponding authors: Songyang Zhang (zhangsongyang@pjlab.org.cn), Kai Chen (chenkai@pjlab.org.cn)
* Code is at https://github.com/open-compass/RaML

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
LLM: 
Updated LLM: 
...
...
Figure 1: Illustration of the reasoning trajectory (𝑡) as the optimization of the LLM parameters 𝜃.
process optimizes the LLM to develop generalized reasoning abilities, identifying an effective meta-initialization
that enables efficient parameter adaptation through reasoning trajectories to produce accurate responses.
This approach provides a theoretical foundation for analyzing LLM reasoning capabilities and training, while
facilitating the application of meta-learning insights to advance LLM reasoning research.
RaML is complemented by comprehensive experiments and analysis involving both models trained from
scratch and publicly available models. Combining the studies in meta-learning [1, 23, 52, 59, 105, 106],
we conduct experiments to investigate key factors influencing LLM reasoning by instantiating them from a
meta-learning lens. Specifically, we explore the effectiveness of two typical LLM reasoning training techniques,
SFT and RL, from the perspective of more stable and effective inner loop optimization. Then, we reveal a
parallel between reasoning trajectory count and support set size. Furthermore, we examine the relationship
between inner loop optimization step and reasoning trajectory token, highlighting the varying contributions
and roles of tokens to the optimization. Lastly, we verify the similarities in task generalization between LLM
reasoning and meta-learning. Finally, based on our analyses, we suggest potential pathways for improving LLM
reasoning capabilities and conduct preliminary confirmative experiments. Our contributions are summarized
as follows:
• To elucidate the reasoning processes of LLMs, we introduce RaML, an interpretation methodology for LLM
reasoning from a meta-learning perspective, supported by a comprehensive theoretical analysis.
• We provide empirical evidence and detailed analysis, demonstrating a strong correspondence between LLM
reasoning and meta-learning principles.
• We contextualize recent advances in LLM reasoning within our framework, offering comprehensions into
their success.
• We present significant insights to enhance LLM reasoning, building on the existing meta-learning research
and analysis addressed in this paper.
2. RaML: Interpreting LLM Reasoning as Meta-Learning
In this section, we elucidate the interpretation methodology for the large language model (LLM) reasoning from
a meta-learning [4, 34, 42, 82, 86] perspective, i.e., RaML. First, we conceptualize the reasoning trajectories
as a pseudo gradient update to the parameters of the LLM (§ 2.2) and subsequently develop a meta-learning
framework to model the training process for the reasoning task (§ 2.3). Lastly, we establish connections between
various training techniques and our proposed definition (§ 2.4). The notations used in this section are listed in
Appendix A.
2.1. Setup
In this paper, we represent the large language model (LLM) as ℳ𝜃, where 𝜃signifies the parameters of the
LLM. We focus on a specific reasoning task that involves a set of questions denoted as 𝒬= {𝑞𝑖}𝑖∈[1,𝑛] and its
corresponding answers 𝒜= {𝑎𝑖}𝑖∈[1,𝑛]. Typically, the LLM is prompted to generate the answer 𝑎based on the
instruction 𝐼and the question 𝑞𝑖:
d (ℳ𝜃(𝐼; 𝑞𝑖)) →𝑎𝑖,
(1)
where d denotes the autoregressive decoding mechanism [79, 107], which is specifically defined as follows:
𝑝𝜃(𝑎𝑖) =
∏︁
0≤𝑗<|𝑎𝑖|
Softmax
(︁
𝐸𝑇
𝑂· ℳ𝜃
(︁
𝐼, 𝑞𝑖, 𝑡, 𝑎0
𝑖, ..., 𝑎𝑗−1
𝑖
)︁)︁[︁
𝑎𝑗
𝑖
]︁
.
(2)
2

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
Here, {𝑎0
𝑖, . . . , 𝑎|𝑎𝑖|
𝑖
} denotes the token set of the answer 𝑎, 𝑡represents the possible intermediate reasoning
trajectories, and 𝐸𝑂indicates the output embeddings of the entire token set (i.e., vocabulary). Intuitively,
ℳ𝜃
(︁
𝐼, 𝑞𝑖, 𝑡, 𝑎0
𝑖, . . . , 𝑎𝑗−1
𝑖
)︁
represents the activation determined by the parameters 𝜃and the inputs, while
the predicted probability is computed through the inner product between the output embeddings and the
activation. The activation, representing the output at the final position, is computed iteratively through the
self-attention and feed-forward layers of the LLM. During training, the LLM ℳis optimized to deliver accurate
answers to each question:
𝒪= max
∑︁
(𝑞𝑖,𝑎𝑖)∈𝒬×𝒜
𝑠(𝑎′
𝑖, 𝑎𝑖),
(3)
where 𝑎′
𝑖means the predicted answer for 𝑞𝑖, and 𝑠(𝑎′, 𝑎) indicates the plausibility of 𝑎′ w.r.t. 𝑎, which also
defines a loss landscape.
2.2. Reasoning Trajectories as Parameter Update
Recent works [13, 27, 72, 110] demonstrate that incorporating intermediate reasoning steps can significantly
enhance the capacity of large language models to execute complex reasoning tasks. Moreover, some studies [5,
25, 35–37, 45] theoretically demonstrate that models based on the transformer architecture can learn to
perform iterative algorithms like multi-step GD with the enhancement of CoT (which we called reasoning
trajectories in this paper). However, these studies primarily focus on explicit numerical optimization problems,
such as linear regression, and demonstrate that LLMs can learn optimization algorithms like multi-step GD in
the reasoning trajectories to solve the problem. In contrast, we conceptualize the reasoning trajectories of an
LLM ℳas a multi-step gradient descent process of the model’s parameters 𝜃, which could be formally represented
by:
𝜃′
𝑖←𝜃′
𝑖−1 + ∆ℳ𝜃′
𝑖−1(𝐼, 𝑞, 𝑡≤𝑖),
𝜃′
0 = 𝜃,
1 ≤𝑖≤|𝑡|,
(4)
where 𝑡denotes a reasoning trajectory, ∆ℳ𝜃′
𝑖−1(𝐼, 𝑞, 𝑡≤𝑖) = −𝜂∇𝜃′
𝑖−1ℒ𝑞(𝜃′
𝑖−1) represents the pseudo gradient
update associated with the reasoning trajectory 𝑡≤𝑖, and 𝜃′
|𝑡| signifies the updated parameters of the LLM in
response to the instruction 𝐼, the query 𝑞, and the reasoning trajectory 𝑡.
In summary, we conceptualize each question 𝑞𝑖as a sophisticated optimization task, with the LLM ℳbeing
optimized to produce the corresponding answer 𝑎𝑖. Prior to generating the final answer, the LLM is guided by
an intermediate reasoning trajectory, which serves as a parameter update mechanism. The overall process is
illustrated in Figure 1.
Pseudo Gradient Update. Without loss of generality, we consider a classic transformer model [107] comprising
a single self-attention layer and a two-layer feed-forward network while disregarding normalization layers and
other components. When using 𝑙= {𝐼, 𝑞} as input, its activation can be expressed as follows:
𝑊𝑇
2
(︀
𝜎
(︀
𝑊𝑇
1
(︀
Softmax
(︀
𝐸𝑙,−1𝑊𝑞𝑊𝑇
𝑘𝐸𝑇
𝑙,:
)︀
𝐸𝑙,:𝑊𝑣
)︀
+ 𝑏1
)︀)︀
+ 𝑏2,
(5)
where 𝐸𝑙,: indicates the input embeddings of the whole sequence 𝑙and 𝐸𝑙,−1 indicates the input embeddings
of the last position of 𝑙. In this context, the parameters 𝜃refers to {𝑊𝑞, 𝑊𝑘, 𝑊𝑣, 𝑊1, 𝑊2, 𝑏1, 𝑏2}. Then, given
a reasoning trajectory 𝑡, when attending to the first token 𝑡0 of 𝑡activation is changed to:
𝑊𝑇
2
(︃
𝜎
(︃
𝑊𝑇
1
(︃
Softmax
(︃
𝐸𝑡,0𝑊𝑞𝑊𝑇
𝑘
[︂𝐸𝑙,:
𝐸𝑡,0
]︂𝑇)︃[︂𝐸𝑙,:
𝐸𝑡,0
]︂
𝑊𝑣
)︃
+ 𝑏1
)︃)︃
+ 𝑏2.
(6)
Proposition 2.1 (One-Step Pseudo Gradient Update). There exists a set of parameters, denoted as 𝜃′
𝑡, which
includes
{︀
𝑊′
𝑞, 𝑊′
𝑘, 𝑊′
𝑣, 𝑊′
1, 𝑊′
2, 𝑏′
1, 𝑏′
2
}︀
, allowing Equation (6) to be expressed in the following form:
𝑊′𝑇
2
(︀
𝜎
(︀
𝑊′𝑇
1
(︀
Softmax
(︀
𝐸𝑙,−1:𝑊′
𝑞𝑊′𝑇
𝑘𝐸𝑇
𝑙,:
)︀
𝐸𝑙,:𝑊′
𝑣
)︀
+ 𝑏′
1
)︀)︀
+ 𝑏′
2,
(7)
where 𝜃′
𝑡represents the one-step update of 𝜃and the increment ∆ℳ𝜃(𝐼, 𝑞, 𝑡0) is only associated with 𝜃, 𝐼, 𝑞, and
𝑡0.
According to Proposition 2.1 (the proof can be found in Appendix B), as the model progressively attends to
the entire reasoning trajectory, the model parameters 𝜃are updated incrementally, a process referred to as the
pseudo gradient update.
3

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
  of q0
  of q1
  of q2
  of q3
Figure 2: Landscape of the plausibility regarding LLMs to generate accurate answers. We apply the methodology
proposed by Li et al. [53]. The questions 𝑞0, 𝑞1, 𝑞2, 𝑞3 are selected from AIME24. Additionally, we project the
trajectory of the pseudo-gradient update onto the landscape (purple line). Please refer to Appendix C.1 for
more details.
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
Figure 3: Visualization of the pseudo-gradient update: The 𝑥-axis represents the normalized indices of corre-
sponding trajectories. 𝑞0, 𝑞1, 𝑞2, 𝑞3 are question selected from AIME24, refer to Appendix C.1 for more details.
Empirical Examples. We provide empirical examples of the pseudo-gradient update using QwQ-32B [101]
to perform reasoning on AIME24, in which the model’s confidence in the answer functions serves as a probe.
Specifically, we calculate the negative log-probability of the answer at each position (denoted aŝ︀ℒ) within the
generated trajectories by appending Final Answer∖n∖boxed{..answer..} at each position. This method
provides an alternative approach to observing the overall optimization objective (Equation (3)), with models
becoming more optimal aŝ︀ℒdecreases. Figure 2 displays the corresponding landscape of the negative log-
probability. As shown in Figure 3, the negative log-probability progressively decreases along the reasoning
trajectories which aligns with our definition. Additional visualizations are provided in Appendix E.
2.3. A Meta-Learning Perspective on LLM Reasoning
Building upon the previous discussion, we present a meta-learning perspective, named RaML, for modeling
the process and the training of the LLM reasoning capability. We consider each question 𝑞𝑖within the question
set as an independent task in the meta-learning. During training (e.g., supervised fine-tuning [30, 43, 79] or
reinforcement learning [27, 67, 73, 87]) on the question set {𝑞𝑖}, the LLM ℳ𝜃is prompted to solve a new
question 𝑞𝑖by following a reasoning trajectory 𝑡. Initially, the parameters 𝜃are updated to 𝜃′
𝑡using one or
more pseudo gradient descent update indicated by reasoning trajectory 𝑡. Subsequently, the LLM is optimized by
pseudo second order gradient, formally expressed as follows:
min
𝜃
∑︁
𝑞𝑖∈𝒬
∑︁
𝑡∈𝒯𝑖
ℒ𝑞𝑖
(︀
ℳ𝜃′
𝑡
)︀
= min
𝜃
∑︁
𝑞𝑖∈𝒬
∑︁
𝑡∈𝒯𝑖
ℒ𝑞𝑖
(︀
ℳ𝜃+Δℳ𝜃(𝐼,𝑞,𝑡)
)︀
,
(8)
where 𝒯𝑖denotes the set of reasoning trajectories corresponding to the question 𝑞𝑖and 𝜃+∆ℳ𝜃(𝐼, 𝑞, 𝑡) indicates
the multi-step update of 𝜃as detailed in Equation (4).
Intuitively, RaML can be perceived as a variant of Model-Agnostic Meta-Learning (MAML, detailed in
4

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
Algorithm 1: Model-Agnostic Meta-
Learning
Input: 𝑝(𝒯): distribution over tasks,
𝛼, 𝛽: step size hyperparameters.
1 Randomly initialize 𝜃;
2 while not done do
3
Sample batch of tasks 𝒯𝑖∼𝑝(𝒯) ;
4
for all 𝒯𝑖do
5
Evaluate ∇ℒ𝒯𝑖(𝑓𝜃) with respect
to 𝐾examples ;
6
Compute adapted parameters
with gradient descent:
𝜃′ = 𝜃−𝛼∇𝜃ℒ𝒯𝑖(𝑓𝜃) ;
7 Update 𝜃←𝜃−𝛽∇∑︀
𝒯𝑖∼𝑝(𝒯) ℒ𝒯𝑖(𝑓𝜃′) ;
Algorithm 2: Meta-Learning Perspective on LLM
Reasoning
Input: ℳ𝜃: LLM, 𝐼: instruction, 𝒬: question set,
𝒯𝑖: reasoning trajectories for each question
𝑖.
1 while not training done do
2
Sample batch of questions 𝑞𝑖from 𝒬;
3
for all 𝑞𝑖do
4
Obtain reasoning trajectories 𝒯𝑖of each 𝑞𝑖
through training data or rollout ;
5
Update 𝜃to 𝜃′
𝑡𝑗by reasoning trajectory
𝑡𝑗∈𝒯𝑖refer to Equation (4) ;
6
Optimize 𝜃through ∑︀
𝑡𝑗∈𝒯𝑖ℒ𝑞𝑖
(︁
ℳ𝜃′
𝑡𝑗
)︁
for
each 𝑞𝑖;
Table 1: Comparison of training techniques, where SFT, PO, and RL mean the abbreviation of supervised
fine-tuning, preference optimization, and reinforcement learning, respectively.
Techniques
Reasoning Trajectories
Outer Loop Loss
SFT
Off-policy
ℒ= −E(𝑥,𝑦)∼𝒟[log 𝑝𝜃(𝑦| 𝑥)] [79]
Off-Policy PO
Off-policy
ℒ= −log 𝜎
(︀
𝑟𝜃(𝑦preferred) −𝑟𝜃(𝑦dispreferred)
)︀
[80]
On-Policy RL
On-policy
ℒ= −E𝑡
[︁
min
(︁
𝑟𝑡(𝜃) ˆ𝐴𝑡, clip(𝑟𝑡(𝜃), 1 −𝜖, 1 + 𝜖) ˆ𝐴𝑡
)︁]︁
[87]
Algorithm 1) [34], where the update of 𝜃using reasoning trajectories function as the inner loop, while the
final optimization of the answer decoding distribution constitutes the outer loop, as outlined in Algorithm 2. In
RaML, the gradient update associated with the latent support set is represented by the reasoning trajectories,
whereas the answer denotes the query set. There are no explicit evaluations (i.e., loss computation and backward)
during the inner loop, as the gradient update is implicitly dictated by the reasoning trajectories. Although the
model weights have not been directly updated, the pseudo update enables the LLM to simulate question-specific
optimization within a specific reasoning trajectory, thereby significantly enhancing the accuracy and stability
of LLM reasoning. During the training process, the parameters of the LLM, denoted as 𝜃, are updated to 𝜃′
𝑞𝑖,𝑡𝑗
according to the given reasoning trajectory 𝑡𝑗in the inner loop. In the outer loop, the parameters of the LLM
are optimized using the second-order gradient (ℒ𝑞𝑖→𝜃′
𝑞𝑖,𝑡𝑗→𝜃). The LLM is optimized to provide an effective
and robust foundation for answering questions (tasks), allowing its parameters to be easily adapted based on the
reasoning trajectories associated with these questions, thereby facilitating answer generation.
Additionally, in the standard MAML process, a few-shot support set is typically required to fine-tune the
model on a new task. In the LLM reasoning scenario, this support set, comprising reasoning trajectories,
is generally generated by the LLM itself. Thus, the inner loop’s optimization process in RaML resembles
Learn-to-Optimize (L2O) [4, 54, 55], which involves learning a parameterized optimizer to automate the
optimization of various tasks by being trained to decode the reasoning trajectories. Specifically, during the LLM
reasoning training, the LLM is trained to function as the meta-optimizer, generating an inner loop optimization
path tailored to the specific question.
2.4. Instantiation of Training Techniques within RaML
We review various training techniques from a meta-learning perspective, including supervised fine-tuning [30,
43], off-policy preference optimization [80], and on-policy reinforcement learning [2, 73, 87, 89]. We propose
to categorize these techniques into two macro-level stages. The first stage involves acquiring reasoning
trajectories and inputting them into the LLM to update its parameters 𝜃, thereby obtaining the output token
distribution through updated 𝜃. Subsequently, the LLM parameters 𝜃are optimized using a specific loss function
based on this output distribution. Since various loss functions lead to the same maximum likelihood estimation
(MLE) [95], we attribute the essential difference between different training techniques to their inner loop
5

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
Table 2: The evaluation performance of Qwen2.5-7B-Base trained using both SFT and Zero-GRPO training
methods. In this context, "Qwen" refers to the abbreviation for Qwen2.5-Math-Instruct, and "Distil-Qwen"
denotes DeepSeek-R1-Distill-Qwen-14B. Green cells indicate the best performance in each column, while Blue
cells indicate the second-best performance.
Techniques
Source
AIME24
MATH500-L5
LiveMathBench-Hard
Pass@8 ↑
mG-Pass@8 ↑
Pass@8 ↑
mG-Pass@8 ↑
Pass@8 ↑
mG-Pass@8 ↑
SFT
Qwen
20.34
7.43
58.42
35.65
26.77
7.43
Distill-Qwen
36.69
10.29
82.98
45.79
25.15
10.46
Zero-GRPO
-
27.37
4.08
71.66
30.48
27.48
8.21
optimization. Inner loop optimization is crucial in meta-learning training, as the meta-gradient is essential for
enhancing meta-learning performance. As illustrated in Table 1, off-policy training techniques obtain reasoning
trajectories through manual collection or synthesis, while on-policy training techniques generate reasoning
trajectories based on the model distribution. From the perspective of learning to optimize, off-policy training
techniques are equivalent to learning from an optimal meta-optimizer, directing the optimization of the inner
loop. In contrast, RL requires independently exploring the inner loop’s optimization path, presenting challenges
due to increased freedom but allowing for potentially greater optimization outcomes.
3. Empirical Analysis on LLM Reasoning from Meta-Learning Perspective
Building on a meta-learning perspective of LLM reasoning, this section explores key factors that influence LLM
reasoning. Specifically, we study and analyze key issues of interest in the research community regarding LLM
reasoning by instantiating them within the framework of meta-learning, referencing relevant research findings
in this domain. We focus on the following problems:
❶Which training strategy, SFT or RL, is more effective for LLM reasoning, and why (§ 3.2)?
❷Why do longer reasoning trajectories enhance reasoning performance (§ 3.3)?
❸What principles behind reasoning-efficiency methodology contribute to the trade-off between cost and
performance (§ 3.3)?
❹Does trajectory-aided reasoning generalize effectively across different domains (§ 3.4)?
3.1. Experiment Setup
Reasoning Task. In this paper, we mainly focus on the mathematical reasoning task due to its broad applicability
and prominence in the research community and we also include other reasoning tasks in § 3.4 for the study of
generalization.
Training.
To minimize the impact of the post-training, we train Qwen2.5-7B-Base [117] from scratch and
conduct experiments on it. We involve SFT for the off-policy training and (Zero-)GRPO [89] for the on-policy
training. The training data, sourced from Open Reasoner Zero [44], initially comprised approximately 57k
questions, refined to 39k through filtering (see Appendix C.2 for details). Synthetic reasoning trajectories are
generated using Qwen2.5-Math-72B-Instruct [118] and DeepSeek-R1-Distill-Qwen-14B [27]. Further training
details are provided in Appendix C.2.
Evaluation. We primarily evaluate performance using three mathematical reasoning benchmarks orthogonal
to the training data: AIME24 1, MATH500 [57] (Level 5 questions selected for greater discrimination), and
LiveMathBench-Hard [60]. We also include GPQA [83] and LiveCodeBench [47] to assess generalization.
Model outputs are generated with a temperature of 1.0, top-𝑝of 0.8, top-𝑘of 50, and a maximum output
length of 16, 384 tokens. We report mG-Pass@𝑘[60] for stability and Pass@𝑘[17] for performance. Additional
evaluation details are in Appendix C.3.
1https://huggingface.co/datasets/AI-MO/aimo-validation-aime
6

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
10
1
10
3
10
5
#Training data
10
20
30
40
50
60
70
Avg@32
DAPO
VAPO
Sky-T1-32B-Preview
Sky-T1-32B-Flash
LIMO
Bespoke-Stratos-32B
s1.1-32B
DeepSeek-R1-Distill-Qwen-32B
Light-R1-32B
OpenThinker-32B
OpenThinker2-32B
Qwen2.5-32B-Base
Qwen2.5-32B-Instruct
Off-Policy v. s. On-Policy on AIME24
10
1
10
3
10
5
#Training data
10
20
30
40
50
60
mG-Pass@32
DAPO
Sky-T1-32B-Preview
Sky-T1-32B-Flash
LIMO
Bespoke-Stratos-32B
s1.1-32B
DeepSeek-R1-Distill-Qwen-32B
Light-R1-32B
OpenThinker-32B
OpenThinker2-32B
Qwen2.5-32B-Base
Qwen2.5-32B-Instruct
Off-Policy v. s. On-Policy on AIME24
On-Policy
Off-Policy
Base Model
Figure 4: Performance of base models, models trained on off-policy data, and models trained on on-policy
data using the AIME24 dataset, with the 𝑥-axis representing the amount of training data. We generate 64
for each question and report Pass@32 and mG-Pass@32. The evaluation includes prominent models such as
Sky-T1-32B [98], Bespoke-Stratos-32B [50], LIMO [124], s1.1-32B [69], OpenThinker-32B [99], Light-R1-
32B [112], DeepSeek-R1-Distill-Qwen-32B [27], DAPO-32B [125], and VAPO-32B [127]. These models are
based on either Qwen2.5-32B or Qwen2.5-32B-Instruct only through SFT or RL (Zero-RL). Since VAPO is not
open source, we copy its results from the original paper.
3.2. Inner Loop Optimization v.s. Reasoning Trajectory Source
The inner loop optimization is crucial in meta-learning, as the results of this process significantly impact the
stability and performance of the final model. In the definition of RaML, the LLM needs to learn as the inner
loop optimizer, generating an optimization trajectory for each query. It is well-documented that training learned
optimizers presents considerable challenges [51]. In this section, we will discuss the development of inner loop
optimizers for SFT and GRPO training techniques. The biggest difference between them is the source of the
reasoning trajectories used for training: 1) on-policy, where the trajectories are generated by the LLM currently
being updated, and 2) off-policy, where the trajectories are generated either by other LLMs or by a previously
trained version of the same LLM (e.g., through reject sampling [92, 126]).
Status Quo of SFT v.s. RL. Recent studies [14, 20, 116] claim the superiority of the on-policy strategy in LLM
reasoning training. For example, GRPO-based DeepSeek-R1-Zero [27] outperforms DeepSeek-V3, which is
trained on large-scale off-policy synthetic data, in mathematical reasoning tasks, scoring 71.0 compared to 39.8
on AIME24, thereby reinforcing the advantages of on-policy strategies. However, as our results in Table 2 and
the evaluation results of community models in Figure 4, GRPO-trained models do not consistently outperform
SFT-trained models for the same base LLM, consistent with findings in [27, 100, 112].
SFT Leads to Stable Inner Loop Optimization. Learning to optimize frequently encounters challenges such
as unstable training, easy divergence, and limited generalization. To address these issues, researchers [76,
103] have suggested employing optimal optimizers as “guardian” optimizers, integrating their features to
ensure convergence and stability. The training reasoning trajectories used by SFT originate from human-
annotated or other advanced reasoning models. These trajectories can be viewed as guides from an oracle
optimizer. Consequently, SFT achieves a stable and effective inner loop optimization process, leading to superior
performance. However, this does not imply that reinforcement learning always has disadvantages. RL provides
greater freedom to explore optimization paths and, given sufficient model capability and exploration steps,
offers a higher theoretical upper limit.
Combination of SFT and RL for Stable Inner Loop Optimization. A straightforward idea involves training
the LLM using an optimal optimizer to stabilize its performance. Subsequently, reinforcement learning can
be employed to explore improved paths for inner-loop optimization. As evidenced in Table 3, the RL model
demonstrates substantial enhancements after supervised fine-tuning.
7

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
Table 3: Performance of Zero-GRPO model and GRPO model based on the SFT cold start.
Techniques
AIME24
MATH500-L5
LiveMathBench-Hard
Pass@8 ↑
mG-Pass@8 ↑
Pass@8 ↑
mG-Pass@8 ↑
Pass@8 ↑
mG-Pass@8 ↑
Zero-GRPO
27.37
4.08
71.66
30.48
27.48
8.21
+ SFT Cold Start
35.87↑31%
11.23↑175%
82.42↑15%
44.92↑47%
42.17↑53%
18.84↑129%
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
Figure 5: Illustration of QwQ’s pseudo-gradient update for both thinking and non-thinking modes and refer
to Appendix E for more examples. We visualize four pairs of correct reasoning trajectories for one question
in AIME24. Compared with thinking trajectories, no-thinking trajectories converge more quickly, which also
easily falls into local optimal points.
Takeaway for SFT v.s. RL
❶SFT provides stable inner loop optimization by training on trajectories from oracle inner loop optimizer
compared with RL.
❷Combining SFT with RL shows significant performance improvements by utilizing SFT for initializing
inner-loop optimization and RL for further exploration.
3.3. Inner Loop Optimization Steps v.s. Reasoning Trajectory Tokens
In RaML, each token in a single reasoning trajectory corresponds to an individual optimization step, and the
length of the trajectory indicates the total number of update steps. We examine these factors by integrating
experimental results with related research studies.
Long Reasoning Trajectories Lead to Superior Performance. As shown in Table 2, models trained with longer
reasoning trajectories consistently outperform those with shorter trajectories, aligning with meta-learning
findings that extended inner-loop updates enhance performance. This observation is consistent with the superior
performance of long CoT reasoning models, such as DeepSeek-R1 [27], suggesting that longer trajectories
increase inner-loop update steps, thereby improving LLM reasoning capabilities.
Different Reasoning Trajectory Tokens Represent Varying Roles of Update. We focus on discussing two
intriguing findings in LLM reasoning. First, advanced reasoning in LLMs has been observed to have an aha
moment [27]. This refers to specific reflection tokens that prompt LLMs to devote additional time to thinking
about questions. These tokens are also utilized to implement test-time scaling [61, 69]. Following the settings
described in § 2.2, we measure the relative changes in thê︀ℒvalue before and after each token position. The
results are presented in Figure 6. We observe that reflection tokens such as “Wait” and “Alternatively” indicate
a significant change in the objective. From an optimization perspective, we propose that these reflection
tokens assist the model in escaping saddle points. As the model gradually approaches a stable state, these
tokens provide a larger gradient, thereby expanding the exploration space to find a better parameter space.
In the following part, we explore the concept of reasoning efficiency, as discussed by various researchers
[61, 78, 100, 119, 130]. This concept involves optimizing the balance between decoding cost and performance
utilizing specific segments, such as the end-of-thinking token delimiter. We hypothesize that these termination
delimiters enhance convergence at the optimization level, akin to the role of momentum in optimization,
facilitating rapid convergence of model parameters within a flatter region. However, this acceleration does
not always lead to the optimal point. Also refer to the settings described in § 2.2, we append the end-
8

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
avg
Wait
Therefore
Alternatively
Hold on
Since
10
0
10
1
10
2
log
0.56
1.22
0.76
16.33
279.50
4.54
Token w. r. t. 
Figure 6: Illustration of the influence of re-
flection tokens. Reflection tokens lead to
sharper objective change.
0
10
20
30
Pass@8
AIME24
0
10
20
30
LiveMathBench-Hard
0
25
50
75
GPQA
0
20
40
LiveCodeBench
0
5
10
mG-Pass@8
AIME24
0
5
10
LiveMathBench-Hard
0
10
20
GPQA
0
5
10
LiveCodeBench
Qwen-Base
SFT-from-Qwen
SFT-from-Distil-Qwen
Zero-RL-from-GRPO
Figure 7: Evaluation results of base, SFT and GRPO models
on AIME24, LiveMathBench-Hard, GPQA-Diamond, and Live-
CodeBench.
of-thinking token delimiter Therefore, after all this, I believe the answer is following the
thinking token delimiter <think>. Figure 5 demonstrates that trajectories using the end-of-thinking token
delimiter achieve quicker convergence, confirming our hypothesis to some extent. Since the QwQ model
does not completely adapt to the no-thinking mode, we include additional experiments related to Qwen3 in
Appendix E. These experiments further substantiate our conclusions.
Takeaway for Reasoning Trajectory Tokens and Inner Loop Optimization Steps
❶Long reasoning trajectories are analogous to performing additional steps of inner-loop optimization,
which improves inner-loop optimization and further enhances outer-loop optimization, i.e., the
reasning performance of LLMs.
❷Different tokens serve distinct functions in the inner loop optimization process. For instance, tokens
associated with reflection patterns promote the exploration of optimization paths, whereas special
tokens regulating the length of reasoning in the recent Long-CoT LLMs facilitate fast-converging
optimization steps.
3.4. Task Generalization v.s. Reasoning Generalization
Meta-learning models typically exhibit strong generalization across tasks with similar distributions, since the
models learn generalized representations for these tasks. We investigate whether this applies to LLM reasoning,
where each question is a distinct task but shares fundamental reasoning skills, suggesting a similar distribution.
We analyze generalization from two perspectives: within-domain generalization (same reasoning domain)
and cross-domain generalization (different reasoning domains). Training data, sourced from Open Reasoner
Zero (§ 3.1), consist of mathematical problems from MATH, making AIME24 and LiveMathBench-Hard suitable
for within-domain evaluation. Results in Figure 7 show improved performance for both SFT and GRPO models
on these benchmarks. For cross-domain generalization, we evaluated SFT and GRPO models on GPQA (scientific
reasoning) and LiveCodeBench (code reasoning). As illustrated in the right section of Figure 7, all trained
models outperformed the base model on both benchmarks.
Our findings align with existing research. For instance, studies have shown that large models trained on
code datasets exhibit strong logical reasoning capabilities [17]. Additionally, research indicates that training
on mathematical and code corpora mutually enhances performance [38, 46, 108].
9

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
1
2
4
8
16
32
#Trajectories
10
20
30
40
Pass@8
AIME24
1
2
4
8
16
32
#Trajectories
60
70
80
Pass@8
MATH500-L5
1
2
4
8
16
32
#Trajectories
20.0
22.5
25.0
27.5
30.0
Pass@8
LiveMathBench-Hard
1
2
4
8
16
32
#Trajectories
10
15
20
25
mG-Pass@8
AIME24
1
2
4
8
16
32
#Trajectories
35
40
45
50
mG-Pass@8
MATH500-L5
1
2
4
8
16
32
#Trajectories
9
10
11
12
mG-Pass@8
LiveMathBench-Hard
Trajectories from Qwen2.5-Math-72B-Instruct
Trajectories from DeepSeek-R1-Distill-Qwen-14B
Figure 8: Evaluation results w.r.t. different number of reasoning trajectories of SFT models on AIME24,
MATH500-L5, and LiveMathBench-Hard.
Takeaway for Reasoning Generalization
Training LLMs using trajectories facilitates the learning of shared features across diverse reasoning
problems. This process, akin to meta-learning, enables the parameters of LLMs to adapt efficiently by
new trajectories and demonstrate generalization to out-of-distribution questions.
3.5. More Discussions about LLM Reasoning
Additional discussions on current research advancements in LLM reasoning are provided in Appendix D within
the framework of RaML. Numerous improvements can be interpreted through the lens of meta-learning,
highlighting the significant potential of comprehending LLM reasoning through RaML.
4. Advancing LLM Reasoning Drawing Inspiration from Meta-Learning
In this section, we undertake several preliminary investigations informed by meta-learning to improve LLM
reasoning, demonstrating the feasibility of enhancing LLM reasoning by integrating meta-learning studies.
4.1. Manipulating Training Reasoning Trajectories per Question
Previous studies [1, 15] highlight that the size of the support set is of paramount importance in improving
performance, stability, and convergence in meta-learning. In RaML, the support set is intrinsically connected
with the number of reasoning trajectories trained per question, prompting the question: Can enhancements in
support set size contribute to more effective training in LLM reasoning? In this preliminary study, we investigate
the impact of increasing the number of training reasoning trajectories per question on LLM reasoning.
SFT. We train Qwen2.5-7B-Base through SFT with {1, 2, 4, 8, 16, 32} synthetic reasoning trajectories, ensuring
equal training frequency per question. Evaluation results (Figure 8) show that increasing the number of
trajectories improves performance and reasoning stability across all benchmarks, suggesting that additional
trajectories enhance supervised fine-tuning outcomes [118].
GRPO. Regarding to GRPO, the support set size corresponds to the number of trajectories in the rollout group
for each prompt (question). To maintain stable advantage estimation in GRPO and ensure a fair comparison,
we fix the rollout group size at 16 and calculated the advantage for each trajectory. During gradient updates,
10

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
1
2
3
4
#Steps
15
20
25
Pass@8
AIME24
1
2
3
4
#Steps
60
65
70
Pass@8
MATH500-L5
1
2
3
4
#Steps
27.5
30.0
32.5
35.0
37.5
Pass@8
LiveMathBench-Hard
1
2
3
4
#Steps
2.5
5.0
7.5
10.0
mG-Pass@8
AIME24
1
2
3
4
#Steps
20
30
40
mG-Pass@8
MATH500-L5
1
2
3
4
#Steps
2.5
5.0
7.5
10.0
12.5
mG-Pass@8
LiveMathBench-Hard
#Update Trajectories=1
#Update Trajectories=2
#Update Trajectories=4
#Update Trajectories=8
#Update Trajectories=16
Figure 9: Evaluation results w.r.t. different number of reasoning trajectories of GRPO models on AIME24,
MATH500-L5, and LiveMathBench-Hard.
however, we randomly select 𝑛∈{1, 2, 4, 8, 16} trajectories to calculate the gradient. Experimental results
shown in Figure 9 demonstrate that: 1) multiple trajectories for a single question significantly enhance model
performance and stability; 2) a larger number of trajectories accelerates convergence. These findings explain
the superior performance and stability of GRPO-based (or other similar RL algorithms) reasoning models [60],
as the GRPO mechanism inherently optimizes for multiple trajectories per question compared with PPO [87]
and naive SFT. It is noteworthy that some studies have attempted to enhance PPO through group sampling
[44, 127] and achieve competitive performance compared with original PPO.
Takeaway for Manipulating Training Reasoning Trajectories per Question
Training on multiple trajectories per question is akin to enlarging the support set to enhance inner-loop
optimization, which consequently leads to better training performance.
4.2. Incentivizing Reasoning Efficiency by Optimization Lens
Optimization Path of Optimal Reasoning Trajectory
Optimization Path of Long Reasoning
Trajectory
Figure 10: Given a long reasoning
trajectory, there exists an optimal
corresponding reasoning trajec-
tory which leverage less tokens.
Recent advanced reasoning models face limitations due to inefficient
and excessively lengthy reasoning trajectories. Although several stud-
ies [61, 100] have attempted to minimize the number of decoding tokens
to mitigate overhead, these approaches frequently lead to decreased rea-
soning performance, thereby presenting a fundamental question: Can we
reduce the number of reasoning tokens without compromising reasoning per-
formance? As previously discussed, each reasoning trajectory corresponds
to an inner-loop optimization trajectory, thus reframing the inquiry as
follows: Can there be a s more effective inner loop optimization path? From
an optimization perspective, as illustrated in Figure 10, there exists such
an optimal inner loop optimization path. In this section, we present a
straightforward yet convincing experiment to validate the existence of this
inner loop optimization path.
Specifically, we employ Qwen3-32B [100] to generate 16 reasoning tra-
jectories for each question in AIME24, MATH500-L5, and LiveMathBench-
Hard. These trajectories serve as foundational optimization paths, and our goal is to refine them to discover
more optimal paths. We propose an heuristic method which condense the reasoning trajectories by using a
11

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
1k
2k
4k
8k
16k
32k
20
40
60
80
Avg@16
80.63
39.79
27.92
AIME24
1k
2k
4k
8k
16k
32k
40
50
60
70
80
90
Avg@16
94.17
83.14
71.12
MATH500-L5
1k
2k
4k
8k
16k
32k
20
30
40
50
60
Avg@16
58.33
27.01
16.37
LiveMathBench-Hard-en
1k
2k
4k
8k
16k
32k
log (#Tokens)
20
40
60
80
Pass@16
93.33
80.00
60.00
1k
2k
4k
8k
16k
32k
log (#Tokens)
60
70
80
90
100
Pass@16
99.25
93.96
90.30
1k
2k
4k
8k
16k
32k
log (#Tokens)
30
40
50
60
70
Pass@16
76.19
59.52
38.10
Thinking Mode (Budget Forcing)
Summarized Reasoning Trajectories
NoThinking Mode
Figure 11: Experimental results of Qwen3-32B with 1) thinking mode, which includes a budget-forcing curve
achieved by truncating decoding with various thinking tokens and prompting the model to generate the answer
by appending ∖n∖n**Final Answer**∖n∖boxed{, 2) summarized trajectory, and 3) nothinking mode on
AIME24, MATH500-L5, and LiveMathBench-Hard-en.
LLM to summarize the original trajectories into shorter variants, which are then used to prompt Qwen3-32B
for answer generation. The summarizations are generated by Qwen2.5-32B-Instruct, deliberately excluding
answers from the summarized reasoning trajectories. We performed four summary generators to reduce the
impact of randomness. Figure 11 displays the experimental results. Notably, we observe that Qwen3-32B’s
performance with summarized reasoning trajectories is comparable to its performance in thinking mode espe-
cially for the Pass@16 metric, while significantly reducing the number of tokens in the reasoning trajectories.
Moreover, Qwen3-32B’s performance using summarized reasoning trajectories surpasses that in no-thinking
mode, even that tThe latter has more tokens.
Our experiments demonstrate that trained long-CoT LLMs have the potential to achieve optimal reasoning
trajectories that require fewer tokens while maintaining comparable reasoning performance. We approximate
these trajectories using a straightforward method, leaving the exploration of more advanced approaches for
future work.
Takeaway for Incentivizing Reasoning Efficiency by Optimization Lens
As an optimization, long reasoning trajectories have corresponding optimal trajectories that maintain
reasoning performance while using fewer tokens. Find a method to reproduce these optimal trajectories
during the decoding stage can enhance reasoning efficiency.
4.3. Future Directions
Building on previous discussions and experiments, we demonstrate the feasibility of advancing LLM reasoning
by drawing inspiration from meta-learning research. In this section, we outline several potential directions for
future work, including:
• Further Understanding Reasoning Trajectories
❶Unlike typical meta-learning frameworks with predefined support sets, the reasoning trajectories
in LLMs are self-generated. This implies that LLMs inherently learn gradient update steps without
needing explicit support sets for fine-tuning. Investigating how LLMs learn to form effective reasoning
trajectories, namely, gradient update steps, presents an intriguing challenge.
❷Tokens contribute differently to the modification of model parameters. What accounts for this disparity
among tokens? Is it connected to their semantic properties, and if so, in what manner?
❸Trajectory-aided reasoning in large language models (LLMs) demonstrates comparable generalization
12

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
abilities across various tasks. What aspects of the learning process contribute to this generalization ability,
and which meta-features are developed through the optimization of reasoning trajectories?
• Towards Enhancing LLM Reasoning
❶Improved Reasoning Trajectory Selection Strategy in LLM Training. During both supervised fine-tuning
and reinforcement learning, reasoning trajectories usually remain constant. Could implementing an
adaptive sampling mechanism, similar to those utilized in meta-learning [59, 120], enhance training
efficacy?
❷Enhancing Reasoning Efficiency Through an Optimization Perspective.
Given that each token offers
a unique contribution to optimization, is there a strategy to discern these contributions to filter out
superfluous tokens, thereby improving reasoning efficiency?
❸Task (Question) Ratio to Enhance Generalization Across Different Domains. Insights from related studies,
such as Collins et al. [22] and Wang et al. [109], in meta-learning, suggest methods to bolster the
reasoning capability of LLMs, enabling them to generalize across domains—for instance, training on
mathematical data and inferring insights from coding data.
5. Related Work
LLM Reasoning. The reasoning capabilities of large language models (LLMs) have progressively advanced
through the development of several key technologies, which have substantially enhanced their performance
on complex tasks. In-Context Learning [9, 13, 31, 65, 84] enables models to perform tasks by interpreting
examples provided in prompts without requiring additional training. However, this approach relies heavily on
the model’s pre-trained knowledge and careful prompt design, limiting its effectiveness for complex reasoning
tasks [64]. The introduction of Chain of Thought (CoT) [10, 110, 122] prompting has significantly improved
LLM performance in areas such as mathematical reasoning, commonsense reasoning, and symbolic reasoning by
guiding the models to produce intermediate reasoning steps. Supervised Fine-Tuning (SFT) further refines the
reasoning capabilities of LLMs by training them with labeled datasets tailored to specific tasks [6, 98, 99, 124].
Reinforcement Learning (RL), through the use of reward mechanisms, has become a critical approach to
optimizing model behavior and enhancing reasoning abilities. Recently, Long-Chain of Thought (Long-CoT)
models have emerged as a notable trend in reasoning research, generating detailed reasoning steps to better
address complex tasks [26, 27, 72, 97, 101].
Understanding LLMs. The remarkable success of LLMs has spurred extensive research into their capabilities.
Early studies [128, 129] explored function approximation, demonstrating that sufficiently large transformers
[107] can universally approximate any continuous sequence-to-sequence mapping on a compact domain.
Subsequent research investigated the computational power of transformers [11, 19, 29, 37, 40, 58, 62, 111,
121]. For example, Feng et al. [33] validated the necessity of chain-of-thought (CoT) prompting for solving
complex problems using circuit complexity theory. Other works [36, 45] demonstrate that transformers
can learn to implement learning algorithms, such as gradient descent, within trajectories. Several studies
[3, 5, 25, 36, 70, 115] have focused on understanding in-context learning (ICL) [13, 31], examining the role
of demonstration examples.
Meta-Learning. Meta-learning, commonly referred to as “learning to learn”, aims to enable models to enhance
their learning strategies by leveraging prior experience across multiple tasks. Early research in this area [7, 104]
explored methods for acquiring learning rules applicable to new tasks, with a particular emphasis on lifelong
learning. These foundational efforts established the basis for creating more adaptable and flexible learning
algorithms, paving the way for subsequent advancements. Recent meta-learning approaches can generally be
categorized into three groups: 1) metric-based methods, which focus on learning a feature space to efficiently
compare samples [16, 93, 96, 131]; 2) model-based methods, which utilize memory mechanisms or other
structures to store and retrieve task-specific information [85, 94, 113]; and 3) optimization-based methods,
which refine the learning process to facilitate rapid adaptation [34, 81, 123].
Additional Discussion. In this part, we aim to elucidate and discuss our work in comparison with several
related studies. Dai et al. [25] interpret in-context learning (ICL) as LLMs generating meta-gradients from
demonstration examples, which are applied to the base GPT model to construct an ICL system. In this work,
each demonstration example serves as one data sample to update the parameters of LLMs. In contrast, our
13

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
research emphasizes trajectory-aided reasoning, viewing each token as an update step and drawing extensive
connections to supervised fine-tuning and reinforcement learning, rather than focusing on demonstration
examples. Additionally, our approach incorporates more general training techniques with explicit parameter
optimization, whereas ICL is constrained by limited demonstration examples, which poses certain limitations.
Another research avenue explored by studies such as Gatmiry et al. [36] and others [45] shows that transformers
can learn to implement learning algorithms, such as gradient descent, within a chain of thought. However,
these studies primarily investigate whether transformers can describe learning algorithms in natural language
to solve practical numerical optimization problems, while our work delves into the internal parameter updates
of the transformer model, applicable to a broader range of problems.
6. Conclusion
This paper introduces a novel perspective on LLM reasoning by integrating it into the meta-learning framework.
Through both theoretical analysis and empirical validation, we illustrate that reasoning trajectories can be
effectively conceptualized as pseudo-gradient updates, enabling a deeper comprehension of how LLMs perform
complex reasoning tasks. Extensive experiments reveal the correlation between meta-learning and LLM
reasoning, suggesting potential avenues for advancing LLM reasoning through meta-learning principles. This
work not only enhances the understanding of LLM reasoning but also offers practical insights for improving
these models using established meta-learning techniques.
14

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
References
[1] Mayank Agarwal, Mikhail Yurochkin, and Yuekai Sun. On sensitivity of meta-learning to support data.
In NeurIPS, pages 20447–20460, 2021. 1, 4.1
[2] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet
Üstün, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learning from human
feedback in llms. In ACL, pages 12248–12267. Association for Computational Linguistics, 2024. 2.4
[3] Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm
is in-context learning? investigations with linear models. In ICLR. OpenReview.net, 2023. 5
[4] Marcin Andrychowicz, Misha Denil, Sergio Gomez Colmenarejo, Matthew W. Hoffman, David Pfau,
Tom Schaul, and Nando de Freitas. Learning to learn by gradient descent by gradient descent. In NIPS,
pages 3981–3989, 2016. 1, 2, 6
[5] Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians: Provable
in-context learning with in-context algorithm selection. In NeurIPS, 2023. 2.2, 5
[6] Edward Beeching, Shengyi Costa Huang, Albert Jiang, Jia Li, Benjamin Lipkin, Zihan Qina, Kashif Rasul,
Ziju Shen, Roman Soletskyi, and Lewis Tunstall. Numinamath 72b cot. https://huggingface.co/
AI-MO/NuminaMath-72B-CoT, 2024. 5
[7] Samy Bengio, Yoshua Bengio, Jocelyn Cloutier, and Jan Gecsei. On the optimization of a synaptic
learning rule. In Optimality in Artificial and Biological Neural Networks, pages 6–8, 1992. 5
[8] Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach
Moshe, Tomer Ronen, Najeeb Nabwani, Ido Shahaf, Oren Tropp, Ehud Karpas, Ran Zilberstein, Jiaqi
Zeng, Soumye Singhal, Alexander Bukharin, Yian Zhang, Tugrul Konuk, Gerald Shen, Ameya Sunil
Mahabaleshwarkar, Bilal Kartal, Yoshi Suhara, Olivier Delalleau, Zijia Chen, Zhilin Wang, David Mosal-
lanezhad, Adi Renduchintala, Haifeng Qian, Dima Rekesh, Fei Jia, Somshubra Majumdar, Vahid Noroozi,
Wasi Uddin Ahmad, Sean Narenthiran, Aleksander Ficek, Mehrzad Samadi, Jocelyn Huang, Siddhartha
Jain, Igor Gitman, Ivan Moshkov, Wei Du, Shubham Toshniwal, George Armstrong, Branislav Kisacanin,
Matvei Novikov, Daria Gitman, Evelina Bakhturina, Jane Polak Scowcroft, John Kamalu, Dan Su, Kezhi
Kong, Markus Kliegl, Rabeeh Karimi, Ying Lin, Sanjeev Satheesh, Jupinder Parmar, Pritam Gundecha,
Brandon Norick, Joseph Jennings, Shrimai Prabhumoye, Syeda Nahida Akter, Mostofa Patwary, Ab-
hinav Khattar, Deepak Narayanan, Roger Waleffe, Jimmy Zhang, Bor-Yiing Su, Guyue Huang, Terry
Kong, Parth Chadha, Sahil Jain, Christine Harvey, Elad Segal, Jining Huang, Sergey Kashirsky, Robert
McQueen, Izzy Putterman, George Lam, Arun Venkatesan, Sherry Wu, Vinh Nguyen, Manoj Kilaru,
Andrew Wang, Anna Warno, Abhilash Somasamudramath, Sandip Bhaskar, Maka Dong, Nave Assaf,
Shahar Mor, Omer Ullman Argov, Scot Junkin, Oleksandr Romanenko, Pedro Larroy, Monika Katariya,
Marco Rovinelli, Viji Balas, Nicholas Edelman, Anahita Bhiwandiwalla, Muthu Subramaniam, Smita
Ithape, Karthik Ramamoorthy, Yuting Wu, Suguna Varshini Velury, Omri Almog, Joyjit Daw, Denys
Fridman, Erick Galinkin, Michael Evans, Katherine Luna, Leon Derczynski, Nikki Pope, Eileen Long, Seth
Schneider, Guillermo Siman, Tomasz Grzegorzek, Pablo Ribalta, Monika Katariya, Joey Conway, Trisha
Saar, Ann Guan, Krzysztof Pawelec, Shyamala Prayaga, Oleksii Kuchaiev, Boris Ginsburg, Oluwatobi
Olabiyi, Kari Briski, Jonathan Cohen, Bryan Catanzaro, Jonah Alben, Yonatan Geifman, Eric Chung, and
Chris Alexiuk. Llama-Nemotron: Efficient reasoning models. CoRR, abs/2505.00949, 2025. D
[9] Amanda Bertsch, Maor Ivgi, Uri Alon, Jonathan Berant, Matthew R. Gormley, and Graham Neubig.
In-context learning with long-context models: An in-depth exploration. CoRR, abs/2405.00200, 2024. 5
[10] Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi,
Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. Graph of
thoughts: Solving elaborate problems with large language models. In AAAI, pages 17682–17690. AAAI
Press, 2024. 5
[11] Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal. On the ability and limitations of transformers to
recognize formal languages. In EMNLP, pages 7096–7116. Association for Computational Linguistics,
2020. 5
15

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
[12] Arne Bjerhammar. Application of calculus of matrices to method of least squares: with special reference
to geodetic calculations. Trans. Roy. Inst. Tech. Stockholm., 1951. B
[13] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss,
Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens
Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack
Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language
models are few-shot learners. In NeurIPS, 2020. 2.2, 5
[14] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie.
Sft or rl? an early investigation into training r1-like reasoning large vision-language models. CoRR,
abs/2504.11468, 2025. 3.2
[15] Jiaxin Chen, Xiao-Ming Wu, Yanke Li, Qimai Li, Li-Ming Zhan, and Fu-Lai Chung. A closer look at the
training strategy for modern meta-learning. In NeurIPS, 2020. 4.1
[16] Jiaxin Chen, Li-Ming Zhan, Xiao-Ming Wu, and Fu-Lai Chung. Variational metric scaling for metric-based
meta-learning. In AAAI, pages 3478–3485. AAAI Press, 2020. 5
[17] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan,
Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger,
Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder,
Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet,
Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel
Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin,
Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua
Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati,
Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech
Zaremba. Evaluating large language models trained on code. CoRR, abs/2107.03374, 2021. 3.1, 3.4
[18] Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang
Zhou, Te Gao, and Wanxiang Che. Towards reasoning era: A survey of long chain-of-thought for
reasoning large language models. CoRR, abs/2503.09567, 2025. 1
[19] David Chiang, Peter Cholak, and Anand Pillay. Tighter bounds on the expressivity of transformer
encoders. In ICML, volume 202 of Proceedings of Machine Learning Research, pages 5544–5562. PMLR,
2023. 1, 5
[20] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le,
Sergey Levine, and Yi Ma. SFT memorizes, RL generalizes: A comparative study of foundation model
post-training. CoRR, abs/2501.17161, 2025. 3.2
[21] Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong
reinforcement learning baseline for model reasoning. CoRR, abs/2504.02546, 2025. D
[22] Liam Collins, Aryan Mokhtari, and Sanjay Shakkottai. Task-robust model-agnostic meta-learning. In
NeurIPS, 2020. 4.3
[23] Liam Collins, Aryan Mokhtari, and Sanjay Shakkottai. How does the task landscape affect MAML
performance? In CoLLAs, volume 199 of Proceedings of Machine Learning Research, pages 23–59. PMLR,
2022. 1
[24] George Cybenko. Approximations by superpositions of a sigmoidal function. MCSS, 2:183–192, 1989. B
[25] Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can gpt learn
in-context? language models secretly perform gradient descent as meta-optimizers. In ACL (Findings),
pages 4005–4019. Association for Computational Linguistics, 2023. 2.2, 5
[26] Google Deepmind.
Gemini 2.5:
Our most intelligent ai model.
https://blog.google/
technology/google-deepmind/gemini-model-thinking-updates-march-2025/
#gemini-2-5-thinking/, 2025. Accessed: 2025-03-26. 1, 5
16

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
[27] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao
Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou,
Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng,
Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen,
Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li,
H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu,
Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li,
J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu,
Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang,
Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng
Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang,
R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu
Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, and S. S. Li. Deepseek-r1: Incentivizing reasoning
capability in llms via reinforcement learning. CoRR, abs/2501.12948, 2025. 1, 2.2, 2.3, 3.1, 4, 3.2, 3.3,
5, C.2
[28] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao,
Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie
Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang,
Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui
Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang
Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan,
Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue
Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning
Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du,
R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi
Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng
Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao
Yun, Tian Pei, Tianyu Sun, W. L. Xiao, and Wangding Zeng. Deepseek-v3 technical report. CoRR,
abs/2412.19437, 2024. 1
[29] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal
transformers. In ICLR. OpenReview.net, 2019. 5
[30] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep
bidirectional transformers for language understanding. In NAACL-HLT, pages 4171–4186. Association
for Computational Linguistics, 2019. 2.3, 2.4
[31] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong
Wu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. A survey on in-context learning. In EMNLP, pages
1107–1128. Association for Computational Linguistics, 2024. 5
[32] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang,
Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien
Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern,
Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe
Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel
Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-
Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan,
Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis
Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah
Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra,
Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet
Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu
Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua
Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li,
Kenneth Heafield, Kevin Stone, and et al. The llama 3 herd of models. CoRR, abs/2407.21783, 2024. 1
17

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
[33] Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the
mystery behind chain of thought: A theoretical perspective. In NeurIPS, 2023. 1, 5
[34] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of
deep networks. In ICML, volume 70 of Proceedings of Machine Learning Research, pages 1126–1135.
PMLR, 2017. 1, 2, 6, 5
[35] Deqing Fu, Tian-Qi Chen, Robin Jia, and Vatsal Sharan. Transformers learn to achieve second-order
convergence rates for in-context linear regression. In NeurIPS, 2024. 2.2
[36] Khashayar Gatmiry, Nikunj Saunshi, Sashank J. Reddi, Stefanie Jegelka, and Sanjiv Kumar.
Can
looped transformers learn to implement multi-step gradient descent for in-context learning? In ICML.
OpenReview.net, 2024. 1, 5
[37] Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papail-
iopoulos. Looped transformers as programmable computers. In ICML, volume 202 of Proceedings of
Machine Learning Research, pages 11398–11442. PMLR, 2023. 1, 2.2, 5
[38] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi,
Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. Deepseek-coder: When the large language
model meets programming - the rise of code intelligence. CoRR, abs/2401.14196, 2024. 3.4
[39] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and
Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In NeurIPS Datasets
and Benchmarks, 2021. C.3
[40] John Hewitt, Michael Hahn, Surya Ganguli, Percy Liang, and Christopher D. Manning. Rnns can generate
bounded hierarchical languages with optimal memory. In EMNLP, pages 1978–2010. Association for
Computational Linguistics, 2020. 5
[41] Kurt Hornik. Approximation capabilities of multilayer feedforward networks. Neural Networks, 4(2):251–
257, 1991. B
[42] Timothy M. Hospedales, Antreas Antoniou, Paul Micaelli, and Amos J. Storkey. Meta-learning in neural
networks: A survey. IEEE Trans. Pattern Anal. Mach. Intell., 44(9):5149–5169, 2022. 1, 2
[43] Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. In
ACL, pages 328–339. Association for Computational Linguistics, 2018. 2.3, 2.4
[44] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-
reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. CoRR,
abs/2503.24290, 2025. 3.1, 4.1, D
[45] Jianhao Huang, Zixuan Wang, and Jason D. Lee. Transformers learn to implement multi-step gradient
descent with chain of thought. CoRR, abs/2502.21212, 2025. 1, 2.2, 5
[46] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang,
Bowen Yu, Kai Dang, An Yang, Rui Men, Fei Huang, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, and
Junyang Lin. Qwen2.5-coder technical report. CoRR, abs/2409.12186, 2024. 3.4
[47] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-
Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of
large language models for code. CoRR, abs/2403.07974, 2024. 3.1, C.3
[48] Zhengbao Jiang, Frank F. Xu, Jun Araki, and Graham Neubig. How can we know what language models
know. Trans. Assoc. Comput. Linguistics, 8:423–438, 2020. 1
[49] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez,
Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with
pagedattention. In SOSP, pages 611–626. ACM, 2023. C.2
18

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
[50] Bespoke Labs. Bespoke-stratos: The unreasonable effectiveness of reasoning distillation, 2025. Accessed:
2025-01-22. 4
[51] Qingfeng Lan, A. Rupam Mahmood, Shuicheng Yan, and Zhongwen Xu. Learning to optimize for
reinforcement learning. RLJ, 2:481–497, 2024. 3.2
[52] Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, and Stefano Soatto.
Meta-learning with
differentiable convex optimization. In CVPR, pages 10657–10665. Computer Vision Foundation / IEEE,
2019. 1
[53] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of
neural nets. In NeurIPS, pages 6391–6401, 2018. 2, 4
[54] Ke Li and Jitendra Malik. Learning to optimize. In ICLR (Poster). OpenReview.net, 2017. 6
[55] Ke Li and Jitendra Malik. Learning to optimize neural nets. CoRR, abs/1703.00441, 2017. 6
[56] Xinzhe Li. A survey on LLM test-time compute via search: Tasks, LLM profiling, search algorithms, and
relevant frameworks. CoRR, abs/2501.10069, 2025. 1
[57] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike,
John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In ICLR. OpenReview.net,
2024. 3.1, C.3
[58] Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers learn
shortcuts to automata. In ICLR. OpenReview.net, 2023. 1, 5
[59] Chenghao Liu, Zhihao Wang, Doyen Sahoo, Yuan Fang, Kun Zhang, and Steven C. H. Hoi. Adaptive
task sampling for meta-learning. In ECCV, volume 12363 of Lecture Notes in Computer Science, pages
752–769. Springer, 2020. 1, 4.3
[60] Junnan Liu, Hongwei Liu, Linchen Xiao, Ziyi Wang, Kuikun Liu, Songyang Gao, Wenwei Zhang, Songyang
Zhang, and Kai Chen. Are your llms capable of stable reasoning? CoRR, abs/2412.13147, 2024. 3.1,
4.1, C.3
[61] Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning models
can be effective without thinking. CoRR, abs/2504.09858, 2025. 3.3, 4.2
[62] William Merrill, Ashish Sabharwal, and Noah A. Smith. Saturated transformers are constant-depth
threshold circuits. Trans. Assoc. Comput. Linguistics, 10:843–856, 2022. 1, 5
[63] Mansfield Merriman. A List of Writings Relating to the Method of Least Squares: With Historical and
Critical Notes, volume 4. Academy, 1877. B
[64] Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. Metaicl: Learning to learn in
context. In NAACL-HLT, pages 2791–2809. Association for Computational Linguistics, 2022. 5
[65] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke
Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? In EMNLP,
pages 11048–11064. Association for Computational Linguistics, 2022. 5
[66] Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang,
Xiaoxue Cheng, Huatong Song, Wayne Xin Zhao, Zheng Liu, Zhongyuan Wang, and Ji-Rong Wen.
Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. CoRR,
abs/2412.09413, 2024. 1
[67] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra,
and Martin A. Riedmiller. Playing atari with deep reinforcement learning. CoRR, abs/1312.5602, 2013.
2.3
[68] Eliakim H Moore. On the reciprocal of the general algebraic matrix. Bull. Am. Math. Soc., 26:294–295,
1920. B
19

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
[69] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke
Zettlemoyer, Percy Liang, Emmanuel J. Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling.
CoRR, abs/2501.19393, 2025. 4, 3.3
[70] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben
Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-
Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse,
Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context
learning and induction heads. CoRR, abs/2209.11895, 2022. 5
[71] OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. 1
[72] OpenAI.
Learning
to
reason
with
llms.
https://openai.com/index/
learning-to-reason-with-llms/, 2024. Accessed: 2024-09-12. 1, 2.2, 5
[73] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke
Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe.
Training language models to follow instructions with human feedback. In NeurIPS, 2022. 2.3, 2.4
[74] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang,
Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie
Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In
NeurIPS, pages 8024–8035, 2019. C.2
[75] Roger Penrose. A generalized inverse for matrices. In Math. Proc. Cambridge Philos. Soc., volume 51,
pages 406–413. Cambridge University Press, 1955. B
[76] Isabeau Prémont-Schwarz, Jaroslav Vitku, and Jan Feyereisl. A simple guard for learned optimizers. In
ICML, volume 162 of Proceedings of Machine Learning Research, pages 17910–17925. PMLR, 2022. 3.2
[77] Yiwei Qin, Xuefeng Li, Haoyang Zou, Yixiu Liu, Shijie Xia, Zhen Huang, Yixin Ye, Weizhe Yuan, Hector
Liu, Yuanzhi Li, and Pengfei Liu. O1 replication journey: A strategic progress report - part 1. CoRR,
abs/2410.18982, 2024. 1
[78] Xiaoye Qu, Yafu Li, Zhaochen Su, Weigao Sun, Jianhao Yan, Dongrui Liu, Ganqu Cui, Daizong Liu,
Shuxian Liang, Junxian He, Peng Li, Wei Wei, Jing Shao, Chaochao Lu, Yue Zhang, Xian-Sheng Hua,
Bowen Zhou, and Yu Cheng. A survey of efficient reasoning for large reasoning models: Language,
multimodality, and beyond. CoRR, abs/2503.21614, 2025. 3.3
[79] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understand-
ing by generative pre-training. OpenAI, 2018. 2.1, 2.3, 1
[80] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea
Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS,
2023. 1, 2.4
[81] Aravind Rajeswaran, Chelsea Finn, Sham M. Kakade, and Sergey Levine. Meta-learning with implicit
gradients. In NeurIPS, pages 113–124, 2019. 5
[82] Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In ICLR. OpenRe-
view.net, 2017. 1, 2
[83] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani,
Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. CoRR,
abs/2311.12022, 2023. 3.1, C.3
[84] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning.
In NAACL-HLT, pages 2655–2671. Association for Computational Linguistics, 2022. 5
20

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
[85] Adam Santoro, Sergey Bartunov, Matthew M. Botvinick, Daan Wierstra, and Timothy P. Lillicrap.
One-shot learning with memory-augmented neural networks. CoRR, abs/1605.06065, 2016. 5
[86] Jürgen Schmidhuber. Evolutionary principles in self-referential learning, or on learning how to learn:
The meta-meta-. hook. Master’s thesis, Technical University of Munich, Germany, 1987. 1, 2
[87] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms. CoRR, abs/1707.06347, 2017. 2.3, 1, 2.4, 4.1
[88] Darsh J Shah, Peter Rushton, Somanshu Singla, Mohit Parmar, Kurt Smith, Yash Vanjani, Ashish Vaswani,
Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, et al. Rethinking reflection in pre-training. CoRR,
abs/2504.04022, 2025. 1
[89] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu,
and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models.
CoRR, abs/2402.03300, 2024. 2.4, 3.1
[90] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin
Lin, and Chuan Wu. Hybridflow: A flexible and efficient RLHF framework. In EuroSys, pages 1279–1297.
ACM, 2025. C.2
[91] Wei Shi, Sihang Li, Tao Liang, Mingyang Wan, Gojun Ma, Xiang Wang, and Xiangnan He. Route sparse
autoencoder to interpret large language models. CoRR, abs/2503.08200, 2025. 1
[92] Avi Singh, John D. Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J.
Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron T. Parisi, Abhishek Kumar, Alexander A. Alemi,
Alex Rizkowsky, Azade Nova, Ben Adlam, Bernd Bohnet, Gamaleldin Fathy Elsayed, Hanie Sedghi,
Igor Mordatch, Isabelle Simpson, Izzeddin Gur, Jasper Snoek, Jeffrey Pennington, Jiri Hron, Kathleen
Kenealy, Kevin Swersky, Kshiteej Mahajan, Laura Culp, Lechao Xiao, Maxwell L. Bileschi, Noah Constant,
Roman Novak, Rosanne Liu, Tris Warkentin, Yundi Qian, Yamini Bansal, Ethan Dyer, Behnam Neyshabur,
Jascha Sohl-Dickstein, and Noah Fiedel. Beyond human data: Scaling self-training for problem-solving
with language models. Trans. Mach. Learn. Res., 2024, 2024. 3.2
[93] Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. In NIPS,
pages 4077–4087, 2017. 5
[94] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In
NIPS, pages 2440–2448, 2015. 5
[95] Gokul Swamy, Sanjiban Choudhury, Wen Sun, Zhiwei Steven Wu, and J. Andrew Bagnell. All roads lead
to likelihood: The value of reinforcement learning in fine-tuning. CoRR, abs/2503.01067, 2025. 2.4
[96] Hao Tang, Zechao Li, Zhimao Peng, and Jinhui Tang. Blockmix: Meta regularization and self-calibrated
inference for metric-based meta-learning. In ACM Multimedia, pages 610–618. ACM, 2020. 5
[97] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao,
Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe
Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu,
Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao,
Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang,
Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang,
Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying
Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He,
Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles,
Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying
Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu,
Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi
k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599, 2025. 1, 5
[98] NovaSky Team. Sky-t1: Train your own o1 preview model within $450. https://novasky-ai.
github.io/posts/sky-t1, 2025. Accessed: 2025-01-09. 4, 5
21

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
[99] OpenThoughts Team. Open Thoughts. https://open-thoughts.ai, 2025. 4, 5, D
[100] Qwen Team. Qwen3: Think deeper, act faster. https://qwenlm.github.io/blog/qwen3/, April
2025. 3.2, 3.3, 4.2, E
[101] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning. https://qwenlm.github.
io/blog/qwq-32b/, March 2025. 1, 2.2, 5, C.1
[102] Fengwei Teng, Zhaoyang Yu, Quan Shi, Jiayi Zhang, Chenglin Wu, and Yuyu Luo. Atom of thoughts for
markov LLM test-time scaling. CoRR, abs/2502.12018, 2025. 1
[103] Benjamin Thérien, Charles-Étienne Joseph, Boris Knyazev, Edouard Oyallon, Irina Rish, and Eugene
Belilovsky. 𝜇lo: Compute-efficient meta-generalization of learned optimizers. CoRR, abs/2406.00153,
2024. 3.2
[104] Sebastian Thrun and Lorien Y. Pratt. Learning to learn: Introduction and overview. In Learning to Learn,
pages 3–17. Springer, 1998. 5
[105] Eleni Triantafillou, Hugo Larochelle, Richard S. Zemel, and Vincent Dumoulin. Learning a universal
template for few-shot dataset generalization. In ICML, volume 139 of Proceedings of Machine Learning
Research, pages 10424–10433. PMLR, 2021. 1
[106] Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku Evci, Kelvin Xu, Ross Goroshin,
Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol, and Hugo Larochelle. Meta-dataset: A dataset
of datasets for learning to learn from few examples. In ICLR. OpenReview.net, 2020. 1
[107] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, pages 5998–6008, 2017. 2.1, 2.2, 5
[108] Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song,
Mingjie Zhan, and Hongsheng Li. Mathcoder: Seamless code integration in llms for enhanced mathe-
matical reasoning. In ICLR. OpenReview.net, 2024. 3.4
[109] Zhe Wang, Jake Grigsby, Arshdeep Sekhon, and Yanjun Qi. ST-MAML : A stochastic-task based method
for task-heterogeneous meta-learning. In UAI, volume 180 of Proceedings of Machine Learning Research,
pages 2066–2074. PMLR, 2022. 4.3
[110] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le,
and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS,
2022. 1, 2.2, 5
[111] Gail Weiss, Yoav Goldberg, and Eran Yahav. Thinking like transformers. In ICML, volume 139 of
Proceedings of Machine Learning Research, pages 11080–11090. PMLR, 2021. 5
[112] Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang,
Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng Jia, and Xiangzheng Zhang. Light-r1: Curriculum
sft, DPO and RL for long COT from scratch and beyond. CoRR, abs/2503.10460, 2025. 4, 3.2, D
[113] Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. In ICLR, 2015. 5
[114] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric
Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara
Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin
Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In EMNLP
(Demos), pages 38–45. Association for Computational Linguistics, 2020. C.2
[115] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context
learning as implicit bayesian inference. In ICLR. OpenReview.net, 2022. 5
[116] Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang.
Iterative preference learning from human feedback: Bridging theory and practice for RLHF under
kl-constraint. In ICML. OpenReview.net, 2024. 3.2
22

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
[117] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng
Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi
Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng
Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren,
Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu.
Qwen2.5 technical report. CoRR, abs/2412.15115, 2024. 1, 3.1
[118] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu,
Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru
Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement.
CoRR, abs/2409.12122, 2024. 3.1, 4.1, C.2
[119] Wenkai Yang, Shuming Ma, Yankai Lin, and Furu Wei. Towards thinking-optimal scaling of test-time
compute for LLM reasoning. CoRR, abs/2502.18080, 2025. 3.3
[120] Huaxiu Yao, Yu Wang, Ying Wei, Peilin Zhao, Mehrdad Mahdavi, Defu Lian, and Chelsea Finn. Meta-
learning with an adaptive task scheduler. In NeurIPS, pages 7497–7509, 2021. 4.3
[121] Shunyu Yao, Binghui Peng, Christos H. Papadimitriou, and Karthik Narasimhan. Self-attention net-
works can process bounded hierarchical languages. In ACL/IJCNLP, pages 3770–3785. Association for
Computational Linguistics, 2021. 5
[122] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan.
Tree of thoughts: Deliberate problem solving with large language models. In NeurIPS, 2023. 5
[123] Feiyang Ye, Baijiong Lin, Zhixiong Yue, Pengxin Guo, Qiao Xiao, and Yu Zhang. Multi-objective meta
learning. In NeurIPS, pages 21338–21351, 2021. 5
[124] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. LIMO: less is more for
reasoning. CoRR, abs/2502.03387, 2025. 4, 5
[125] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu,
Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang,
Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli
Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin
Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: an open-source LLM reinforcement learning
system at scale. CoRR, abs/2503.14476, 2025. 4, D
[126] Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou. Scaling
relationship on learning mathematical reasoning with large language models. CoRR, abs/2308.01825,
2023. 3.2
[127] Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang,
TianTian Fan, Zhengyin Du, Xiangpeng Wei, Xiangyu Yu, Gaohong Liu, Juncai Liu, Lingjun Liu, Haibin
Lin, Zhiqi Lin, Bole Ma, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Ru Zhang, Xin Liu, Mingxuan
Wang, Yonghui Wu, and Lin Yan. VAPO: efficient and reliable reinforcement learning for advanced
reasoning tasks. CoRR, abs/2504.05118, 2025. 4, 4.1, D
[128] Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv Kumar. Are
transformers universal approximators of sequence-to-sequence functions? In ICLR. OpenReview.net,
2020. 5
[129] Chulhee Yun, Yin-Wen Chang, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv
Kumar. O(n) connections are expressive enough: Universal approximability of sparse transformers. In
NeurIPS, 2020. 5
[130] Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, and He He. Reasoning models
know when they’re right: Probing hidden states for self-verification. CoRR, abs/2504.05419, 2025. 3.3
[131] Yangguang Zhang, Can Wang, Qihao Shi, Yan Feng, and Chun Chen. Adversarial gradient-based meta
learning with metric-based test. Knowl. Based Syst., 263:110312, 2023. 5
23

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
A. Notations
Table 4: Illustration of notations used in the paper.
ℳ
the large language model
𝜃
the parameters of the large language model
𝑞
the question
𝒬
the question set
𝑞𝑖
the 𝑖-th question in the question set
𝑞𝑗
𝑖
the 𝑗-th token of the 𝑖-th question
𝑎
the answer
𝒜
the answer set
𝑎𝑖
the 𝑖-th answer of the 𝑖-th question
𝑎𝑗
𝑖
the 𝑗-th token of the 𝑖-th answer
| · |
the length of tokens
𝐼
the instruction
d
the autoregressive decoding mechanism
Softmax
the softmax function
𝜎
the activation function
𝑡
the reasoning trajectory
𝒯
the set of reasoning trajectory
𝐸𝑜
the whole output token embedding of LLM
𝐸𝑥,:
the input token embedding of the sequence 𝑥
𝐸𝑥,𝑖
the input token embedding of the 𝑖-th token in the sequence 𝑥
Softmax(·)[𝑥]
the value of softmax vector in the entry corresponding to 𝑥
𝑝(·)
the probability distribution of one token sequence determined by
the LLM
∆ℳ𝜃(·)
the variation of the parameter 𝜃corresponding to the inputs
𝜃′
𝑖
the 𝑖-th step updated parameters
𝜃′
𝑡
the updated parameters 𝜃corresponding to the reasoning trajectory
𝑡
[︃
𝑥
𝑦
]︃
the concatenation of 𝑥and 𝑦
𝑊𝑘, 𝑊𝑞, 𝑊𝑣
the parameters in self-attention layer
𝑊1, 𝑊2, 𝑏1, 𝑏2
the parameters in feed-forward network
ℒ𝑞
the loss corresponding to the question
B. Proof of Proposition 2.1
Proof. Recall that our objective is to determine the set {𝑊′
𝑞, 𝑊′
𝑘, 𝑊′
𝑣, 𝑊′
1, 𝑊′
2, 𝑏′
1, 𝑏′
2} such that:
𝑊𝑇
2
(︃
𝜎
(︃
𝑊𝑇
1
(︃
Softmax
(︃
𝐸𝑡,0𝑊𝑞𝑊𝑇
𝑘
[︂𝐸𝑙,:
𝐸𝑡,0
]︂𝑇)︃[︂𝐸𝑙,:
𝐸𝑡,0
]︂
𝑊𝑣
)︃
+ 𝑏1
)︃)︃
+ 𝑏2 =
𝑊′𝑇
2
(︀
𝜎
(︀
𝑊′𝑇
1
(︀
Softmax
(︀
𝐸𝑙,−1:𝑊′
𝑞𝑊′𝑇
𝑘𝐸𝑇
𝑙,:
)︀
𝐸𝑙,:𝑊′
𝑣
)︀
+ 𝑏′
1
)︀)︀
+ 𝑏′
2,
(9)
where 𝐸𝑙,: ∈R|𝑙|×𝑑, 𝐸𝑙,−1, 𝐸𝑡,0 ∈R1×𝑑, 𝑊𝑞, 𝑊′
𝑞, 𝑊𝑘, 𝑊′
𝑘, 𝑊𝑣, 𝑊′
𝑣∈R𝑑×𝑑, and 𝑊1, 𝑊′
1, 𝑊2, 𝑊′
2 ∈R𝑑×𝑑.
24

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
We might as well let 𝑊′
1, 𝑊′
2, 𝑏′
1, 𝑏′
2 equal to 𝑊1, 𝑊2, 𝑏1, 𝑏2 (❶), respectively, as follows:
𝑊′
1 = 𝑊1,
𝑊′
2 = 𝑊2,
𝑏′
1 = 𝑏1,
𝑏′
2 = 𝑏2.
(10)
Then, we only need to establish the following equality:
Softmax
(︃
𝐸𝑡,0𝑊𝑞𝑊𝑇
𝑘
[︂𝐸𝑙,:
𝐸𝑡,0
]︂𝑇)︃[︂𝐸𝑙,:
𝐸𝑡,0
]︂
𝑊𝑣=
Softmax
(︀
𝐸𝑙,−1:𝑊′
𝑞𝑊′𝑇
𝑘𝐸𝑇
𝑙,:
)︀
𝐸𝑙,:𝑊′
𝑣.
(11)
For simplicity, we refine Equation (11) into several parts:
𝑄= 𝐸𝑡,0𝑊𝑞∈R1×𝑑,
𝐾𝑇= 𝑊𝑇
𝑘[𝐸𝑙,:, 𝐸𝑡,0]𝑇∈R(|𝑙|+1)×𝑑,
𝑉= [𝐸𝑙,:, 𝐸𝑡,0] 𝑊𝑣∈R(|𝑙|+1)×𝑑,
𝑄′ = 𝐸𝑙,−1:𝑊′
𝑞∈R1×𝑑,
𝐾′𝑇= 𝑊′𝑇
𝑘𝐸𝑇
𝑙,: ∈R|𝑙|×𝑑,
𝑉′ = 𝐸𝑙,:𝑊′
𝑣∈R|𝑙|×𝑑,
(12)
where 𝑑is the dimension size of embeddings.
Initially, considering the matrices 𝑄, we will demonstrate the existence of a linear transformation matrix
𝑃∈R𝑑×𝑑such that:
𝐸𝑡,0𝑃= 𝐸𝑙,−1:.
(13)
To support this assertion, we reference Theorem B.1:
Theorem B.1. Let 𝑈and 𝑉be vector spaces, and let {𝑏1, 𝑏2, . . . , 𝑏𝑛} denote a basis of 𝑈. For 𝑛vectors 𝑣𝑖∈𝑉,
there exists a linear transformation 𝑇: 𝑈→𝑉such that 𝑇(𝑏𝑖) = 𝑣𝑖for each 𝑖= 1, 2, . . . , 𝑛.
Proof. We begin by defining a linear transformation 𝑇: 𝑈→𝑉. Let 𝑢be a vector in 𝑈, expressed as
𝑢= 𝑢1𝑏1+𝑢2𝑏2+· · ·+𝑢𝑛𝑏𝑛, where the set {𝑏1, 𝑏2, . . . , 𝑏𝑛} constitutes a basis and the coefficients 𝑢1, 𝑢2, . . . , 𝑢𝑛
are determined by 𝑢. The linear transformation 𝑇is constructed as follows:
𝑇(𝑢) = 𝑢1𝑣1 + 𝑢2𝑣2 + · · · + 𝑢𝑛𝑣𝑛.
(14)
It is evident that this transformation 𝑇satisfies 𝑇(𝑏𝑖) = 𝑣𝑖for each index 𝑖.
Based on Theorem B.1, if we define a basis which involves 𝐸𝑡,0 (e.g., {𝐸𝑡,0, 0, . . . , 0}) and construct a
corresponding vector space, we can derive a linear transformation matrix 𝑃such that Equation (13) holds, with
𝑃being solely dependent on 𝐸𝑡,0. Therefore, if we let 𝑊′
𝑞= 𝑃𝑊𝑞(❷), then we have 𝑄= 𝑄′. Consequently,
we simplify Equation (11) to:
Softmax
(︀
𝑄𝐾𝑇)︀
𝑉= Softmax
(︀
𝑄𝐾′𝑇)︀
𝑉′
(15)
Now considering the matrix
[︂𝐸𝑙,:
𝐸𝑡,0
]︂
∈R(|𝑙|+1)×𝑑and 𝐸𝑙,: ∈R|𝑙|×𝑑, we can consistently identify a vector
𝐶∈R1×|𝑙| such that:
𝐸𝑡,0 ≈𝐶𝐸𝑙,:.
(16)
We examine the existence of 𝐶in two cases: 1) if 𝐸𝑡,0 lies within the span of the row vectors of 𝐸𝑙,;,
then 𝐶obviously exists; 2) if 𝐸𝑡,0 does not lie within the span of the row vectors of 𝐸𝑙,;, an approximate
25

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
solution for 𝐶can be derived using various methods, such as the least squares method [63]. Then, let
𝑀=
[︀
𝐼𝑙, 𝐶𝑇]︀
∈R|𝑙|×(|𝑙|+1), it follows that:
[︂𝐸𝑙,:
𝐸𝑡,0
]︂
≈𝑀𝑇𝐸𝑙,:.
(17)
We can express this relationship mathematically as follows:
Softmax
(︀
𝑄𝐾𝑇)︀
𝑉= Softmax
(︃
𝑄𝑊𝑇
𝑘
[︂𝐸𝑙,:
𝐸𝑡,0
]︂𝑇)︃[︂𝐸𝑙,:
𝐸𝑡,0
]︂
𝑊𝑣
≈Softmax
(︀
𝑄𝑊𝑇
𝑘𝐸𝑇
𝑙,:𝑀
)︀
𝑀𝑇𝐸𝑙,:𝑊𝑣
⇒Softmax
(︀
𝑄𝐾′𝑇)︀
𝑉′
= Softmax
(︀
𝑄𝑊′𝑇
𝑘𝐸𝑇
𝑙,:
)︀
𝐸𝑙,:𝑊′
𝑣.
(18)
Thus, we obtain (❸):
𝑊′
𝑘= 𝐸†
𝑙,:𝑀𝑇𝐸𝑙,;𝑊𝑘,
𝑊′
𝑣= 𝐸†
𝑙,:𝑀𝑇𝐸𝑙,;𝑊𝑣.
(19)
This construction ensures the validity of Equation (11). In this context, 𝐸†
𝑙,: indicates the Moore–Penrose
pseudoinverse [12, 68, 75] of 𝐸𝑙,:.
Building upon the previous discussions (❶,❷,❸), we demonstrate the existence of a parameter set:
{𝑊′
𝑞, 𝑊′
𝑘, 𝑊′
𝑣, 𝑊′
1, 𝑊′
2, 𝑏′
1, 𝑏′
2}
such that:
𝑊𝑇
2
(︃
𝜎
(︃
𝑊𝑇
1
(︃
Softmax
(︃
𝐸𝑡,0𝑊𝑞𝑊𝑇
𝑘
[︂𝐸𝑙,:
𝐸𝑡,0
]︂𝑇)︃[︂𝐸𝑙,:
𝐸𝑡,0
]︂
𝑊𝑣
)︃
+ 𝑏1
)︃)︃
+ 𝑏2 =
𝑊′𝑇
2
(︀
𝜎
(︀
𝑊′𝑇
1
(︀
Softmax
(︀
𝐸𝑙,−1:𝑊′
𝑞𝑊′𝑇
𝑘𝐸𝑇
𝑙,:
)︀
𝐸𝑙,:𝑊′
𝑣
)︀
+ 𝑏′
1
)︀)︀
+ 𝑏′
2,
(20)
which proves the Proposition 2.1. And this parameter set may not be the only viable option. For example,
according to the universal approximation theorem [24, 41], a feed-forward network can be utilized to address
differences in attention computations and provide a greater degree of freedom for 𝑊′
𝑞, 𝑊′
𝑘, and 𝑊′
𝑣.
C. Implementation Details of Experiments
C.1. Implementation Details of Visualization of Pseudo Gradient Update
Data Preparation. We select four questions from AIME2024 as follows:
Details of 𝑞0
Question
Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she
walks at a constant speed of 𝑠kilometers per hour, the walk takes her 4 hours, including 𝑡minutes
spent in the coffee shop. When she walks 𝑠+ 2 kilometers per hour, the walk takes her 2 hours and 24
minutes, including 𝑡minutes spent in the coffee shop. Suppose Aya walks at 𝑠+ 1
2 kilometers per hour.
Find the number of minutes the walk takes her, including the 𝑡minutes spent in the coffee shop.
Answer
204
26

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
Details of 𝑞1
Question
There exist real numbers 𝑥and 𝑦, both greater than 1, such that log𝑥(𝑦𝑥) = log𝑦
(︀
𝑥4𝑦)︀
= 10. Find 𝑥𝑦.
Answer
025
Details of 𝑞2
Question
Find the largest possible real part of
(75 + 117𝑖)𝑧+ 96 + 144𝑖
𝑧
where 𝑧is a complex number with |𝑧| = 4.
Answer
540
Details of 𝑞3
Question
Let △𝐴𝐵𝐶have circumcenter 𝑂and incenter 𝐼with 𝐼𝐴⊥𝑂𝐼, circumradius 13, and inradius 6. Find
𝐴𝐵· 𝐴𝐶.
Answer
468
Visualization of Pseudo Gradient Update. We leverage QwQ-32B [101] to generate trajectories for these four
questions. Then for each trajectory, we calculate the negative log-probability of Final Answer∖n∖boxed..answer..
at each position. Algorithm 3 outlines the overall process.
Algorithm 3: Computation of Empirical Examples of Pseudo Gradient Update
Input: ℳ: QwQ-32B, 𝐼: instruction, 𝑞: question from AIME2024, 𝑡: the trajectory generated by
QwQ-32B, 𝑎: the answer sequence, i.e., Final Answer∖n∖boxed...answer..., 𝑠: step
size.
1 for i ∈[0, |𝑡|) do
2
Obtain input by 𝐼⊕𝑞⊕𝑡:𝑖⊕𝑎;
3
Feed input to 𝑀and get logits 𝑙𝑎of answer sequence ;
4
Compute the negative log-probability using 𝑙𝑎;
Visualization of Landscape.
We refer to the methodology proposed by Li et al. [53]. Assuming the set
parameters of QwQ-32B is denoted by {𝜃𝑘} (excluding the embedding matrix), we randomly select two vectors,
{𝜃1,𝑘} and {𝜃2,𝑘}, for each parameter. We then edit the parameters by adding 𝛼1𝜃1,𝑘+ 𝛼2𝜃2,𝑘and compute
the negative log-probability given only the instruction and question to form the point set {(𝛼1, 𝛼2,̂︀ℒ𝛼1,𝛼2)}.
Finally, we visualize this point set to reveal the landscape. The overall process is summarized as Algorithm 4.
Project the Pseudo Gradient Update to Landscape. To project the trajectory of the pseudo-gradient update
onto the landscape, we fix one direction corresponding to the time dimension and identify the closest contour
to the correspondinĝ︀ℒto determine another direction.
27

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
Algorithm 4: Computation of Landscape
Input: ℳ: QwQ-32B, 𝐼: instruction, 𝑞: question from AIME2024, 𝑎: the answer sequence, i.e.,
Final Answer∖n∖boxed..answer...
1 Obtain random vectors {𝜃1,𝑘}, {𝜃2,𝑘} for each parameter 𝜃𝑘of ℳ;
2 for i ∈[−1, 1, 𝑠] do
3
for j ∈[−1, 1, 𝑠] do
4
Get edited parameters {𝜃′
𝑘} by adding 𝑖· 𝜃1,𝑘+ 𝑗· 𝜃2,𝑘;
5
Obtain input by 𝐼⊕𝑞⊕𝑎;
6
Feed input to 𝑀and get logits 𝑙𝑎of answer sequence ;
7
Compute the negative log-probability using 𝑙𝑎;
C.2. Training Details
Dataset Processing.
To maintain the validity and verifiability of the question set, we clean and filter the
original dataset. Initially, we exclude incomplete questions as well as those lacking answers. Subsequently, we
remove questions requiring reasoning with images or other external information. To further ensure verifiability,
we employ Math-Verify 2 to examine each question and exclude those that could not be verified. Finally, we
eliminate irrelevant characters, such as URLs and HTML tags, resulting in approximately 39k questions with
corresponding answers.
Training of SFT. We first synthesize training trajectories from Qwen2.5-Math-72B-Instruct [118] and DeepSeek-
R1-Distill-Qwen-14B [27]. From the entire question set, we sample 10k questions and use the sampling
parameters shown in Table 5 to generate reasoning trajectories with the prompt, Please solve the following
mathematical problem step by step and put your final answer in \boxed, resulting in 640k trajectories. We then
Table 5: Sampling parameters leveraging for reasoning trajectories synthesis.
Qwen2.5-Math-72B-Instruct
DeepSeek-R1-Distill-Qwen-14B
Temperature
0.7
0.7
Top-𝑝
1.0
1.0
Top-𝑘
50
50
Max Tokens
8192
36784
Rollout Number
64
64
filter out trajectories with incorrect answers, retaining approximately ∼470k for Qwen2.5-Math-72B-Instruct
and approximately ∼550k for DeepSeek-R1-Distill-Qwen-14B. During training, we utilize the parameters listed
in Table 6.
2https://github.com/huggingface/Math-Verify
28

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
Table 6: Training parameters for SFT.
Parameter
Max Response Length
18432
Train Batch Size
256
Learning Rate
1𝑒-5
Total Epochs
1
Table 7: Training parameters for GRPO.
Parameter
Max Prompt Length
1024
Max Response Length
16384
Rollout Temperature
1.0
Rollout Number
16
Train Batch Size
1024
Learning Rate
1𝑒-6
Total Epochs
1
System Prompt of GRPO
A conversation between a User and an Assistant. The User poses a question, and the Assistant provides
a solution. The Assistant’s response follows these structured steps:
1. Reasoning Process: The Assistant reflects on the problem using a reasoning process enclosed within
<think> and </think> tags.
2. Conclusion: The Assistant reaches a conclusion, which is enclosed within <conclusion> and
</conclusion> tags. The final answer is highlighted within \boxed...final answer....
3. Answer Format: The complete response should be formatted as:
<think>
...reasoning process...
</think>
<conclusion>
...conclusion...
The answer is \boxed...final answer...
</conclusion>
Training of GRPO. For the GRPO training, we use the complete question set and apply the parameters listed
in Table 7. We adhere to the DeepSeek-R1-style system prompt, as presented in the System Prompt of GRPO
box. And for the reward design, we assign the trajectory with the correct answer and correct format the score
1, the trajectory with the false answer and correct format the score 0.0, trajectory with the correct answer and
false format the score −0.5, and trajectory with the false answer and false format the score −1, formally:
𝑅(𝑦′, 𝑦) =
⎧
⎪
⎪
⎪
⎨
⎪
⎪
⎪
⎩
1
answer_match(𝑦′, 𝑦)
and
format_correct(𝑦′),
0
¬answer_match(𝑦′, 𝑦)
and
format_correct(𝑦′),
−0.5
answer_match(𝑦′, 𝑦)
and
¬format_correct(𝑦′),
−1
¬answer_match(𝑦′, 𝑦)
and
¬format_correct(𝑦′),
(21)
where 𝑦indicates the ground-truth and 𝑦′ indicates the trajectory. We employ the Math-Verify package to
ascertain the equivalence of 𝑦and 𝑦′.
Details of Hardware and Software. All the training tasks are conducted based on veRL [90], cooperated
with Pytorch [74] 2.6.0, Transformers [114] 4.51.3, vLLM [49] 0.8.4. We conduct all experiments on clusters
equipped with NVIDIA A800 GPUs and Intel(R) Xeon(R) Platinum 8336C CPUs.
C.3. Evaluation Details
Benchmarks. The following details describe our evaluation benchmarks:
• AIME24. AIME243 consists of 30 challenging questions from the 2024 American Invitational Mathematics
Examination (AIME).
3https://huggingface.co/datasets/AI-MO/aimo-validation-aime
29

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
• MATH500. The original MATH dataset [39] comprises 12, 500 problems from American high school
mathematics competitions. MATH500 [57], a widely used subset of its test split, includes only Level 5
questions in this study.
• LiveMathBench. LiveMathBench [60] is a continuously updated dataset of challenging mathematical
problems. We use the December 2024 hard split, comprising 45 questions in English and Chinese.
• GPQA. GPQA [83] dataset is a challenging, professional multiple-choice science question-answering dataset.
We use its diamond subset, comprising 198 questions.
• LiveCodeBench. LiveCodeBench [47] is a benchmark designed for a comprehensive and uncontaminated
evaluation of the code-related capabilities of LLMs. It incorporates questions from LeetCode, AtCoder, and
Codeforces.
Metrics.
We use Pass@𝑘and mG-Pass@𝑘[60] as evaluation metrics. We generate 𝑛responses for each
question and assume the number of correct responses is 𝑐. Then the metrics are computed as:
• Pass@𝑘.
Pass@𝑘= Equestions
[︃
1 −
(︀𝑛−𝑐
𝑘
)︀
(︀𝑛
𝑘
)︀
]︃
.
(22)
• mG-Pass@𝑘.
mG-Pass@𝑘= Equestions
⎡
⎣2
𝑘
𝑘
∑︁
𝑖=⌈𝑘/2⌉+1
𝑐
∑︁
𝑗=𝑖
(︀𝑐
𝑗
)︀
·
(︀𝑛−𝑐
𝑘−𝑗
)︀
(︀𝑛
𝑘
)︀
⎤
⎦.
(23)
D. More Discussions on Recent LLM Reasoning Progress
In this section, we focus on recent research developments and discuss the essential improvements they
implemented to enhance performance within our framework. We involve the following representative works:
OpenThoughts [99], Light-R1 [112], Open-Reason-Zero [44], DAPO [125], VAPO[127], GPG [21], Llama
Nemotron [8].
Data Filtering. Works such as Light-R1 [112] use strategies like diversity and difficulty filtering to obtain
high-quality data. From a meta-learning perspective, this approach can be seen as a sample mining strategy,
optimizing the distribution of training task sets to enhance the efficiency of model training.
Synthetic Data From Strong Reasoning LLMs. Works such as OpenThoughts [99] and Llama Nemotron [8]
utilize a more advanced reasoning LLM, such as DeepSeek-R1, to generate multiple trajectories for each training
question, resulting in training data for SFT. This approach effectively expands the size of the support set to
stabilize inner loop optimization, thereby achieving improved results. On the other hand, this is equivalent to
distilling the optimization path from the already trained model (strong reasoning LLMs) to the small model.
Clip Higher for Clipper Surrogate Loss of RL. DAPO [125] proposes using a higher clipping range to promote
exploration during the GRPO training process. Similarly, removing the KL penalty term and entropy loss
in GRPO can achieve the same effect. From an optimization perspective, these improvements expand the
exploration space of the optimization path, facilitating the model’s ability to explore extreme points. Increasing
the diversity of training data in supervised fine-tuning also contributes to this effect.
Dynamic Sampling During RL. Recent studies [21, 125] aim to balance the ratio of correct to incorrect
trajectories during rollout by employing dynamic sampling or introducing bias. This strategy equalizes the
positive and negative gradients in the inner loop, thereby alleviating model overfitting to a particular class.
Group-Sampling for PPO. In classic reinforcement learning methodologies, algorithms typically generate only
a single trajectory per problem instance. Recent advancements [44, 127] have introduced group sampling in
algorithms such as PPO, allowing the generation of multiple trajectories for each problem. From the perspective
of this study, this improvement corresponds to expanding the support set, thereby enhancing inner-loop
optimization.
30

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
E. More Experimental Results
QwQ’s Pseudo Gradient Update. Figure 12 illustrates more visualizations of the pseudo-gradient update of
QwQ.
QwQ3’s Pseudo Gradient Update in Thinking/Nothinking Mode. Figure 13 illustrates more visualizations
of the pseudo-gradient update of QwQ in thinking and nothinking modes.
Qwen3’s Pseudo Gradient Update. Following the methodology described in § 2.2 and Appendix C.1, we
monitor the pseudo-gradient update of Qwen3-32B [100] under thinking mode, as illustrated in Figure 14. We
similarly observe that the reasoning trajectories of Qwen3 exhibit a parameter update effect.
Qwen3’s Pseudo Gradient Update in Thinking/Nothinking Mode. Referring to the settings in § 3.3, we
examine the differences between thinking mode and no-thinking mode, as shown in Figure 15. It is evident
that, due to the specific optimization of Qwen3, its no-thinking token delimiter (i.e., </think>) demonstrates
a more pronounced gradient descent effect. The delimiter </think> enables the model to swiftly update to an
extreme point in the appropriate direction with a larger step size. However, this update is susceptible to falling
into local minima, which accounts for the performance gap between Qwen3’s no-thinking mode and thinking
mode.
Pseudo Gradient Update of False Reasoning Trajectories.
Figure 16 illustrates the curve of pseudo
gradient updates associated with incorrect reasoning trajectories. It is evident that the curve representing these
trajectories does not show a downward trend, underscoring the strong connection between reasoning paths
and optimization processes.
31

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
Figure 12a: More visualizations of the pseudo-gradient update of QwQ.
32

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
Figure 12b: More visualizations of the pseudo-gradient update of QwQ.
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
Figure 13a: More visualizations of QwQ’s pseudo-gradient update for both thinking and no-thinking modes.
33

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
Figure 13b: More visualizations of QwQ’s pseudo-gradient update for both thinking and no-thinking modes.
34

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
Figure 14a: Visualizations of the pseudo-gradient update of Qwen3.
35

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
Figure 14b: Visualizations of the pseudo-gradient update of Qwen3.
36

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
0.00
0.25
0.50
0.75
1.00
Trajectory0
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory1
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory2
w/ Thinking
w/o Thinking
0.00
0.25
0.50
0.75
1.00
Trajectory3
w/ Thinking
w/o Thinking
Normalized Token Indice
Figure 15: Visualizations of Qwen3’s pseudo-gradient update for both thinking and no-thinking modes.
37

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
Figure 16a: Illustration Qwen3’s pseudo-gradient update corresponding to false reasoning trajectories.
38

Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective
0.0
0.5
1.0
q0
0.0
0.5
1.0
q1
0.0
0.5
1.0
q2
0.0
0.5
1.0
q3
Normalized Token Index
Trajectory0
Trajectory1
Trajectory2
Trajectory3
Figure 16b: Illustration Qwen3’s pseudo-gradient update corresponding to false reasoning trajectories.
39
