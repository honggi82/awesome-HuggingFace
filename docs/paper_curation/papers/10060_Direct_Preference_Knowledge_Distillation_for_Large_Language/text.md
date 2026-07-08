# arXiv:2406.19774v2[cs.CL]7Apr2025

## Direct Preference Knowledge Distillation for Large Language Models

Yixing Li1*, Yuxian Gu2*, Li Dong3, Dequan Wang1, Yu Cheng4, Furu Wei 3 1Shanghai Jiao Tong University 2Tsinghua University 3Microsoft Research 4The Chinese University of Hong Kong {lyxing0, dequanwang}@sjtu.edu.cn guyx21@mails.tsinghua.edu.cn {lidong1,fuwei}@microsoft.com chengyu@cse.cuhk.edu.hk

### Abstract

In the field of large language models (LLMs), Knowledge Distillation (KD) is a critical technique for transferring capabilities from teacher models to student models. However, existing KD methods face limitations and challenges in the distillation of LLMs, including efficiency and insufficient measurement capabilities of traditional Kullback-Leibler (KL) divergence. It is shown that LLMs can serve as an implicit reward function, which we define as a supplement to KL divergence. In this work, we propose Direct Preference Knowledge Distillation (DPKD) for LLMs. DPKD utilizes distribution divergence to represent the preference loss and the implicit reward function. We re-formulate KD of LLMs into two stages: first optimizing an objective consisting of implicit reward and reverse KL divergence and then improving the preference probability of teacher outputs over student outputs. We conducted experiments on various datasets with LLM parameters ranging from 120M to 13B and demonstrate the broad applicability and effectiveness of our DPKD approach. Meanwhile, we prove the value and effectiveness of the introduced implicit reward and output preference in KD through experiments and theoretical analysis. The DPKD method outperforms the baseline method in both output response precision and exact match percentage.

### 1 Introduction

In the era of Large Language Models (LLMs), a series of models and techniques have demonstrated great capabilities (Zhang et al., 2022a; Chowdhery et al., 2023), and their powerful capabilities are usually ascribed to the increase in the size of the training data and the scale (Kaplan et al., 2020; Anil et al., 2023). However, the increase in the size of large models also brings expensive computing costs and difficulties (Hoffmann et al., 2022).

* Contribution during internship at Microsoft Research.

How to maintain performance and conduct efficient training while increasing the scale of language models has become an important issue. Knowledge distillation (KD; Hinton et al., 2015) trains a relatively compact student model by simulating the output distribution and behavior of the teacher model, which is an effective method under limited computing resources.

Recent work has recognized the shortcomings of KL divergence in traditional KD in which of LLMs (Gu et al., 2023), and explored other forms of KL divergence like reverse KL divergence. Work on KL divergence (Wu et al., 2024) proposed different KLD metrics for different stages of their respective shortcomings, and proposed improvement methods to expand KL divergence into flexible distance form. In fact, KL divergence is insufficient under the condition of a stronger teacher model (Huang et al., 2022; Cho and Hariharan, 2019; Son et al.,

- 2021), requiring the introduction of additional objectives and innovative knowledge distillation procedures. We start from another perspective and consider a novel optimization objective to compensate for the above problems while maintaining high efficiency.

In this work, we propose Direct Preference Knowledge Distillation (DPKD) for the knowledge distillation of LLMs. It is shown that LLMs can serve as an implicit reward function (Yuan et al., 2024; Rafailov et al., 2024b). Due to the deficiency of KL divergence (Gu et al., 2023; Huang et al.,

- 2022), we define implicit reward function as a supplement to the KL distance. We re-formulate the KD of white-box LLMs as follows, first maximize the optimization function consisting of implicit reward and reverse KL divergence, and then improve the preference probability of teacher outputs over student outputs. These settings compensate for the shortcomings of KL divergence while preserving high training efficiency. We derive the training objective to obtain a concise final form, and demon-

strate the significance of the reward and preference form we introduced in KD through theoretical derivation.

Finally, we conducted experiments on our DPKD method in the instruction tuning task. We utilized various families of LLMs including GPT2 (Radford et al., 2019) and OPT (Zhang et al., 2022b) with parameter sizes ranging from 120M to 13B and multiple datasets to validate our method. We evaluate the results with Rouge-L (Lin, 2004) metric. Our results show that the DPKD method outperforms the baseline and has advantages over a wide range of generation lengths. Additionally, we conducted experiments on the implicit reward function in the reformulation of distillation to verify its effectiveness, and experiments on the forms of preference to show the potential of preference modeling in KD and provide ideas for subsequent research. Our contributions are thus three-fold:

- • We propose a novel framework called Direct Preference Knowledge Distillation (DPKD), which provides a new perspective besides designing different KL divergences for knowledge distillation of large models.
- • We compare DPKD to five baseline methods across two commonly used LLMs (GPT-2 and OPT) and three datasets. We also provide a detailed derivation and theoretical analysis to demonstrate the effectiveness of the reward and preference model introduced in this paper.
- • We provide additional experiments on other preference objectives and observations on implicit reward. We illustrate the effectiveness and potential of our reformulation of the KD process, providing inspiration for subsequent work.

### 2 Methods

#### 2.1 Preliminaries

- 2.1.1 Sequence-Level Knowledge Distillation Sequence-level KD is often formulated as an optimization problem. Given a fixed teacher model with output distribution p and a student

model with output distribution qθ, the optimization goal is to minimize the distribution distance

between p and qθ. The distance is measured by Kullback-Leibler divergence (KLD). In the case of KD for LLMs, forward-KLD and reverseKLD (rKLD) are the most studied, respective defined as forward-KLD =Ex∼px,y∼p log p(y|x)

qθ(y|x) and

reverse-KLD =Ex∼px,y∼qθ log qθ(y|x)

p(y|x) . The reverse KL divergence is more suitable for KD while the distribution of LLMs is more complicated.

#### 2.1.2 Preference Model

Given two comparable objects or events, a common model for comparing the probabilities of their selection is the Bradley-Terry (BT; Bradley and Terry, 1952) model. We consider output y1 and y2, and the reward function that measures the gain of choosing an output is denoted r(y). BT model can be used to measure the probability that we choose y1 instead of y2:

exp (r (y1)) exp (r (y1)) + exp (r (y2))

Pr (y1 ≻ y2) =

= σ (r (y1) − r (y2)) ,

(1)

where σ is the function σ(x) := 1+exp(1 −x). 2.2 DPKD: Direct Preference Knowledge

Distillation

In the context of knowledge distillation of LLMs, the difference in distribution measured by KL divergence is often regarded as the only criterion for the distillation target. But KL divergence is insufficient. KL divergence measures the difference between two distributions, but it is not really a distance because it is asymmetric (Manning and Schutze, 1999). Researchers have improved the KL divergence by combining two symmetrical items (Jensen-Shannon divergence (Nielsen, 2021)). In the field of KD, researchers have explored the inadequacy of KL divergence in KD (Cho and Hariharan, 2019; Mirzadeh et al., 2020). Some works (Gu et al., 2023; Wu et al., 2024) hope to redesign KL divergence, and some works(Huang et al., 2022) add new objectives besides KL divergence. We conducted a toy experiment to further illustrate the inadequacy of KLD in Figure 2 and Section 4.3.

Based on the above considerations, we denote the implicit reward rp (y|x) as a supplement to the KL divergence in the optimization objective. We formulate the optimization goal as:

max

E [ rp(y|x) − β KLD (qθ(y|x))∥p(y|x)) ]. (2)

θ

We can get the optimal solution of Equation 2 through the following steps. Firstly transform Equation 2 as follows:

E log qθ(y | x)

1 β

min

r(x, y)

p(y | x) −

θ

(3)

E log qθ(y | x)

= min

q∗(y | x) − log Z(x) ,

θ

where q∗(y|x) := Z(1x)p(y|x)exp β 1r(x,y) , and Z (x) is the scaling function of the distribution, which is required to be independent of θ and y. It

is defined as Z(x) := y p(y|x)exp β 1r(x,y) . Following the work, we can derive the optimal solution of Equation 3:

qθ(y | x) = q∗(y | x). (4)

From Equation 4 we can obtain that:

r∗(x, y) = β log q∗(y | x) p(y | x)

+ β log Z(x). (5)

Given the same prompt x, the outputs of the student model and the teacher model are denoted as ys and yt respectively. The purpose of KD is to fit the distribution of the student model to teacher model, which can also be understood as we expect the student model to have a greater probability of outputting results similar to those of the teacher model. From the BT model, we can obtain the following probability:

p∗ (yt ≻ ys | x)

= σ β log qθ (yt | x) p (yt | x) − β log qθ (ys | x) p (ys | x)

(6)

.

The complete derivation of the above procedure is in the Appendix A. Since our goal is to maximize the probability of model outputing yt rather than ys, which is maxθ Elog p∗ (yt ≻ ys | x). It is formulated as:

−E log σ β log qθ (yt) p (yt) − β log qθ (ys) p (ys)

min

θ

. (7)

where qθ (yt) denotes qθ (yt | x) for short, and same for p and ys.

- 2.2.1 Learning Objective Following the framework derivation of DPKD, the loss is defined as:

L = −E log σ β log qθ (yt) p (yt) − β log qθ (ys) p (ys)

. (8)

It is shown by works (Meng et al., 2024; Gu et al., 2023) that models tend to introduce bias and

Algorithm 1 Direct Preference Knowledge Distillation (DPKD)

Input: Instruction tuning datasets D; Pre-training corpus Dp; Fine-tuned teacher model with distribution p; Initialized student model conducted SFT on Dp with output distribution qθ;

Output: Student model parameter conducted distillation training;

- 1: for epoch in epochs do
- 2: for batch in datasets D and Dp do
- 3: Compute responses from teacher and student model and obtain yt and ys;
- 4: Compute four log items log p(yt), log p(ys), log qθ(yt) and log qθ(ys);
- 5: Compute distill loss Lkd = − (x∼D); log σ |y β

t| log qθ(yt)

p(yt) − |yβ

s| log qθ(ys)

p(ys) ;

- 6: Compute language modeling loss Lpt = − d∼Dp log qθ(d);
- 7: Compute loss L = Lkd + λ · Lpt and gradient ∇θL;
- 8: Gradient Update θ(t+1) = θ(t) − α∇θL;
- 9: end for
- 10: end for
- 11: return Student model parameter θ.

produce short responses without length normalization. We add the length normalization factor to the distillation loss as follows:

log qθ (yt) p (yt) −

log qθ (ys) p (ys)

β |ys|

β |yt|

L = −E log σ

. (9)

Following work (Ouyang et al., 2022; Gu et al.,

- 2023), we add the language modeling loss in order to preserve performance on canonical NLP benchmarks. The complete algorithm process is shown in Algorithm 1. 3 Analysis 3.1 Gradient Derivation

Based on our target definition, we can derive the gradient of DPKD through the basic chain rule and perform analysis similar to work (Rafailov et al.,

- 2024b). From Equation 2.2.1, we can perform vari-

able substitution and define u := β log qθ(ys|x)

p(ys|x) − β log qθ(yt|x)

p(yt|x) , and the gradient of the loss can be expressed as

∇θL = −∇θE(x,yt,ys)∼D [log σ (u)]

σ′(u) σ(u) ∇θ(u) .

(10)

= −E(x,yt,ys)∼D

We can derive the complete form of gradient:

∇θL = − E βσ β log qθ (yt) p (yt) − β log qθ (ys)

p (ys) [∇θ log qθ (yt) − ∇θ log qθ (ys)]] .

(11)

The full derivation of the gradient is in the Appendix B.

- 3.2 Optimizing DPKD Is Optimizing The Q Function

Following work (Rafailov et al., 2024a), we conduct a theoretical analysis of DPKD and relate the reward function in KD we introduced to the Q function..We consider the sequence generation task, and the data set D = {(x,y)}N, where the prompt is denoted as x = {x0,··· ,xl−1}. We denote the vocabulary as V, each word x is included in the vocabulary V. Given input prompt x, the model output is y = {y0,··· ,ym−1}, where m is the maximum generated length. At each time step, the generated yt is conditionally generated based on the generated sequence {x,yt−1}.

From another perspective of Markov Decision Process (MDP), the process of sequence generation by the model is regarded as the generation process of a Markov decision chain. The state is denoted as st = {x0,...,xl−1,y0,...,yt−1}, and the action is yt ∈ V, which is chosen based on generated sequence st. The transition function from the current state to the next state is the LLMs we selected. We analyze the implicit reward function of DPKD from Q function, which is a perspective in reinforcement learning. The general represent of Q is:

Qθ(st, at) = E[Rt+1 + γRt+2 + ... | st, at]. (12)

Following the MDP perspective of sequence generation and according to the work , the fixed point solution of Equation 12 is:

∗(st,at)−V ∗(st))/β. (13)

qθ∗(at|st) = e(Q

Any valid Q function needs to satisfy the Belmman equation, from which we can write the current step Q∗ function for the optimal strategy (that is, the most optimal student model parameters) and the reward function r as:

Q∗(st, at) = r(st, at) + β log p(at|st) + V ∗(st+1), (14)

where V ∗(s) is defined to be zero if s is the end of sequence (EOS). Following work (Rafailov et al., 2024a), we can use the Q function instead of r and substitute it into the Bellman equation.

Now we can use the Q function to re-derive the DPKD method. By transforming the Equation 14 and summing over time t, and substitute Equation 13 into the result, we could obtain:

T−1

r(st, at) = V ∗(s0) +

t=0

T−1

β log qθ∗(at|st) p(at|st)

. (15)

t=0

According to the multi-step generalization of the Bradley-Terry preference model, namely the Plackett-Luce model, and substituting the Equation 15 into it, we can obtain:

p(τt ⪰ τs) = σ β log qθ (yt) p (yt) − β log qθ (ys) p (ys)

, (16)

where τ refers to the sequence trajectory generated by the model, and in the sequence generation task of the large model, it refers to the generated text. Replace τt and τs with yt and ys respectively, and the above formula is exactly the same as Equation 6. Thus we get the complete process of deriving the DPKD of this work from the Q function. Therefore, we can link the DPKD method of this work with the concept of Q function in reinforcement learning and Markov decision chain.

- 3.3 Your Language Model Is Secretly Implicit Reward Function During KD

We implicitly optimized a reward function during the KD process by redefining the formula and objectives of KD, and our experimental results demonstrate the significance of the reward function in the KD training process. From the perspective of theoretical analysis, the reward function serves not only as a weight term in the gradient of our DPKD but also as a crucial link between our DPKD method and the Q function in reinforcement learning.

Furthermore, the results presented in Section

- 4.4 indicate that various preference expressions related to reward functions yield significantly different results. Notably, some preference expressions demonstrate promising results, despite minimal parameter fine-tuning. This suggests that employing reward functions to articulate preferences in the knowledge distillation process is effective and highlights the potential for further exploration, which also offers valuable insights for future research.

### 4 Experiments

#### 4.1 Experiments Setup

Tasks and Metrics. We conduct experiments and analyses on the task of instruction tuning. The task of instruction tuning range from summarizing to completing requests, requiring the model to generate responses based on the provided instructions, prompts, and inputs. To ensure that the model produces high-quality results of sufficient length, we specifically select data with instructions and outputs exceeding 10 words in the dataset. The Rouge-L score (Lin, 2004) is employed to assess the quality of model generation, as Wang et al. (2022) has shown that Rouge-L is appropriate for large-scale instruction tuning evaluation.

Datasets. For the datasets, we use the following datasets: (1) Dolly (Conover et al., 2023) including ∼15k instruction/response generated in capability domains from the InstructGPT, (2) SelfInst (Wang et al., 2023) consisting of 252 expertwritten tasks and their instructions motivated by user-oriented applications, (3) SUPERNATURALINSTRUCTIONS (S-NI; Wang et al., 2022) containing over 1.5k tasks and covering 76 distinct task types. We filter the data whose length exceeds the model processing length and select the part with response length [11,∞] as training data to keep the same settings as the validation set and test machine.

Models. For teacher models, we select GPT-2 (Radford et al., 2019) (1.5B) and OPT (Zhang et al., 2022b) (13B). The student models are GPT-2 with 120M and 340M parameters and OPT with 1.3B parameters respectively. These models have been pre-trained on the Dp datasets.

Baselines. The baseline methods we selected include: (1) Supervised Fine-Tuning (SFT) directly fine-tune on labels of dataset; (2) KD (Hinton et al., 2015) is also called word-level KD, which uses the teacher’s output distributions as supervision;

- (3) Sequence-level KD (SeqKD; Kim and Rush, 2016; Zhou et al., 2024) directly distills the student model on the data generated by the teacher model;
- (4) MiniLLM (Gu et al., 2023) leverages reverse KL divergence to perform distillation on LLMs ;(5) Adaptive Kullback-Leiber (AKL; Wu et al., 2024) uses a combination of forward and reverse KL with dynamic weights to perform distillation.

Model #Param Method DollyEval SelfInst S-NI

1.5B Teacher 27.6 14.3 27.6

SFT 23.3 10.0 16.3 KD 22.8 10.8 13.4

GPT-2

SeqKD 22.7 10.1 16.4 MiniLLM 24.6 13.2 25.3

120M

AKL 23.9 - 19.2 DPKD (ours) 24.6 13.8 25.4

1.5B Teacher 27.6 14.3 27.6

SFT 25.5 13.0 25.1 KD 25.0 12.0 23.7

GPT-2

340M

SeqKD 25.3 12.6 22.9

MiniLLM 25.4 15.6 27.4 DPKD (ours) 26.0 14.3 27.6

13B Teacher 29.2 18.4 30.4

SFT 26.0 11.4 23.1 KD 25.4 12.2 21.9

OPT

1.3B

SeqKD 26.1 12.7 21.4

MiniLLM 26.7 14.8 28.6 DPKD (ours) 27.2 14.9 28.7

Table 1: Experimental results with GPT-2 and GPT model families. The teacher models of the GPT-2 and OPT are 1.5B and 13B respectively. The student model ranges from 120M to 1.3B. The scores are RougeL scores, and the experiments cover the DollyEval, SelfInst, and S-NI datasets. All experiments are conducted with the same seed. AKL results are taken from (Wu et al., 2024)

#### 4.2 Results

We conducted experiments based on Algorithm 1 and the experimental settings in Section 4.1, and the experimental results are presented in Table 1. In this section, we conduct a preliminary analysis of the results including the RougeL score, the GPT4 score, and an example of the implicit reward function introduced in this article.

From the experimental results, we could see that our method demonstrates great performance compared to the various baseline. We conduct KD experiments on datasets with different distributions and LLMs of different sizes, showing that our method is applicable to a wide range of models and data and retains good performance. In some specific datasets, the student model trained by our knowledge distillation method is even close to the performance of the teacher model.

We collected the output texts of DPKD and the baseline method of GPT-2 Base, and evaluated them together with the labels of the dataset by GPT4 to evaluate the quality of the instruction tracking generated text of each method in Figure 1. Although the output results of each method are still

[Figure 1]

- Figure 1: GPT-4 evaluation of different methods. Our DPKD method outperforms other baselines and is closest to the reference responses.

[Figure 2]

- Figure 2: Illustration of the relation of rKLD, implicit reward and Rouge-L. We construct differentiated models by adding random noise to the base model. Lighter colors indicate higher Rouge-L scores.

a certain distance away from the label, the generation results of the method proposed in this paper are closer to the label evaluations and surpass those of the baselines.

#### 4.3 Reward Observation

In this section, our analysis highlights the inadequacies of rKLD and emphasizes the significance of the reward function introduced in this paper. We conducted an experiment to further illustrate the deficiencies of rKLD discussed in Section 2.2 and the corresponding changes in reward and RougeL. As shown in Figure 2, we added random noise to the basic model and simultaneously calculated both the rKLD and the estimated implicit reward presented in this paper during inference. Notably, while the rKLD fluctuates at a relatively low value of 0.001, the Rouge-L score exhibits a fluctuation as high as 6.23.

Furthermore, we analyze the implicit reward function discussed in this paper and examine the role of the reward function introduced in KD from both theoretical and experimental perspectives. Ac-

[Figure 3]

Figure 3: We report the reverse KL divergence, and estimated implicit reward during training. The color of points represents the training epochs. The end of the training falls in the upper left corner, where the KL divergence is low and the reward is high.

cording to the gradient derivation presented in Section 3.1, the reward function acts as a weight in the gradient update process. When the estimate of the reward function indicates that the current output is biased towards an error, the weight value will increase, resulting in a larger gradient. Conversely, when the model’s performance improves and the output distributions of the student and teacher models become similar, the previously mentioned weight term will decrease, leading the model to converge.

We aim to analyze the trend of KL divergence and the reward function of DPKD. To achieve this, we utilize an unbiased approximation of the reward function, which is rˆ∗(x,y) = β log qθ(y|x)

p(y|x) . Based on this and the definition of reward estimation, we plot the reward function and KL divergence for the same model (in this case, we use the GPT-2 Base) at different training periods.

We illustrate the trend of reward during training and the reverse KL divergence distance between the student and teacher models across training epochs in Figure 3. From the visualization results, we observe that while some model checkpoints exhibit very close KL divergence values to those of the teacher model (all within the range of 0.04±0.02), the differences in reward values still lead to variations in the RougeL scores. From the curve trend in the figure, it can be seen that the optimal point of model training occurs at a left-point arc vertex, while the student model has the lowest KL divergence from the teacher and the reward function is at a higher value. The magnitude of the reward function may be related to the hyperparameter β used during training, which adjusts the relative propor-

[Figure 4]

[Figure 5]

[Figure 6]

Figure 4: Reward, KLD, and reverse KLD curves during the distillation process of GPT-2 Base. KLD and reverse KLD show similar trends.

tions of the KL divergence and the reward function.

- Figure 4 illustrates the changes in the implicit

reward function, KLD, and rKLD during training. We can see that the estimated implicit reward gradually increases within a specific range. The trends of KLD and rKLD are generally similar, but the exact values differ slightly, which is consistent with the conclusions of previous work (Gu et al., 2023; Wu et al., 2024).

#### 4.4 Variants of Preference Objective

In the subject of fine-tuning language models with human feedback, there are many optimization methods based on preference. These various optimization techniques are all based on the original work DPO, and modify aspects such as the preference format and normalization methods. In the context of knowledge distillation for large models, the function form derived in this paper represents a relatively basic form. So similarly, we also conducted preliminary experiments on different methods for calculating preferences, with the aim of inspiring future research. It is important to note that the experiments conducted in this chapter did not perform special parameter fine-tuning.

We experimented with three other forms of preference: IPO, CPO, SimPO. Their experiment results and definitions are shown in Table 2. From the experimental results, we could see that although other forms of preference do not surpass our method, they demonstrate a certain level of effectiveness. (Pal et al., 2024) illustrated that in the context of learning from preference data, the

#### Preference Form RougeL

IPO (Azar et al., 2024) 18.78 CPO (Xu et al., 2024) 24.31 SimPO (Meng et al., 2024) 24.46 DPKD (Ours) 24.62

- Table 2: Variants of preference objective. The definitions of each preference are as follows: (1) SimPO = −Elog σ |y β

t| log qθ (yt|x) − |yβ

s| log qθ (ys|x) − γ ;

- (2) CPO = −E log σ β log qθ(yt|x) qθ(yt|x) − log qθ (yt|x) ;

- (3) IPO = E log qθ(yt|x)

p(yt|x) − log qθ(ys|x)

p(ys|x) − 21τ

2

.

Setting RougeL ↑ rKLD ↓ Reward ↑

DPKD 24.62 0.07 0.97 w/o LM Loss 23.47 0.24 0.93 w/o Length Norm. 22.79 0.23 0.90

- Table 3: Ablation studies of DPKD. “rKLD” (reverse KL divergence) is the minimum value of the student distance from the teacher model during training. “Reward” is the maximum value. LM loss represents the language modeling loss introduced in Section 2.2.1

form of the preference also affects how the relative scores of yt and ys grow during the training process. Our findings suggest that these problems and solutions are also applicable to the field of knowledge distillation.

#### 4.5 Ablation Studies

We conduct ablation experiments on the experimental settings and measure the scores on the GPT2 Base model and DollyEval. We evaluate the results of omitting length normalization and language modeling loss. In Table 3, KLD and rKLD represent the minimum values of KL divergence and reverse KL divergence from the teacher model during training, respectively. To ensure consistent distribution conditions, the distribution difference is calculated based on the output of the first token. The reward is the maximum value during training and is calculated using the same method described above. From the results, we can see that the components introduced in the experiment have a positive effect on the results, especially the improvement of implicit reward. The comprehensive method performs well with the teacher model under both reverse KL and forward KL metrics, achieving the highest implicit reward and score.

[Figure 7]

- Figure 5: RougeL score with different ranges of generation lengths. In the case of different ranges of reference label lengths, DPKD scores higher than the baseline. In particular, DPKD stands out when the golden response length is in the middle range. The raw RougeL scores of each method are provided in the Appendix C.

Method (0,30) (30,70) (70,∞)

SeqKD 5.15 0.00 0.00 MiniLLM 5.30 0.93 0.63 DPKD 5.57 1.85 0.63

- Table 4: Exact match with different ranges of generation lengths. Our method outperforms the baselines in different ranges of reference lengths, and the results of other baselines tend to be 0 when the length of the golden response is longer.

#### 4.6 Results of Different Generation Lengths

In this section, we conduct experiments on subsets of the test set with varying response lengths to assess the effectiveness of our approach in scenarios with different output length requirements. We divide the DollyEval test set into three subsets according to the length of the golden response: (0,30), (30,70) and (70,∞), where 30 and 70 are the median and mean of the response lengths, respectively. We then evaluate the response results of various baseline methods and DPKD on these subsets.

The results of the RougeL and exact match are shown in Figure 5 and Table 4. Our DPKD method performs well on all subsets of response length splits. Notably, DPKD exceeds the baseline by the most in the results with response lengths in the middle. In the exact match results, the scores of many baseline methods drop to 0 when the golden response length is longer, while DPKD still performs well and exceeds the baseline.

### 5 Related Work

Knowledge Distillation (KD) is a widely used technique in model training and fine-tuning, and was first proposed in the work (Bucila et al., 2006), (Hinton et al., 2015) further enriched and conducted studies on KD. Researchers have explored the application of KD in text generation tasks (Song et al., 2020; Zhang et al., 2023; Jiao et al., 2020; Sun et al., 2019). The standard form of KD in NLP is to minimize the KLD between student model and teacher, including train on teacher-generated text (Kim and Rush, 2016; Taori et al., 2023) and teacher output at each step (Sanh et al., 2019). There are also many works (Agarwal et al., 2023; Passalis and Tefas, 2018; Wen et al., 2023) exploring the metric of distribution distance in KD. Works (Huang et al., 2022; Cho and Hariharan, 2019; Mirzadeh et al., 2020) proposed inadequacy of KLD when the student and teacher model sizes significantly differ.

KD for LLM has received attention in recent years (Zhang et al., 2022a; Touvron et al., 2023). KD in large models is usually divided into blackbox distillation (Taori et al., 2023; Chiang et al., 2023) and white-box distillation (Kim et al., 2024). The black-box KD takes the text generated by the teacher as knowledge and performs distillation, while the wite-box KD use the model intermediate layers or output logits distribution (Jiao et al., 2020; Wang et al., 2020). Recent works have recognized the inadequacy of KL divergence in knowledge distillation of LLMs (Gu et al., 2023) and applied reverse KL divergence to KD of LLMs. Concurrent work (Wu et al., 2024) illustrates the shortcomings of reverse KLD and KLD and designs a KL divergence that adaptively allocates weights.

### 6 Conclusion

In this work we propose a novel method for knowledge distillation of LLMs from the perspective of direct preference learning. We introduce implicit reward and output preference models in knowledge distillation of LLMs and re-formulate the goal of KD. We perform theoretical derivation to obtain a new distillation framework. Experiments show that this framework performs well and is applicable to a wide range of models and data. In addition, we conducts extensive studies on the reward function, preference formula form, showing its importance and providing insights for subsequent work.

### Limitations

We recognize several limitations of this work. Firstly, the form of preference will affect the situation of model learning yt and ys. In some cases, direct preference learning will cause the model to focus too much on the relative probability of the two, rather than on the absolute probability of yt that we expect to obtain. As mentioned in Section

- 4.4, there are many works that explore and improve different forms of preference. This paper conducts preliminary experiments on these forms of preference and shows that some of the methods may be effective. However, in the field of instruction fine-tuning involved in this paper, the above direct preference situation is not serious. However, the effectiveness of other methods in the preliminary experiments of this paper shows that there may be other problems caused by preferences in the field of LLM KD, and the performance can be further improved through design methods. Section 4.4 has introduced preliminary experiments, and we hope that the DPKD methods and this part of the experiment can provide inspiration for future work.

### References

Rishabh Agarwal, Nino Vieillard, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. 2023. Gkd: Generalized knowledge distillation for auto-regressive sequence models. arXiv preprint arXiv:2306.13649.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. arXiv preprint arXiv:2305.10403.

Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. 2024. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pages 4447–4455. PMLR.

Ralph Allan Bradley and Milton E. Terry. 1952. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39:324.

Cristian Bucila, Rich Caruana, and Alexandru Niculescu-Mizil. 2006. Model compression. In Knowledge Discovery and Data Mining.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion

Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing gpt-4 with 90%* chatgpt quality.

Jang Hyun Cho and Bharath Hariharan. 2019. On the efficacy of knowledge distillation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4794–4802.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113.

Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. 2023. Free dolly: Introducing the world’s first truly open instructiontuned llm.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang.

2023. Knowledge distillation of large language models. ArXiv, abs/2306.08543.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556.

Tao Huang, Shan You, Fei Wang, Chen Qian, and Chang Xu. 2022. Knowledge distillation from a stronger teacher. Advances in Neural Information Processing Systems, 35:33716–33727.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. 2020. TinyBERT: Distilling BERT for natural language understanding. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4163– 4174, Online. Association for Computational Linguistics.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Gyeongman Kim, Doohyuk Jang, and Eunho Yang. 2024. Promptkd: Distilling student-friendly knowledge for generative language models via prompt tuning. arXiv preprint arXiv:2402.12842.

Yoon Kim and Alexander M. Rush. 2016. Sequencelevel knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1317–1327, Austin, Texas. Association for Computational Linguistics.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Christopher Manning and Hinrich Schutze. 1999. Foundations of statistical natural language processing. MIT press.

Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734.

Seyed Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, and Hassan Ghasemzadeh. 2020. Improved knowledge distillation via teacher assistant. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 5191–5198.

Frank Nielsen. 2021. On a variational definition for the jensen-shannon symmetrization of distances based on the information radius. Entropy, 23(4):464.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. 2024. Smaug: Fixing failure modes of preference optimisation with dpo-positive. arXiv preprint arXiv:2402.13228.

Nikolaos Passalis and Anastasios Tefas. 2018. Learning deep representations with probabilistic knowledge transfer. In Proceedings of the European Conference on Computer Vision (ECCV), pages 268–284.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners.

Rafael Rafailov, Joey Hejna, Ryan Park, and Chelsea Finn. 2024a. From r to q: Your language model is secretly a q-function. arXiv preprint arXiv:2404.12358.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2024b. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. ArXiv, abs/1910.01108.

Wonchul Son, Jaemin Na, Junyong Choi, and Wonjun Hwang. 2021. Densely guided knowledge distillation using multiple teacher assistants. In Proceedings

of the IEEE/CVF International Conference on Computer Vision, pages 9395–9404.

Kaitao Song, Hao Sun, Xu Tan, Tao Qin, Jianfeng Lu, Hongzhi Liu, and Tie-Yan Liu. 2020. Lightpaff: A two-stage distillation framework for pre-training and fine-tuning. arXiv preprint arXiv:2004.12817.

Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. 2019. Patient knowledge distillation for bert model compression. In Conference on Empirical Methods in Natural Language Processing.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https:// github.com/tatsu-lab/stanford_alpaca.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. Minilm: deep selfattention distillation for task-agnostic compression of pre-trained transformers. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA. Curran Associates Inc.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13484–13508, Toronto, Canada. Association for Computational Linguistics.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. 2022. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5085–5109, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Yuqiao Wen, Zichao Li, Wenyu Du, and Lili Mou. 2023. f-divergence minimization for sequence-level knowledge distillation. arXiv preprint arXiv:2307.15190.

Taiqiang Wu, Chaofan Tao, Jiahao Wang, Zhe Zhao, and Ngai Wong. 2024. Rethinking kullback-leibler divergence in knowledge distillation for large language models. arXiv preprint arXiv:2404.02657.

Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. 2024. Contrastive preference optimization: Pushing the boundaries of llm performance in machine translation. arXiv preprint arXiv:2401.08417.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. 2024. Self-rewarding language models. arXiv preprint arXiv:2401.10020.

Rongzhi Zhang, Jiaming Shen, Tianqi Liu, Jialu Liu, Michael Bendersky, Marc Najork, and Chao Zhang. 2023. Do not blindly imitate the teacher: Using perturbed loss for knowledge distillation. arXiv preprint arXiv:2305.05010.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022a. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022b. Opt: Open pre-trained transformer language models. ArXiv, abs/2205.01068.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. 2024. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36.

### A Complete Derivation of DPKD Objective

The optimization goal of first step is defined as:

max

E [ rp(y|x) − β KLD (qθ(y|x))∥p(y|x)) ] (17)

θ

Transform Equation 17 as follows:

E r(x, y) − β log qθ(y | x)

max

p(y | x)

θ

E log qθ(y | x)

1 β

r(x, y)

= min

p(y | x) −

θ

E log qθ(y | x)

= min

Z(x)p(y | x) exp β 1 r(x, y) − log Z(x)

1

θ

E log qθ(y | x)

= min

q∗(y | x) − log Z(x)

θ

(18)

where q∗(y|x) := Z(1x)p(y|x)exp β 1r(x,y) , and Z (x) is the scaling function of the distribution, which is required to be independent of θ and y. It

is defined as Z(x) := y p(y|x)exp β 1r(x,y) .

Substituting into the optimization target, we can get the optimization target:

Ex Ey log qθ(y | x)

min

q∗(y | x) − log Z(x)

θ

= min

Ex [KLD (qθ(y | x)∥q∗(y | x)) − log Z(x)]

θ

(19)

Since Z (x) is a scaling function independent of θ, the minimum value from the optimization objective 19 is achieved by minimizing the first KL divergence term. We can derive the optimal solution of Equation 19:

qθ(y | x) = q∗(y | x) (20)

From Equation 20 we can obtain that:

r∗(x, y) = β log q∗(y | x) p(y | x)

+ β log Z(x) (21)

Given the same prompt x, the outputs of the student model and the teacher’s model are respectively denoted as ys and yt. The purpose of KD is to fit the distribution of the student model to teacher model, which can also be understood as we expect the student model to have a greater probability of outputting results similar to the teacher model. From the BT model, we can obtain the following probability:

p∗ (yt ≻ ys | x)

= σ β log qθ (yt | x) p (yt | x) − β log qθ (ys | x) p (ys | x)

(22)

Since our goal is to maximize the probability of model outputing yt rather than ys, which is:

max

E log p∗ (yt ≻ ys | x) (23)

θ

−E log σ β log qθ (yt) p (yt) − β log qθ (ys) p (ys)

min

(24)

θ

where qθ (yt) denotes qθ (yt | x) for short, and same for p and ys.

### B Complete Derivation of DPKD Gradient

Based on our target definition, we can derive the gradient of DPKD through the basic chain rule and perform analysis. From Equation 2.2.1, we can perform variable substitution and define u :=

β log qθ(ys|x)

p(ys|x) − β log qθ(yt|x)

p(yt|x) , and the gradient of the loss can be expressed as

∇θL = −∇θE(x,yt,ys)∼D [log σ (u)]

σ′(u) σ(u) ∇θ(u)

(25)

= −E(x,yt,ys)∼D

Define the variable u as follows:

u = β log qθ (ys | x) p (ys | x) − β log qθ (yt | x) p (yt | x)

(26)

Substitute and perform gradient deformation:

∇θLDPKD (θ) = −E(x,yt,ys)∼D

σ′(u) σ(u) ∇θ(u) (27)

Since σ′(x) = σ(x)(1 − σ(x)), we can derive the complete form of gradient:

∇θL = − E βσ β log qθ (yt) p (yt) − β log qθ (ys)

p (ys) [∇θ log qθ (yt) − ∇θ log qθ (ys)]]

(28)

For further analysis, we make a representable approximation of the reward function form in Equation 21

rˆθ(x, y) = β log qθ(y | x) p(y | x)

(29)

and substitute it into the Equation 28:

∇θL = − β E [σ (ˆrθ(x, yt) − rˆθ(x, ys)) [∇θ log qθ (yt) − ∇θ log qθ (ys)]]

(30)

### C Complete Result of Different Generation Lengths

Method (0, 30) (30, 70) (70, ∞) KD 28.04 21.65 15.07 SeqKD 29.32 24.67 16.16

SFT 30.10 25.28 16.09 DPKD 30.98 26.69 17.26

- Table 5: Exact value of RougeL with different ranges of generation lengths.

The complete result of Figure 5 is shown in Table

- 5. From the results, we can see that as the length of the golden response increases, the absolute value of RougeL decreases. This is because the difficulty of the corresponding task of the instruction increases. However, we can still see the advantage of DPKD, as well as the huge lead of DPKD in the middle range of length.

