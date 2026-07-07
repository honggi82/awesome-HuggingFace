# arXiv:2602.05261v1[cs.CL]5Feb2026

## Length-Unbiased Sequence Policy Optimization: Revealing and Controlling Response Length Variation in RLVR

Fanfan Liu ∗, Youyang Yin ∗, Peng Shi , Siqi Yang , Zhixiong Zeng , Haibo Qiu Meituan

### Abstract

Recent applications of Reinforcement Learning with Verifiable Rewards (RLVR) to Large Language Models (LLMs) and Vision-Language Models (VLMs) have demonstrated significant success in enhancing reasoning capabilities for complex tasks. During RLVR training, an increase in response length is often regarded as a key factor contributing to the growth of reasoning ability. However, the patterns of change in response length vary significantly across different RLVR algorithms during the training process. To provide a fundamental explanation for these variations, this paper conducts an in-depth analysis of the components of mainstream RLVR algorithms. We present a theoretical analysis of the factors influencing response length and validate our theory through extensive experimentation. Building upon these theoretical findings, we propose the Length-Unbiased Sequence Policy Optimization (LUSPO) algorithm. Specifically, we rectify the length bias inherent in Group Sequence Policy Optimization (GSPO), rendering its loss function unbiased with respect to response length and thereby resolving the issue of response length collapse. We conduct extensive experiments across mathematical reasoning benchmarks and multimodal reasoning scenarios, where LUSPO consistently achieves superior performance. Empirical results demonstrate that LUSPO represents a novel, state-of-the-art optimization strategy compared to existing methods such as GRPO and GSPO.

### 1 Introduction

Reinforcement Learning with Verifiable Rewards (RLVR) has become a cornerstone technique for advancing the capabilities of language models [DeepSeek, 2025]. Leveraging large-scale RLVR, these models are empowered to address complex tasks, including sophisticated mathematics and high-level programming, through extended and nuanced reasoning. Through iterative optimization and exposure to diverse reward signals, RLVR-trained language models develop the ability to generate longer, more coherent, and contextually relevant responses. This facilitates nuanced reasoning, allowing the models to break down complex problems into manageable steps and provide detailed, logical explanations. As a result, RLVR not only improves the accuracy and reliability of language model outputs, but also significantly expands their applicability across domains requiring sophisticated reasoning and decision-making.

Among contemporary RLVR strategies, GRPO [Shao et al., 2024] has emerged as a prominent method, demonstrating strong performance in DeepSeek-R1 [DeepSeek, 2025]. Nonetheless, GRPO’s improper utilization of importance sampling introduced instability in the training process for Mixtureof-Experts (MoE) architectures. GSPO [Zheng et al., 2025] subsequently resolved this limitation by employing sequence-level importance weighting [Zheng et al., 2023], thereby enhancing stability in MoE training.

Although GRPO has achieved notable success, averaging the contribution of all tokens within each trajectory in the GRPO objective leads to a length bias. This bias causes the model to give larger

∗Equal contribution

gradient updates to shorter responses, encouraging brevity in correct answers. On the other hand, for negative advantages (i.e., incorrect responses), longer outputs are penalized less, making the policy favor longer responses when the answer is incorrect.

GSPO exhibits the same issue as well, Moreover, the GSPO’s objective function further exacerbates this bias. Sequence-level clipping leads to a substantially greater number of tokens being truncated compared to token-level clipping. Additionally, the clip-higher mechanism [Yu et al., 2025] disproportionately removes negative sample tokens, resulting in a pronounced imbalance between positive and negative sample tokens. This disparity amplifies the response-level length bias inherent in the GSPO’s objective function, where positive samples incentivize the model to generate shorter outputs. Over time, this bias causes the model to produce increasingly brief responses, undermining overall training efficacy.

| |GRPO GSPO<br><br>| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

450

ResponseLength

400

350

300

250

200

0 100 200 300 400 500 600 700

Step

- Figure 1: Response length during RLVR training for Qwen2.5-VL-7B-Instruct. Under strictly controlled experimental settings (with all conditions except for the loss function kept constant), we compared the response length curves of GRPO and GSPO. It can be observed that GRPO induces the model to generate longer responses, while GSPO leads the model to gradually shorten its response length during training.

- As illustrated in the figure 1, training Qwen2.5-VL-7B-Instruct [Bai et al., 2025] under identical experimental settings, GSPO exhibited response length collapse during training, while GRPO did not show this phenomenon.

In this work, we revisit the GRPO’s and GSPO’s objective function and present a thorough analysis. To eliminate its inherent length bias, we introduce the Length-Unbiased Sequence Policy Optimization (LUSPO) algorithm, which implements a straightforward yet impactful modification: scaling each sequence’s loss by its own length. This adjustment alleviates the degraded performance observed with GSPO in dense models, while maintaining robust training dynamics in MoE models, and significantly accelerates the growth of response length during training and improves performance on reasoning tasks, both in text-only and multimodal settings.

To substantiate the utility of our approach, we perform comprehensive empirical studies across diverse architectures, encompassing dense, MoE, text-only, and vision-language (VL) models. Experimental results show that LUSPO can effectively eliminate the length bias of the GSPO’s objective function, ensuring stable training. Moreover, evaluations on a suite of text-only and multimodal benchmarks—including AIME24 [Zhang and Math-AI, 2024], AIME25 [Zhang and Math-AI, 2025], MathVista [Lu et al., 2024], MathVision [Wang et al., 2024], and others—demonstrate notable improvements over GRPO and GSPO. For example, training Qwen2.5-7B-Base [Qwen, 2025a] and Qwen3-30B-A3B-Instruct [Qwen, 2025b] with LUSPO yields up to 2.9% and 6.9% higher accuracy on AIME24 compared to GSPO, meanwhile, training Qwen2.5-VL-7B-Instruct [Bai et al., 2025] with LUSPO results in up to 1.6% higher accuracy than GRPO and 0.5% higher accuracy than GSPO on the MathVista-Mini.

In summary, our key contributions are:

- • We conduct a detailed analysis of the objective functions of GRPO and GSPO, and clarify the underlying reasons for their inherent length bias.
- • We propose the LUSPO algorithm, which neutralizes GSPO’s length bias through a principled modification.
- • We conduct extensive experiments across multiple models to validate the effectiveness and generalizability of our method.

### 2 Related Work

#### 2.1 Reinforcement Learning with Verifiable Rewards

With the tremendous success of DeepSeek-R1 [DeepSeek, 2025], RLVR has become widely adopted for post-training large language models. At the same time, a number of RLVR algorithms have been proposed by researchers. The most representative among them is Group Relative Policy Optimization (GRPO) [Shao et al., 2024], which eliminates the necessity for a value model by calculating the relative advantage of each response within a group corresponding to the same query. To enhance the effectiveness of reinforcement learning, Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) [Yu et al., 2025] incorporates four main techniques: Clip-Higher, Dynamic Sampling, TokenLevel Policy Gradient Loss, and Overlong Reward Shaping. Moreover, Dr.GRPO [Liu et al., 2025] proposes an unbiased optimization approach that enhances token efficiency without compromising reasoning performance.

#### 2.2 RLVR on MoE models

Currently, an increasing number of leading large models are actively exploring and adopting the MoE architecture. Significant progress is being made in areas such as efficient training, expert routing algorithms, and sparse activation techniques. MoE models are poised to become a foundational architecture for next-generation general-purpose large models, driving major advancements in AI capabilities for complex reasoning and cross-domain integration.The Selection of an appropriate RLVR algorithm for post-training MoE models has become critically important.

However, GRPO and its various extensions rely on token-level importance ratios, which tend to exhibit high variance—particularly in Mixture-of-Experts (MoE) models, where routing diversity and longer responses further amplify token-level fluctuations. This increased variance makes unstable updates more likely during training.

To tackle this problem, Group Sequence Policy Optimization (GSPO) [Zheng et al., 2025] introduces an importance ratio defined at the sequence level, rather than at the token level, and applies sequencelevel clipping, reward assignment, and optimization. GSPO exhibits substantial improvements over GRPO in terms of training stability, efficiency, and overall performance. Importantly, GSPO inherently addresses the stability issues associated with RL training of large Mixture-of-Experts (MoE) models, removing the necessity for intricate stabilization techniques. In addition, Soft Adaptive Policy Optimization (SAPO) [Gao et al., 2025] is a token-adaptive and smooth reinforcement learning algorithm developed to mitigate the instability and inefficiency found in hard-clipped policy optimization for large language models. Instead of relying on discontinuous clipping, SAPO utilizes a temperature-controlled soft gating mechanism, and applies asymmetric temperatures to more effectively manage negative-token gradients. This approach delivers a more stable and informative optimization signal throughout training.

### 3 Preliminaries

Notation In this work, we model an autoregressive language model parameterized by θ as a policy πθ. Let x denote a query and D be the query set. Given a response y to an input query x, its likelihood under the policy πθ is expressed as πθ(y|x) = |ty=1| πθ(yt|x,y<t) where |y| represents the number of tokens in y. Each query-response pair (x,y) can be evaluated by a verifier r, which assigns a reward value r(x,y) to the pair.

#### 3.1 GRPO: Group Relative Policy Optimization

For each query x ∈ D, GRPO [Shao et al., 2024] samples a group of responses from the behavior policy, computes their rewards and optimizes the following objective:

|yi| t=1 min wi,t(θ) Ai,t,clip(wi,t(θ),1 − ε,1 + ε) Ai,t

G i=1

JGRPO(θ) = Ex∼D,{y

1 G

1 |yi|

i}Gi=1∼πθold(·|x)

(1) where G denotes the number of responses generated for each query x (i.e., the size of the group), and the importance ratio wi,t(θ) and advantage Ai,t of token yi,t are defined as follows:

πθ(yi,t|x,yi,<t) πθ

, (2)

wi,t(θ) =

(yi,t|x,yi,<t)

old

r(x,yi) − mean {r(x,yi)}Gi=1 std {r(x,yi)}Gi=1

, (3)

Ai,t = Ai =

respectively, with all tokens in yi sharing a common advantage given by Ai.

#### 3.2 GSPO: Group Sequence Policy Optimization

The use of the token-level importance weigh ππθ(yi,t|x,yi,<t)

θold(yi,t|x,yi,<t) in GRPO presents certain issues. In contrast, the sequence-level importance weight π

θ(y|x)

πθold(y|x) is theoretically well-founded in the context of language generation: it quantifies the extent to which a response y sampled from πθ

(·|x) differs

old

from the current policy πθ(·|x). This measure is naturally compatible with sequence-level rewards and provides an intuitive basis for the clipping mechanism.

Building on this intuitive insight, GSPO [Zheng et al., 2025] adopts the following sequence-level optimization objective:

JGSPO(θ) = Ex∼D,{y

i}Gi=1∼πθold(·|x)

G

1 G

min si(θ) Ai,clip(si(θ),1 − ε,1 + ε) Ai (4)

i=1

where the group-based advantage is estimated as

r(x,yi) − mean r(x,yi)Gi=1 std r(x,yi)Gi=1

, (5)

Ai =

and the importance ratio si(θ) is defined according to the sequence likelihood [Zheng et al., 2023]:

si(θ) =

### 4 Algorithm

πθ(yi|x) πθ

(yi|x)

old

  1

1 |yi|

= exp

|yi|

 . (6)

|yi|

πθ(yi,t|x,yi,<t) πθ

log

(yi,t|x,yi,<t)

old

t=1

Response length is a key metric in RLVR for large language models. By extending the response length, the model is able to explore a wider range of reasoning patterns and develop more advanced problem-solving strategies throughout the training process.

#### 4.1 Analysis of Response Length Variation

There are many factors that influence changes in response length during training, which mainly fall into two aspects.

550

ViRL39K

ViRL39K+DAPO-MATH-17K

500

450

ResponseLength

400

350

300

250

200

0 100 200 300 400 500 600 700

Step

Figure 2: Response length during the training of Qwen2.5-VL-7B-Instruct with GSPO on different datasets exhibits different trends.

One aspect is the reward-driven changes in response length. For example, explicitly adding incentives or penalties for response length in the reward function can directly encourage longer or shorter responses. Alternatively, implicit effects may occur: if longer answers are more likely to receive correct rewards, the model tends to generate longer responses; if shorter answers are favored, the response length decreases accordingly, this may be related to the choice of training data. As illustrated in the figure 2, training Qwen2.5-VL-7B-Instruct on ViRL39k [Wang et al., 2025] and DAPO-MATH17K [Yu et al., 2025] led to a growth in response length, while only training on ViRL39k caused response length collapse. This is because DAPO-MATH-17K requires relatively more complex reasoning processes.

Another aspect, and also the main focus of this paper, is the preference for response length embedded in the loss function, as exemplified by approaches like GRPO Shao et al. [2024] and GSPO [Zheng et al., 2025]. An in-depth analysis will be presented in the subsequent subsections.

#### 4.2 Response Length Bias in GSPO

In the GRPO’s loss function, the inner layer first computes the average over tokens within each trajectory, and the outer layer then averages across all trajectories. This calculation leads to the following issue:

For long responses, since the number of tokens in a trajectory is large, dividing by the trajectory length means each token in a long response contributes less to the loss, while for short responses, each token contributes more.

If the sample is correct, tokens in short responses have a higher weight, so the model tends to generate shorter responses. If the sample is incorrect, the model avoids very short responses and favors longer responses. Thus, step accuracy influences response length: high accuracy leads to shorter responses, while low accuracy leads to longer ones.

In a single step of the GRPO algorithm, the contributions to the gradient from positive and negative samples are generally balanced, the impact of this length bias on training is not very significant. However, this length bias has a more pronounced impact on training process of GSPO.

Although GSPO’s loss function replaces token-level importance ratios with an importance ratio based on sequence likelihood, it still does not address the aforementioned issue of length bias.

- At the same time, GSPO adopts sequence-level clipping, which results in a significant increase in the proportion of tokens being clipped compared to GRPO. Additionally, in practical applications, GSPO

applies the Clip-Higher operation, causing the number of clipped negative samples to exceed that of positive samples. This indirectly leads to the gradients in a single step being dominated by positive samples, making the model’s responses tend to become increasingly shorter.

- 4.3 Length-Unbiased Sequence Policy Optimization

To address this issue, we proposed Length-Unbiased Sequence Policy Optimization (LUSPO). LUSPO introduces a simple yet effective modification to the GSPO’s objective by scaling each sequence’s loss according to its own length. This adjustment directly mitigates the response-level length bias inherent in GSPO, ensuring that longer sequences are not unfairly penalized during training. As a result, LUSPO not only alleviates the collapse in response length observed with GSPO, but also promotes more balanced and stable learning dynamics across both dense and Mixture-of-Experts (MoE) architectures.

LUSPO’s optimization objective is as follows:

JLUSPO(θ) = Ex∼D,{y

i}Gi=1∼πθold(·|x)

1 G

G

i=1

min si(θ) Ai,clip(si(θ),1 − ε,1 + ε) Ai · |yi|

(7) where the group-based advantage estimation and importance ratio take the same form as in GSPO.

- 4.4 Gradient Analysis The gradient of the LUSPO objective can be derived as follows (with clipping omitted for clarity):

∇θJLUSPO(θ) = ∇θEx∼D,{y

i}Gi=1∼πθold(·|x)

1 G

G

i=1

si(θ) Ai · |yi|

= Ex∼D,{y

i}Gi=1∼πθold(·|x)

1 G

G

i=1

si(θ) Ai · |yi| · ∇θ log si(θ)

= Ex∼D,{y

i}Gi=1∼πθold(·|x)

1 G

G

i=1

πθ(yi|x) πθold(yi|x)

1 |yi|

Ai · |yi| ·

1 |yi|

|yi|

t=1

∇θ log πθ(yi,t|x, yi,<t)

= Ex∼D,{y

i}Gi=1∼πθold(·|x)

1 G

G

i=1

πθ(yi|x) πθold(yi|x)

1 |yi|

Ai

|yi|

t=1

∇θ log πθ(yi,t|x, yi,<t) . (8)

Similarly, we can derive the gradient of the GSPO objective as follows:

∇θJGSPO(θ) = Ex∼D,{y

i}Gi=1∼πθold(·|x)

1 G

G i=1

πθ(yi|x) πθold(yi|x)

1 |yi|

Ai|y1

i|

|yi| t=1 ∇θ log πθ(yi,t|x,yi,<t) .

(9)

By comparing the gradients of the LUSPO and GSPO objectives, it can be observed that LUSPO eliminates the length-dependent bias present in GSPO for each trajectory.

- 5 Experiment

- 5.1 Training Setup

In this work, to assess the generalizability of our method, we performed comprehensive experiments across both dense and Mixture-of-Experts (MoE) architectures, as well as on text-only and visionlanguage (VL) models.

#### Implement Details

For model backbones, we utilize Qwen2.5-7B-Base [Qwen, 2025a] to represent dense models and Qwen3-30B-A3B-Instruct [Qwen, 2025b] to represent MoE models, both for text-only tasks. For multimodal evaluations, Qwen2.5-VL-7B-Instruct [Bai et al., 2025] serves as the model for training.

We trained Qwen2.5-7B-Base and Qwen2.5-VL-7B-Instruct on 8 Nvidia H800 GPUs and trained Qwen3-30B-A3B-Instruct on 4 x 8 Nvidia H800 GPUs. We trained these models mainly on verl [Sheng et al., 2024] framework.

Regarding hyperparameters, we adopt the AdamW [Loshchilov and Hutter, 2019] optimizer with a fixed learning rate of 1 × 10−6, coupled with a linear warm-up [Vaswani et al., 2017] spanning 20 rollout steps. Each prompt batch comprises 128 items, with 8 responses sampled per prompt during rollout. Training utilizes a mini-batch size of 16, and the maximum generation length is set to 32,768 and 4,096 tokens for text-only and VL models, respectively. For the Clip-Higher mechanism [Yu et al., 2025], we configure ϵlow to 2 × 10−3 and ϵhigh to 2.5 × 10−3, striking an effective balance between exploration and exploitation. The top-p is set to 0.7 and the temperature is set to 1.0 for the actor rollout.

#### Dataset

Table 1: Description of datasets. Dataset Description

math questions paired with corresponding single integer answers

DAPO-17K-MATH

Math/Phys/Chem/Bio charts/diagrams/tables-based reasoning broader STEM social science topics

ViRL39K

We trained on the DAPO-MATH-17K [Yu et al., 2025] dataset, while the VL model was trained on the ViRL39K [Wang et al., 2025] dataset. As shown in table 1, both datasets focus on scientific-related problems. Therefore, our primary evaluation focuses on mathematical and logical tasks, which serve

- as robust testbeds for our algorithm and can be seamlessly extended to other domains.

#### Reward

The reward we designed during training consists of three components, taking into account accuracy, format, and response length.

R = Raccuracy + Rformat + Roverlong (10)

The accuracy reward Raccuracy ∈ {0,1}, depending on whether the answer is correct. The format reward Rformat ∈ {0,0.5}, depending on whether the required format specified in the prompt is followed (figure 3).

###### System Prompt

Solve the question. When the user asks a question, first think through the reasoning process internally. Present this reasoning process explicitly, enclosed within <think> </think> tags. Then, provide the final answer in LaTeX format, wrapped in $...$, and ensure the answer is enclosed in the \boxed{} command. The answer should be placed within <answer> </answer> tags. For example: <think> Since $1+1=2$, so the answer is $2$. </think> <answer> The answer is $\boxed{2}$ </answer>. The output must always begin with <think> and end with </answer>.

Figure 3: System prompt used during VL model training

The overlong reward is defined as follows:

Roverlong(y) = min 0,

(Lmax − Lbuffer) − |y| Lbuffer × 0.3 (11)

where Lmax is the maximum generation length, Lbuffer is set to 512 for Qwen2.5-VL-7B-Instruct and 4096 for Qwen2.5-7B-Base and Qwen3-30B-A3B-Instruct and |y| is the length of response.

#### 5.2 Main Results

We adopt GRPO and GSPO as our primary baseline algorithms for comparison. For text-only models, we evaluated their performance on benchmarks, including AMC23, AIME24 [Zhang and Math-AI, 2024], AIME25 [Zhang and Math-AI, 2025] and MATH500 [Lightman et al., 2023]. For Qwen3-30B-A3B-Instruct, due to it’s powerful capabilities, we conduct evaluations only on AIME.

- Table 2: Comparison of GSPO and LUSPO-trained Qwen2.5-7B-Base and Qwen3-30B-A3B-Instruct models on text-only benchmarks.

Base model + Algorithm AMC23 AIME24 AIME25 MATH500 Avg. Qwen2.5-7B-Base

w/o RLVR 35.8 1.6 3.6 60.8 25.5 GSPO 55.3 11.8 11.2 71.0 37.3 LUSPO 58.3 14.7 13.9 78.4 41.3 ∆ GSPO +3.0 +2.9 +2.7 +7.4 +4.0

1

Qwen3-30B-A3B-Instruct w/o RLVR 97.5 60.0 57.2 96.2 GSPO — 76.7 59.2 — 68.0 LUSPO — 83.6 76.3 — 80.0 ∆ GSPO — +6.9 +17.1 — +12.0

Notably, as shown in table 2 both the dense model (Qwen2.5-7B-Base) and the Mixture-of-Experts (MoE) model (Qwen3-30B-A3B-Instruct) demonstrated significant improvements.

- Table 3: Comparison of GRPO, GSPO and LUSPO-trained Qwen2.5-VL-7B-Instruct models on multimodal benchmarks.

Base model + Algorithm MathVista-mini MathVision MathVerse(Vision Only) DynaMath WeMath LogicVista Avg. Qwen2.5-VL-7B-Instruct

w/o RLVR 67.4 26.2 41.1 20.2 34.5 45.6 39.2 GRPO 72.8 28.7 46.8 26.2 43.3 46.5 44.0 GSPO 73.9 27.7 45.6 25.9 40.7 47.7 43.6 LUSPO 74.4 28.0 47.1 24.6 45.6 53.7 45.6 ∆ GRPO +1.6 -0.7 +0.3 -1.6 +2.5 +7.2 +1.6 ∆ GSPO +0.5 +0.3 +1.5 +0.7 +5.1 +6.0 +2.0

In addition to evaluating text-only models, we further assessed the effectiveness of the proposed LUSPO algorithm in the multimodal domain. Table 3 presents a comparative analysis of LUSPO, GRPO and GSPO on various multimodal benchmarks, including MathVista-mini [Lu et al., 2024], MathVision [Wang et al., 2024], MathVerse(Vision Only) [Zhang et al., 2024], DynaMath [Zou et al., 2024], WeMath [Qiao et al., 2025] and LogicVista [Xiao et al., 2024], using the Qwen2.5VL-7B-Instruct model. The results demonstrate that LUSPO consistently outperforms GRPO and GSPO, indicating its strong generalization capability across both textual and visual-language tasks. Especially on the Wemath and LogicVista benchmarks, LUSPO achieves improvements of 5.1% and 6.0% over GSPO, respectively.

It is worth noting that in evaluations on several multimodal benchmarks, GSPO even achieves lower average scores than GRPO. This is because, as previously discussed, GSPO further amplifies the length bias arises from the operation of averaging the gradient contributions of each token within a trajectory.

#### 5.3 Training Dynamics Response Length

5000

GSPO

LUSPO

4000

ResponseLength

3000

2000

1000

0 20 40 60 80 100 120 140

Step

(a) Qwen2.5-7B-Base

7500

GSPO

LUSPO

7000

6500

ResponseLength

6000

5500

5000

4500

4000

3500

0 10 20 30 40 50

Step

(b) Qwen3-30B-A3B-Instruct

550

GSPO

LUSPO

500

450

ResponseLength

400

350

300

250

200

0 100 200 300 400 500 600 700

Step

(c) Qwen2.5-VL-7B-Instruct

Figure 4: Training curves of GSPO and LUSPO across reponse length.

As illustrated in figure 4, given the same number of training steps, LUSPO exhibits a more rapid increase in response length compared to GSPO. This accelerated growth in response length is indicative of enhanced model capability, further underscoring the effectiveness of the LUSPO algorithm.

In particular, as shown in figure 4c, within the VL model, GSPO results in a pronounced collapse of response length, which severely restricts the model’s capacity for exploration and complex reasoning. By contrast, LUSPO effectively mitigates this issue by maintaining a stable and sufficiently long response length throughout training. This stability allows the model to better leverage multimodal information and sustain its reasoning ability. This phenomenon of response length collapse also corroborates our earlier analysis that GSPO exacerbates length bias.

Table 4: Average length of generated responses on validation set.

Response Length GSPO LUSPO

Base Model

Qwen2.5-7B-Base 2611 3940 Qwen3-30B-A3B-Instruct 6757 11014

- Table 4 presents the average response length for GSPO and LUSPO on validation set. For both models the average response length of LUSPO is nearly 1.5 times that of GSPO.

#### Accuracy Reward

Accuracy reward obtained during training has long been recognized as a fundamental metric for monitoring the progress and effectiveness of reinforcement learning algorithms.

GSPO

0.35

LUSPO

0.30

AccuracyReward

0.25

0.20

0.15

0.10

0.05

0 20 40 60 80 100

Step

(a) Qwen2.5-7B-Base

0.8

GSPO

GSPO

LUSPO

LUSPO

0.52

0.7

0.50

AccuracyReward

AccuracyReward

0.48

0.6

0.46

0.5

0.44

0.4

0.42

0 100 200 300 400 500

0 10 20 30 40 50

Step

Step

(b) Qwen3-30B-A3B-Instruct

(c) Qwen2.5-VL-7B-Instruct

Figure 5: Training curves of GSPO and LUSPO across accuracy reward.

- As shown in the figure 5, we present the accuracy reward of the models during training. It is evident that, under the same number of training steps, LUSPO consistently outperforms GSPO for both dense and MoE models.

This is precisely because LUSPO eliminates the inherent length bias present in GSPO. As a result,

- at the same number of training steps, building on the previous analysis of response length during training, LUSPO produces longer responses, giving the model a larger exploration space and making it easier to solve more complex problems.

#### Validation Scores

The validation curve during training is crucial for assessing a model’s generalization ability. By comparing the reward and validation curves, we can identify issues such as overfitting or underfitting. If the validation performance plateaus or begins to decline while the reward continues to improve, it indicates that the model may be memorizing the training data rather than learning general patterns.

GSPO

LUSPO

0.15

Avg@32

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |

0.05

0.00

0 2 4 6 8 10 12 14

Step

(a) Qwen2.5-7B-Base

0.85

GSPO

LUSPO

0.80

0.75

Avg@32

0.70

0.65

0.60

0 1 2 3 4 5 6

Step

(b) Qwen3-30B-A3B-Instruct

Figure 6: Comparison of avg@32 scores on AIME24 for LUSPO and GSPO across training.

During training, for Qwen2.5-7B-Base [Qwen, 2025a] and Qwen3-30B-A3B-Instruct[Qwen, 2025b], we evaluate the avg@32 metric on AIME24 [Zhang and Math-AI, 2024] every 10 steps, and plot the results as shown in the figure 6.

- As shown in the figure 6, LUSPO not only achieves higher rewards than GSPO during training, but also demonstrates significantly improved performance on the validation set.

#### 5.4 Ablation Study

- Table 5: Comparison of GSPO and LUSPO trained on ViRL39k and DAPO-MATH-17k on multimodal benchmarks.

Base model + Algorithm MathVista-mini MathVision MathVerse(Vision Only) DynaMath WeMath LogicVista Avg. Qwen2.5-VL-7B-Instruct

GSPO 75.3 28.3 45.2 24.4 41.3 47.2 43.6 LUSPO 75.8 28.8 49.9 26.3 45.2 49.4 45.9 ∆ GSPO +0.5 +0.5 +4.7 +2.1 +3.9 +2.2 +2.3

To additionally validate the robustness of our method, we also conducted training using GSPO and LUSPO on the ViRL39k and DAPO-MATH-17k, which does not cause response length collapse. Specifically, for each question in DAPO-MATH-17k, we added an extra blank image to facilitate training with VL models. The results are shown in the table 5, and LUSPO still consistently outperforms GSPO across all five benchmarks.

GSPO

LUSPO

600

550

ResponseLength

500

450

400

350

0 100 200 300 400 500

Step

Figure 7: Response length curve during training on ViRL39k and DAPO-MATH-17k.

Similarly, as shown in the figure 7, the response length during training with LUSPO is consistently higher than that with GSPO.

### 6 Conclusion

In this work, based on the objective functions of GRPO and GSPO, we conduct an in-depth analysis of the reasons behind their response length bias. To address this issue and improve the stability of RLVR training for large language models, we introduce Length-Unbiased Sequence Policy Optimization (LUSPO), a novel reinforcement learning algorithm for training large language models. By applying a length-aware adjustment to sequence-level optimization, LUSPO addresses the response length bias inherent in GSPO, resulting in improved training stability and performance in both text-only and multimodal tasks. Extensive experiments on different model types and comprehensive evaluations on diverse benchmarks demonstrate the robustness and effectiveness of our method.

### References

DeepSeek. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645

(8081):633–638, September 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL http://dx.doi.org/10.1038/s41586-025-09422-z.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization,

2025. URL https://arxiv.org/abs/2507.18071.

Chujie Zheng, Pei Ke, Zheng Zhang, and Minlie Huang. Click: Controllable text generation with sequence likelihood contrastive learning, 2023. URL https://arxiv.org/abs/2306.03350.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

- Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2024, 2024.
- Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2025, 2025.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=QWTCcxMpPA.

Qwen. Qwen2.5 technical report, 2025a. URL https://arxiv.org/abs/2412.15115. Qwen. Qwen3 technical report, 2025b. URL https://arxiv.org/abs/2505.09388. Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min

Lin. Understanding r1-zero-like training: A critical perspective. In Conference on Language Modeling (COLM), 2025.

Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, and Junyang Lin. Soft adaptive policy optimization, 2025. URL https: //arxiv.org/abs/2511.20347.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. URL https: //arxiv.org/abs/1711.05101.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step, 2023. URL https://arxiv.org/abs/2305.20050.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024.

Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models, 2024. URL https://arxiv.org/abs/2411.00836.

Runqi Qiao, Qiuna Tan, Guanting Dong, MinhuiWu MinhuiWu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma GongQue, Shanglin Lei, Yifan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20023–20070, 2025.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts, 2024. URL https://arxiv.org/abs/2407.04973.

