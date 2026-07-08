## SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization

# arXiv:2511.06411v2[cs.AI]29Jan2026

Zhi Zheng1 Yu Gu2 Wei Liu1 Yee Whye Teh3 Wee Sun Lee1

### Abstract

Token Prob

Token Prob

Token Prob

read

0.6 0.3 0.1

read

0.6 0.3 0.1

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

The soft-thinking reasoning paradigm has demonstrated superior performance over traditional discrete-token Chain-of-Thought (CoT) reasoning in various scenarios. However, while discretetoken CoT reasoning can be reinforced through advanced reinforcement learning with verifiable rewards (RLVR) techniques such as group relative policy optimization (GRPO), extending the softthinking reasoning with such strong techniques remains challenging. This difficulty stems from the complexities of injecting stochasticity into softthinking tokens and updating soft-thinking policies accordingly. As a result, previous attempts to combine soft-thinking with RLVR typically underperform their discrete-token RLVR counterparts. To fully unlock the potential of soft-thinking, this paper presents a powerful policy optimization algorithm, SofT-GRPO. It injects the Gumbel noise into token probabilities with Gumbel-Softmax for controllable stochasticity, and leverages the Gumbel reparameterization trick to achieve accurate credit assignment to LLM soft-thinking policies. We conduct experiments over LLMs ranging from 1.5B to 7B parameters, where SofT-GRPO enables LLMs with soft-thinking to slightly outperform discrete-token CoT GRPO on Pass@1 (+0.13% on average accuracy), and brings a substantial uplift on Pass@32 (+2.19% on average)2.

read

... ...

0.6 0.3 0.1

... ...

Query Q LLM

see LLM

Query Q LLM

[Figure 5]

[Figure 6]

see LLM

... ...

Query Q LLM

see LLM

try

try

try

(a) LLM Reasoning with Discrete-Token CoT

Embeddings of Token  ,

Embeddings of Token  ,

Embeddings of Token  ,

Token Prob

Token Prob

Token Prob

read

0.6 0.3 0.1

read

0.6 0.3 0.1

read

0.6 0.3 0.1

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

... ...

... ...

... ...

Query Q LLM

Query Q LLM

see LLM

see LLM

Query Q LLM

see LLM

try

try

try

(b) LLM Reasoning with Soft-Thinking (Zhang et al., 2025b)

Rollout a Group of Discrete-Token CoTs with Multinomial Sampling

Rollout a Group of Discrete-Token CoTs with Multinomial Sampling

Rollout a Group of Discrete-Token CoTs with Multinomial Sampling

rea d

0.6 0.4 0.1

- 1

3

- 2

rea d

rea d

0.6 0.4 0.1

0.6 0.4 0.1

... ...

Query Q

rea d

0.6 0.4 0.1

LLM

see LLM

- 1

3

- 2

read

0.6 0.3 0.1

... ...

- 1

3

- 2

rea d

0.6 0.4 0.1

... ...

[Figure 13]

[Figure 14]

Query Q

Query Q

rea d

LLM

see LLM

0.6 0.4 0.1

... ...

LLM

see LLM

Query Q

... ...

LLM

see LLM

read

0.6 0.3 0.1

Query Q LLM

... ...

try

see LLM

read

0.6 0.3 0.1

[Figure 15]

[Figure 16]

Query Q

... ...

LLM

see LLM

[Figure 17]

[Figure 18]

Query Q

LLM

see LLM

... ...

try

Query Q LLM

... ...

1

try

see LLM

Query Q LLM

try

see LLM

try

GRPO ≈

 log

try

1

try

1

Updating the Discrete-Token CoT Policies

try

try

GRPO ≈

 log

GRPO ≈

 log

Updating the Discrete-Token CoT Policies

Updating the Discrete-Token CoT Policies

(c) GRPO (Shao et al., 2024) for Discrete-Token CoT

Rollout a Group of Soft-Thinking CoTs with Gaussian Noise on Soft Tokens

Token Prob + Gaussian Noise  ~ (0, 2)∈ 

Token Prob +Gaussian noise

Rollout a Group of Soft-Thinking CoTs with Gaussian Noise on Soft Tokens

Token Prob +Gaussian noise

... = + 

read

0.7 0.4 0.1

- 1

3

- 2

... = + 

read

0.7 0.4 0.1

...  = ...

Token Prob + Gaussian Noise  ~ (0, 2)∈ 

read

Token Prob +Gaussian noise

0.6 0.3 0.1

Query Q LLM

[Figure 19]

Token Prob +Gaussian noise

see LLM

[Figure 20]

Query Q LLM

... = + 

see LLM

 = + 

read

0.7 0.4 0.1

- 1

3

- 2

Query Q LLM

... = + 

see LLM

read

0.7 0.4 0.1

...  = ...

read

0.6 0.3 0.1

Query Q LLM

[Figure 21]

try

see LLM

[Figure 22]

Query Q LLM

try

see LLM

 = + 

Query Q LLM

try

see LLM

1

- 1

- 2 2

try

try

try

|| − ||2)

SoftTokens ≈

 (−

Updating Soft-Thinking CoT Policies with Gaussian Reparameterization Trick

1

- 1

- 2 2

|| − ||2)

SoftTokens ≈

 (−

Updating Soft-Thinking CoT Policies with Gaussian Reparameterization Trick

(d) Existing Work of Soft-Thinking + GRPO (Butt et al., 2025)

Rollout a Group of Soft-Thinking CoTs with Gumbel Noise

Rollout a Group of Soft-Thinking CoTs with Gumbel Noise

Gumbel noise

=log + 

Token Prob

Gumbel noise

=log + 

Token Prob

0.3

read

0.7 0.4 0.1

   (  / )

- 1

3

- 2

Gumbel noise

=log + 

Token Prob

=

Gumbel noise

=log + 

0.3

Token Prob

read

0.7 0.4 0.1

   (  / )

... ...

Query Q LLM exp(  / )

Query Q LLM

see LLM

Gumbel noise

=

=log + 

Token Prob

0.3

read

[Figure 23]

0.6 0.3 0.1

   (  / ) exp(  / )

... ...

-0.2

[Figure 24]

0.3

read

0.7 0.4 0.1

Query Q LLM exp(  / )

   (  / )

- 1

3

- 2

see LLM

Gumbel noise

=

=log + 

Token Prob

-0.2

... ...

=

0.3

read

0.7 0.4 0.1

   (  / )

... ...

see LLM

Query Q LLM exp(  / )

Query Q LLM

see LLM

=

-0.2

0.3

try

read

0.1

[Figure 25]

0.6 0.3 0.1

   (  / ) exp(  / )

... ...

-0.2

[Figure 26]

Query Q LLM exp(  / )

see LLM

=

### 1. Introduction

try

0.1

-0.2

... ...

see LLM

try

0.1

-0.2

try

0.1

1

try

0.1

SofT−GRPO ≈

 [−(  −log )−exp(−(  −log ))]

try

0.1

Updating Soft-Thinking CoT Policies with Gumbel Reparameterization Trick

1

SofT−GRPO ≈

 [−(  −log )−exp(−(  −log ))]

Reasoning with large language models (LLMs) has demonstrated impressive versatility in diverse domains (Sprague et al., 2024). However, most existing reasoning methods rely on the generation of discrete language tokens, which

Updating Soft-Thinking CoT Policies with Gumbel Reparameterization Trick

(e) SofT-GRPO (Ours) for Reinforcing Soft-Thinking

1School of Computing, National University of Singapore, Singapore 2School of Intelligence Science and Technology, Nanjing University, China 3Department of Statistics, University of Oxford, United Kingdom. Correspondence to: Wei Liu <weiliu87@nus.edu.sg>.

Preprint. January 30, 2026.

2The codes are available at https://github.com/ zz1358m/SofT-GRPO-master

Figure 1. The soft-thinking pattern (b) passes the expectation of token embeddings to the next LLM step (Zhang et al., 2025b), which can surpass the conventional discrete-token CoT (a) without any fine-tuning. However, employing the GRPO algorithm (c) will boost the performance of discrete-token CoT over soft-thinking, and existing attempts (d) of applying RLVR to soft-thinking derive inferior performances. The proposed SofT-GRPO (e) provides the first powerful RLVR algorithm, which can outperform the discretetoken CoT with GRPO on Pass@1, especially Pass@K.

will limit their ability to represent certain abstract concepts that cannot be accurately represented by a discrete token (Zhang et al., 2025b). In pursuit of better expression of abstract ideas, Zhang et al. (2025b) presents the soft-thinking reasoning pattern. As shown in Figure 1 (b), it replaces each discrete token in the chain-of-thought (CoT) with a continuous representation: the weighted sum of d-dimensional token embeddings, taking the output probabilities as the weights. With suitable sampling techniques—such as with the Gumbel-Softmax technique (Wu et al., 2025a) or the Dirichlet resampling technique (Zhuang et al., 2025)—softthinking can outperform discrete-token CoT on a wide range of tasks without requiring any fine-tuning (Wu et al., 2025a).

Recently, reinforcement learning with verifiable rewards (RLVR) approaches have become standard for improving discrete-token CoT reasoning (Wang et al., 2024; Liu et al., 2025b; Yu et al., 2025). The most representative RLVR method, Group Relative Policy Optimization (GRPO; Figure 1 (c)) (Shao et al., 2024), samples groups of CoT trajectories per query and reinforces the policy toward higherreward examples. Notably, RLVR fine-tuning can yield discrete-token CoT performance that surpasses that of softthinking reasoning from the original, unfine-tuned LLMs. Moreover, after discrete-token GRPO, the model’s softthinking ability often does not improve as substantially as its discrete-token reasoning ability.

However, extending RLVR methods to soft-thinking reasoning remains challenging (Jain & Rappazzo, 2025). As illustrated in Figure 1 (b), in contrast to the stochastic nature of sampling-based discrete-token CoTs, soft-thinking reasoning produces deterministic soft-tokens, lacking both randomness and gradients for RLVR. So, extending RLVR to soft-thinking should (1) introduce controllable stochasticity to adequately explore diverse soft-thinking paths, and (2) enable unbiased policy updates that properly assign the credits of high-reward soft-thinking samples to LLM logits. Previous work (Butt et al., 2025) (see Figure 1 (d)) attempts to address stochasticity by injecting Gaussian noise into soft tokens and applying a Gaussian reparameterization trick for RLVR gradients, but such approaches have struggled to match the performance of discrete-token RLVR, largely due to difficulties in attributing RLVR advantages to logits.

To address these challenges and fully unlock the LLM reasoning ability by soft-thinking, we propose a powerful policy optimization algorithm, SofT-GRPO. As shown in Figure 1(e), to introduce controllable stochasticity in the rollout process, SofT-GRPO samples groups of soft-thinking reasoning paths by injecting Gumbel noises into the output token probabilities and employs the Gumbel-Softmax technique to avoid invalid soft-tokens outside the pre-trained discrete-token embedding space. To accurately assign the reward credits to the output probability of LLMs

in soft-thinking policy updates, SofT-GRPO leverages the reparameterization trick on the Gumbel distribution. We perform thorough evaluations of SofT-GRPO on three representative LLMs from 1.5B to 7B parameters and across five numerical reasoning benchmarks. Experimental results show that SofT-GRPO-enhanced soft-thinking not only surpasses discrete-token GRPO on the averaged Pass@1 but also yields substantial gains on Pass@32, highlighting the practical advantages of robust policy optimization for softthinking reasoning. Our contributions are as follows:

- • We present SofT-GRPO, a powerful RLVR algorithm designed with the soft-thinking reasoning paradigm. It adopts the Gumbel-Softmax technique into the group rollout process, actively obtaining diverse but valid softthinking reasoning paths.
- • SofT-GRPO proposes a novel gradient estimation approach via Gumbel reparameterization, first achieving precise attribution of improvements to the LLM’s output probability distributions in policy optimization.
- • Over eight in-domain and out-of-domain benchmarks, SofT-GRPO improves the LLM’s reasoning ability with soft-thinking, outperforming the discrete-token GRPO, especially at higher sample rates (Pass@16 or Pass@32).

### 2. Preliminaries

#### 2.1. Discrete-Token CoT Reasoning

To solve a |Q|-token question Q = (q1,...,q|Q|), the discrete-token CoT reasoning process generates |R| rea-

soning CoT tokens R = (r1,...,r|R|) before predicting the answer A = (a1,...,a|A|) (Guo et al., 2025; Sprague et al., 2024). Each of Q, R, and A is a token sequence over the vocabulary T (i.e., Q,R,A ∈ T ∗), where T is the set of all possible language tokens. Language reasoning tokens and answer tokens are generated with the next-token prediction (NTP) policy of LLM πθ as follows:

###### |R|

p(R,A|Q) =

t=1

πθ(rt|[Q,(r1,...,rt−1)])

###### |A|

πθ(at|[Q,R,(a1,...,at−1)]),

t=1

(1)

where [·, ·] and [·, ·, ·] denotes concatenation.

Supervised Fine-tuning (SFT) for Discrete-Token Reasoning. As straightforward fine-tuning methods for LLM reasoning, the SFT methods (Wu et al., 2025b; Zheng & Lee, 2025) collect high-quality CoT labels for each question and then fine-tune LLMs for correct predictions. However, SFT methods highly rely on the quality of CoT labels, which are hard to obtain for complex datasets. Moreover, there are

1 G

EQ∼D,{R}G

JGRPO(θ) =

g=1,{A}Gg=1∼p(·,·|Q) G

|Rg|+|Ag|

1 |Rg| + |Ag|

min pg,tAˆg,clip(pg,t,1 − ϵ,1 + ϵ)Aˆg − βDKL(πθ||πθ

)

(2)

ref

t=1

g=1

πθ(ag,t|[Q,Rg,(ag,1,...,ag,t−1)]) πθold(ag,t|[Q,R,(ag,1,...,ag,t−1)]) if t > |Rg|

f(Ag) − mean(f(A))Gg=1 std(f(A))Gg=1

A ˆg =

, pg,t =

πθ(rg,t|[Q,(rg,1,...,rg,t−1)])

πθold(rg,t|[Q,(rg,1,...,rg,t−1)]) if t ≤ |Rg|.

also concerns about the out-of-domain generalization ability of SFT (Chu et al., 2025).

RLVR Fine-tuning for Discrete-Token Reasoning. RLVR fine-tuning methods such as GRPO (Liu et al., 2024), Dr. GRPO (Liu et al., 2025b), and DAPO (Yu et al., 2025) sample several CoTs [R,A] and assign a reward to each of

- them based on the quality of the answers A. The standard discrete-token GRPO algorithm (Shao et al., 2024) samples

G CoTs for each question Q with sampling policy πθ

old

and optimizes the reasoning policy towards CoTs with higher rewards. Its loss function is shown in Eq. (2), where Aˆg represents the advantage function for the g-th CoT, the reward function f(Ag) = 1 if and only if the answer Ag is correct, ϵ is the clipping hyperparameter, pg,t is calculated in an off-policy way for both the current policy πθ and the sampling policy πθ

old

, and DKL represents the KL-divergence of πθ and a reference policy πθ

ref

. RLVR methods can improve the discrete-token reasoning performance of LLMs across a wide range of reasoning problems without the need for CoT labels. However, for each CoT in each LLM step, discrete-token GRPO updates the probability of only one token based on the reward. As mentioned in Yue et al. (2025), making RLVR does not really incentivize reasoning capacity in LLMs beyond the base model.

- 2.2. Soft-Thinking Reasoning Paradigm

At each of the CoT reasoning steps, conventional discretetoken reasoning is constrained to selecting one token from the token set T , which may hinder the model’s ability to express certain abstract concepts that cannot be easily represented by a single deterministic token (Zhang et al., 2025b), thus undermining the LLM’s expression ability in reasoning. To represent abstract concepts, Zhang et al. (2025b) presents the soft-thinking paradigm, which replaces discretetoken reasoning steps R with a soft-thinking reasoning path S = (s1,...,s|S|). Each token si ∈ Rd (noted soft token) is a real vector, which is the sum of token embeddings weighted by their respective output probabilities. They are

- then fed into the next LLM step as follows: pt = πθ(·|[Q,(s1,...,st−1)])),

|T |

(3)

pi · ei,

st =

i=1

where pi ∈ [0,1] is the predicted probability of token i in pt and ei ∈ Rd is the LLM embeddings of token i. As detailed in Appendix A.3, the soft tokens lie in the convex hull of token embeddings, making it possible to maintain validity while transmitting more information.

Based on Eq. (3), Wu et al. (2025a) find that most pretrained LLMs tend to be single-threaded and the vanilla soft-thinking reasoning may lead to greedy feedback loop that suppresses alternative reasoning paths and undermines the benefits of transmitting more informative soft-tokens. So, they propose to introduce sampling strategies in the soft-thinking reasoning process for randomness, employing Gumbel noises and Gumbel-Softmax technique (Maddison et al., 2016; Jang et al., 2016) based on the output probability pt as follows:

exp(gi/τg) |T | i=1 exp(gi/τg)

gi = log pi + ϵi, yi =

,

(4)

|T |

yi · ei,

st =

i=1

where ϵi is a scalar noise sampled from the Gumbel distribution Gumbel(0,1), and τg is the temperature of GumbelSoftmax. Besides Gumbel-Softmax, Wu et al. (2025a) also tries to use the Dirichlet resampling technique as follows:

|T |

(x1,...,x|T |) ∼ Dirichlet(α,pt), st =

xi · ei,

i=1

(5) where α is a scaling parameter. Empirically, incorporated with Gumbel noises and the Gumbel-Softmax technique, the soft-thinking pattern can outperform conventional discretetoken CoT on a broad range of tasks, including numerical, code, and scientific reasoning, without requiring any finetuning. (Wu et al., 2025a).

#### 2.3. Related Work on RLVR Soft-Thinking

However, after being boosted with the RLVR fine-tuning, discrete-token reasoning will clearly outperform softthinking. So, Butt et al. (2025) tries to similarly improve the soft-thinking reasoning ability using GRPO by adding Gaussian noise on the soft-token st in Eq. (3) as follows:

sˆt = st + N(0,σ2Id). (6)

- ① SGLang Rollout Agent: Rollout a Group of Soft-Thinking CoTs with the Sampling Policy and Gumbel Noise .

SofT−GRPO =

1

(

1 | |

 [−( ’ −log )−exp(−( ’ −log ))]− [−  −exp(−  )])+

1 | |

(  log − log  ’ )

Query Q LLM

read

try

see LLM

0.7 0.4 0.1

 ’ ...

Token Prob  ’

0.3

-0.2

0.1

Gumbel noise

 ’ =log  ’ + 

 ’ =

   ( ’ / ) exp( ’ / )

<think> ... </think>

Answer Query Q LLM

read

try

see LLM

0.7 0.4 0.1

 ’ ...

Token Prob  ’

0.3

-0.2

0.1

Gumbel noise

 ’ =log  ’ + 

 ’ =

   ( ’ / ) exp( ’ / )

<think> ... </think>

Answer Query Q LLM

read

try

see LLM

0.6 0.3 0.1

 ’ ...

Token Prob  ’

0.3

-0.2

0.1

Gumbel noise

 ’ =log  ’ + 

<think> ... </think> Answer

3 2

1

Log prob of the curent policy Log prob of the sampling policy

 ’ =

   ( ’ / ) exp( ’ / )

③ Policy Optimization Agent: Updating Soft-Thinking CoT Policies with the Gumbel Reparameterization Trick with  ’ .

[Figure 27]

[Figure 28]

Query Q

Answer

 ’  ’

Concatenate |S| steps

CoT S

[ Q , S , ] LLM

read

try

see

0.65 (0.6+0.05) 0.25 (0.6-0.05) 0.1 (0.1+0)

Token Reconstructed Prob If Advantage 1 >0: Encourage  ’ with estimated noises.

Else when 1 <0: Discourage  ’ with estimated noises.

[Figure 29]

( ’ −log )

Query Q

Answer

 ’  ’

Concatenate |S| steps

CoT S

[ Q , S , ] LLM

read

try

see

0.65 (0.6+0.05) 0.25 (0.6-0.05) 0.1 (0.1+0)

Token Reconstructed Prob If Advantage 1 >0: Encourage  ’ with estimated noises.

Else when 1 <0: Discourage  ’ with estimated noises.

[Figure 30]

( ’ −log )

Query Q

Answer

 ’  ’

Concatenate |S| steps

CoT S

[ Q , S , ] LLM

read

try

see

0.65 (0.6+0.05) 0.25 (0.6-0.05) 0.1 (0.1+0)

Token Reconstructed Prob If Advantage 1 >0: Encourage  ’ with estimated noises.

Else when 1 <0: Discourage  ’ with estimated noises.

[Figure 31]

(i.e.,  ’ −log )

- ② Policy Optimization Agent: Reconstructing Probabilities with  ’ and Current LLM .

- Figure 2. The pipeline of the proposed SofT-GRPO algorithm. In training with each Query Q, the SofT-GRPO first generates a group of G soft-thinking reasoning paths with Gumbel noises and the Gumbel-Softmax technique (Wu et al., 2025a). We transmit the value gi′ and

yi′ for the loss calculation afterward. Then, we reconstruct the soft-thinking input. Finally, we update the soft-thinking policy with the off-policy proximal policy optimization (Schulman et al., 2017), optimizing soft-thinking policies with Gumbel reparameterization.

In deriving the soft-thinking probability in GRPO, they restore the value of sˆt in the rollout and calculate the log probability for the soft-thinking reasoning path S with the Gaussian reparameterization trick as follows, keeping the updating process of the answering part A unchanged:

- 1

- 2σ2||sˆt − st||22 . (7)

p(sˆt) ∝ exp −

Although this method can improve the soft-thinking ability of base LLMs and show improvements compared to discretetoken GRPO on the Pass@32 metrics (i.e., the pass rate with 32 attempts), there is a severe degradation in the average accuracy. This is mainly due to the following two drawbacks: (1) Adding noises on soft-tokens may help avoid inputs outside the pre-trained discrete-token embedding space, but adding noise to soft-tokens instead of logits is not direct and may theoretically mismatch the LLM predictions. As analyzed in Appendix C.1, token embeddings are linearly dependent (|T | ≫ d), so they cannot attribute which probability pi is beneficial to the effectiveness of added noise. Moreover, the added noise may even be impossible to be represented by the embeddings. (2) They do not use advanced sampling methods for effectiveness in training (e.g., incorporating Dirichlet resampling or Gumbel-Softmax technique (Wu et al., 2025a)), which will undermine the performance of sampled soft-thinking reasoning paths.

As a result, existing attempts of applying GRPO to softthinking will not keep its advantage over the discrete-token CoT under the no-finetune setting.

### 3. SofT-GRPO: Reinforcing Soft-Thinking Policy with Gumbel Reparameterization

To fully unlock the potential of soft-thinking for better LLM reasoning, we present the RLVR algorithm SofT-GRPO. As shown in Figure 2, it first samples soft-thinking reasoning paths with controllable randomness using the GumbelSoftmax technique (Wu et al., 2025a). In the following policy update stage, we propose a novel SofT-GRPO loss function with the Gumbel reparameterization trick, which accurately assigns reward credits to token probabilities.

#### 3.1. Group Rollout with Gumbel Noise

As an off-policy RLVR algorithm, SofT-GRPO samples a group of G soft-thinking CoTs for each query Q during rollout and saves them in a replay buffer. A principal challenge for applying RLVR to soft-thinking is that vanilla softthinking generates deterministic soft tokens, as each st is a fixed weighted sum of embeddings based solely on model probabilities (Eq. (3)). So, to explore diverse and powerful soft-thinking reasoning paths in the rollout process, we introduce the Gumbel-Softmax resampling technique (shown in Eq. (4)) as follows:

(·|[Q,(s1,...,st−1)]), gi′ = log pi + ϵi, yi′ =

##### (p1,...,p|T |) = πθ

old

exp(gi′/τg) |T | i=1 exp(g′

,

i/τg)

|T |

yi′ · ei,

st =

i=1

(9)

1 G

EQ∼D,{S}G

JSofT-GRPO(θ) =

g=1,{A}Gg=1∼p(·,·|Q) G

|Sg|+|Ag|

1 |Sg| + |Ag|

min pg,tAˆg,clip(pg,t,1 − ϵ,1 + ϵ)Aˆg − βDKL(πθ||πθ

)

ref

(8)

t=1

g=1

 

πθ(ag,t|[Q,S,(ag,1,...,ag,t−1)]) πθold(ag,t|[Q,S,(ag,1,...,ag,t−1)]) if t > |Sg| exp |Ti=1| − (gi′ − log pi) − exp(−gi′ + log pi) − − ϵi − exp(−ϵi) if t ≤ |Sg|.

pg,t =



where ei is the embedding of the i-th token, πθ

is the sampling policy used in rollout, and ϵi is a scalar Gumbel noise. Using the inverse transform sampling (Jang et al., 2016), we sample ϵi = −log(−log(u)) where u ∼ Uniform(0,1). Besides bringing stochasticity, as discussed in Section 2.2, resampling with Gumbel-Softmax will bring potential improvements on the quality of reasoning paths.

calculating off-policy probabilities, we first reconstruct the soft-thinking reasoning paths S = (s1,...,s|S|) from the restored yi′ values in the rollout process. Then, the probability is calculated with the restored gi′ values as follows:

old

p(g′|[Q,(s1,...,st−1)],θ) =

 ,

 

|T |

−(gi′ − log pi) − exp(−(gi′ − log pi))

Due to limitations in the amount of fine-tuning data, finetuning LLMs with soft-thinking needs to preserve the pretrained knowledge. So, applying RLVR to soft-thinking should also keep the generated soft tokens st within the pretrained discrete-token embedding space. In contrast to noise injection based on other distributions (e.g., Dirichlet or Gaussian), Gumbel-Softmax can better maintain the stability of the soft-token embeddings when providing stochasticity. As established in Theorem 3.1 (see Appendix D for proof), it produces a relaxation that maintains samples close to the discrete token distribution, thus mitigating the risk of generating out-of-vocabulary soft tokens.

exp

i=1

where (p1,...,p|T |) = πθ(·|[Q,(s1,...,st−1)]).

(12) Finally, the SofT-GRPO loss function is Eq. (8), where π and πref are token probability distributions that take history soft-tokens S and answer tokens A as input. pg,t is calculated as the probability ratio between current policy pθ (Eq. (11)) and sampling policy pθ

(Eq. (12)). Intuitively, SofT-GRPO recalculates the noise corresponding to g′ obtained in rollout and encourages this noise only when the current soft-thinking trajectory leads to a higher Aˆg. Unlike discrete-token GRPO in Eq. (2), SofT-GRPO updates the probability of several tokens at each LLM step t.

old

Theorem 3.1 (Gumbel-max Trick). Let (p1,...,pn) be nonnegative, and ϵ1,...,ϵn independent samples from Gumbel(0,1) (Maddison et al., 2016),

### 4. Experiments

pj n i=1 pi

. (10)

Pr j = arg max

(log pi + ϵi) =

i

In this section, we implement the proposed SofT-GRPO algorithm to reinforce the soft-thinking reasoning of three LLMs, including DeepSeek-R1-Distill-Qwen-1.5B, LLaMA-

#### 3.2. Gumbel Reparameterization for Loss Function

- 3.2-3B-Instruct, and DeepSeek-R1-Distill-Qwen-7B.
- 4.1. Implementation Details

In RLVR with discrete tokens (Eq. (1)), the probability of CoT trajectories can be obtained from the output Categorical distributions. Calculating such a probability is impossible in vanilla soft-thinking. To address this, SofT-GRPO proposes using the Gumbel reparameterization trick after obtaining the Gumbel noise ϵ during the rollout process. We take ϵ as RLVR actions, consider transforming ϵ to soft tokens st as a part of the transition, and estimate the next-soft-token probability with the density of Gumbel noise as follows:

Detailed Settings. In practice, we follow the general setting in soft-thinking works (Wu et al., 2025a), enabling both the top-p and top-k sampling strategies, considering only tokens with k highest probabilities instead of the whole token set |T | and normalizing their probabilities. In both training and inference, we set top-p as 0.95 and top-k as 5, the temperature of LLMs τ = 1, and the temperature in Gumbel-Softmax τg = 0.1.

|T |

p(g′|[Q,(s1,...,st−1)],θold) = exp

−ϵi−exp(−ϵi) ,

i=1

Training & Testing Settings. We employ DeepScaler (Luo et al., 2025) as the training dataset, which contains 40,315 queries. In implementing SofT-GRPO, we use SGLang (Zheng et al., 2024) for the rollout process and the verl-0.4.x framework (Sheng et al., 2024) for RLVR.

(11)

where g′ = (g1′ ,...,g|T′ |) is the gi′ values in Eq. (9) collected from the replay buffer. We provide the derivation

for this equation in Appendix C.3. Following GRPO, SofTGRPO requires off-policy probabilities for clipping. When

- Table 1. Experiment results of baselines and the proposed SofT-GRPO on five numerical reasoning benchmarks. We cover 3 LLMs from 1.5B to 7B and two reasoning patterns, i.e., discrete-token CoT and soft-thinking reasoning. @1 metrics denote the Mean@32 values, where we run each method 32 times on the dataset for average Pass@1 accuracies. @16 and @32 denote the Pass@16 and Pass@32 values on the dataset, respectively. We multiply all the results by 100 to highlight the differences between results. The best result on each metric and dataset is underlined, the best average result is bolded, and the second-best average result is shaded.

|Dataset<br><br>|AIME2024|AIME2025<br><br>|AMC23|MATH-500|GSM8K<br><br>|Average|
|---|---|---|---|---|---|---|
|Metrics<br><br>|@1 @16 @32|@1 @16 @32|@1 @16 @32<br><br>|@1 @16 @32|@1 @16 @32<br><br>|@1 @16 @32|

DeepSeek-R1-Distill-Qwen-1.5B Base LLM Discrete-Token CoT Reasoning Pattern No-Finetune 30.6 70.0 73.3 23.0 46.7 53.3 70.7 92.5 95.0 84.6 97.8 97.8 81.5 95.8 96.7 58.09 80.54 83.23 + GRPO 31.8 66.7 76.7 25.3 46.7 46.7 77.3 95.0 95.0 87.1 97.4 97.8 84.9 95.1 95.8 61.28 80.16 82.39 Soft-Thinking Reasoning Pattern No-Finetune 27.3 66.7 70.0 23.8 46.7 53.3 69.9 95.0 95.0 79.4 93.2 96.6 81.0 94.6 97.1 56.28 79.23 82.41 + GRPO 29.2 70.0 73.3 25.4 46.7 53.3 75.8 95.0 95.0 86.3 96.8 98.2 84.9 95.6 96.4 60.31 80.81 83.26 + SofT-GRPO 32.6 76.7 80.0 26.1 50.0 53.3 76.4 97.5 97.5 86.3 97.4 98.0 85.5 96.1 97.0 61.39 83.54 85.18 LLaMA-3.2-3B-Instruct Base LLM Discrete-Token CoT Reasoning Pattern No-Finetune 4.4 20.0 26.7 0.3 0.3 1.0 18.3 65.0 75.0 38.1 75.6 84.0 67.9 92.1 94.6 25.79 50.61 56.26

- + GRPO 7.3 23.3 26.7 0.5 3.3 3.3 27.3 62.5 67.5 48.3 77.2 82.6 79.6 95.4 96.5 32.60 52.35 55.32 Soft-Thinking Reasoning Pattern

No-Finetune 3.4 16.7 16.7 0.2 6.7 6.7 17.6 70.0 77.5 36.7 76.0 81.4 66.9 91.6 94.7 24.96 52.18 55.39

- + GRPO 8.0 20.0 23.3 0.7 3.3 10.0 27.3 70.0 75.0 47.8 76.8 81.8 79.2 94.8 96.3 32.60 53.00 57.28

+ SofT-GRPO 7.7 23.3 26.7 0.3 10.0 10.0 31.3 67.5 67.5 47.2 77.6 83.4 77.6 96.4 97.7 32.83 54.96 57.06 DeepSeek-R1-Distill-Qwen-7B Base LLM Discrete-Token CoT Reasoning Pattern No-Finetune 55.7 80.0 80.0 39.4 66.7 66.7 89.9 97.5 97.5 93.6 98.8 99.2 89.5 96.7 97.0 73.62 87.94 88.07

- + GRPO 54.0 80.0 80.0 40.4 66.7 70.0 89.2 95.0 95.0 93.5 98.4 99.0 91.4 96.1 96.6 73.69 87.24 88.12 Soft-Thinking Reasoning Pattern

No-Finetune 55.3 80.0 83.3 39.2 66.7 70.0 90.2 95.0 97.5 93.3 98.8 99.0 89.3 96.6 97.0 73.44 87.41 89.38

- + GRPO 55.5 80.0 80.0 37.8 63.3 66.7 90.1 97.5 97.5 93.6 98.6 99.0 91.8 96.7 96.9 73.76 87.23 88.01

+ SofT-GRPO 53.2 80.0 83.3 40.4 60.0 73.3 89.6 97.5 97.5 93.3 98.6 99.0 92.1 97.2 97.7 73.74 86.66 90.18

We involve five numerical reasoning benchmarks as indomain test sets (AIME2024, AIME2025, AMC23, MATH500, and GSM8K). We also employ a scientific reasoning benchmark (GPQA Diamond) and two code benchmarks (HumanEval and MBPP) for out-of-domain evaluation. The maximum generation length is 8192 in training and 32768 in testing. Test answers are verified using the Math Verify package (Kydl´ıˇcek, 2025). All experiments are implemented on a node of 8× NVIDIA H200 GPUs (141 GB VRAM each), and detailed parameters are shown in Appendix D.

Baselines. We use both conventional discrete-token CoT reasoning and soft-thinking reasoning as baselines for SofTGRPO. For discrete-token CoT, we include discrete-token GRPO trained on the same dataset and base LLMs without fine-tuning as baselines. For baselines with soft-thinking, we implement the soft-thinking reasoning pattern with the Gumbel-Softmax technique, using the best temperature setting τg = 0.5 proposed in Wu et al. (2025a). Under both patterns, we follow the parameters in (Wu et al., 2025a), setting the temperature τ of LLMs to 0.6 (we discuss this setting in Appendix E.2), top-p to 0.95, and top-k to 30.

Metrics. We adopt the general metrics of Mean@32 (the average Pass@1 accuracy over 32 runs), Pass@16, and Pass@32, which measure the average probability of covering the correct answer within 16 and 32 runs, respectively.

#### 4.2. Main Result

As shown in Table 1, SofT-GRPO can lead to a clear and consistent improvement from the No-Finetune results under the soft-thinking reasoning pattern. Moreover, SofT-GRPO outperforms applying soft-thinking on top of the discretetoken trained GRPO on average, demonstrating superior soft-thinking reasoning ability.

For Pass@1, SofT-GRPO consistently outperforms the discrete-token trained GRPO across LLMs of three sizes

- (+0.13% on average accuracy). In contrast, comparing the

+GRPO results on the discrete-token and soft-thinking patterns, performing soft-thinking on top of the discrete-token trained GRPO gives inconsistent improvement.

As observed in Yue et al. (2025), GRPO can lead to decreased performance on Pass@K over the No-Finetune model, even though it improves performance for Pass@1. We can observe the same result in Table 1. In contrast, the SofT-GRPO leads to a clear improvement in the Pass@16

- (+1.80% on average) and Pass@32 (+2.19% on average) metrics. Appendix E.2 shows that the superiority will also hold for different temperatures. As detailed in Appendix F, the improvement in Pass@16 and Pass@32 is likely due to the fact that for each sample in each LLM reasoning step, SofT-GRPO can reinforce the output probability of several tokens instead of focusing on one in GRPO.

- Table 2. Average accuracies on out-of-domain datasets. We cover GPQA Diamond, HumanEval, and MBPP. @1 metrics denote the Mean@32 values, where we run the methods 32 times on the dataset for average Pass@1 accuracies. @8, @16, and @32 denote the Pass@8, Pass@16, and Pass@32 values on the dataset, respectively.

|Dataset|GPQA Diamond<br><br>|HumanEval|MBPP|Average<br><br>|
|---|---|---|---|---|
|Metrics<br><br>|@1 @8 @16 @32|@1 @8 @16 @32<br><br>|@1 @8 @16 @32<br><br>|@1 @8 @16 @32|

DeepSeek-R1-Distill-Qwen-1.5B Base LLM Discrete-Token CoT Reasoning Pattern No-Finetune 36.7 84.3 92.4 96.0 68.1 87.2 90.2 93.9 65.5 84.8 89.1 91.1 56.77 85.45 90.59 93.64

- + GRPO 35.4 77.8 88.4 93.4 72.2 90.9 92.7 94.5 68.1 85.2 87.2 90.0 58.56 84.62 89.41 92.65 Soft-Thinking Reasoning Pattern

No-Finetune 36.0 83.8 91.9 97.0 67.2 89.6 91.5 92.7 64.7 84.4 86.0 88.7 55.98 85.97 89.79 92.79

- + GRPO 36.5 81.3 91.4 94.4 71.8 89.0 92.7 95.1 68.1 84.8 87.9 90.3 58.82 85.05 90.68 93.28

+ SofT-GRPO 37.3 82.8 89.9 95.5 71.2 88.4 91.5 94.5 68.8 88.7 90.3 91.4 59.08 86.65 90.54 93.80

- Table 3. Experiments on using majority voting to boost the performances on AIME2024, AIME2025, AMC23 and GSM8K. @1 represents the Pass@1 result from Table 1, which is averaged over 32 runs. M@16 and M@32 represent Major@16 and Major@32, respectively.

|Dataset<br><br>|AIME2024<br><br>|AIME2025|AMC23<br><br>|MATH-500|GSM8K<br><br>|Average|
|---|---|---|---|---|---|---|
|Metrics<br><br>|@1 M@16 M@32<br><br>|@1 M@16 M@32<br><br>|@1 M@16 M@32|@1 M@16 M@32<br><br>|@1 M@16 M@32<br><br>|@1 M@16 M@32|

DeepSeek-R1-Distill-Qwen-1.5B Base LLM Discrete-Token CoT Reasoning Pattern No Fine-tune 30.6 56.7 60.0 23.0 36.7 40.0 70.7 90.0 95.0 84.6 92.6 92.4 81.5 89.5 89.5 58.1 73.1 75.4 + GRPO 31.8 53.3 50.0 25.3 36.7 30.0 77.3 92.5 92.5 87.1 91.6 91.4 84.9 90.1 90.4 61.3 72.8 70.9 Soft-Thinking Reasoning Pattern No Fine-tune 27.3 56.7 60.0 23.8 33.3 33.3 69.9 95.0 92.5 79.4 88.8 89.8 81.0 89.3 89.3 56.3 72.6 73.0 + GRPO 29.2 43.3 46.7 25.4 30.0 30.0 75.8 92.5 92.5 86.3 91.0 91.4 84.9 90.5 90.8 60.3 69.5 70.3 + SofT-GRPO 32.6 60.0 63.3 26.1 33.3 36.7 76.4 92.5 95.0 86.3 92.2 92.0 85.5 90.6 90.5 61.4 73.7 75.5

- 4.3. Comparison in Out-of-Domain Datasets

Table 4. Comparison of the proposed SofT-GRPO to the RLVR fine-tuning method proposed in (Butt et al., 2025). We compare to their reported results under the same training dataset and LLM. Soft Tokens* represents their reported results fine-tuned under the soft-thinking pattern.

Besides numerical reasoning, we conduct out-of-domain experiments to evaluate the general reasoning ability of SofTGRPO. As shown in Table 2, the LLMs fine-tuned with SofT-GRPO on numerical questions can still demonstrate advantages from Pass@1 to Pass@32 on a scientific reasoning benchmark (GPQA Diamond) and two code benchmarks (HumanEval and MBPP).

|Dataset|MATH-500<br><br>|GSM8K|
|---|---|---|
|Metrics<br><br>|@1 @32<br><br>|@1 @32|

LLaMA-3.2-3B-Instruct Base LLM Discrete-Token CoT Reasoning Pattern No-Finetune 38.1 84.0 67.9 94.6 + GRPO 48.3 82.6 79.6 96.5 Soft-Thinking Reasoning Pattern + Soft Tokens* 41.3 77.9 75.5 95.2

#### 4.4. Boosting SofT-GRPO with Majority Voting

To further exploit the advantage of SofT-GRPO on Pass@32, in this subsection, we design to boost SofT-GRPO with majority voting (Chen et al., 2024). As shown in Table 3, SofTGRPO with majority-voting can outperform No-Finetune LLM and LLM fine-tuned with discrete-token GRPO across Major@16 (the accuracy of the most common answer in 16 runs) and Major@32. The results show that SofT-GRPOfine-tuned LLMs can be strengthened into better reasoning solvers with majority voting.

47.2 83.4 77.6 97.7

+ SofT-GRPO

(+5.9) (+5.5) (+2.1) (+2.5)

et al. (2025) also requires transitioning the d-dimensional vector between rollout workers and RLVR workers, making implementation more difficult.

#### 4.6. Comparison in the Token Efficiency

#### 4.5. Comparison to Butt et al. (2025)

In addition to accuracy, token efficiency is an important metric for LLMs. As shown in Appendix E.1, SofT-GRPO reduces thinking length compared to No-Finetune, and does not greatly increase tokens compared to discrete-token GRPO. Notably, SofT-GRPO leads to a clear reduction in thinking length for the LLaMA-3.2-3B-Instruct model, highlighting its effectiveness in reducing computational costs.

We conduct comparison experiments with an existing RLVR algorithm for the soft-thinking pattern, the method in Butt et al. (2025). We refer to this method as Soft Tokens in Table 4, where the proposed SofT-GRPO can demonstrate a clear improvement compared to the results reported in Butt et al. (2025). Compared to SofT-GRPO, the algorithm in Butt

Table 5. Ablation studies on the noise added in the proposed SofT-GRPO. We highlight the performance differences on the ablation variants (i.e., adding Dirichlet noises or Gaussian noises).

|Dataset<br><br>|AIME2024|AIME2025<br><br>|AMC23|MATH-500|GSM8K<br><br>|Average|
|---|---|---|---|---|---|---|
|Metrics|@1 @16 @32<br><br>|@1 @16 @32|@1 @16 @32|@1 @16 @32<br><br>|@1 @16 @32|@1 @16 @32|

DeepSeek-R1-Distill-Qwen-1.5B Base LLM Soft-Thinking Reasoning Pattern

|No-Finetune<br><br>+ GRPO|27.3 66.7 70.0 29.2 70.0 73.3<br><br>|23.8 46.7 53.3 25.4 46.7 53.3<br><br>|69.9 95.0 95.0 75.8 95.0 95.0<br><br>|79.4 93.2 96.6 86.3 96.8 98.2|81.0 94.6 97.1 84.9 95.6 96.4<br><br>|56.3 79.2 82.4 60.3 80.8 83.3|
|---|---|---|---|---|---|---|
|+ SofT-GRPO (Original, Gumbel)<br><br>+ SofT-GRPO (Dirichlet Noise)<br><br>+ SofT-GRPO (Gaussian Noise)|32.6 76.7 80.0 25.4 60.0 70.0<br><br>(-7.2) (-16.7) (-10) 21.9 60.0 73.0 (-10.7) (-16.7) (-7)<br><br>|26.1 50.0 53.3 22.9 56.7 63.3 (-3.2) (+6.7) (+10) 20.0 40.0 43.3<br><br>(-6.1) (-10) (-10)<br><br>|76.4 97.5 97.5 68.4 97.5 97.5 (-8) (0) (0)<br><br>65.6 97.5 97.5 (-10.8) (0) (0)|86.3 97.4 98.0 83.6 97.0 97.8 (-2.7) (-0.4) (-0.2) 79.9 95.2 96.6 (-6.5) (-2.2) (-1.4)<br><br>|85.5 96.1 97.0 83.6 97.0 97.8 (-1.8) (+0.9) (+0.8) 81.8 94.5 95.5 (-3.7) (-1.7) (-1.5)<br><br>|61.4 83.5 85.2 56.8 81.6 85.3 (-4.6) (-1.9) (+0.1) 53.8 77.4 81.2 (-7.6) (-6.1) (-4)|

0.85

0.6

0.550

AverageValidation(GSM8K)Accuracies

0.525

0.80

AverageTrainingRewards

AverageTrainingRewards

0.5

0.500

0.75

0.475

0.4

0.70

0.450

0.65

0.3

0.425

Discrete-token GRPO

0.60

Discrete-token GRPO

SofT-GRPO with Direclet Noise ( =4) SofT-GRPO with Direclet Noise ( =1) SofT-GRPO with Gaussian Noise ( =0.05)

SofT-GRPO with Direclet Noise ( =4) SofT-GRPO with Direclet Noise ( =1) SofT-GRPO with Gaussian Noise ( =0.05)

0.400

- SofT-GRPO with top-p=0.95, g=0.1

- SofT-GRPO with top-p=1.0, g=0.1

0.2

0.55

0.375

SofT-GRPO with Gumbel Noise ( g=0.1)

SofT-GRPO with top-p=0.95, g=0.25

SofT-GRPO with Gumbel Noise ( g=0.1)

0.50

0.350

0.1

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

GRPO Steps

GRPO Steps

GRPO Steps

(a) Training reward curve of variants on added noises

(b) Validation accuracy curve of variants on added noises

(c) Training reward curve of different hyperparameters

- Figure 3. Smoothed training or validation curves of ablation studies (the dashed background contains the data points). (a) discusses the setting of adding Gumbel noise in SofT-GRPO. (b,c) discusses the setting of top-p=0.95 and the Gumbel-Softmax temperature τg = 0.1.

### 5. Discussion

#### 5.1. Ablation on the Added Noise

In SofT-GRPO, we employ the Gumbel-Softmax technique for controllable stochasticity. As discussed in Section 2.2, adding Dirichlet noise can be another choice, so we compare the performance of the original SofT-GRPO and its two variants, adding Dirichlet noise or Gaussian noise to probabilities pi in training. The reward curve in training is shown in Figure 3(a), the validation curve is shown in Figure 3(b), and the final performance of variants is shown in Table 5. Compared to adding the original Gumbel noises, LLMs do not gain improvements using Dirichlet noise, and adding the Gaussian noise will cause a poorer initial performance.

#### 5.2. Ablation on Hyper-Parameters

SofT-GRPO sets the top-p as 0.95 and the Gumbel temperature as τg = 0.1. To demonstrate the reason for these settings, in Figure 3(c), we compare the training reward curve of the original SofT-GRPO with two variants (varying top-p to 1.0 or τg to 0.25). Results show that both variants will cause a collapse in the training process. As analyzed in Appendix E.3, the two variants will cause a substantial increase in the KL divergence between πθ and πθ

. So, we attribute this kind of collapse to the fact that fine-tuning

ref

under the soft-thinking pattern may lead to soft-token inputs outside the pre-trained discrete-token embedding space.

#### 5.3. Visualization: Soft-Thinking after SofT-GRPO

In Appendix H, we provide a visualization of the softthinking reasoning path after SofT-GRPO fine-tuning. As shown in Figure 11, high probability paths preserve interpretability, and we also observe new emerged distributions with high mass on a token as well as its antonym (Yeah and No), indicating preservation of multiple reasoning branches.

### 6. Conclusion

This paper presents a powerful RLVR algorithm, SofTGRPO, to reinforce LLMs under the soft-thinking reasoning pattern. It integrates controllable stochasticity with GumbelSoftmax and updates the soft-thinking policy with Gumbel reparameterization. It can demonstrate superior numerical, scientific, and code reasoning ability compared to the conventional discrete-token GRPO on Pass@1, especially Pass@16 and Pass@32. It can also be boosted into a better solver with majority voting. This work demonstrates the advantages of soft-thinking in LLM reasoning. Future works include extending the method for models such as Vision Language Models.

### Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Arora, D. and Zanette, A. Training language models to reason efficiently. arXiv preprint arXiv:2502.04463, 2025.

Butt, N., Kwiatkowski, A., Labiad, I., Kempe, J., and Ollivier, Y. Soft tokens, hard truths. arXiv preprint arXiv:2509.19170, 2025.

Chen, L., Davis, J. Q., Hanin, B., Bailis, P., Stoica, I., Zaharia, M., and Zou, J. Are more llm calls all you need? towards scaling laws of compound inference systems. arXiv preprint arXiv:2403.02419, 2024.

Chen, X., Zhao, A., Xia, H., Lu, X., Wang, H., Chen, Y., Zhang, W., Wang, J., Li, W., and Shen, X. Reasoning beyond language: A comprehensive survey on latent chain-of-thought reasoning, 2025. URL https: //arxiv.org/abs/2505.16782.

Cheng, J. and Van Durme, B. Compressed chain of thought: Efficient reasoning through dense representations. arXiv preprint arXiv:2412.13171, 2024.

Chow, Y., Tennenholtz, G., Gur, I., Zhuang, V., Dai, B., Thiagarajan, S., Boutilier, C., Agarwal, R., Kumar, A., and Faust, A. Inference-aware fine-tuning for best-ofn sampling in large language models. arXiv preprint arXiv:2412.15287, 2024.

Chu, T., Zhai, Y., Yang, J., Tong, S., Xie, S., Schuurmans, D., Le, Q. V., Levine, S., and Ma, Y. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Dai, M., Yang, C., and Si, Q. S-grpo: Early exit via reinforcement learning in reasoning models. arXiv preprint arXiv:2505.07686, 2025.

Deng, J., Pang, L., Wei, Z., Xu, S., Duan, Z., Xu, K., Song, Y., Shen, H., and Cheng, X. Latent reasoning in llms as a vocabulary-space superposition. arXiv preprint arXiv:2510.15522, 2025.

Feng, S., Fang, G., Ma, X., and Wang, X. Efficient reasoning models: A survey. arXiv preprint arXiv:2504.10903, 2025.

Goyal, S., Ji, Z., Rawat, A. S., Menon, A. K., Kumar, S., and Nagarajan, V. Think before you speak: Training language models with pause tokens. In ICLR, 2024. URL https: //openreview.net/forum?id=k5E1Yw5u3Q.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., and Tian, Y. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.

Hao, Z., Wang, H., Liu, H., Luo, J., Yu, J., Dong, H., Lin, Q., Wang, C., and Chen, J. Rethinking entropy interventions in rlvr: An entropy change perspective. arXiv preprint arXiv:2510.10150, 2025.

Hu, Z., Wang, Y., He, Y., Wu, J., Zhao, Y., Ng, S.-K., Breazeal, C., Luu, A. T., Park, H. W., and Hooi, B. Rewarding the rare: Uniqueness-aware rl for creative problem solving in llms. arXiv preprint arXiv:2601.08763, 2026.

Jain, A. and Rappazzo, B. Learning to reason with mixture of tokens. arXiv preprint arXiv:2509.21482, 2025.

Jang, E., Gu, S., and Poole, B. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.

Kydl´ıˇcek, H. Math-Verify: Math Verification Library, 2025. URL https://github.com/huggingface/ math-verify.

Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Liu, J., Li, Y., Fu, Y., Wang, J., Liu, Q., and Shen, Y. When speed kills stability: Demystifying rl collapse from the inference-training mismatch, september 2025a.

Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., and Lin, M. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025b.

Liu, Z., Liu, J., He, Y., Wang, W., Liu, J., Pan, L., Hu, X., Xiong, S., Huang, J., Hu, J., et al. Part i: Tricks or traps? a deep dive into rl for llm reasoning. arXiv preprint arXiv:2508.08221, 2025c.

Luo, M., Tan, S., Wong, J., Shi, X., Tang, W. Y., Roongta, M., Cai, C., Luo, J., Li, L. E., Popa, R. A., and Stoica, I. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. Notion Blog.

Maddison, C. J., Mnih, A., and Teh, Y. W. The concrete distribution: A continuous relaxation of discrete random variables. arXiv preprint arXiv:1611.00712, 2016.

Mahdavi, S., Li, M., Liu, K., Liao, R., and Thrampoulidis, C. Beyond accuracy: A policy gradient reweighting approach for pass@ k maximization in llms. In 2nd AI for Math Workshop@ ICML 2025, 2025.

Peng, R., Ren, Y., Yu, Z., Liu, W., and Wen, Y. Simko: Simple pass@ k policy optimization. arXiv preprint arXiv:2510.14807, 2025.

Qi, P., Liu, Z., Pang, T., Du, C., Lee, W. S., and Lin, M. Optimizing anytime reasoning via budget relative policy optimization. arXiv preprint arXiv:2505.13438, 2025a.

Qi, P., Liu, Z., Zhou, X., Pang, T., Du, C., Lee, W. S., and Lin, M. Defeating the training-inference mismatch via fp16. arXiv preprint arXiv:2510.26788, 2025b.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Shen, Z., Yan, H., Zhang, L., Hu, Z., Du, Y., and He, Y. Codi: Compressing chain-of-thought into continuous space via self-distillation. arXiv preprint arXiv:2502.21074, 2025.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Sprague, Z., Yin, F., Rodriguez, J. D., Jiang, D., Wadhwa, M., Singhal, P., Zhao, X., Ye, X., Mahowald, K., and Durrett, G. To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning. arXiv preprint arXiv:2409.12183, 2024.

Su, D., Zhu, H., Xu, Y., Jiao, J., Tian, Y., and Zheng, Q. Token assorted: Mixing latent and text tokens for improved language model reasoning. arXiv preprint arXiv:2502.03275, 2025.

Tan, W., Li, J., Ju, J., Luo, Z., Luan, J., and Song, R. Think silently, think fast: Dynamic latent compression of llm reasoning chains. arXiv preprint arXiv:2505.16552, 2025.

Walder, C. and Karkhanis, D. Pass@ k policy optimization: Solving harder reinforcement learning problems. arXiv preprint arXiv:2505.15201, 2025.

Wang, C., Li, Z., Bai, J., Zhang, Y., Cui, S., Zhao, Z., and Wang, Y. Arbitrary entropy policy optimization breaks the exploration bottleneck of reinforcement learning. arXiv preprint arXiv:2510.08141, 2025a.

Wang, J., Wu, Z., Lai, F., Lian, S., and Zeng, Z. Synadapt: Learning adaptive reasoning in large language models via synthetic continuous chain-of-thought. arXiv preprint arXiv:2508.00574, 2025b.

Wang, S., Zhang, S., Zhang, J., Hu, R., Li, X., Zhang, T., Li, J., Wu, F., Wang, G., and Hovy, E. Reinforcement learning enhanced llms: A survey. arXiv preprint arXiv:2412.10400, 2024.

Wei, X., Liu, X., Zang, Y., Dong, X., Cao, Y., Wang, J., Qiu, X., and Lin, D. Sim-cot: Supervised implicit chain-ofthought. arXiv preprint arXiv:2509.20317, 2025.

Wu, C., Lu, J., Ren, Z., Hu, G., Wu, Z., Dai, D., and Wu, H. Llms are single-threaded reasoners: Demystifying the working mechanism of soft thinking. arXiv preprint arXiv:2508.03440, 2025a.

Wu, X.-K., Chen, M., Li, W., Wang, R., Lu, L., Liu, J., Hwang, K., Hao, Y., Pan, Y., Meng, Q., et al. Llm finetuning: Concepts, opportunities, and challenges. Big Data and Cognitive Computing, 9(4):87, 2025b.

Xu, Y., Guo, X., Zeng, Z., and Miao, C. Softcot: Soft chain-of-thought for efficient reasoning with llms, 2025a.

Xu, Y., Guo, X., Zeng, Z., and Miao, C. Softcot++: Testtime scaling with soft chain-of-thought reasoning. arXiv preprint arXiv:2505.11484, 2025b.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S., and Huang, G. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In NeurIPS, 2025.

Zelikman, E., Harik, G. R., Shao, Y., Jayasiri, V., Haber, N., and Goodman, N. Quiet-star: Language models can teach themselves to think before speaking. In First Conference on Language Modeling, 2024. URL https://arxiv.

org/abs/2403.09629.

Zhang, J., Zhu, Y., Sun, M., Luo, Y., Qiao, S., Du, L., Zheng, D., Chen, H., and Zhang, N. Lightthinker: Thinking stepby-step compression. arXiv preprint arXiv:2502.15589, 2025a.

Zhang, Z., He, X., Yan, W., Shen, A., Zhao, C., Wang, S., Shen, Y., and Wang, X. E. Soft thinking: Unlocking the reasoning potential of llms in continuous concept space. arXiv preprint arXiv:2505.15778, 2025b.

Zheng, L., Yin, L., Xie, Z., Sun, C. L., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., et al. Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024.

Zheng, Z. and Lee, W. S. Reasoning-cv: Fine-tuning powerful reasoning llms for knowledge-assisted claim verification. arXiv preprint arXiv:2505.12348, 2025.

Zhu, R.-J., Peng, T., Cheng, T., Qu, X., Huang, J., Zhu, D., Wang, H., Xue, K., Zhang, X., Shan, Y., et al. A survey on latent reasoning. arXiv preprint arXiv:2507.06203, 2025.

Zhuang, Y., Liu, L., Singh, C., Shang, J., and Gao, J. Text generation beyond discrete token sampling. arXiv preprint arXiv:2505.14827, 2025.

### Appendix Contents

- 1. Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- (a) RLVR on Discrete-Token CoT Reasoning
- (b) Latent Reasoning
- (c) The Working Mechanism of Soft-thinking

- 2. Prompt of Language Reasoning and Latent Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 3. Motivation and Theoretical Proof . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- (a) Motivation: Mismatch in Butt et al. (2025) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .16
- (b) Proof for Theorem 3.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- (c) Derivation of Eq. (8), Eq. (11) and Eq. (12) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- 4. Detailed Parameters for Training & Testing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .19
- 5. Supplementary of Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .20

- (a) Supplementary of Token Efficiency
- (b) Experiments under Different Temperatures
- (c) Supplementary Ablation on Hyper-Parameters
- (d) Pass@K up to K=1024
- (e) P-values for Significance on Pass@K

- 6. Discussion & Analysis on Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- 7. Baselines & Datasets & Licenses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .26

- (a) Baselines
- (b) Datasets
- (c) Inference Framework
- (d) Licenses

- 8. Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

### A. Related Work

In this section, we will discuss recent developments on discrete-token RLVR, a similar domain of soft-thinking, the latent reasoning methods, and the working mechanism of soft-thinking.

#### A.1. RLVR on Discrete-Token CoT Reasoning

Deepseek-R1-zero (Guo et al., 2025) has demonstrated remarkable performance with discrete-token RLVR. And there has been a wide collection of RLVR methods for discrete-token CoT policy optimization, such as GRPO (Shao et al., 2024), Dr. GRPO (Liu et al., 2025b), DAPO (Yu et al., 2025), and Lite PPO (Liu et al., 2025c). To further improve these algorithms towards some specific goals, recent works focus on developing RLVR methods for efficient responses (Feng et al., 2025) or controlling the entropy in CoT generation for better exploration-exploitation balance (Hao et al., 2025).

To generate concise discrete-token CoTs, Arora & Zanette (2025) modifies the reward function by adding penalties on the generation length. Qi et al. (2025a) and Dai et al. (2025) turn to formulate the fine-tuning process for concise language CoT as a multi-objective optimization task with the trade-off between token efficiency and accuracy. They truncate the generation of discrete-token CoT at several thinking budgets and optimize the overall performance over them.

RLVR methods with entropy control try to make a better balance between exploration and exploitation. They try to control the entropy in the training process to stimulate the exploration within the groups of CoTs in GRPO (Wang et al., 2025a).

This paper only considers GRPO as the only RLVR pipeline, because we believe these modifications (e.g., considering entropy or DAPO, which proposes to remove the KL divergence term) should be orthogonal to the two major contributions of SofT-GRPO: Sampling with Gumbel-Softmax and credit assignment with Gumbel reparameterization. In the future, we would like to do a comprehensive investigation to check this.

#### A.2. Latent Reasoning

Similar to the soft-thinking pattern, latent reasoning methods pass continuous vectors between LLM steps. These methods fully decouple the reasoning process from explicit natural language (which soft-thinking does not do) and perform inference in the hidden space of the model. Generally, latent CoT methods are often diverse from each other and can be mainly divided into token-wise auto-regressive methods and auxiliary strategies (Chen et al., 2025; Zhu et al., 2025). Token-wise auto-regressive methods transform the reasoning process into ’soft thoughts’ with dense latent embeddings (Hao et al., 2024) or specialized tokens (e.g., pause (Goyal et al., 2024; Zelikman et al., 2024)). These methods focus on transferring the original reasoning policy in the language domain to a latent embedding space, including curriculum learning (e.g., Coconut (Hao et al., 2024), LightThinker (Zhang et al., 2025a), SIM-COT (Wei et al., 2025)), self-distillation (e.g., CODI (Shen et al., 2025)), and one-shot compression (e.g., CoLaR (Tan et al., 2025), SynAdapt (Wang et al., 2025b), Latent-SFT (Deng et al., 2025)). Auxiliary strategies (e.g., SoftCoT (Xu et al., 2025a)) generate latent embeddings from an auxiliary module and inject them into the frozen main model (Cheng & Van Durme, 2024; Xu et al., 2025b; Su et al., 2025).

Due to the existing token-wise auto-regressive methods completely treating the language CoTs as the label, these methods can hardly surpass or even reach the performance level of reasoning LLMs. So empirically, these methods can effectively improve the token efficiency compared to language CoT, but there is a clear performance drop. Auxiliary strategies, instead, can effectively boost the performance of the original LLM, sacrificing the running efficiency.

Compared to latent reasoning methods mainly aiming at better efficiency, the proposed SofT-GRPO aims at reinforcing the accuracy of the soft-thinking pattern to surpass discrete-token CoT with GRPO on general reasoning tasks.

#### A.3. The Working Mechanism of Soft-thinking

As illustrated in (Zhang et al., 2025b), soft-thinking is effective because it transforms the exponential path expansion of reasoning into a tractable and parallelizable process, rigorously approximating full path-summation without sacrificing efficiency or accuracy.

- • Continuous Concept Space: Soft Thinking replaces discrete token choices with “concept tokens,” which are probability-weighted mixtures over all token embeddings. This enables the model to represent more abstract and nuanced concepts beyond individual tokens.

- • Parallel Path Exploration: By retaining the full probability distribution at each reasoning step, Soft Thinking allows the model to implicitly explore multiple reasoning trajectories in parallel, rather than committing early to a single discrete path.

Remarkably, LLMs can directly interpret and process soft tokens without the need for additional fine-tuning. Intuitively, this is because soft tokens are constrained to a well-confined space, specifically the convex hull formed by the model’s embedding vectors.

• First-Order Linearization for Effectiveness: Moreover, the effectiveness of the soft-thinking paradigm can be understood as a recursive first-order (linear) approximation of the full discrete autoregressive path-sum as follows:

- 1. For discrete reasoning tokens R = (r1,...,r|R|), answer A = (a1,...,a|A|), and LLM policy πθ, we have:

p(A | Q) =

R

|R|

t=1

πθ(rt | [Q,(r1,...,rt−1)])

|A|

t=1

πθ(at | [Q,R,(a1,...,at−1)]). (13)

- 2. Soft-Thinking Tokens: At each reasoning step, define the soft token as the expected embedding:

pt ∼ πθ(· | [Q,(s1,...,st−1)]), st =

|T |

i=1

pt,i · ei, (14)

where pt is the token probability distribution, ei is the embedding of token i.

- 3. Then we can recursively replace each sum over discrete tokens by soft tokens. For example, for the first reasoning step:

p(A|Q) =

r1

πθ(r1|Q)p(A|Q,r1) (15)

≈ p(A|Q, Er

1

[er

1

]) (16) = p(A|Q, s1). (17)

Here, s1 = |Ti=1| πθ(r1 = i|Q) · ei is the soft token for the first step. For later steps, recursively apply the same procedure:

p(A|Q,s1) =

r2

πθ(r2|Q,s1)p(A|Q,s1,r2) (18)

≈ p(A|Q,s1,s2), (19)

where s2 =

|T |

i=1

πθ(r2 = i|Q,s1) · ei. (20)

- 4. Recursive Soft-Token Approximation: By recursively applying this linearization at each step, we approximate the original intractable path sum by a deterministic soft-token chain:

p(A | Q) ≈

###### |S|

###### |A|

πθ(st | [Q,(s1,...,st−1)])

t=1

t=1

πθ(at | [Q,S,(a1,...,at−1)]). (21)

where S = (s1,...,s|S|). Thus, each st replaces the sum over all possible rt, acting as a linear surrogate.

However, it is important to note that not all points in the embedding space (i.e., convex hull of token embeddings) are equally interpretable or accessible to the LLM. For particularly smooth or spread-out distributions (making the approximation in Eq. (15) fail), the expressiveness of soft tokens remains limited. We believe this limitation is closely related to the “collapse” phenomenon discussed in Section 5.2, where the model’s reasoning may degrade or repeat when encountering out-of-distribution soft tokens.

### B. Prompt of Language Reasoning and Latent Generation

In this part, we show the prompt we adopt for reasoning problems, including the in-domain numerical reasoning, out-ofdomain GPQA reasoning, and code reasoning. We inherit the prompts in Zhang et al. (2025b) for out-of-domain benchmarks. We use the same prompt for both the soft-thinking reasoning pattern and the discrete-token reasoning pattern.

Prompt for Numerical Reasoning. user {Question} Let’s think step by step and output the final answer within \boxed{}

Prompt for GPQA Reasoning. user Please solve the following multiple-choice question. Please show your choice in the answer field with only the choice letter, e.g.,”answer”: ”C”. {Question}

Prompt for Code Reasoning (HumanEval). user Please solve the programming task below in Python. Code should be wrapped in a markdown code block. “‘python {Question} “‘

Prompt for Code Reasoning (MBPP). user Please solve the programming task with test cases below in Python. Make sure your code satisfies the following requirements:

- 1. The function name and signature must match exactly as specified in the test cases.
- 2. Your code should be wrapped in a markdown code block without including any test cases. Task: {Question} Test Cases: “‘python {TestCases} “‘

The blue part represents the specific question (query Q), the brown part represents the possible test cases provided in the MBPP code reasoning benchmark. We use the code provided in Zhang et al. (2025b) in the verification process of the responses.

### C. Motivation and Theoretical Proof

- C.1. Motivation: Mismatch in Butt et al. (2025) Butt et al. (2025) adds Gaussian noise on the soft-token inputs as follows:

st =

and calculates the log probability as follows:

|T |

pi · ei, sˆt = st + N(0,σ2Id). (22)

i=1

- 1

- 2σ2||sˆt − st||22 + Constant. (23)

log p(sˆt) = −

Let E ∈ R|T |×d be the embedding matrix, and each token probability vector p ∈ ∆|T |−1 corresponds to the soft input s = E⊤p ∈ Rd.

Assume the observed noisy soft input is sˆ, and we define a likelihood

- 1

- 2σ2∥sˆ − E⊤p∥22. (24)

log p(sˆ | p) ∝ −

Suppose we want to regard this as a likelihood on p. In general, the mapping p  → E⊤p is many-to-one: since |T | > d, the kernel of E⊤ is nontrivial, so

∃p1 ̸= p2, with E⊤p1 = E⊤p2. (25) Thus, the same s may correspond to infinitely many p. This means that, under this Gaussian model, two different token mixtures can lead to the exact same log-probability value. The information about the original token distribution is partially lost in the embedding projection, unless E is invertible (which it is not). Thus, the use of a Gaussian noise model on the embedding space gives a mismatch to the true simplex-based probability geometry.

- In summary of Drawback 1: Due to the non-injectivity (non-invertibility) of the embedding transformation, the model log p(sˆt) ∝ −∥sˆt − st∥2 does not define a true likelihood on the simplex of token probabilities.

The above mismatch is not only due to the non-invertibility of the embedding matrix. Even if we restrict p to be sparse (nonzero only on a top-k set, which is a general setting of LLMs or a common nature of LLM predictions), and even if the corresponding submatrix EK is invertible, the process of adding Gaussian noise in the d-dimensional embedding space fundamentally breaks the connection to sparse token distributions.

More specifically, after adding Gaussian noise, with a general top-k setting (e.g., k=10 to 30, and k << d) the perturbed embedding sˆt = st+ϵ (with ϵ ∼ N(0,σ2Id)) will almost surely not lie in the convex hull of any set of k token embeddings. In other words,

∀ sˆt, for almost all ϵ, ∄ k-sparse p such that sˆt = E⊤p. (26) The set of all top-k soft-token embeddings forms a low-dimensional union of simplices in Rd, which is a measure-zero subset of the space. The probability of a randomly perturbed embedding sˆt coinciding with a legal top-k mixture is thus zero.

Therefore, defining the likelihood p(sˆt) as a simple Gaussian on embedding space cannot be interpreted as a likelihood on the space of top-k soft tokens—not only due to non-invertibility or nonlinearity, but more fundamentally because most sˆt produced by noise are not realizable by any top-k soft-token distribution.

- In summary of Drawback 2: Under the general top-k setting, the likelihood p(sˆt) as a simple Gaussian on the embedding space cannot be interpreted as a likelihood on the space of top-k embeddings.

#### C.2. Proof for Theorem 3.1

Theorem 3.1 (Gumbel-max Trick) Let (p1,...,pn) be nonnegative real numbers, not all zero. Let g1,...,gn be independent samples from Gumbel(0,1). Then,

pj n i=1 pi

. (27)

Pr j = arg max

(gi + log pi) =

i

Proof. For any j ∈ {1,...,n},

(gi + log pi) = Pr(gj + log pj ≥ gi + log pi, ∀i ̸= j)

Pr j = arg max

i

+∞

Pr(gi ≤ gj + log pj − log pi)fg

=

(gj)dgj,

j

−∞ i̸=j

where fg

(g) = e−g−exp(−g) is the PDF of the standard Gumbel distribution. Pr(gi ≤ t) = FGumbel(t) = exp(−e−t), so

j

The total probability is

Let S = 1 + i̸=j p

pj =

i

=

Pr(gi ≤ gj + log pj − log pi)

i̸=j

exp −e−(g

j+log pj−log pi)

=

i̸=j

 −

 

e−(g

j+log pj−log pi)

= exp

i̸=j

 .

 −e−g

- pi

- pj

= exp

j

i̸=j

 dgj

 −e−g

+∞

- pi

- pj

−gj exp

e−g

e−e

j

j

−∞

i̸=j

 

 dgj.

 −e−g

 1 +

+∞

- pi

- pj

e−g

=

exp

j

j

−∞

i̸=j

n i=1 pi

, dgj = −dyy , y ∈ (0,+∞),

pj . Substitute y = e−g

j

0

dy y

y exp(−yS) −

y=+∞

=

+∞

1 S

exp(−yS)dy =

0

pj n i=1 pi

=

.

Thus, combining all the above equations,

pj n i=1 pi

Pr j = arg max

(gi + log pi) =

.

i

| |
|---|

#### C.3. Derivation of Eq. (8), Eq. (11) and Eq. (12)

The Gumbel-Softmax rollout of SofT-GRPO is given in Eq. (9). On the determinism of the mapping. It is crucial to note that, for fixed logits (or probabilities) pt and Gumbel noises ϵt, the resulting soft token st is a deterministic function of (pt,ϵt):

st = SoftEmbed(pt,ϵt), (28)

Hence, in our analysis, we always track the generative randomness via ϵt (consider ϵt as the action in RLVR), and not via st directly. Cause every g′ can map to one deterministic soft-token st, but st can map to multiple g′, calculating density on g′ avoids the ill-posedness of attempting to define sampling densities directly in the soft-token space. So, we consider outputting g′ as an action and put the generation of st as a part of the environment transition.

- Eq. (11): log-likelihood under the old policy. During rollout (generated by the old policy θold), we sample Gumbel noises ϵt = (ϵt,1,...,ϵt,|T |). The randomness of st is entirely attributable to ϵt, given pt. The joint density factorizes as

p(ϵt) =

|T |

i=1

fGumbel(ϵt,i), fGumbel(x) = exp − x − exp(−x) ,

where fGumbel represents the probability density function of a Gumbel distribution. Hence, the corresponding log-likelihood is

log p(ϵt) =

|T |

i=1

log fGumbel(ϵt,i) =

|T |

i=1

− ϵt,i − exp(−ϵt,i) ,

which gives Eq. (11). In this formulation, the density is over ϵt, while st is merely a deterministic function thereof.

- Eq. (12): log-likelihood under the current policy. For off-policy correction, we reuse the same realized perturbations from rollout. Equivalently, we store the perturbed logits.

##### gt,i′ ≜ log poldt,i + ϵt,i,

so that, under a new policy θ with probabilities pt,i = πθ(i | [Q,(s1,...,st−1)]), the implied noise that would have generated the same gt,i′ under the new policy is

ϵt,i = gt,i′ − log pt,i.

Therefore, the log-density of the same realized perturbation under the new policy is

|T |

|T |

log fGumbel(gt,i′ − log pt,i) =

i=1

i=1

− (gt,i′ − log pt,i) − exp − (gt,i′ − log pt,i) ,

which is exactly Eq. (12). This preserves consistency in credit assignment and importance sampling, as all stochasticity and probability mass are correctly attributed to ϵt rather than st.

Eq. (8): Off-Policy Ratio Cause the transformation from g′ to st is fixed, the reparameterization process of SofT-GRPO considers the added noise as action. So, using the Eq. (11) and Eq. (12), we can get the final result in Eq. 8 for the soft-thinking part.

p(g′|[Q,(s1,...,st−1)],θ) p(g′|[Q,(s1,...,st−1)],θold)

exp |Ti=1| −(gi′ − log pi) − exp(−gi′ + log pi) exp |Ti=1| −ϵi − exp(−ϵi)

=

 

 .

|T |

− (gi′ − log pi) − exp(−gi′ + log pi) − − ϵi − exp(−ϵi)

=exp

i=1

(29)

### D. Detailed Parameters for Training & Testing

All our experiments are performed on 8 × H200 GPUs in about 2 to 3 days. Compared to discrete-token GRPO, SofT-GRPO requires ×2 to ×3 in time, mainly due to the more time consumption for soft-thinking rollout (Zhang et al., 2025b; Wu et al., 2025b). We show the experimental configurations in 6 for No-Finetune, 7 for GRPO, and 8 for SofT-GRPO.

Table 6. Parameters of No-Finetune Parameter Value

NO-FINETUNE TESTING under Discrete-Token Reasoning Paradigm

Maximum response length 32768 tokens Sampling temperature 0.6 (top-p, top-k) (0.95, 30)

NO-FINETUNE TESTING under Soft-Thinking Reasoning Paradigm Maximum response length 32768 tokens

- Sampling temperature 0.6

Gumbel-temperature τg 0.5 (top-p, top-k) (0.95, 30)

Table 7. Parameters of Discrete-Token GRPO Parameter Value

DISCRETE-TOKEN GRPO TRAINING Maximum response length 8192 tokens

- Sampling temperature 1.0 (top-p, top-k) (1, -1) Group Size G 8 Learning rate 1 × 10−6 KL loss coefficient β 0.001 Policy clipping parameter ϵ 0.2

DISCRETE-TOKEN GRPO TESTING under Discrete-Token Reasoning Paradigm Maximum response length 32768 tokens

- Sampling temperature 0.6 (top-p, top-k) (0.95, 30)

GRPO TESTING under Soft-Thinking Reasoning Paradigm Maximum response length 32768 tokens

- Sampling temperature 0.6

Gumbel-temperature τg 0.5 (top-p, top-k) (0.95, 30)

Table 8. Parameters of SofT-GRPO Parameter Value

SOFT-GRPO TRAINING Maximum response length 8192 tokens

- Sampling temperature 1.0 (top-p, top-k) (0.95, 5) Group Size G 8 Gumbel-temperature τg 0.1 Learning rate 1 × 10−6 KL loss coefficient β 0.001 Policy clipping parameter ϵ 0.2

SOFT-GRPO TESTING under Soft-Thinking Reasoning Paradigm Maximum response length 32768 tokens

- Sampling temperature 1.0

Gumbel-temperature τg 0.1 (top-p, top-k) (0.95, 5)

### E. Supplementary of Experiments

Table 9. Experiments on the token efficiency for baselines and the proposed SofT-GRPO. #Token values in Table represent the number of tokens across all queries, and the #Token c values represent the number of tokens across correct queries.

|Dataset|AIME2024<br><br>|AIME2025<br><br>|AMC23|MATH-500|GSM8K<br><br>|Average|
|---|---|---|---|---|---|---|
|Metrics|#Token #Token c<br><br>|#Token #Token c<br><br>|#Token #Token c<br><br>|#Token #Token c<br><br>|#Token #Token c<br><br>|#Token #Token c<br><br>|

DeepSeek-R1-Distill-Qwen-1.5B Base LLM Discrete-Token CoT Reasoning Pattern

No-Finetune 16241.6 14997.8 16416.3 13448.8 10052.2 9394.2 5616.6 5368.1 1839.3 1772.5 10033.2 8996.3 + GRPO 8927.6 8535.6 8039.4 6414.2 5001.3 4672.8 3106.1 2958.5 1417.1 1370.6 5298.3 4790.3

Soft-Thinking Reasoning Pattern

No-Finetune 17857.8 16191.3 17569.4 14582.4 11269.9 10482.0 4015.9 3888.3 1699.3 1649.9 10482.5 9358.8 + GRPO 9383.2 8934.5 8007.2 7325.8 5203.7 4894.4 3233.6 3131.7 1385.5 1349.9 5442.7 5127.2 + SofT-GRPO 11039.6 10756.1 10519.6 7831.3 5900.2 5630.4 3549.5 3399.2 1577.6 1542.5 6517.3 5831.9

LLaMA-3.2-3B-Instruct Base LLM Discrete-Token CoT Reasoning Pattern

No-Finetune 5010.5 3334.1 4297.6 2430.9 2790.7 2398.1 1850.1 1494.2 236.6 224.6 2837.1 1976.4 + GRPO 8991.8 6368.9 8443.8 11088.3 4998.1 3978.5 3334.8 2463.0 502.0 453.6 5254.1 4870.5

Soft-Thinking Reasoning Pattern

No-Finetune 4859.2 2727.7 4934.8 3735.2 3259.9 2624.0 2086.1 1494.1 243.5 227.8 3076.7 2161.8 + GRPO 9749.0 7627.8 8498.9 10041.4 5389.9 4987.3 3701.9 2764.6 541.9 505.8 5576.3 5185.4 + SofT-GRPO 829.7 862.9 893.1 879.6 911.5 927.9 632.8 575.7 294.6 292.9 712.3 707.8

DeepSeek-R1-Distill-Qwen-7B Base LLM Discrete-Token CoT Reasoning Pattern

No-Finetune 13120.7 11511.4 14347.7 11750.7 6346.0 5987.1 3998.9 3939.0 1061.1 1032.0 7774.9 6844.0 + GRPO 7795.9 7116.1 8003.5 7369.8 4050.9 3719.7 2473.3 2405.5 1146.7 1118.7 4694.0 4346.0

Soft-Thinking Reasoning Pattern

No-Finetune 13017.4 11888.6 14116.8 11507.4 6346.7 6043.6 3947.4 3858.6 996.3 962.2 7684.9 6852.1 + GRPO 7464.9 6531.4 8291.6 7160.4 3931.7 3736.7 2473.2 2423.0 1117.0 1104.5 4655.7 4191.2 + SofT-GRPO 8035.8 7556.8 8381.7 7873.7 4008.0 3843.5 2630.2 2583.5 1293.2 1276.4 4869.8 4626.8

#### E.1. Supplementary of Token Efficiency

As shown in Section 4.6, besides the performance, the token efficiency of LLMs is also an important metric. In this section, we compare the token efficiency of baselines and SofT-GRPO in Table 9. Compared to No-Finetune variants, SofT-GRPO can demonstrate a clear refinement in both the token efficiency across all queries and the token efficiency across correct queries. Compared to the discrete-token GRPO, SofT-GRPO will not cause severe token improvement. Specifically, we observe a severe reduction in the thinking length of the LLaMA-3.2-3B-Instruct model. As shown in Figure 4, unlike GRPO, SofT-GRPO maintains and even enhances the token efficiency compared to the base LLM when training progresses, demonstrating the effectiveness of SofT-GRPO in reducing computational consumption.

2000

AverageNumberofTokensinResponses

Discrete-token GRPO

SofT-GRPO

1800

1600

1400

1200

1000

800

600

0 100 200 300 400 500 600

GRPO Steps

Figure 4. Token consumption curve on LLaMA-3.2-3B-Instruct Base LLM during training.

#### E.2. Experiments under Different Temperatures

In Section 4, we adopt the setting of τ = 0.6 for discrete-token CoT. To investigate whether other temperature settings will break our observations in our main experiments, similar to Yue et al. (2025), we try temperatures from the collection of τ = 0.6, τ = 0.8, τ = 1.0, τ = 1.2, and τ = 1.4.

As shown in Figure 5, we conduct experiments on the 1.5B LLMs (i.e., DeepSeek-R1-Distill-Qwen-1.5B Base LLM) for their average Pass@K accuracies across five numerical reasoning benchmarks (i.e., AIME2024, AIME2025, AMC23, MATH-500, GSM8K), where the proposed SofT-GRPO can demonstrate outstanding results compared to GRPO and No-Finetune variants with various temperatures, from Pass@1 to Pass@32.

85

80

75

70

Accuracy(%)

65

60

55

- No-Finetune temperature=0.6

- No-Finetune temperature=0.8

- No-Finetune temperature=1.0

- No-Finetune temperature=1.2

- No-Finetune temperature=1.4

- GRPO temperature=0.6

- GRPO temperature=0.8

- GRPO temperature=1.0

- GRPO temperature=1.2

- GRPO temperature=1.4

50

SofT-GRPO

45

@1 @2 @4 @8 @16 @32 Pass@k

Figure 5. Running discrete-token CoT methods (GRPO and No-finetune) with more temperature options on DeepSeek-R1-Distill-Qwen1.5B Base LLM. Pass@K represents the pass rate within at most k runs, and Pass@1 is additionally averaged from 32 runs. Experiments are run on the five datasets in Table 1 for the average.

#### E.3. Supplementary Ablation on Hyper-Parameters

0.025

1e 5

- SofT-GRPO with top-p=0.95, g=0.1

- SofT-GRPO with top-p=1.0, g=0.1

AverageKLDivergencebetweenPPOPolicies

AverageKLDivergencetoReferenceModel

4

SofT-GRPO with top-p=0.95, g=0.25

0.020

2

0.015

0

0.010

2

- SofT-GRPO with top-p=0.95, g=0.1

- SofT-GRPO with top-p=1.0, g=0.1

0.005

4

SofT-GRPO with top-p=0.95, g=0.25

0.000

0 100 200 300 400 500 600

0 100 200 300 400 500 600

GRPO Steps

GRPO Steps

(a) Training KL Divergence curve between πθref and πθ

(b) Training Proximal Policy Optimization (PPO) (Schulman et al.,

2017) KL Divergence curve between πθold and πθ Figure 6. KL Divergence curves in SofT-GRPO over DeepSeek-R1-Distill-Qwen-1.5B Base LLM.

In this subsection, we further investigate the hyperparameter settings. As briefly shown in Section 5.2, adopting a higher top-p or τg will cause collapses in training. We attribute these collapses to the case that some soft-thinking tokens may become incomprehensible for LLMs.

As shown in Figure 6(a), the variants (varying top-p to 1.0 or varying τg to 0.25) will demonstrate higher divergence between the fixed pre-trained πθ

and the current policy, which can be an indicator of the inputs outside the pre-trained embedding

ref

space. Recently, Liu et al. (2025a); Qi et al. (2025b) provides excellent insight into the collapse situation in the RLVR fine-tuning. When collapse is caused by a precision issue, they observe a very high KL divergence between πθ

and πθ (Refer to Figure

old

- 3 in (Qi et al., 2025b), 10−3 even higher). However, in the variants shown in Figure 6(b), we find that their KL divergence in PPO policies is less than 10−5, indicating that the variants of SofT-GRPO are less likely to have precision issues.

#### E.4. Pass@K up to K=1024

Table 1 and Figure 5 show the Pass@K for K at most 64. In Yue et al. (2025), they conduct experiments for K up to 1024. We adopt this setting. Confined by limited computational resources, we implement the proposed SofT-GRPO over DeepSeek-R1-Distill-Qwen-1.5B for 1024 runs, as well as the GRPO and No-Finetune baselines with the discrete-token CoT reasoning pattern.

As shown in Figure 7, we calculate Pass@16 to Pass@1024 on AIME2024 and AMC 23. As a difference from the result in Table 1, we obtain Pass@K for each K runs instead of the first K runs. For example, when calculating Pass@32, we divide the 1024 runs into 32 groups, calculating their Pass@K value and averaging. In Table 1 instead, we calculate the Pass@K at first 32 runs.

Figure 7 obtains the same result pattern compared to the results in (Yue et al., 2025), where GRPO will outperform the No-finetune version when K is small, but will be left behind when K increases. SofT-GRPO can consistently outperform the GRPO and no-finetune counterparts on all K (from 16 to 1024).

Averaged Pass@k (every chunk) on AIME2024

AveragedPass@K(calculatedonceeveryKsamples)

Averaged Pass@k (every chunk) on AMC23

AveragedPass@K(calculatedonceeveryKsamples)

84

DeepSeek-R1-Distill-Qwen-1.5B

- 96

- 97

- 98

- 99

- 100

+ GRPO

82

+ SofT-GRPO

80

78

76

74

72

DeepSeek-R1-Distill-Qwen-1.5B

+ GRPO

+ SofT-GRPO

70

16 32 64 128 256 512 1024

16 32 64 128 256 512 1024

K

K

(a) Pass@16 to Pass@1024 on AIME2024

(b) Pass@16 to Pass@1024 on AMC23

Figure 7. Pass@16 to Pass@1024 on AIME2024 and AMC 23. The fluctuation in Figure (a) occurs because AIME only has 30 instances, and SofT-GRPO correctly solved three out of four (1024/256) problems when K=256. In Figure (b), all algorithms ultimately achieved complete correctness. Consistent with this, we checked CoT to prove that the correct results did not come from random guessing.

#### E.5. P-values for Significance on Pass@K

Besides the results in Table 1 and Figure 5, in this section, we use results from 1024 runs to demonstrate the significance test for Pass@16 and Pass@32 (dividing 64 groups for Pass@16 from the 1024 runs and 32 groups of Pass@32). We calculated the t-test for the one-sided hypothesis test (See Figure 7, the average result of SofT-GRPO outperforms the GRPO and No-Finetune baselines), and taking 0.05 as the threshold, the results in Figure 8 and 9 show that in Pass@16 with 64 runs, and Pass@32 with 32 runs:

- • SofT-GRPO can significantly outperform GRPO on AIME2024 with Pass@16 (0.0067), AMC23 with Pass@32 (0.022), with quite small values (0.13) on both AIME2024 with Pass@32 and AMC23 with Pass@16.
- • SofT-GRPO can get quite small p-values (0.081 and 0.054) on AMC23 with Pass@16 and Pass@32.

Pass@16 block matrix p-values (aime2024)

1.0

[Figure 32]

|[Figure 33]| |
|---|---|
| | |
| | |
| | |
| | |

No-Finetune

0.8

0.6

p-value

0.051

GRPO

0.4

0.2

0.2 0.0067

SofT-GRPO

0.0

No-Finetune GRPO SofT-GRPO

(a) P-scores of Pass@16 over SofT-GRPO, GRPO, and No-finetune on AIME2024

Pass@32 block matrix p-values (aime2024)

1.0

[Figure 34]

|[Figure 35]| |
|---|---|
| | |
| | |
| | |
| | |

No-Finetune

0.8

0.6

p-value

0.18

GRPO

0.4

0.2

0.41 0.13

SofT-GRPO

0.0

No-Finetune GRPO SofT-GRPO

(b) P-scores of Pass@32 over SofT-GRPO, GRPO, and No-finetune on AIME2024

Figure 8. Significance test on AIME2024.

Pass@16 block matrix p-values (amc23)

1.0

[Figure 36]

|[Figure 37]| |
|---|---|
| | |
| | |
| | |
| | |

No-Finetune

0.8

0.6

p-value

0.39

GRPO

0.4

0.2

0.081 0.13

SofT-GRPO

0.0

No-Finetune GRPO SofT-GRPO

(a) P-scores of Pass@16 over SofT-GRPO, GRPO, and No-finetune on AMC23

Pass@32 block matrix p-values (amc23)

1.0

[Figure 38]

|[Figure 39]| |
|---|---|
| | |
| | |
| | |
| | |

No-Finetune

0.8

0.6

p-value

0.34

GRPO

0.4

0.2

0.054 0.022

SofT-GRPO

0.0

No-Finetune GRPO SofT-GRPO

(b) P-scores of Pass@32 over SofT-GRPO, GRPO, and No-finetune on AMC23

Figure 9. Significance test on AMC23.

### F. Discussion & Analysis on Results

In this section, we will provide a detailed analysis of the improvement in Pass@K. Yue et al. (2025) first validates that RLVR techniques in discrete-token GRPO will not incentivize reasoning capacity in LLMs beyond the base model, demonstrating equal or even inferior Pass@K values when k improves. So, Recent studies (Walder & Karkhanis, 2025; Mahdavi et al., 2025; Peng et al., 2025) try to develop RL methods that optimize the Pass@K accuracy directly. Among them, Walder & Karkhanis (2025); Chow et al. (2024); Mahdavi et al. (2025); Hu et al. (2026) focus on reallocating the reward within the group, encouraging rewards of negative discrete-token CoTs in groups with high Pass@K value. Peng et al. (2025) try to improve the model’s Pass@K performance by avoiding the over-concentration on the top-1 selection within the token set.

In SimKO, at high-entropy (forking) tokens, positive (correct) samples are rewarded not only on the sampled token but spread across the top-K most probable tokens, promoting output diversity and better Pass@K. For negative (incorrect) samples, SimKO applies stronger penalties to the top-1 token and lighter penalties to others, which prevents the output probabilities from collapsing onto a single choice. This entropy-aware and asymmetric strategy encourages exploration, outperforming traditional RL methods that only focus on the sampled token.

In the proposed SofT-GRPO, we have the policy update in the soft-thinking part (t-th token) as:

 

 

|T |

[−gi′ + log pi − exp(−gi′ + log pi)] − [−ϵi − exp(−ϵi)]

∇θL(θ) = ∇θ

i=1

|T |

∇θ (log pi − exp(−gi′ + log pi))

=

i=1

|T |

(∇θ log pi − exp(−gi′ + log pi)∇θ log pi)

=

i=1

|T |

(1 − exp(−gi′ + log pi))∇θ log pi

=

i=1

|T |

(1 − pi · exp(−gi′))∇θ log pi.

=

i=1

This means that instead of updating the policy only for the selected token, in SofT-GRPO, it inherently highlights: (1). Higher importance component (gi′) in the soft tokens, (2). lower prob (pi). For example, for positive instances, SofT-GRPO inherently improves the probability of the main component in each soft-tokens and the low-probability components. For Negative instances, SofT-GRPO inherently gives penalties to the probability of the main component in each soft-tokens and the low-probability components.

Figure 10 verifies this conclusion: the entropy of token probability distributions will converge to nearly 0 with discrete-token GRPO, but keep stable and even improve with SofT-GRPO.

0.8

EntropyoftheDistributionpoverTokens

0.7

0.6

0.5

0.4

0.3

0.2

Discrete-token GRPO

0.1

SofT-GRPO

0.0

0 100 200 300 400 500 600

GRPO Steps

Figure 10. Entropy curve on DeepSeek-R1-Distill-Qwen-7B during training.

### G. Baselines & Datasets & Licenses

In our experiments, we evaluate and compare the following model baselines and datasets. For each, we detail the official website and usage license.

#### G.1. Baselines

We mainly include No-Finetune base LLMs, Discrete-Token GRPO, and the method in Butt et al. (2025) (noted Soft Token) as baselines.

Base LLMs This paper includes DeepSeek-R1-Distill-Qwen-1.5B, LLaMA-3.2-3B-Instruct, and DeepSeek-R1-DistillQwen-7B as base LLMs.

Discrete-Token GRPO We utilize the latest verl (Sheng et al., 2024) https://github.com/volcengine/verl/ tree/main and vLLM rollout to implement discrete-token GRPO with default parameters.

Butt et al. (2025) Soft Token Due to the requirement of passing d-dimensional inputs sˆ between the rollout workers and the verl policy optimization workers, implementing this algorithm requires a high amount of communication between the rollout workers and the policy update workers. So, we only report the results in (Butt et al., 2025) for comparison in Table 4.

#### G.2. Datasets

This paper covers five in-domain numerical reasoning datasets (i.e., AIME2024, AIME2025, AMC23, MATH-500, and GSM8K (Cobbe et al., 2021)), one out-of-domain scientific reasoning dataset GPQA-Diamond (Rein et al., 2024), and two out-of-domain code reasoning datasets (i.e., HumanEval and MBPP). These datasets are provided in https://github.

com/eric-ai-lab/Soft-Thinking.

#### G.3. Inference Framework

The inference framework of our implementation is built on SGLang (Zheng et al., 2024), maximizing efficiency via continuous batching, RadixAttention for KV cache reuse, and compressed finite state machines for faster structured output decoding. In the standard flow, the scheduler flattens discrete token inputs into continuous tensors for GPU execution; generated tokens are returned to update the Radix tree state for subsequent steps.

SGLang’s optional overlap scheduler pipelines execution by issuing the next batch ahead via negative integer slot addresses. Compared to Zhang et al. (2025b), we enabled this for Soft-Thinking by storing generated soft probabilities and indices in VRAM-allocated future maps. A custom kernel resolves the slot addresses to fetch these states just-in-time. This design should be faster by eliminating PCIe bottlenecks, allowing non-blocking scheduling with data kept entirely on the GPU.

###### G.4. Licenses For Base LLM, Dataset, and frameworks, we list their Licenses in Table 10.

Table 10. A summary of licenses.

Resources Type License URL DeepSeek-R1-Distill-Qwen-1.5B Base LLM MIT License https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B LLaMA-3.2-3B-Instruct Base LLM Llama 3.2 License https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct DeepSeek-R1-Distill-Qwen-7B Base LLM MIT License https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B verl RL-framework Apache-2.0 license https://github.com/volcengine/verl verl-0.4.x RL-framework Apache-2.0 license https://github.com/volcengine/verl/tree/v0.4.x SGLang for soft-thinking Inference-framework MIT License https://github.com/eric-ai-lab/Soft-Thinking AIME2024, AIME2025, GSM8K Dataset MIT License https://github.com/eric-ai-lab/Soft-Thinking AMC23, MATH-500 Dataset Available Online https://github.com/eric-ai-lab/Soft-Thinking HumanEval, GPQA-Diamond Dataset MIT License https://github.com/eric-ai-lab/Soft-Thinking MBPP Dataset Apache-2.0 license https://github.com/eric-ai-lab/Soft-Thinking

### H. Visualization

In this section, in Figure 11 and Figure 12, we provide visualization examples for the outcome of SofT-GRPO on two examples (we show yi numbers that are larger than 0.01 in soft-tokens). Our observations are as follows:

- • Compared to Figure 4 in Zhang et al. (2025b), SofT-GRPO preserves the multiply-path thinking pattern, thus providing better soft-thinking reasoning performances.
- • As discussed in Wu et al. (2025a), SofT-GRPO outcomes can observe possible latent search trees. See the 288th soft-token in Figure 11, it blends two discrete tokens with contrary meanings in its top-2 (No and Yeah, probably with totally different embeddings), representing a more abstract concept than any language token. Moreover, to check whether this pattern has already existed in No-Finetune LLM, we searched their output on GSM8K (a total of 1319 instances), where we could not find this probability distribution. So, we can recognize that this probability distribution is highly possible to emerge after SofT-GRPO.
- • Similarly to Figure 4 in Zhang et al. (2025b), SofT-GRPO can preserve interpretability; the contents in the red box (Okay, so I need to...) can be a good explanation for the path of soft-thinking reasoning.

Top-K Decoding Visualization (GSM8K Instance 4) Soft-Thinking

|Okay<br><br>0.96<br><br>,<br><br>1.00<br><br>so<br><br>1.00<br><br>I<br><br>0.91<br><br>need<br><br>1.00<br><br>to<br><br>1.00 1.00<br><br>out<br><br>1.00<br><br>how<br><br>1.00<br><br>many<br><br>1.00 0.72<br><br>James<br><br>1.00<br><br>runs<br><br>1.00<br><br>in<br><br>1.00<br><br>a<br><br>1.00<br><br>week<br><br>1.00<br><br>.<br><br>0.89<br><br>He<br><br>0.51<br><br>says<br><br>0.99<br><br>he<br><br>1.00 1.00<br><br>to<br><br>1.00<br><br>run<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0.04<br><br>James<br><br>0.09<br><br>total<br><br>0.28<br><br>just<br><br>0.11<br><br>Let<br><br>0.48| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

Answering

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23

|1.00<br><br>3<br><br>1.00<br><br>s<br><br>1.00<br><br>prints<br><br>1.00<br><br>,<br><br>1.00<br><br>and<br><br>1.00<br><br>each<br><br>1.00 1.00<br><br>is<br><br>1.00 1.00<br><br>6<br><br>1.00<br><br>0<br>1.00 1.00<br><br><br>.<br><br>1.00<br><br>Hmm<br><br>1.00<br><br>,<br><br>1.00<br><br>let<br><br>1.00<br><br>'s<br><br>1.00<br><br>break<br><br>1.00<br><br>this<br><br>1.00<br><br>down<br><br>1.00<br><br>step<br><br>1.00<br><br>by<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | |

24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46

|step<br><br>1.00<br><br>.<br><br>1.00<br><br>First<br><br>1.00<br><br>,<br><br>1.00<br><br>James<br><br>1.00<br><br>runs<br><br>0.97 1.00<br><br>3<br><br>1.00<br><br>s<br><br>1.00<br><br>prints<br><br>1.00<br><br>a<br><br>0.66<br><br>week<br><br>1.00<br><br>.<br><br>1.00<br><br>Each<br><br>1.00 1.00<br><br>is<br><br>1.00 0.86<br><br>6<br><br>1.00<br><br>0<br>1.00 1.00<br><br><br>.<br><br>1.00<br><br>So<br><br>1.00<br><br>,<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|does<br><br>0.03<br><br>every<br><br>0.34<br><br>a<br><br>0.14| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69

|to<br><br>0.98<br><br>find<br><br>1.00<br><br>the<br><br>1.00<br><br>total<br><br>1.00<br><br>distance<br><br>1.00<br><br>he<br><br>1.00<br><br>runs<br><br>1.00<br><br>in<br><br>1.00<br><br>a<br><br>1.00<br><br>week<br><br>1.00<br><br>,<br><br>1.00<br><br>I<br><br>1.00<br><br>need<br><br>1.00<br><br>to<br><br>1.00<br><br>multiply<br><br>1.00<br><br>the<br><br>1.00 1.00<br><br>of<br><br>1.00<br><br>s<br><br>1.00<br><br>prints<br><br>1.00<br><br>by<br><br>1.00<br><br>the<br><br>1.00<br><br>distance<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|that<br><br>0.02| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92

|of<br><br>1.00<br><br>each<br><br>1.00 1.00<br><br>.<br><br>1.00<br><br>That<br><br>1.00<br><br>makes<br><br>1.00<br><br>sense<br><br>1.00 1.00<br><br>if<br><br>1.00<br><br>he<br><br>1.00<br><br>does<br><br>1.00<br><br>the<br><br>1.00<br><br>same<br><br>1.00<br><br>distance<br><br>1.00<br><br>each<br><br>1.00<br><br>time<br><br>1.00<br><br>,<br><br>1.00<br><br>the<br><br>0.96<br><br>total<br><br>1.00<br><br>distance<br><br>1.00<br><br>is<br><br>1.00<br><br>just<br><br>1.00<br><br>the<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0.04| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115

|sum<br><br>1.00<br><br>of<br><br>1.00<br><br>each<br><br>0.99 1.00<br><br>.<br><br>1.00<br><br>So<br><br>0.99<br><br>,<br><br>1.00<br><br>math<br><br>1.00<br><br>em<br><br>1.00 1.00<br><br>,<br><br>1.00<br><br>that<br><br>1.00<br><br>would<br><br>1.00<br><br>be<br><br>1.00<br><br>:<br><br>1.00<br><br>Total<br><br>1.00 1.00<br><br>=<br><br>1.00 1.00<br><br>of<br><br>1.00<br><br>s<br><br>1.00<br><br>prints<br><br>1.00<br><br>×<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0.01| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

|[Figure 40]| |
|---|---|
| | |
| | |
| | |
| | |
| | |

116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138

|Distance<br><br>1.00<br><br>per<br><br>1.00 1.00 1.00<br><br>Pl<br><br>1.00<br><br>ugging<br><br>1.00<br><br>in<br><br>1.00<br><br>the<br><br>1.00 1.00<br><br>:<br><br>1.00<br><br>Total<br><br>1.00 1.00<br><br>=<br><br>1.00 1.00<br><br>3<br><br>1.00<br><br>×<br><br>1.00 1.00<br><br>6<br><br>1.00<br><br>0<br>1.00 1.00<br><br><br>Let<br><br>1.00<br><br>me<br><br>1.00 0.97| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|do<br><br>0.03| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

0.9

139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

|that<br><br>1.00<br><br>.<br><br>1.00 1.00<br><br>3<br><br>1.00<br><br>multiplied<br><br>0.99<br><br>by<br><br>1.00 1.00<br><br>6<br><br>1.00<br><br>0<br>1.00<br><br><br>.<br><br>0.98<br><br>Hmm<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>3<br><br>1.00<br><br>times<br><br>1.00 1.00<br><br>6<br><br>1.00<br><br>0<br>1.00<br><br><br>is<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>8<br><br>1.00<br><br>0<br>1.00<br>| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|...<br><br>0.02| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184

0.8

|.<br><br>1.00<br><br>So<br><br>0.92<br><br>,<br><br>1.00<br><br>is<br><br>0.93<br><br>that<br><br>1.00<br><br>it<br><br>0.98<br><br>?<br><br>0.92<br><br>Wait<br><br>1.00<br><br>,<br><br>1.00<br><br>let<br><br>1.00<br><br>me<br><br>1.00 1.00<br><br>-check<br><br>1.00<br><br>.<br><br>1.00 0.91<br><br>6<br><br>1.00<br><br>0<br>1.00 1.00<br><br><br>per<br><br>1.00 1.00<br><br>,<br><br>1.00<br><br>times<br><br>1.00 1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Wait<br><br>0.08 0.07<br><br>right<br><br>0.01<br><br>?<br><br>0.08<br><br>If<br><br>0.09| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207

|3<br><br>1.00<br><br>s<br><br>1.00<br><br>prints<br><br>1.00<br><br>.<br><br>0.72<br><br>Yeah<br><br>0.99<br><br>,<br><br>1.00<br><br>that<br><br>0.98 1.00<br><br>be<br><br>1.00 0.91<br><br>1<br><br>1.00<br><br>8<br><br>1.00<br><br>0<br>1.00 1.00<br><br><br>total<br><br>1.00<br><br>.<br><br>1.00<br><br>I<br><br>1.00<br><br>can<br><br>0.83<br><br>also<br><br>1.00<br><br>think<br><br>1.00<br><br>of<br><br>1.00<br><br>it<br><br>1.00<br><br>as<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|:<br><br>0.28 0.02<br><br>straightforward<br><br>0.09<br><br>don<br><br>0.17| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230

0.7

|0.99<br><br>6<br><br>1.00<br><br>0<br>1.00<br><br><br>+<br><br>1.00 1.00<br><br>6<br><br>1.00<br><br>0<br>1.00<br><br><br>+<br><br>1.00 1.00<br><br>6<br><br>1.00<br><br>0<br>1.00<br><br><br>.<br><br>1.00<br><br>That<br><br>1.00<br><br>adds<br><br>0.50 1.00<br><br>1<br><br>1.00<br><br>8<br><br>1.00<br><br>0<br>1.00<br><br><br>as<br><br>1.00<br><br>well<br><br>1.00<br><br>.<br><br>1.00<br><br>So<br><br>1.00<br><br>,<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0.01<br><br>'s<br><br>0.50| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

Probability

231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253

|James<br><br>1.00<br><br>runs<br><br>1.00 1.00<br><br>1<br><br>0.99<br><br>8<br><br>1.00<br><br>0<br>1.00 1.00<br><br><br>in<br><br>1.00<br><br>a<br><br>1.00<br><br>week<br><br>1.00<br><br>.<br><br>1.00<br><br>That<br><br>1.00<br><br>seems<br><br>1.00<br><br>right<br><br>1.00 0.74 0.80<br><br>3<br><br>0.99<br><br>s<br><br>0.98<br><br>prints<br><br>1.00<br><br>,<br><br>1.00<br><br>each<br><br>1.00 0.62<br><br>6<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|.<br><br>0.26<br><br>each<br><br>0.20<br><br>6<br><br>0.01<br><br>times<br><br>0.02<br><br>being<br><br>0.38| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

0.6

254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276

|0<br>1.00 0.99<br><br><br>,<br><br>1.00<br><br>so<br><br>1.00 0.55<br><br>3<br><br>1.00<br><br>times<br><br>1.00 1.00<br><br>6<br><br>1.00<br><br>0<br>1.00<br><br><br>.<br><br>0.51<br><br>No<br><br>0.92<br><br>,<br><br>1.00<br><br>wait<br><br>1.00<br><br>,<br><br>1.00<br><br>I<br><br>0.96<br><br>just<br><br>1.00<br><br>want<br><br>1.00<br><br>to<br><br>1.00<br><br>make<br><br>1.00<br><br>sure<br><br>1.00<br><br>I<br><br>1.00<br><br>didn<br><br>0.56| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0.45<br><br>is<br><br>0.49<br><br>Yeah<br><br>0.08<br><br>hold<br><br>0.04<br><br>'m<br><br>0.44| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299

|'t<br><br>1.00<br><br>mis<br><br>0.94<br><br>read<br><br>1.00<br><br>the<br><br>1.00<br><br>question<br><br>1.00<br><br>.<br><br>1.00<br><br>It<br><br>1.00<br><br>says<br><br>1.00 1.00<br><br>3<br><br>1.00<br><br>s<br><br>1.00<br><br>prints<br><br>1.00<br><br>a<br><br>1.00<br><br>week<br><br>1.00<br><br>,<br><br>1.00<br><br>each<br><br>1.00 1.00<br><br>6<br><br>1.00<br><br>0<br>1.00 1.00<br><br><br>.<br><br>1.00<br><br>So<br><br>1.00<br><br>,<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0.06| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

0.5

300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322

|1.00<br><br>3<br><br>1.00<br><br>s<br><br>1.00<br><br>prints<br><br>1.00<br><br>,<br><br>1.00<br><br>so<br><br>0.62 1.00<br><br>3<br><br>1.00<br><br>times<br><br>1.00 1.00<br><br>6<br><br>1.00<br><br>0<br>1.00<br><br><br>is<br><br>1.00 0.89<br><br>1<br><br>1.00<br><br>8<br><br>1.00<br><br>0<br>1.00<br><br><br>.<br><br>1.00<br><br>I<br><br>1.00<br><br>think<br><br>1.00<br><br>that<br><br>1.00<br><br>'s<br><br>1.00<br><br>solid<br><br>0.96| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|each<br><br>0.38 0.11 0.04| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345

|.<br><br>1.00<br><br>I<br><br>1.00<br><br>don<br><br>1.00<br><br>'t<br><br>1.00<br><br>think<br><br>1.00<br><br>I<br><br>1.00<br><br>made<br><br>0.95<br><br>a<br><br>1.00 1.00<br><br>here<br><br>0.96<br><br>.<br><br>1.00<br><br>So<br><br>0.77<br><br>,<br><br>1.00<br><br>the<br><br>0.92<br><br>final<br><br>1.00 1.00 0.83<br><br>be<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>8<br><br>1.00<br><br>0<br>1.00 1.00<br>| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|in<br><br>0.01<br><br>'ve<br><br>0.05<br><br>there<br><br>0.03<br><br>It<br><br>0.23 0.08<br><br>is<br><br>0.17| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

0.4

346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368

|.<br><br>0.97<br><br>**<br><br>1.00<br><br>Final<br><br>1.00 1.00<br><br>**<br><br>1.00<br><br>James<br><br>0.51<br><br>total<br><br>1.00 1.00<br><br>James<br><br>1.00<br><br>runs<br><br>1.00<br><br>a<br><br>1.00<br><br>week<br><br>1.00<br><br>is<br><br>1.00<br><br>\<br><br>1.00<br><br>boxed<br><br>1.00<br><br>{<br><br>1.00<br><br>1<br><br>1.00<br><br>8<br><br>1.00<br><br>0<br>1.00<br><br><br>}.<br><br>1.00 1.00 1.00<br><br>James<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | |

369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391

|1.00<br><br>to<br><br>1.00<br><br>run<br><br>1.00 1.00<br><br>3<br><br>1.00<br><br>s<br><br>1.00<br><br>prints<br><br>1.00<br><br>a<br><br>1.00<br><br>week<br><br>1.00<br><br>,<br><br>1.00<br><br>and<br><br>1.00<br><br>each<br><br>1.00 1.00<br><br>is<br><br>1.00 1.00<br><br>6<br><br>1.00<br><br>0<br>1.00 1.00<br><br><br>.<br><br>1.00<br><br>To<br><br>1.00<br><br>find<br><br>1.00<br><br>the<br><br>1.00<br><br>total<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | |

392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414

|distance<br><br>1.00<br><br>he<br><br>1.00<br><br>runs<br><br>1.00<br><br>in<br><br>1.00<br><br>a<br><br>1.00<br><br>week<br><br>1.00<br><br>,<br><br>1.00<br><br>we<br><br>1.00<br><br>multiply<br><br>1.00<br><br>the<br><br>1.00 1.00<br><br>of<br><br>1.00<br><br>s<br><br>1.00<br><br>prints<br><br>1.00<br><br>by<br><br>1.00<br><br>the<br><br>1.00<br><br>distance<br><br>1.00<br><br>of<br><br>1.00<br><br>each<br><br>1.00 1.00<br><br>:<br><br>1.00<br><br>\<br><br>1.00<br><br>[<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | |

415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437

|\<br><br>1.00<br><br>text<br><br>1.00<br><br>{<br><br>1.00<br><br>Total<br><br>1.00 1.00<br><br>}<br><br>1.00<br><br>=<br><br>1.00<br><br>\<br><br>1.00<br><br>text<br><br>1.00<br><br>{<br><br>1.00<br><br>Number<br><br>1.00<br><br>of<br><br>1.00<br><br>s<br><br>1.00<br><br>prints<br><br>1.00<br><br>}<br><br>1.00<br><br>\<br><br>1.00<br><br>times<br><br>1.00<br><br>\<br><br>1.00<br><br>text<br><br>1.00<br><br>{<br><br>1.00 1.00<br><br>per<br><br>1.00 1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | |

438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460

|}<br><br>1.00<br><br>\<br><br>1.00<br><br>]<br><br>1.00<br><br>Pl<br><br>1.00<br><br>ugging<br><br>1.00<br><br>in<br><br>1.00<br><br>the<br><br>1.00 1.00<br><br>:<br><br>1.00<br><br>\<br><br>1.00<br><br>[<br><br>1.00<br><br>\<br><br>1.00<br><br>text<br><br>1.00<br><br>{<br><br>1.00<br><br>Total<br><br>1.00 1.00<br><br>}<br><br>1.00<br><br>=<br><br>1.00 1.00<br><br>3<br><br>1.00<br><br>\<br><br>1.00<br><br>times<br><br>1.00 1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | |

461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483

|6<br><br>1.00<br><br>0<br>1.00<br><br><br>=<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>8<br><br>1.00<br><br>0<br>1.00 1.00<br><br><br>\<br><br>1.00<br><br>]<br><br>1.00<br><br>Thus<br><br>1.00<br><br>,<br><br>1.00<br><br>the<br><br>1.00<br><br>total<br><br>1.00 1.00<br><br>James<br><br>1.00<br><br>runs<br><br>1.00<br><br>a<br><br>1.00<br><br>week<br><br>1.00<br><br>is<br><br>1.00<br><br>\<br><br>1.00<br><br>boxed<br><br>1.00<br><br>{<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | |

484 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506

|1<br><br>1.00<br><br>8<br><br>1.00<br><br>0<br>1.00<br><br><br>}.<br><br>1.00<br><br><|end of sentence|><br><br>1.00| | | | |
|---|---|---|---|---|
| | | | | |

507 508 509 510 511

Figure 11. Visualization: DeepSeek-R1-Distill-Qwen-1.5B + SofT-GRPO on the 4-th instance of GSM8K.

Top-K Decoding Visualization (MATH-500 Instance 4) Soft-Thinking

Answering

|Alright<br><br>0.65<br><br>,<br><br>1.00<br><br>so<br><br>1.00<br><br>I<br><br>1.00<br><br>need<br><br>1.00<br><br>to<br><br>1.00<br><br>figure<br><br>1.00<br><br>out<br><br>1.00<br><br>how<br><br>1.00<br><br>many<br><br>1.00<br><br>positive<br><br>1.00<br><br>whole<br><br>1.00<br><br>-number<br><br>1.00<br><br>div<br><br>1.00<br><br>isors<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>has<br><br>1.00<br><br>.<br><br>1.00<br><br>Hmm<br><br>1.00<br><br>,<br><br>1.00<br><br>div<br><br>1.00<br><br>isors<br><br>1.00<br><br>,<br><br>1.00<br><br>right<br><br>1.00<br><br>,<br><br>1.00<br><br>factors<br><br>1.00<br><br>of<br><br>1.00<br><br>a<br><br>1.00<br><br>number<br><br>1.00<br><br>,<br><br>1.00<br><br>that<br><br>1.00<br><br>'s<br><br>1.00<br><br>what<br><br>0.92<br><br>I<br><br>1.00<br><br>remember<br><br>1.00<br><br>from<br><br>0.96<br><br>math<br><br>1.00<br><br>class<br><br>1.00<br><br>.<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Okay<br><br>0.35<br><br>the<br><br>0.08<br><br>.<br><br>0.04| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42

|I<br><br>0.94<br><br>think<br><br>1.00<br><br>it<br><br>0.75<br><br>means<br><br>1.00<br><br>numbers<br><br>1.00<br><br>that<br><br>1.00<br><br>divide<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>without<br><br>1.00<br><br>leaving<br><br>1.00<br><br>a<br><br>1.00<br><br>remainder<br><br>1.00<br><br>.<br><br>1.00<br><br>So<br><br>1.00<br><br>,<br><br>1.00<br><br>to<br><br>0.99<br><br>find<br><br>1.00<br><br>the<br><br>1.00<br><br>number<br><br>1.00<br><br>of<br><br>1.00<br><br>positive<br><br>0.72<br><br>div<br><br>1.00<br><br>isors<br><br>1.00<br><br>,<br><br>1.00<br><br>I<br><br>1.00<br><br>probably<br><br>1.00<br><br>need<br><br>1.00<br><br>to<br><br>1.00<br><br>factor<br><br>1.00<br><br>ize<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>into<br><br>1.00<br><br>its<br><br>1.00<br><br>prime<br><br>1.00<br><br>factors<br><br>1.00<br><br>first<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|So<br><br>0.06<br><br>the<br><br>0.25<br><br>div<br><br>0.28| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84

|,<br><br>0.99<br><br>and<br><br>1.00<br><br>then<br><br>1.00<br><br>use<br><br>1.00<br><br>that<br><br>0.73<br><br>to<br><br>1.00<br><br>determine<br><br>0.96<br><br>how<br><br>0.99<br><br>many<br><br>1.00<br><br>div<br><br>1.00<br><br>isors<br><br>1.00<br><br>there<br><br>1.00<br><br>are<br><br>1.00<br><br>.<br><br>0.96<br><br>Let<br><br>1.00<br><br>me<br><br>1.00<br><br>start<br><br>1.00<br><br>by<br><br>1.00<br><br>recalling<br><br>1.00<br><br>how<br><br>1.00<br><br>to<br><br>1.00<br><br>factor<br><br>1.00<br><br>a<br><br>1.00<br><br>number<br><br>1.00<br><br>into<br><br>1.00<br><br>primes<br><br>1.00<br><br>.<br><br>1.00<br><br>I<br><br>1.00<br><br>think<br><br>1.00<br><br>it<br><br>1.00<br><br>'s<br><br>1.00<br><br>called<br><br>1.00<br><br>prime<br><br>1.00<br><br>factor<br><br>1.00<br><br>ization<br><br>1.00<br><br>.<br><br>1.00<br><br>So<br><br>1.00<br><br>,<br><br>1.00<br><br>I<br><br>0.94<br><br>need<br><br>1.00<br><br>to<br><br>1.00<br><br>break<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|some<br><br>0.27<br><br>calculate<br><br>0.04<br><br>.<br><br>0.04 0.06| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126

|down<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>into<br><br>1.00<br><br>primes<br><br>1.00<br><br>that<br><br>0.70<br><br>multiply<br><br>1.00<br><br>together<br><br>1.00<br><br>to<br><br>1.00<br><br>give<br><br>0.71 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>.<br><br>1.00<br><br>Starting<br><br>1.00<br><br>with<br><br>1.00<br><br>the<br><br>1.00<br><br>smallest<br><br>1.00<br><br>prime<br><br>1.00<br><br>number<br><br>0.74<br><br>,<br><br>1.00<br><br>which<br><br>1.00<br><br>is<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>.<br><br>1.00<br><br>Is<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>divisible<br><br>0.76<br><br>by<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>?<br><br>1.00<br><br>Yes<br><br>0.97<br><br>,<br><br>1.00<br><br>because<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|.<br><br>0.30<br><br>get<br><br>0.29<br><br>,<br><br>0.26<br><br>even<br><br>0.24<br><br>Yeah<br><br>0.03| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168

|it<br><br>1.00<br><br>'s<br><br>1.00<br><br>an<br><br>1.00<br><br>even<br><br>1.00<br><br>number<br><br>1.00<br><br>.<br><br>1.00<br><br>So<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>divided<br><br>1.00<br><br>by<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>is<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>8<br><br>1.00<br><br>.<br><br>1.00<br><br>Okay<br><br>0.99<br><br>,<br><br>1.00<br><br>so<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>is<br><br>1.00<br><br>a<br><br>1.00<br><br>prime<br><br>1.00<br><br>factor<br><br>1.00<br><br>,<br><br>0.77<br><br>and<br><br>1.00<br><br>now<br><br>1.00<br><br>I<br><br>1.00<br><br>need<br><br>1.00<br><br>to<br><br>1.00<br><br>factor<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>8<br><br>1.00<br><br>.<br><br>1.00<br><br>Again<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|.<br><br>0.02<br><br>So<br><br>0.01<br><br>.<br><br>0.21| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210

|,<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>8<br><br>1.00<br><br>is<br><br>1.00<br><br>even<br><br>1.00<br><br>,<br><br>1.00<br><br>so<br><br>1.00<br><br>I<br><br>1.00<br><br>can<br><br>1.00<br><br>divide<br><br>1.00<br><br>by<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>once<br><br>1.00<br><br>more<br><br>1.00<br><br>.<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>8<br><br>1.00<br><br>divided<br><br>1.00<br><br>by<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>is<br><br>1.00 1.00<br><br>4<br><br>1.00<br><br>9<br><br>1.00<br><br>.<br><br>1.00<br><br>Now<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>4<br><br>1.00<br><br>9<br><br>1.00<br><br>is<br><br>1.00<br><br>...<br><br>1.00<br><br>hmm<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>times<br><br>1.00 1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

|[Figure 41]| |
|---|---|
| | |
| | |
| | |
| | |
| | |

211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252

|7<br><br>1.00<br><br>is<br><br>1.00 1.00<br><br>4<br><br>1.00<br><br>9<br><br>1.00<br><br>,<br><br>1.00<br><br>right<br><br>1.00<br><br>?<br><br>1.00<br><br>So<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>4<br><br>1.00<br><br>9<br><br>1.00<br><br>is<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>squared<br><br>1.00<br><br>,<br><br>0.96<br><br>so<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>is<br><br>1.00<br><br>another<br><br>0.97<br><br>prime<br><br>1.00<br><br>factor<br><br>1.00<br><br>.<br><br>0.98<br><br>So<br><br>1.00<br><br>,<br><br>1.00<br><br>putting<br><br>1.00<br><br>it<br><br>1.00<br><br>all<br><br>1.00<br><br>together<br><br>1.00<br><br>,<br><br>1.00<br><br>the<br><br>1.00<br><br>prime<br><br>1.00<br><br>factors<br><br>1.00<br><br>of<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>are<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|.<br><br>0.04<br><br>a<br><br>0.02<br><br>,<br><br>0.02| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.9

253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294

|1.00<br><br>2<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>,<br><br>1.00<br><br>and<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>.<br><br>0.99<br><br>Wait<br><br>1.00<br><br>,<br><br>1.00<br><br>let<br><br>1.00<br><br>me<br><br>1.00<br><br>write<br><br>1.00<br><br>that<br><br>1.00<br><br>out<br><br>0.99<br><br>:<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>=<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>×<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>×<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>×<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>.<br><br>1.00<br><br>So<br><br>0.97<br><br>,<br><br>1.00<br><br>in<br><br>1.00<br><br>terms<br><br>0.98| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|.<br><br>0.01<br><br>Alternatively<br><br>0.03<br><br>exponential<br><br>0.02| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336

|of<br><br>1.00<br><br>ex<br><br>1.00<br><br>ponents<br><br>1.00<br><br>,<br><br>1.00<br><br>that<br><br>1.00<br><br>would<br><br>1.00<br><br>be<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>2<br><br>1.00<br><br>×<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>2<br><br>1.00<br><br>.<br><br>1.00<br><br>So<br><br>1.00<br><br>,<br><br>1.00<br><br>the<br><br>0.96<br><br>prime<br><br>1.00<br><br>factor<br><br>1.00<br><br>ization<br><br>1.00<br><br>is<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>squared<br><br>0.93<br><br>times<br><br>0.95 1.00<br><br>7<br><br>1.00<br><br>squared<br><br>1.00<br><br>.<br><br>1.00<br><br>I<br><br>1.00<br><br>remember<br><br>1.00<br><br>that<br><br>0.89<br><br>to<br><br>1.00<br><br>find<br><br>1.00<br><br>the<br><br>1.00<br><br>number<br><br>1.00<br><br>of<br><br>1.00<br><br>div<br><br>1.00<br><br>isors<br><br>1.00<br><br>,<br><br>1.00<br><br>we<br><br>0.96| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|each<br><br>0.04<br><br>2<br><br>0.07<br><br>multiplied<br><br>0.05<br><br>from<br><br>0.10<br><br>you<br><br>0.04| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.8

337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378

|use<br><br>1.00<br><br>the<br><br>1.00<br><br>ex<br><br>1.00<br><br>ponents<br><br>1.00<br><br>in<br><br>0.96<br><br>the<br><br>1.00<br><br>prime<br><br>1.00<br><br>factor<br><br>1.00<br><br>ization<br><br>1.00<br><br>.<br><br>1.00<br><br>The<br><br>0.98<br><br>formula<br><br>1.00<br><br>,<br><br>0.96<br><br>if<br><br>0.99<br><br>I<br><br>1.00<br><br>recall<br><br>1.00<br><br>correctly<br><br>1.00<br><br>,<br><br>1.00<br><br>is<br><br>1.00<br><br>to<br><br>1.00<br><br>add<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>to<br><br>1.00<br><br>each<br><br>1.00<br><br>of<br><br>1.00<br><br>the<br><br>1.00<br><br>ex<br><br>1.00<br><br>ponents<br><br>1.00<br><br>and<br><br>1.00<br><br>then<br><br>1.00<br><br>multiply<br><br>1.00<br><br>them<br><br>1.00<br><br>together<br><br>1.00<br><br>.<br><br>1.00<br><br>So<br><br>1.00<br><br>,<br><br>1.00<br><br>for<br><br>1.00<br><br>each<br><br>0.95<br><br>prime<br><br>1.00<br><br>,<br><br>1.00<br><br>we<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|of<br><br>0.04<br><br>For<br><br>0.02<br><br>is<br><br>0.04<br><br>a<br><br>0.05| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420

|take<br><br>1.00<br><br>its<br><br>1.00<br><br>exponent<br><br>1.00<br><br>,<br><br>1.00<br><br>add<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>,<br><br>1.00<br><br>and<br><br>1.00<br><br>multiply<br><br>1.00<br><br>.<br><br>1.00<br><br>So<br><br>1.00<br><br>,<br><br>1.00<br><br>for<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>2<br><br>1.00<br><br>,<br><br>1.00<br><br>the<br><br>1.00<br><br>exponent<br><br>1.00<br><br>is<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>,<br><br>1.00<br><br>so<br><br>1.00<br><br>we<br><br>1.00<br><br>add<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>to<br><br>1.00<br><br>get<br><br>1.00 1.00<br><br>3<br><br>1.00<br><br>.<br><br>1.00<br><br>For<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>2<br><br>1.00<br><br>,<br><br>1.00<br><br>exponent<br><br>1.00<br><br>is<br><br>1.00 1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462

|2<br><br>1.00<br><br>,<br><br>1.00<br><br>so<br><br>0.91<br><br>add<br><br>0.77 1.00<br><br>1<br><br>1.00<br><br>to<br><br>1.00<br><br>get<br><br>1.00 1.00<br><br>3<br><br>1.00<br><br>.<br><br>1.00<br><br>Then<br><br>1.00<br><br>,<br><br>1.00<br><br>multiply<br><br>1.00 0.98<br><br>3<br><br>1.00<br><br>and<br><br>1.00 1.00<br><br>3<br><br>1.00<br><br>together<br><br>1.00<br><br>,<br><br>1.00<br><br>which<br><br>1.00<br><br>is<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>.<br><br>1.00<br><br>Therefore<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>has<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>positive<br><br>1.00<br><br>whole<br><br>0.92<br><br>-number<br><br>1.00<br><br>div<br><br>1.00<br><br>isors<br><br>1.00<br><br>.<br><br>1.00<br><br>Wait<br><br>0.46| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|But<br><br>0.03<br><br>we<br><br>0.03<br><br>Let<br><br>0.14<br><br>add<br><br>0.09<br><br>again<br><br>0.21<br><br>them<br><br>0.02<br><br>div<br><br>0.08<br><br>Hmm<br><br>0.37| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.7

Probability

463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504

|,<br><br>1.00<br><br>let<br><br>1.00<br><br>me<br><br>1.00<br><br>double<br><br>0.93<br><br>-check<br><br>1.00<br><br>that<br><br>1.00<br><br>.<br><br>0.56<br><br>Maybe<br><br>1.00<br><br>by<br><br>1.00<br><br>listing<br><br>1.00<br><br>them<br><br>1.00<br><br>out<br><br>0.99<br><br>to<br><br>0.99<br><br>ensure<br><br>1.00<br><br>I<br><br>1.00<br><br>haven<br><br>0.78<br><br>'t<br><br>1.00<br><br>missed<br><br>1.00<br><br>anything<br><br>1.00<br><br>.<br><br>1.00<br><br>The<br><br>0.95<br><br>positive<br><br>0.99<br><br>div<br><br>1.00<br><br>isors<br><br>1.00<br><br>of<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>should<br><br>1.00<br><br>be<br><br>1.00<br><br>all<br><br>1.00<br><br>the<br><br>1.00<br><br>numbers<br><br>1.00<br><br>that<br><br>1.00<br><br>can<br><br>1.00<br><br>divide<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>without<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|'m<br><br>0.03<br><br>verify<br><br>0.07<br><br>to<br><br>0.44<br><br>all<br><br>0.01<br><br>didn<br><br>0.20<br><br>Starting<br><br>0.05<br><br>div<br><br>0.01| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.6

505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546

|leaving<br><br>1.00<br><br>a<br><br>1.00<br><br>remainder<br><br>1.00<br><br>.<br><br>1.00<br><br>So<br><br>1.00<br><br>,<br><br>1.00<br><br>starting<br><br>1.00<br><br>from<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>,<br><br>0.97<br><br>then<br><br>0.98 1.00<br><br>2<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>4<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>4<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>8<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>4<br><br>1.00<br><br>9<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>8<br><br>1.00<br><br>,<br><br>1.00<br><br>and<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|and<br><br>0.03<br><br>since<br><br>0.02| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588

|.<br><br>1.00<br><br>Let<br><br>1.00<br><br>me<br><br>1.00<br><br>count<br><br>1.00<br><br>these<br><br>1.00<br><br>:<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>4<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>4<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>8<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>4<br><br>1.00<br><br>9<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>8<br><br>1.00<br><br>,<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>.<br><br>1.00<br><br>That<br><br>1.00<br><br>'s<br><br>1.00 1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630

|9<br><br>1.00<br><br>numbers<br><br>1.00<br><br>,<br><br>1.00<br><br>so<br><br>1.00<br><br>yeah<br><br>1.00<br><br>,<br><br>1.00<br><br>that<br><br>1.00<br><br>matches<br><br>1.00<br><br>what<br><br>0.44<br><br>earlier<br><br>1.00<br><br>calculation<br><br>0.67<br><br>.<br><br>1.00<br><br>Alternatively<br><br>1.00<br><br>,<br><br>1.00<br><br>I<br><br>1.00<br><br>could<br><br>1.00<br><br>think<br><br>0.89<br><br>of<br><br>1.00<br><br>it<br><br>1.00<br><br>as<br><br>1.00<br><br>each<br><br>1.00<br><br>combination<br><br>1.00<br><br>of<br><br>1.00<br><br>the<br><br>1.00<br><br>ex<br><br>1.00<br><br>ponents<br><br>1.00<br><br>of<br><br>1.00<br><br>the<br><br>1.00<br><br>prime<br><br>1.00<br><br>factors<br><br>1.00<br><br>.<br><br>1.00<br><br>So<br><br>1.00<br><br>,<br><br>1.00<br><br>for<br><br>1.00<br><br>prime<br><br>0.96 1.00<br><br>2<br><br>1.00<br><br>,<br><br>1.00<br><br>which<br><br>1.00<br><br>is<br><br>1.00<br><br>squared<br><br>1.00<br><br>,<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|my<br><br>0.26<br><br>the<br><br>0.30<br><br>result<br><br>0.33<br><br>use<br><br>0.11 0.04| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.5

631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672

|the<br><br>1.00<br><br>ex<br><br>1.00<br><br>ponents<br><br>1.00<br><br>can<br><br>1.00<br><br>be<br><br>1.00 1.00<br><br>0<br>1.00<br><br><br>,<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>,<br><br>1.00<br><br>or<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>.<br><br>1.00<br><br>For<br><br>1.00<br><br>prime<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>,<br><br>1.00<br><br>which<br><br>0.90<br><br>is<br><br>1.00<br><br>also<br><br>0.99<br><br>squared<br><br>1.00<br><br>,<br><br>1.00<br><br>the<br><br>1.00<br><br>ex<br><br>1.00<br><br>ponents<br><br>1.00<br><br>can<br><br>1.00<br><br>be<br><br>1.00 1.00<br><br>0<br>1.00<br><br><br>,<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>,<br><br>1.00<br><br>or<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>.<br><br>0.70<br><br>So<br><br>1.00<br><br>,<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|.<br><br>0.14<br><br>similarly<br><br>0.10<br><br>as<br><br>0.16| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.4

673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714

|the<br><br>1.00<br><br>number<br><br>1.00<br><br>of<br><br>1.00<br><br>combinations<br><br>1.00<br><br>would<br><br>0.93<br><br>be<br><br>1.00 0.98<br><br>3<br><br>1.00<br><br>(<br><br>1.00<br><br>for<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>)<br><br>1.00<br><br>multiplied<br><br>1.00<br><br>by<br><br>1.00 1.00<br><br>3<br><br>1.00<br><br>(<br><br>1.00<br><br>for<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>),<br><br>1.00<br><br>which<br><br>1.00<br><br>is<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>.<br><br>1.00<br><br>Each<br><br>1.00<br><br>combination<br><br>1.00<br><br>gives<br><br>1.00<br><br>a<br><br>1.00<br><br>unique<br><br>1.00<br><br>divisor<br><br>1.00<br><br>,<br><br>0.94<br><br>so<br><br>1.00<br><br>that<br><br>1.00<br><br>also<br><br>1.00<br><br>makes<br><br>0.99<br><br>sense<br><br>1.00<br><br>.<br><br>1.00<br><br>I<br><br>1.00<br><br>think<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|is<br><br>0.07<br><br>(<br><br>0.02<br><br>:<br><br>0.06| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756

|I<br><br>1.00<br><br>'m<br><br>1.00<br><br>confident<br><br>1.00<br><br>now<br><br>1.00<br><br>that<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>has<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>positive<br><br>1.00<br><br>div<br><br>1.00<br><br>isors<br><br>1.00<br><br>.<br><br>1.00<br><br>**<br><br>1.00<br><br>Final<br><br>1.00<br><br>Answer<br><br>1.00<br><br>**<br><br>1.00<br><br>The<br><br>1.00<br><br>number<br><br>1.00<br><br>of<br><br>1.00<br><br>positive<br><br>1.00<br><br>whole<br><br>1.00<br><br>-number<br><br>1.00<br><br>div<br><br>1.00<br><br>isors<br><br>1.00<br><br>of<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>is<br><br>1.00<br><br>\<br><br>1.00<br><br>boxed<br><br>1.00<br><br>{<br><br>1.00<br><br>9<br><br>1.00<br><br>}.<br><br>1.00<br><br></think><br><br>1.00 1.00<br><br>To<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798

|determine<br><br>1.00<br><br>how<br><br>1.00<br><br>many<br><br>1.00<br><br>positive<br><br>1.00<br><br>whole<br><br>1.00<br><br>-number<br><br>1.00<br><br>div<br><br>1.00<br><br>isors<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>has<br><br>1.00<br><br>,<br><br>1.00<br><br>we<br><br>1.00<br><br>start<br><br>1.00<br><br>by<br><br>1.00<br><br>finding<br><br>1.00<br><br>its<br><br>1.00<br><br>prime<br><br>1.00<br><br>factor<br><br>1.00<br><br>ization<br><br>1.00<br><br>.<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>is<br><br>1.00<br><br>an<br><br>1.00<br><br>even<br><br>1.00<br><br>number<br><br>1.00<br><br>,<br><br>1.00<br><br>so<br><br>1.00<br><br>we<br><br>1.00<br><br>divide<br><br>1.00<br><br>by<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>:<br><br>1.00<br><br>\[<br><br>1.00 1.00<br><br>1<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840

|9<br><br>1.00<br><br>6<br><br>1.00<br><br>\<br><br>1.00<br><br>div<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>=<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>8<br><br>1.00<br><br>\<br><br>1.00<br><br>]<br><br>1.00<br><br>9<br><br>1.00<br><br>8<br><br>1.00<br><br>is<br><br>1.00<br><br>also<br><br>1.00<br><br>even<br><br>1.00<br><br>,<br><br>1.00<br><br>so<br><br>1.00<br><br>we<br><br>1.00<br><br>divide<br><br>1.00<br><br>by<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>again<br><br>1.00<br><br>:<br><br>1.00<br><br>\[<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>8<br><br>1.00<br><br>\<br><br>1.00<br><br>div<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>=<br><br>1.00 1.00<br><br>4<br><br>1.00<br><br>9<br><br>1.00<br><br>\<br><br>1.00<br><br>]<br><br>1.00<br><br>4<br><br>0.97<br><br>9<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882

|is<br><br>1.00<br><br>not<br><br>1.00<br><br>even<br><br>1.00<br><br>,<br><br>1.00<br><br>but<br><br>1.00<br><br>it<br><br>1.00<br><br>is<br><br>1.00<br><br>a<br><br>0.93<br><br>perfect<br><br>1.00<br><br>square<br><br>1.00<br><br>of<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>:<br><br>1.00<br><br>\[<br><br>1.00 1.00<br><br>4<br><br>1.00<br><br>9<br><br>1.00<br><br>=<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>\<br><br>1.00<br><br>times<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>\<br><br>1.00<br><br>]<br><br>1.00<br><br>Thus<br><br>1.00<br><br>,<br><br>1.00<br><br>the<br><br>1.00<br><br>prime<br><br>1.00<br><br>factor<br><br>1.00<br><br>ization<br><br>1.00<br><br>of<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>is<br><br>1.00<br><br>:<br><br>1.00<br><br>\[<br><br>1.00 1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924

|1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>=<br><br>1.00 1.00<br><br>2<br><br>1.00<br><br>^<br><br>1.00<br><br>2<br><br>1.00<br><br>\<br><br>1.00<br><br>times<br><br>1.00 1.00<br><br>7<br><br>1.00<br><br>^<br><br>1.00<br><br>2<br><br>1.00<br><br>\<br><br>1.00<br><br>]<br><br>1.00<br><br>To<br><br>1.00<br><br>find<br><br>1.00<br><br>the<br><br>1.00<br><br>number<br><br>1.00<br><br>of<br><br>1.00<br><br>positive<br><br>1.00<br><br>div<br><br>1.00<br><br>isors<br><br>1.00<br><br>,<br><br>1.00<br><br>we<br><br>1.00<br><br>use<br><br>1.00<br><br>the<br><br>1.00<br><br>formula<br><br>1.00<br><br>for<br><br>0.64<br><br>the<br><br>1.00<br><br>number<br><br>1.00<br><br>of<br><br>1.00<br><br>div<br><br>1.00<br><br>isors<br><br>1.00<br><br>,<br><br>1.00<br><br>which<br><br>1.00<br><br>involves<br><br>1.00<br><br>adding<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>to<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966

|each<br><br>1.00<br><br>of<br><br>1.00<br><br>the<br><br>1.00<br><br>ex<br><br>1.00<br><br>ponents<br><br>1.00<br><br>in<br><br>1.00<br><br>the<br><br>1.00<br><br>prime<br><br>1.00<br><br>factor<br><br>1.00<br><br>ization<br><br>1.00<br><br>and<br><br>1.00<br><br>then<br><br>1.00<br><br>multiplying<br><br>1.00<br><br>the<br><br>1.00<br><br>results<br><br>1.00<br><br>:<br><br>1.00<br><br>\[<br><br>1.00<br><br>(<br><br>1.00<br><br>2<br><br>1.00<br><br>+<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>)(<br><br>1.00<br><br>2<br><br>1.00<br><br>+<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>)<br><br>1.00<br><br>=<br><br>1.00 1.00<br><br>3<br><br>1.00<br><br>\<br><br>1.00<br><br>times<br><br>1.00 1.00<br><br>3<br><br>1.00<br><br>=<br><br>1.00 1.00<br><br>9<br><br>1.00<br><br>\<br><br>1.00<br><br>]<br><br>1.00<br><br>Therefore<br><br>1.00<br><br>,<br><br>1.00| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008

|the<br><br>1.00<br><br>number<br><br>1.00<br><br>of<br><br>1.00<br><br>positive<br><br>1.00<br><br>whole<br><br>1.00<br><br>-number<br><br>1.00<br><br>div<br><br>1.00<br><br>isors<br><br>1.00<br><br>of<br><br>1.00 1.00<br><br>1<br><br>1.00<br><br>9<br><br>1.00<br><br>6<br><br>1.00<br><br>is<br><br>1.00<br><br>\<br><br>1.00<br><br>(\<br><br>1.00<br><br>boxed<br><br>1.00<br><br>{<br><br>1.00<br><br>9<br><br>1.00<br><br>}\<br><br>1.00<br><br>).<br><br>1.00<br><br><|end of sentence|><br><br>1.00| | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |

1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030

Figure 12. Visualization: DeepSeek-R1-Distill-Qwen-1.5B + SofT-GRPO on the 4-th instance of MATH-500.

