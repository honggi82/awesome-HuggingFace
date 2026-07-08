## arXiv:2510.08696v1[cs.LG]9Oct2025

Don’t Waste Mistakes: Leveraging Negative RL-Groups via Confidence Reweighting

Yunzhen Feng2,∗, Parag Jain1, Anthony Hartshorn1, Yaqi Duan2,†, Julia Kempe1,2,† 1Meta Superintelligence Labs, 2New York University

†Joint advising

Reinforcement learning with verifiable rewards (RLVR) has become a standard recipe for improving large language models (LLMs) on reasoning tasks, with Group Relative Policy Optimization (GRPO) widely used in practice. Yet GRPO wastes substantial compute on negative groups: groups in which no sampled response is correct yield zero advantage and thus no gradient. We ask whether negative groups can be leveraged without extra supervision. Starting from a maximum-likelihood (MLE) objective in reward modeling, we show that the MLE gradient is equivalent to a policy gradient for a modified value function. This value function adds a confidence-weighted penalty on incorrect responses, imposing larger penalties on more confident mistakes. We refer to this as Likelihood Estimation with Negative Samples (LENS). LENS modifies GRPO to assign non-zero, confidence-dependent rewards to incorrect generations, making negative groups informative and converting previously wasted samples into useful gradient updates. On the MATH benchmark with Llama-3.1-8B and Qwen-2.5-3B, the proposed variant consistently outperforms GRPO baseline, with significant gains on harder items. These results demonstrate a principled and practical way to “rescue” negative groups, improving efficiency and performance in RLVR.

Date: October 13, 2025 Correspondence: Yunzhen Feng at yf2231@nyu.edu

1 Introduction

Large language models (LLMs) fine-tuned with reinforcement learning and verifiable rewards (RLVR) (Shao et al., 2024; Guo et al., 2025) have shown strong gains on complex reasoning tasks, with algorithms such as Group Relative Policy Optimization (GRPO) (Shao et al., 2024; Guo et al., 2025) emerging as practical defaults. A persistent inefficiency, however, is how these methods handle negative groups—the generation group in which no sampled response is correct. In GRPO and its variants, such groups contribute zero advantage and therefore no gradient signal. This is especially common at the start of training and on harder reasoning problems, where negative groups can constitute a substantial fraction of compute, effectively wasting already-generated trajectories.

We therefore ask: can we learn from negative groups without additional supervision in a principled way? Our starting point is deliberately simple: to learn from negative groups, the natural approach is reward modeling that distinguishes correct from incorrect answers, optimized with maximum likelihood (MLE). From this likelihood perspective, the MLE gradient is equivalent to a policy gradient on a modified RLVR value function. The modified value adds a confidence-weighted penalty for incorrect responses: the more confident the model is in a wrong answer, the larger the penalty. Intuitively, it discourages overconfident failure modes, thereby encouraging exploration of lower-probability yet plausible alternatives.

This equivalence lets us modify GRPO directly. It yields a drop-in change in which incorrect generations receive non-zero, confidence-dependent rewards (i.e., lower rewards when confidence is higher). As a result, negative groups now provide informative advantage estimates, converting previously wasted samples into useful gradient updates and promoting exploration on hard negatives. We term this algorithm LENS: Likelihood Estimation with Negative Samples.

We evaluate LENS on mathematical reasoning using the MATH benchmark with Llama-3.1-8B-Instruct and

# Self-Normalized Tail Inequalities for Matrix Martingales

Self-Normalized Tail Inequalities for Matrix Martingales

Proof of inequality (4): By comparing derivatives, one can show that

then

#### Self-Normalized Tail Inequalities for Matrix Martingales

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧ September 12, 2025 Abstract

qA>

qa>Vt  11 a.

then

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧ September 12, 2025 Abstract

u2/2 1   u/3

eu   u   1 

for any 0 < u < 3.

t Vt  11 At & sup

then

##### Self-Normalized Tail Inequalities for Matrix Martingales

qA>

qa>Vt  11 a.

Setting u = (p2z + 1) 1 for z > 0 and performing routine algebra yields

then

then

qA>

qa>Vt  11 a.

t Vt  11 At & sup

a

then

qA>

qa>Vt  11 a.

Proof of inequality (4): By comparing derivatives, one can show that

u2/2 1   u/3 

1 4z2

. Hence,

qA>

qa>Vt  11 a.

t Vt  11 At & sup

t Vt  11 At & sup

qA>

qa>Vt  11 a.

a

t Vt  11 At & sup

u2/2 1   u/3

t Vt  11 At & sup

eu   u   1 

a

###### Self-Normalized Tail Inequalities for Matrix Martingales

for any 0 < u < 3.

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧ September 12, 2025 Abstract

⌘  

exp⇣ 1 p2z + 1

a

1 p2z + 1   1 

1 4z2

for any z > 0.

Setting u = (p2z + 1) 1 for z > 0 and performing routine algebra yields

###### Self-Normalized Tail Inequalities for Matrix Martingales

a

a

Answer Space O for a question q

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧ September 12, 2025 Abstract

###### Self-Normalized Tail Inequalities for Matrix Martingales

###### Self-Normalized Tail Inequalities for Matrix Martingales

Now let z = Rt1 pkW⌧k2 (or z = Rt1 p"/2). Then g✓

u2/2 1   u/3 

1 4z2

###### Self-Normalized Tail Inequalities for Matrix Martingales

Proof of inequality (4): By comparing derivatives, one can show that

. Hence,

###### Self-Normalized Tail Inequalities for Matrix Martingales

###### Self-Normalized Tail Inequalities for Matrix Martingales

###### Self-Normalized Tail Inequalities for Matrix Martingales

◆ =

⇢exp⇣ Rt p2kW⌧k2 + Rt

Rt p2kW⌧k2 + Rt   1 

⌘  

Answer Space O for a question q

t p2kW⌧k2 + Rt

1 R2

Answer Space O for a question q

###### for Matrix Martingales

###### Self-Normalized Tail Inequalities for Matrix Martingales

Answer Space O for a question q

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧ September 12, 2025 Abstract

⌘  

exp⇣ 1 p2z + 1

u2/2 1   u/3

Answer Space O for a question q

###### Self-Normalized Tail Inequalities for Matrix Martingales

###### Self-Normalized Tail Inequalities for Matrix Martingales

###### Self-Normalized Tail Inequalities for Matrix Martingales

1 p2z + 1   1 

1 4z2

eu   u   1 

t2 4kW⌧k2

for any 0 < u < 3.

Yaqi Duan⇧

Yaqi Duan⇧

⇧

for any z > 0.

Answer Space O for a question q

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧ September 12, 2025 Abstract



, which establishes inequality (4).

Setting u = (p2z + 1) 1 for z > 0 and performing routine algebra yields

Now let z = Rt1 pkW⌧k2 (or z = Rt1 p"/2). Then g✓

Yaqi Duan⇧

Yaqi Duan⇧

Yaqi Duan⇧

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧ September 12, 2025 Abstract

Yaqi Duan⇧

Correction Rate p? P

Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

Yaqi Duan⇧

◆ =

⇢exp⇣ Rt p2kW⌧k2 + Rt

Rt p2kW⌧k2 + Rt   1 

u2/2 1   u/3 

1 4z2

⌘  

###### Self-Normalized Tail Inequalities for Matrix Martingales

t p2kW⌧k2 + Rt

1 R2

. Hence,

⇧

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧ September 12, 2025 Abstract

⇥

⇤

Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

Yaqi Duan⇧

Yaqi Duan⇧

Correction Rate p? P

Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

Correction Rate p? P

Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

###### References

Correction Rate p? P

Stern School of Business, New York University⇧

Answer o is correct for question q

0 1

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

t2 4kW⌧k2

Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

⇥

⇤

Correction Rate p? P

⌘  

exp⇣ 1 p2z + 1

⇥

⇤

⇥

⇤



, which establishes inequality (4).

1 p2z + 1   1 

1 4z2

[1] J. Tropp. Freedman’s inequality for matrix martingales. 2011.

then

then

then

Correction Rate p? P

September 12, 2025

September 12, 2025

for any z > 0.

Answer o is correct for question q

0 1

Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

⇥

⇤

Answer o is correct for question q

0 1

Answer o is correct for question q

0 1

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧ September 12, 2025 Abstract

Now let z = Rt1 pkW⌧k2 (or z = Rt1 p"/2). Then g✓

E := ⇢k✓bt   ✓t?kVt   ",

 max(Vt)  min(Vt)  ` 

⇥

⇤

then

then

qA>

qA>

qA>

qa>Vt  11 a.

qa>Vt  11 a.

qa>Vt  11 a.

September 12, 2025

September 12, 2025

then

Answer o is correct for question q

0 1

September 12, 2025

September 12, 2025

then

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

t Vt  11 At & sup

t Vt  11 At & sup

t Vt  11 At & sup

Answer o is correct for question q

0 1

then

then

###### Abstract

###### Abstract

qA>

qA>

qa>Vt  11 a.

qa>Vt  11 a.

September 12, 2025

◆ =

⇢exp⇣ Rt p2kW⌧k2 + Rt

Rt p2kW⌧k2 + Rt   1 

qA>

qa>Vt  11 a.

qA>

qa>Vt  11 a.

⌘  

###### References

t p2kW⌧k2 + Rt

1 R2

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

t Vt  11 At & sup

t Vt  11 At & sup

t Vt  11 At & sup

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

then

then

qA>

qA>

qa>Vt  11 a.

qa>Vt  11 a.

a

a

a

###### Abstract

###### Abstract

September 12, 2025

September 12, 2025

t Vt  11 At & sup

h✓bt 1,ai +  tqa>Vt  11 a  h✓bt 1,Ati +  tqA>

###### Abstract

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

###### Abstract

a

t Vt  11 At

[1] J. Tropp. Freedman’s inequality for matrix martingales. 2011.

t Vt  11 At & sup

t Vt  11 At & sup

qA>

qA>

qa>Vt  11 a.

qa>Vt  11 a.

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

###### Abstract

a

a

t2 4kW⌧k2

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

a

D(q) Normalization

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.



, which establishes inequality (4).

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

D(q) Normalization

GRPO Reward = (

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

E := ⇢k✓bt   ✓t?kVt   ",

 max(Vt)  min(Vt)  ` 

t Vt  11 At & sup

t Vt  11 At & sup

quadratic variation.

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

###### Abstract

###### Abstract

a

a

then

 h✓bt 1,ai   h✓bt 1,Ati    tqa>Vt  11 a   O(1) If

 tqA>

t Vt  11 At    tqa>Vt  11 a +

D(q) Normalization

Answer Space O for a question q

qA>

qa>Vt  11 a.

D(q) Normalization

1 3 0 7

###### GRPO Reward = (

Answer Space O for a question q

Answer Space O for a question q

Answer Space O for a question q

D(q) Normalization

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

a

a

###### GRPO Reward = (

###### GRPO Reward = (

###### GRPO Reward = (

###### = (

t Vt  11 At & sup

1 3 0 7

D(q) Normalization

1 3 0 7

1 3 0 7

1 3 0 7

h✓bt 1,ai +  tqa>Vt  11 a  h✓bt 1,Ati +  tqA>

### GRPO Reward = (

Answer Space O for a question q

Answer Space O for a question q

1 3 0 7

Answer Space O for a question q

a

###### References

###### GRPO Reward = (

###### GRPO Reward = (

qa>Vt  11 a &  t 1 , then

t Vt  11 At

###### GRPO Reward = (

###### GRPO Reward = (

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

⇥

⇤

###### GRPO Reward = (

###### GRPO Reward = (

###### GRPO Reward = (

1 3 0 7

1 3 0 7

Answer Space O for a question q

Answer Space O for a question q

1 3 0 7

###### GRPO Reward = (

1 3 0 7

sup

Correction Rate P

1 3 0 7

Answer o is correct for question q

0 1

###### GRPO Reward = (

1 3 0 7

1 3 0 7

[1] J. Tropp. Freedman’s inequality for matrix martingales. 2011.

1 3 0 7

a

1 3 0 7

1 3 0 7

 h✓bt 1,ai   h✓bt 1,Ati    tqa>Vt  11 a   O(1) If

 tqA>

t Vt  11 At    tqa>Vt  11 a +

Answer Space O for a question q

Answer Space O for a question q

Answer Space O for a question q

Desirable Policy ⇡?

###### GRPO Reward = (

###### GRPO Reward = (

Desirable Policy ⇡?

E := ⇢k✓bt   ✓t?kVt   ",

 max(Vt)  min(Vt)  ` 

Correction Rate P Answer o is correct for question q 0 1

Correction Rate P Answer o is correct for question q 0 1

Correction Rate P Answer o is correct for question q 0 1

qA>

qa>Vt  11 a.

1 3 0 7

1 3 0 7

Desirable Policy ⇡?

t Vt  11 At & sup

⇥

⇤

###### GRPO Reward = (

Desirable Policy ⇡?

Correction Rate P Answer o is correct for question q 0 1

Correction Rate P Answer o is correct for question q 0 1

1 3 0 7

Correction Rate P

Desirable Policy ⇡?

qa>Vt  11 a &  t 1 , then

a

Answer o is correct for question q

0 1

⇡(o | q)

Probability ⇡(o | q)

Probability ⇡(o | q)

###### ⇡(o | q)

⇥

⇤

Desirable Policy ⇡?

sup

Correction Rate P Answer o is correct for question q 0 1

Correction Rate P Answer o is correct for question q 0 1

h✓bt 1,ai +  tqa>Vt  11 a  h✓bt 1,Ati +  tqA>

⇡(o | q)

Correction Rate P

Answer o is correct for question q

0 1

then

⇥

⇥

⇤

⇤

a

Probability ⇡(o | q)

Probability ⇡(o | q)

t Vt  11 At

Probability ⇡(o | q)

Answer Space O for a question q

Correction Rate P

Correction Rate P

qA>

qa>Vt  11 a.

Answer o is correct for question q

Answer o is correct for question q

0 1

0 1

Probability ⇡(o | q)

⇡(o | q)

qA>

qa>Vt  11 a.

Probability ⇡(o | q)

t Vt  11 At & sup

⇡(o | q)

then

⇡(o | q)

⇡(o | q)

 h✓bt 1,ai   h✓bt 1,Ati    tqa>Vt  11 a   O(1) If

 tqA>

t Vt  11 At    tqa>Vt  11 a +

t Vt  11 At & sup

8 ><

8 ><

8 ><

8 ><

###### Reward Modeling via MLE Policy Optimization

qA>

qa>Vt  11 a.

a

8

Probability ⇡(o | q)

Probability ⇡(o | q)

All zero ⇒ Everything discarded

a

1 3  

1 3  

1 3  

= > >

⇡(o | q)

1 3  

t Vt  11 At & sup

1 3  

⇡(o | q)

8 ><

8 ><

###### Reward Modeling via MLE Policy Optimization Reparameterization

###### Reward Modeling via MLE Policy Optimization Reparameterization

8 ><

a

Reward Modeling via MLE Policy Optimization

LENS Reward =

LENS Reward =

LENS Reward =

1 3  

1 3  

⇡(o | q) D(q)   ⇡(o | q)

⇡(o | q) D(q)   ⇡(o | q)

⇡(o | q) D(q)   ⇡(o | q)

⇡(o | q) (q)   ⇡(o | q)

Answer Space O for a question q

8 ><

8 ><

1 3  

qa>Vt  11 a &  t 1 , then

LENS Reward =

8 ><

Answer Space O for a question q

⇡(o | q) D(q)   ⇡(o | q)

>:

>:

>:

7

7

7

7

1 3  

sup

>:

Reward Modeling via MLE Policy Optimization Reparameterization

1 3  

Reward Modeling via MLE Policy Optimization Reparameterization

LENS Reward =

LENS Reward =

1 3  

⇡(o | q)

7

8 ><

LENS Reward =

⇡(o | q) D(q)   ⇡(o | q)

⇡(o | q) D(q)   ⇡(o | q)

8 ><

⇡(o | q) D(q)   ⇡(o | q)

8 ><

8 ><

a

8 ><

>:

>:

7

7

LENS Reward =

8

Answer Space O for a question q

7

1 3  

1 3  

:

LENS Reward =

1 3  

1 3  

⇡(o | q) D(q)   ⇡(o | q)

LENS Reward =

⇡(o | q) D(q)   ⇡(o | q)

⇡(o | q) D(q)   ⇡(o | q)

>:

qA>

qa>Vt  11 a.

⇥

⇤

>:

1 3  

7

>:

7

LENS Reward =

8 ><

Correction Rate P

7

Answer o is correct for question q

0 1

###### 1 Introduction

LENS Reward =

LENS Reward =

t Vt  11 At & sup

⇡(o | q) D(q)   ⇡(o | q)

8 ><

LENS Reward =

⇡(o | q) D(q)   ⇡(o | q)

⇡(o | q) D(q)   ⇡(o | q)

>:

⇡(o | q) D(q)   ⇡(o | q)

7

>:

>:

7

7

a

>:

###### 1 Introduction

r1? =  0.14   0.26   0.35   1.04   0.02   0.05   0.64   0.85   0.22   0.01   0.43

r1? =  0.   0.   0.   1.   0.   0.   0.   0.   0.   0.   0.

r1? =  0.14   0.26   0.35   1.04   0.02   0.05   0.64   0.85   0.22   0.01   0.43

⇥

⇤

7

1 3  

LENS Reward =

1 3  

⇡(o | q) D(q)   ⇡(o | q)

Correction Rate P

Answer o is correct for question q

0 1

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

>:

Answer Space O for a question q

r1? =  0.14   0.26   0.35   1.04   0.02   0.05   0.64   0.85   0.22   0.01   0.43

r1? =  0.14   0.26   0.35   1.04   0.02   0.05   0.64   0.85   0.22   0.01   0.43

r1? =  0.14   0.26   0.35   1.04   0.02   0.05   0.64   0.85   0.22   0.01   0.43

7

LENS Reward =

###### 1 Introduction

###### 1 Introduction

⇡(o | q) D(q)   ⇡(o | q)

r1? =  0.14   0.26   0.35   1.04   0.02   0.05   0.64   0.85   0.22   0.01   0.43

r1? =  0.14   0.26   0.35   1.04   0.02   0.05   0.64   0.85   0.22   0.01   0.43

LENS Reward =

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

8 ><

⇡(o | q) D(q)   ⇡(o | q)

>:

###### 1 Introduction

###### 1 Introduction

8

7

r1? =  0.14   0.26   0.35   1.04   0.02   0.05   0.64   0.85   0.22   0.01   0.43

r1? =  0.14   0.26   0.35   1.04   0.02   0.05   0.64   0.85   0.22   0.01   0.43

###### 1 Introduction

>:

7

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

1 3  

###### 1 Introduction

###### 1 Introduction

###### 1 Introduction

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Differences emerge

###### 1 Introduction

###### 1 Introduction

Low-confidence error ⇒ Light penalty

High-confidence error ⇒ Heavy penalty

#### 1 Introduction

⇒ Clear signal

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

###### 1 Introduction

###### 1 Introduction

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

### LENS Reward =

⇡(o | q) D(q)   ⇡(o | q)

mulated in terms of a deterministic

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

##### 1 Introduction

>:

Theorem (Matrix Freedman). Consider a matrix martingale {Y k : k = 0,1,2,...} whose values are self-adjoint matrices with dimension d, and let {Xk : k = 1,2,3,...} be the di↵erence sequence. Assume that the di↵erence sequence is uniformly bounded in the sense that

7

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

self-regularized

1 Introduction

Figure 1 Overview of our approach. Standard approaches like GRPO assign a uniform reward of 0 to all incorrect answers. This provides no learning signal, causing these samples to be discarded. Our method, LENS, is derived from reward modeling via Maximum Likelihood Estimation (MLE) and assigns non-zero, confidence-dependent rewards to incorrect responses. This creates a clear learning signal where differences emerge from the samples, converting previously discarded information into useful gradient updates.

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Theorem (Matrix Freedman). Consider a matrix martingale {Y k : k = 0,1,2,...} whose values are self-adjoint matrices with dimension d, and let {Xk : k = 1,2,3,...} be the di↵erence sequence. Assume that the di↵erence sequence is uniformly bounded in the sense that

Theorem (Matrix Freedman). Consider a matrix martingale {Y k : k = 0,1,2,...} whose values are self-adjoint matrices with dimension d, and let {Xk : k = 1,2,3,...} be the di↵erence sequence. Assume that the di↵erence sequence is uniformly bounded in the sense that

 max(Xk)  R almost surely for k = 1,2,3,....

1

1

1

Theorem (Matrix Freedman). Consider a matrix martingale {Y k : k = 0,1,2,...} whose values are self-adjoint matrices with dimension d, and let {Xk : k = 1,2,3,...} be the di↵erence sequence. Assume that the di↵erence sequence is uniformly bounded in the sense that

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

1

1

# 1 Introduction

1

1

1

1

Theorem (Matrix Freedman). Consider a matrix martingale {Y k : k = 0,1,2,...} whose values are self-adjoint matrices with dimension d, and let {Xk : k = 1,2,3,...} be the di↵erence sequence. Assume that the di↵erence sequence is uniformly bounded in the sense that

 max(Xk)  R almost surely for k = 1,2,3,....

1

1

9

 max(Xk)  R almost surely for k = 1,2,3,....

1

 max(Xk)  R almost surely for k = 1,2,3,....

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Qwen-2.5-3B-Base. In both settings, our GRPO variant consistently outperforms the GRPO baseline across all Pass@k metrics. Stratifying by difficulty, we find that gains are concentrated on the Levels 4-5 subsets (hard items), consistent with repurposed negative groups driving increased exploration for hard questions. We train on two distinct math training datasets to demonstrate the generality of our method.

1

9

1

Theorem (Matrix Freedman). Consider a matrix martingale {Y k : k = 0,1,2,...} whose values are self-adjoint matrices with dimension d, and let {Xk : k = 1,2,3,...} be the di↵erence sequence. Assume that the di↵erence sequence is uniformly bounded in the sense that

 max(Xk)  R almost surely for k = 1,2,3,....

Theorem (Matrix Freedman). Consider a matrix martingale {Y k : k = 0,1,2,...} whose values are self-adjoint matrices with dimension d, and let {Xk : k = 1,2,3,...} be the di↵erence sequence. Assume that the di↵erence sequence is uniformly bounded in the sense that

1

We summarize our contributions as follows:

9

9

- • We introduce a likelihood framework, Likelihood Estimation with Negative Samples (LENS), that explicitly connects reward modeling and policy optimization.
- • LENS yields a principled value function whose additional term penalizes overconfident incorrect answers, formalizing how negative-group signals should be used and calibrated within the objective.
- • We propose a GRPO variant that assigns non-zero, confidence-dependent rewards to incorrect generations, thereby leveraging negative groups rather than wasting them. It is plug-and-play with negligible computational overhead.
- • Empirical results support our algorithm’s effectiveness and show increased exploration, as reflected in Pass@k.

Theorem (Matrix Freedman). Consider a matrix martingale {Y k : k = 0,1,2,...} whose values are self-adjoint matrices with dimension d, and let {Xk : k = 1,2,3,...} be the di↵erence sequence. Assume that the di↵erence sequence is uniformly bounded in the sense that

 max(Xk)  R almost surely for k = 1,2,3,....

9

9

 max(Xk)  R almost surely for k = 1,2,3,....

9

9

9

9

9

1

9

9

9

9

1

9

9

Theorem (Matrix Freedman). Consider a matrix martingale {Y k : k = 0,1,2,...} whose values are self-adjoint matrices with dimension d, and let {Xk : k = 1,2,3,...} be the di↵erence sequence. Assume that the di↵erence sequence is uniformly bounded in the sense that

9

9

9

 max(Xk)  R almost surely for k = 1,2,3,....

1

###### 2 Related Work

 max(Xk)  R almost surely for k = 1,2,3,....

RLVR. Recent work has shown that reinforcement learning (RL) can effectively refine LLMs for reasoning. In RLVR, the LLM is treated as a policy that generates a chain-of-thought (CoT) reasoning process, and it receives a deterministic reward based on whether the final answer can be algorithmically verified. Recent works (Shao et al., 2024; Guo et al., 2025; Team et al., 2025) show that RLVR can elicit emergent reasoning behaviors and dramatically boost math and coding performance compared to the base model. Underlying most of these RLVR methods is the Group Relative Policy Optimization (GRPO) algorithm (Shao et al., 2024). GRPO is an efficient variant of Proximal Policy Optimization (PPO) (Schulman et al., 2017) that drops the value network and instead computes advantages from grouped outputs. In this way, with a group of all incorrect generations, the advantage is 0, and these groups do not contribute to the optimization. In this work, we try to make use of these negative groups.

1

Learning from negatives. Recent work has emphasized that negative samples are not merely noise but a useful

training signal in LLM reasoning. One direction explores asymmetric treatment of positives and negatives in REINFORCE-style training: Roux et al. (2025) introduce an asymmetric variant of importance sampling to speed up learning. Arnal et al. (2025) demonstrate that asymmetric REINFORCE, and in particular reducing the signal from negative samples, can be beneficial when data is off-policy. Lyu et al. (2025) propose to reweight positive and negative samples at the token level using a learned reward model combined with log-likelihood. Zhu et al. (2025) demonstrate that training only on negatives, assigning reward −1 to incorrect and 0 to correct answers, can outperform baselines on Pass@k for large k.

Another line of work argues that entirely wrong completions may still contain valuable sub-signals. Chen et al. (2025a) assign fractional rewards within all-negative groups, Yang et al. (2025) mine correct sub-steps from long chains of thought, and Li et al. (2024b) leverage negative rationales through a dual-LoRA distillation framework. These methods demonstrate that even within incorrect trajectories, certain steps are worth reinforcing, particularly in long reasoning traces where correct and incorrect steps alternate. A key drawback of these approaches is that evaluating intermediate reasoning steps is labor-intensive, and accurate automation remains underexplored. Several concurrent works also propose various approaches to address negative groups, which we summarize in Appendix A.

Our contribution is to provide a framework that stratifies reward signals within negative samples using only outcome rewards and probability, balancing computational efficiency with the benefits of learning from structured negatives.

- 3 Preliminaries and Motivation We start with background on policy optimization and the motivation for our method.

- 3.1 Language Model Reasoning as Policy Optimization

We begin with a basic setting: given a question q ∈ Q, a language model π is tasked with generating an answer o ∈ O. To evaluate correctness, we assume the existence of a reward function r⋆ : Q × O → {0,1}, which assigns 1 if the answer o is correct for the given question q, and 0 otherwise.

The ultimate goal of training the language model is to improve its accuracy rate. Formally, this corresponds to maximizing the expected reward:

maximizeπ J(π) := E[r⋆(q,o)], where q ∼ ξ, o ∼ π(· | q). (1)

Here ξ denotes the distribution of questions. Equation (1) is the central criterion: it asks us to design a policy π that maximizes the expected correctness of generated responses.

- 3.2 Motivation: Negative Groups in RLVR

In practice, Group Relative Policy Optimization (GRPO) has become a default algorithm for optimizing LLM reasoning ability for the objective in Equation (1). Concretely, for each verifiable question q, we draw a group of G candidates {oi}Gi=1 ∼ πθ

(· | q), obtain scalar rewards ri := r⋆(q,oi) ∈ {0,1}, and form zero-mean, unit-variance group advantages

old

ri − mean({rj}j∈[G]) std({rj}j∈[G])

. (2)

ri =

With outcome-only rewards, the same advantage Ai,t = ri is assigned to all tokens t in response oi. GRPO then maximizes a clipped PPO-style surrogate with an explicit per-token KL regularizer to a fixed reference πref:

|oi|

G

1 G

1 |oi|

JGRPO(πθ) = Eq,{o

i}

t=1

i=1

min ρi,t Ai,t, clip(ρi,t,1 − ϵ,1 + ϵ) Ai,t , (3)

where ρi,t := ππθ(oi,t|q,oi,<t)

θold(oi,t|q,oi,<t) is the correction for off-policy samples. We omit the KL divergence term following the common practice as β = 0.

GRPO is a practical policy-gradient method for LLMs because it computes advantages from group-relative statistics rather than a learned value function (critic). This makes it simple and robust for long-form reasoning, where sequences are long and rewards arrive only after a complete solution.

However, GRPO wastes substantial compute on negative groups. If an entire group is incorrect, i.e., all rewards {ri} are zero, the advantages collapse to zero, yielding no contribution to the policy gradient. Figure 2 shows the fraction of all-negative groups during training with group size G = 16: despite 16 generations per prompt, nearly 45% of groups are all-negative early in training, and about 35% remain even by the end. These groups consume substantial generation compute yet contribute no learning signal.

Negative Group Ratio of GRPO

Data points

0.55

Smooth curve

0.50

NegativeGroupRatio

0.45

0.40

0.35

0.30

0.25

0.20

0.15

###### 4 A Likelihood-Based Framework for Reasoning

0 200 400 600 800 1000 1200 1400

Training Step

Figure 2 Negative group ratio during GRPO training of Llama-3.1-8B-Instruct with MATH and Numina 1.5. G = 16.

We now seek to find a principled framework to use the negative groups. A direct route is reward modeling: train a model to discriminate correct from incorrect responses. We develop a likelihood-based formulation of reward modeling and show how it connects to policy optimization.

4.1 From Policy Learning to Reward Modeling

While our goal is to optimize the policy, the task becomes clearer when re-expressed through reward modeling. To illustrate this connection, we turn to a simple multiple-choice example.

Illustrative Example: Multiple-Choice Reasoning. Suppose a single question q comes with six possible answers: A,B,C,D,E,F. Out of these, only A and B are correct. We can think of an unknown ground-truth probability function

p⋆(q,o) = P Answer o is correct for question q .

For math problems, this function is deterministic: each answer is either correct (p⋆ = 1) or incorrect (p⋆ = 0) and p⋆ = r⋆. More generally, however, p⋆ could take fractional values in [0,1] to reflect varying confidence or partial correctness.

In this example, the desirable optimal policy π⋆ for Equation (1) is one that selects only from the correct options. For instance:

π⋆(A | q) = π⋆(B | q) = 21, π⋆(C | q) = ··· = π⋆(F | q) = 0.

This π⋆ randomly chooses between the correct answers A and B.1 This relationship can be expressed more generally as

p⋆(q,o) =

1 D(q)

π⋆(o | q), (4)

where D(q) is a normalizing factor defined by

−1

p⋆(q,o)

###### . (5)

D(q) =

o∈O

1Here we select an optimal policy that chooses uniformly at random among all correct answers. In more general settings we may have preferences over which correct answers to favor; for example, one might prefer shorter correct answers to longer ones. We extend the framework to incorporate a preference function, as discussed in Appendix C.

Proof of inequality (4): By comparing derivatives, one can show that

u2/2 1   u/3

eu   u   1 

for any 0 < u < 3.

Setting u = (p2z + 1) 1 for z > 0 and performing routine algebra yields

Proof of inequality (4): By comparing derivatives, one can show that

u2/2 1   u/3 

1 4z2

then

. Hence,

u2/2 1   u/3

eu   u   1 

for any 0 < u < 3.

qA>

###### qa>Vt  11 a.

then

then

Setting u = (p2z + 1) 1 for z > 0 and performing routine algebra yields

t Vt  11 At & sup

⌘  

exp⇣ 1 p2z + 1

1 p2z + 1   1 

1 4z2

qA>

qa>Vt  11 a.

qA>

qa>Vt  11 a.

for any z > 0.

a

t Vt  11 At & sup

u2/2 1   u/3 

1 4z2

t Vt  11 At & sup

Now let z = Rt1 pkW⌧k2 (or z = Rt1 p"/2). Then g✓

. Hence,

a

a

◆ =

⇢exp⇣ Rt p2kW⌧k2 + Rt

Rt p2kW⌧k2 + Rt   1 

⌘  

t p2kW⌧k2 + Rt

1 R2

⌘  

exp⇣ 1 p2z + 1

1 p2z + 1   1 

1 4z2

###### Answer Space O for a question q

for any z > 0.

then

Answer Space O for a question q

t2 4kW⌧k2

Answer Space O for a question q

Now let z = Rt1 pkW⌧k2 (or z = Rt1 p"/2). Then g✓



, which establishes inequality (4).

qA>

qa>Vt  11 a.

then

t Vt  11 At & sup

◆ =

⇢exp⇣ Rt p2kW⌧k2 + Rt

Rt p2kW⌧k2 + Rt   1 

qA>

qa>Vt  11 a.

⌘  

t p2kW⌧k2 + Rt

1 R2

t Vt  11 At & sup

a

Correction Rate p? P

a

Correction Rate p? P

###### References

⇥

⇤

t2 4kW⌧k2

Correction Rate p? P

⇥

⇤



, which establishes inequality (4).

Answer o is correct for question q

0 1

⇥

⇤

Answer o is correct for question q

0 1

Answer Space O for a question q

Answer Space O for a question q

[1] J. Tropp. Freedman’s inequality for matrix martingales. 2011.

Answer o is correct for question q

0 1

E := ⇢k✓bt   ✓t?kVt   ",

 max(Vt)  min(Vt)  ` 

then

then

then

qA>

qa>Vt  11

qA>

qa>Vt  11 a.

qA>

qa>Vt  11 a.

###### References

t Vt  11 At & sup

t Vt  11 At & sup

Correction Rate p? P

t Vt  11 At & sup

Correction Rate p? P

⇥

⇤

D(q) Normalization

a

a

###### D(q) Normalization

h✓bt 1,ai +  tqa>Vt  11 a  h✓bt 1,Ati +  tqA>

then

a

⇥

⇤

Answer o is correct for question q

0 1

[1] J. Tropp. Freedman’s inequality for matrix martingales. 2011.

###### D(q) Normalization

qA>

qa>Vt  11 a.

t Vt  11 At

Answer o is correct for question q

0 1

then

t Vt  11 At & sup

qA>

qa>Vt  11 a.

E := ⇢k✓bt   ✓t?kVt   ",

 max(Vt)  min(Vt)  ` 

a

t Vt  11 At & sup

Answer Space O for a question q

Answer Space O for a question q

Answer Space O for a question q

 h✓bt 1,ai   h✓bt 1,Ati    tqa>Vt  11 a   O(1) If

 tqA>

t Vt  11 At    tqa>Vt  11 a +

a

D(q) Normalization

Answeroiscorrectforquestionq ⇤ 01

?CorrectionRatep

Answer Space O for a question q

Desirable Policy ⇡?

Desirable Policy ⇡?

h✓bt 1,ai +  tqa>Vt  11 a  h✓bt 1,Ati +  tqA>

Answer Space O for a question q

Desirable Policy ⇡?

D(q) Normalization

then

t Vt  11 At

qa>Vt  11 a &  t 1 , then

Correction Rate ? ⇥

Correction Rate p? P

Correction Rate p? P

qA>

qa>Vt  11 a.

⇥

⇤

⇤

⇥

⇤

⇥

⇤

t Vt  11 At & sup

OAnswerSpaceforaquestionq

sup

q 1>aVa. t1

Correction Rate P

Desirable Policy ⇡?

Answer o is correct for question q

PolicyOptimization

RewardModelingviaMLE

Reparameterization

⇥

⇤

Answer is correct for question q

0 1

Answer o is correct for question q

0 1

Answer o is correct for question q

0 1

a

a

Correction Rate P

Answer o is correct for question q

0 1

 h✓bt 1,ai   h✓bt 1,Ati    tqa>Vt  11 a   O(1) If

 tqA>

t Vt  11 At    tqa>Vt  11 a +

qA>

qa>Vt  11 a.

D(q)Normalization

###### Reward Modeling via MLE Policy Optimization

?DesirablePolicy⇡

Desirable Policy ⇡?

Answer Space O for a question q

t Vt  11 At & sup

###### Reward Modeling via MLE Policy Optimization Reparameterization

###### Reward Modeling via MLE Policy Optimization Reparameterization

###### Reward Modeling via MLE Policy Optimization Reparameterization

qa>Vt  11 a &  t 1 , then

D(q) Normalization

q) Normalization

D(q) Normalization

a

& 1VAsuptt t1

⇥

⇤

a

sup

Correction Rate P

Answer o is correct for question q

0 1

9

a

Answer Space O for a question q

###### Reward Modeling via MLE Policy Optimization

qA>

qa>Vt  11 a.

Desirable Policy ⇡?

Desirable Policy ?

Desirable Policy ⇡?

t Vt  11 At & sup

8

a

q>A

Answer Space O for a question q

Figure 3 An optimal policy π⋆ is derived from reward probabilities p⋆ through normalization (see Equation (4)). This approach reframes the task of finding the best policy as a more straightforward statistical problem: learning a reward model from data.

8

⇥

P

Intuitively, D(q) ∈ (0,1] captures the difficulty of the question. If only one answer is correct, D(q) = 1, indicating a hard question. If multiple answers are correct, D(q) becomes smaller, signaling an easier question.

then

In practice, we do not have direct access to the full probability function p⋆. Instead, we observe data samples of the form (q,o,r), where r ∼ Bernoulli p⋆(q,o) . Reward modeling then fits a model pθ to these observations to approximate p⋆. Through the relation in Equation (4), we can recover one optimal policy π⋆. Therefore, policy learning reduces to the statistical task of estimating reward probabilities.

9

9

Maximum Likelihood Estimation (MLE) as the Learning Principle. Formally, suppose we are given an i.i.d. dataset D = {(qi,oi,ri)}ni=1. If we have an estimate of the difficulty D(qi) (as defined in Equation (5)), we can reparameterize the probability model as

9

9

1 D(q)

πθ(o | q), (6)

pθ(q,o) =

9

9

9

where πθ belongs to a parametric policy class. The straightforward way to solve pθ is through the maximum likelihood (equivalently, cross-entropy minimization) objective:

9

9

9

n

1 n

ri · log pθ(qi,oi) + (1 − ri) · log 1 − pθ(qi,oi) . (7)

minimizeθ L0(θ) = −

i=1

Plugging in the reparameterization yields the equivalent form:

|minimizeθ L(θ) = −<br><br>1 n<br><br>n<br><br>i=1<br><br>ri · log πθ(oi | qi) + (1 − ri) · log 1 −<br><br>πθ(oi | qi) D(qi)<br><br>.|
|---|

(8)

This formulation makes explicit the bridge between policy learning and reward modeling: by estimating p⋆, we implicitly learn a good policy πθ that maximizes accuracy.

- 4.2 Calibrating Policy Gradient via MLE.

We now turn to the algorithmic perspective: how can the maximum likelihood objective (8) guide policy gradient methods? Our first step is to analyze the gradient of the MLE loss. This is summarized in Theorem 1.

- Theorem 1. The gradient of the log-likelihood L(θ) with respect to the parameters θ is given by

n

πθ(oi | qi) D(q) − πθ(oi | qi) · ∇θ log πθ(oi | qi). (9)

1 n

∇θ L(θ) = −

ri − (1 − ri)

i=1

Comparison with Policy Gradient. For reference, the standard policy gradient expression for maximizing the accuracy objective in Equation (1) is

∇θ J(πθ) = E r · ∇θ log πθ(o | q) .

Classical algorithms such as REINFORCE, PPO, and GRPO are all built upon this form. In practice, the raw reward r is often replaced by an advantage estimate A to reduce variance. However, in GRPO, when all answers in a batch are incorrect (i.e., r = 0), the gradient contribution vanishes entirely (after centralization). This explains why negative groups are typically discarded in existing methods.

MLE Perspective. Theorem 1 sheds new light on this issue. The first term of the gradient, ri · ∇θ log πθ(oi | qi),

matches the standard policy gradient signal: positive samples (ri = 1) encourage the model to increase probability mass on correct answers.

But critically, the MLE gradient also contains an additional negative sample contribution: − (1 − ri)

πθ(oi | qi) D(qi) − πθ(oi | qi) · ∇θ log πθ(oi | qi).

Although typically smaller in scale, this term is non-negligible when only negative answers are observed, or when negative samples dominate the data. In other words, discarding negative groups overlooks a legitimate part of the gradient revealed by the MLE formulation.

Calibrated Policy Gradient. Motivated by this observation, we propose a unified modification to REINFORCEtype algorithms for LLM reasoning. Specifically, we replace the raw reward r = r⋆(q,o) with a calibrated reward that incorporates both positive and negative contributions:

|r = r − (1 − r)<br><br>πθ(o | q) D(q) − πθ(o | q)<br><br>.|
|---|

(10)

When the generation is correct (r = 1), the calibrated reward is unchanged: r = r = 1. The adjustment applies only to incorrect samples. In negative groups, r = 0 for every candidate, but the policy confidences πθ

(o | q) differ; consequently, the adjusted rewards r also differ across candidates, reflecting their relative confidence. This ensures that negative groups contribute informative gradients rather than being discarded, thereby yielding a more statistically principled update rule.

old

We provide the proof and show that the estimator is consistent in Appendix B.1: if the model is correctly specified (i.e., π⋆ = πθ⋆ ∈ {πθ}θ∈Θ), then the true parameter vector θ⋆ is a maximizer of the population log-likelihood.

- 4.3 Confidence Weighted Value Function

After introducing the calibrated policy gradient, we can interpret it as solving a modified policy optimization problem with a redefined value function JMLE(πθ). The next theorem formalizes this perspective: in the on-policy setting, the MLE gradient coincides with the gradient of this specially constructed value function. The proof is deferred to Section B.2.

- Theorem 2. If we collect dataset D according to qi ∼ ξ and oi ∼ πθ(· | qi), then the gradient of the (population) log-likelihood function L(θ) is identical to the gradient of the following value function JMLE(πθ):

maximizeθ JMLE(πθ) = J+(πθ) − J−(πθ), (11) where

θ(·|q) r⋆(q,o) , (12a) J−(πθ) := Eq∼ξ,o∼π

J+(πθ) := Eq∼ξ,o∼π

θ(·|q) w πθ(o | q)/D(q) 1 − r⋆(q,o) . (12b) Here the weight function w(·) is defined as

1 z

1 1 − z − 1 for any 0 ≤ z < 1. (13)

log

w(z) :=

Proof of inequality (4): By comparing derivatives, one can show that

u2/2 1   u/3

eu   u   1 

for any 0 < u < 3.

Setting u = (p2z + 1) 1 for z > 0 and performing routine algebra yields

Proof of inequality (4): By comparing derivatives, one can show that

###### Self-Normalized Tail Inequalities for Matrix Martingales

u2/2 1   u/3 

1 4z2

then

. Hence,

u2/2 1   u/3

eu   u   1 

for any 0 < u < 3.

qA>

###### qa>Vt  11 a.

then

then

Setting u = (p2z + 1) 1 for z > 0 and performing routine algebra yields

t Vt  11 At & sup

⌘  

exp⇣ 1 p2z + 1

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

1 p2z + 1   1 

1 4z2

qA>

qa>Vt  11 a.

qA>

###### qa>Vt  11 a.

for any z > 0.

a

t Vt  11 At & sup

u2/2 1   u/3 

1 4z2

t Vt  11 At & sup

Now let z = Rt1 pkW⌧k2 (or z = Rt1 p"/2). Then g✓

. Hence,

a

###### Self-Normalized Tail Inequalities for Matrix Martingales

a

◆ =

⇢exp⇣ Rt p2kW⌧k2 + Rt

Rt p2kW⌧k2 + Rt   1 

⌘  

###### Self-Normalized Tail Inequalities for Matrix Martingales

t p2kW⌧k2 + Rt

1 R2

###### Self-Normalized Tail Inequalities for Matrix Martingales

September 13, 2025 Abstract

⌘  

exp⇣ 1 p2z + 1

1 p2z + 1   1 

1 4z2

###### Answer Space O for a question q

for any z > 0.

###### Self-Normalized Tail Inequalities for Matrix Martingales

then

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

Answer Space O for a question q

t2 4kW⌧k2

###### Answer Space O for a question q

Now let z = Rt1 pkW⌧k2 (or z = Rt1 p"/2). Then g✓



, which establishes inequality (4).

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

qA>

qa>Vt  11 a.

###### Self-Normalized Tail Inequalities for Matrix Martingales

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

then

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

t Vt  11 At & sup

◆ =

⇢exp⇣ Rt p2kW⌧k2 + Rt

Rt p2kW⌧k2 + Rt   1 

qA>

qa>Vt  11 a.

⌘  

###### Self-Normalized Tail Inequalities for Matrix Martingales

###### Self-Normalized Tail Inequalities for Matrix Martingales

t p2kW⌧k2 + Rt

1 R2

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

t Vt  11 At & sup

a

###### Self-Normalized Tail Inequalities for Matrix Martingales

###### Self-Normalized Tail Inequalities for Matrix Martingales

###### GRPO Reward = (

###### Self-Normalized Tail Inequalities for Matrix Martingales

###### Self-Normalized Tail Inequalities for Matrix Martingales

1 3 0 7

Correction Rate p? P

September 13, 2025 Abstract

a

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

Correction Rate p? P

###### References

⇥

⇤

t2 4kW⌧k2

Correction Rate p? P

⇥

⇤



, which establishes inequality (4).

September 13, 2025 Abstract

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

Answer o is correct for question q

0 1

⇥

⇤

September 13, 2025 Abstract

Answer o is correct for question q

0 1

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

Answer Space O for a question q

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

⇧

⇧

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

Yaqi Duan⇧ Department of Technology, Operations and Statistics Stern School of Business, New York University⇧

Answer Space O for a question q

[1] J. Tropp. Freedman’s inequality for matrix martingales. 2011.

Answer o is correct for question q

0 1

September 13, 2025 Abstract

Probability ⇡(o | q)

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

E := ⇢k✓bt   ✓t?kVt   ",

 max(Vt)  min(Vt)  ` 

then

then

###### GRPO Reward = (

September 13, 2025 Abstract

then

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

⇧

⇧

1 3 0 7

qA>

qa>Vt  11

qA>

qa>Vt  11 a.

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

8 ><

qA>

qa>Vt  11 a.

###### References

September 13, 2025 Abstract

September 13, 2025 Abstract

t Vt  11 At & sup

t Vt  11 At & sup

Correction Rate p? P

1 3  

###### GRPO Reward = (

t Vt  11 At & sup

September 13, 2025 Abstract

September 13, 2025 Abstract

Correction Rate p? P

1 3 0 7

⇥

⇤

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

D(q) Normalization

a

a

LENS Reward =

###### D(q) Normalization

September 13, 2025 Abstract

September 13, 2025 Abstract

h✓bt 1,ai +  tqa>Vt  11 a  h✓bt 1,Ati +  tqA>

then

###### GRPO Reward = (

⇡(o | q) D(q)   ⇡(o | q)

a

⇥

⇤

Answer o is correct for question q

0 1

###### GRPO Reward = (

[1] J. Tropp. Freedman’s inequality for matrix martingales. 2011.

>:

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

###### D(q) Normalization

7

1 3 0 7

qA>

qa>Vt  11 a.

t Vt  11 At

1 3 0 7

Answer o is correct for question q

0 1

then

Probability ⇡(o | q)

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

t Vt  11 At & sup

qA>

qa>Vt  11 a.

###### GRPO Reward = (

E := ⇢k✓bt   ✓t?kVt   ",

 max(Vt)  min(Vt)  ` 

a

t Vt  11 At & sup

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

This work provides tail bounds for martingale that depend on realized predictable quadratic variation.

1 3 0 7

Answer Space O for a question q

Answer Space O for a question q

###### GRPO Reward = (

###### GRPO Reward = (

Answer Space O for a question q

Probability ⇡(o | q)

 h✓bt 1,ai   h✓bt 1,Ati    tqa>Vt  11 a   O(1) If

 tqA>

t Vt  11 At    tqa>Vt  11 a +

8 ><

a

1 3 0 7

1 3 0 7

0.2 0.4 0.6 0.8 2 3 4 5 6

###### GRPO Reward = (

###### GRPO Reward = (

D(q) Normalization

1 3  

1 3 0 7

1 3 0 7

Answeroiscorrectforquestionq ⇤ 01

?CorrectionRatep

Answer Space O for a question q

Probability ⇡(o | q)

Desirable Policy ⇡?

Probability ⇡(o | q)

8 ><

LENS Reward =

###### = (

###### = (

###### GRPO Reward = (

###### GRPO Reward = (

⇡(o | q) D(q)   ⇡(o | q)

Desirable Policy ⇡?

h✓bt 1,ai +  tqa>Vt  11 a  h✓bt 1,Ati +  tqA>

Answer Space O for a question q

>:

1 3  

then

Desirable Policy ⇡?

1 3 0 7

1 3 0 7

1 3 0 7

1 3 0 7

7

D(q) Normalization

This formulation provides insight into the behavior of the MLE optimizer. The objective JMLE(πθ) balances two components:

then

1 z

1 1   z   1

Probability ⇡(o | q)

t Vt  11 At

qa>Vt  11 a &  t 1 , then

Correction Rate ? ⇥

Correction Rate p? P

Correction Rate p? P

qA>

qa>Vt  11 a.

8 ><

LENS Reward =

qA>

qa>Vt  11 a.

w(z) =

log

Probability ⇡(o | q)

Probability ⇡(o | q)

⇡(o | q) D(q)   ⇡(o | q)

8 ><

⇥

⇤

⇤

⇥

⇤

>:

1 3  

⇥

⇤

t Vt  11 At & sup

OAnswerSpaceforaquestionq

sup

t Vt  11 At & sup

7

q 1>aVa. t1

Correction Rate P

Probability ⇡(o | q)

Probability ⇡(o | q)

Desirable Policy ⇡?

Answer o is correct for question q

PolicyOptimization

RewardModelingviaMLE

Reparameterization

1 3  

⇥

⇤

then

then

Answer is correct for question q

0 1

Answer o is correct for question q

0 1

Answer o is correct for question q

0 1

[Figure 1]

a

8 ><

a

Correction Rate P

LENS Reward =

J+(πθ): This is the standard policy gradient objective (REINFORCE), which maximizes the expected reward. It encourages the policy πθ to take actions (i.e., propose answers o) that are likely to be correct.

Answer o is correct for question q

0 1

0.2 0.4 0.6 0.8 2 3 4 5 6

a

⇡(o | q) D(q)   ⇡(o | q)

 h✓bt 1,ai   h✓bt 1,Ati    tqa>Vt  11 a   O(1) If

 tqA>

t Vt  11 At    tqa>Vt  11 a +

8 ><

8 ><

qA>

qa>Vt  11 a.

qA>

qa>Vt  11 a.

LENS Reward =

1 3  

>:

7

⇡(o | q) D(q)   ⇡(o | q)

1 3  

1 3  

8 ><

8 ><

>:

###### ⇡(o | q)

###### ⇡(o | q)

Probability ⇡(o | q)

Probability ⇡(o | q)

t Vt  11 At & sup

t Vt  11 At & sup

⇡✓(o | q) ⇡ 0 ⇡✓(o | q) ⇡ D(q)

7

qA>

qa>Vt  11 a.

LENS Reward =

D(q)Normalization

###### Reward Modeling via MLE Policy Optimization

1 3  

1 3  

⇡(o | q) D(q)   ⇡(o | q)

0.2 0.4 0.6 0.8 2 3 4 5 6

?DesirablePolicy⇡

LENS Reward =

LENS Reward =

Desirable Policy ⇡?

Answer Space O for a question q

⇡(o | q) D(q)   ⇡(o | q)

⇡(o | q) D(q)   ⇡(o | q)

>:

7

a

a

t Vt  11 At & sup

>:

>:

1 z

1 1   z   1

7

7

LENS Reward =

LENS Reward =

⇡(o | q) D(q)   ⇡(o | q)

⇡(o | q) D(q)   ⇡(o | q)

###### Reward Modeling via MLE Policy Optimization Reparameterization

###### Reward Modeling via MLE Policy Optimization Reparameterization

Answer Space O for a question q

w(z) =

log

###### Reward Modeling via MLE Policy Optimization Reparameterization

>:

>:

8 ><

8 ><

7

7

qa>Vt  11 a &  t 1 , then

D(q) Normalization

q) Normalization

D(q) Normalization

0.2 0.4 0.6 0.8 2 3 4 5 6

a

###### 1 Introduction

& 1VAsuptt t1

1 3  

1 3  

1 3  

1 3  

= > >

= > >

1 z

1 1   z   1

⇥

⇤

0.2 0.4 0.6 0.8 2 3 4 5 6

J−(πθ): This term acts as a penalty for incorrect answers. The cost of being incorrect, 1 − r⋆, is re-weighted by w πθ(o | q)/D(q) , which represents the policy’s own “odds” of its prediction being correct. The penalty is most severe when the policy is highly confident but wrong (as πθ → D−, w → ∞). Conversely, the penalty is negligible when the policy is uncertain and wrong (as πθ → 0+, w → 0). It encourages diversity in the negative responses / exploration in the negative space.

a

sup

[Figure 2]

w(z) =

log

Correction Rate P

Answer o is correct for question q

0 1

0.2 0.4 0.6 0.8 2 3 4 5 6

0.2 0.4 0.6 0.8 2 3 4 5 6

0.2 0.4 0.6 0.8 2 3 4 5 6

Answer Space O for a question q

LENS Reward =

LENS Reward =

Answer Space O for a question q

9

a

⇡(o | q) D(q)   ⇡(o | q)

⇡(o | q) D(q)   ⇡(o | q)

⇡(o | q) D(q)   ⇡(o | q)

⇡(o | q) D(q)   ⇡(o | q)

Answer Space O for a question q

⇡✓(o | q) ⇡ 0 ⇡✓(o | q) ⇡ D(q)

>:

>:

1 z

1 1   z   1

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

0.2 0.4 0.6 0.8 2 3 4 5 6

0.2 0.4 0.6 0.8 2 3 4 5 6

⇥

⇤

7

7

7

7

w(z) =

log

Correction Rate P

Answer o is correct for question q

0 1

###### Reward Modeling via MLE Policy Optimization

qA>

qa>Vt  11 a.

Desirable Policy ⇡?

1 z

1 1   z   1

Desirable Policy ?

Desirable Policy ⇡?

1 z

1 1   z   1

1 z

1 1   z   1

⇡✓(o | q) ⇡ 0 ⇡✓(o | q) ⇡ D(q)

1 z

1 1   z   1

w(z) =

log

t Vt  11 At & sup

w(z) =

log

w(z) =

log

w(z) =

log

###### 1 Introduction

1 z

1 1   z   1

1 z

1 1   z   1

⇥

⇤

⇥

⇤

8

w(z) =

log

w(z) =

log

Correction Rate P

Correction Rate P

0. . . .

0. . . .

Answer o is correct for question q

0 1

0.2 0.4 0.6 0.8 2 3 4 5 6

0.2 0.4 0.6 0.8 2 3 4 5 6

Answer o is correct for question q

0 1

a

⇡✓(o | q) ⇡ 0 ⇡✓(o | q) ⇡ D(q)

###### 1 Introduction

q>A

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

⇡✓(o | q) ⇡ 0 ) w ⇡ 0 ⇡✓(o | q) ⇡ D(q) ) w ! +1

⇡✓(o | q) ⇡ 0 ) w ⇡ 0 ⇡✓(o | q) ⇡ D(q) ) w ! +1

⇡✓(o | q) ⇡ 0 ⇡✓(o | q) ⇡ D(q)

⇡✓(o | q) ⇡ 0 ⇡✓(o | q) ⇡ D(q)

Answer Space O for a question q

⇡✓(o | q) ⇡ 0 ) w ⇡ 0 ⇡✓(o | q) ⇡ D(q) ) w ! +1

⇡✓(o | q) ⇡ 0 ) w ⇡ 0 ⇡✓(o | q) ⇡ D(q) ) w ! +1

###### 1 Introduction

1 z

1 1   z   1

1 z

1 1   z   1

1 z

1 1   z   1

1 z

1 1   z   1

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Light penalty

Heavy penalty

(z

(z

w(z) =

log

w(z) =

log

###### 1 Introduction

###### 1 Introduction

The objective JMLE(πθ) creates a powerful dynamic. It not only drives the policy to maximize rewards but, more critically, it uses the penalty term J−(πθ) to enforce “principled exploration”. By penalizing misplaced confidence, the agent is forced to explore diverse responses rather than exploiting a potentially flawed understanding. This balance between exploitation and exploration is essential for learning a well-calibrated policy.

###### 1 Introduction

###### 1 Introduction

###### 1 Introduction

###### 1 Introduction

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

8

Figure 4 Illustration of the weight function w(z).

1

⇥

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

P

⇡✓(o | q) ⇡ 0 ⇡✓(o | q) ⇡ D(q)

⇡✓(o | q) ⇡ 0 ⇡✓(o | q) ⇡ D(q)

⇡✓(o | q) ⇡ 0 ⇡✓(o | q) ⇡ D(q)

⇡✓(o | q) ⇡ 0 ⇡✓(o | q) ⇡ D(q)

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

###### 1 Introduction

###### 1 Introduction

1

1

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

Classical martingale concentration inequalities, such as Freedman’s inequality [1], are formulated in terms of a deterministic variance bound. This requires the predictable quadratic variation process to be controlled by a ﬁxed constant, which in turn often relies on stationarity or uniform variance assumptions. In contrast, the results below establish self-regularized bounds: the variance term is allowed to be random and path-dependent, and the tail probability automatically adapts to its realized scale. This removes the need for stationarity, yielding sharper and more widely applicable inequalities.

1

###### 5 Proposed Modification to GRPO

1

1

then

1

1

1

The likelihood framework naturally led to a theoretically-grounded modification to GRPO’s advantage function, directly incorporating the insights from the JMLE(πθ) = J+(πθ) − J−(πθ) objective to enhance exploration and policy calibration. The core of our proposal is to replace the original reward with our adjusted reward r from Equation (10). The adjusted reward directly implements the gradient of our theoretical objective. The calibrated reward is then normalized and the obtained advantage is used in Equation (3). We do not modify the GRPO loss function.

1

1

1

1

1

9

9

9

5.1 Implementation and Practical Considerations

We calibrate rewards using the ratio πθ

D(q)−πθold which requires careful handling, particularly in how the probability πθ

old

9

and the difficulty factor D(q) are estimated and used.

old

9

9

old Term. For LLMs with long generations, raw sequence probabilities are dominated by length: per-token probabilities tend to be of similar magnitude, so the sequence probability decays roughly as γ|o| for some γ ∈ (0,1). Consequently, plugging πθ

πθ

9

9

9

in directly makes the adjustment sparse: length-driven decay pushes most candidates’ terms to 0, while a single dominant candidate gets a much larger value. To mitigate this, we use the length-normalized (geometric-mean) probability

old

9

(o | q)1/|o|.

π¯θ

(o | q) := πθ

old

old

9

In Appendix C we show that our likelihood framework naturally generalizes to incorporate preferences over correct generations (e.g., in the example in Section 4.1, we can make π⋆(A | q) = ρ(q,A) and π⋆(B | q) = ρ(q,B), rather than 0.5 and 0.5); empirically, the above substitution is equivalent to a calibrated reward that encodes a length preference for correct generations.

9

9

EstimatingD(q). The true difficulty function D(q) (as defined in Equation (5)) is unknown and acts as a key hyperparameter controlling learning dynamics. Smaller D(q) increases the penalty on confident but incorrect predictions, encouraging broader exploration to avoid overconfidence. This mechanism allows tuning between exploiting correct answers and exploring uncertain ones.

A direct estimator follows from importance sampling:

G

−1

−1

−1

r⋆(q,o) πθ

1 G

ri πθ

p⋆(o′ | q)

. (14)

= Eo∼π

≈

Dimp(q) =

θold

(o | q)

(oi | q)

old

old

o′∈O

i=1

In this formulation, we approximate the expectation with a Monte Carlo average over a group of G samples {(oi,ri)}Gi=1 drawn from πθ

. For numerical stability, we should conservatively overestimate D(q) so that the denominator D(q) − π¯θ

old

is positive and well-conditioned. Concretely, over the G candidates in the group we set

old

π¯θ

D(q) = max Dimp(q), 2 · max

(oi | q) ,

old

1≤i≤G

which keeps the calibrated rewards in [−1,1]. Dimp(q) is undefined for negative groups as all ri are zero. In that case we fall back to

π¯θ

D(q) = 2 · max

1≤i≤G

(oi | q).

old

Handling Invariance. GRPO’s group-wise normalization enjoys a useful sign invariance: regardless of how many generations are correct, after normalization all incorrect generations have negative advantages and all correct generations have positive advantages. We aim to preserve this property under our calibration. Consider the extreme mixed group with one correct and G − 1 incorrect generations; the calibrated rewards might look like [1,0,−1,...,−1]. To maintain sign invariance, we scale all negative calibrated rewards by 1/G.

Calibrated Reward (per sample). In combination, our calibrated reward is

π¯θ

(oi | q) D(q) − π¯θ

1 G

old

ri := ri − (1 − ri)

,

(oi | q)

old

with

 

(oj | q) , (mixed group), 2 · maxj π¯θ

max Dimp(q), 2 · maxj π¯θ

old

D(q) =



(oj | q), (negative group).

old

Final Objective. In negative groups, the only signal comes from confidence differences rather than a verifiable reward, so we treat it as a weaker, auxiliary signal. For those groups we use de-meaning only in the normalization for simplicity, and we introduce the only hyperparameter, α, to down-weight their contribution:

Jours = JGRPO[mixed groups] + α · JGRPO[negative groups].

- 6 Experimental Results

We now empirically test the effectiveness of our algorithm.

Set-up. We evaluate our method on mathematical reasoning. Since a single RL run is affected by randomness, we conduct training with two different datasets to ensure robust results: 1) the MATH training split combined with the DAPO dataset (Yu et al., 2025a), and 2) the MATH training split combined with Numina 1.5 (Li et al., 2024a). For DAPO dataset, we perform two independent runs with different random seeds. All evaluations are on the MATH test set. We consider two models, Llama-3.1-8B-Instruct (Dubey et al., 2024) and Qwen-2.5-3B-Base (Yang et al., 2024) 2, and compare our method against the baseline GRPO. We report details and results on Numina in Appendix E.

2Following prior work, we apply RL to the Qwen base model (Liu et al., 2025b), which already follows instructions and produces outputs in the required format, whereas for Llama we use the instruction-tuned model (Arnal et al., 2025). This allows us to remove the format reward in RLVR.

###### (a) Llama-3.1-8B-Instruct

MATH-500 Eval

MATH-500 Levels 4-5 Eval

0.600

0.40

GRPO baseline

GRPO baseline

0.575

0.38

Ours

Ours

0.550

Accuracy

Accuracy

0.36

0.525

0.34

0.500

0.32

0.475

0.450

0.30

0 200 400 600 800 1000 1200 1400 1600

0 200 400 600 800 1000 1200 1400 1600

Training Step

Training Step

###### (b) Qwen-2.5-3B-Base

MATH-500 Eval

MATH-500 Levels 4-5 Eval

0.70

0.60

GRPO baseline

GRPO baseline

0.68

Ours

Ours

0.55

Accuracy

Accuracy

0.66

0.50

0.64

0.45

0.62

0.60

0.40

0 200 400 600 800 1000 1200 1400 1600

0 200 400 600 800 1000 1200 1400 1600

Training Step

Training Step

Figure 5 Comparison of our algorithm and GRPO baseline: performance on the full MATH test set and the Levels 4–5 (hard) subset. Top: Llama-3.1-8B-Instruct; bottom: Qwen-2.5-3B-Base. The accuracy is averaged across all 16 generations during evaluation and over two independent runs. Training set: MATH + DAPO. Our algorithm brings improvement for both models.

Training protocol. To stress-test learning from negative groups, we use a possibly large G and sample 16 completions per question. Each gradient update uses a global batch of 512 trajectories (32 questions × 16 samples). We decode with temperature 1.0 and cap generations at 4,096 tokens. We do not add any KL regularization following common practices. The negative ratio α is set to 0.25 for all models. No format rewards are added to the scalar reward.

Evaluation. At evaluation time, we use temperature 1.0 and top-p 1.0 to evaluate the model in the plain setup as training, and report Pass@k for k ∈ {1,2,4,8,16}. We present evaluation curves during training for both the full MATH dataset, and the MATH Levels 4-5 subset to understand the performance on hard questions. We use Math-Verify (Kydlíček, 2025) as the verifier function for both training and evaluation.

Results. We report training curves for Llama and Qwen in Figure 5. The curves are averaged across two independent runs. The full training results are in Appendix E. Across both models, LENS consistently attains higher accuracy than the GRPO baseline throughout training. On the hard split of MATH, LENS shows substantial additional gains, indicating that the method effectively converts negative groups, which often correspond to hard instances where no candidate is initially correct, into useful learning signals. As a result, when the GRPO curve saturates, LENS continues to improve. These results indicate that our method learns effectively through exploration and explicitly leverages negative groups, yielding stronger performance on difficult problems. Moreover, training remains stable for >1,000 steps without ad hoc tricks or collapse. Training results using Numina set are included in Appendix E, where we observe consistent improvements with identical hyperparameters.

We further report Pass@k in Table 1. Compared with the GRPO baseline, LENS achieves higher Pass@k for k ∈ {1,2,4,8,16}, with the improvement at Pass@16 also significant. These results indicate that our algorithm consistently improves Pass@k for all k, rather than only Pass@1, and that its confidence-based design enables these exploration gains. Appendix D.2 presents ablations that separately evaluate the effect of

- Table 1 Pass@k results on MATH with Llama-3.1-8B-Instruct and Qwen-2.5-3B-Base, averaged over two independent runs. Training set: MATH + DAPO.

Model Algorithm Pass@1 Pass@2 Pass@4 Pass@8 Pass@16 Llama-3.1-8B-Instruct

GRPO baseline 54.09 61.09 67.05 69.29 72.70 LENS (Ours) 56.63 63.05 68.36 72.18 75.34

GRPO baseline 67.06 72.86 77.12 80.83 82.67 LENS (Ours) 68.59 74.79 79.33 82.62 84.44

Qwen-2.5-3B-Base

adjusted rewards in mixed and negative groups, showing strong improvements from negative groups alone.

- 7 Discussion

In this paper, we start from an observation. In GRPO, any generation group in which all samples are incorrect does not contribute to the optimization, even though these generations already consume substantial compute. We ask a question: can we use this data in a principled way? We develop a theoretical framework that begins with reward modeling using both positive and negative data, connects it to policy optimization, and shows that the MLE objective corresponds to an adjusted value function. The adjustment adds a confidence-weighted penalty for incorrect generations. This view yields a calibrated reward that fits seamlessly into GRPO. Empirically, we demonstrate effectiveness on both Llama and Qwen models, with improvements across all Pass@k scores.

Our empirical algorithm builds on the connection between reward modeling and policy optimization, and the framework can also incorporate preference, as shown in Appendix C. We study the simple case and leave further exploration of preference-aware variants for future work. To balance the impact of negative groups and mixed groups, we introduce a single tunable hyperparameter. A natural direction is to quantify the contributions of both sources in theory and design an objective that is free of hyperparameters. Our framework also covers nonbinary reward signals theoretically, and we defer a systematic experimental study of this setting to future work.

Acknowledgment

The authors would like to sincerely thank Dulhan Jayalath, Lovish Madaan, and Yuda Song for their technical guidance. An initial part of this work was completed while YF was an intern at Meta, and YF would like to thank Cheng Zhang for hosting. YF and JK acknowledge support from the Simons Foundation through the Collaborative Grant “The Physics of Learning and Neural Computation” and by the NSF through NRT Award 1922658. YD acknowledges support from NSF Grant DMS-2413812.

References

Charles Arnal, Gaëtan Narozniak, Vivien Cabannes, Yunhao Tang, Julia Kempe, and Remi Munos. Asymmetric reinforce for off-policy reinforcement learning: Balancing positive and negative rewards, 2025. https://arxiv.org/ abs/2506.20520.

Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018.

Shicong Cen, Jincheng Mei, Katayoon Goshvadi, Hanjun Dai, Tong Yang, Sherry Yang, Dale Schuurmans, Yuejie Chi, and Bo Dai. Value-incentivized preference optimization: A unified approach to online and offline rlhf. arXiv preprint arXiv:2405.19320, 2024.

Peter Chen, Xiaopeng Li, Ziniu Li, Xi Chen, and Tianyi Lin. Spectral policy optimization: Coloring your incorrect reasoning in grpo. arXiv preprint arXiv:2505.11595, 2025a.

Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, and Guang Shi. Pass@ k training for adaptively balancing exploration and exploitation of large reasoning models. arXiv preprint arXiv:2508.10751, 2025b.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. arXiv preprint arXiv:2506.14758, 2025.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407, 2024.

Yunzhen Feng, Julia Kempe, Cheng Zhang, Parag Jain, and Anthony Hartshorn. What characterizes effective reasoning? revisiting length, review, and structure of cot. arXiv preprint arXiv:2509.19284, 2025a.

Yunzhen Feng, Ariel Kwiatkowski, Kunhao Zheng, Julia Kempe, and Yaqi Duan. PILAF: Optimal human preference sampling for reward modeling. In Forty-second International Conference on Machine Learning, 2025b.

Jingtong Gao, Ling Pan, Yejing Wang, Rui Zhong, Chi Lu, Qingpeng Cai, Peng Jiang, and Xiangyu Zhao. Navigate the unknown: Enhancing llm reasoning with intrinsic motivation guided exploration. arXiv preprint arXiv:2505.17621, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Hynek Kydlíček. Math-Verify: Math Verification Library, 2025. https://github.com/huggingface/math-verify. Thanh-Long V Le, Myeongho Jeon, Kim Vu, Viet Lai, and Eunho Yang. No prompt left behind: Exploiting zero-variance

prompts in llm reinforcement learning via entropy-guided advantage shaping. arXiv preprint arXiv:2509.21880, 2025.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024a.

Pengyi Li, Matvey Skripkin, Alexander Zubrey, Andrey Kuznetsov, and Ivan Oseledets. Confidence is all you need: Few-shot rl fine-tuning of language models. arXiv preprint arXiv:2506.06395, 2025.

Yiwei Li, Peiwen Yuan, Shaoxiong Feng, Boyuan Pan, Bin Sun, Xinglin Wang, Heda Wang, and Kan Li. Turning dust into gold: Distilling complex reasoning capabilities from llms by leveraging negative data. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 18591–18599, 2024b.

Wei Liu, Siya Qi, Xinyu Wang, Chen Qian, Yali Du, and Yulan He. Nover: Incentive training for language models via verifier-free reinforcement learning. arXiv preprint arXiv:2505.16022, 2025a.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025b.

Chengqi Lyu, Songyang Gao, Yuzhe Gu, Wenwei Zhang, Jianfei Gao, Kuikun Liu, Ziyi Wang, Shuaibin Li, Qian Zhao, Haian Huang, et al. Exploring the limit of outcome reward for learning mathematical reasoning. arXiv preprint arXiv:2502.06781, 2025.

Gongrui Nan, Siye Chen, Jing Huang, Mengyu Lu, Dexun Wang, Chunmei Xie, Weiqi Xiong, Xianzhou Zeng, Qixuan Zhou, Yadong Li, et al. Ngrpo: Negative-enhanced group relative policy optimization. arXiv preprint arXiv:2509.18851, 2025.

Mihir Prabhudesai, Lili Chen, Alex Ippoliti, Katerina Fragkiadaki, Hao Liu, and Deepak Pathak. Maximizing confidence alone improves reasoning. arXiv preprint arXiv:2505.22660, 2025.

Nicolas Le Roux, Marc G. Bellemare, Jonathan Lebensold, Arnaud Bergeron, Joshua Greaves, Alex Fréchette, Carolyne Pelletier, Eric Thibodeau-Laufer, Sándor Toth, and Sam Work. Tapered off-policy reinforce: Stable and efficient reinforcement learning for llms, 2025. https://arxiv.org/abs/2503.14286.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Yunhao Tang, Kunhao Zheng, Gabriel Synnaeve, and Rémi Munos. Optimizing language models for inference time objectives using reinforcement learning. arXiv preprint arXiv:2503.19595, 2025.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Tengyang Xie, Dylan J Foster, Akshay Krishnamurthy, Corby Rosset, Ahmed Awadallah, and Alexander Rakhlin. Exploratory preference optimization: Harnessing implicit q*-approximation for sample-efficient rlhf. arXiv preprint arXiv:2405.21046, 2024.

Wei Xiong, Chenlu Ye, Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, Jiang Bian, Nan Jiang, and Tong Zhang. Reinforce-ada: An adaptive sampling framework for reinforce-style llm training. arXiv preprint arXiv:2510.04996, 2025.

Zhongwen Xu and Zihan Ding. Single-stream policy optimization. arXiv preprint arXiv:2509.13232, 2025.

Qwen An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxin Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yi-Chao Zhang, Yunyang Wan, Yuqi Liu, Zeyu Cui, Zhenru Zhang, Zihan Qiu, Shanghaoran Quan, and Zekun Wang. Qwen2.5 technical report. ArXiv, abs/2412.15115, 2024.

Zhaohui Yang, Yuxiao Ye, Shilei Jiang, Chen Hu, Linjing Li, Shihong Deng, and Daxin Jiang. Unearthing gems from stones: Policy optimization with negative sample augmentation for llm reasoning. arXiv preprint arXiv:2505.14403, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025a.

Tianyu Yu, Bo Ji, Shouli Wang, Shu Yao, Zefan Wang, Ganqu Cui, Lifan Yuan, Ning Ding, Yuan Yao, Zhiyuan Liu, et al. Rlpr: Extrapolating rlvr to general domains without verifiers. arXiv preprint arXiv:2506.18254, 2025b.

Shenao Zhang, Donghan Yu, Hiteshi Sharma, Han Zhong, Zhihan Liu, Ziyi Yang, Shuohang Wang, Hany Hassan, and Zhaoran Wang. Self-exploring language models: Active preference elicitation for online alignment. arXiv preprint arXiv:2405.19332, 2024.

Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590, 2025.

Tianyu Zheng, Tianshun Xing, Qingshui Gu, Taoran Liang, Xingwei Qu, Xin Zhou, Yizhi Li, Zhoufutu Wen, Chenghua Lin, Wenhao Huang, et al. First return, entropy-eliciting explore. arXiv preprint arXiv:2507.07017, 2025.

Xiangxin Zhou, Zichen Liu, Anya Sims, Haonan Wang, Tianyu Pang, Chongxuan Li, Liang Wang, Min Lin, and Chao Du. Reinforcing general reasoning without verifiers. arXiv preprint arXiv:2505.21493, 2025.

Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The surprising effectiveness of negative reinforcement in llm reasoning. arXiv preprint arXiv:2506.01347, 2025.

- A Other Related Works

Exploration in RL. Enhancing exploration during RL training is an important part for all RL algorithms. In RLHF, Xie et al. (2024); Cen et al. (2024); Zhang et al. (2024) use the base model likelihood as an exploration bonus, nudging the policy toward outputs that are plausible yet seldom sampled. Closest in theoretical spirit to our view is Feng et al. (2025b), which studies the MLE objective of reward modeling to derive a principled exploration method. In the reasoning setting, Gao et al. (2025) employ Random Network Distillation (Burda et al., 2018) to encourage novel solution traces. Other works (Cheng et al., 2025; Zheng et al., 2025) promote exploration through entropy based objectives. Finally, Chen et al. (2025b) optimize a pass@k objective (Tang et al., 2025) to increase batch diversity during training. However, these approaches do not propose to differentiate rewards inside negative groups and focus mainly on mixed groups.

Asymmetric treatment of positive and negative outputs. A few recent work introduce asymmetric treatment of positive and negative generations during REINFORCE-style training. (Roux et al., 2025) introduces an asymmetric variant of importance sampling to speed up learning. Arnal et al. (2025) demonstrate that asymmetric REINFORCE, and in particular reducing the signal from negative generations, can be beneficial when data is off-policy.

Using Confidence in RLVR. Confidence proxies have also been applied in RLVR, mainly proposed as a surrogate for the rule-based verifier. Zhao et al. (2025) use the KL divergence between the per token generation probability and a uniform distribution. Zhou et al. (2025); Yu et al. (2025b); Liu et al. (2025a) take the log prob of generating the reference answer conditioned on the existing CoT as the reward. Li et al. (2025) leverage confidence scores at test time for light tuning and report gains. Prabhudesai et al. (2025) similarly optimize the entropy of response tokens as the reward. In all of these studies, the rule-based reward is replaced with a confidence-based proxy and light training is performed. Most works do not train beyond one hundred steps and focus only on Qwen models, which raises concerns about generalization and the risk of reward hacking without a bag of tricks. In contrast, we do not aim to replace rule based rewards; instead, we propose to make use of negative groups in GRPO in a principled way. We demonstrate effectiveness on both Llama and Qwen and show stable training for more than one thousand five hundred steps.

Concurrent works addressing Negative Groups in GRPO Several concurrent works also propose various approaches to address negative groups. Xu and Ding (2025) leverage an on-the-fly baseline so that negative groups have a non-zero baseline and the advantage is non-zero. Similarly, Nan et al. (2025) employ advantage calibration to adjust the baseline. Le et al. (2025) leverage entropy to create differentiation within negative groups. Xiong et al. (2025) propose addressing negative groups by adaptively allocating more generation samples to hard problems. Our work provides a more theory-grounded motivation.

- B Proofs

- B.1 Proof of Theorem 1 We now provide the proof of Theorem 1 and a comment on the estimator consistency.

Proof of Theorem 1. Let πθ ≡ πθ(o | q) and D ≡ D(q) for notational brevity. The gradient of each individual term in the loss L(θ) with respect to θ is found using the chain rule:

1 − r D − πθ ∇θ πθ .

r πθ −

πθ D

∇θ r · log πθ + (1 − r) · log 1 −

=

By applying the identity for the gradient of a logarithm, ∇θ πθ = πθ · ∇θ log πθ, we can express the result as:

πθ

D − πθ ∇θ log πθ , which provides the final result.

r − (1 − r)

Consistency of the Estimator. A key property of this estimator is its consistency under ideal conditions. If the model is correctly specified (i.e., πθ⋆ ∈ {πθ}θ∈Θ), then the true parameter vector θ⋆ is a maximizer of the population log-likelihood. This can be verified by observing that the gradient ∇θ L(θ) evaluates to zero at θ = θ⋆. By taking the conditional expectation of the gradient’s inner term with respect to r, given q and o, we find:

πθ⋆(o | q) D(q) − πθ⋆(o | q)

Er|q,o r − (1 − r)

Using E[r | q,o] = p⋆(o | q) and the definition p⋆ = π⋆/D, this becomes:

p⋆ 1 − p⋆

πθ⋆ D − πθ⋆

= p⋆ − p⋆ = 0.

= p⋆ − (1 − p⋆)

= p⋆ − (1 − p⋆)

Since the conditional expectation of the term multiplying ∇θ log πθ is zero, the full expectation is zero, confirming that θ⋆ is a stationary point.

- B.2 Proof of Theorem 2 We will show that ∇θ JMLE(πθ) is equivalent to ∇θ L(θ) when µ = πθ. First, the target gradient from Theorem 1, with the sampling policy µ set to the model policy πθ, is:

∇θ L(θ) µ=π

θ

πθ D − πθ · ∇θ log πθ(o | q) . (15)

= Eq∼ξ,o∼π

θ(·|q) r − (1 − r)

Next, we rigorously compute the gradient of J(πθ) = J+(πθ) − J−(πθ). The gradient of the positive term is standard:

θ(·|q) r · ∇θ log πθ . (16)

∇θ J+(πθ) = Eq∼ξ,o∼π

For the negative term, J−(πθ) = Eo∼π

θ

w(πθ/D) · (1 − r) , we use the product rule and derive

(1 − r) w(πθ/D) + (πθ/D) · w′(πθ/D) · ∇θ log πθ . (17) Now we compute w(z) + z · w′(z):

∇θ J−(πθ) = Eq,o∼π

θ

z

1−z + D log(1 − z) z2

w(z) + z · w′(z) = −log(1 − z) z − 1 + z

z 1 − z

1 1 − z − 1 =

.

=

This is exactly the term we needed. Substituting this result back into the gradient for J−(πθ):

πθ

D − πθ · ∇θ log πθ . (18) Finally, combining the gradients for the positive and negative parts of J(πθ):

∇θ J−(πθ) = Eq,,o∼π

(1 − r)

θ

πθ

D − πθ · ∇θ log πθ . (19) This expression is identical to the MLE gradient in equation 15. The equivalence is proven.

∇θ JMLE(πθ) = ∇θ J+(πθ) − ∇θ J−(πθ) = Eq,,o∼π

r − (1 − r)

θ

- C A Preference-Aware Framework

The framework introduced in Section 4.1 assumed that when multiple answers are correct, the optimal policy distributes probability mass uniformly across them. For example, if both A and B are correct answers to a question q, we had π⋆(A | q) = π⋆(B | q) = 0.5. However, uniformity may not always reflect the true

reasoning process. In practice, we might prefer some answers over others. For instance, A could be easier to infer, shorter in form, or more natural to express. In such cases, a more realistic distribution might be π⋆(A | q) = 0.9 and π⋆(B | q) = 0.1.

From the perspective of chain-of-thought reasoning, preferences can capture properties such as the length of the reasoning trajectory or the similarity of an answer to outputs from a reference language model. To encode this flexibility, we introduce a nonnegative preference function:

ρ(q,o) ≥ 0, which adjusts the weight assigned to each (q,o) pair.

Modified Framework. With the preference function, we adjust the relation between policy πθ and correctness probabilities. Specifically, we define

pθ(q,o) =

where the difficulty factor D(q) is updated as

1 D(q) · ρ(q,o)

πθ(o | q), (20)

D(q) =

p⋆(q,o) · ρ(q,o)

o∈O

−1

. (21)

Intuitively, D(q) still measures how hard the question is, but it now accounts for the internal weighting across candidate answers.

The maximum likelihood estimation (MLE) problem under this new framework becomes

min

θ

1 n

L(θ) = −

n

i=1

πθ(oi | qi) D(qi) · ρ(qi,oi)

ri · log πθ(oi | qi) + (1 − ri) · log 1 −

. (22)

The corresponding gradient of the log-likelihood is

1 n

∇θ L(θ) = −

n

i=1

πθ(oi | qi) D(qi) · ρ(qi,oi) − πθ(oi | qi) · ∇θ log πθ(oi | qi). (23)

ri − (1 − ri)

Compared to the uniform case, the gradient now incorporates the additional signal encoded by ρ, ensuring that both positive and negative samples are scaled according to the chosen preference structure.

Examples of Preference Functions. To illustrate the flexibility of this framework, we describe some concrete choices of ρ:

Preference as the data collection distribution. Suppose we take ρ(q,o) = µ(o | q), where µ is the distribution used to collect the dataset D. Then the difficulty factor D(q) can be approximated by:

−1

1 |OD(q)|

r⋆(q,o)

D(q) ≈

,

o∈OD(q)

where OD(q) denotes the set of observed answers to question q in D. In words, D(q) can be estimated as the inverse of the empirical correctness rate.

Preference as the policy itself. If we further set µ = πθ, then the negative calibration term simplifies to

πθ(oi | qi) D(qi) · ρ(qi,oi) − πθ(oi | qi)

=

1 D(qi) − 1

.

In this case, the weight for negative samples is exactly the correction rate of the current policy πθ. Equivalently, in the ordinary policy gradient formulation, each question should be reweighted by its correction rate. Although this choice does not produce the “confidence-based” weighting we ultimately seek, it highlights that commonly used uniform weights (e.g., Arnal et al. (2025); Zhu et al. (2025)) emerge as a special case of our framework.

Preference as a function of response length. Now, consider a preference function that depends on the length of the candidate answer (Feng et al., 2025a):

ρ(q,o) := γ|o| for a fixed parameter γ ∈ (0,1). Define the shorthand

1

π¯θ(o | q) := πθ(o | q)

|o| . The negative-sample reward can then be expressed as

πθ(o | q) D(q) · ρ(q,o) − πθ(o | q)

rθ(o | q) = −

π¯θ(o | q)|o| D(q) · γ|o| − π¯θ(o | q)|o| .

= −

1 |o|

For large |o|, we have D(q)

≈ 1. If γ is chosen on the same scale as π¯θ, this weight simplifies to

1 |o|

1 |o|

−1

−1

· γ π¯θ(o | q)

· γ π¯θ(o | q) − 1

|o|

D(q)

D(q)

1 |o|

rθ(o | q) = −

− 1

≈ −

π¯θ(o | q) D(q)

π¯θ(o | q) γ − π¯θ(o | q)

1 |o|

1 |o|

= −

≈ −

·

·

.

1 |o|

· γ − π¯θ(o | q)

Therefore, in practice, it is convenient to set negative-sample reward

1 |o|

rθ(o | q) := −

with γ > 0 properly tuned.

π¯θ(o | q) γ − π¯θ(o | q)

·

1 |o|

= −

1 |o|

πθ(o | q)

·

1 |o|

γ − πθ(o | q)

###### D Experiment Details

- D.1 Hyperparameters

We use a learning rate 3e − 7 for Llama-3.1-8B-Instruct and a learning rate 1e − 6 for Qwen-2.5-3B-Base. The base model requires a larger learning rate while the instruct model has gone through the RLHF stages so a smaller learning rate is better. Prior works (Zhu et al., 2025; Arnal et al., 2025) have used the same setup. The batch size is set to be 512, with 32 questions and 16 generations for each. We use a clipping ratio of 0.2 for all the models to mitigate the impact of off-policy data. We set the maximum number of off-policy updates to 4; in VeRL (Sheng et al., 2024), this is implemented by using a training batch size as 128 (4×32).

We set temperature and top-p to 1.0 during both training and evaluation for both models. We combine the MATH training set of 7,500 samples with a random subset of 17,500 samples from DAPO (Yu et al., 2025a).

- D.2 Ablation

We also conduct an ablation to understand where the improvement comes from. In our algorithm, we modify the reward for all incorrect generations in both mixed and negative groups as in Equation 10. Compared with GRPO, we adjust rewards for incorrect generations within mixed groups, and negative groups now have nonzero advantages. To quantify the contribution of each component, we use the Llama model and consider two settings: (i) modify only the incorrect generations in mixed groups while keeping advantages for negative

- Table 2 Ablation results of pass@k on MATH with Llama-3.1-8B-Instruct.

|Algorithm<br><br>|Pass@1 Pass@2 Pass@4 Pass@8 Pass@16|
|---|---|
|GRPO baseline LENS with only mixed groups LENS with only negative groups<br><br>|56.88 65.42 72.08 78.34 82.80<br>57.42 65.82 73.08 78.80 83.20<br>58.14 66.48 73.46 79.79 83.40<br>|
|LENS (Ours)<br><br>|58.64 66.08 73.98 79.46 83.40|

groups at zero, and (ii) modify only the incorrect generations in negative groups while leaving mixed groups unchanged. This design isolates the effect of each part. We refer to these variants as LENS with only mixed groups and LENS with only negative groups. The training set is MATH and Numina 1.5. The pass@k results are reported in Table 2.

The results show that both components help improve performance. Specifically, adjusting the reward in mixed groups encourages exploration in batches that already contain a correct answer. This helps the model reinforce correct samples while rejecting incorrect generations. As a result, LENS with only mixed groups yields its largest gains at pass@1. LENS with only negative groups also improves over GRPO and in several cases nearly matches the full LENS, suggesting that a substantial portion of the improvement arises from the negative groups.

- E Additional Results

We report additional results of training with MATH + Numina 1.5 (Li et al., 2024a). These complementary results, omitted from the main paper for space, are summarized as follows. Figure 6 shows training curves for Llama and Qwen trained on MATH and Numina 1.5. Table 3 reports the Pass@k results. Due to computational constraints, we perform only one training run. On this training set, we significantly improve Pass@k for larger k, indicating greater diversity.

- Table 3 Pass@k results on MATH with Llama-3.1-8B-Instruct and Qwen-2.5-3B-Base.

Model Algorithm Pass@1 Pass@2 Pass@4 Pass@8 Pass@16 Llama-3.1-8B-Instruct

GRPO baseline 56.88 65.42 72.08 78.34 82.80 LENS (Ours) 58.64 66.08 73.98 79.46 83.40

GRPO baseline 65.88 72.39 77.82 82.05 85.13 LENS (Ours) 68.46 74.74 79.75 83.54 86.28

Qwen-2.5-3B-Base

###### (a) Llama-3.1-8B-Instruct

MATH-500 Eval

MATH-500 Levels 4-5 Eval

0.42

0.58

0.40

0.56

Accuracy

Accuracy

0.38

0.54

0.36

0.52

| |
|---|

| |
|---|

GRPO baseline

GRPO baseline

0.50

0.34

Ours

Ours

| |
|---|

| |
|---|

0.48

0.32

| |
|---|

0 200 400 600 800 1000 1200 1400

0 200 400 600 800 1000 1200 1400

Training Step

Training Step

###### (b) Qwen-2.5-3B-Base

MATH-500 Eval

MATH-500 Levels 4-5 Eval

0.7

| |
|---|

| |
|---|

| |
|---|

0.5

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

0.6

| |
|---|

| |
|---|

0.4

Accuracy

Accuracy

| |
|---|

0.5

| |
|---|

0.3

| |
|---|

0.4

GRPO baseline

GRPO baseline

Ours

Ours

0.2

0.3

0 200 400 600 800 1000 1200 1400

0 200 400 600 800 1000 1200 1400

Training Step

Training Step

###### Figure 6 Comparison of our algorithm and GRPO baseline: performance on the full MATH test set and the Levels 4–5 (hard) subset. Top: Llama-3.1-8B-Instruct; bottom: Qwen-2.5-3B-Base. The accuracy is averaged over all 16 generations during the evaluation. Our algorithm brings improvement for both models. The training set is MATH and Numina 1.5

