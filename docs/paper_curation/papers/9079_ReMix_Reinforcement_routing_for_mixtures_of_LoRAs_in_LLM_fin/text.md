[Figure 1]

[Figure 2]

[Figure 3]

Selection ℑ

Selection ℑ Model Input

Selection ℑ

| | | | | | |
|---|---|---|---|---|---|

LoRA Pool NonActivated Activated

Model Input

Model Input

###### Layer Input

[Figure 4]

| |
|---|

- Layer #1

| | | | | |
|---|---|---|---|---|

- Layer #2

- Layer #1

| | | | | |
|---|---|---|---|---|

- Layer #2

| | | | | |
|---|---|---|---|---|

| | | | | |
|---|---|---|---|---|

- Layer #1

| | | | | |
|---|---|---|---|---|

- Layer #2

| | | | | |
|---|---|---|---|---|

Router

Constant Weights

…

| |
|---|

[Figure 5]

[Figure 6]

### …

### …

### …

| | | | | |
|---|---|---|---|---|

[Figure 7]

Layer #L

Layer #L

| | | | | |
|---|---|---|---|---|

| | | | | |
|---|---|---|---|---|

Layer #L

| | | | | |
|---|---|---|---|---|

Layer Output

SFT Loss ℒ ℑ

SFT Loss ℒ ℑ

SFT Loss ℒ ℑ

Our ReMix Router

Selection Probabilities 𝑄 ℑ

RLOO Gradient Estimator 𝑮𝑷

# arXiv:2603.10160v1[cs.LG]10Mar2026

Figure 1: Finetuning procedure of our proposed ReMix.

## REMIX: REINFORCEMENT ROUTING FOR MIXTURES OF LORAS IN LLM FINETUNING

Ruizhong Qiu∗ University of Illinois Urbana-Champaign, IL, USA rq5@illinois.edu

Hanqing Zeng, Yinglong Xia, Yiwen Meng, Ren Chen Meta AI Menlo Park, CA, USA {zengh,yxia,ywmeng,renchen}@meta.com

Jiarui Feng Washington University St. Louis, WA, USA feng.jiarui@wustl.edu

Dongqi Fu Meta AI Sunnyvale, CA, USA dongqifu@meta.com

Qifan Wang, Jiayi Liu, Jun Xiao, Xiangjun Fan, Benyu Zhang, Hong Li Meta AI Menlo Park, CA, USA {wqfcr,liujiayi,junxiao,maxfan,byzhang,hongli}@meta.com

Zhining Liu, Hyunsik Yoo, Zhichen Zeng, Tianxin Wei, Hanghang Tong University of Illinois Urbana-Champaign, IL, USA {liu326,hy40,zhichenz,twei10,htong}@illinois.edu

ABSTRACT

Low-rank adapters (LoRAs) are a parameter-efficient finetuning technique that injects trainable low-rank matrices into pretrained models to adapt them to new tasks. Mixture-of-LoRAs models expand neural networks efficiently by routing each layer input to a small subset of specialized LoRAs of the layer. Existing Mixture-of-LoRAs routers assign a learned routing weight to each LoRA to enable end-to-end training of the router. Despite their empirical promise, we discover, both theoretically and empirically, that the routing weights often collapse to only one LoRA even when we activate k > 1 LoRAs during finetuning. When one LoRA has a dominantly large weight, then the computation of the other k − 1

∗Work done during an internship at Meta AI.

LoRAs are essentially wasted because using k > 1 would have similar accuracy to k = 1. This essentially limits the number of effective LoRAs and thus severely hinders the realized expressive power of existing Mixture-of-LoRAs models. In this work, we attribute this weakness to the nature of learnable routing weights and rethink the fundamental design of the router. To address this critical issue, we propose a simple yet effective router design that we call Reinforcement Routing for Mixture-of-LoRAs (ReMix). Our key idea is using non-learnable routing weights to ensure all active LoRAs to be equally effective, with no single LoRA dominating the routing weights. However, such non-learnable routing weights make it infeasible to directly train routers via gradient descent. In response, we further propose an unbiased gradient estimator for the router and employ the reinforce leave-one-out (RLOO) technique to reduce the variance of the estimator. Our gradient estimator also enables to scale up training compute to boost the predictive performance of our ReMix. Extensive experiments demonstrate that our proposed ReMix significantly outperforms state-of-the-art parameter-efficient finetuning methods under a small number of activated parameters.

- 1 INTRODUCTION

ESS( (l)) 1.15

1.0

0.933

Dominating Dominated

| |
|---|

Parameter-efficient fine-tuning (PEFT) aims to reduce the number of trainable parameters while achieving strong task performance (e.g., He et al., 2022; R¨uckl´e et al., 2020; Jie et al., 2023). Among PEFT methods, low-rank adapters (LoRAs, Hu et al., 2021) have become particularly prominent due to their simplicity and effectiveness. By injecting lightweight low-rank matrices into pretrained weight matrices, LoRAs allow downstream adaptation with a small fraction of trainable parameters, making them particularly attractive for resource-constrained settings and large-scale multi-task deployments.

0.8

(l)RoutingWeighti

0.6

0.4

0.2

0.022 0.005 0.010

0.003 0.015 0.005 0.007

0.0

1 2 3 4 5 6 7 8

LoRA i

Figure 2: We often observe that only one LoRA has a dominating routing weight that is close to one at deeper layers even when we activate k > 1 LoRAs during finetuning. Consequently, the computation of the other k − 1 LoRAs are essentially wasted because using k > 1 would have similar accuracy to k = 1. (The dominating LoRA can differ on different inputs.)

Building on the success of LoRAs, researchers have proposed Mixture-of-LoRAs to further enhance parameter efficiency and expressive power (e.g., Huang et al., 2023; Wang et al., 2023b; Tian et al., 2024; Zeng et al., 2025a). The key idea is to route each input through a small pool of LoRAs per layer, thereby enabling specialization of LoRAs across different input distributions. Central to this framework is the router, which assigns routing weights across a pool of multiple LoRAs. Current approaches rely on learned routing weights, trained jointly with task objectives via gradient descent. In principle, such routers should flexibly allocate inputs across LoRAs and balance capacity usage.

Despite their empirical promise, we theoretically reveal a striking weakness of existing Mixtureof-LoRAs routers: routing weights can be extremely imbalanced, often collapsing to a very small number of LoRAs with high probability. Furthermore, we empirically observe that the imbalance even worsens as finetuning progresses, where the effective number of LoRAs drops to 1 quickly even though we activate k > 1 LoRAs during finetuning. This essentially disables all other LoRAs, thereby limiting the realized expressive power of the mixture. When only one LoRA has a dominating weight, the computation of the other k − 1 LoRAs are essentially wasted because using k > 1 would have similar accuracy to k = 1. We call this critical issue routing weight collapse.

To address this critical limitation, we revisit the fundamental design of the router. Instead of relying on learned continuous weights that tend to result in routing weight collapse, we propose a simple yet effective design called Reinforcement Routing for Mixture-of-LoRAs (ReMix), which enforces a constant routing weights across all activated LoRAs. This ensures that all active LoRAs contribute equally, avoiding collapse into a single dominant LoRA. Since non-learnable weights prevent direct training via backpropagation, we reformulate the router training problem as reinforcement learning (RL), where we view the supervised finetuning loss as the negative reward and the router as the

policy model of RL. We then propose an unbiased, RLOO-based gradient estimator tailored for our proposed router. This unbiased estimator enables stable training and scales efficiently to large compute budgets, unlocking the full potential of mixture-based parameter-efficient finetuning. Our main contributions are as follows.

- • Theoretical insights on routing weight collapse: We theoretically reveal and empirically observe a fundamental limitation of routers: We observe that for each given input, often only one LoRA has a dominating routing weight that is close to one. This extreme imbalance essentially disables all other LoRAs and severely limits the expressive power of the model.
- • Simple yet effective router: To address routing imbalance, we propose a new router design with a constant routing weight across all activated LoRAs. Our design does not introduce any additional inference cost over existing Mixture-of-LoRAs methods.
- • Reinforcement learning for router training: To address the non-differentiability of our proposed router, we reformulate the router training problem as reinforcement learning and propose an unbiased, RLOO-based gradient estimator tailored for our proposed router.
- • Empirical evaluation: Through extensive experiments across diverse benchmarks, we demonstrate that ReMix consistently outperforms state-of-the-art parameter-efficient finetuning methods under the same parameter budgets.

- 2 MOTIVATION: ROUTING WEIGHT COLLAPSE

Evolution of Routing Weights

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

In this section, we analyze and reveal a critical limitation of existing Mixture-of-LoRAs routers: the extreme imbalance in routing weights assigned to different LoRAs. After introducing preliminaries in Section 2.1, we first make a fundamental theoretical analysis showing that the number of effective LoRAs per layer is severely limited. Then, we corroborate this finding with empirical evidence from our experiments.

7

(l)ESS()

5

3

- 0 1000 2000 3000 4000 5000 Finetuning Step
- 1

Figure 3: Routing weight collapse even worsens as finetuning progresses. The effective support size ESS(π(l)) often drops to 1 quickly during finetuning.

- 2.1 PRELIMINARIES: MIXTURE OF LORAS

Mixture-of-LoRAs is a type of parameter-efficient adapter that enhances the capacity of large models using only a small number of LoRAs and a lightweight router to dynamically select the LoRAs for each input.

Let D denote the hidden dimensionality of the model. Following prior work, we apply LoRAs to feedforward layers in the LLM, and all other layers are frozen. Let x(l),y(l) ∈ RD denote the input and the output of feedforward layer l (l = 1,...,L), respectively. Let n denote the number of LoRAs we use in the mixture. Each LoRA i = 1,...,n is a linear map parameterized as a

low-rank decomposition Bi(l)A(il) ∈ RD×D, where A(il) ∈ Rr×D and Bi(l) ∈ RD×r are learnable parameters, and r ≪ D is the rank of LoRAs. A router of a layer l is a small neural network

parameterized by a matrix P(l) ∈ Rn×D that predicts a categorical distribution over n LoRAs via the softmax operation:

#### π(l) := softmax(P(l)x(l)) ∈ Rn. (1)

Here, πi(l) := (π(l))i represents the routing weight assigned to the i-th LoRA. Given the routing weights, the output of a typical Mixture-of-LoRAs layer is computed as:

n

#### πi(l)Bi(l)A(il)x(l). (2)

y(l) := W(l)x(l) +

i=1

where W(l) ∈ RD×D denotes the frozen weight of the layer l. This formulation intends to differentiably select a specialized subset of LoRAs for each given layer input x(l).

- 2.2 THEORETICAL ANALYSIS

We make a fundamental theoretical analysis showing that the number of effective LoRAs is severely limited. Recall that the output of a Mixture-of-LoRAs layer is a weighted sum of the LoRA outputs, where the routing weights are typically normalized via a softmax function. While this design allows for end-to-end training, we show that it introduces a strong tendency for the router to concentrate most of the weight on only one or two LoRAs.

To quantify the effective number of LoRAs, we use the effective support size notion from information theory. For routing weights π(l) ̸= 0, the effective support size (ESS) is defined as (Grendar, 2006)

ESS(π(l)) :=

n i=1 |πi(l)| 2 n i=1 |πi(l)|2

= ∥π(l)∥1 ∥π(l)∥2

2

. (3)

The intuition of ESS(π(l)) is that it measures the number of LoRAs with relatively large routing weights. For example, if π(l) is one-hot, then we have ESS(π(l)) = 1; if π(l) is uniform over n LoRAs, then we have ESS(π(l)) = n. Note that ESS(π(l)) concerns only about the utilization of LoRAs for each given input, not the overall utilization of each LoRA over the entire dataset. With the help of this notion of ESS, we formally state our theoretical observation in the following Theorem 1.

- Theorem 1 (routing weight collapse). Suppose that the router parameter matrix P(l) follows i.i.d. Gaussian initialization with variance σ2 > 0 (e.g., Kaiming initialization, He et al., 2015). Then for any 0 < δ < 1, with probability at least 1 − δ, the effective support size of π(l) is at most

ESS(π(l)) ≤

  1+

1 exp δσ∥x

(l)∥2

3 2

√

π

ln 3 ln n+ √2π 2n−1log2 n−1

− ln(n − 1)

  

2

.

- The proof of Theorem 1 is deferred to Appendix B.1. Our Theorem 1 shows that with high probability, only an extremely small number of LoRAs have relatively large routing weights. For instance, if σ = 1, and there are n = 8 LoRAs, and x(l) is a Rademacher random vector in RD=1024, then our Theorem 1 shows that with probability at least 84.19%, at most two LoRAs have relatively large routing weights. Since each routing weight is a coefficient in front of each LoRA, a relatively small routing weight would essentially disable that LoRA. Moreover, those extremely small routing weights also vanish the gradient back-propagated to the corresponding LoRAs and consequently hinder the learning process of these LoRAs. Therefore, this phenomenon severely limits the realized expressive power and performance of the Mixture-of-LoRAs model.

- 2.3 EMPIRICAL ANALYSIS

To further validate our theoretical result in Theorem 1, we conduct a case study on the routing weights across LoRAs in MixLoRA (Li et al., 2024a), a popular Mixture-of-LoRAs method. Specifically, we track the routing weights of the last layer throughout the training process on the GSM8K dataset (a mathematical reasoning dataset) and compute the distributions and the ESS of the routing weights.

To visualize the distribution of routing weights, we plot a typical histogram of routing weights during finetuning, shown in Figure 2. We often observe that only one LoRA has a dominating routing weight close to one at deeper layers while all other seven LoRAs have negligibly small routing weights. The observation echoes our Theorem 1 that the learned routing weights are indeed extremely imbalanced. The extremely limited number of effective LoRAs severely restricts the realized expressive power of the Mixture-of-LoRAs model: when only one LoRA has a dominating weight, the computation of the other k−1 LoRAs are essentially wasted because using k > 1 would have similar accuracy to k = 1.

To further study how the distribution of routing weights evolve over the finetuning process, we plot the ESS of the worst routing weights at each training step, as shown in Figure 3. In fact, the imbalance even worsens as the finetuning process progresses. We often observe that the effective

support size ESS(π(l)) often drops to 1 quickly during finetuning. For instance, even though the ESS is around 4 at step 0, the ESS quickly decreases to 1 since only step 1000 and never increases thereafter. As a remark, different inputs can have different dominating LoRAs, but they still suffer from routing weight collapse.

These results highlight a fundamental limitation of current Mixture-of-LoRAs routers: despite the potential for increased expressivity via multiple LoRAs, the model essentially activates only an extremely small subset for each given input. This motivates our proposed method, which aims to ensure a more balanced and effective use of other available LoRAs.

- 3 SIMPLE YET EFFECTIVE METHOD: REMIX

In this section, we propose a simple yet effective router design called Reinforcement Routing for Mixture-of-LoRAs (ReMix). First, we introduce the adapter architecture in Section 3.1. Then, we describe the finetuning procedure in Section 3.2 and the inference procedure in Section 3.3.

- 3.1 ADAPTER ARCHITECTURE: NON-LEARNABLE WEIGHT In this subsection, we introduce the adapter architecture of our proposed method ReMix.

Given layer input x(l) ∈ RD, we first produce an n-way categorical routing distribution q(l) := softmax(P(l)x(l)) ∈ Rn≥0 over the n LoRAs, where P(l) ∈ Rn×D denotes the learnable parameter matrix of the router. Then, we use the routing distribution q(l) to select the k LoRAs I(l) := {i(1l),...,i(kl)} to activate. The LoRA selection procedure differs between finetuning and inference, which we will describe later in Sections 3.2 & 3.3.

To address the extreme imbalance of routing weights in existing Mixture-of-LoRAs models (Section 2), we assign the a constant routing weight ω > 0 to all the k activated LoRAs and zero routing weights to all non-activated LoRAs. Formally, our routing weights π(l) are defined as

πi(l) := ω [i∈I(l)] =

ω, if i ∈ I(l), 0, if i ∈/ I(l),

i = 1,...,n, (4)

where ω is either LoRA-type ω := 2/kr (Hu et al., 2021) or rsLoRA-type ω := 2/

√

kr (Kalajdzievski, 2023). In fact, our method is not sensitive to ω, as shown in Section 4.8.

Notably, our design ensures that ESS(π(l)) = k, which is in stark contrast to existing learnable routing weights (Theorem 1). Finally, we compute the layer output y(l) ∈ RD as a π(l)-weighted sum over k activated LoRAs. Using the sparse nature of our routing weights π(l), the computation of layer output y(l) can be simplified as follows:

y(l) := W(l)x(l) +

n

i=1

πi(l)Bi(l)A(il)x(l)W(l)x(l) + ω

k

j=1

B(l) i(jl)

A(l) i(jl)

x(l). (5)

- 3.2 FINETUNING PROCEDURE: RLOO In this subsection, we describe how to train our proposed ReMix during finetuning.

Let I := (I(1),...,I(L)) denote the collection of activated LoRAs of the entire LLM for a given model input, and we call I a selection. Let L(I) denote the supervised finetuning (SFT) loss when

activated LoRAs are I. Regarding LoRA parameters A(il), Bi(l), since the LLM output is differentiable w.r.t. LoRA parameters, we can simply use their gradients GA(l)

#### := ∇A(l)

#### L(I), GB(l)

#### := ∇B(l)

i

i

i

L(I) to train them.

i

Regarding router parameters, however, the LLM output is not differentiable w.r.t. router parameters P(l) because routing weights πi(l) are a constant hyperparameter ω. Consequently, we cannot directly compute their gradients ∇P(l)

L(I) as it is not defined. To address this nondifferentiability, we propose sampling each I(l) from the corresponding routing distribution q(l)

i

so that EI(l)∼q(l)[L(I)] depends on router parameters P(l). This enables EI(l)∼q(l)[L(I)] to be differentiable w.r.t. router parameters P(l). Hence, we propose using GP(l) := ∇P(l)EI(l)∼q(l)[L(I)] as a surrogate gradient of P(l). Formally, given the routing distribution q(l) := softmax(P(l)x(l)), we sample k LoRAs (i(1l),...,i(kl)) ∼ q(l) from q(l) without replacement to compose the activated LoRA subset I(l) := (i(1l),...,i(kl)), where sampling without replacement ensures that the k activated LoRAs are mutually distinct.

However, due to the exponentially many possibilities of I, it is computationally intractable to straightforwardly compute GP(l) = ∇P(l)EI(l)∼q(l)[L(I)] by definition. To address this intractability, we alternatively consider router training as a reinforcement learning (RL) problem, where we view the SFT loss L(I) as the negative reward and the routers q(l) as the policy model. With this alternative view, we are able to employ the policy gradient estimator in RL to estimate the surrogate gradient GP(l). Formally, we independently sample M selections J1,...,JM, where M represents the training compute budget. Write each selection as Im =: (Im(l))Ll=1 =: ((i(m,jl) )kj=1)Ll=1 (m = 1,...,M), where Im(l) denotes the ordered set of selected LoRAs at the l-th layer in the m-th selection Jm, and i(m,jl) denotes the j-th selected LoRA at the l-th layer in the m-th selection Jm. Due to sampling without replacement, the probability of each selection Jm is

q(l) i

(l) m,j

Q(Jm) := Ll=1 kj=1

. Under this factorization, we further derive the RLOO gradi-

1− jj−′=11 q(l)

(l) m,j′

i

ent estimator (Kool et al., 2019) to estimate the surrogate gradient GP(l):

1 M−1

GP(l) :=

M

1 M−1

L(Im) − L ∇P(l) log Q(Jm)

m=1

k

M

L(Im) − L

m=1

j=1

q(l)

i(m,jl)

∇P(l) log

#### ,

j−1

q(l)

1−

i(m,jl) ′

j′=1

where L := M1 Mm=1 L(Im) is the average SFT loss across the M selections.It can be shown that our RLOO gradient estimator is unbiased: EJ

#### 1,...,Jm[ GP(l)] = GP(l).

- 3.3 INFERENCE PROCEDURE: TOP-k SELECTION

In this subsection, we describe how our proposed ReMix selects the LoRAs to activate during inference. While it is possible to randomly sample the LoRAs like the finetuning procedure, here we propose a better, theoretically optimal approach to LoRA selection.

Our following Theorem 2 shows that the optimal strategy is in fact top-k selection as long as the router is trained sufficiently well.

- Theorem 2 (optimality of top-k selection). Let I(l)∗ = {i(1l)∗,...,i(kl)∗} denote the optimal subset of LoRAs for a given model input. As long as the router q(l) is trained sufficiently well such that PI(l)∼q(l)[I(l) = I(l)∗] > 21, then the LoRAs i with top-k qi(l) are guaranteed to constitute the best subset I(l)∗: argtopkni=1 qi(l) = I(l)∗.

- The proof of Theorem 2 is deferred to Appendix B.2. Notably, our Theorem 2 shows that when sampling yields the optimal subset with probability above 50%, then top-k selection substantially improves this probability to 100%. Intuitively speaking, as long as the router is trained sufficiently

well, then the optimal choices of LoRAs are in fact those i with top-k qi(l). Motivated by Theorem 2, we employ top-k LoRA selection (instead of random sampling) during inference:

n argtopk i=1

I(l) = {i(1l),...,i(kl)} :=

qi(l). (6)

- Table 1: Comparison with existing parameter-efficient finetuning methods. Our ReMix consistently outperforms all baseline methods while maintaining strong parameter efficiency. We use the same parameter count budget for all methods to ensure fair comparison; for each method, we search for the best parameter count and report the accuracy under the best parameter count.

GSM8K HumanEval ARC-c Average Accuracy Params Pass@1 Params Accuracy Params Accuracy Params No Tuning

Type Method

Zero-Shot 04.78 N/A 13.41 N/A 22.03 N/A 13.41 N/A Few-Shot 55.95 N/A 17.68 N/A 81.36 N/A 51.66 N/A

Prefix Tuning 02.65 0.034B 00.00 0.034B 28.47 0.004B 10.37 0.024B Prompt Tuning 04.70 0.000B 26.22 0.000B 23.73 0.000B 18.22 0.000B P-Tuning 34.19 0.001B 27.44 0.001B 43.05 0.001B 34.89 0.001B

Prefix Injection

(IA)3 08.57 0.001B 31.10 0.001B 23.39 0.001B 21.02 0.001B LoRA 59.21 0.168B 26.83 0.084B 83.05 0.084B 56.36 0.112B DoRA 55.34 0.043B 31.10 0.169B 83.39 0.169B 56.61 0.127B rsLoRA 62.47 0.042B 28.66 0.021B 82.71 0.021B 57.95 0.028B

Weight Modulation

VB-LoRA 34.27 0.677B 29.27 0.673B 23.73 0.674B 29.09 0.675B MixLoRA 61.87 0.068B 28.05 0.116B 82.37 0.119B 57.43 0.101B HydraLoRA 62.47 0.092B 20.12 0.079B 82.71 0.082B 55.10 0.084B ReMix (Ours) 65.66 0.106B 32.93 0.090B 83.73 0.016B 60.77 0.070B

Mixture

- 4 EXPERIMENTS

- 4.1 EXPERIMENTAL SETUP

Baselines. We comprehensively compare our proposed ReMix against various types of baseline methods. (i) No tuning methods: testing the base LLM directly under zero-shot and few-shot prompting. (ii) Prefix injection methods: Prefix Tuning (Li & Liang, 2021), Prompt Tuning (Lester et al., 2021), and P-Tuning (Liu et al., 2021b). (iii) Weight modulation methods: (IA)3 (Liu et al.,

- 2022), LoRA (Hu et al., 2021), DoRA (Liu et al., 2024a), and rsLoRA (Kalajdzievski, 2023). (iv) Mixture methods: VB-LoRA (Li et al., 2024b), MixLoRA, (Li et al., 2024a), and HydraLoRA (Tian et al., 2024). For each baseline method, we perform a hyperparameter search and report the best results.

Datasets & evaluation metrics. We finetune the base LLM and evaluate them on a diverse set of benchmarks, including GSM8K (Cobbe et al., 2021) to evaluate mathematical reasoning capabilities, HumanEval (Chen et al., 2021) to evaluate code generation capabilities, and ARC-c (Clark et al., 2018) to evaluate knowledge recall capabilities. For HumanEval, since HumanEval does not contain a training set, we follow Tian et al. (2024) to finetune the base LLM on CodeAlpaca (Chaudhary,

- 2023) and report the Pass@1 metric on HumanEval. For all other datasets, we finetune the base LLM on their training split and report the accuracy metric on their test split. In this work, we use Llama

3 8B (Dubey et al., 2024) as the base LLM. Besides that, we also report the number of activated parameters (in billion, B) under the best-performing hyperparameters.

Implementation details. We train all methods using the same number of epochs, learning rate schedule, gradient accumulation steps and machine type. All methods are trained using the LLaMAFactory (Zheng et al., 2024) framework and evaluated using the OpenCompass (Contributors, 2023) framework. For the no-tuning few-shot method, we use 4 shots for GSM8K and HumanEval and 5 shots for ARC-c.

- 4.2 MAIN RESULTS

We evaluate the performance of various fine-tuning strategies on three representative tasks: HumanEval (code generation), GSM8K (math reasoning), and ARC-c (knowledge recall). As shown in Table 1, our ReMix consistently outperforms all baselines across these benchmarks while maintaining strong parameter efficiency.

From a performance standpoint, ReMix surpasses all baseline methods, achieving an average accuracy improvement of 2.82 over the strongest competing approach. Specifically, ReMix outperforms the best Prefix Injection baseline by a substantial 25.88, the best Weight Modulation baseline by 2.82, and the strongest Mixture competitor by 3.34 on average across the three tasks. On HumanEval, ReMix achieves a Pass@1 of 32.93, outperforming the best baseline, (IA)3, by 1.83. For

- Table 2: The fact that our method significantly outperforms rank-kr LoRA clearly shows that our ReMix can activate diverse LoRA subsets: If the activated subset were always the same subset, then the results would be the same as rank-kr LoRA.

Method k = 1 k = 2 k = 4 Rank-kr LoRA 56.10 54.51 59.21 Ours: k rank-r LoRAs 56.18 59.67 64.22

Table 3: Our ReMix is not sensitive to routing weight ω.

Method LoRA-type ω rsLoRA-type ω ReMix (Ours) 53.30 55.72

GSM8K, ReMix attains an accuracy of 65.66, showing a clear gain of 3.19 over the best competitors (rsLoRA and HydraLoRA). On ARC-c, ReMix reaches 83.73, exceeding the best-performing lowrank method DoRA by 0.34. These results highlight the consistent advantages of our reinforcementtrained router across diverse task types. Notably, within the Mixture methods, ReMix provides consistent improvements, suggesting that reinforcement-guided, balance-aware routing enhances both reasoning-intensive tasks (e.g., GSM8K) and generation tasks (e.g., HumanEval), while preserving strong retrieval performance on ARC-c.

In terms of parameter efficiency, ReMix achieves these performance gains with a competitive budget of only 0.070B trainable parameters. Compared to other mixture methods, this represents a 90% reduction relative to the most parameter-heavy baseline VB-LoRA (0.675B), and a 31% reduction compared to the most effective baseline MixLoRA (0.101B). Even when compared to the lightweight rsLoRA (0.028B), ReMix delivers a +2.82 average-accuracy improvement at the cost of only 0.042B more parameters, demonstrating a superior accuracy-to-parameter trade-off. Overall, these results confirm that reinforcement-guided mixture routing achieves state-of-the-art accuracy with minimal and often reduced parameter overhead.

- 4.3 ABLATION STUDIES

No RLOO No top-k Ours

45

50

55

60

65

Accuracy(%)

Figure 4: Both of our proposed RLOO and top-k selection contribute significantly to the strong performance of our ReMix.

To understand the contributions of the key components in our proposed ReMix (i.e., RLOO for router training and top-k LoRA selection for inference), we conducte ablation studies on GSM8K comparing its performance against the ablated variants with each component removed. The results are presented in Figure 4, which visualizes the accuracy achieved by different configurations.

From Figure 4, we observe that our full ReMix method achieves the highest accuracy among all ablated variants. When removing the RLOO from our finetuning procedure ReMix (No RLOO), we observe a significant drop in accuracy compared to the full ReMix, indicating that RLOO plays a crucial role in enhancing the model’s performance. Similarly, disabling the top-k LoRA selection (No top-k) also results in lower accuracy than the complete ReMix, demonstrating the importance of this component in optimizing the model performance. These findings underscore the value of integrating both RLOO and top-k selection into our ReMix method.

- 4.4 DIVERSITY OF ACTIVATED LORA SUBSETS

In this subsection, we empirically verify the diversity of activated LoRA subsets. If the activated subset were always the same subset, then this method would be the same as rank-kr LoRA. Therefore, we compare our method (a mixture of k rank-r LoRAs) with a single rank-kr LoRA, which has the same number of LoRA parameters. The results for r = 8 on GSM8K under various k are presented in Table 2.

The fact that our ReMix significantly outperforms rank-kr LoRA demonstrates the diversity of selected LoRA subsets. For instance, our accuracy 64.22 for k = 4 significantly outperforms the

Table 4: Our ReMix significantly outperforms MixLoRA under similar training time.

Method Per-Step Time Total Time Accuracy

MixLoRA 8.95s 1:12:56 50.34 ReMix (Ours) 9.87s 1:28:21 58.38

Table 5: Our ReMix consistently improves as we scale up the number k of activated LoRAs.

Method k = 1 k = 2 k = 3 k = 4 ReMix (Ours) 56.18 59.67 61.33 64.22

accuracy 59.21 of rank-32 LoRA. This clearly demonstrates that our method is able to choose different subsets appropriately.

- 4.5 TRAINING EFFICIENCY

In this subsection, we study the training efficiency of our proposed method. Note that MixLoRA can be regarded as an ablated variant where our reinforcement router is replaced with an ordinary learnable router. Hence, we compare our ReMix and MixLoRA under comparable training time to show the training efficiency of our proposed ReMix. The results are presented in Table 4.

As shown in Table 4, our ReMix achieves an accuracy of 58.38% with a total training time of 1:28:21, while MixLoRA achieves an accuracy of 50.34% in 1:12:56. Although our ReMix consumes only 10% more training time than MixLoRA, it yields a substantial relative improvement of 15.97% in accuracy. This demonstrates that our ReMix still retains strong performance even under small training compute budget.

- 4.6 SCALING THE NUMBER OF ACTIVATED LORAS

In this subsection, we study how scaling the number k of activated LoRAs benefits the predictive accuracy. Intuitively, the choice of k depends on the tradeoff between efficiency and accuracy.

Theoretically, since the number of size-k subsets (i.e., nk ) increases with k when k ≤ n/2, we would expect accuracy to increase with k correspondingly. To verify this, we conduct experiments

with n = 8 under various k on GSM8K. The results are shown in Table 5. Indeed, as shown in the table, Our ReMix consistently achieves stronger results under larger k whenever k ≤ n/2.

- 4.7 SCALING THE TRAINING COMPUTE

Since our ReMix incorporates RL-based gradient estimator, we can effectively scale up training compute by increasing the number M of sampled selections. To evaluate how training compute scaling benefits our ReMix, we examine its performance under varying numbers M of sampled selections. As shown in Figure 5, increasing M from 2 to 32 leads to a steady improvement in accuracy, rising from 56.03% to 58.83%. This indicates that our ReMix effectively leverages additional computational resources to enhance its performance. Notably, the consistent gains observed across different scales suggest that further increases in M could yield even better results. This demonstrates that ReMix offers a favorable trade-off between training efficiency and performance. In stark contrast, existing methods do not benefit from training compute scaling because their deterministic training has fixed training compute, which underscores the unique advantage offered by ReMix in utilizing increased training compute to achieve improved outcomes.

- 55

- 56

- 57

- 58

- 59

58.83%

Accuracy(%)

57.47%

56.79%

56.03%

2 4 8 32

Number M of sampled selections

Figure 5: Our proposed ReMix can further benefit from scaling up the training compute. In contrast, HydraLoRA and MixLoRA have fixed training compute and thus cannot benefit from scaling up training compute.

- 4.8 LORA V.S. RSLORA ROUTING WEIGHT

#### √

We compare LoRA-type ω := 2/kr and rsLoRA-type ω := 2/

kr under k = 3 on GSM8K. The results are shown in Table 3. We find that our method ReMix is not sensitive to the choice of ω. As shown in Table 3, the performance has only very small difference under these two versions of ω.

- 5 RELATED WORK

Parameter-efficient fine-tuning (PEFT) aims to reduce the number of trainable parameters while achieving strong task performance. Due to the page limit, please refer to Appendix A for related work on general PEFT. More recent efforts in PEFT have explored new multi-LoRA architectures that go beyond single low-rank adapters by explicitly restructuring how multiple LoRA modules are organized and combined, offering advantages on complex data distributions. LoraHub (Huang et al., 2023) introduces a dynamic composition framework that integrates multiple LoRAs at the architectural level, enabling cross-task generalization without retraining by assembling adapters into a unified pipeline. MultiLoRA (Wang et al., 2023b) modifies the structural initialization of LoRA subspaces and horizontally expands adapters across layers, thereby mitigating the dominance of top singular vectors and achieving more balanced representations in multi-task learning. HydraLoRA (Tian et al., 2024) departs from the symmetric LoRA design and proposes an asymmetric architecture that decouples the projection and update pathways, substantially improving parameter and training efficiency. Beyond linear compositions, S’MoRE (Zeng et al., 2025a) integrates LoRA with mixture-of-experts style routing by hierarchically decomposing expert weights into low-rank residual components and routing them through a structured multi-layer architecture. Meanwhile, LoRAFlow (Wang et al., 2024) rethinks the architecture for generative tasks by embedding a lightweight, token-level fusion gate that dynamically modulates multiple LoRAs during inference, and MultLFG (Roy et al., 2025) introduces a frequency-aware fusion mechanism that structurally guides LoRA composition across denoising steps.

- 6 CONCLUSION

In this paper, we investigate the problem of imbalanced routing weights that hinder effective LoRA utilization, and propose a reinforcement-based router design named ReMix to address this problem. Extensive experiments across diverse benchmarks demonstrate that our ReMix consistently outperforms state-of-the-art parameter-efficient finetuning methods, achieving superior predictive power and computational efficiency.

REFERENCES

Wenxuan Bao, Ruxi Deng, Ruizhong Qiu, Tianxin Wei, Hanghang Tong, and Jingrui He. Latte: Collaborative test-time adaptation of vision-language models in federated learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

Burak Bartan, Ruizhong Qiu, Rafael Esteves, Yuwei Ren, Weiliang Will Zeng, and An Chen. FineAMP: optimization-based automatic mixed precision quantization for efficient diffusion model inference. The 17th International OPT Workshop on Optimization for Machine Learning, 2025.

Zygmunt Wilhelm Birnbaum. An inequality for Mill’s ratio. The Annals of Mathematical Statistics, 13(2):245–246, 1942.

Sahil Chaudhary. Code alpaca: An instruction-following llama model for code generation. https: //github.com/sahil280114/codealpaca, 2023.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pond´e de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol,

Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. CoRR, abs/2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.

Sirui Chen, Yunzhe Qi, Mengting Ai, Yifan Sun, Ruizhong Qiu, Jiaru Zou, and Jingrui He. Influence-preserving proxies for gradient-based data selection in LLM finetuning. In The Fourteenth International Conference on Learning Representations, 2026.

Yifan Chen, Devamanyu Hazarika, Mahdi Namazifar, Yang Liu, Di Jin, and Dilek Hakkani-Tur. Inducer-tuning: Connecting prefix-tuning and adapter-tuning. arXiv preprint arXiv:2210.14469, 2022.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.

OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/opencompass, 2023.

Chengming Cui, Tianxin Wei, Ziyi Chen, Ruizhong Qiu, Zhichen Zeng, Zhining Liu, Xuying Ning, Duo Zhou, and Jingrui He. AdaFuse: adaptive ensemble decoding with test-time scaling for LLMs. arXiv preprint, 2026.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Robert D. Gordon. Values of Mills’ ratio of area to bounding ordinate and of the normal probability integral for large values of the argument. The Annals of Mathematical Statistics, 12(3):364–366, 1941.

Marian Grendar. Entropy and effective support size. Entropy, 8(3):169–174, 2006. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing

human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pp. 1026–1034, 2015.

Shwai He, Liang Ding, Daize Dong, Miao Zhang, and Dacheng Tao. Sparseadapter: An easy approach for improving the parameter-efficiency of adapters. arXiv preprint arXiv:2210.04284, 2022.

Xinyu He, Jian Kang, Ruizhong Qiu, Fei Wang, Jose Sepulveda, and Hanghang Tong. On the sensitivity of individual fairness: Measures and robust algorithms. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, 2024.

Xinyu He, Chenhan Xiao, Haoran Li, Ruizhong Qiu, Zhe Xu, Yang Weng, Jingrui He, and Hanghang Tong. PowerGrow: feasible co-growth of structures and dynamics for power grid synthesis. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2026.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. Lorahub: Efficient cross-task generalization via dynamic lora composition. arXiv preprint arXiv:2307.13269, 2023.

Shibo Jie, Haoqing Wang, and Zhi-Hong Deng. Revisiting the parameter efficiency of adapters from the perspective of precision redundancy. In Proceedings of the IEEE/CVf international conference on computer vision, pp. 17217–17226, 2023.

Damjan Kalajdzievski. A rank stabilization scaling factor for fine-tuning with LoRA. arXiv preprint arXiv:2312.03732, 2023.

Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 REINFORCE samples, get a baseline for free! In ICLR 2019 workshop: Deep RL Meets Structured Prediction, 2019.

Minh Le, Chau Nguyen, Huy Nguyen, Quyen Tran, Trung Le, and Nhat Ho. Revisiting prefix-tuning: Statistical benefits of reparameterization among prompts. arXiv preprint arXiv:2410.02200, 2024.

Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.

Dengchun Li, Yingzi Ma, Naizheng Wang, Zhiyuan Cheng, Lei Duan, Jie Zuo, Cal Yang, and Mingjie Tang. Mixlora: Enhancing large language models fine-tuning with lora based mixture of experts. arXiv preprint arXiv:2404.15159, 2024a.

Gaotang Li, Ruizhong Qiu, Xiusi Chen, Heng Ji, and Hanghang Tong. Beyond log likelihood: Probability-based objectives for supervised fine-tuning across the model capability continuum. ICLR 2026 Workshop on Scaling Post-training for LLMs, 2026.

Mufei Li, Dongqi Fu, Limei Wang, Si Zhang, Hanqing Zeng, Kaan Sancak, Ruizhong Qiu, Haoyu Peter Wang, Xiaoxin He, Xavier Bresson, Yinglong Xia, Chonglin Sun, and Pan Li. Haystack engineering: Context engineering meets the long-context challenge in large language models. NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling, 2025a.

Ting-Wei Li, Ruizhong Qiu, and Hanghang Tong. Graph data selection for domain adaptation: A model-free approach. In Advances in Neural Information Processing Systems 38, 2025b.

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021.

Yang Li, Shaobo Han, and Shihao Ji. VB-LoRA: Extreme parameter efficient fine-tuning with vector banks. Advances in Neural Information Processing Systems, 37:16724–16751, 2024b.

Xiao Lin, Zhining Liu, Dongqi Fu, Ruizhong Qiu, and Hanghang Tong. BackTime: backdoor attacks on multivariate time series forecasting. In Advances in Neural Information Processing Systems 37, 2024.

Xiao Lin, Zhicheng Tang, Weilin Cong, Mengyue Hang, Kai Wang, Yajuan Wang, Zhichen Zeng, Ting-Wei Li, Hyunsik Yoo, Zhining Liu, Xuying Ning, Ruizhong Qiu, Wen-Yen Chen, Shuo Chang, Rong Jin, Huayu Li, and Hanghang Tong. Mixture of sequence: Theme-aware mixtureof-experts for long-sequence recommendation. In Proceedings of the ACM Web Conference 2026, 2026.

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. Advances in Neural Information Processing Systems, 35:1950–1965, 2022.

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, KwangTing Cheng, and Min-Hung Chen. DoRA: Weight-decomposed low-rank adaptation. In Forty-first International Conference on Machine Learning, 2024a.

Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. Ptuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks. arXiv preprint arXiv:2110.07602, 2021a.

Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. Gpt understands, too. arXiv preprint arXiv:2103.10385, 2021b.

Zhining Liu, Ruizhong Qiu, Zhichen Zeng, Hyunsik Yoo, David Zhou, Zhe Xu, Yada Zhu, Kommy Weldemariam, Jingrui He, and Hanghang Tong. Class-imbalanced graph learning without class rebalancing. In Proceedings of the 41st International Conference on Machine Learning, 2024b.

Zhining Liu, Ruizhong Qiu, Zhichen Zeng, Yada Zhu, Hendrik Hamann, and Hanghang Tong. AIM: attributing, interpreting, mitigating data-encoded unfairness. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024c.

Zhining Liu, Ze Yang, Xiao Lin, Ruizhong Qiu, Tianxin Wei, Yada Zhu, Hendrik Hamann, Jingrui He, and Hanghang Tong. Breaking silos: Adaptive model fusion unlocks better time series forecasting. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

Aleksandar Petrov, Philip HS Torr, and Adel Bibi. When do prompting and prefix-tuning work? a theory of capabilities and limitations. arXiv preprint arXiv:2310.19698, 2023.

Aniket Roy, Maitreya Suin, Ketul Shah, and Rama Chellappa. Multlfg: Training-free multi-lora composition using frequency-domain guidance. arXiv preprint arXiv:2505.20525, 2025.

Andreas R¨uckl´e, Gregor Geigle, Max Glockner, Tilman Beck, Jonas Pfeiffer, Nils Reimers, and Iryna Gurevych. Adapterdrop: On the efficiency of adapters in transformers. arXiv preprint arXiv:2010.11918, 2020.

Zhengxiang Shi and Aldo Lipani. Dept: Decomposed prompt tuning for parameter-efficient finetuning. arXiv preprint arXiv:2309.05173, 2023.

Chunlin Tian, Zhan Shi, Zhijiang Guo, Li Li, and Chengzhong Xu. Hydralora: An asymmetric lora architecture for efficient fine-tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Mojtaba Valipour, Mehdi Rezagholizadeh, Ivan Kobyzev, and Ali Ghodsi. Dylora: Parameter efficient tuning of pre-trained models using dynamic search-free low-rank adaptation. arXiv preprint arXiv:2210.07558, 2022.

Chaozheng Wang, Yuanhang Yang, Cuiyun Gao, Yun Peng, Hongyu Zhang, and Michael R Lyu. No more fine-tuning? an experimental evaluation of prompt tuning in code intelligence. In Proceedings of the 30th ACM joint European software engineering conference and symposium on the foundations of software engineering, pp. 382–394, 2022.

Dingsu Wang, Yuchen Yan, Ruizhong Qiu, Yada Zhu, Kaiyu Guan, Andrew J. Margenot, and Hanghang Tong. Networked time series imputation via position-aware graph enhanced variational autoencoders. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023a.

Hanqing Wang, Bowen Ping, Shuo Wang, Xu Han, Yun Chen, Zhiyuan Liu, and Maosong Sun. Lora-flow: Dynamic lora fusion for large language models in generative tasks. arXiv preprint arXiv:2402.11455, 2024.

Yiming Wang, Yu Lin, Xiaodong Zeng, and Guannan Zhang. Multilora: Democratizing lora for better multi-task learning. arXiv preprint arXiv:2311.11501, 2023b.

Tianxin Wei, Xuying Ning, Xuxing Chen, Ruizhong Qiu, Yupeng Hou, Yan Xie, Shuang Yang, Zhigang Hua, and Jingrui He. CoFiRec: coarse-to-fine tokenization for generative recommendation. arXiv preprint, 2025.

Tianxin Wei, Ting-Wei Li, Zhining Liu, Xuying Ning, Ze Yang, Jiaru Zou, Zhichen Zeng, Ruizhong Qiu, Xiao Lin, Dongqi Fu, Zihao Li, Mengting Ai, Duo Zhou, Wenxuan Bao, Yunzhe Li, Gaotang Li, Cheng Qian, Yu Wang, Xiangru Tang, Yin Xiao, Liri Fang, Hui Liu, Xianfeng Tang, Yuji Zhang, Chi Wang, Jiaxuan You, Heng Ji, Hanghang Tong, and Jingrui He. Agentic reasoning for large language models: A survey. arXiv preprint, 2026a.

Tianxin Wei, Ruizhong Qiu, Yifan Chen, Yunzhe Qi, Jiacheng Lin, Wenxuan Bao, Wenju Xu, Sreyashi Nag, Ruirui Li, Hanqing Lu, Zhengyang Wang, Chen Luo, Hui Liu, Suhang Wang, Jingrui He, Qi He, and Xianfeng Tang. DiffKGW: stealthy and robust diffusion model watermarking. Transactions on Machine Learning Research, 2026b.

Zhe Xu, Ruizhong Qiu, Yuzhong Chen, Huiyuan Chen, Xiran Fan, Menghai Pan, Zhichen Zeng, Mahashweta Das, and Hanghang Tong. Discrete-state continuous-time diffusion for graph generation. In Advances in Neural Information Processing Systems 37, 2024.

Adam X Yang, Maxime Robeyns, Xi Wang, and Laurence Aitchison. Bayesian low-rank adaptation for large language models. arXiv preprint arXiv:2308.13111, 2023.

Hyunsik Yoo, Zhichen Zeng, Jian Kang, Ruizhong Qiu, David Zhou, Zhining Liu, Fei Wang, Charlie Xu, Eunice Chan, and Hanghang Tong. Ensuring user-side fairness in dynamic recommender systems. In Proceedings of the ACM Web Conference 2024, 2024.

Hyunsik Yoo, SeongKu Kang, Ruizhong Qiu, Charlie Xu, Fei Wang, and Hanghang Tong. Embracing plasticity: Balancing stability and plasticity in continual recommender systems. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2025a.

Hyunsik Yoo, Ruizhong Qiu, Charlie Xu, Fei Wang, and Hanghang Tong. Generalizable recommender system during temporal popularity distribution shifts. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2025b.

Qi Yu, Zhichen Zeng, Yuchen Yan, Zhining Liu, Baoyu Jing, Ruizhong Qiu, Ariful Azad, and Hanghang Tong. PlanetAlign: a comprehensive Python library for benchmarking network alignment. In The Fourteenth International Conference on Learning Representations, 2026.

Yuhang Zang, Wei Li, Kaiyang Zhou, Chen Huang, and Chen Change Loy. Unified vision and language prompt learning. arXiv preprint arXiv:2210.07225, 2022.

Hanqing Zeng, Yinglong Xia, Zhuokai Zhao, Gilbert Jiang, Qiang Zhang, Jiayi Liu, Lizhu Zhang, Xiangjun Fan, and Benyu Zhang. S’MoRE: Structural mixture of residual experts for llm finetuning. arXiv preprint arXiv:2504.06426, 2025a.

Zhichen Zeng, Ruizhong Qiu, Zhe Xu, Zhining Liu, Yuchen Yan, Tianxin Wei, Lei Ying, Jingrui He, and Hanghang Tong. Graph mixup on approximate Gromov–Wasserstein geodesics. In Proceedings of the 41st International Conference on Machine Learning, 2024.

Zhichen Zeng, Mengyue Hang, Xiaolong Liu, Xiaoyi Liu, Xiao Lin, Ruizhong Qiu, Tianxin Wei, Zhining Liu, Siyang Yuan, Chaofei Yang, Yiqun Liu, Hang Yin, Jiyan Yang, and Hanghang Tong. Hierarchical LoRA MoE for efficient CTR model scaling. arXiv preprint, 2025b.

Zhichen Zeng, Wenxuan Bao, Xiao Lin, Ruizhong Qiu, Tianxin Wei, Xuying Ning, Yuchen Yan, Chen Luo, Monica Xiao Cheng, Jingrui He, and Hanghang Tong. Subspace alignment for visionlanguage model test-time adaptation. arXiv preprint, 2026a.

Zhichen Zeng, Ruizhong Qiu, Wenxuan Bao, Tianxin Wei, Xiao Lin, Yuchen Yan, Tarek F. Abdelzaher, Jiawei Han, and Hanghang Tong. Pave your own path: Graph gradual domain adaptation on fused Gromov–Wasserstein geodesics. Transactions on Machine Learning Research, 2026b.

Zhichen Zeng, Qi Yu, Xiao Lin, Ruizhong Qiu, Xuying Ning, Tianxin Wei, Yuchen Yan, Jingrui He, and Hanghang Tong. Harnessing consistency for robust test-time LLM ensemble. Findings of EACL 2026, 2026c.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adalora: Adaptive budget allocation for parameterefficient fine-tuning. arXiv preprint arXiv:2303.10512, 2023.

Yunkai Zhang, Qiang Zhang, Diji Yang, Ryan Lin, Ruizhong Qiu, Benyu Zhang, Hanchao Yu, Jason Liu, Yinglong Xia, Zhuokai Zhao, Lizhu Zhang, Xiangjun Fan, Zhuoran Yu, Abhishek Kumar, and Zeyu Zheng. Guiding generative recommender systems with structured human priors via multi-head decoding. In Proceedings of the ACM Web Conference 2026, 2026.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

Jiaru Zou, Yikun Ban, Zihao Li, Yunzhe Qi, Ruizhong Qiu, Ling Yang, and Jingrui He. Transformer copilot: Learning from the mistake log in LLM fine-tuning. In Advances in Neural Information Processing Systems 38, 2025a.

Jiaru Zou, Xiyuan Yang, Ruizhong Qiu, Gaotang Li, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, and Ling Yang. Latent collaboration in multi-agent systems. arXiv preprint, 2025b.

CONTENTS

- A Related Work (Cont’d) 16
- B Theoretical Proofs 16

- B.1 Proof of Theorem 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- B.2 Proof of Theorem 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- A RELATED WORK (CONT’D)

While neural networks has been prevalent in various domains (Cui et al., 2026; He et al., 2026; Yu et al., 2026; He et al., 2024; Yoo et al., 2025a;b; 2024; Wang et al., 2023a), Transformers have nowadays become the de facto neural architecture (Wei et al., 2026a; 2025; 2026b; Chen et al., 2026; Li et al., 2026; 2025a;b; Zeng et al., 2026a;b;c; 2025b; 2024; Zhang et al., 2026; Lin et al., 2026; 2024; Bartan et al., 2025; Zou et al., 2025a;b; Liu et al., 2025; 2024b;c; Bao et al., 2025; Xu et al., 2024). PEFT methods for Transformers can be broadly categorized into four groups: prompt tuning, prefix tuning, adapter-based methods, and low-rank adaptation methods. Early methods such as prompt tuning (Liu et al., 2021a; Shi & Lipani, 2023; Lester et al., 2021; Zang et al., 2022; Wang et al., 2022) and prefix tuning (Li & Liang, 2021; Le et al., 2024; Chen et al., 2022; Petrov et al., 2023) introduce small continuous prompts, but often struggle to scale to deeper layers or larger models due to limited expressivity. Adapter-based methods (He et al., 2022; R¨uckl´e et al., 2020; Jie et al., 2023) mitigate some of these issues by inserting lightweight bottleneck modules into transformer layers. However, as the depth and dimensionality of models increase, the parameter overhead of adapters can become substantial, creating significant bottlenecks in computation and scalability. To address these limitations, low-rank adaptation methods (Hu et al., 2021; Valipour et al., 2022; Zhang et al., 2023; Yang et al., 2023) are proposed. These methods inject rank-constrained updates into weight matrices, striking a favorable balance between expressivity and parameter cost, and have become a de facto standard for many adaptation tasks. Specifically, LoRA (Hu et al., 2022) introduces two trainable low-rank matrices while keeping the original model weights frozen. By training these matrices to approximate parameter perturbations, LoRA achieves effective fine-tuning with minimal overhead. Building on this idea, DyLoRA (Valipour et al., 2022) dynamically trains LoRA modules across a range of ranks within a predefined budget rather than fixing the rank. AdaLoRA (Zhang et al., 2023) reformulates parameter perturbations using singular value decomposition (SVD), finetuning across the three SVD components for improved flexibility. Laplace-LoRA (Yang et al., 2023) takes a Bayesian perspective, applying a post-hoc Laplace approximation to the posterior distribution over LoRA parameters, thereby offering a principled uncertainty-aware extension.

- B THEORETICAL PROOFS B.1 PROOF OF THEOREM 1 Before stating our proof of Theorem 1, we present a few technical lemmata that we will employ.

2/2, Φ(z) := −∞ z φ(x)dx, and Φ(z) := 1 − Φ(z) (z ∈ R) denote the probability density function, the cumulative distribution function, and the complementary cumulative distribution function of the standard Gaussian distribution N(0,1), respectively.

Let φ(z) := √12πe−x

- Lemma 3 (a Gaussian gap estimate). For every z ∈ R and α > 0,

α √2π

. (7)

Φ(z + α) − Φ(z) ≤

−t2/2

Proof. Since Φ′(t) = φ(t) = e

√2π , then

Φ(z + α) − Φ(z) =

z+α

Φ′(t)dt =

z

2/2

z+α

e−t

√2π

dt ≤

z

z+α

1 √2π

α √2π

dt =

.

z

| |
|---|

- Lemma 4 (a Gaussian upper-tail gap estimate). For any z ≥ 0 and any α > 0,

Φ(z + α) − Φ(z) ≤

√

2π(Φ(α) − Φ(0))φ(z) ≤ α φ(z). (8)

Proof. Define a function h : R≥0 → R as

h(z) :=

Φ(z + α) − Φ(z) φ(z)

, z ≥ 0. (9)

Since Φ′(z) = φ(z), and φ′(z) = −zφ(z), then

h′(z) =

(φ(z + α) − φ(z))Φ′(z) + (Φ(z + α) − Φ(z))(−φ′(z)) φ(z)2

(10)

=

(φ(z + α) − φ(z))φ(z) + (Φ(z + α) − Φ(z))(zφ(z)) φ(z)2

(11)

=

φ(z + α) − φ(z) + z(Φ(z + α) − Φ(z)) φ(z)

(12)

=

z+α z φ′(t)dt+z z z+α Φ′(t)dt

φ(z)

(13)

=

z+α z (−tφ(t))dt+z z z+α φ(t)dt

φ(z)

(14)

= −

z+α z (t − z)φ(t)dt

φ(z)

< 0. (15)

Hence, h(z) is a decreasing function. It follows from Lemma 3 that

Φ(z + α) − Φ(z) φ(z)

= h(z) ≤ h(0) =

Φ(α) − Φ(0) φ(0)

=

√

2π(Φ(α) − Φ(0)) (16)

=

√

2π

α

0

Φ′(t)dt =

√

2π

α

0

e−t

2/2

√2π

dt ≤

√

2π

α

0

1 √2π

dt =

α

0

dt = α.

| |
|---|

- Lemma 5 (a Gaussian inverse estimate). For every 0 < v ≤ 21,

1 v

φ(Φ−1(1 − v)) ≤ v 2ln

. (17)

Proof. Let z := Φ−1(1 − v) ≥ 0, so that v = 1 − Φ(z) = Φ(z). Note that it is equivalent to show that

ln

φ(z)2 2Φ(z)2

1 Φ(z) ≥

. (18)

Define a function h : R≥0 → R as

h(z) := ln

φ(z)2 2Φ(z)2

1 Φ(z) −

, z ≥ 0. (19)

Since z ≥ 0, then by Gordon (1941),

φ(z) Φ(z) ≥ z ≥

z 2

. (20)

and by Birnbaum (1942),

φ(z) Φ(z) ≤

2 √z2 + 4 − z

=

√z2 + 4 + z 2

z 2

=

+

√z2 + 4 2

. (21)

Together, we have

√z2 + 4 2

z 2

z 2

φ(z) Φ(z) ≤

. (22) Furthermore, since Φ′(z) = −φ(z), and φ′(z) = −zφ(z),

<

+

2

φ(z) Φ(z)

φ(z) Φ(z) −

φ(z) Φ(z)

h′(z) =

(23)

1 + z

√z2 + 4 2

√z2 + 4 2 −

z 2

φ(z) Φ(z)

φ(z) Φ(z) −

z 2

φ(z) Φ(z)

(24) ≥ 0. (25)

=

+

+

Hence, the function h(z) is non-decreasing w.r.t. z. It follows that for any 0 < v ≤ 12,

φ(Φ(1 − v))2 2v2

φ(z)2 2Φ(z)2

1 v −

1 Φ(z) −

1 π

> 0. (26)

= h(z) ≥ h(0) = ln2 −

ln

= ln

#### Therefore, φ(Φ−1(1 − v)) ≤ v 2ln v1.

| |
|---|

- Lemma 6 (a Gamma integral). Let Γ(β) denote the Gamma function (β > 0). For any α,β > 0, 1

0

tα−1 ln

1 t

β−1

dt =

Γ(β) αβ

. (27) In particular,

1

0

t ln

1 t

dt =

Γ 2 1 + 1 (1 + 1)1/

2+1 =

√π 4√2

. (28)

Proof. Let z := α ln 1t. Then,

1

0

tα−1 ln

1 t

β−1

dt =

+∞

0

e−z/α α−1

z α

β−1 e−z/α α

dz (29)

=

1 αβ

+∞

0

e−zzβ−1 dz =

1 αβ

Γ(β).

| |
|---|

- Lemma 7 (a mixed integral estimate). For every α > 0 and 0 < β ≤ α, β

√π 4√2

√

α t

te−t ln

lnα (1 − e−β+ln(β+1)) +

. (30)

dt ≤

0

Proof. Since ln 1t ≥ 0 only when 0 ≤ t ≤ 1, then by the triangle inequality,

α t

1 t ≤ lnα + [0≤t≤1] ln

1 t

(31)

ln

= lnα + ln

√

√

1 t

1 t

. (32) It follows from Lemma 6 that

≤

lnα + [0≤t≤1] ln

=

lnα + [0≤t≤1] ln

√

β

β

α t

1 t

te−t ln

te−t

dt (33)

dt ≤

lnα + [0≤t≤1] ln

0

0

√

min{1,β}

β

1 t

te−t dt+

te−t ln

dt (34)

=

lnα

0

0

√

β

1

1 t

te−t dt+

te−t ln

dt (35)

≤

lnα

0

0

√

β

1

1 t

te−t dt+

dt (36)

≤

t ln

lnα

0

0

√π 4√2

√

lnα (1 − e−β+ln(β+1)) +

=

.

| |
|---|

- Lemma 8 (an integer function bound). For every integer n ≥ 3, √2

(n − 2)2

ln(n − 2)(1 − e−

n

2 −1+ln n2 ) +

√π 4√2 ≤

3 2

π ln3

√

lnn n(n − 1)

. (37)

Proof. Define a function h : N≥3 → R:

h(n) :=

√2

(n−2)2 ln(n − 2)(1 − e−n2 −1+ln n2 ) +

√π 4√2

√

ln n n(n−1)

, n ≥ 3. (38)

Note that when n ≥ 9, we have

h(n) ≤

√2

(n−2)2 ln(n − 2) +

√π 4√2

√

ln n n(n−1)

=

√2

(n−2)2 ln(n − 2) +

√π 4√2

√

ln n n(n−1)

(39)

=

√

2

ln(n − 2) lnn

+

1 4

π lnn

1 +

2 n − 2

+

3 (n − 2)2

(40)

≤

√

2 +

1 4

π lnn

1 +

2 n − 2

+

3 (n − 2)2

(41)

≤

√

2 +

1 4

π ln9

1 +

2 9 − 2

+

3 (9 − 2)2

(42)

=

√

2 +

1 4

π ln9

72 49

< 2.52. (43)

It follows that

√2

(n−2)2 ln(n − 2) +

√π 4√2

√

ln n n(n−1)

= h(n) (44)

≤ max h(3),h(4),...,h(8),

√

2 +

1 4

π ln9

72 49

(45)

= h(3) =

3 2

π ln3

< 2.54.

| |
|---|

- Lemma 9 (a Gaussian integral estimate). For every integer n ≥ 3,

√

+∞

lnn n(n − 1)

3 2

π ln3

Φ(z)n−2φ(z)2 dz ≤

0

. (46)

Proof. Let v := 1 − Φ(z) and t := (n − 2)v. By the fact that 1 − v ≤ e−v and Lemmas 5, 7, & 8,

+∞

1/2

1/2

Φ(z)n−2φ(z)2 dz =

(1 − v)n−2φ(Φ−1(1 − v))dv ≤

(e−v)n−2φ(Φ−1(1 − v))dv

0

0

0

(47) ≤

√2 (n − 2)2

n 2 −1

1/2

n − 2 t

1 v

(e−v)n−2v 2ln

te−t ln

dt (48)

dv =

0

0

√2 (n − 2)2

√π 4√2

n

2 −1+ln n2 ) +

ln(n − 2)(1 − e−

(49)

≤

√

√

3 2

lnn n(n − 1)

lnn n(n − 1)

π ln3

≤

< 2.54

.

| |
|---|

With the technical lemmata above, we are now ready to prove Theorem 1.

- Proof of Theorem 1. Let ξ := P(l)x(l) denote the logits of routing weights, so that π(l) = softmax(ξ). Let ξ(1) ≥ ··· ≥ ξ(n) denote the order statistics of ξ (i.e., ξ(1) is the largest entry of ξ, ξ(2) is the second largest entry of ξ, etc.). Note that

n i=1 softmax(ξ)i 2 n i=1 softmax(ξ)2i

n i=1 πi(l) 2

ESS(π(l)) = ∥π(l)∥21 ∥π(l)∥22

(50)

=

=

n i=1(πi(l))2

i 2

n i=1 eξ(i) 2

n i=1 eξ(i) 2

n i=1 eξ

(51)

n i=1(eξ(i))2 ≤

=

=

n i=1(eξi)2

(eξ(1))2

n

n

2

2

1 eξ(1)−ξ(2)

1 eξ(1)−ξ(i)

(52)

≤ 1 +

= 1 +

i=2

i=2

n − 1 eξ(1)−ξ(2)

2

2

1 eξ(1)−ξ(2)−ln(n−1)

. (53) Since P(l) have i.i.d. N(0,σ2) entries, then ξ = P(l)x(l) have i.i.d N(0,σ2∥x(l)∥22) entries. Let

= 1 +

= 1 +

1

. (54)

κ :=

3 2

ln 3 lnn + √2π2 1

π

n−log2 n−1

For any 0 < δ < 1, with z(i) := σξ∥(xi)−0

(l)∥2 (i = 1,2), by Lemmas 3, 4, & 9, P[ξ(1) − ξ(2) ≤ δκσ∥x(l)∥2] = P[z(1) − z(2) ≤ δκ] (55)

+∞

z(2)+δκ

n(n − 1)φ(z(1))φ(z(2))Φ(z(2))n−2 dz(1) dz(2) (56)

=

−∞

z(2)

+∞

z(2)+δκ

φ(z(1))dz(1) φ(z(2))Φ(z(2))n−2 dz(2) (57)

= n(n − 1)

−∞

z(2)

+∞

(Φ(z(2) + δκ) − Φ(z(2)))φ(z(2))Φ(z(2))n−2 dz(2) (58)

= n(n − 1)

−∞

+∞

0

(Φ(z(2) + δκ) − Φ(z(2)))φ(z(2))Φ(z(2))n−2 dz(2) (59)

= n(n − 1)

+

−∞

0

+∞

0

δκ √2π

φ(z(2))Φ(z(2))n−2 dz(2) +

δκφ(z(2))φ(z(2))Φ(z(2))n−2 dz(2) (60)

≤ n(n − 1)

−∞

0

+∞

0

1 √2π

φ(z(2))Φ(z(2))n−2 dz(2) +

Φ(z(2))n−2φ(z(2))2 dz(2) (61)

= δκn(n − 1)

−∞

0

+∞

Φ(0)n−1 − Φ(−∞)n−1 n − 1

1 √2π

Φ(z(2))n−2φ(z(2))2 dz(2) (62)

= δκn(n − 1)

+

0

+∞

1 √2π(n − 1)2n−1 +

Φ(z(2))n−2φ(z(2))2 dz(2) (63)

= δκn(n − 1)

0

√

1 √2π(n − 1)2n−1 +

3 2

lnn n(n − 1)

π ln3

(64)

≤ δκn(n − 1)

3 2

n √2π2n−1 = δκ

1

3 2

π ln3

π ln3

√2π2n−log2 n−1 = δ. (65) This implies P[ξ(1) − ξ(2) > δκσ∥x(l)∥2] ≥ 1 − δ. It follows that with probability at least 1 − δ,

= δκ

lnn +

lnn +

2

2

1 eδκσ∥x(l)∥2−ln(n−1)

1 eξ(1)−ξ(2)−ln(n−1)

ESS(π(l)) ≤ 1 +

(66)

≤ 1 +





2

1 exp

=

1 +

#### .

| |
|---|

 

 

δσ∥x(l)∥2

− ln(n − 1)

3 2

ln 3 lnn + √2π2 1

π

n−log2 n−1

B.2 PROOF OF THEOREM 2 Before stating our proof of Theorem 1, we present a technical lemma that we will employ.

To simplify notation, we omit the superscript (l) in this proof. For an ordered subset I = (i1,...,ik) ⊆ {1,...,n}, let q(I) denote the probability of sampling an ordered subset I from q without replace:

k

#### qi

. (67)

j

Q(I) = Q(i1,...,ik) :=

1 − jj′−=11 qi

j′

j=1

Let Pk denote the set of permutations over {1,...,n}. For ϖ ∈ Pk, define the permutation action as ϖ(i1,...,ik) := (iϖ(1),...,iϖ(k)). Let Q(I) denote the probability of sampling an unordered subset I from q without replacement:

Q(ϖ(I)). (68)

Q(I) = PI∼q[I] =

ϖ∈Pn

- Lemma 10 (swapping a pair). Given a size-k subset I ⊆ {1,...,n}, for a LoRA i ∈ I and another

LoRA i† ∈ {1,...,n} \ I, if qi ≤ qi†, then replacing i with i† increases the unordered sampling probability:

#### Q((I \ {i}) ∪ {i†}) ≥ Q(I). (69)

Proof. Say I = (i1,...,ik). Without loss of generality, say i1 = i, and let I† := (i†,i2,...,ik) denote the ordered subset after replacing i with i†. For any permutation ϖ ∈ Pk, let jϖ := ϖ−1(1) denote the order of i under permutation ϖ (i.e., ϖ(I)j

= i). Since qi ≤ qi†, then Q(ϖ(I†)) Q(ϖ(I))

ϖ

1 − jj′−=11 qi

k

qi† qi

j′

(70)

=

#### 1 − qi† + qi − jj′−=11 qi

j′

j=jϖ+1

k

1 1 − q

qi† qi

(71)

=

i†−qi 1− jj−′=11 qij′

j=jϖ+1

k

qi† qi

qi† qi ≥ 1. (72)

≥

1 =

j=jϖ+1

This means Q(ϖ(I†)) ≥ Q(ϖ(I)). It follows that

Q((I \ {i}) ∪ {i†}) = Q(I†) =

Q(ϖ(I†)) (73)

ϖ∈Pn

Q(ϖ(I)) = Q(I).

≥

| |
|---|

ϖ∈Pn

We are now ready to prove Theorem 2.

- Proof of Theorem 2. Suppose that

n argtopk i=1

I† :=

qi ̸= I∗, (74)

where we break ties in argtop arbitrarily. We will show that this premise leads to a contradiction. Recall that by definition,

- 1

- 2

Q(I∗) = PI∼q[I = I∗] >

. (75)

#### Since I† ̸= I∗, then k∩ := |I∗ ∩ I†| < k. Say I∗ \ I† = {i∗1,...,i∗k−k∩}, I† \ I∗ = {i†1,...,i†k−k∩}. Construct a series of subsets inductively as follows. Define I0 := I∗. For

j = 1,...,k −k∩, define Ij by replacing i∗j from Ij−1 with i†j and inheriting all other LoRAs from Ij−1. Finally, we have Ik−k∩ = I†. Since I† consists of LoRAs i with top-k qi, then qi∗

for all j = 1,...,k−k∩. Hence, by Lemma 10, Q( Ij) ≥ Q( Ij−1) for all j = 1,...,k−k∩. Together,

#### ≤ qi†

j

j

- 1

- 2

Q(I†) = Q( Ik−k∩) ≥ Q( Ik−k∩−1) ≥ ··· ≥ Q( I0) = Q(I∗) >

. (76) It follows that

- 1

- 2

- 1

- 2

Q(I†) + Q(I∗) >

= 1. (77) However, this contradicts the fact that

+

Q(I†) + Q(I∗) ≤

falsifying the premise. Therefore,

n argtopk i=1

I

Q(I) = 1, (78)

qi = I∗.

| |
|---|

