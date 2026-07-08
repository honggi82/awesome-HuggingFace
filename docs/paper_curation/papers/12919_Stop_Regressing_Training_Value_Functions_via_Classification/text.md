## arXiv:2403.03950v1[cs.LG]6Mar2024

[Figure 1]

# Stop Regressing: Training Value Functions via Classification for Scalable Deep RL

###### Jesse Farebrother1,2,*, Jordi Orbay1,†, Quan Vuong1,†, Adrien Ali Taïga1,†, Yevgen Chebotar1, Ted Xiao1, Alex Irpan1, Sergey Levine1, Pablo Samuel Castro1,3,†, Aleksandra Faust1, Aviral Kumar1,†, Rishabh Agarwal1,3,*

*Equal Contribution, †Core Contribution, 1Google DeepMind, 2Mila, McGill University, 3Mila, Université de Montréal

Value functions are a central component of deep reinforcement learning (RL). These functions, parameterized by neural networks, are trained using a mean squared error regression objective to match bootstrapped target values. However, scaling value-based RL methods that use regression to large networks, such as high-capacity Transformers, has proven challenging. This difficulty is in stark contrast to supervised learning: by leveraging a cross-entropy classification loss, supervised methods have scaled reliably to massive networks. Observing this discrepancy, in this paper, we investigate whether the scalability of deep RL can also be improved simply by using classification in place of regression for training value functions. We demonstrate that value functions trained with categorical cross-entropy significantly improves performance and scalability in a variety of domains. These include: single-task RL on Atari 2600 games with SoftMoEs, multi-task RL on Atari with large-scale ResNets, robotic manipulation with Q-transformers, playing Chess without search, and a language-agent Wordle task with high-capacity Transformers, achieving state-of-the-art results on these domains. Through careful analysis, we show that the benefits of categorical cross-entropy primarily stem from its ability to mitigate issues inherent to value-based RL, such as noisy targets and non-stationarity. Overall, we argue that a simple shift to training value functions with categorical cross-entropy can yield substantial improvements in the scalability of deep RL at little-to-no cost.

### 1. Introduction

A clear pattern emerges in deep learning breakthroughs – from AlexNet (Krizhevsky et al., 2012) to Transformers (Vaswani et al., 2017) – classification problems seem to be particularly amenable to effective training with large neural networks. Even in scenarios where a regression approach appears natural, framing the problem instead as a classification problem often improves performance (Torgo and Gama, 1996; Rothe et al., 2018; Rogez et al., 2019). This involves converting real-valued targets into categorical labels and minimizing categorical cross-entropy rather than the mean-squared error. Several hypotheses have been put forward to explain the superiority of this approach, including stable gradients (Imani and White, 2018; Imani et al., 2024), better representations (Zhang et al., 2023), implicit bias (Stewart et al., 2023), and dealing with imbalanced data (Pintea et al., 2023) – suggesting their potential utility beyond supervised regression.

Unlike trends in supervised learning, value-based reinforcement learning (RL) methods primarily rely on regression. For example, deep RL methods such as deep Q-learning (Mnih et al., 2015) and actor-critic (Mnih et al., 2016) use a regression loss, such as mean-squared error, to train a value function from continuous scalar targets. While these value-based deep RL methods, powered by regression losses, have led to high-profile results (Silver et al., 2017), it has been challenging to scale them up to large networks, such as high-capacity transformers. This lack of scalability has been attributed to several issues (Kumar et al., 2021, 2022; Agarwal et al., 2021; Lyle et al., 2022; Le Lan et al., 2023; Obando-Ceron et al., 2024), but what if simply reframing the regression problem as classification can enable the same level of scalability achieved in supervised learning?

Corresponding author(s): jfarebro@cs.mcgill.ca, aviralkumar@google.com, rishabhagarwal@google.com

Regression (MSE) Classification (HL-Gauss)

Single-task RL: SoftMoE (8 experts)

Training Generalist Policies: ResNet-101

Scaling Beyond Atari: High-capacity Transformers

NormalizedPerformance

+115%

+82%

200

+67% +70%

+43%

+29%

100

0

LanguageAgent: Wordle(CQL)

OnlineAtari (DQN)

Chess (Q-function

Robotic Manipulation (Q-Transformer)

Online Multi-taskAtari

Offline Multi-gameAtari

Distillation)

(ScaledQL)

(IMPALA)

- Figure 1 | Performance gains from HL-Gauss cross-entropy loss (§3.1) over MSE regression loss for training valuenetworks with modern architectures, including MoEs (§4.2.1), ResNets (§4.2), and Transformers (§4.3). The x-axis labels correspond to domain name, with training method in brackets. For multi-task RL results, we report gains with ResNet-101 backbone, the largest network in our experiments. For Chess, we report improvement in performance gap relative to the teacher Stockfish engine, for the 270M transformer. For Wordle, we report results with behavior regularization of 0.1.

In this paper, we perform an extensive study to answer this question by assessing the efficacy of various methods for deriving classification labels for training a value-function with a categorical cross-entropy loss. Our findings reveal that training value-functions with cross-entropy substantially improves the performance, robustness, and scalability of deep RL methods (Figure 1) compared to traditional regression-based approaches. The most notable method (HL-Gauss; Imani and White, 2018) leads to consistently 30% better performance when scaling parameters with Mixture-of-Experts in single-task RL on Atari (Obando-Ceron et al., 2024); 1.8 − 2.1× performance in multi-task setups on Atari (Kumar et al., 2023; Ali Taïga et al., 2023); 40% better performance in the language-agent task of Wordle (Snell et al., 2023); 70% improvement for playing chess without search (Ruoss et al., 2024); and 67% better performance on large-scale robotic manipulation with transformers (Chebotar et al., 2023). The consistent trend across diverse domains, network architectures, and algorithms highlights the substantial benefits of treating regression as classification in deep RL, underscoring its potential as a pivotal component as we move towards scaling up value-based RL.

With strong empirical results to support the use of cross-entropy as a “drop-in” replacement for the mean squared error (MSE) regression loss in deep RL, we also attempt to understand the source of these empirical gains. Based on careful diagnostic experiments, we show that the categorical cross-entropy loss offers a number of benefits over mean-squared regression. Our analysis suggests that the categorical cross-entropy loss mitigates several issues inherent to deep RL, including robustness to noisy targets and allowing the network to better use its capacity to fit non-stationary targets. These findings not only help explain the strong empirical advantages of categorical cross-entropy in deep RL but also provide insight into developing more effective learning algorithms for the field.

### 2. Preliminaries and Background

Regression as classification. We take a probabilistic view on regression where given input 𝑥 ∈ ℝ𝑑 we seek to model the target as a conditional distribution 𝑌 | 𝑥 ∼ N(𝜇 = ˆ𝑦(𝑥;𝜃), 𝜎2) for some fixed variance 𝜎2 and predictor function ˆ𝑦 : ℝ𝑑 × ℝ𝑘 → ℝ parameterized by the vector 𝜃 ∈ ℝ𝑘. The maximum

Softmax Project

Neural Network

Target Distribution

|Cross-Entropy Loss|
|---|

- Figure 2 | Regression as Classification. Data points 𝒙𝑖 are transformed by a neural network to produce a categorical distribution via a softmax. The prediction ˆ𝑦 is taken to be the expectation of this categorical distribution. The logits of the network are reinforced by gradient descent on the cross-entropy loss with respect to a target distribution whose mean is the regression target 𝑦𝑖. Figure 3 depicts three methods for constructing and projecting the target distribution in RL.

likelihood estimator for data {𝑥𝑖, 𝑦𝑖}𝑖𝑁=1 is characterized by the mean-squared error (MSE) objective,

###### ∑︁𝑁

(ˆ𝑦(𝑥𝑖;𝜃) − 𝑦𝑖)2 ,

min

𝜃

𝑖=1

with the optimal predictor being ˆ𝑦(𝑥;𝜃∗) = 𝔼 [𝑌 | 𝑥].

Instead of learning the mean of the conditional distribution directly, an alternate approach is to learn a distribution over the target value, and then, recover the prediction ˆ𝑦 as a statistic of the distribution. To this end, we will construct the target distribution 𝑌 | 𝑥 with probability density function 𝑝(𝑦 | 𝑥) such that our scalar target can be recovered as the mean of this distribution 𝑦 = 𝔼𝑝 [𝑌 | 𝑥]. We can now frame the regression problem as learning a parameterized distribution ˆ𝑝(𝑦 | 𝑥;𝜃) that minimizes the KL divergence to the target 𝑝(𝑦 | 𝑥),

∫

###### ∑︁𝑁

min

𝑝(𝑦 | 𝑥𝑖) log (ˆ𝑝(𝑦 | 𝑥𝑖;𝜃)) 𝑑𝑦 (2.1)

𝜃

Y

𝑖=1

which is the cross-entropy objective. Finally, our prediction can be recovered as ˆ𝑦(𝑥;𝜃) = 𝔼ˆ𝑝 [𝑌 | 𝑥;𝜃]. Given this new problem formulation, in order to transform the distribution learning problem into a tractable loss we restrict ˆ𝑝 to the set of categorical distributions supported on [𝑣min, 𝑣max] with 𝑚 evenly spaced locations or “classes”, 𝑣min ≤ 𝑧1 < · · · < 𝑧𝑚 ≤ 𝑣max defined as,

∑︁ 𝑚

###### ∑︁𝑚

𝑝𝑖 𝛿𝑧𝑖 : 𝑝𝑖 ≥ 0,

𝑝𝑖 = 1 , (2.2)

Z =

𝑖=1

𝑖=1

where 𝑝𝑖 is the probability associated with location 𝑧𝑖 and 𝛿𝑧𝑖 is the Dirac delta function at location 𝑧𝑖. The final hurdle is to define a procedure to construct the target distribution 𝑌 | 𝑥 and its associated projection onto the set of categorical distributions Z. We defer this discussion to §3 where we discuss various methods for performing these steps in the context of RL.

Reinforcement Learning (RL). We consider the reinforcement learning (RL) problem where an agent interacts with an environment by taking an action 𝐴𝑡 ∈ A in the current state 𝑆𝑡 ∈ S and subsequently prescribed a reward 𝑅𝑡+1 ∈ ℝ before transitioning to the next state 𝑆𝑡+1 ∈ S according to the environment transition probabilities. The return numerically describes the quality of a sequence of actions as the cumulative discounted sum of rewards 𝐺𝑡 = ∞𝑘=0 𝛾𝑘𝑅𝑡+𝑘+1 where 𝛾 ∈ [0, 1) is the discount factor. The agent’s goal is to learn the policy 𝜋 : S → P(A) that maximizes the expected return. The action-value function allows us to query the expected return from taking action 𝑎 in state 𝑠 and following policy 𝜋 thereafter: 𝑞𝜋(𝑠, 𝑎) = 𝔼𝜋 [𝐺𝑡 | 𝑆𝑡 = 𝑠, 𝐴𝑡 = 𝑎].

Deep Q-Networks (DQN; Mnih et al., 2015) proposes to learn the approximately optimal stateaction value function 𝑄(𝑠, 𝑎;𝜃) ≈ 𝑞𝜋∗(𝑠, 𝑎) with a neural network parameterized by 𝜃. Specifically,

DQN minimizes the mean-squared temporal difference (TD) error from transitions (𝑆𝑡, 𝐴𝑡, 𝑅𝑡+1, 𝑆𝑡+1) sampled from dataset D,

|TDMSE(𝜃) = 𝔼D ( T𝑄)(𝑆𝑡, 𝐴𝑡;𝜃−) − 𝑄(𝑆𝑡, 𝐴𝑡;𝜃)<br><br>2|
|---|

(2.3)

where 𝜃− is a slow moving copy of the parameters 𝜃 that parameterize the “target network” and

( T𝑄)(𝑠, 𝑎;𝜃−) = 𝑅𝑡+1 + 𝛾 max

𝑄(𝑆𝑡+1, 𝑎′;𝜃−) 𝑆𝑡 = 𝑠, 𝐴𝑡 = 𝑎 ,

𝑎′

is the sample version of the Bellman optimality operator which defines our scalar regression target. Most deep RL algorithms that learn value functions use variations of this basic recipe, notably regressing to predictions obtained from a target value network.

In addition to the standard online RL problem setting, we also explore the offline RL setting where we train agents using a fixed dataset of environment interactions (Agarwal et al., 2020; Levine et al., 2020). One widely-used offline RL method is CQL (Kumar et al., 2020) that jointly optimizes the TD error with a behavior regularization loss with strength 𝛼, using the following training objective:

𝛼 𝔼D log ∑︁

min

exp(𝑄(𝑆𝑡+1, 𝑎′;𝜃)) − 𝔼D [𝑄(𝑆𝑡, 𝐴𝑡;𝜃)] + TDMSE(𝜃), (2.4)

𝜃

𝑎′

This work aims to replace the fundamental mean-squared TD-error objective with a classification-style cross-entropy loss for both value-based and actor-critic methods, in both offline and online domains.

### 3. Value-Based RL with Classification

In this section, we describe our approach to cast the regression problem appearing in TD-learning as a classification problem. Concretely, instead of minimizing the squared distance between the scalar Q-value and its TD target (Equation 2.3) we will instead minimize the distance between categorical distributions representing these quantities. To employ this approach, we will first define the categorical representation for the action-value function 𝑄(𝑠, 𝑎).

Categorical Representation. We choose to represent 𝑄 as the expected value of a categorical distribution 𝑍 ∈ Z. This distribution is parameterized by probabilities ˆ𝑝𝑖(𝑠, 𝑎;𝜃) for each location or “class” 𝑧𝑖 which are derived from the logits 𝑙𝑖(𝑠, 𝑎;𝜃) through the softmax function:

###### ∑︁𝑚

exp (𝑙𝑖(𝑠, 𝑎;𝜃))

ˆ𝑝𝑖(𝑠, 𝑎;𝜃) · 𝛿𝑧𝑖, ˆ𝑝𝑖(𝑠, 𝑎;𝜃) =

𝑄(𝑠, 𝑎;𝜃) = 𝔼 [ 𝑍(𝑠, 𝑎;𝜃) ] , 𝑍(𝑠, 𝑎;𝜃) =

.

𝑚 𝑗=1 exp 𝑙𝑗(𝑠, 𝑎;𝜃)

𝑖=1

To employ the cross-entropy loss (Equation 2.1) for TD learning, it is necessary that the target distribution is also a categorical distribution, supported on the same locations 𝑧𝑖, . . . , 𝑧𝑚. This allows for the direct computation of the cross-entropy loss as:

|TDCE(𝜃) = 𝔼D<br><br>∑︁ 𝑚<br><br>𝑖=1<br><br>𝑝𝑖(𝑆𝑡, 𝐴𝑡;𝜃−) log ˆ𝑝𝑖(𝑆𝑡, 𝐴𝑡;𝜃)|
|---|

, (3.1)

where the target probabilities 𝑝𝑖 are defined such that 𝑚𝑖=1 𝑝𝑖(𝑆𝑡, 𝐴𝑡;𝜃−) 𝑧𝑖 ≈ ( T𝑄)(𝑆𝑡, 𝐴𝑡;𝜃−). In the subsequent sections, we explore two strategies for obtaining the target probabilities 𝑝𝑖(𝑆𝑡, 𝐴𝑡;𝜃−).

Two-Hot HLGauss Categorical Distributional RL

| |
|---|

| |
|---|

| | |
|---|---|
| | |

| |
|---|

- Figure 3 | Visualizing target-value categorical distribution in cross-entropy based TD learning. While Two-Hot (left, §3.1) puts probability mass on exactly two locations, HL-Gauss (middle, §3.1) distributes the probability mass to neighbouring locations (which is akin to smoothing the target value). CDRL (right, §3.2) models the categorical return distribution, distributing probability mass proportionally to neighboring locations.

##### 3.1. Constructing Categorical Distributions from Scalars

The first set of methods we outline will project the scalar target ( T𝑄)(𝑆𝑡, 𝐴𝑡;𝜃−) onto the categorical distribution supported on {𝑧𝑖}𝑚𝑖=1. A prevalent but naïve approach for the projection step involves discretizing the scalar into one of 𝑚 bins where 𝑧𝑖 represents the center of the bin. The resulting onehot distribution is “lossy” and induces errors in the 𝑄-function. These errors would compound as more Bellman backups are performed, resulting in more biased estimates, and likely worse performance. To combat this, we first consider the “two-hot” approach (Schrittwieser et al., 2020) that represents a scalar target exactly via a unique categorical distribution that puts non-zero densities on two locations that the target lies between (see Figure 3; Left).

A Two-Hot Categorical Distribution. Let 𝑧𝑖 and 𝑧𝑖+1 be the locations which lower and upper-bound the TD target 𝑧𝑖 ≤ ( T𝑄)(𝑆𝑡, 𝐴𝑡;𝜃−) ≤ 𝑧𝑖+1. Then, the probability, 𝑝𝑖 and 𝑝𝑖+1, put on these locations is:

𝑝𝑖(𝑆𝑡, 𝐴𝑡;𝜃−) = ( T𝑄)(𝑆𝑡, 𝐴𝑡;𝜃−) − 𝑧𝑖 𝑧𝑖+1 − 𝑧𝑖

𝑧𝑖+1 − ( T𝑄)(𝑆𝑡, 𝐴𝑡;𝜃−) 𝑧𝑖+1 − 𝑧𝑖

, 𝑝𝑖+1(𝑆𝑡, 𝐴𝑡;𝜃−) =

. (3.2)

For all other locations, the probability prescribed by the categorical distribution is exactly zero. In principle, this Two-Hot transformation provides a uniquely identifiable and a non-lossy representation of the scalar TD target to a categorical distribution. However, Two-Hot does not fully harness the ordinal structure of discrete regression. Specifically, the classes are not independent and instead have a natural ordering, where each class intrinsically relates to its neighbors.

The class of Histogram Losses introduced by Imani and White (2018) seeks to exploit the ordinal structure of the regression task by distributing probability mass to neighboring bins – akin to label smoothing in supervised classification (Szegedy et al., 2016). This is done by transforming a noisy version of the target value into a categorical distribution where probability mass can span multiple bins near the target (See Figure 3; Center), rather than being restricted to two locations.

Histograms as Categorical Distributions. Formally, define the random variable 𝑌 | 𝑆𝑡, 𝐴𝑡 with probability density 𝑓𝑌|𝑆𝑡,𝐴𝑡 and cumulative distribution function 𝐹𝑌|𝑆𝑡,𝐴𝑡 whose expectation is ( T𝑄)(𝑆𝑡, 𝐴𝑡;𝜃−). We can project the distribution 𝑌 | 𝑆𝑡, 𝐴𝑡 onto the histogram with bins of width 𝜍 = (𝑣max − 𝑣min)/𝑚 centered at 𝑧𝑖 by integrating over the interval [𝑧𝑖 − 𝜍/2, 𝑧𝑖 + 𝜍/2] to obtain the probabilities,

𝑝𝑖(𝑆𝑡, 𝐴𝑡;𝜃−) = ∫ 𝑧

𝑖+𝜍/2

𝑓𝑌|𝑆𝑡,𝐴𝑡(𝑦 | 𝑆𝑡, 𝐴𝑡) 𝑑𝑦

𝑧𝑖−𝜍/2

= 𝐹𝑌|𝑆𝑡,𝐴𝑡(𝑧𝑖 + 𝜍/2 | 𝑆𝑡, 𝐴𝑡) − 𝐹𝑌|𝑆𝑡,𝐴𝑡(𝑧𝑖 − 𝜍/2 | 𝑆𝑡, 𝐴𝑡) . (3.3)

We now have a choice for the distribution 𝑌 | 𝑆𝑡, 𝐴𝑡. We follow the suggestion of Imani and White (2018) in using the Gaussian distribution 𝑌 | 𝑆𝑡, 𝐴𝑡 ∼ N(𝜇 = ( T𝑄)(𝑆𝑡, 𝐴𝑡;𝜃−), 𝜎2) where the variance 𝜎2 is a hyper-parameter that can control the amount of label smoothing applied to the resulting categorical distribution. We refer to this method as HL-Gauss.

How should we tune 𝜎 in practice? HL-Gauss requires tuning the standard deviation 𝜎, in addition to the bin width 𝜍 and distribution range [𝑣𝑚𝑖𝑛, 𝑣𝑚𝑎𝑥]. 99.7% of the samples obtained by sampling from a standard Normal distribution should lie within three standard deviations of the mean with high confidence, which corresponds to approximately 6 · 𝜎/𝜍 bins. Thus, a more interpretable hyperparameter that we recommend tuning is 𝜎/𝜍: setting it to 𝐾/6 distributes most of the probability mass to ⌈𝐾⌉ + 1 neighbouring locations for a mean value centered at one of the bins. Unless specified otherwise, we set 𝜎/𝜍 = 0.75 for our experiments, which distributes mass to approximately 6 locations.

##### 3.2. Modelling the Categorical Return Distribution

In the previous section, we chose to construct a target distribution from the usual scalar regression target representing the expected return. Another option is to directly model the distribution over future returns using our categorical model 𝑍, as done in distributional RL (Bellemare et al., 2023). Notably, C51 (Bellemare et al., 2017), an early distributional RL approach, use the categorical representation along with minimizing the cross-entropy between the predicted distribution 𝑍 and the distributional analogue of the TD target. To this end, we also investigate C51 as an alternative to Two-Hot and HL-Gauss for constructing the target distribution for our cross-entropy objective.

Categorical Distributional RL. The first step to modelling the categorical return distribution is to define the analogous stochastic distributional Bellman operator on 𝑍,

∑︁𝑚

( T𝑍)(𝑠, 𝑎;𝜃−) =𝐷

ˆ𝑝𝑖(𝑆𝑡+1, 𝐴𝑡+1;𝜃−) · 𝛿𝑅𝑡+1+𝛾𝑧𝑖 𝑆𝑡 = 𝑠, 𝐴𝑡 = 𝑎 ,

𝑖=1

where 𝐴𝑡+1 = argmax𝑎′ 𝑄(𝑆𝑡+1, 𝑎′). As we can see, the stochastic distributional Bellman operator has the effect of shifting and scaling the locations 𝑧𝑖 necessitating the categorical projection, first introduced by Bellemare et al. (2017). At a high level, this projection distributes probabilities proportionally to the immediate neighboring locations 𝑧𝑗−1 ≤ 𝑅𝑡+1 + 𝛾𝑧𝑖 ≤ 𝑧𝑗 (See Figure 3; Right). To help us identify these neighboring locations we define ⌊𝑥⌋ = argmax{𝑧𝑖 : 𝑧𝑖 ≤ 𝑥} and ⌈𝑥⌉ = argmin{𝑧𝑖 : 𝑧𝑖 ≥ 𝑥}. Now the probabilities for location 𝑧𝑖 can be written as,

∑︁𝑚

𝑝𝑖(𝑆𝑡, 𝐴𝑡;𝜃−) =

ˆ𝑝𝑗(𝑆𝑡+1, 𝐴𝑡+1;𝜃−) · 𝜉𝑗(𝑅𝑡+1 + 𝛾𝑧𝑖) (3.4)

𝑗=1

𝑥 − 𝑧𝑗 𝑧𝑗+1 − 𝑧𝑗

𝑧𝑗+1 − 𝑥 𝑧𝑗+1 − 𝑧𝑗

1{⌊𝑥⌋ = 𝑧𝑗} +

1{⌈𝑥⌉ = 𝑧𝑗} .

𝜉𝑗(𝑥) =

For a complete exposition of the categorical projection, see Bellemare et al. (2023, Chapter 5).

### 4. Evaluating Classification Losses in RL

The goal of our experiments in this section is to evaluate the efficacy of the various target distributions discussed in Section 3 combined with the categorical cross-entropy loss (3.1) in improving performance and scalability of value-based deep RL on a variety of problems. This includes several single-task and multi-task RL problems on Atari 2600 games as well as domains beyond Atari including language agents, chess, and robotic manipulation. These tasks consist of both online and offline RL problems. For each task, we instantiate our cross-entropy losses in conjunction with a strong value-based RL approach previously evaluated on that task. Full experimental methodologies including hyperparameters for each domain we consider can be found in Appendix B.

Online RL: Atari 200M Aggregate Statistics

Offline RL: Atari CQL

IQMNormalizedScore

IQM

Optimality Gap

HL-Gauss Two-Hot

1.5

C51 MSE

HL-Gauss C51 MSE

1.0

0.5

Two-Hot

0.0

1.2 1.4 1.5

0.2 0.3 0.3 0.3

0 25 50 75 100 Number of Gradient Updates (x 62.5k)

Human Normalized Score

- Figure 4 | Regression vs cross-entropy losses for (Left) Online RL and (Right) Offline RL (§4.1). HL-Gauss and CDRL outperform MSE, with HL-Gauss performing the best. Moreover, Two-Hot loss underperforms MSE but is more stable with prolonged training in offline RL, akin to other cross-entropy losses. See §4.1 for more details.

##### 4.1. Single-Task RL on Atari Games

We first evaluate the efficacy of HL-Gauss, Two-Hot, and C51 (Bellemare et al., 2017) – an instantiation of categorical distributional RL, on the Arcade Learning Environment (Bellemare et al., 2013). For our regression baseline we train DQN (Mnih et al., 2015) on the mean-squared error TD objective which has been shown to outperform other regression based losses (Ceron and Castro, 2021). Each method is trained with the Adam optimizer, which has been shown to reduce the performance discrepancy between regression-based methods and distributional RL approaches (Agarwal et al., 2021).

Evaluation. Following the recommendations by Agarwal et al. (2021), we report the interquartile mean (IQM) normalized scores with 95% stratified bootstrap confidence intervals (CIs), aggregated across games with multiple seeds each. We report human-normalized aggregated scores across 60 Atari games for online RL. For offline RL, we report behavior-policy normalized scores aggregated across 17 games, following the protocol in Kumar et al. (2021).

Online RL results. Following the setup of Mnih et al. (2015), we train DQN for 200M frames with the aforementioned losses. We report aggregated human-normalized IQM performance and optimality gap across 60 Atari games in Figure 4. Observe that HL-Gauss substantially outperforms the Two-Hot and MSE losses. Interestingly, HL-Gauss also improves upon categorical distributional RL (C51), despite not modelling the return distribution. This finding suggests that the loss (categorical cross-entropy) is perhaps the more crucial factor for C51, as compared to modelling the return distribution.

Offline RL results. The strong performance of HL-Gauss with online DQN, which involves learning from self-collected interactions, raises the question of whether it would also be effective in learning from offline datasets. To do so, we train agents with different losses on the 10% Atari DQN replay dataset (Agarwal et al., 2020) using CQL (§2) for 6.25M gradient steps. As shown in Figure 4, HL-Gauss and C51 consistently outperform MSE, while Two-Hot shows improved stability over MSE but underperforms other classification methods. Notably, HL-Gauss again surpasses C51 in this setting. Furthermore, consistent with the findings of Kumar et al. (2021), utilizing the mean squared regression loss results in performance degradation with prolonged training. However, cross-entropy losses (both HL-Gauss and C51) do not show such degradation and generally, remain stable.

##### 4.2. Scaling Value-based RL to Large Networks

In supervised learning, particularly for language modeling (Kaplan et al., 2020), increasing the parameter count of a network typically improves performance. However, such scaling behavior remain elusive for value-based deep RL methods, where naive parameter scaling can hurt performance (Ali Taïga et al., 2023; Kumar et al., 2023; Obando-Ceron et al., 2024). To this end, we investigate the efficacy of our classification methods, as an alternative to MSE regression loss in deep RL, towards enabling better performance with parameter scaling for value-networks.

Online RL (Atari): SoftMoE Expert Scaling.

IQMNormalizedScore

MoE+ HL-Gauss

4.5

4.0

MoE+MSE

3.5

3.0

MSE

2.5

1 2 4 8 Number of Experts

- Figure 5 | MoE scaling curves for HL-Gauss and MSE on Online RL. HL-Gauss, with a single expert, outperform all regression configurations. Both HL-Gauss and MSE scale similarly when employing SoftMoE, with HL-Gauss providing ≈ 30% IQM improvement. SoftMoE also mitigates negative scaling observed with MSE alone. See §4.2.1 for more details.

Multi-Task RL: Scaling Capacity

Asteroids (63 Variants)

IQMNormalizedScore

HL-Gauss

- 1

- 2

- 3

MSE

IMPALA-CNNResNet-18ResNet-34ResNet-50ResNet-101

Figure 6 | Scaling curves on Multi-task Online RL. Results for actor-critic IMPALA with ResNets on Asteroids. HLGauss outperforms MSE and notably reliably scales better with larger networks. Since human scores are not available for variants, we report normalized scores using a baseline IMPALA agent with MSE loss. See §4.2.2 for more details.

##### 4.2.1. Scaling with Mixture-of-Experts

Recently, Obando-Ceron et al. (2024) demonstrate that while parameter scaling with convolutional networks hurts single-task RL performance on Atari, incorporating Mixture-of-Expert (MoE) modules in such networks improves performance. Following their setup, we replace the penultimate layer in the architecture employed by Impala (Espeholt et al., 2018) with a SoftMoE (Puigcerver et al., 2024) module and vary the number of experts in {1, 2, 4, 8}. Since each expert is a copy of the original penultimate layer, this layer’s parameter count increases by a factor equal to the number of experts. The only change we make is to replace the MSE loss in SoftMoE DQN, as employed by Obando-Ceron et al. (2024), with the HL-Gauss cross-entropy loss. We train on the same subset of 20 Atari games used by Obando-Ceron et al. (2024) and report aggregate results over five seeds in Figure 5.

As shown in Figure 5, we find that HL-Gauss consistently improves performance over MSE by a constant factor independent of the number of experts. One can also observe that SoftMoE + MSE seems to mitigate some of the negative scaling effects observed with MSE alone. As SoftMoE + MSE uses a softmax in the penultimate layer this could be providing similar benefits to using a classification loss but as we will later see these benefits alone cannot be explained by the addition of the softmax.

##### 4.2.2. Training Generalist Policies with ResNets

Next, we consider scaling value-based ResNets (He et al., 2016) in both offline and online settings to train a generalist video game-playing policy on Atari. In each case, we train a family of differently sized Q-networks for multi-task RL, and report performance as a function of the network size.

Multi-task Online RL. Following Ali Taïga et al. (2023), we train a multi-task policy capable of playing Atari game variants with different environment dynamics and rewards (Farebrother et al., 2018). We evaluate two Atari games: 63 variants for Asteroids and 29 variants for Space Invaders. We employ a distributed actor-critic method, IMPALA (Espeholt et al., 2018), and compare the standard MSE critic loss with the cross-entropy based HL-Gauss loss. Our experiments investigate the scaling properties of these losses when moving from Impala-CNN (≤ 2M parameters) to larger ResNets (He et al., 2016) up to ResNet-101 (44M parameters). We evaluate multi-task performance after training for 15 billion frames, and repeat each experiment with five seeds.

Results for Asteroids are presented in Figure 6, with additional results on Space Invaders presented in Figure D.3. We observe that in both environments HL-Gauss consistently outperforms

MSE. Notably, HL-Gauss scales better, especially on Asteroids where it even slightly improves performance with larger networks beyond ResNet-18, while MSE performance significantly degrades.

Multi-game Offline RL. We consider the the setup from Kumar et al. (2023), where we modify their recipe to use a non-distributional HL-Gauss loss, in place of distributional C51. Specifically, we train a single generalist policy to play 40 different Atari games simultaneously, when learning from a “near-optimal” training dataset, composed of replay buffers obtained from online RL agents trained independently on each game. This multi-game RL setup was originally proposed by Lee et al. (2022). The remaining design choices (e.g., feature normalization; the size of the network) are kept identical.

As shown in Figure 7, HL-Gauss scales even better than the C51 results from Kumar et al. (2023), resulting in an improvement of about 45% over the best prior multi-game result available with ResNet101 (80M parameters) as measured by the IQM human normalized score (Agarwal et al., 2021). Furthermore, while the performance of MSE regression losses typically plateaus upon increasing model capacity beyond ResNet-34, HL-Gauss is able to leverage this capacity to improve performance, indicating the efficacy of classification-based cross-entropy losses. Additionally, when normalizing against scores obtained by a DQN agent, we show in Figure D.4 that in addition to performance, the rate of improvement as the model scale increases tends to also be larger for the HL-Gauss loss compared to C51.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Residual Neural Network

Amidar Asterix Yars Zaxxon Revenge

Offline RL (Multi-Game): Scaling Capacity

IQMNormalizedScore

1.50

HL-Gauss

Multi-Game DT (200M)

1.20

C51

0.90

MSE

Two-Hot

0.60

31M 60M 79M Number of Parameters

- Figure 7 | Scaling curves on Multi-game Atari (Offline RL). IQM human normalized score for ResNet-{34,50,101}, with spatial embeddings, to play 40 Atari games simultaneously using a single value network (Kumar et al., 2023). HL-Gauss enables remarkable scaling, substantially outperforming categorical distributional RL (C51) and regression (MSE) losses used by prior work, as well as the multi-game Decision Transformer (Lee et al., 2022). See §4.2.2 for more details and Figure D.4 for a version of these results reported in terms of DQN normalized scores, another commonly used metric.

##### 4.3. Value-Based RL with Transformers

Next, we evaluate the applicability of the HL-Gauss cross-entropy loss beyond Atari. To do so, we consider several tasks that utilize high-capacity Transformers, namely, a language-agent task of playing Wordle, playing Chess without inference-time search, and robotic manipulation.

##### 4.3.1. Language Agent: Wordle

To evaluate whether classification losses enhance the performance of value-based RL approaches on language agent benchmarks, we compare HL-Gauss with MSE on the task of playing the game of Wordle1. Wordle is a word guessing game in which the agent gets 6 attempts to guess a word. Each turn the agent receives environment feedback about whether guessed letters are in the true word. The dynamics of this task are non-deterministic. More generally, the task follows a turn-based structure,

1www.nytimes.com/games/wordle/index.html

T S

I M E S

C A N T

|S|
|---|

|W|
|---|

|O|
|---|

|R|
|---|

M P D

Transformer Value Function

T S W O R

</a> </s>

Offline RL (Wordle): Transformer (125M)

HL-Gauss MSE

0.45

SuccessRate

0.40

0.35

0.30

0.1 0.3 1.0 Behavior Regularizer Strength ( )

- Figure 8 | Regression vs cross-entropy loss for Wordle (Offline RL). Comparing HL-Gauss cross-entropy loss with MSE regression loss for a transformer trained with offline RL on Wordle dataset (Snell et al., 2023). Here, we evaluate the success rate of guessing the word in one turn given a partially played Wordle game (e.g., image on left). HL-Gauss leads to substantially higher success rates for varying strengths of behavior regularization. See §4.3.1 for more details.

reminiscent of dialogue tasks in natural language processing. This experiment is situated in the offline RL setting, where we utilize the dataset of suboptimal game-plays provided by Snell et al. (2023). Our goal is to train a GPT-like, decoder-only Transformer, with 125M parameters, representing the Q-network. See Figure 8 (left) for how the transformer model is used for playing this game.

On this task, we train the language-based transformer for 20K gradient steps with an offline RL approach combining Q-learning updates from DQN with a CQL-style behavior regularizer (§2), which corresponds to standard next-token prediction loss (in this particular problem). As shown in Figure 8, HL-Gauss outperforms MSE, for multiple coefficients controlling the strength of CQL regularization.

##### 4.3.2. Grandmaster-level Chess without Search

Transformers have demonstrated their effectiveness as general-purpose algorithm approximators, effectively amortizing expensive inference-time computation through distillation (Ruoss et al., 2024; Lehnert et al., 2024). In this context, we explore the potential benefits of using HL-Gauss to convert scalar action-values into classification targets for distilling a value-function. Using the setup of Ruoss et al. (2024), we evaluate HL-Gauss for distilling the action-value function of Stockfish 16 — the strongest available Chess engine that uses a combination of complex heuristics and explicit search into a causal transformer. The distillation dataset comprises 10 million chess games annotated by the Stockfish engine, yielding 15 billion data points (Figure 9, left).

We train 3 transformer models of varying capacity (9M, 137M, and 270M parameters) on this dataset, using either HL-Gauss or 1-Hot classification targets. We omit MSE as Ruoss et al. (2024) demonstrate that 1-Hot targets outperform MSE on this task. The effectiveness of each model is evaluated based on its ability to solve 10,000 chess puzzles from Lichess, with success measured by the accuracy of the generated action sequences compared to known solutions. Both the setup and results are presented in Figure 9 (right). While the one-hot target with the 270M Transformer from Ruoss et al. (2024) outperformed an AlphaZero baseline without search, HL-Gauss closes the performance gap with the substantially stronger AlphaZero with 400 MCTS simulations (Schrittwieser et al., 2020).

##### 4.3.3. Generalist Robotic Manipulation with Offline Data

Finally, we evaluate whether cross-entropy losses can improve performance on a set of large-scale vision-based robotic manipulation control tasks from Chebotar et al. (2023). These tasks present a simulated 7-DoF mobile manipulator, placed in front of a countertop surface. The goal is to

| |
|---|
|[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]|
| |

| | |
|---|---|
| | |

Offline RL (Distill): Chess Transformer Scaling

HL-Gauss

AlphaZero (w/ MCTS)

Accuracy(10k)

LichessPuzzle

0.95

Ruoss et al. (One-Hot)

0.90

0.85

9M 137M 270M Number of Parameters

- Figure 9 | Grandmaster-level Chess without Search. (Left) Dataset generation for Q-value distillation on Chess. (Right) Scaling Curves. Following the setup from Ruoss et al. (2024), where they train Transformer models to play chess via supervised learning on Stockfish 16 Q-values and then follow greedy policy for evaluation. As the results show, HL-Gauss outperforms one-hot targets used by Ruoss et al. (2024) and nearly matches the performance of AlphaZero with tree search.

[Figure 13]

0 20k 40k 60k 80k Training Steps

0.0

0.1

0.2

SuccessRate

HL-Gauss

MSE

Robotics Manipulation: Q-Transformer

- Figure 10 | Generalist robotic manipulation with offline data: (Left) Robot platform and (Right) HL-Gauss vs MSE on simulated vision-based manipulation. The robotic manipulation problem (§4.3.3) uses the setup from Chebotar et al.

- (2023). The image on the left shows the 7 degree of freedom mobile manipulator robot used for these experiments. In the plots, error bars show 95% CIs. Note that utilizing a HL-Gauss enables significantly faster learning to a better point.

control this manipulator to successfully grasp and lift 17 different kitchen objects in the presence of distractor objects, clutter, and randomized initial poses. We generate a dataset of 500, 000 (successful and failed) episodes starting from a small amount of human-teleoperated demonstrations (40, 000 episodes) by replaying expert demonstrations with added sampled action noise, reminiscent of failed autonomously-collected rollouts obtained during deployment or evaluations of a behavioral cloning policy trained on the human demonstration data.

We train a Q-Transformer model with 60M parameters, following the recipe in Chebotar et al. (2023), but replace the MSE regression loss with the HL-Gauss classification loss. As shown in Figure 10, HL-Gauss results in 67% higher peak performance over the regression baseline, while being much more sample-efficient, addressing a key limitation of the prior regression-based approach.

### 5. Why Does Classification Benefit RL?

Our experiments demonstrate that classification losses can significantly improve the performance and scalability of value-based deep RL. In this section, we perform controlled experiments to understand why classification benefits value-based RL. Specifically, we attempt to understand how the categorical cross-entropy loss can address several challenges specific to value-based RL including representation learning, stability, and robustness. We will also perform ablation experiments to uncover the reasons behind the superiority of HL-Gauss over other categorical targets.

Online RL: Cross-Entropy Ablation

MSE+Softmax MSE Cross-Entropy (C51) Cross-Entropy (HL Gauss)

0.00 0.50 1.00 1.50 IQM Normalized Score

- Figure 11 | Evaluating the learning stability of softmax parameterization (§5.1.1) in online RL on Atari. Categorical representation of Q-values does not benefit MSE + Softmax relative to MSE, implying that the cross-entropy loss is critical.

Offline RL: Double DQN

Offline RL: CQL

MSE (+Softmax)

Lossfunction

MSE

Cross-Entropy (C51)

Cross-Entropy (HL-Gauss)

0.0 0.5 1.0 IQM Normalized Score

0.0 0.5 1.0 1.5 IQM Normalized Score

Figure 12 | Evaluations of the learning stability of MSE+Softmax (§5.1.1) in Offline RL on Atari. We do not observe any substantial gains from using a softmax operator with the MSE loss for either architecture. This implies that the cross-entropy loss is critical.

#### Online RL: HL-Gauss Label Smoothing

IQMNormalizedScore

Bins

21 51

1.5

| |
|---|
| |
| |

101 201

1.0

Two-Hot

0.5

0.0

10 1 100

- Figure 13 | Sweeping the ratio 𝜎/𝜍 for different number of bins in Online RL on Atari.. A wide range of 𝜎 values outperform Two-Hot, which corresponds to not using any label smoothing, implying that HL-Gauss does benefit from a label smoothing like effect. Furthermore, the optimal amount of label smoothing as prescribed by 𝜎 is independent of bin width 𝜍. This implies that the HL-Gauss is leveraging the structure of the regression problem and the gains cannot be purely attributed to reduced overfitting from label smoothing (§5.1.2).

##### 5.1. Ablation Study: What Components of Classification Losses Matter?

Classification losses presented in this paper differ from traditional regression losses used in value-based RL in two ways: (1) parameterizing the output of the value-network to be a categorical distribution in place of a scalar, and (2) strategies for converting scalar targets into a categorical target. We will now understand the relative contribution of these steps towards the performance of cross-entropy losses.

##### 5.1.1. Are Categorical Representations More Performant?

As discussed in §3.1, we parameterize the Q-network to output logits that are converted to probabilities of a categorical distribution by applying the “softmax” operator. Using softmax leads to bounded Qvalues and bounded output gradients, which can possibly improve RL training stability (Hansen et al., 2024). To investigate whether our Q-value parameterization alone results in improved performance without needing a cross-entropy loss, we train Q-functions with the same parameterization as Eq (3.1) but with MSE. We do not observe any gains from using softmax in conjunction with the MSE loss in both online (Figure 11) and offline RL (Figure 12). This highlights that the use of the cross-entropy loss results in the bulk of the performance improvements.

##### 5.1.2. Why Do Some Cross-Entropy Losses Work Better Than Others?

Our results indicate that HL-Gauss outperforms Two-Hot, despite both these methods using a crossentropy loss. We hypothesize that the benefits of HL-Gauss could stem from two reasons: 1) HL-

Online RL: Env Stochasticity

Reward Noise ε ∼ U(0, η)

IQMNormalizedScore

Deterministic

1.8

η = 0.1

η = 0.3

η = 1.0

Sticky Actions

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

IQMNormalizedScore

1.5

1.6

1.0

1.4

1.2

0.5

1.0

0.0

MSE C51 HL-Gauss

HL-Gauss MSE

HL-Gauss

HL-Gauss

MSE

MSE

Loss function

Figure 15 | Cross-entropy vs regression losses when varying environment stochasticity in online RL on Atari (§4.1). HL-Gauss only outperforms MSE under deterministic dynamics. Details are in §5.2.1

- Figure 14 | HL-Gauss vs. MSE when trained using noisy rewards in an offline RL setting on Atari (§4.1). Performance of HL-Gauss degrades slower than MSE as noise increases. Details are in §5.2.1.

.

Gauss reduces overfitting by spreading probability mass to neighboring locations; and 2) HL-Gauss generalizes across a specific range of target values, exploiting ordinal structure in the regression problem. The first hypothesis would be more consistent with how label smoothing addresses overfitting in classification problems (Szegedy et al., 2016).

We test these hypotheses in the online RL setting across a subset of 13 Atari games. To do so, we fix the value range [𝑣min, 𝑣max] while simultaneously varying the number of categorical bins in {21, 51, 101, 201} and the ratio of deviation 𝜎 to bin width 𝜍 in {0.25, 0.5, 0.75, 1.0, 2.0}. We find that a wide range of 𝜎 values for HL-Gauss outperform Two-Hot, indicating that spreading probability mass to neighbouring locations likely results in less overfitting. Interestingly, we notice that the second hypothesis is also at play, as the optimal value of 𝜎 seems to be independent of number of bins, indicating that HL-Gauss generalizes best across a specific range of target values and is indeed leveraging the ordinal nature of the regression problem. Thus, the gains from HL-Gauss cannot be entirely attributed to overfitting, as is believed to be the case for label smoothing.

##### 5.2. What Challenges Does Classification Address in Value-Based RL?

Having seen that the performance gains of cross-entropy losses stem from both the use of a categorical representation of values and distributed targets, we now attempt to understand which challenges in value-based RL cross-entropy losses address, or at least, partially alleviate.

##### 5.2.1. Is Classification More Robust to Noisy Targets?

Classification is less prone to overfitting to noisy targets than regression, as it focuses on the categorical relationship between the input and target rather than their exact numerical relationship. We investigate whether classification could better deal with noise induced by stochasticity in RL.

- (a) Noisy Rewards. To test robustness of classification to stochasticity in rewards, we consider an offline RL setup where we add random noise 𝜀𝑡, sampled uniformly from (0, 𝜂), to each dataset reward 𝑟𝑡. We vary the noise scale 𝜂 ∈ {0.1, 0.3, 1.0} and compare the performance of cross-entropy based HL-Gauss with the MSE loss. As shown in Figure 14, the performance of HL-Gauss degrades more gracefully than MSE as the noise scale increases.
- (b) Stochasticity in Dynamics. Following Machado et al. (2018), our Atari experiments use sticky actions — with 25% probability, the environment will execute the previous action again, instead

of the agent’s executed action — resulting in non-deterministic dynamics. Here, we turn off sticky actions to compare different losses on deterministic Atari (60 games). As shown in Figure 15, while cross-entropy based HL-Gauss outperforms MSE with stochastic dynamics, they perform comparably under deterministic dynamics while outperforming distributional C51.

Overall, the benefits of cross-entropy losses can be partly attributed to less overfitting to noisy targets, an issue inherent to RL environments with stochastic dynamics or rewards. Such stochasticity issues may also arise as a result of dynamics mis-specification or action delays in real-world embodied RL problems, implying that a cross-entropy loss is a superior choice in those problems.

##### 5.2.2. Does Classification Learn More Expressive Representations?

It is well known that just using the mean-squared regression error alone does not produce useful representations in value-based RL, often resulting in low capacity representations (Kumar et al., 2021) that are incapable of fitting target values observed during subsequent training. Predicting a categorical distribution rather than a scalar target can lead to better representations (Zhang et al., 2023), that retain the representational power to model value functions of arbitrary policies that might be encountered over the course of value learning (Dabney et al., 2021). Lyle et al. (2019) showed that gains from C51 can be partially attributed to improved representations but it remains unknown whether they stem from backing up distributions of returns or the use of cross-entropy loss.

To investigate this question, following the protocol in Farebrother et al. (2023), we study whether a learned representation, corresponding to penultimate feature vectors, obtained from value-networks trained online on Atari for 200M frames, still retain the necessary information to re-learn a policy from scratch. To do so, we train a Q-function with a single linear layer on top of frozen representation (Farebrother et al., 2023), akin to how self-supervised representations are evaluated in vision (He et al., 2020). As shown in Figure 16, cross-entropy losses result in better performance with linear probing. This indicates that their learned representations are indeed better in terms of supporting the value-improvement path of a policy trained from scratch (Dabney et al., 2021).

Online RL: Linear RL from Fixed Features

IQM

Optimality Gap

HL-Gauss

C51

MSE

0.2 0.3

0.6 0.7 0.8

Human Normalized Score

- Figure 16 | Evaluating representations using linear probing (§5.2.2) on Atari. This experiment follows the protocol of Farebrother et al. (2023). Optimality gap refers to the distance from human-level performance and lower is better. In both plots, HL-Gauss scores best, indicating its learned representations are the most conducive to downstream tasks.

##### 5.2.3. Does Classification Perform Better Amidst Non-Stationarity?

Non-stationarity is inherent to value-based RL as the target computation involves a constantly evolving argmax policy and value function. Bellemare et al. (2017) hypothesized that classification might mitigate difficulty of learning from a non-stationary policy, but did not empirically validate it. Here, we investigate whether classification can indeed handle target non-stationarity better than regression.

Synthetic setup: We first consider a synthetic regression task on CIFAR10 presented in Lyle et al.

Synt Magnitude

|Synthetic:<br><br>0|Regression<br><br>b = 8|w/<br><br>b = 16|Increasing<br><br>b = 24|
|---|---|---|---|
| | | | |

b =

100

10 1

###### MSE

L2 Regression

10 2

Two-Hot

10 3

HL Gauss

5k 10k 15k 20k

Gradient Step

- Figure 17 | Synthetic magnitude prediction experiment to simulate non-stationarity on CIFAR10 (§5.2.3). Nonstationarity is simulated by fitting networks with different losses on an increasing sequences of biases over gradient steps. Crossentropy losses are less likely to lose plasticity.

Offline RL: SARSA

Offline RL: CQL

MSE

Lossfunction

Cross-Entropy (2-Hot)

Cross-Entropy (HL-Gauss)

0.0 0.5 1.0 IQM Normalized Score

0.0 0.5 1.0 1.5 IQM Normalized Score

Figure 18 | Offline QL vs SARSA to ablate policy nonstationarity on Atari (§5.2.3). HL-Gauss gains over MSE vanish with SARSA. This is evidence that some of the benefits from classification stem from dealing with non-stationarity in value-based RL.

- (2024), where the regression target corresponds to mapping an input image 𝑥𝑖 through a randomly initialized neural network 𝑓𝜃− to produce high-frequency targets 𝑦𝑖 = sin(105 · 𝑓𝜃−(𝑥𝑖)) + 𝑏 where 𝑏 is a constant bias that can control for the magnitude of the targets. When learning a value function with TD, the prediction targets are non-stationary and often increase in magnitude over time as the policy improves. We simulate this setting by fitting a network with different losses on the increasing sequence of bias 𝑏 ∈ {0, 8, 16, 24, 32}. See details in Appendix B.4. As shown in Figure 17, classification losses retain higher plasticity under non-stationary targets compared to regression.

Offline RL: To control non-stationarity in an RL context, we run offline SARSA, which estimates the value of the fixed data-collection policy, following the protcol in Kumar et al. (2022). Contrary to Q-learning, which use the action which maximizes the learned Q-value at the next state 𝑆𝑡+1 for computing the Bellman target (§2), SARSA uses the action observed at the next timestep (𝑆𝑡+1, 𝐴𝑡+1) in the offline dataset. As shown in Figure 18, most of the benefit from HL-Gauss compared to the MSE loss vanishes in the offline SARSA setting, adding evidence that some of the benefits from classification stem from dealing with non-stationarity in value-based RL.

To summarize, we find that the use of cross-entropy loss itself is central to obtain good performance in value-based RL, and while these methods do not address any specific challenge, they enable value-based RL methods to deal better with non-stationarity, induce highly-expressive representations, and provide robustness against noisy target values.

### 6. Related Work

Prior works in tabular regression (Weiss and Indurkhya, 1995; Torgo and Gama, 1996) and computer vision (Van Den Oord et al., 2016; Kendall et al., 2017; Rothe et al., 2018; Rogez et al., 2019) have replaced regression with classification to improve performance. Most notably, Imani and White (2018) proposed the HL-Gauss cross-entropy loss for regression and show its efficacy on small-scale supervised regression tasks, outside of RL. Our work complements these prior works by illustrating for the first time that a classification objective trained with cross-entropy, particularly HL-Gauss, can enable effectively scaling for value-based RL on a variety of domains, including Atari, robotic manipulation, chess, and Wordle.

Several state-of-the-art methods in RL have used the Two-Hot cross-entropy loss without any analysis, either as an “ad-hoc” trick (Schrittwieser et al., 2020), citing benefits for sparse rewards (Hafner et al., 2023), or simply relying on folk wisdom (Hessel et al., 2021; Hansen et al., 2024). However, in our experiments, Two-Hot performs worse than other cross-entropy losses and MSE. We believe this

is because Two-Hot does not effectively distribute probability to neighboring classes, unlike C51 and HL-Gauss (see §5.1.2 for an empirical investigation).

Closely related is the line of work on categorical distributional RL. Notably, Achab et al. (2023) offer an analysis of categorical one-step distributional RL, which corresponds precisely to the TwoHot algorithm discussed herein with the similarity of these two approaches not being previously recognized. Additionally, the work of Bellemare et al. (2017) pioneered the C51 algorithm, and while their primary focus was not on framing RL as classification, our findings suggest that the specific loss function employed may play a more significant role in the algorithm’s success than modeling the return distribution itself. Several methods find that categorical distributional RL losses are important for scaling offline value-based RL (Kumar et al., 2023; Springenberg et al., 2024), but these works do not attempt to isolate which components of this paradigm are crucial for attaining positive scaling trends. We also note that these findings do not contradict recent theoretical work (Wang et al., 2023; Rowland et al., 2023) which argues that distributional RL brings statistical benefits over standard RL orthogonal to use of a cross entropy objective or the categorical representation.

Prior works have characterized the representations learned by TD-learning (Bellemare et al., 2019; Lyle et al., 2021; Le Lan et al., 2022, 2023; Kumar et al., 2021, 2022) but these prior works focus entirely on MSE losses with little to no work analyzing representations learned by cross-entropy based losses in RL. Our linear probing experiments in §5.2.2 try to fill this void, demonstrating that value-networks trained with cross-entropy losses learn better representations than regression. This finding is especially important since Imani and White (2018) did not find any representational benefits of HL-Gauss over MSE on supervised regression, indicating that the use of cross-entropy might have substantial benefits for TD-based learning methods in particular.

### 7. Conclusion

In this paper, we showed that framing regression as classification and minimizing categorical crossentropy instead of the mean squared error yields large improvements in performance and scalability of value-based RL methods, on a wide variety of tasks, with several neural network architectures. We analyzed the source of these improvements and found that they stem specifically from the ability of the cross-entropy loss in enabling more expressive representations and handling noise and nonstationarity in value-based RL better. While the cross-entropy loss alone does not fully alleviate any of these problems entirely, our results show the substantial difference this small change can make.

We believe that strong results with the use categorical cross-entropy has implications for future algorithm design in deep RL, both in theory and practice. For instance, value-based RL approaches have been harder to scale and tune when the value function is represented by a transformer architecture and our results hint that classification might provide for a smooth approach to translate innovation in value-based RL to transformers. From a theoretical perspective, analyzing the optimization dynamics of cross-entropy might help devise improved losses or target distribution representations. Finally, while we did explore a number of settings, further work is required to evaluate the efficacy of classification losses in other RL problems such as those involving pre-training, fine-tuning, or continual RL.

### Acknowledgements

We would like to thank Will Dabney for providing feedback on an early version of this paper. We’d also like to thank Clare Lyle, Mark Rowland, Marc Bellemare, Max Schwarzer, Pierluca D’oro, Nate Rahn, Harley Wiltzer, Wesley Chung, and Dale Schuurmans, for informative discussions. We’d also like to acknowledge Anian Ruoss, Grégoire Delétang, and Tim Genewein for their help with the Chess

training infrastructure. This research was supported by the TPU resources at Google DeepMind, and the authors are grateful to Doina Precup and Joelle Baral for their support.

### Author Contributions

JF led the project, implemented histogram-based methods, ran all the single-task online RL experiments on Atari, Q-distillation on Chess, jointly proposed and ran most of the analysis experiments, and contributed significantly to paper writing.

JO and AAT set up and ran the multi-task RL experiments and helped with writing. QV ran the robotic manipulation experiments and YC helped with the initial set-up. TX helped with paper writing and AI was involved in discussions. SL advised on the robotics and Wordle experiments and provided feedback. PSC helped set up the SoftMoE experiments and hosted Jesse at GDM. PSC and AF sponsored the project and took part in discussions.

AK advised the project, proposed offline RL analysis for non-stationarity and representation learning, contributed significantly to writing, revising, and the narrative, and set up the robotics and multi-game scaling experiments. RA proposed the research direction, advised the project, led the paper writing, ran offline RL and Wordle experiments, and helped set up all of the multi-task scaling and non-Atari experiments.

### References

Mastane Achab, Réda Alami, Yasser Abdelaziz Dahou Djilali, Kirill Fedyanin, and Eric Moulines. One-step distributional reinforcement learning. CoRR, abs/2304.14421, 2023.

Rishabh Agarwal, Dale Schuurmans, and Mohammad Norouzi. An optimistic perspective on offline reinforcement learning. In International Conference on Machine Learning (ICML), 2020.

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G. Bellemare. Deep reinforcement learning at the edge of the statistical precipice. Neural Information Processing Systems (NeurIPS), 2021.

Adrien Ali Taïga, Rishabh Agarwal, Jesse Farebrother, Aaron Courville, and Marc G. Bellemare. Investigating multi-task pretraining and generalization in reinforcement learning. In International Conference on Learning Representations (ICLR), 2023.

Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research (JAIR), 47: 253–279, 2013.

Marc G. Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning. In International Conference on Machine Learning (ICML), 2017.

Marc G. Bellemare, Will Dabney, Robert Dadashi, Adrien Ali Taïga, Pablo Samuel Castro, Nicolas Le Roux, Dale Schuurmans, Tor Lattimore, and Clare Lyle. A geometric perspective on optimal representations for reinforcement learning. In Neural Information Processing Systems (NeurIPS), 2019.

Marc G. Bellemare, Will Dabney, and Mark Rowland. Distributional reinforcement learning. MIT Press, 2023.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/ google/jax.

Pablo Samuel Castro, Subhodeep Moitra, Carles Gelada, Saurabh Kumar, and Marc G. Bellemare. Dopamine: A Research Framework for Deep Reinforcement Learning. CoRR, abs/1812.06110, 2018.

Johan Samir Obando Ceron and Pablo Samuel Castro. Revisiting rainbow: Promoting more insightful and inclusive deep reinforcement learning research. In International Conference on Machine Learning (ICML), 2021.

Yevgen Chebotar, Quan Vuong, Karol Hausman, Fei Xia, Yao Lu, Alex Irpan, Aviral Kumar, Tianhe Yu, Alexander Herzog, Karl Pertsch, et al. Q-transformer: Scalable offline reinforcement learning via autoregressive q-functions. In Conference on Robot Learning (CoRL), 2023.

Will Dabney, André Barreto, Mark Rowland, Robert Dadashi, John Quan, Marc G. Bellemare, and David Silver. The value-improvement path: Towards better representations for reinforcement learning. In AAAI Conference on Artificial Intelligence, 2021.

Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. In International Conference on Machine Learning (ICML), 2018.

Jesse Farebrother, Marlos C. Machado, and Michael Bowling. Generalization and regularization in DQN. CoRR, abs/1810.00123, 2018.

Jesse Farebrother, Joshua Greaves, Rishabh Agarwal, Charline Le Lan, Ross Goroshin, Pablo Samuel Castro, and Marc G. Bellemare. Proto-value networks: Scaling representation learning with auxiliary tasks. In International Conference on Learning Representations (ICLR), 2023.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy P. Lillicrap. Mastering diverse domains through world models. CoRR, abs/2301.04104, 2023.

Nicklas Hansen, Hao Su, and Xiaolong Wang. TD-MPC2: Scalable, robust world models for continuous control. In International Conference on Learning Representations (ICLR), 2024.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Matteo Hessel, Ivo Danihelka, Fabio Viola, Arthur Guez, Simon Schmitt, Laurent Sifre, Theophane Weber, David Silver, and Hado van Hasselt. Muesli: Combining improvements in policy optimization. In International Conference on Machine Learning (ICML), 2021.

Daniel Ho, Kanishka Rao, Zhuo Xu, Eric Jang, Mohi Khansari, and Yunfei Bai. Retinagan: An objectaware approach to sim-to-real transfer. In IEEE International Conference on Robotics and Automation (ICRA), 2021.

Ehsan Imani and Martha White. Improving regression performance with distributional losses. In International Conference on Machine Learning (ICML), 2018.

Ehsan Imani, Kai Luedemann, Sam Scholnick-Hughes, Esraa Elelimy, and Martha White. Investigating the histogram loss in regression. CoRR, abs/2402.13425, 2024.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. CoRR, abs/2001.08361, 2020.

Alex Kendall, Hayk Martirosyan, Saumitro Dasgupta, Peter Henry, Ryan Kennedy, Abraham Bachrach, and Adam Bry. End-to-end learning of geometry and context for deep stereo regression. In IEEE International Conference on Computer Vision (ICCV), 2017.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Neural Information Processing Systems (NeurIPS), 2012.

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Neural Information Processing Systems (NeurIPS), 2020.

Aviral Kumar, Rishabh Agarwal, Dibya Ghosh, and Sergey Levine. Implicit under-parameterization inhibits data-efficient deep reinforcement learning. In International Conference on Learning Representations (ICLR), 2021.

Aviral Kumar, Rishabh Agarwal, Tengyu Ma, Aaron Courville, George Tucker, and Sergey Levine. Dr3: Value-based deep reinforcement learning requires explicit regularization. In International Conference on Learning Representations (ICLR), 2022.

Aviral Kumar, Rishabh Agarwal, Xinyang Geng, George Tucker, and Sergey Levine. Offline Q-Learning on Diverse Multi-Task Data Both Scales and Generalizes. In International Conference on Learning Representations (ICLR), 2023.

Charline Le Lan, Stephen Tu, Adam Oberman, Rishabh Agarwal, and Marc G. Bellemare. On the generalization of representations in reinforcement learning. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022.

Charline Le Lan, Stephen Tu, Mark Rowland, Anna Harutyunyan, Rishabh Agarwal, Marc G. Bellemare, and Will Dabney. Bootstrapped representations in reinforcement learning. In International Conference on Machine Learning (ICML), 2023.

Kuang-Huei Lee, Ofir Nachum, Mengjiao (Sherry) Yang, Lisa Lee, Daniel Freeman, Sergio Guadarrama, Ian Fischer, Winnie Xu, Eric Jang, Henryk Michalewski, and Igor Mordatch. Multi-game decision transformers. In Neural Information Processing Systems (NeurIPS), 2022.

Lucas Lehnert, Sainbayar Sukhbaatar, Paul Mcvay, Michael Rabbat, and Yuandong Tian. Beyond a*: Better planning with transformers via search dynamics bootstrapping. CoRR, abs/2402.14083, 2024.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. CoRR, abs/2005.01643, 2020.

Clare Lyle, Marc G. Bellemare, and Pablo Samuel Castro. A comparative analysis of expected and distributional reinforcement learning. In AAAI Conference on Artificial Intelligence, 2019.

Clare Lyle, Mark Rowland, Georg Ostrovski, and Will Dabney. On the effect of auxiliary tasks on representation dynamics. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2021.

Clare Lyle, Mark Rowland, and Will Dabney. Understanding and preventing capacity loss in reinforcement learning. In International Conference on Learning Representations (ICLR), 2022.

Clare Lyle, Zeyu Zheng, Khimya Khetarpal, Hado van Hasselt, Razvan Pascanu, James Martens, and Will Dabney. Disentangling the causes of plasticity loss in neural networks. CoRR, abs/2402.18762, 2024.

Marlos C. Machado, Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, and Michael Bowling. Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents. Journal of Artificial Intelligence Research (JAIR), 61:523–562, 2018.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. Nature, 518 (7540):529–533, 2015.

Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning (ICML), 2016.

Johan Obando-Ceron, Ghada Sokar, Timon Willi, Clare Lyle, Jesse Farebrother, Jakob Foerster, Gintare Karolina Dziugaite, Doina Precup, and Pablo Samuel Castro. Mixtures of experts unlock parameter scaling for deep rl. CoRR, abs/2402.08609, 2024.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Neural Information Processing Systems (NeurIPS), 2019.

Silvia L. Pintea, Yancong Lin, Jouke Dijkstra, and Jan C. van Gemert. A step towards understanding why classification helps regression. In IEEE International Conference on Computer Vision (ICCV), pages 19972–19981, 2023.

Joan Puigcerver, Carlos Riquelme Ruiz, Basil Mustafa, and Neil Houlsby. From sparse to soft mixtures of experts. In International Conference on Learning Representations (ICLR), 2024.

Gregory Rogez, Philippe Weinzaepfel, and Cordelia Schmid. Lcr-net++: Multi-person 2d and 3d pose detection in natural images. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 42(5):1146–1161, 2019.

Rasmus Rothe, Radu Timofte, and Luc Van Gool. Deep expectation of real and apparent age from a single image without facial landmarks. International Journal of Computer Vision (IJCV), 126(2-4): 144–157, 2018.

Mark Rowland, Yunhao Tang, Clare Lyle, Rémi Munos, Marc G. Bellemare, and Will Dabney. The statistical benefits of quantile temporal-difference learning for value estimation. In International Conference on Machine Learning (ICML), 2023.

Anian Ruoss, Grégoire Delétang, Sourabh Medapati, Jordi Grau-Moya, Li Kevin Wenliang, Elliot Catt, John Reid, and Tim Genewein. Grandmaster-level chess without search. CoRR, abs/2402.04494, 2024.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy P. Lillicrap, and David Silver. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588 (7839):604–609, 2020.

David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. Mastering the game of go without human knowledge. Nature, 550(7676):354–359, 2017.

Charlie Victor Snell, Ilya Kostrikov, Yi Su, Sherry Yang, and Sergey Levine. Offline RL for natural language generation with implicit language q learning. In International Conference on Learning Representations (ICLR), 2023.

Jost Tobias Springenberg, Abbas Abdolmaleki, Jingwei Zhang, Oliver Groth, Michael Bloesch, Thomas Lampe, Philemon Brakel, Sarah Bechtle, Steven Kapturowski, Roland Hafner, et al. Offline actorcritic reinforcement learning scales to large models. CoRR, abs/2402.05546, 2024.

Lawrence Stewart, Francis Bach, Quentin Berthet, and Jean-Philippe Vert. Regression as classification: Influence of task formulation on neural network features. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2023.

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Luís Torgo and João Gama. Regression by classification. In Brazilian Symposium on Artificial Intelligence, pages 51–60. Springer, 1996.

Aäron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In International Conference on Machine Learning (ICML), 2016.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Neural Information Processing Systems (NeurIPS), 2017.

Kaiwen Wang, Kevin Zhou, Runzhe Wu, Nathan Kallus, and Wen Sun. The benefits of being distributional: Small-loss bounds for reinforcement learning. In Neural Information Processing Systems (NeurIPS), 2023.

Sholom M Weiss and Nitin Indurkhya. Rule-based machine learning methods for functional prediction. Journal of Artificial Intelligence Research (JAIR), 3:383–403, 1995.

Shihao Zhang, Linlin Yang, Michael Bi Mi, Xiaoxu Zheng, and Angela Yao. Improving deep regression with ordinal entropy. In International Conference on Learning Representations (ICLR), 2023.

### A. Reference Implementations

import jax import jax.scipy.special import jax.numpy as jnp

def hl_gauss_transform( min_value: float, max_value: float, num_bins: int, sigma: float,

):

"""Histogram loss transform for a normal distribution.""" support = jnp.linspace(min_value, max_value, num_bins + 1, dtype=jnp.float32)

def transform_to_probs(target: jax.Array) -> jax.Array: cdf_evals = jax.scipy.special.erf((support - target) / (jnp.sqrt(2) * sigma)) z = cdf_evals[-1] - cdf_evals[0] bin_probs = cdf_evals[1:] - cdf_evals[:-1] return bin_probs / z

def transform_from_probs(probs: jax.Array) -> jax.Array: centers = (support[:-1] + support[1:]) / 2 return jnp.sum(probs * centers)

return transform_to_probs, transform_from_probs

- Listing 1 | An implementation of HL-Gauss (Imani and White, 2018) in Jax (Bradbury et al., 2018).

import torch import torch.special import torch.nn as nn import torch.nn.functional as F

class HLGaussLoss(nn.Module):

def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float): super().__init__() self.min_value = min_value self.max_value = max_value self.num_bins = num_bins self.sigma = sigma self.support = torch.linspace(

min_value, max_value, num_bins + 1, dtype=torch.float32 )

def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

return F.cross_entropy(logits, self.transform_to_probs(target)) def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:

cdf_evals = torch.special.erf( (self.support - target.unsqueeze(-1)) / (torch.sqrt(torch.tensor(2.0)) * self.sigma)

) z = cdf_evals[..., -1] - cdf_evals[..., 0] bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1] return bin_probs / z.unsqueeze(-1)

def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor: centers = (self.support[:-1] + self.support[1:]) / 2 return torch.sum(probs * centers, dim=-1)

- Listing 2 | An implementation of HL-Gauss (Imani and White, 2018) in PyTorch (Paszke et al., 2019).

- B. Experimental Methodology In the subsequent sections we outline the experimental methodology for each domain herein.

##### B.1. Atari

Both our online and offline RL regression baselines are built upon the Jax (Bradbury et al., 2018) implementation of DQN+Adam in Dopamine (Castro et al., 2018). Similarly, each of the classification methods (i.e., HL-Gauss and Two-Hot) were built upon the Jax (Bradbury et al., 2018) implementation of C51 in Dopamine (Castro et al., 2018). Hyperparameters for DQN+Adam are provided in Table B.1 along with any hyperparameter differences for C51 (Table B.2), Two-Hot (Table B.2), and HL-Gauss (Table B.3). Unless otherwise stated the online RL results in the paper were ran for 200M frames on 60 Atari games with five seeds per game. The offline RL results were ran on the 17 games in Kumar et al. (2021) with three seeds per game. The network architecture for both the online and offline results is the standard DQN Nature architecture that employs three convolutional layers followed by a single non-linear fully-connected layer before outputting the action-values.

##### Table B.1 | DQN+Adam Hyperparameters.

Discount Factor 𝛾 0.99 𝑛-step 1 Minimum Replay History 20, 000 agent steps Agent Update Frequency 4 environment steps Target Network Update Frequency 8, 000 agent steps Exploration 𝜖 0.01 Exploration 𝜖 decay 250, 000 agent steps Optimizer Adam Learning Rate 6.25 × 10−5 Adam 𝜖 1.5 × 10−4 Sticky Action Probability 0.25 Maximum Steps per Episode 27, 000 agent steps Replay Buffer Size 1, 000, 000 Batch Size 32

- Table B.2 | C51 & Two-Hot Hyperparameters. Difference in hyperparameters from DQN+Adam Table B.1.

Number of Locations 51 [𝑣min, 𝑣max] [−10, 10]

Learning Rate 0.00025 Adam 𝜖 0.0003125

- Table B.3 | HL-Gauss Hyperparameters. Difference in hyperparameters from C51 Table B.2.

Smoothing Ratio 𝜎/𝜍 0.75

##### B.1.1. Mixtures of Experts

All experiments ran with SoftMoE reused the experimental methodology of Obando-Ceron et al. (2024). Specifically, we replace the penultimate layer of the DQN+Adam in Dopamine (Castro et al., 2018) with a SoftMoE (Puigcerver et al., 2024) module. The MoE results were ran with the Impala ResNet architecture (Espeholt et al., 2018). We reuse the same set of 20 games from Obando-Ceron et al. (2024) and run each configuration for five seeds per game. All classification methods reused the parameters from Table B.2 for C51 and Two-Hot or Table B.3 for HL-Gauss.

##### B.1.2. Multi-Task & Multi-Game

The multi-task and multi-game results follow exactly the methodology outlined in Ali Taïga et al. (2023) and Kumar et al. (2023) respectively. We reuse the hyperparameters for HL-Gauss outlined in Table B.3. For multi-task results each agent is run for five seeds per game. Due to the prohibitive compute of the multi-game setup we run each configuration for a single seed.

##### B.2. Chess

We follow exactly the setup in Ruoss et al. (2024) with the only difference being the use of HL-Gauss with a smoothing ratio 𝜎/𝜍 = 0.75. Specifically, we take the action-values produced by Stockfish and project them a categorical distribution using HL-Gauss. As Ruoss et al. (2024) was already performing classification we reuse the parameters of their categorical distribution, those being, 𝑚 = 128 bins evenly divided between the range [0, 1]. For each parameter configuration we train a single agent and report the evaluation puzzle accuracy. Puzzle accuracy numbers for one-hot and AlphaZero w/ MCTS were taken directly from Ruoss et al. (2024, Table 6).

##### B.3. Robotic manipulation experiments.

We study a large-scale vision-based robotic manipulation setting on a mobile manipulator robot with 7 degrees of freedom, which is visualized in Figure 10 (left). The tabletop robot manipulation domain consists of a tabletop with various randomized objects spawned on top of the countertop. A RetinaGAN is applied to transform the simulation images closer to real-world image distributions, following the method in (Ho et al., 2021). We implement a Q-Transformer policy following the procedures in (Chebotar et al., 2023). Specifically, we incorporate autoregressive 𝑄-learning by learning 𝑄-values per action dimension, incorporate conservative regularization to effectively learn from suboptimal data, and utilize Monte-Carlo returns.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Figure B.1 | Robot manipulation domain. The simulated robot manipulation (§4.3.3) consists of a tabletop with randomized objects. A learned RetinaGAN transformation is applied to make the visual observation inputs more realistic.

##### B.4. Regression Target Magnitude & Loss of Plasticity

To assess whether classification losses are more robust when learning non-stationary targets of increasing magnitude we leverage the synthetic setup from Lyle et al. (2024). Specifically, we train a convolutional neural network that takes CIFAR 10 images 𝑥𝑖 as input and outputs a scalar prediction: 𝑓𝜃 : ℝ32×32×3 → ℝ. The goal is to fit the regression target,

𝑦𝑖 = sin(𝑚 𝑓𝜃−(𝑥𝑖)) + 𝑏

where 𝑚 = 105, 𝜃− are a set of randomly sampled target parameters for the same convolutional architecture, and 𝑏 is a bias that changes the magnitude of the prediction targets. It is clear that increasing 𝑏 shouldn’t result in a more challenging regression task.

When learning a value function with TD methods the regression targets are non-stationary and hopefully increasing in magnitude (corresponding to an improving policy). To simulate this setting we consider fitting the network 𝑓𝜃 on the increasing sequence 𝑏 ∈ {0, 8, 16, 24, 32}. For each value 𝑏

we sample a new set of target parameters 𝜃− and regress towards 𝑦𝑖 for 5, 000 gradient steps with a batch size of 512 with the Adam optimizer using a learning rate of 10−3.

We evaluate the Mean-Squared Error (MSE) throughout training for three methods: Two-Hot, HLGauss, and L2 regression. For both Two-Hot and HL-Gauss we use a support of [−40, 40] with 101 bins. Figure 17 depicts the MSE throughout training averaged over 30 seeds for each method. One can see that the network trained with L2 regression does indeed loose its ability to rapidly fit targets of increasing magnitude, consistent with Lyle et al. (2024). On the other hand, the classification methods are more robust and tend to converge to the same MSE irrespective of the target magnitude 𝑏. Furthermore, we can see that HL-Gauss outperforms Two-Hot, consistent with our previous findings. These results help provide some evidence that perhaps one of the reasons classification outperforms regression is due to the network being more “plastic” under non-stationary targets.

### C. Per-Game Atari Results

AirRaid

Alien

Amidar

Assault

Asterix

HL-Gauss Two Hot

1,500

HL-Gauss

HL-Gauss

EpisodeReturn

C51 MSE

10,000

MSE

20,000

MSE

- 1,500

- 2,000

- 1,000

- 2,000

- 3,000

- 4,000

C51 MSE

Two Hot C51

HL-Gauss

7,500

1,000

15,000

C51 MSE

HL-Gauss

C51

5,000

10,000

1,000

Two Hot

500

Two Hot

Two Hot

5,000

2,500

500

Asteroids

Atlantis

BankHeist

BattleZone

BeamRider

Two Hot

HL-Gauss

HL-Gauss

Two Hot

HL-Gauss

C51 MSE

EpisodeReturn

MSE

1,250

1,000 HL-Gauss

MSE

800,000

MSE

6,000

HL-Gauss

C51

C51

C51 MSE

Two Hot

C51

750

1,000

20,000

600,000

- 3,000

- 4,500

Two Hot

Two Hot

500

400,000

750

10,000

250

200,000

1,500

500

Berzerk

Bowling

Boxing

Breakout

Carnival MSE

HL-Gauss

MSE

HL-Gauss Two Hot C51

HL-Gauss

90

EpisodeReturn

C51

300

Two Hot C51

MSE

- 3,000

- 4,500 HL-Gauss Two Hot

600

60

Two Hot C51

40

MSE

Two Hot

200

MSE

450

30

HL-Gauss

C51

30

100

0

300

1,500

-30

20

150

Centipede

ChopperCommand

CrazyClimber

DemonAttack

DoubleDunk

C51

HL-Gauss

HL-Gauss

Two Hot

6,000

EpisodeReturn

C51

MSE

10,000

HL-Gauss

MSE

120,000

10

30,000

HL-Gauss

Two Hot

- 3,000

- 4,500

Two Hot C51

8,000

90,000

C51

0

20,000

6,000

Two Hot C51

60,000

- -20

- -10

HL-Gauss Two Hot

4,000

10,000

MSE

1,500

MSE

30,000

MSE

2,000

ElevatorAction

Enduro

FishingDerby MSE

Freeway

Frostbite

C51

HL-Gauss Two Hot

HL-Gauss Two Hot

25 HL-Gauss

HL-Gauss

EpisodeReturn

- 1,500

- 2,000

30

C51

60,000

C51

- 3,000

- 4,500

MSE

0

C51 MSE

Two Hot

Two Hot

C51 MSE

45,000

20

- -75

- -50

- -25

1,000

30,000

MSE

10

Two Hot

1,500

500

15,000

HL-Gauss

0

0

Gopher

Gravitar

Hero

IceHockey

Jamesbond

1,250

HL-Gauss

C51

MSE

EpisodeReturn

10,000

MSE

30,000 HL-Gauss

Two Hot C51

HL-Gauss Two Hot C51

800

Two Hot

1,000

MSE

7,500

C51

0 HL-Gauss

MSE

600

HL-Gauss

750

20,000

MSE

Two Hot

5,000

C51

400

500

Two Hot

-10

10,000

2,500

200

250

JourneyEscape MSE

Kangaroo

Krull

KungFuMaster

MontezumaRevenge

Two Hot

C51 MSE

MSE

HL-Gauss

- 6,000

- 7,500

EpisodeReturn

HL-Gauss

1,500

MSE

C51

10,000

C51

20,000 HL-Gauss Two Hot

- -15,000

- -10,000

- -5,000

Two Hot C51

C51

7,500

1,000

- 3,000

- 4,500 HL-Gauss Two Hot

5,000

10,000

500

2,500

MSE

Two Hot

0

HL-Gauss

HL-Gauss Two Hot

MsPacman MSE

NameThisGame

Phoenix

Pitfall

Pong

HL-Gauss Two Hot

C51 MSE

20

15,000

HL-Gauss Two Hot

- 1,000

- 2,000

- 3,000

- 4,000

C51 MSE

EpisodeReturn

C51

MSE

12,000

HL-Gauss Two Hot

C51 MSE

10

- -300

- -200

- -100

10,000

9,000

0

C51

6,000

HL-Gauss Two Hot

5,000

- -20

- -10

3,000

Pooyan

PrivateEye

Qbert

Riverraid

RoadRunner

HL-Gauss

HL-Gauss

HL-Gauss

MSE

HL-Gauss

EpisodeReturn

15,000

MSE

15,000

MSE

12,000

C51

- 1,000

- 2,000

- 3,000

- 4,000

45,000

MSE

C51

Two Hot C51

Two Hot

Two Hot C51

9,000

10,000

10,000

30,000

Two Hot

6,000

Two Hot

5,000

C51 MSE

15,000

5,000

3,000

0

HL-Gauss

Robotank MSE

Seaquest

Skiing

Solaris

SpaceInvaders

Two Hot

8,000 HL-Gauss

MSE

- -30,000

- -25,000

- -20,000

- -15,000

EpisodeReturn

HL-Gauss

60

Two Hot

45,000

C51

- 1,500

- 2,000 HL-Gauss Two Hot

6,000

C51

C51

45

HL-Gauss

30,000

C51

C51 MSE

4,000

30

MSE

15,000

1,000

2,000

MSE

15

Two Hot

Two Hot

HL-Gauss

StarGunner

Tennis

TimePilot

Tutankham

UpNDown

Two Hot C51

HL-Gauss Two Hot

60,000

MSE

C51

20

EpisodeReturn

Two Hot

10,000

HL-Gauss Two Hot

200 HL-Gauss

12,000

MSE

MSE

45,000

10

MSE

C51

7,500

C51

9,000

MSE

150

C51

0

HL-Gauss

HL-Gauss

Two Hot

30,000

5,000

6,000

100

- -20

- -10

15,000

3,000

2,500

50

Venture

VideoPinball

WizardOfWor

YarsRevenge

Zaxxon

C51 MSE

40,000 HL-Gauss

HL-Gauss

10,000 HL-Gauss

EpisodeReturn

450,000 HL-Gauss

HL-Gauss Two Hot

1,200

12,000

MSE

C51

MSE

7,500

30,000

Two Hot C51

900

9,000

Two Hot

Two Hot

300,000

Two Hot

MSE

5,000

600

6,000

20,000

MSE

C51

150,000

2,500

300

3,000

C51

10,000

0

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200

Iteration

Iteration

Iteration

Iteration

Iteration

###### Figure C.1 | Training curves on single-task online RL (§4.1) for all 60 Atari games. All games ran for 200M frames and ran for: DQN(Adam), C51, Two-Hot, and HL-Gauss.

Online RL: Per-Game Improvement of HNS for HL-Gauss vs. MSE

PercentImprovement

1,000.0

100.0

10.0

1.0

- -100.0

- -10.0

- -1.0

SkiingPhoenixPrivateEyeKrullJamesbondAlienUpNDownTennisBowlingAsteroidsCarnivalMsPacmanStarGunnerAmidarRobotankFishingDerbyJourneyEscapeMontezumaRevengeKungFuMasterPong QbertVentureRiverraidBattleZoneBoxingRoadRunnerPitfallCrazyClimberAirRaidGopherAtlantisKangarooDoubleDunkChopperCommandBeamRiderGravitarPooyanBerzerkZaxxonAssaultFreewayTimePilotIceHockeyYarsRevengeBreakoutTutankhamBankHeistHeroFrostbiteElevatorActionSpaceInvadersEnduroNameThisGameWizardOfWorVideoPinballAsterixDemonAttackSeaquestCentipedeSolaris

Online RL: Atari 200M IQM

IQMNormalizedScore

HL-Gauss

1.5

C51 MSE

1.2

Two Hot

0.9

0.6

0.3

0.0

0 25 50 75 100 125 150 175 200

Iteration

- Figure C.2 | HL-Gauss vs MSE per game in single-task online RL (§4.2.2). (Left) Each column displays the relative final performance of HL-Gauss with respect to MSE in the single-task online RL training curves. This is a summary of the curves displayed in Figure C.1. Note that HL-Gauss outperforms MSE in ≈ 3/4 of all games, and that HL-Gauss scores at least 10% higher on 1/2 of all games. (Right) IQM normalized training curves throughout training.

### D. Additional Results

Multi-task Space Invaders (29 variants)

IMPALA-CNN HL Gauss

IMPALA-CNN MSE

2.5

ResNet-18 HL Gauss

ResNet-18 MSE

ResNet-34 HL Gauss

IQMNormalizedScore

ResNet-34 MSE

2.0

ResNet-50 HL Gauss

ResNet-50 MSE

ResNet-101 HL Gauss

ResNet-101 MSE

1.5

1.0

0.5

0.0

0 200 400 600 800 1000 1200 1400 Number of Frames (in millions)

Figure D.1 | Multi-task online RL (§4.2.2) training curves for Space Invaders trained concurrently on 29 game variants. Note that for every architecture, the HLGauss variant scales better than its respective MSE variant.

Multi-task Asteroids (63 variants)

3.5

IMPALA-CNN HL Gauss

IMPALA-CNN MSE

ResNet-18 HL Gauss

3.0

ResNet-18 MSE

ResNet-34 HL Gauss

IQMNormalizedScore

ResNet-34 MSE

2.5

ResNet-50 HL Gauss

ResNet-50 MSE

ResNet-101 HL Gauss

2.0

ResNet-101 MSE

1.5

1.0

0.5

0.0

0 200 400 600 800 1000 1200 1400 Number of Frames (in millions)

Figure D.2 | Multi-task online RL (§4.2.2) training curves for Asteroids trained concurrently on 63 game variants. These results investigate the scaling properties per architecture of MSE critic loss and cross-entropy HL-Gauss loss. Note that with the architectures larger than Resnet 18, HL-Gauss keeps improving while MSE performance drops after 1300M frames. These larger architectures also all reach higher peak IQM scores with HL-Gauss.

Multi-Task RL: Scaling Capacity

Space Invaders (29 Variants)

IQMNormalizedScore

- 1

- 2

- 3

HL-Gauss MSE

IMPALA-CNNResNet-18ResNet-34ResNet-50ResNet-101

Figure D.3 | Scaling curves on Multi-task Online RL. Online RL scaling results with actor-critic IMPALA with ResNets on Space Invaders. HL-Gauss outperforms MSE for all models. Since human scores are not available for variants, we report normalized scores using a baseline IMPALA agent with MSE loss. See §4.2.2 for more details.

Offline RL (Multi-Game): Scaling Capacity

IQMNormalizedScore

2.00

HL-Gauss

Multi-Game DT (200M)

1.50

C51

MSE

1.00

Two-Hot

0.50

31M 60M 79M Number of Parameters

Figure D.4 | Multi-task Offline RL results presented in terms of DQN normalized scores. Note that when aggregate results are computed with DQN normalization, HLGauss exhibits a faster rate of improvement than C51 as the number of parameters scales up.

