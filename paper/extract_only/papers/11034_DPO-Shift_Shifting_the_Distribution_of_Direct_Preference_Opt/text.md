arXiv:2502.07599v2[cs.CL]6Jun2025

# DPO-Shift: Shifting the Distribution of Direct Preference Optimization

### Xiliang Yang∗

School of Mathematics South China University of Technology xlyangscut@gamil.com

Feng Jiang School of Data Science The Chinese University of Hong Kong, Shenzhen jiangfeng@cuhk.edu.cn

Qianen Zhang School of Data Science The Chinese University of Hong Kong, Shenzhen zhangqianen@cuhk.edu.cn

Lei Zhao Institute of Translational Medicine and National Center for Translational Medicine Shanghai Jiao Tong University zhaolei@sjtu.edu.cn

Xiao Li† School of Data Science The Chinese University of Hong Kong, Shenzhen lixiao@cuhk.edu.cn

## Abstract

Direct Preference Optimization (DPO) and its variants have become increasingly popular for aligning language models with human preferences. These methods aim to teach models to better distinguish between chosen (or preferred) and rejected (or dispreferred) responses. However, prior research has identified that the probability of the chosen responses often decreases during training, and this phenomenon is known as likelihood displacement. To tackle this challenge, in this work, we introduce DPO-Shift to controllably shift the distribution of the chosen probability. Then, we show that DPO-Shift exhibits a fundamental trade-off between improving the chosen probability and sacrificing the reward margin, as supported by both theoretical analysis and experimental validation. Furthermore, we demonstrate the superiority of DPO-Shift over DPO on a downstream task by a designed win rate experiment. We believe this study shows that the likelihood displacement issue of DPO can be effectively mitigated with a simple, theoretically grounded solution.

## 1 Introduction

There has been a growing interest in guiding large language models (LLMs) to generate safe and helpful content to align with human values and intentions, or, taken together, preferences. One of the

∗Most of the work of Xiliang Yang was done when he was with School of Data Science, The Chinese University of Hong Kong, Shenzhen.

†Corresponding author.

Preprint. Under review.

most important methods in this field is known as Reinforcement Learning from Human Feedback (RLHF) [9, 20, 28]. However, multi-stage optimization procedure is raised in these methods, which includes the training of a reward model and the policy model to maximize the reward. Such optimization and computational burden make it challenging to use and analyze, despite its ability to improve the quality of generated responses [6, 1, 31].

Background and Related Works. Recently, DPO [26] and its variants [18, 4, 30, 36, 13, 24] is attracting more and more attention. Given a pair of samples (x,yw,yl) from the dataset, where x is the prompt, and yw and yl represent the chosen and rejected responses, respectively—annotated by strong large language models or humans—the loss of DPO is designed to maximize the margin between the reward of the chosen response and the rejected response for the model πθ. Being offline algorithms, its simplicity makes DPO more applicable and stable. The main difference between DPO and RLHF lies in the treatment of reward function. DPO proposes to directly parameterize it with the policy model, therefore eliminating the need to train an extra reward model and largely simplifying the training process.

However, it has been reported that both log πθ(yw|x) and log πθ(yl|x) often decrease simultaneously during the training process of DPO; see, e.g., [21, 38, 25, 29, 23, 17, 27]. There are several names used to describe such a phenomenon, and we adopt the term “likelihood displacement” [27] in this work. Though DPO still maximizes the reward margin even with this likelihood displacement issue, it remains unfavorable as it causes an unexpected increase in probabilities for responses that are neither preferred nor dispreferred. Unfortunately, this problem hasn’t been noticed nor been solved in some of the latest works in this area [34, 33, 16, 19, 22, 37].

Prior work has attributed this phenomenon to limitations in model capacity [29], the presence of multiple training samples or output tokens [21], and the initial SFT phase [25]. Existing studies, such as [27], have provided theoretical insights into addressing this gap and proposed solving the likelihood displacement problem by filtering the datasets.

Main Contributions. In this paper, we propose DPO-Shift, aiming to solve the likelihood displacement issue of DPO, by adding a parameter function f(λ) to the rejected reward in the Bradley–Terry (BT) model [7], which is detailed in (3).

We briefly illustrate in Figure 1a that, by choosing a proper f(λ) in DPO-Shift, we successfully achieve a balance between the distribution of log πθ(yw|x) and the reward margin. The first row corresponds to a specific choice of f(λ) of our proposed DPO-Shift, where we observe an increased chosen probability compared to DPO (depicted in the second row). This improvement is accompanied by only a slight decrease in the reward margin defined as the difference r(x,yw) − r(x,yl). In fact, we can achieve reward margins that are nearly as high as that of DPO by choosing f(λ) properly; see Section 3. The detailed training process of these two methods is established in Figure 1.

DPO-Shift

1000

1.00

300

yw yL

mean = 0.565

log()yxp

750

0.75

Frequency

| |
|---|

Density

350

500

0.50

DPO-Shift yw DPO-Shift yL

400

250

0.25

DPO yw DPO yL

450

0

0.00

1500 1000 500 0 logp (y x)

2 1 0 1 2 3 r(x, yw) r(x, yL)

Step

0.6

DPO-Shift Reward Margin

DPO

(,)(,)xyxyrrwL

DPO Reward Margin

0.4

mean = 0.582

Frequency

Density

0.2

100 200 300 400 500

r(x, yw) r(x, yL)

logp (y x)

Step

(a) Distribution comparison of log pθ(y | x) and reward margins using DPO and DPO-Shift on UltraFeedback (Llama 3-8B). Left: Chosen vs. rejected logprobabilities. Right: Reward margin.

(b) Training process for DPO and DPO-Shift on UltraFeedback (Llama 3-8B)

Figure 1: Comparison of log probabilities and reward margins between DPO and DPO-Shift. Our main contributions can be summarized as follows.

- (C.1) We propose DPO-Shift to mitigate the likelihood displacement issue of DPO by controllably shifting the distribution of the chosen probability. This is achieved through a new parameter

- function f(λ) introduced in DPO-Shift. Our approach is as simple as DPO and does not require any modifications to the dataset.
- (C.2) We provide a theoretical analysis for the proposed DPO-Shift without imposing additional assumptions. The analysis guarantees that DPO-Shift mitigates the likelihood displacement issue while introducing a fundamental trade-off. Specifically, our theory reveals that DPO-

Shift improves the chosen probability log πθ(yw|x) at the cost of reducing the reward margin that DPO seeks to maximize. Furthermore, the trade-off is explicitly controlled by f(λ), offering practical guidance on its selection. Our findings suggest that f(λ) should be chosen relatively close to 1 to achieve an improvement in the chosen probability over DPO, while only slightly sacrificing the reward margin.

Experimentally, we conduct thorough ablation studies to train the Llama 3-8B and Qwen 2-7B models on the UltraFeedback and Capybara-preferences datasets with fine grained choices of f(λ). The experiment results corroborate our analysis, clearly demonstrating that the likelihood displacement issue is largely mitigated by DPO-Shift and the fundamental trade-off exists between the chosen probability and reward margin in DPO-Shift.

- (C.3) We use downstream experiments to illustrate the improved performance of DPO-Shift over DPO. In particular, we train the Llama 3-8B and Qwen 2-7B models on the UltraFeedback dataset. Then, to fully demonstrate the superiority that DPO-Shift can controllably shift the chosen probability, we conduct a designed win rate experiment. The result shows that DPO-Shift can consistently outperform DPO.

We believe that our study provides a simple yet efficient approach for mitigating the likelihood displacement issue of DPO.

## 2 DPO-Shift: Formulation and Analysis

### 2.1 DPO, likelihood displacement, and DPO-Shift

Consider the preference dataset Dpref = {(x,yw,yl)}, where x is the prompt, yw and yl are the chosen and rejected responses for the prompt x, respectively. Given the prompt x, two responses y1 and y2 are first generated from models or directly taken from static dataset, then the chosen yw and rejected yl are selected from {y1,y2} by human or other strong language models. DPO consists of two steps, including supervised fine-tuning (SFT) and preference optimization.

Supervised Fine-tuning (SFT). In this stage, the LLM πθ is trained to maximize the log-likelihood of y given x with cross-entropy loss:

LSFT(πθ) = −E(x,y)∼D

[log πθ(y|x)],

min

SFT

θ

Here, DSFT = {(x,y)} is the normal prompt-response dataset used for auto-regressive language modeling and πθ(y|x) = |iy=1| πθ(yi|x,y1:i−1). For convenience, we refer to the model after the SFT stage as the “SFTed model".

Preference Optimization (PO). In this stage, DPO parameterizes the reward model with the LLM πθ via

πθ(y|x) πref(y|x)

+ log Z(x) , (1)

r(x,y) = β log

where Z(x) is the partition function and πref is known as the reference model and usually set to be the “SFTed model".

Incorporating this reward function into the Bradley-Terry (BT) model, i.e., P(yw > yl|x) = σ(r(x,yw) − r(x,yl)). Then, by maximizing the log-likelihood of P(yw > yl|x), DPO arrives at the following objective function:

πθ(yl|x) πref(yl|x)

πθ(yw|x) πref(yw|x) − β log

. (2)

LDPO(π) = −E log σ β log

Here, the expectation is taken over the dataset Dpref. The model after the PO state is called the “POed model".

Likelihood Displacement. Let us recall the distributions of the likelihood log πθ(yw|x) and log πθ(yl|x) of the DPO model in Figure 1. It is easy to observe that the highest likelihood region for both the chosen and rejected responses are decreased dramatically after DPO. Though some other likelihood regions of the chosen and rejected responses have increased after DPO, the averaged likelihood of log πθ(yw|x) and log πθ(yl|x) over the entire test set is overall decreasing according to our experiment results, aligning with the likelihood displacement phenomenon observed in the existing literature [29, 21, 27, 23, 38, 17, 25, 15]. In conclusion, the likelihood displacement occurs not only in the training stage but also in the test dataset, which is counter-intuitive and can be harmful to the model’s generalization ability.

An important factor causing the likelihood displacement issue during the PO stage gives rise to the semantic similarity between the chosen yw and rejected yl pairs in Dpref, as observed in the existing works; see, e.g., [29, 15, 27, 21]. This is indeed implied by the generation process of contemporary preference datasets. For instance, the UltraFeedback dataset [11] is generated by using different LLMs to response to the same prompt x, and then it selects yw and yl using GPT4. To demonstrate it, we pick the following examples from UltraFeedback:

- Q1: ...Select from female and male... Solution: chosen: Female. rejected: Female.
- Q2: Write the right answer to the question based on... chosen: Dan, the protagonist, got a coke out of the cooler. rejected: Dan got coke out of the cooler.

This partly explains that the model tends to assign similar probabilities to both responses. In the DPO objective function, it seeks to maximize the margin between the probability of the chosen and rejected responses even if they are semantically similar. Hence, instead of the ideal case where it maximizes the chosen probability while minimizes the rejected one, it often reduces the probability of both responses with similar semantic structures, though their margin is enlarged. This leads to the likelihood displacement issue. Consequently, the model favors responses that are neither chosen nor rejected.

DPO-Shift. To address this counter-intuitive and harmful likelihood displacement issue of DPO, we introduce DPO-Shift in this work. The motivation behind the proposed method is to alleviate the problem caused by the similarity of the chosen and rejected pairs. As we analyzed previously, the chosen probability decreases accordingly when the DPO objective maximizes the margin between two semantically similar responses. Based on this observation, we propose to add a real-valued function 0 < f(λ) < 1 to the reward of the rejected response. This helps the BT model to rank correctly by reducing the confrontation between two semantically similar responses, potentially mitigating the likelihood displacement issue of DPO. Mathematically, our proposed formulation is displayed in the following:

πθ(yl|x) πref(yl|x)

πθ(yw|x) πref(yw|x) − f(λ) · β log

. (3)

LDPO-Shift(π) = −E log σ β log

### 2.2 Analysis for DPO-Shift

We analyze the effects of DPO-Shift for two important quantities, including the likelihood of the chosen response log πθ(yw|x) and the indicator function of the reward margin 1{(x,yw,yl)|log π

θ(yw|x)

θ(yl|x)

πref(yw|x) − log π

πref(yl|x) > 0}. The latter reflects the model’s ability to align with human preferences and is implicitly maximized by DPO’s objective. We define the two target functions as follows:

- ω1(θ) = E[log πθ (yw|x)], (4)
- ω2(θ) = E 1 log

πθ(yw|x) πref(yw|x) − log

πθ(yl|x) πref(yl|x)

> 0 . (5)

The likelihood displacement issue of DPO enlarges ω2 while decreases ω1. To provide an analytical characterization, we alter the discontinuous ω2 and consider its smoothed version

πθ(yw|x) πref(yw|x) − γ log

πθ(yl|x) πref(yl|x)

, (6)

ω2(θ) = E σ γ log

where γ is the smoothing factor. To compare DPO-Shift with the original DPO, we introduce two functions measuring the gaps between targets after one step of optimization (i.e., updating θt to θt+1) with different objective functions:

#### − ω1(θt+1)

#### g1(t + 1) = ω1(θt+1)

DPO-Shift

#### ,g2(t + 1) = ω2(θt+1)

DPO

We characterize the two gap functions in the following theorem. Theorem 2.1. Given θt and learning rate η and denote

#### − ω2(θt+1)

#### .

DPO

DPO-Shift

(7)

πθ(yl|x) πref(yl|x) − γ log

πθ(yw|x) πref(yw|x)

c(θ) = γσ f(λ) · γ log

,

π (yl|x) πref(yl|x) − log

π (yw|x) πref(yw|x)

. We have

η1(θ) = ησ log

g1(t + 1) = (1 − f(λ))u1, g2(t + 1) = (1 − f(λ))u2. (8) Here,

- u1 = E c(θ) · ∇θ log πθ (yl|x)⊤ ∇θ log πθ (yw|x) ,
- u2 = E η1(θ) ∇θ log πθ (yl|x)⊤ ∇θ log πθ (yw|x) − ∥∇θ log πθ (yl|x)∥2 .

(9)

The proof of Theorem 2.1 is deferred to Appendix B. It is worth mentioning that our derivation in the proof of Theorem 2.1 applies to every single sample, and hence the result in Theorem 2.1 applies to u1 and u2 defined using any specific dataset. We state the results in expectation in order to be data-independent.

This theorem explains the pros and cons of DPO-Shift, yielding indications for choosing f(λ). We provide detailed discussions below.

Fundamental Trade-off. We are interested in characterizing the sign of the two gap functions using

#### (8). To compute u1 and u2 on a specific Dpref, we define the following sample-based version: ui1 = ci(θ) · ∇θ log πθ yil|xi ⊤ ∇θ log πθ yiw|xi , ui2 = η1 ∇θ log πθ yil|xi ⊤ ∇θ log πθ yiw|xi − ∇θ log πθ yil|xi 2 .

Then, we compute the sample average u1 = i ui1/|Dref| and u2 = i ui2/|Dref|. On the test set of UltraFeedback, when setting γ = 1 and πθ to be the SFTed Llama 3-8B, we obtain that

- u1 = 4.84 × 108. For u2, since η1 is bounded, we set it to be 1 and obtain u2 = −4.28 × 109. In terms of frequency, 71.4% of {ui1} are positive and 81.7% of {ui2} are negative. These results indicate a clear positivity of u1 while a clear negativity of u2. Indeed, positivity of u1 is expected due to the semantic similarity between yw and yl in contemporary Dpref.

Since we choose 0 < f(λ) < 1, we have 1 − f(λ) > 0. In this case, g1 > 0 as u1 > 0. This immediately concludes that DPO-Shift improve the chosen probability (or likelihood) log πθ(yw|x) compared to DPO, solving the undesired likelihood displacement issue. However, there is an explicit trade-off to achieving this improvement. Since u2 can be negative as shown in our test result, g2 can be negative, leading to reduced reward margin of DPO-Shift compared to DPO.

In summary, DPO-Shift improves the chosen probability over DPO, at the cost of decreasing the reward margin. This also yields the indications for choosing f(λ), which we describe below.

Indications for Choosing f(λ). As analyzed previously, the important trade-off exists in DPO-Shift. A slightly deeper analysis reveals that a smaller f(λ) leads to more increase in chosen probability while a more severe drop in the reward margin. Thus, this indicates that choosing a relatively large f(λ), i.e., close to 1, helps to balance both sides. That is, the chosen probability is improved reasonably compared to DPO, in the meanwhile the reward margin of DPO-Shift is only decreased slightly. The balance controlled by f(λ) is thoroughly demonstrated by our experiment results in Section 3.

For the strategy of choosing of f(λ), the first one is to fix it all along the optimization process, i.e., f(λ) = λ. We denote this strategy as fixed. We also propose to vary λ between the minimal λmin and the maximal λmin along with time t. We denote this strategy as f(λmin,λmax,t). In this paper, we mainly have linear increase/decrease between λmin < 1 and λmax = 1. Set the maximal iteration steps to be T, the detailed strategy for the linear increase version of f(λmin,λmax,t) is t/T(λmax − λmin) + λmin, while the linearly decreased version is t/T(λmin − λmax) + λmax. They are separately denoted as linear_increase and linear_decrease.

Can f(λ) be Chosen Larger than 1? By default, we choose f(λ) < 1 in DPO-Shift to achieve chosen probability improvement, which is based on the hypothesis that u1 is generally positive. If we encounter the case where u1 < 0, e.g., when most pairs yw and yl are dissimilar to each other,

- u2 must be negative as well. Interestingly, in this case Theorem 2.1 suggests that DPO-Shift with f(λ) > 1 can lead to simultaneous improvements for both the chosen probability and reward margin.

However, the event u1 < 0 is likely to be very rare, given the general similarity between yw and yl in existing Dpref. Using f(λ) > 1 when u1 > 0 leads to catastrophic results. For instance, we trained Llama 3-8B on UltraFeedback with fixed f(λ) = 1.05, and the model quickly exhibited crash behavior, producing unreadable responses. Therefore, choosing f(λ) > 1 should be done with great care unless u1 is clearly negative and the overwhelming majority of {ui1} has negative values. Though choosing f(λ) > 1 is highly discouraged, the above analysis provides a possible direction to further improve DPO-Shift. Before implementing the PO state, we can first calculate {ui1} for the entire training set. Then, we assign a specific f(λi) for each data point, i.e., f(λi) < 1 when ui1 > 0 and f(λi) > 1 when ui1 < 0. Adopting such a carefully crafted approach requires a significant amount of additional implementation code and falls beyond the scope of this work. We leave it as future work.

- 3 Experimental Results

In this section, we present the main results of our experiments, highlighting the performance of DPO-Shift on various benchmarks and ablation studies.

### 3.1 Experimental Setup

We perform our experiment on two models, Llama 3-8B [2] and Qwen 2-7B [5], under base setup. We mainly carry out the following two parts of experiments.

Verification Experiment. This part of the experiment is designed to validate the theoretical results presented in Section 2.2. Specifically, we consider three strategies for selecting f(λ) in (3): fixed, linear_increase, and linear_decrease. We design a series of ablation studies for each of the strategy by altering the choice of f(λ). For the fixed strategy, we perform an ablation study by evaluating fixed f(λ) from the range [0.5,0.55,0.6,0.65,0.75,0.8,0.85,0.9,0.95]. For the linear_increase and linear_decrease strategies, we set λmin to values in [0.75,0.85,0.95] and fix λmax = 1. We compute the log πθ(yw|x), log πθ(yl|x), and reward margin for models trained with these f(λ) strategies on the test sets of their respective training datasets. To further illustrate the phase transition phenomenon in these probability distributions as f(λ) varies from f(λ) < 1 to f(λ) = 1, we extend the fixed strategy ablation study by including f(λ) values of [0.96,0.97,0.98,0.99].

Downstream Performance Evaluation. This experiment is primarily designed to evaluate the general performance of the model trained using our proposed method. For the two SFTed models trained on the UltraFeedback dataset, we evaluate the win rate of DPO-Shift against DPO using 2,000 randomly selected questions from the Capybara dataset and the test_prefs split of the UltraFeedback dataset. For the evaluation, we employ the Llama3.3:70b-instruct-q4_K_M model provided in the ollama library, a 4-bit quantized version of the latest Llama 3.3-Instruct 70B, which delivers performance comparable to Llama 3.1-405B.

Additional experiment details are deferred to Appendix A.1.

DPO-Shift with f( )=0.55

DPO-Shift with f( )=0.75

DPO-Shift with f( )=0.95

DPO

1000

yw yL

750

Frequency

Frequency

Frequency

Frequency

| |
|---|

500

250

0

1500 1000 500 0 logp (y x)

logp (y x)

logp (y x)

logp (y x)

2.0

1.5

mean = 0.289

mean = 0.461

mean = 0.588

mean = 0.582

Density

Density

Density

Density

1.0

0.5

0.0

2 0 2 r(x, yw) r(x, yL)

r(x, yw) r(x, yL)

r(x, yw) r(x, yL)

r(x, yw) r(x, yL)

- Figure 2: Distribution for log πθ(yw|x), log πθ(yl|x) (up) and reward margin as well as the mean on test set split of UltraFeedback for Llama 3-8B trained on UltraFeedback with fixed strategy. Only limited cases of f(λ) are listed. For a full ablation study, please refer to Appendix A.3. The ranges of the y-axis of all subfigures are the same.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.5 0.6 0.7 0.8 0.9 1.0

lambda

0.70

0.72

- 0.74

Llama 3-8B trained on ultrafeedback

- Figure 3: Reward accuracy vs different f(λ) on the UltraFeedback test set for Llama 3-8B trained on UltraFeedback with fixed, where f(λ) is selected from 0.5 to 0.95.

0.00200

1.0

|DPO|mean|=0.582| | |
|---|---|---|---|---|
|DPO|-Shift|mean=|0.588| |
| | | | | |
| | | | | |
| | | | | |

DPO-shift

DPO

0.00175

0.8

0.00150

0.00125

0.6

Density

0.00100

0.4

0.00075

0.00050

0.2

0.00025

0.00000

0.0

1000 500 0 logp (yw x)

2 1 0 1 2 3

r(x, yw) r(x, yL)

Figure 4: The comparison between DPO-Shift with f(λ) = 0.99 and DPO on the UltraFeedback test set for Llama 3-8B. Left: Distribution for log πθ(yw|x). Right: Reward accuracy and margin distribution.

### 3.2 Experimental Verification for DPO-Shift

We report some representative results from our ablation studies for fixed, linear_increase and linear_decrease. We mainly evaluate them with 2 metrics, including distribution for log πθ(yw|x) and log πθ(yl|x), and distribution for reward margin.

We first study fixed strategy. The two metrics for evaluation, including distributions for log πθ(yw|x) and log πθ(yl|x), and distribution for reward margin are demonstrated in Figure 2. Thorough ablation experiment results on different datasets and models are supplemented in Appendix A.3, where we have more choices of f(λ) for Llama 3-8B trained on UltraFeedback, new experiments for Llama 3-8B trained on Capybara, new experiments for Qwen 2-7B trained on UltraFeedback, and new experiments for Qwen 2-7B trained on Capybara. In Figure 2, we observe a consistent “shift” in the distribution of log πθ(yw|x) and log πθ(yl|x), moving from high-probability regions to lowerprobability regions with increasing f(λ). This shift is accompanied by a mass transition in the reward margin, shifting from areas less than 0 to areas greater than 0. The extent of this “shift” largely depends on the value of f(λ). Specifically, when f(λ) is very small, it leads to higher probabilities for the language model on the chosen answer yw. However, this comes at the cost of decreased and shifted reward margin. Thus, small f(λ) can result in “over-fitting" to the chosen answer, losing the contrastive learning ability against the rejected one and reducing the model’s performance. Fortunately, by selecting a relatively larger f(λ) that is closer to 1, an increase in the reward margin with only a limited trade-off in the chosen probability is observed. This aligns with our analysis proposed in Section 2.2 that larger f(λ) helps balance both metrics. For example, f(λ) = 0.9,0.95 achieves a higher probability of the chosen responses than DPO while maintaining almost the same reward margin.

DPO-Shift with f( ) = 0.75

DPO-Shift with min = 0.75

DPO-Shift with f( ) = 0.75

DPO-Shift with min = 0.75

1000

1.0

###### yw yL

800

0.8

mean = 0.461

mean = 0.517

Frequency

Frequency

600

0.6

Density

Density

400

0.4

200

0.2

0

0.0

1500 1000 500 0 logp (y x)

2 0 2 r(x, yw) r(x, yL)

r(x, yw) r(x, yL)

logp (y x)

- Figure 5: Comparison between fixed strategy with f(λ) = 0.75 and linear_decrease with λmin = 0.75 on the test set split of UltraFeedback for Llama 3-8B trained on UltraFeedback. Left:

Distribution for log πθ(yw|x) and log πθ(yl|x). Right: Reward accuracy and distribution for reward margin.

In addition to the reward margin, we also display the reward of DPO-Shift with several choices of f(λ) and DPO in Figure 3, where the reward accuracy is defined as the frequency ratio of r(x,yw) − r(x,yl) > 0 over the test dataset split. Similar to the reward margin, it illustrates a clear increase in reward accuracy along with increasing f(λ), further corroborating our theory.

Additionally, we conduct experiments with more fine grains to investigate the effect of f(λ) on the distribution of log πθ(yw|x) and log πθ(yl|x) as it approaches 1. In the case of fixed, we select a new set of fixed f(λ)s, including [0.96,0.97,0.98,0.99]. We report the comparison between fixed with f(λ) = 0.99 and the original DPO with the distribution of chosen probability and reward margin in the Figure 4. Compared to DPO, an obvious shift of the distribution log πθ(yw|x) can be observed, while the distribution of the reward margin remains nearly unchanged. The full experimental results for f(λ) ∈ [0.96,0.97,0.98,0.99] are reported in Appendix A.5, in which the results are consistent with our analysis.

To achieve a possibly better trade-off between the two distributions, we conduct experiments on linear_increase and linear_decrease strategies for choosing f(λ). We report the result for linear_decrease with λmin = 0.75 in Figure 5. It can be seen that this dynamic f(λ) strategy achieves a more satisfactory balance between reward accuracy and log πθ(yw|x) compared to fixed with f(λ) = 0.75. Thorough ablation experiment results for linear_increase and linear_decrease strategies are provided in Appendix A.4, where we include more fine-grained choices of λmin, the dataset Capybara, and the model Qwen 2-7B.

Summary. By setting f(λ) properly in DPO-Shift, we can achieve controllable trade-off between chosen probability and reward accuracy. A carefully chosen For example, if one chooses f(λ) = 0.95 in DPO-Shift, the likelihood displacement issue can be largely mitigated, while the reward margin can be kept nearly unchanged compared to DPO.

### 3.3 Downstream Performance

When comparing the answers generated by DPO and DPO-Shift, we observe that the model trained with DPO generally generates longer and less meaningful responses, while DPO-Shift tends to produce more concise and high-quality responses. Based on these observations, we adopt perplexity, a widely used evaluation metric [10, 8, 35] for language models. Perplexity quantifies how well a probability model predicts a sample of data, essentially measuring the model’s uncertainty about the data. Lower perplexity indicates that the model is better at predicting the sample. The perplexity of DPO and DPO-Shift trained on UltraFeedback with fixed f(λ) = 0.95 is evaluated on the chosen responses from the test split of the UltraFeedback dataset. The results are 4.475 for DPO-Shift and 18.996 for DPO, further demonstrating the potential advantage of DPO-Shift.

### f(λ) strategy Win Lose

SFT 34.20% 65.80% fixed 0.55 51.40% 49.60% fixed 0.75 57.30% 42.70% fixed 0.95 68.95% 31.05% linear_increase 0.95 67.40% 32.60% linear_decrease 0.95 72.15% 27.85% SimPO [18] 56.75% 43.25% KTO [14] 61.00% 39.00% IPO [4] 60.50% 39.50%

Table 1: Win rate experiment against DPO using Llama 3-8B trained on the UltraFeedback dataset and tested with questions from the test split of UltraFeedback. Results for DPO-Shift using all three strategies including fixed, linear_increase, linear_decrease are shown.

To fully demonstrate the better alignment of DPO-Shift with the chosen response, we conduct a win rate experiment. In this setup, the judge model is provided with the question, the reference answer from the dataset, and the answers generated by DPO-Shift and DPO. The judge model then judges the responses of DPO-Shift and DPO based on their general quality and how close to the reference answer they are. We put the judge prompts in Appendix A.2.

We test the win rate using the test split of UltraFeedback for Llama 3-8B, which is trained on UltraFeedback. In Table 1, we report the results for DPO-Shift and other preference optimization methods including SimPO [18], KTO [14] and IPO [4]. To ensure every method is tested at its best performance, we directly adopt the checkpoints provided by SimPO3, where extensive hyperparameter grid search has been conducted. Note that the experimental setups for the SimPO, KTO, and IPO checkpoints are identical to ours, as we strictly follow the same configuration as SimPO. Therefore, these models are directly comparable.

One can observe from Table 1 that DPO-Shift consistently outperforms DPO once f(λ) is chosen properly, i.e., when f(λ) is relatively closer to 1, corroborating our analysis in Section 2.2 and the verification experiments in Section 3.2. Specifically, DPO-Shift wins DPO 72.15%−27.85%

2 = 22.15% more if the linear_decrease with λmin = 0.95 strategy is chosen for DPO-Shift. Our method also exhibits superior performance compared to SimPO, KTO, and IPO, as DPO-Shift wins DPO more compared to these methods. For instance, KTO wins 61% against DPO, while DPO-Shift using linear_decrease with λmin = 0.95 wins 72.15% against DPO, further illustrating the advantage of our method. In Appendix A.6, we also display the win rate experiment of the Qwen 2-7B model. As SimPO does not provide checkpoints for Qwen 2-7B, we only compare DPO-Shift against DPO, while omitting other methods. The results for Qwen 2-7B align with our observations from Table 1 Summary. DPO-Shift outperforms DPO in terms of downstream performance when f(λ) is set properly to achieve a good balance between the chosen probability and the reward margin.

### 3.4 Ablation Study: Comparison with Other Potential Methods

To clearly distinguish DPO-Shift from other baseline alternatives, we supplement the comparison on the mean of log πθ(yw|x), log πθ(yl|x), and the reward margin results with other existing preference optimization methods including IPO, SimPO, and KTO. To lift log πθ(yw|x) in the DPO stage, we note that it is common practice in the community to combine the original DPO objective with the SFT loss, which is denoted as α-DPO in our paper:

α |yw|

LDPO-α(π) = LDPO(π) −

log π(yw|x),

This approach may also mitigate the likelihood displacement issue by properly selecting α. The experiment results are presented in Table 2, where DPO-Shift uses f(λ) = 0.95.

We tried different α and found that α = 0.1 gives almost the same log probability of the chosen response compared to DPO-Shift. It can be observed that α-DPO works, as it mitigates the likelihood

3https://github.com/princeton-nlp/SimPO?tab=readme-ov-file

displacement issue. However, though it achieves almost the same log probability of the chosen response as DPO-Shift, it has a higher log probability of the rejected response. This is reasonable as α-DPO lifts the probability of yw by directly adding a SFT term on yw, at the price of reducing the effects of the DPO term and hence weakening the confrontation between yw and yl. Consequently, the reward margin of this approach decreased and is inferior to DPO-Shift. As for other existing preference optimization methods, DPO-Shift demonstrates consistent advantages on all metrics compared to SimPO and KTO. In the case of IPO, although it exhibits a higher reward margin, it does so at the cost of significantly lower chosen log probability, which further worsens the likelihood displacement issue.

Summary. The fundamental property of DPO-Shift lies in that it can increase log πθ(yw|x) (i.e., mitigating the likelihood displacement issue), while keeping the reward margin nearly unchanged compared to DPO once f(λ) is chosen properly. This, together with Table 1, further justifies the superiority of DPO-Shift.

Metric SFT DPO-Shift DPO IPO SimPO KTO α-DPO log πθ(yw|x) -299.09 -330.86 -425.47 -591.85 -404.46 -360.93 -314.59 log πθ(yl|x) -278.60 -363.67 -457.25 -644.37 -412.29 -420.29 -343.73 Reward margin N/A 0.58 0.58 0.73 0.28 0.49 0.49

Table 2: Comparison between DPO-Shift and other potential methods.

## 4 Conclusion and Discussions on Limitations

In this work, we proposed DPO-Shift, which controllably shifts the distribution of the chosen probability. Our method guarantees to mitigate the likelihood displacement issue of DPO while introducing a fundamental trade-off between the chosen probability and reward margin. By selecting f(λ) carefully, the chosen probability can be improved, in the meanwhile the reward margin is only slightly sacrificed. Extensive ablation experiments across various models and datasets confirm the validity of our theoretical findings. To further demonstrate the advantages of DPO-Shift, we conducted experiments on downstream task using a designed win rate experiment. Clear performance improvements over DPO were observed, highlighting the robustness and effectiveness of our approach.

Limitations. We only consider a global f(λ) for all the data points. A more crafted strategy on the selection of f(λ) can be a possible direction to further improve DPO-Shift, as we commented at the end of Section 2.2. We leave it as future work.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. ArXiv, abs/2303.08774, 2023.
- [2] AI@Meta. Llama 3 model card. 2024.
- [3] argilla. Capybara-preferences dataset card. 2024.
- [4] Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pages 4447–4455. PMLR, 2024.
- [5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [6] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

- [7] Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39:324, 1952.
- [8] Stanley F Chen, Douglas Beeferman, and Roni Rosenfeld. Evaluation metrics for language models. 1998.
- [9] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.
- [10] Nathan Cooper and Torsten Scholak. Perplexed: Understanding when large language models are confused. arXiv preprint arXiv:2404.06634, 2024.
- [11] Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. UltraFeedback: Boosting language models with high-quality feedback. In ICML, 2024.
- [12] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. In EMNLP, 2023.
- [13] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. KTO: Model alignment as prospect theoretic optimization. ArXiv, abs/2402.01306, 2024.
- [14] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.
- [15] Jiwoo Hong, Noah Lee, and James Thorne. ORPO: Monolithic preference optimization without reference model. ArXiv, abs/2403.07691, 2024.
- [16] Aobo Kong, Wentao Ma, Shiwan Zhao, Yongbin Li, Yuchuan Wu, Ke Wang, Xiaoqian Liu, Qicheng Li, Yong Qin, and Fei Huang. Sdpo: Segment-level direct preference optimization for social agents. arXiv preprint arXiv:2501.01821, 2025.
- [17] Zhihan Liu, Miao Lu, Shenao Zhang, Boyi Liu, Hongyi Guo, Yingxiang Yang, Jose Blanchet, and Zhaoran Wang. Provably mitigating overoptimization in rlhf: Your sft loss is implicitly an adversarial regularizer. arXiv preprint arXiv:2405.16436, 2024.
- [18] Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734, 2024.
- [19] Motoki Omura, Yasuhiro Fujita, and Toshiki Kataoka. Entropy controllable direct preference optimization. arXiv preprint arXiv:2411.07595, 2024.
- [20] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke E. Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Francis Christiano, Jan Leike, and Ryan J. Lowe. Training language models to follow instructions with human feedback. In NeurIPS, 2022.
- [21] Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. Smaug: Fixing failure modes of preference optimisation with DPO-positive. arXiv preprint arXiv:2402.13228, 2024.
- [22] Junshu Pan, Wei Shen, Shulin Huang, Qiji Zhou, and Yue Zhang. Pre-dpo: Improving data utilization in direct preference optimization using a guiding reference model. arXiv preprint arXiv:2504.15843, 2025.
- [23] Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. arXiv preprint arXiv:2404.19733, 2024.
- [24] Ryan Park, Rafael Rafailov, Stefano Ermon, and Chelsea Finn. Disentangling length from quality in direct preference optimization. ArXiv, abs/2403.19159, 2024.

- [25] Rafael Rafailov, Joey Hejna, Ryan Park, and Chelsea Finn. From r to Q∗: Your language model is secretly a Q-function. arXiv preprint arXiv:2404.12358, 2024.
- [26] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2023.
- [27] Noam Razin, Sadhika Malladi, Adithya Bhaskar, Danqi Chen, Sanjeev Arora, and Boris Hanin. Unintentional unalignment: Likelihood displacement in direct preference optimization. arXiv preprint arXiv:2410.08847, 2024.
- [28] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020.
- [29] Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jeff Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn, and Aviral Kumar. Preference fine-tuning of llms should leverage suboptimal, on-policy data. arXiv preprint arXiv:2404.14367, 2024.
- [30] Yunhao Tang, Zhaohan Daniel Guo, Zeyu Zheng, Daniele Calandriello, Rémi Munos, Mark Rowland, Pierre Harvey Richemond, Michal Valko, Bernardo Ávila Pires, and Bilal Piot. Generalized preference optimization: A unified approach to offline alignment. arXiv preprint arXiv:2402.05749, 2024.
- [31] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.
- [32] Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Shengyi Huang, Kashif Rasul, Alvaro Bartolome, Alexander M. Rush, and Thomas Wolf. The Alignment Handbook.
- [33] Teng Xiao, Yige Yuan, Zhengyu Chen, Mingxiao Li, Shangsong Liang, Zhaochun Ren, and Vasant G Honavar. Simper: A minimalist approach to preference alignment without hyperparameters. arXiv preprint arXiv:2502.00883, 2025.
- [34] Teng Xiao, Yige Yuan, Huaisheng Zhu, Mingxiao Li, and Vasant G Honavar. Cal-dpo: Calibrated direct preference optimization for language model alignment. arXiv preprint arXiv:2412.14516, 2024.
- [35] Frank F Xu, Uri Alon, Graham Neubig, and Vincent Josua Hellendoorn. A systematic evaluation of large language models of code. In Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming, pages 1–10, 2022.
- [36] Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. Contrastive preference optimization: Pushing the boundaries of LLM performance in machine translation. ArXiv, abs/2401.08417, 2024.
- [37] Huimin Xu, Xin Mao, Feng-Lin Li, Xiaobao Wu, Wang Chen, Wei Zhang, and Anh Tuan Luu. Full-step-dpo: Self-supervised preference optimization with step-wise rewards for mathematical reasoning. arXiv preprint arXiv:2502.14356, 2025.
- [38] Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, et al. Advancing llm reasoning generalists with preference trees. arXiv preprint arXiv:2404.02078, 2024.

##### [39] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In NeurIPS Datasets and Benchmarks Track, 2023.

## Contents

- 1 Introduction 1
- 2 DPO-Shift: Formulation and Analysis 3

- 2.1 DPO, likelihood displacement, and DPO-Shift . . . . . . . . . . . . . . . . . . . . 3
- 2.2 Analysis for DPO-Shift . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 3 Experimental Results 6

- 3.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.2 Experimental Verification for DPO-Shift . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.3 Downstream Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.4 Ablation Study: Comparison with Other Potential Methods . . . . . . . . . . . . . 9

- 4 Conclusion and Discussions on Limitations 10

- A Supplemented Experimental results 15

- A.1 Additional Experimental details . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- A.2 Llama 3.3-70B Prompts for Win Rate Experiment . . . . . . . . . . . . . . . . . . 15
- A.3 Ablation Studies for fixed . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- A.4 Ablation Studies for linear_increase and linear_decrease . . . . . . . . . . 20
- A.5 Ablation Studies for Fine-grained fixed . . . . . . . . . . . . . . . . . . . . . . . 24
- A.6 Supplementary Results for Win Rate Experiment . . . . . . . . . . . . . . . . . . 25

- B Proof of Theorem 2.1 26

## A Supplemented Experimental results

### A.1 Additional Experimental details

For the SFT stage, we train our models for one epoch on UltraChat-200k dataset [12] to obtain an SFT model with a learning rate of 2e-5. For Llama 3-8B model, we directly use the off-the-shelf models from [18] (princeton-nlp/Llama-3-Base-8B-SFT), which follows the same training strategy.

For the Preference optimization stage, we aim to verify the robustness of DPO-Shift. To this end, we perform preference optimization on UltraFeedback dataset [11], Capybara-preferences dataset [3]. Notice that since the Capybara dataset lacks a given test dataset, we randomly divide 5% of the training dataset into test sets for subsequent analysis and observations. We start from the two SFTed models and fine-tune them with these two datasets. As for the training parameter, we mainly follow the experimental detail in [18], and train the model for 1 epoch with a learning rate of 6e-7. The optimizer we choose is paged_Adamw_32bit with the implementation from bitsandbytes.

Computation Environment. The training experiments in this paper were conducted using 8×A800 80GB GPUs and 4×A100 40GB GPUs, based on the alignment handbook repository [32]. Specifically, all Llama 3 8B-related experiments were performed on 4×A100 GPUs, while Qwen 2-7B experiments utilized 8×A800 GPUs. For downstream performance evaluation, MT-Bench was assessed using the llm_judge submodule from the FastChat [39] repository and the win rate experiments were conducted with the Ollama repository.

### A.2 Llama 3.3-70B Prompts for Win Rate Experiment

You are tasked with comparing the responses of two assistants, Assistant A and Assistant B, to a user’s question. Additionally, you will be provided with a reference answer to evaluate the quality of the responses from both assistants.

First, output only ’A’ or ’B’ to indicate your judgment on which response is better. Then, provide a one-sentence explanation for your choice. The principles for your judgment should be based on the following criteria:

- 1. The most important criterion is to select the response whose meaning is essentially closer to the reference answer.
- 2. Do not judge the quality of the two responses based on their length.
- 3. Evaluate the responses based on their helpfulness, relevance, accuracy, depth, and conciseness.

User’s Question: {question}

Reference Answer: {ref_answer}

- Assistant A’s Response: {response_compare}
- Assistant B’s Response: {response_baseline}

### A.3 Ablation Studies for fixed

DPO-Shift with f( )=0.5

DPO-Shift with f( )=0.55

DPO-Shift with f( )=0.6

SFTed model

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

DPO-Shift with f( )=0.65

DPO-Shift with f( )=0.7

DPO-Shift with f( )=0.75

DPO-Shift with f( )=0.8

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

DPO-Shift with f( )=0.85

DPO-Shift with f( )=0.9

DPO-Shift with f( )=0.95

DPO

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

- Figure 6: Distribution for log πθ(yw|x) and log πθ(yl|x) on test set split of UltraFeedback for Llama 3-8B trained on UltraFeedback, where DPO-Shift uses fixed strategy. The ranges of the y-axis of all subfigures are the same.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

r(x, yw) r(x, yL)

SFTed model

r(x, yw) r(x, yL)

mean = 0.236

DPO-Shift with f( ) = 0.5

r(x, yw) r(x, yL)

mean = 0.289

DPO-Shift with f( ) = 0.55

r(x, yw) r(x, yL)

mean = 0.336

DPO-Shift with f( ) = 0.6

r(x, yw) r(x, yL)

mean = 0.375

DPO-Shift with f( ) = 0.65

r(x, yw) r(x, yL)

mean = 0.433

DPO-Shift with f( ) = 0.7

r(x, yw) r(x, yL)

mean = 0.461

DPO-Shift with f( ) = 0.75

r(x, yw) r(x, yL)

mean = 0.508

DPO-Shift with f( ) = 0.8

r(x, yw) r(x, yL)

mean = 0.552

DPO-Shift with f( ) = 0.85

r(x, yw) r(x, yL)

mean = 0.565

DPO-Shift with f( ) = 0.9

r(x, yw) r(x, yL)

mean = 0.588

DPO-Shift with f( ) = 0.95

r(x, yw) r(x, yL)

mean = 0.582

DPO

- Figure 7: Distribution for reward margin and its mean on test set split of UltraFeedback for Llama 3-8B trained on UltraFeedback, where DPO-Shift uses fixed strategy. The ranges of the y-axis of all subfigures are the same.

2000 1500 1000 500 0

2000 1500 1000 500 0

2000 1500 1000 500 0

2000 1500 1000 500 0

DPO-Shift with f( )=0.65

DPO-Shift with f( )=0.7

DPO-Shift with f( )=0.75

DPO-Shift with f( )=0.8

2000 1500 1000 500 0

2000 1500 1000 500 0

2000 1500 1000 500 0

2000 1500 1000 500 0

DPO-Shift with f( )=0.85

DPO-Shift with f( )=0.9

DPO-Shift with f( )=0.95

DPO

2000 1500 1000 500 0

2000 1500 1000 500 0

2000 1500 1000 500 0

2000 1500 1000 500 0

- Figure 8: Distribution for log πθ(yw|x) and log πθ(yl|x) on test set split of Capybara for Llama 3-8B trained on Capybara, where DPO-Shift uses fixed strategy. The ranges of the y-axis of all subfigures are the same.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

r(x, yw) r(x, yL)

SFTed model

r(x, yw) r(x, yL)

mean = 0.032

DPO-Shift with f( ) = 0.5

r(x, yw) r(x, yL)

mean = 0.038

DPO-Shift with f( ) = 0.55

r(x, yw) r(x, yL)

mean = 0.047

DPO-Shift with f( ) = 0.6

r(x, yw) r(x, yL)

mean = 0.029

DPO-Shift with f( ) = 0.65

r(x, yw) r(x, yL)

mean = 0.074

DPO-Shift with f( ) = 0.7

r(x, yw) r(x, yL)

mean = 0.089

DPO-Shift with f( ) = 0.75

r(x, yw) r(x, yL)

mean = 0.108

DPO-Shift with f( ) = 0.8

r(x, yw) r(x, yL)

mean = 0.129

DPO-Shift with f( ) = 0.85

r(x, yw) r(x, yL)

mean = 0.156

DPO-Shift with f( ) = 0.9

r(x, yw) r(x, yL)

mean = 0.187

DPO-Shift with f( ) = 0.95

r(x, yw) r(x, yL)

mean = 0.221

DPO

- Figure 9: Distribution for reward margin and its mean on test set split of Capybara for Llama 3-8B trained on Capybara, where DPO-Shift uses fixed strategy. The ranges of the y-axis of all subfigures are the same.

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

DPO-Shift with f( )=0.65

DPO-Shift with f( )=0.7

DPO-Shift with f( )=0.75

DPO-Shift with f( )=0.8

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

DPO-Shift with f( )=0.85

DPO-Shift with f( )=0.9

DPO-Shift with f( )=0.95

DPO

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

- Figure 10: Distribution for log πθ(yw|x) and log πθ(yl|x) on test set split of UltraFeedback for Qwen 2-7B trained on UltraFeedback, where DPO-Shift uses fixed strategy. The ranges of the y-axis of all subfigures are the same.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

r(x, yw) r(x, yL)

SFTed model

r(x, yw) r(x, yL)

mean = 0.060

DPO-Shift with f( ) = 0.5

r(x, yw) r(x, yL)

mean = 0.075

DPO-Shift with f( ) = 0.55

r(x, yw) r(x, yL)

mean = 0.097

DPO-Shift with f( ) = 0.6

r(x, yw) r(x, yL)

mean = 0.124

DPO-Shift with f( ) = 0.65

r(x, yw) r(x, yL)

mean = 0.144

DPO-Shift with f( ) = 0.7

r(x, yw) r(x, yL)

mean = 0.143

DPO-Shift with f( ) = 0.75

r(x, yw) r(x, yL)

mean = 0.173

DPO-Shift with f( ) = 0.8

r(x, yw) r(x, yL)

mean = 0.211

DPO-Shift with f( ) = 0.85

r(x, yw) r(x, yL)

mean = 0.252

DPO-Shift with f( ) = 0.9

r(x, yw) r(x, yL)

mean = 0.343

DPO-Shift with f( ) = 0.95

r(x, yw) r(x, yL)

mean = 0.456

DPO

- Figure 11: Distribution for reward margin and its mean on test set split of UltraFeedback for Qwen

- 2-7B trained on UltraFeedback, where DPO-Shift uses fixed strategy. The ranges of the y-axis of all subfigures are the same.

2000 1500 1000 500 0

2000 1500 1000 500 0

2000 1500 1000 500 0

2000 1500 1000 500 0

DPO-Shift with f( )=0.65

DPO-Shift with f( )=0.7

DPO-Shift with f( )=0.75

DPO-Shift with f( )=0.8

2000 1500 1000 500 0

2000 1500 1000 500 0

2000 1500 1000 500 0

2000 1500 1000 500 0

DPO-Shift with f( )=0.85

DPO-Shift with f( )=0.9

DPO-Shift with f( )=0.95

DPO

2000 1500 1000 500 0

2000 1500 1000 500 0

2000 1500 1000 500 0

2000 1500 1000 500 0

- Figure 12: Distribution for log πθ(yw|x) and log πθ(yl|x) on test set split of Capybara for Qwen

- 2-7B trained on Capybara, where DPO-Shift uses fixed strategy. The ranges of the y-axis of all subfigures are the same.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

r(x, yw) r(x, yL)

SFTed model

r(x, yw) r(x, yL)

mean = 0.008

DPO-Shift with f( ) = 0.5

r(x, yw) r(x, yL)

mean = 0.011

DPO-Shift with f( ) = 0.55

r(x, yw) r(x, yL)

mean = 0.015

DPO-Shift with f( ) = 0.6

r(x, yw) r(x, yL)

mean = 0.023

DPO-Shift with f( ) = 0.65

r(x, yw) r(x, yL)

mean = 0.029

DPO-Shift with f( ) = 0.7

r(x, yw) r(x, yL)

mean = 0.032

DPO-Shift with f( ) = 0.75

r(x, yw) r(x, yL)

mean = 0.041

DPO-Shift with f( ) = 0.8

r(x, yw) r(x, yL)

mean = 0.123

DPO-Shift with f( ) = 0.85

r(x, yw) r(x, yL)

mean = 0.072

DPO-Shift with f( ) = 0.9

r(x, yw) r(x, yL)

mean = 0.096

DPO-Shift with f( ) = 0.95

r(x, yw) r(x, yL)

mean = 0.129

DPO

- Figure 13: Distribution for reward margin and its mean on test set split of Capybara for Qwen 2-7B trained on Capybara, where DPO-Shift uses fixed strategy. The ranges of the y-axis of all subfigures are the same.

#### A.4 Ablation Studies for linear_increase and linear_decrease

DPO-Shift increase min=0.75

DPO-Shift increase min=0.85

DPO-Shift increase min=0.95

SFTed model

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

DPO-Shift decrease min=0.75

DPO-Shift decrease min=0.85

DPO-Shift decrease min=0.95

DPO

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

- Figure 14: Distribution for log πθ(yw|x) and log πθ(yl|x) on test set split of Ultrafeedback for Llama

- 3-8B trained on UltraFeedback, where DPO-Shift uses linear_increase and linear_decrease strategies. The ranges of the y-axis of all subfigures are the same.

r(x, yw) r(x, yL)

SFTed model

r(x, yw) r(x, yL)

mean = 0.554

DPO-Shift increase min=0.75

r(x, yw) r(x, yL)

mean = 0.592

DPO-Shift increase min=0.85

r(x, yw) r(x, yL)

mean = 0.580

DPO-Shift increase min=0.95

r(x, yw) r(x, yL)

mean = 0.517

DPO-Shift decrease min=0.75

r(x, yw) r(x, yL)

mean = 0.533

DPO-Shift decrease min=0.85

r(x, yw) r(x, yL)

mean = 0.575

DPO-Shift decrease min=0.95

r(x, yw) r(x, yL)

mean = 0.582

DPO

- Figure 15: Distribution for reward margin and its mean on test set split of Ultrafeedback for Llama

- 3-8B trained on UltraFeedback, where DPO-Shift uses linear_increase and linear_decrease strategies. The ranges of the y-axis of all subfigures are the same.

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

DPO-Shift decrease min=0.75

DPO-Shift decrease min=0.85

DPO-Shift decrease min=0.95

DPO

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

- Figure 16: Distribution for log πθ(yw|x) and log πθ(yl|x) on test set split of Capybara for Llama 3-8B trained on Capybara, where DPO-Shift uses linear_increase and linear_decrease strategies. The ranges of the y-axis of all subfigures are the same.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

r(x, yw) r(x, yL)

SFTed model

r(x, yw) r(x, yL)

mean = 0.129

DPO-Shift increase min=0.75

r(x, yw) r(x, yL)

mean = 0.159

DPO-Shift increase min=0.85

r(x, yw) r(x, yL)

mean = 0.199

DPO-Shift increase min=0.95

r(x, yw) r(x, yL)

mean = 0.150

DPO-Shift decrease min=0.75

r(x, yw) r(x, yL)

mean = 0.175

DPO-Shift decrease min=0.85

r(x, yw) r(x, yL)

mean = 0.203

DPO-Shift decrease min=0.95

r(x, yw) r(x, yL)

mean = 0.221

DPO

- Figure 17: Distribution for reward margin and its mean on test set split of Capybara for Llama 3-8B trained on Capybara, where DPO-Shift uses linear_increase and linear_decrease strategies. The ranges of the y-axis of all subfigures are the same.

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

DPO-Shift decrease min=0.75

DPO-Shift decrease min=0.85

DPO-Shift decrease min=0.95

DPO

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

- Figure 18: Distribution for log πθ(yw|x) and log πθ(yl|x) on test set split of UltraFeedback for Qwen 2-7B trained on UltraFeedback, where DPO-Shift uses linear_increase and linear_decrease strategies. The ranges of the y-axis of all subfigures are the same.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

r(x, yw) r(x, yL)

SFTed model

r(x, yw) r(x, yL)

mean = 0.209

DPO-Shift increase min=0.75

r(x, yw) r(x, yL)

mean = 0.261

DPO-Shift increase min=0.85

r(x, yw) r(x, yL)

mean = 0.379

DPO-Shift increase min=0.95

r(x, yw) r(x, yL)

mean = 0.304

DPO-Shift decrease min=0.75

r(x, yw) r(x, yL)

mean = 0.359

DPO-Shift decrease min=0.85

r(x, yw) r(x, yL)

mean = 0.426

DPO-Shift decrease min=0.95

r(x, yw) r(x, yL)

mean = 0.456

DPO

- Figure 19: Distribution for reward margin and its mean on test set split of UltraFeedback for Qwen 2-7B trained on UltraFeedback, where DPO-Shift uses linear_increase and linear_decrease strategies. The ranges of the y-axis of all subfigures are the same.

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

DPO-Shift decrease min=0.75

DPO-Shift decrease min=0.85

DPO-Shift decrease min=0.95

DPO

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

1500 1000 500 0

- Figure 20: Distribution for log πθ(yw|x) and log πθ(yl|x) on test set split of Capybara for Qwen

- 2-7B trained on Capybara, where DPO-Shift uses linear_increase and linear_decrease strategies. The ranges of the y-axis of all subfigures are the same.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

r(x, yw) r(x, yL)

SFTed model

r(x, yw) r(x, yL)

mean = 0.045

DPO-Shift increase min=0.75

r(x, yw) r(x, yL)

mean = 0.054

DPO-Shift increase min=0.85

r(x, yw) r(x, yL)

mean = 0.068

DPO-Shift increase min=0.95

r(x, yw) r(x, yL)

mean = 0.057

DPO-Shift decrease min=0.75

r(x, yw) r(x, yL)

mean = 0.065

DPO-Shift decrease min=0.85

r(x, yw) r(x, yL)

mean = 0.072

DPO-Shift decrease min=0.95

r(x, yw) r(x, yL)

mean = 0.129

DPO

- Figure 21: Distribution for reward margin and its mean on test set split of Capybara for Qwen 2-7B trained on Capybara, where DPO-Shift uses linear_increase and linear_decrease strategies. The ranges of the y-axis of all subfigures are the same.

### A.5 Ablation Studies for Fine-grained fixed

DPO-s with f( )=0.96

DPO-s with f( )=0.97

1400 1200 1000 800 600 400 200 0

1400 1200 1000 800 600 400 200 0

DPO-s with f( )=0.98

DPO-s with f( )=0.99

1400 1200 1000 800 600 400 200 0

1400 1200 1000 800 600 400 200 0

- Figure 22: Distribution for log πθ(yw|x) and log πθ(yl|x) on test set split of UltraFeedback for Llama 3-8B trained on UltraFeedback, where DPO-Shift uses fixed strategy. The ranges of the y-axis of all subfigures are the same.

r(x, yw) r(x, yL)

mean = 0.569

DPO-shift with f( ) = 0.96

r(x, yw) r(x, yL)

mean = 0.577

DPO-shift with f( ) = 0.97

r(x, yw) r(x, yL)

mean = 0.603

DPO-shift with f( ) = 0.98

r(x, yw) r(x, yL)

mean = 0.586

DPO-shift with f( ) = 0.99

- Figure 23: Distribution for reward margin and reward accuracy on test set split of UltraFeedback for Llama 3-8B trained on UltraFeedback, where DPO-Shift uses fixed strategy. The ranges of the y-axis of all subfigures are the same.

- A.6 Supplementary Results for Win Rate Experiment

f(λ) strategy Win Lose

SFT 30.70% 69.30% fixed 0.55 49.15% 50.85% fixed 0.75 59.40% 40.60% fixed 0.95 69.60% 31.40% increase_linear 0.95 71.95% 29.05% decrease_linear 0.95 73.30% 26.70%

Table 3: Win rate experiment against DPO using Qwen-2 7B trained on the UltraFeedback dataset and tested with questions from the test split of UltraFeedback. Results for DPO-Shift using fixed, linear_increase, linear_decrease are included.

- B Proof of Theorem 2.1 We consider our modified PO loss function:

πθ(yw|x) πref(yw|x) − f(λ) · β log

πθ(yl|x) πref(yl|x) where f(λ) is a real valued function with f(λ) < 1. We compute its gradient w.r.t θ:

Lλ−DPO(πθ,πref) = −E(x,y

w,yl)∼Dpref σ β log

σ′(u) σ(u) ∇θu

∇θLλ−DPO(πθ,πref) = −E(x,y

w,yl)∼Dpref

πθ(yw|x) πref(yw|x) − f(λ) · β log

πθ(yl|x) πref(yl|x)

u := β log

notice that σ′(x) = σ(x)(1 − σ(x)),1 − σ(x) = σ(−x). Then we proceed with the final gradient

πθ(yl|x) πref(yl|x) − β log

πθ(yw|x) πref(yw|x)

∇θLλ−DPO(πθ,πref) = −E(x,y

w,yl)∼Dpref βσ f(λ) · β log

- (10)

× [∇θ log π (yw|x) − f(λ) · ∇θ log π (yl|x)]]

- (11)

We then simplify it with the following notation

πθ(yl|x) πref(yl|x) − β log

πθ(yw|x) πref(yw|x) c2 = f(λ)c1

c1 := cθ(λ,yw,yl) = βσ f(λ) · β log

then we have

∇θLλ−DPO(πθ,πref) = −E(x,y

w,yl)∼Dpref [c1∇θ log πθ (yw|x) − c2∇θ log πθ (yl|x)]. Then we upgrade θt+1 with the following: θt+1 ← θt + ηE(x,y

(yl|x)]. We first look into

w,yl)∼Dpref [c1∇θ log πθ

#### (yw|x) − c2∇θ log πθ

t

t

(yw|x) then

#### w1(θt) = log πθ

t

#### (yl|x))⊤ (∇θ log πθ

(yw|x)) (12)

#### w1(θt+1) = w1(θt) + η (c1∇θ log πθ

#### (yw|x) − c2∇θ log πθ

t

t

t

#### (yl|x)⊤ ∇θ log πθ

#### (yw|x)||2 − c2∇θ log πθ

= w1(θt) + η c1 ||∇θ log πθ

#### (yw|x)

t

t

t

(13) then

#### g1(t + 1) = η(c1 − c2)∇θ log π (yl|x)⊤ ∇θ log π (yw|x)

We compute ∇θ log π yil|xi ⊤ ∇θ log π yiw|xi for the SFTed Llama 3-8B model. In terms of frequency, 71.4% of them turn out to be positive. Consequently, we can choose a c2 as small as possible to increase the chosen probability.

However, choosing small c2 can cause performance drop. To evaluate the performance of the model, we look into the reward margin:

(yw|x) πref(yw|x) − log

(yl|x) πref(yl|x)

πθ

πθ

ω2(θt) = E 1 log

t

t

> 0

To analyze, we alternate it with its smoothed version:

(yw|x) πref(yw|x) − γ log

(yl|x) πref(yl|x)

πθ

πθ

ω2(θt) = E σ γ log

t

t

,

we abuse notation and use β → +∞ as hyper parameter. Then with first order Taylor’s expansion for (ω′,θ′) and (ω,θ). Remark 1. Given the fact that they are using the same expectation E(x,y

w,yl)∼Dpref [·], we omit it for the sake of simplicity 2. Given the assumption that η is small enough, we can ignore second order.

(yl|x) πref(yl|x) − log

#### (yw|x) πref(yw|x) · (∇θ log πθ

πθ

πθ

ω2(θt+1) = ω2(θt) + η(θt2+1 − θt2)⊤σ log

t

t

(yw|x) − ∇θ log π (yl|x))

t

#### (yl|x))⊤ (∇θ log πθ

= ωt2 + η1 (c1∇θ log πθ

#### (yw|x) − c2∇θ log πθ

#### (yw|x) − ∇θ log πθ

(yl|x))

t

t

t

t

= ωt2 + η1[c1 ||∇θ log πθ

#### (yl|x)||2 − (c1 + c2)∇θ log πθ

#### (yw|x)||2 + c2 ||∇θ log πθ

t

t

#### (yl|x)⊤ ∇θ log πθ

#### (yw|x)] η1 := ησ (log πθ

t

t

(yl|x)) then we have

#### (yw|x) − log πθ

t

t

g2(t + 1) = η1 (c2 − c1)||∇θ log π (yl|x)||2 + (c1 − c2)∇θ log π (yl|x)⊤ ∇θ log π (yw|x)

= η1 (c1 − c2) ∇θ log π (yl|x)⊤ ∇θ log π (yw|x) − ||∇θ log π (yl|x)||2

#### yil|xi ⊤ ∇θ log πθ

We compute ∇θ log πθ

yil|xi |2 for the SFTed Llama

#### yiw|xi − |∇θ log πθ

t

t

t

- 3-8B model. In terms of frequency, 81.7% of them turn out to be positive.

which is consistent with our observation that accuracy drops greatly with small c2 , therefore we need to strike a balance between the accuracy and chosen probability.

