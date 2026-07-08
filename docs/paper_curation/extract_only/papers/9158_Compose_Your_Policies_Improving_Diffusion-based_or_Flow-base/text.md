# arXiv:2510.01068v2[cs.RO]10Mar2026

COMPOSE YOUR POLICIES! IMPROVING DIFFUSIONBASED OR FLOW-BASED ROBOT POLICIES VIA TESTTIME DISTRIBUTION-LEVEL COMPOSITION

Jiahang Cao1,2,3˚ Yize Huang1,4˚ Hanzhong Guo1 Rui Zhang1 Mu Nan1 Weijian Mai1 Jiaxu Wang5 Hao Cheng5 Jingkai Sun1,2 Gang Han2 Wen Zhao2 Qiang Zhang2 Yijie Guo2 Qihao Zheng3 Chunfeng Song3 Xiao Li4 Ping Luo1 Andrew F. Luo1: 1The University of Hong Kong 2Beijing Innovation Center of Humanoid Robotics 3Shanghai AI Lab 4Shanghai Jiaotong University 5The Hong Kong University of Science and Technology

ABSTRACT

Diffusion-based models for robotic control, including vision-language-action (VLA) and vision-action (VA) policies, have demonstrated significant capabilities. Yet their advancement is constrained by the high cost of acquiring large-scale interaction datasets. This work introduces an alternative paradigm for enhancing policy performance without additional model training. Perhaps surprisingly, we demonstrate that the composed policies can exceed the performance of either parent policy. Our contribution is threefold. First, we establish a theoretical foundation showing that the convex composition of distributional scores from multiple diffusion models can yield a superior one-step functional objective compared to any individual score. A Gr¨onwall-type bound is then used to show that this single-step improvement propagates through entire generation trajectories, leading to systemic performance gains. Second, motivated by these results, we propose General Policy Composition (GPC), a training-free method that enhances performance by combining the distributional scores of multiple pre-trained policies via a convex combination and test-time search. GPC is versatile, allowing for the plug-and-play composition of heterogeneous policies, including VA and VLA models, as well as those based on diffusion or flow-matching, irrespective of their input visual modalities. Third, we provide extensive empirical validation. Experiments on Robomimic, PushT, and RoboTwin benchmarks, alongside realworld robotic evaluations, confirm that GPC consistently improves performance and adaptability across a diverse set of tasks. Further analysis of alternative composition operators and weighting strategies offers insights into the mechanisms underlying the success of GPC. These results establish GPC as a simple yet effective method for improving control performance by leveraging existing policies. Our project page is in https://sagecao1125.github.io/GPC-Site/.

1 INTRODUCTION

Diffusion Policies (DPs) (Chi et al., 2023; Ho et al., 2020; Song et al., 2020a) have emerged as a powerful method for policy parameterization in robot learning, enabling the representation of complex, multi-modal action distributions – a key advantage for policies conditioning on high-dimensional inputs like vision and language in domains from manipulation (Ze et al., 2024b; Zhu et al., 2024; Liu et al., 2024a) to navigation (Sridhar et al., 2024; Zhang et al., 2024a). Despite this progress, the advancement of diffusion- and flow-based policies is fundamentally constrained by scaling challenges related to both model capacity and data availability. Performance can plateau due to the intrinsic representational limits of a given model, yet scaling up the model architecture also requires

˚Equal contribution jiahang@connect.hku.hk :Corresponding author aluo@hku.hk

+12%

FlowPolicy Place Burger Fries

[Figure 1]

Robomimic 1.0

[Figure 2]

| |[Figure 3]<br><br>+8%<br><br>|
|---|---|
| | |
| | |
| | |
| | |

𝜋

[Figure 4]

GPC (ours)

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Failure

0.9

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Success

[Figure 17]

| |[Figure 18]<br><br>[Figure 19]<br><br>DP + DP3<br><br>RDT + DP<br><br>RDT + DP3<br><br>Put Object Cabinet<br><br>+11%<br><br>|
|---|---|
| | |
| | |
| | |

0.8

[Figure 20]

[Figure 21]

Distribution-level Composition

Pre-trained Distribution of Policy A RoboTwin

Can Square Lift

[Figure 22]

RDT

[Figure 23]

[Figure 24]

[Figure 25]

DP3

[Figure 26]

[Figure 27]

0.8

[Figure 28]

[Figure 29]

GPC (ours)

[Figure 30]

[Figure 31]

[Figure 32]

Hanging Mug

[Figure 33]

[Figure 34]

Failure

Composed Distribution

+15%

0.6

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Plausible Area Implausible Area

[Figure 39]

[Figure 40]

0.4

[Figure 41]

Success Cases Failure Cases

Pre-trained Distribution of Policy B

Place Burger Put Object Stack Bowls

Test-time Weight Searching 

(a) (b) (c)

- Figure 1: Illustration of General Policy Composition. (a) Distributions from pre-trained stateof-the-art diffusion- or flow-based policies can be composed to construct a stronger policy without additional training, with a test-time search over composition weights picking the best parent-policy mix; score composition corresponds to the product of probabilistic density functions (PDFs), steering sampling toward consensus regions. (b) GPC can yield consistent gains across a diverse set of tasks. (c) We find the optimal weight when composing two models can vary depending on the task.

the collection of costly interaction datasets to fully capture the potential performance benefit (Black et al., 2024). Conventional post-training strategies offer limited solutions; supervised fine-tuning requires expensive data collection (Ouyang et al., 2022), while reinforcement learning introduces the complexity of reward engineering and extensive online interaction (Hu et al., 2025).

To overcome these limitations, this work introduces an alternative paradigm: creating stronger policies by composing existing, pre-trained models. While prior work has explored static model composition (Du & Kaelbling, 2024; Wang et al., 2024c), we find that the optimal weighting is not universal but is instead highly task-dependent, even for a fixed set of parent policies. Drawing inspiration from compositional generative modeling, we first establish a theoretical foundation showing that a convex combination of distributional scores can yield a provably superior objective for policy improvement. This principle underpins our proposed method of General Policy Composition (GPC, Fig. 1). GPC is a training-free framework that, at inference time, combines the distributional scores of multiple pre-trained policies via convex combination and test-time search. This approach flexibly integrates heterogeneous models – spanning diffusion- and flow-based architectures, VA and VLA modalities, and diverse sensory inputs – to form a more capable policy, all without modifying the base models. Crucially, we demonstrate that the resulting composed policy can exceed the performance of any of its individual parent policies.

We validate GPC through extensive experiments in both simulation and real-world environments, demonstrating consistent outperformance against single-policy baselines. Our analysis extends to alternative composition operators (e.g., logical AND/OR) and various weighting configurations, offering broader insights into why and when composition is effective. Our contributions are summarized as follows: (i) We establish a theoretical foundation for robot policy composition, proving that the convex combination of distributional scores can yield an improved functional objective and that this advantage propagates to the system level. (ii) We propose General Policy Composition (GPC), a flexible, training-free framework that combines pre-trained policies across different modalities and architectures into a more expressive policy. (iii) We conduct extensive evaluations in simulation and the real world, demonstrating the consistent performance gains of GPC while analyzing key design choices to guide future research in policy composition.

2 RELATED WORK

Composable Generative Models. Composability refers to the ability to combine multiple components or distributions into a unified representation while preserving the properties of the individual elements. (i) Visual Generation: Energy-based models (EBMs) (Hinton, 2002; Du & Mordatch, 2019; Grathwohl et al., 2020) support compositionality by summing energies, allowing factor-level combinations. Du et al. (2020) unified perspectives of compositionality for visual generation. Liu et al. (2021) further improved EBMs for scene generation by factorizing relational structures. Skreta et al. (2024) introduced the superposition of diffusion models by using itoˆ estimation. (ii) Language

Success

Policies with different visual modalities

[Figure 42]

Diffusion-based Policies

Lower Error

πDP

Denoise

[Figure 43]

- Score A
- Score B

DP DP3 MDT

###### VA

- DP with Modality A

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

- DP with Modality B

Composed Score

…

Action

𝑓

[Figure 48]

VLA Florence DP

RDT

Test-time Success

πDP

Execution

𝑓

GPC

Denoise

Flow-based Policies

[Figure 49]

[Figure 50]

Composed Score …

Action

Flow Policy

VA

[Figure 51]

[Figure 52]

πRDT

Success

[Figure 53]

𝒘𝟏 𝒘𝟐

𝜋0

Florence FP

VLA

Score C

iteration

RDT with Modality B

{ , 𝒘∗}

…

Lower Error

Pre-trained State-of-the-art Policies Pool

Policies with different backbones

𝑓 :

Test-time Searching for 𝒘∗

- Figure 2: Overview of our proposed General Policy Composition. Combining distributional scores from pre-trained diffusion-based or flow-based policies on different conditions (e.g., visual modalities and network backbones), GPC can generate expressive and adaptable action trajectories through convex score combination without additional training.

Generation: Du et al. (2023b) combined outputs using multi-agent debate for robust language generation. Lifshitz et al. (2025) proposed multi-agent verification at test-time for improvement.

Diffusion Models in Robot Learning. Due to their flexibility and representational power, diffusion models (Ho et al., 2020; Song et al., 2020a; Nichol & Dhariwal, 2021) offer a novel way to represent policies. The concept of Diffusion Policy (Chi et al., 2023) was first proposed to model action spaces using diffusion, significantly enhancing expressiveness. Since then, numerous advancements have been made: multimodal DP such as MDT (Reuss et al., 2024), trajectory extraction approaches like AWE (Shi et al., 2023). DP3 (Ze et al., 2024b) utilizes point cloud representations to achieve state-of-the-art performance, and VLA models, e.g., Octo (Team et al., 2024), π0 (Black et al., 2024) and RDT (Liu et al., 2024a). In this work, our GPC can be adopted to various general diffusion-based (e.g., DP, DP3, MDT, and RDT) or flow-based policies (e.g., flow policy and π0), demonstrating great flexibility (Detailed in App. G).

Compositional Diffusion Models in Robotics. Recent work has explored the use of compositional diffusion models in robotics. Janner et al. (2022) applied compositionality to diffusion-based planning. Yang et al. (2023b) tackled continuous constraint satisfaction in robotic planning, and Luo et al. (2024) improved motion planning by learning potential fields. In addition, Policy Composition (PoCo) (Wang et al., 2024c) does constraint-based, task-based, and input modality-based composition; however, it does not explore the weights between policies. Moreover, it has not been validated in widely adopted simulation environments and provides limited analysis of the underlying composition mechanisms. In contrast, our proposed GPC framework offers broader generality by enabling composition across both VA and VLA models, regardless of input visual modality, and we also deliver deeper insights into the task-dependent weight searching for policy composition.

- 3 PRELIMINARIES

Diffusion models (Sohl-Dickstein et al., 2015) are based on a generative process that iteratively denoises a random noise distribution to generate samples. The following equation describes the update rule for the diffusion process based on Langevin dynamics (Song et al., 2020b):

τt´1 “ αt τt ` βt sθpτt,tq ` γt η, η „ N`

˘

0,σt2I

, (1)

where sθpτt,tq denotes the learned score function, αt,βt,γt are coefficients determined by the noise schedule and the choice of solver. Different sampling methods, such as DDPM (Ho et al., 2020), DDIM (Song et al., 2020a), or ODE/SDE-based solvers (Song et al., 2020b), can be recovered by specifying these coefficients accordingly.

Closely related to diffusion models are energy-based models (Hinton, 2002; Du & Mordatch, 2019), which define probability distributions through learnable energy functions. The connection arises because the gradient of the energy function in EBMs plays a role analogous to the score function in diffusion models. Further progress is made with compositional EBMs (Du et al., 2020), where multiple energy functions can be combined by summing their contributions.

- 4 THEORETICAL ANALYSIS OF CONVEX SCORE COMPOSITION

We first provide a mathematical justification for why convex score composition can improve policy performance. Our analysis shows that (i) at the functional level, convex combinations of scores from pre-trained policies can yield lower score error, and (ii) at the system level, sampling error is bounded by score error through the stability of the sampling dynamics. These results establish convex score composition as a principled foundation for policy improvement, which directly motivates our proposed General Policy Composition framework (in Sec. 5).

- 4.1 FUNCTIONAL-LEVEL IMPROVEMENT

We begin with the question of whether combining score estimators can yield better approximations to the true score s˚. The following result shows that there exists a convex combination of two estimators whose error is no greater than that of the better individual estimator, and strictly smaller unless their errors are perfectly aligned.

|Proposition 4.1 (Single-step improvement via convex combination). Let two score estimators be ε1 “ s˚`b1`η1 and ε2 “ s˚`b2`η2, with deterministic biases bi and random zero-mean noise ηi that plays the role of the diffusion component in the time-reversed stochastic dynamics (e.g., a reverse-time ODE). For any convex weight w P r0,1s, define εpwq “ wε1 ` p1 ´ wqε2. Then the mean-squared error Qpwq “ E}εpwq´s˚}2 is a convex quadratic in w. Its minimizer w‹ satisfies<br><br>Qpw‹q ď mintQp0q,Qp1qu, with strict inequality in most non-trivial cases.|
|---|

- See proof in App. B. Intuitively, each estimator deviates from the true score in a different way. A convex combination can cancel out these errors, achieving a better score estimator. Unless the two models make identical errors, the true score s˚ lies closer to some interior point, ensuring that a weighted average achieves smaller error. This establishes that convex score composition can reduce estimation error at each step.

4.2 SYSTEM-LEVEL STABILITY

While Prop. 4.1 shows improvement at the functional level, it remains to understand how score errors propagate into trajectory sampling. The following proposition establishes a stability guarantee: the terminal error is controlled by the cumulative score error.

|Proposition 4.2 (Score-to-sample stability). Let x˚ptq denote the oracle trajectory derived from the true score s˚, and xsˆptq denote the approximate trajectory derived from an estimator sˆ, both starting from the same initial condition. They satisfy<br><br>x9˚ptq “ Fpt,x˚ptq,s˚pt,x˚ptqqq, x9sˆptq “ Fpt,xsˆptq,sˆpt,xsˆptqqq, where F represents the underlying dynamics that map the score into the state update. Suppose F is Lipschitz in px,sq with constants Lxptq,Lsptq, and sˆ is Lipschitz in x with constant Λˆptq. Assume the score error admits a uniform bound κptq (i.e., }sˆ ´ s˚} ď κptq). Define L˜ptq “ Lxptq ` LsptqΛˆptq. Then for all T ą 0,<br><br>E}xsˆpTq ´ x˚pTq} ď<br><br>˜ż T<br><br>0<br><br>e2ştT<br><br>L˜pτqdτ Lsptq2 dt¸1{2˜ż T<br><br>0<br><br>κptq2 dt¸1{2.|
|---|

- See proof in App. C. This result shows that the sampling dynamics are stable: the terminal error grows at most exponentially with the Lipschitz constants, and is directly bounded by the integrated score error. Thus, reducing score error at each step translates to reducing the overall trajectory error.

- 4.3 IMPLICATIONS FOR POLICY COMPOSITION Combining Prop. 4.1 and Prop. 4.2 yields a direct implication for composed policies.

|Corollary 4.1 (Convex score combination tightens the sampling error bound). Let Bpsˆq denote the upper bound on the expected sampling error derived in Proposition 4.2 (specifically Eq. 21). If a convex combination scomp “ ws1 ` p1 ´ wqs2 satisfies<br><br>ż T<br><br>0<br><br>E}scomp ´ s˚}2dt ă min<br><br>i<br><br>ż T<br><br>0<br><br>E}si ´ s˚}2dt, then the corresponding theoretical error bound is strictly reduced:<br><br>Bpscompq ă min<br><br>i<br><br>Bpsiq.|
|---|

- See proof in App. D. Once functional-level improvement is established by obtaining an optimal w˚ (Prop. 4.1), stability ensures this advantage propagates along the trajectory (Prop. 4.2), making convex score composition provably superior to relying on individual scores.

This theoretical analysis provides a clear justification for convex score composition: it can improve accuracy at each functional step and propagate this advantage through stable sampling dynamics, leading to system-level gains. These results directly motivate GPC, which leverages convex score combination to build stronger policies from pre-trained components. While the theory guarantees the existence of optimal weights, finding them analytically is intractable; hence, in practice we employ test-time searching to identify effective weighting strategies, as explored in Sec. 6.

- 5 OUR METHOD: GENERAL POLICY COMPOSITION

Building on the mathematical foundation in Sec. 4, we now present our method, General Policy Composition, as illustrated in Fig. 2. The key idea is to leverage convex score composition to combine multiple pre-trained policies into a stronger and more expressive one. We first revisit the mathematical formulation of compositional diffusion models in Sec. 5.1, which provides a basis for composing policies conditioned on different factors. We then introduce our method in Sec. 5.2, where GPC convexly combines the scores of diffusion or flow-based policies across modalities, architectures, or VA/VLA settings. Finally, we extend this framework in Sec. 5.3 to include alternative composition operators, offering a broader view of policy composition beyond convex averaging.

- 5.1 COMPOSITIONAL DIFFUSION MODELS

The key idea of the compositional diffusion model (CDM) is to model the distribution of a trajectory τ conditioned on multiple concepts ci, similar to the compositional EBMs. Mathematically under an independence assumption, we can express the joint probability of the trajectory τ based on the set of concepts tc1,...,cnu in Eq. 2, and further reformulate the conditional terms by parameterizing ppci|τq9´

¯α, as follows:

ppτ|ciq ppτq

źn

ppci|τq, (2)

ppτ|c1,...,cnq 9 ppτ,c1,...,cnq “ ppτq

i“1

ˆ

˙α , with ppci|τq9ˆ

˙α , (3)

źn

ppτ|ciq ppτq

ppτ|ciq ppτq

9 ppτq

i“1

where ppci|τq can be interpreted as an implicit classifier (Ho & Salimans, 2022) and α serves as a weighting factor that modulates the influence of each concept on the overall trajectory distribution.

Then, the score function of the composed distribution can be derived directly from Eq. 3:

ÿn

`∇τ log ppτ|ciq ´ ∇τ log ppτq˘

. (4)

∇τ log ppτ|c1,...,cnq “ ∇τ log ppτq `

α

i“1

Using the relationship between the score function of the distribution and noise (Bao et al., 2022), i.e., ϵθpτt,tq “ ´στ∇τ log ppτq, we can express the update rule for CDM with the ϵ parameterization :

ÿn

`

##### ϵθpτt,t,ciq ´ ϵθpτt,tq˘

, (5)

ϵˆpτt,t,cq “ ϵθpτt,tq `

wi

i“1

###### DPimg DPpcd GPC (ours)

[Figure 54]

[Figure 55]

###### Alg. 1: General Policy Composition Sampling

[Figure 56]

Input: Pre-trained policies π1, π2, weights w1, w2 (i.e., 1 ´ w1), policies’ conditions c1, c2

Failure

Success Success

- 1: for w1 “ 0.0, 0.1, . . . , 0.9, 1.0 : // test-time searching
- 2: Initialize noise trajectory τN „ Np0, Iq per action
- 3: for t “ N, . . . , 1 : // denoising steps
- 4: s1 Ð π1pτt, t, c1q
- 5: s2 Ð π2pτt, t, c2q # score estimation
- 6: sˆcomp Ð w1 ˚ s1 ` w2 ˚ s2 # score composition
- 7: τt´1 Ð αt τt ` βt sˆcomp ` γt
- 8: Return: Action trajectory τ0
- 9: Evaluate SR and store in reward pools Rpw1q Return: Optimal weights w1˚ Ð arg maxw1 Rpw1q

[Figure 57]

[Figure 58]

[Figure 59]

Success

Success

Failure

[Figure 60]

[Figure 61]

[Figure 62]

Success Success Better 

- Figure 3: Visualization results of different diffusion policies and the composed policy with GPC. Our proposed GPC can be successful even when one part of the DP fails, and shows better performance when both parts of the DP work.

Algorithm 1: GPC Sampling. Policies are combined via test-time score composition into a stronger policy.

where ϵθpτt,t,ciq{ϵθpτt,tq represents the noise estimation at time step t for trajectory τt conditioned on the individual concept ci or without condition. The weights wi modulate the influence of each concept on the overall noise estimate. This formulation represents a generalization of the classifierfree guidance (CFG) (Ho & Salimans, 2022) technique commonly used in generative models.

- 5.2 GENERAL POLICY COMPOSITION

Based on the previous foundation, we can now apply the CDM to diffusion policy for robotic tasks. The joint probability distribution of the trajectory conditioned on different modes ci (e.g., different visual modality input or different model architecture) can be expressed as follows:

ppτ|c1,...,cnq 9 ppτ|c1qppτ|c2q¨¨¨ppτ|cnq. (6) While one could in principle apply CFG sampling as in Eq. 5, our theoretical analysis in Sec. 4 shows that convex combinations of scores can provide a better functional objective and propagate stability through sampling dynamics. Motivated by this result, we construct our compositional policy by directly combining the score functions from multiple conditional diffusion policies via convex combination. This formulation not only inherits the stability guarantees established in theory but also enables flexible integration of diverse conditional information.

Formally, let sˆcomppτt,t,cq denotes the composed score, the update rule of GPC is defined as:

sˆcomppτt,t,cq “

ÿn

i“1

wisθpτt,t,ciq, with

ÿn

i“1

wi “ 1, (7)

where sθpτt,t,ciq denotes the score estimate conditioned on concepts ci (e.g., visual modality or policy architecture), and wi represents the weight of convex combination assigned to each concept, ensuring a balanced contribution from all source distributions in the final trajectory estimate.

This convex combination ensures that the composite score remains within the feasible convex hull of individual policies, preventing divergence toward extreme or unstable behaviors. Intuitively, the GPC formulation balances information from different conditions, yielding a more stable and coherent generative trajectory (e.g., Fig. 3). GPC sampling process is shown in Alg. 1.

- 5.3 GPC WITH SUPERPOSITION

Apart from using the score convex combination, our GPC framework naturally connects to the principle of superposition (Skreta et al., 2024), which encompasses: (i) Logical OR corresponds to sampling from a mixture of distributions, which is implemented by weighting with the softmax function at each sampling time: wi1´t “ softmax

`

˘

, where wi1´t determines the relative contribution of each policy’s score in sampling time t, T and l are constants; (ii) Logical AND enforces agreement among policies, corresponding to the intersection of their distributions. This is achieved by solving a linear system to compute the weights such that dlog ptpτ|ciq “ dlog ptpτ|cjq, ensuring consistency across different policies during sampling. In this work, we leverage these formulations to instantiate GPC with logical OR and AND operators as the application in Sec. 6.5.

T log ptpτ|ciq ` ℓ

- Table 1: Experiment results on Robomimic and PushT. The table shows the success rate Ò. Our GPC yields a noticeable average improvement compared with the base policies.

Method Generative Mode Model Type

Robomimic PushT

Can Lift Square PushT Average Base Policies

Diffusion Policy (DP) Diffusion VA 34.50 98.50 2.00 21.75 39.19 Mamba Policy (MP) Diffusion VA 5.00 98.50 3.00 12.06 29.64 Flow Policy (FP) Flow Matching VA 95.00 13.00 77.50 54.25 59.94 Florence Policy-D Diffusion VLA 61.50 97.00 46.50 40.00 61.25 Florence Policy-F Flow Matching VLA 89.00 98.50 88.50 39.38 78.84 π0 Flow Matching VLA 96.50 99.00 92.50 57.69 86.42

Composed Policies via Convex Score Combination DP+MP Diffusion VA & VA 34.50 99.50 8.00 23.63 41.41 +2.22% Florence-Policy-D+DP Diffusion VLA & VA 62.50 100.00 61.50 43.06 66.76 +5.51% Florence-Policy-D+MP Diffusion VLA & VA 63.00 100.00 54.50 40.88 64.60 +3.35% Florence-Policy-F+FP Flow Matching VLA & VA 98.50 98.50 92.50 56.06 86.39 +7.55% π0+FP Flow Matching VLA & VA 99.50 100.00 94.00 62.25 88.94 +2.52%

- Table 2: Experiment results on RoboTwin with 6 diverse bimanual manipulation tasks. GPC achieves an obvious increase with up to 7% improvement on the success rate.

RoboTwin 2.0 Hanging Mug Open Laptop Place Burger Fries Put Object Cabinet Stack Bowls Three Turn Switch Average

Method Model Type

Base Policies

DPimg VA 0.10 0.74 0.49 0.56 0.52 0.38 0.46 DPpcd VA 0.21 0.93 0.72 0.71 0.64 0.71 0.65 RDT VLA 0.13 0.69 0.46 0.32 0.47 0.30 0.40

Composed Policies via Convex Score Combination DPimg + DPpcd VA & VA 0.23 0.93 0.78 0.82 0.71 0.71 0.70 +5% RDT + DPimg VLA & VA 0.18 0.80 0.57 0.59 0.66 0.38 0.53 +7% RDT + DPpcd VLA & VA 0.36 0.94 0.83 0.78 0.73 0.71 0.72 +7%

- 6 EXPERIMENT

We conduct experiments to investigate three key questions: (i) How does GPC perform in simulation and real-world experiments? (ii) How do different weight configurations influence the performance of GPC across various scenarios? (iii) How can the advantages of the composed DP be explained?

- 6.1 EXPERIMENT SETTINGS

Environment Settings. We evaluate on Robomimic (Mandlekar et al., 2022), which includes three manipulation tasks (Can, Lift, Square), PushT (Florence et al., 2021), and RoboTwin (Mu et al., 2025), a suite of dual-arm collaborative tasks where we select representative ones from versions 1.0 and 2.0 (Chen et al., 2025). We further perform four real-world experiments: Place Bottles, Hang Mug, Close Table, and Punch Holes, with the setups in Fig. 6. More details are in App. I.

Baselines. For Robomimic and PushT, we compare against three VA models: DP (Chi et al., 2023), Mamba Policy (MP) (Cao et al., 2025b), Flow Policy (FP, the flow matching version of DP), and three VLA models: Florence-based MDT (Reuss et al., 2024), Florence-based Flow-based MDT, and a revised π0 (Black et al., 2024) built upon Florence VLM (Xiao et al., 2024). For RoboTwin, we adopt two VA models: DP, DP3 (Ze et al., 2024b), and a VLA model RDT (Liu et al., 2024a).

Training and Testing Details. Since GPC is training-free, we directly use pre-trained policies trained based on their original settings (more details are in App. I). Each setting is evaluated over 200 rollouts (100 for RoboTwin), and we report the average success rate (SR). For composition, we employ our GCP and search over weighting coefficients from 0.0 to 1.0 in steps of 0.1.

Optimized weight searching (Optional). Although GPC in principle allows arbitrary convex weights, our framework naturally suggests that the policy with the stronger score should receive a larger weight. This intuition is empirically validated by our findings in Sec. 6.3. Therefore, in practical use, once the relative strengths of the base policies are known, one can directly bias the stronger policy to have weight ą0.5, instead of exhaustively searching over the entire interval r0.1,0.9s. This simple heuristic both respects the underlying score-composition principle and significantly reduces the searching time.

###### Table 3: Experiment results of our method under different composition configurations. These results highlight GPC’s versatility and the importance of weight tuning across policies.

|Scenario Task DPimg DPpcd<br><br>|Weight Scheduling in GPC<br><br>˚|∆|
|---|---|---|
| |0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9| |
|Both Policies Perform Well<br><br>Empty Cup Place 0.42 0.62 Dual Bottles Pick (Hard) 0.49 0.64 Shoe Place 0.37 0.36<br><br>|0.70 0.86 0.84 0.86 0.84 0.84 0.76 0.68 0.61 0.69 0.63 0.71 0.66 0.64 0.65 0.63 0.56 0.58 0.47 0.52 0.56 0.59 0.60 0.59 0.59 0.53 0.41<br><br>|+24% +7% +23%|
|Both Policies Perform Bad<br><br>Dual Shoes Place 0.08 0.23 Pick Apple Messy 0.05 0.26<br><br>|0.19 0.17 0.19 0.20 0.20 0.17 0.16 0.14 0.09 0.25 0.17 0.21 0.15 0.13 0.08 0.08 0.06 0.08<br><br>|+0% +0%|
|Policy A ą Policy B Dual Bottles Pick (Easy) 0.77 0.36<br><br>|0.52 0.64 0.70 0.75 0.82 0.81 0.80 0.85 0.80<br><br>|+8%|
|Policy A ă Policy B Block Hammer Beat 0.00 0.76<br><br>|0.61 0.3 0.18 0.15 0.12 0.07 0.00 0.00 0.00<br><br>|+0%|

˚: The number set t0.1,...,0.9u denotes the weight of DPimg (i.e., w1), corresponding to the noise estimation of GPC as ϵˆM˚ “ w1 ˚ ϵDP

. When w1 equals to 0.0 and 1.0, GPC degenerates into DPpcd and DPimg, respectively.

img ` w2 ˚ ϵDP

Density

pcd

1.0 0.8 0.6 0.4 0.2 0.0

[Figure 63]

| |[Figure 64]<br><br>DPpcd<br><br>|[Figure 65]<br><br>Scattered|
|---|
|
|---|---|
| | |
| | |
| | |
| | |
| | |

###### RoboTwin

###### Robomimic Can

DPimg w1=0.7, w2=0.3

w1=0.5, w2=0.5 Florence Flow w1=0.7, w2=0.3 w1=0.9, w2=0.1

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

| |[Figure 70]<br><br>DPimg<br><br>Mean Value<br><br>|[Figure 71]<br><br>Noisy|
|---|
|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

SampleValue

SR=0.49 SR=0.63 SR=0.64

SR=0.89 SR=0.92 SR=0.90

DPpcd

w1=0.4, w2=0.6 w1=0.3, w2=0.7

FlowP w1=0.2, w2=0.8 w1=0.1, w2=0.9

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

| |[Figure 76]<br><br>GPC<br><br>|[Figure 77]<br><br>Coherent|
|---|
| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

SR=0.64 SR=0.66 SR=0.71

SR=0.95 SR=0.98 SR=0.98

(a) Sample Distribution under Different Modality (b) Sample Distribution under Different Models

Execution Time

Figure 5: Sample distribution through execution time. GPC yields more coherent distributions than baselines.

- Figure 4: Visual analysis of GPC under different compositions. GPC generalizes across (a) modalities and (b) architectures, with appropriate weighting yielding accurate distributions with better SR than individual policies.

- 6.2 MAIN RESULTS: GPC ACROSS ARCHITECTURES AND MODALITIES

Simulation Results. Our results demonstrate that GPC is broadly applicable across both diffusionand flow-based policies, and works under a range of general settings: (i) Same input modality, different architectures. GPC successfully composes policies trained on the same modality but with different network architectures. For instance, in Tab. 1, combining two VA policies (DP+MP) yields a noticeable average improvement of +2.22% over the base policies, while combining a VA and VLA model (Florence-D+DP) achieves a larger increase of +5.51%. (ii) Different modalities, similar architectures. GPC also supports the integration of heterogeneous modalities. In Tab. 2, combining RGB-based and point cloud-based DPs (DPimg+DPpcd) improves the average SR from 0.46/0.65 to 0.70 (+5%), confirming that convex score composition can exploit complementary information even within the same sensory domain. (iii) Different modalities, different architectures. GPC enables flexible integration across modalities and architectures. For example, combining a VLA model with a VA policy (RDT+DPpcd) produces consistent improvements, raising the average SR to +7% compared to DPpcd, and surprisingly, +32% compared to RDT itself.

Real-world Results. In real-world evaluations (in Tab. 5), GPC shows consistently stronger performance than single-policy baselines. For instance, in the Clean Table task it achieves 14/20 successes, surpassing base policies. Similarly, it delivered gains in Place Bottles (13/20 vs. 7/20 and 11/20).

Overall, to answer question one, across diverse tasks and benchmarks, GPC consistently improves performance, with an average increase of up to +7.55% on Robomimic & PushT, +7% on RoboTwin, and +10% in real-world tasks. These results validate that convex score composition provides a robust and general principle for composing policies, regardless of model type or input modality.

- 6.3 INFLUENCE OF WEIGHT CONFIGURATIONS ON GPC PERFORMANCE

To analyze the second question, we evaluate GPC performance across multiple tasks under different weight configurations in Tab. 3. Several findings are summarized:

Finding 1: When both policies have moderate accuracy (e.g., ą30%), GPC often achieves higher accuracy under appropriate weight configurations compared to base policies. For in-

[Figure 78]

- Table 4: Results of GPC with superposition, highlighting performance increase by strong compositional operators.

Table 5: Real-world experiment results, demonstrating the effectiveness of GPC.

|Method|Place Bottles Hang Mug Clean Table Punch Holes|
|---|---|
|DPimg|7/20 5/20 12/20 7/20|
|DPpcd<br><br>|11/20 6/20 7/20 6/20|
|GPC (ours)|13/20 7/20 14/20 9/20<br><br>|

Robomimic PushT

Method

Can Lift Square PushT Average Base Policies

Diffusion Policy (DP) 34.50 98.50 2.00 21.75 39.19 Mamba Policy (MP) 5.00 98.50 3.00 12.06 29.64 Florence Policy-D 61.50 97.00 46.50 40.00 61.25

[Figure 79]

[Figure 80]

GPC

Composed Policies via Logical AND Composition DP+MP 84.00 99.50 48.00 28.18 64.92 +25.73% Florence-Policy-D+DP 90.50 100.00 90.00 36.31 79.20 +17.95% Florence-Policy-D+MP 83.00 100.00 90.00 37.38 77.60 +16.35% Composed Policies via Logical OR Composition DP+MP 82.50 99.50 44.00 29.13 63.78 +24.59% Florence-Policy-D+DP 83.50 100.00 89.00 37.87 77.59 +16.34% Florence-Policy-D+MP 86.50 100.00 86.50 38.44 77.86 +16.61%

[Figure 81]



DPimg

[Figure 82]



[Figure 83]

[Figure 84]

[Figure 85]

DPpcd

Figure 6: Real-world setup and results.

Table 7: Per–action-chunk inference latency in RoboMimic. The overhead of GPC is modest and purely computational.

Table 6: Comparison of training/finetuning time vs. GPC weight search. Teval “ Nrollout ˆ Tper rollout. For RoboMimic, TevalRoboMimic« 200˚5s “ 0.27 hr. For realworld, TevalReal« 20 ˚ 30s “ 0.17 hr.

Method Setting Time cost Training from scratch 1M+ demos, N GPUs 14d (OpenVLA 7B), 30d (RDT 1B) Finetuning 100 demos, 1 GPU ą5h (DP 200M+), ą8h (RDT 1B) GPC (full search) 9 weights (w “ 0.1:0.9) 9Teval („ 2.5hrs) GPC (optimized) 4 weights (wstrong P r0.6,0.9s) 4Teval („ 1hr)

Method Time per chunk (s)

DP 0.09 Florence-Policy-D 0.06 GPC (DP + FP-D) 0.13

stance, in the Empty Cup Place task, DPimg and DPpcd achieve 0.42 and 0.62, respectively, while GPC peaks at 0.86 (+24%) with w1“0.4, surpassing both unimodal DPs. This improvement reflects the composition of diffusion scores capturing a more generalized distribution that reduces the reliance on specific conditions, consistent with the theoretical advantages of compositional models.

#### Finding 2: When one policy has significantly lower accuracy, GPC struggles to surpass the

[Figure 86]

highest accuracy of the better-performing base policies. For example, in the Pick Apple Messy task, DPpcd achieves 0.26 and DPimg achieves only 0.05. GPC peaks at 0.25, falling short of DPpcd. This suggests that low-accuracy scores from weaker modalities can significantly impact the joint distribution, diminishing the overall performance of the composed policy.

#### Finding 3: The improvement of GPC is always maximized when the better-performing base

[Figure 87]

policy holds a larger weight in GPC. For instance, in Dual Bottles Pick (Easy), where DPimg achieves 0.77, GPC reaches 0.85 with w1“0.8, leveraging the stronger DP effectively. This highlights the necessity of assigning higher weights to the better-performing distribution to maximize the effectiveness of GPC, leading the composed policy toward consensus.

These findings highlight GPC’s versatility in leveraging the strengths of different conditions and the importance of appropriately tuning weights to each policy’s performance.

- 6.4 TIME AND COMPUTATIONAL EFFICIENCY OF GPC

GPC introduces two main sources of extra cost compared to using a single base policy: (i) additional weight search evaluations (Snell et al., 2024), and (ii) increased inference cost per action chunk.

Weight search as repeated evaluations. During weight search, we fix a candidate weight configuration, run N rollouts to estimate the SR, and then move to the next configuration. Thus, the total search cost is Tsearch “ Nsearch ˚ Teval, Teval “ Nrollout ˚ Tper rollout. As shown in Tab. 6, sweeping w from 0.1 to 0.9 leads to Nsearch “ 9 and about 2.5 hours. With our optimized searching method, we can restrict the search to wstrong P r0.6,0.9s, reducing to Ncfg “ 4 („ 1 hour). Compared to days of training from scratch or hours of finetuning, this makes GPC a highly efficient alternative.

Inference-time overhead. During rollout, GPC incurs extra compute from querying multiple base policies per action chunk. Tab. 7 reports the measured inference latency in RoboMimic: the GPC composition increases latency from 0.09 s to 0.13 s per chunk, which is modest in practice. This overhead is purely computational and can be further reduced with stronger hardware, optimized inference runtimes, or future engineering improvements (e.g., model compression (Høeg et al., 2024) or distillation (Liu et al., 2026)).

Table 8: GPC can be applied with different actionchunk lengths and infer time steps. Results are in Robomimic.

Table 9: GPC with three base policies on RoboMimic.

Method Can Lift Square

Method Setting Success Rate

Flow Policy 0.95 0.13 0.77 Florence-Policy-F 0.89 0.98 0.88 π0 0.61 0.96 0.92 GPC (best 2-policy) 0.99 1.00 0.94 GPC (FP+FP-F+π0) 1.00 1.00 0.94

DP DDPM, chunk 8, 5 steps 0.50 Florence-Policy-D DDIM, chunk 16, 10 steps 0.53

GPC (DP+FP-D) DDIM, chunk 16, 10 steps 0.66

- 6.5 COMPREHENSIVE ANALYSIS OF GPC EFFECTIVENESS

Analysis on GPC’s Superiority via Visualization. For the third question, Fig. 4 illustrates how GPC improves sample distributions under different settings: (i) GPC under different modalities. In Fig. 4(a), DPimg and DPpcd learn distinct distributions. By adjusting the convex weights, GPC adapts smoothly between them. This demonstrates how GPC leverages knowledge from different modalities to form a more complete distribution. (ii) GPC under different architectures. In Fig. 4(b), both Florence and FlowP learn broadly similar distributions, yet each exhibits localized biases. Through convex composition, GPC expands coverage and enhances precision. This shows that even when base models learn similar representations, GPC refines their alignment and achieves stronger results. Overall, these visualizations confirm that GPC generalizes across modalities and architectures, with appropriate weighting yielding broader and more accurate distributions than individual policies.

Analysis on Execution-Time Sample Distributions. Fig. 5 shows the evolution of execution-time sample distributions. Baselines DPimg and DPpcd produce scattered or noisy patterns, particularly in later stages, indicating instability and higher variance. In contrast, GPC yields coherent and concentrated distributions, ensuring greater stability and mitigating error accumulation during execution.

Experiment Results on GPC with Superposition. We further evaluate GPC under superposition settings. As shown in Tab. 4, composing DP and MP with logical AND boosts the SR to 64.92 (+25.73%), while Florence-D + DP under logical OR reaches 77.59 (+16.34%). These results highlight the potential of superposition to amplify policy performance through stronger composition operators. However, superposition also has clear limitations. It is not directly applicable to flow-based models, and the requirement to recompute weights at every step increases inference cost.

GPC with heterogeneous inference steps and chunk sizes. GPC is also flexible with different inference steps and chunk sizes. For diffusion steps, we simply pick a unified sampler and number of inference steps at test time and run all policies under this common configuration, e.g., the results in RoboTwin with DP+RDT. For action-chunk mismatch (assume HA ě HB), we sample a shared noise trajectory of length HA, apply policy B only on the first HB steps, and take a convex combination of scores on the overlap while keeping the tail from policy A. Results in Tab. 8 confirm that this heterogeneous-chunk composition also yields clear gains.

GPC with multiple base policies. The score-composition rule of GPC directly extends to three or more base policies via a convex combination of their scores. Tab. 9 reports results with three policies on RoboMimic. Three-policy GPC either matches or improves upon the best two-policy configuration, and consistently outperforms the individual base policies. This shows that GPC scales beyond pairwise composition and can effectively leverage diversity across multiple pretrained policies.

- 7 DISCUSSION

Limitations. Our GPC demonstrates clear effectiveness across a wide range of experiments. Despite this strength, certain limitations remain. First, test-time weight search is restricted by a fixed discretization, which may overlook optimal values; future work could explore adaptive or automatic search strategies. Second, we mainly study dual/triple-policy composition, while scaling to more policies increases computation. Addressing this may require feature sharing or compact representations to enable efficient multi-policy integration.

Conclusion. We introduced General Policy Composition, a training-free framework that improves robotic control by combining the distributional scores of pre-trained policies. Our theoretical analysis establishes that convex score composition leads to step-wise and trajectory-level improvements, while our experiments on diverse benchmarks and real-world setups confirm consistent performance gains. GPC is simple, versatile, and widely applicable, providing a foundation for future research in policy composition as a means to enhance performance without additional training resources.

REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Adilzhan Adilkhanov, Amir Yelenov, Assylkhan Seitzhanov, Ayan Mazhitov, Azamat Abdikarimov, Danissa Sandykbayeva, Daryn Kenzhebek, Dinmukhammed Mukashev, Ilyas Umurbekov, Jabrail Chumakov, et al. Survey on vision-language-action models. arXiv preprint arXiv:2502.06851, 2025.

Anurag Ajay, Yilun Du, Abhi Gupta, Joshua Tenenbaum, Tommi Jaakkola, and Pulkit Agrawal. Is conditional generative modeling all you need for decision-making? arXiv preprint arXiv:2211.15657, 2022.

Anurag Ajay, Seungwook Han, Yilun Du, Shuang Li, Abhi Gupta, Tommi Jaakkola, Josh Tenenbaum, Leslie Kaelbling, Akash Srivastava, and Pulkit Agrawal. Compositional foundation models for hierarchical planning. Advances in Neural Information Processing Systems (NeurIPS), 36: 22304–22325, 2023.

Shan An, Ziyu Meng, Chao Tang, Yuning Zhou, Tengyu Liu, Fangqiang Ding, Shufang Zhang, Yao Mu, Ran Song, Wei Zhang, et al. Dexterous manipulation through imitation learning: A survey. arXiv preprint arXiv:2504.03515, 2025.

Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In International Conference on Learning Representations (ICLR), 2022.

Jose Barreiros, Andrew Beaulieu, Aditya Bhat, Rick Cory, Eric Cousineau, Hongkai Dai, ChingHsin Fang, Kunimatsu Hashimoto, Muhammad Zubair Irshad, Masha Itkina, et al. A careful examination of large behavior models for multitask dexterous manipulation. arXiv preprint arXiv:2507.05331, 2025.

Suneel Belkhale, Tianli Ding, Ted Xiao, Pierre Sermanet, Quon Vuong, Jonathan Tompson, Yevgen Chebotar, Debidatta Dwibedi, and Dorsa Sadigh. Rt-h: Action hierarchies using language. arXiv preprint arXiv:2403.01823, 2024.

Richard Bellman. The stability of solutions of linear differential equations. Duke Mathematical Journal, 1943.

Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash Kumar. Roboagent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking. In International Conference on Robotics and Automation (ICRA), pp. 4788–4795. IEEE, 2024.

Johan Bjorck, Fernando Casta˜neda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. zpi 0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alexander Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. Robotics: Science and Systems (RSS), 2023.

Qingwen Bu, Hongyang Li, Li Chen, Jisong Cai, Jia Zeng, Heming Cui, Maoqing Yao, and Yu Qiao. Towards synergistic, generalized, and efficient dual-system for robotic manipulation. arXiv preprint arXiv:2410.08001, 2024.

Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

Remi Cadene, Simon Alibert, Alexander Soare, Quentin Gallouedec, Adil Zouitine, Steven Palma, Pepijn Kooijmans, Michel Aractingi, Mustafa Shukor, Dana Aubakirova, Martino Russi, Francesco Capuano, Caroline Pascal, Jade Choghari, Jess Moss, and Thomas Wolf. Lerobot: State-of-the-art machine learning for real-world robotics in pytorch. https://github.com/ huggingface/lerobot, 2024.

Wenxiao Cai, Iaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. In International Conference on Robotics and Automation (ICRA), pp. 9490–9498. IEEE, 2025.

Jiahang Cao, Qiang Zhang, Hanzhong Guo, Jiaxu Wang, Hao Cheng, and Renjing Xu. Modalitycomposable diffusion policy via inference-time distribution-level composition. arXiv preprint arXiv:2503.12466, 2025a.

Jiahang Cao, Qiang Zhang, Jingkai Sun, Jiaxu Wang, Hao Cheng, Yulin Li, Jun Ma, Yecheng Shao, Wen Zhao, Gang Han, et al. Mamba policy: Towards efficient 3d diffusion policy with hybrid selective state models. International Conference on Intelligent Robots and Systems (IROS), 2025b.

Joao Carvalho, An T Le, Mark Baierl, Dorothea Koert, and Jan Peters. Motion planning diffusion: Learning and planning of robot motions with diffusion models. In International Conference on Intelligent Robots and Systems (IROS), pp. 1916–1923. IEEE, 2023.

Augustin-Louis Cauchy. Sur les formules qui r´esultent de l’emploie du signe et sur¿ ou¡, et sur les moyennes entre plusieurs quantit´es. Cours d’Analyse, 1er Partie, Analyse Algebrique, pp. 373–377, 1821.

Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Xiao Ma, et al. Gr-3 technical report. arXiv preprint arXiv:2507.15493, 2025.

Lili Chen, Shikhar Bahl, and Deepak Pathak. Playfusion: Skill acquisition via diffusion from language-annotated play. In Conference on Robot Learning (CoRL), pp. 2012–2029. PMLR, 2023.

Shang-Fu Chen, Hsiang-Chun Wang, Ming-Hao Hsu, Chun-Mao Lai, and Shao-Hua Sun. Diffusion model-augmented behavioral cloning. In International Conference on Machine Learning (ICML), pp. 7486–7510. PMLR, 2024.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Qiwei Liang, Zixuan Li, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Zaitian Gongye, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu Yin, Sifei Liu, and Xiaolong Wang. Navila: Legged robot vision-language-action model for navigation. arXiv preprint arXiv:2412.04453, 2024a.

Xuxin Cheng, Kexin Shi, Ananye Agarwal, and Deepak Pathak. Extreme parkour with legged robots. In International Conference on Robotics and Automation (ICRA), pp. 11443–11450. IEEE, 2024b.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research (IJRR), pp. 02783649241273668, 2023.

Hao-Tien Lewis Chiang, Zhuo Xu, Zipeng Fu, Mithun George Jacob, Tingnan Zhang, TsangWei Edward Lee, Wenhao Yu, Connor Schenck, David Rendleman, Dhruv Shah, et al. Mobility vla: Multimodal instruction navigation with long-context vlms and topological graphs. arXiv preprint arXiv:2407.07775, 2024.

Armand Comas, Yilun Du, Christian Fernandez Lopez, Sandesh Ghimire, Mario Sznaier, Joshua B Tenenbaum, and Octavia Camps. Inferring relational potentials in interacting systems. In International Conference on Machine Learning (ICML), pp. 6364–6383. PMLR, 2023.

Can Cui, Pengxiang Ding, Wenxuan Song, Shuanghao Bai, Xinyang Tong, Zirui Ge, Runze Suo, Wanqi Zhou, Yang Liu, Bofang Jia, et al. Openhelix: A short survey, empirical analysis, and open-source dual-system vla model for robotic manipulation. arXiv preprint arXiv:2505.03912, 2025.

Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. In International Conference on Machine Learning (ICML), pp. 10041–10071. PMLR, 2024.

Sudeep Dasari, Oier Mees, Sebastian Zhao, Mohan Kumar Srirama, and Sergey Levine. The ingredients for robotic diffusion transformers. In International Conference on Robotics and Automation (ICRA), pp. 15617–15625. IEEE, 2025.

Danny Driess, Jost Tobias Springenberg, Brian Ichter, Lili Yu, Adrian Li-Bell, Karl Pertsch, Allen Z Ren, Homer Walke, Quan Vuong, Lucy Xiaoyang Shi, et al. Knowledge insulating visionlanguage-action models: Train fast, run fast, generalize better. arXiv preprint arXiv:2505.23705, 2025.

Maximilian Du and Shuran Song. Dynaguide: Steering diffusion polices with active dynamic guidance. arXiv preprint arXiv:2506.13922, 2025.

Yilun Du and Leslie Kaelbling. Compositional generative modeling: A single model is not all you need. arXiv preprint arXiv:2402.01103, 2024.

Yilun Du and Igor Mordatch. Implicit generation and modeling with energy based models. Advances in Neural Information Processing Systems (NeurIPS), 32, 2019.

Yilun Du, Shuang Li, and Igor Mordatch. Compositional visual generation with energy based models. Advances in Neural Information Processing Systems (NeurIPS), 33:6637–6647, 2020.

Yilun Du, Shuang Li, Yash Sharma, Josh Tenenbaum, and Igor Mordatch. Unsupervised learning of compositional energy concepts. Advances in Neural Information Processing Systems (NeurIPS), 34:15608–15620, 2021.

Yilun Du, Conor Durkan, Robin Strudel, Joshua B Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc. In International conference on Machine Learning (ICML), pp. 8489–8510. PMLR, 2023a.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In International Conference on Machine Learning (ICML), 2023b.

Shichao Fan, Quantao Yang, Yajie Liu, Kun Wu, Zhengping Che, Qingjie Liu, and Min Wan. Diffusion trajectory-guided policy for long-horizon robot manipulation. arXiv preprint arXiv:2502.10040, 2025.

AI Figure. Helix: A vision-language-action model for generalist humanoid control. Figure AI News, 2024.

Roya Firoozi, Johnathan Tucker, Stephen Tian, Anirudha Majumdar, Jiankai Sun, Weiyu Liu, Yuke Zhu, Shuran Song, Ashish Kapoor, Karol Hausman, et al. Foundation models in robotics: Applications, challenges, and the future. The International Journal of Robotics Research (IJRR), 44 (5):701–739, 2025.

Pete Florence, Corey Lynch, Andy Zeng, Oscar Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, and Jonathan Tompson. Implicit behavioral cloning. Conference on Robot Learning (CoRL), November 2021.

Letian Fu, Huang Huang, Gaurav Datta, Lawrence Yunliang Chen, William Chung-Ho Panitch, Fangchen Liu, Hui Li, and Ken Goldberg. In-context imitation learning via next-token prediction. arXiv preprint arXiv:2408.15980, 2024.

Guido Fubini. Sugli integrali multipli. Rend. Acc. Naz. Lincei, 16:608–614, 1907.

Timur Garipov, Sebastiaan De Peuter, Ge Yang, Vikas Garg, Samuel Kaski, and Tommi Jaakkola. Compositional sculpting of iterative generative processes. Advances in Neural Information Processing Systems (NeurIPS), 36:12665–12702, 2023.

Koffivi Fid`ele Gbagbe, Miguel Altamirano Cabrera, Ali Alabbas, Oussama Alyunes, Artem Lykov, and Dzmitry Tsetserukou. Bi-vla: Vision-language-action model-based system for bimanual robotic dexterous manipulations. In International Conference on Systems, Man, and Cybernetics (SMC), pp. 2864–2869. IEEE, 2024.

Theophile Gervet and Zhou Xiao. Act3d: 3d feature field transformers for multi-task robotic manipulation. In Conference on Robot Learning (CoRL). Proceedings of Machine Learning Research, 2023.

Will Grathwohl, Kuan-Chieh Wang, J¨orn-Henrik Jacobsen, David Duvenaud, and Richard Zemel. Learning the stein discrepancy for training and evaluating energy-based models without sampling. In International Conference on Machine Learning (ICML), pp. 3732–3747. PMLR, 2020.

Will Grathwohl, Kevin Swersky, Milad Hashemi, David Duvenaud, and Chris Maddison. Oops i took a gradient: Scalable sampling for discrete distributions. In International Conference on Machine Learning (ICML), pp. 3831–3841. PMLR, 2021.

Thomas Hakon Gronwall. Note on the derivatives with respect to a parameter of the solutions of a system of differential equations. Annals of Mathematics, 20(4):292–296, 1919.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In Conference on Language Modeling (CoLM), 2024.

Huy Ha, Pete Florence, and Shuran Song. Scaling up and distilling down: Language-guided robot skill acquisition. In Conference on Robot Learning (CoRL), pp. 3766–3777. PMLR, 2023.

Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. Neural Computation, 14(8):1771–1800, 2002.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NeurIPS), 33:6840–6851, 2020.

Sigmund H Høeg, Yilun Du, and Olav Egeland. Streaming diffusion policy: Fast policy synthesis with variable noise diffusion models. arXiv preprint arXiv:2406.04806, 2024.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. International Conference on Learning Representations (ICLR), 1(2):3, 2022.

Jiaheng Hu, Rose Hendrix, Ali Farhadi, Aniruddha Kembhavi, Roberto Mart´ın-Mart´ın, Peter Stone, Kuo-Hao Zeng, and Kiana Ehsani. Flare: Achieving masterful and adaptive robot policies with large-scale reinforcement learning fine-tuning. In International Conference on Robotics and Automation (ICRA), pp. 3617–3624. IEEE, 2025.

Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations, 2024. URL https://arxiv.org/abs/2412.14803.

Haojie Huang, Karl Schmeckpeper, Dian Wang, Ondrej Biza, Yaoyao Qian, Haotian Liu, Mingxi Jia, Robert Platt, and Robin Walters. Imagination policy: Using generative point cloud models for learning manipulation policies. In Conference on Robot Learning (CoRL), pp. 5150–5165. PMLR, 2025a.

Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. In International Conference on Machine Learning (ICML), pp. 20413–20451, 2024.

Siyuan Huang, Liliang Chen, Pengfei Zhou, Shengcong Chen, Zhengkai Jiang, Yue Hu, Peng Gao, Hongsheng Li, Maoqing Yao, and Guanghui Ren. Enerverse: Envisioning embodied future space for robotics manipulation. arXiv preprint arXiv:2501.01895, 2025b.

Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. In Conference on Robot Learning (CoRL), pp. 540–562. PMLR, 2023.

Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. In Conference on Robot Learning (CoRL), pp. 4573–4602. PMLR, 2025c.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0. 5: a vision-languageaction model with open-world generalization, 2025. URL https://arxiv. org/abs/2504.16054, 1(2): 3, 2025.

Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In International Conference on Machine Learning (ICML), pp. 9902–

9915. PMLR, 2022. Johan Ludwig William Valdemar Jensen. Sur les fonctions convexes et les in´egalit´es entre les valeurs moyennes. Acta mathematica, 30(1):175–193, 1906.

Yueru Jia, Jiaming Liu, Sixiang Chen, Chenyang Gu, Zhilue Wang, Longzan Luo, Lily Lee, Pengwei Wang, Zhongyuan Wang, Renrui Zhang, et al. Lift3d foundation policy: Lifting 2d large-scale pretrained models for robust 3d robotic manipulation. arXiv preprint arXiv:2411.18623, 2024.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea open-world dataset and g0 dual-system vla model. arXiv preprint arXiv:2509.00576, 2025.

Joshua Jones, Oier Mees, Carmelo Sferrazza, Kyle Stachowicz, Pieter Abbeel, and Sergey Levine. Beyond sight: Finetuning generalist robot policies with heterogeneous sensors via language grounding. arXiv preprint arXiv:2501.04693, 2025.

Siddharth Karamcheti, Suraj Nair, Annie S Chen, Thomas Kollar, Chelsea Finn, Dorsa Sadigh, and Percy Liang. Language-driven representation learning for robotics. arXiv preprint arXiv:2302.12766, 2023.

Tsung-Wei Ke, Nikolaos Gkanatsios, and Katerina Fragkiadaki. 3d diffuser actor: Policy diffusion with 3d scene representations. In Conference on Robot Learning (CoRL), pp. 1949–1974. PMLR, 2025.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. In Robotics: Science and Systems (RSS), 2024.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, et al. Openvla: An open-source vision-language-action model. In Conference on Robot Learning (CoRL), pp. 2679–2713. PMLR, 2025.

Daphne Koller and Nir Friedman. Probabilistic graphical models: principles and techniques. MIT press, 2009.

S´ebastien Lachapelle, Divyat Mahajan, Ioannis Mitliagkas, and Simon Lacoste-Julien. Additive decoders for latent variables identification and cartesian-product extrapolation. Advances in Neural Information Processing Systems (NeurIPS), 36:25112–25150, 2023.

Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024a.

Shuang Li, Yilun Du, Joshua B Tenenbaum, Antonio Torralba, and Igor Mordatch. Composing

ensembles of pre-trained models via iterative consensus. arXiv preprint arXiv:2210.11522, 2022. Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. arXiv preprint

arXiv:2503.00200, 2025.

Xiang Li, Varun Belagali, Jinghuan Shang, and Michael S Ryoo. Crossway diffusion: Improving diffusion-based visuomotor policy via self-supervised learning. In International Conference on Robotics and Automation (ICRA), pp. 16841–16849. IEEE, 2024b.

Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, Hanbo Zhang, and Huaping Liu. Towards generalist robot policies: What matters in building vision-language-action models. arXiv preprint arXiv:2412.14058, 2024c.

Shalev Lifshitz, Sheila A McIlraith, and Yilun Du. Multi-agent verification: Scaling test-time compute with multiple verifiers. Conference on Language Modeling (CoLM), 2025.

Fanqi Lin, Yingdong Hu, Pingyue Sheng, Chuan Wen, Jiacheng You, and Yang Gao. Data scaling laws in imitation learning for robotic manipulation. In International Conference on Learning Representations (ICLR), 2024a.

Ming-Yi Lin, Ou-Wen Lee, and Chih-Ying Lu. Embodied ai with large language models: A survey and new hri framework. In International Conference on Advanced Robotics and Mechatronics (ICARM), pp. 978–983. IEEE, 2024b.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, Sixiang Chen, Mengzhen Liu, et al. Hybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model. arXiv preprint arXiv:2503.10631, 2025a.

Nan Liu, Shuang Li, Yilun Du, Josh Tenenbaum, and Antonio Torralba. Learning to compose visual relations. Advances in Neural Information Processing Systems (NeurIPS), 34:23166–23178, 2021.

Nan Liu, Yilun Du, Shuang Li, Joshua B Tenenbaum, and Antonio Torralba. Unsupervised compositional concepts discovery with text-to-image generative models. In International Conference on Computer Vision (ICCV), pp. 2085–2095, 2023.

Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022.

Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024a.

Songming Liu, Bangguo Li, Kai Ma, Lingxuan Wu, Hengkai Tan, Xiao Ouyang, Hang Su, and Jun Zhu. Rdt2: Exploring the scaling limit of umi data towards zero-shot cross-embodiment generalization. arXiv preprint arXiv:2602.03310, 2026.

Weiyu Liu, Tucker Hermans, Sonia Chernova, and Chris Paxton. Structdiffusion: Object-centric diffusion for semantic rearrangement of novel objects. In CoRL Workshop on Language and Robotics, 2022.

Yang Liu, Weixing Chen, Yongjie Bai, Xiaodan Liang, Guanbin Li, Wen Gao, and Liang Lin. Aligning cyber space with physical world: A comprehensive survey on embodied ai. IEEE/ASME Transactions on Mechatronics (TMECH), 2025b.

Yijun Liu, Yuwei Liu, Yuan Meng, Jieheng Zhang, Yuwei Zhou, Ye Li, Jiacheng Jiang, Kangye Ji, Shijia Ge, Zhi Wang, et al. Spatial policy: Guiding visuomotor robotic manipulation with spatial-aware modeling and reasoning. arXiv preprint arXiv:2508.15874, 2025c.

Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Maximilian Du, and Chelsea Finn. Bidirectional decoding: Improving action chunking via closed-loop resampling. arXiv e-prints, pp. arXiv–2408, 2024b.

Yiyang Lu, Yufeng Tian, Zhecheng Yuan, Xianbang Wang, Pu Hua, Zhengrong Xue, and Huazhe Xu. H3dp: Triply-hierarchical diffusion policy for visuomotor learning. arXiv preprint arXiv:2505.07819, 2025.

Yunhao Luo, Chen Sun, Joshua B Tenenbaum, and Yilun Du. Potential based diffusion motion planning. In International Conference on Machine Learning (ICML), 2024.

Xiao Ma, Sumit Patidar, Iain Haughton, and Stephen James. Hierarchical diffusion policy for kinematics-aware multi-task robotic manipulation. In Computer Vision and Pattern Recognition (CVPR), pp. 18081–18090, 2024a.

Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. A survey on visionlanguage-action models for embodied ai. arXiv preprint arXiv:2405.14093, 2024b.

Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li FeiFei, Silvio Savarese, Yuke Zhu, and Roberto Mart´ın-Mart´ın. What matters in learning from offline human demonstrations for robot manipulation. In Conference on Robot Learning (CoRL), pp. 1678–1690. PMLR, 2022.

David McAllister, Songwei Ge, Brent Yi, Chung Min Kim, Ethan Weber, Hongsuk Choi, Haiwen Feng, and Angjoo Kanazawa. Flow matching policy gradients. arXiv preprint arXiv:2507.21053, 2025.

Utkarsh Aashu Mishra, Shangjie Xue, Yongxin Chen, and Danfei Xu. Generative skill chaining: Long-horizon skill planning with diffusion models. In Conference on Robot Learning (CoRL), pp. 2905–2925. PMLR, 2023.

Eleonora Misino, Giuseppe Marra, and Emanuele Sansone. Vael: Bridging variational autoencoders and probabilistic logic programming. Advances in Neural Information Processing Systems (NeurIPS), 35:4667–4679, 2022.

Yao Mu, Tianxing Chen, Zanxin Chen, Shijia Peng, Zhiqian Lan, Zeyu Gao, Zhixuan Liang, Qiaojun Yu, Yude Zou, Mingkun Xu, et al. Robotwin: Dual-arm robot benchmark with generative digital twins. In Computer Vision and Pattern Recognition Conference (CVPR), pp. 27649–27660, 2025.

Kevin P Murphy. Probabilistic machine learning: an introduction. MIT press, 2022. Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models.

In International Conference on Machine Learning (ICML), pp. 8162–8171. PMLR, 2021.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems (NeurIPS), 35:27730–27744, 2022.

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In International Conference on Robotics and Automation (ICRA), pp. 6892–6903. IEEE, 2024.

Daojie Peng, Jiahang Cao, Qiang Zhang, and Jun Ma. Lovon: Legged open-vocabulary object navigator. arXiv preprint arXiv:2507.06747, 2025.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

Aaditya Prasad, Kevin Lin, Jimmy Wu, Linqi Zhou, and Jeannette Bohg. Consistency policy: Accelerated visuomotor policies via consistency distillation. In Robotics: Science and Systems (RSS), 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), pp. 8748–8763. PmLR, 2021.

Allen Z Ren, Justin Lidard, Lars L Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion policy policy optimization. arXiv preprint arXiv:2409.00588, 2024.

Moritz Reuss, Omer¨ Erdin¸c Ya˘gmurlu, Fabian Wenzel, and Rudolf Lioutikov. Multimodal diffusion transformer: Learning versatile behavior from multimodal goals. Robotics: Science and Systems (RSS), 2024.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Ranjan Sapkota, Yang Cao, Konstantinos I Roumeliotis, and Manoj Karkee. Vision-language-action

models: Concepts, progress, applications and challenges. arXiv preprint arXiv:2505.04769, 2025. Vaibhav Saxena, Yotto Koga, and Danfei Xu. Constrained-context conditional diffusion models for

imitation learning. arXiv preprint arXiv:2311.01419, 2023. Atharva Sehgal, Arya Grayeli, Jennifer J Sun, and Swarat Chaudhuri. Neurosymbolic grounding for compositional world models. arXiv preprint arXiv:2310.12690, 2023.

Rui Shao, Wei Li, Lingsen Zhang, Renshan Zhang, Zhiyang Liu, Ran Chen, and Liqiang Nie. Large vlm-based vision-language-action models for robotic manipulation: A survey. arXiv preprint arXiv:2508.13073, 2025.

SP Sharan, Ruihan Zhao, Zhangyang Wang, Sandeep P Chinchali, et al. Plan diffuser: Grounding llm planners with diffusion models for robotic manipulation. In Bridging the Gap between Cognitive Science and Robot Learning in the Real World: Progresses and New Directions, 2024.

Lucy Xiaoyang Shi, Archit Sharma, Tony Z Zhao, and Chelsea Finn. Waypoint-based imitation learning for robotic manipulation. In Conference on Robot Learning (CoRL), pp. 2195–2209. PMLR, 2023.

Marta Skreta, Lazar Atanackovic, Joey Bose, Alexander Tong, and Kirill Neklyudov. The superposition of diffusion models using the itˆo density estimator. In International Conference on Learning Representations (ICLR), 2024.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning (ICML), pp. 2256–2265. pmlr, 2015.

Houshang H Sohrab. Basic real analysis, volume 231. Springer, 2003. Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Interna-

tional Conference on Learning Representations (ICLR), 2020a.

Mingchen Song, Xiang Deng, Zhiling Zhou, Jie Wei, Weili Guan, and Liqiang Nie. A survey on diffusion policy for robotic manipulation: Taxonomy, analysis, and future directions. Authorea Preprints, 2025.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems (NeurIPS), 32, 2019.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020b.

Ajay Sridhar, Dhruv Shah, Catherine Glossop, and Sergey Levine. Nomad: Goal masked diffusion policies for navigation and exploration. In IEEE International Conference on Robotics and Automation (ICRA), pp. 63–70. IEEE, 2024.

Jocelin Su, Nan Liu, Yanbo Wang, Joshua B Tenenbaum, and Yilun Du. Compositional image decomposition with diffusion models. In International Conference on Machine Learning (ICML), pp. 46823–46842. PMLR, 2024.

Zhanyi Sun and Shuran Song. Latent policy barrier: Learning robust visuomotor policies by staying in-distribution. arXiv preprint arXiv:2508.05941, 2025.

Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

Brian S Thomson, Judith B Bruckner, and Andrew M Bruckner. Elementary real analysis, volume 1.

ClassicalRealAnalysis. com, 2008. Leonida Tonelli. Sull’integrazione per parti. Rend. Acc. Naz. Lincei, 5(18):246–253, 1909. Julen Urain, Ajay Mandlekar, Yilun Du, Mahi Shafiullah, Danfei Xu, Katerina Fragkiadaki, Georgia

Chalvatzaki, and Jan Peters. Deep generative models in robotics: A survey on learning from multimodal demonstrations. arXiv preprint arXiv:2408.04380, 2024.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems (NeurIPS), 30, 2017.

Robert Verkuil, Ori Kabeli, Yilun Du, Basile IM Wicky, Lukas F Milles, Justas Dauparas, David Baker, Sergey Ovchinnikov, Tom Sercu, and Alexander Rives. Language models generalize beyond natural proteins. BioRxiv, pp. 2022–12, 2022.

Vitalis Vosylius, Younggyo Seo, Jafar Uruc¸, and Stephen James. Render and diffuse: Aligning image and action spaces for diffusion-based behaviour cloning. arXiv preprint arXiv:2405.18196, 2024.

Andrew Wagenmaker, Mitsuhiko Nakamoto, Yunchu Zhang, Seohong Park, Waleed Yagoub, Anusha Nagabandi, Abhishek Gupta, and Sergey Levine. Steering your diffusion policy with latent space reinforcement learning. arXiv preprint arXiv:2506.15799, 2025.

Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe HansenEstruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning (CoRL), pp. 1723–1736. PMLR, 2023.

Dian Wang, Stephen Hart, David Surovik, Tarik Kelestemur, Haojie Huang, Haibo Zhao, Mark Yeatman, Jiuguang Wang, Robin Walters, and Robert Platt. Equivariant diffusion policy. arXiv preprint arXiv:2407.01812, 2024a.

Lirui Wang. Robot Fleet Learning From Heterogeneous Data. PhD thesis, Massachusetts Institute of Technology, 2025.

Lirui Wang, Xinlei Chen, Jialiang Zhao, and Kaiming He. Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers. Advances in Neural Information Processing Systems (NeurIPS), 37:124420–124450, 2024b.

Lirui Wang, Jialiang Zhao, Yilun Du, Edward H Adelson, and Russ Tedrake. Poco: Policy composition from and for heterogeneous robot learning. Robotics: Science and Systems (RSS), 2024c.

Qianhao Wang, Yinqian Sun, Enmeng Lu, Qian Zhang, and Yi Zeng. Mtdp: Modulated transformer diffusion policy model. arXiv e-prints, pp. arXiv–2502, 2025a.

Shaoan Wang, Jiazhao Zhang, Minghan Li, Jiahang Liu, Anqi Li, Kui Wu, Fangwei Zhong, Junzhi Yu, Zhizheng Zhang, and He Wang. Trackvla: Embodied visual tracking in the wild. arXiv preprint arXiv:2505.23189, 2025b.

Yanwei Wang, Lirui Wang, Yilun Du, Balakumar Sundaralingam, Xuning Yang, Yu-Wei Chao, Claudia P´erez-D’Arpino, Dieter Fox, and Julie Shah. Inference-time policy steering through human interactions. In International Conference on Robotics and Automation (ICRA), pp. 15626– 15633. IEEE, 2025c.

Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, et al. Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation. Robotics and Automation Letters (RAL), 2025a.

Junjie Wen, Yichen Zhu, Minjie Zhu, Zhibin Tang, Jinming Li, Zhongyi Zhou, Xiaoyu Liu, Chaomin Shen, Yaxin Peng, and Feifei Feng. Diffusionvla: Scaling robot foundation models via unified diffusion and autoregression. In International Conference on Machine Learning (ICML), 2025b.

Zehang Weng, Haofei Lu, Danica Kragic, and Jens Lundell. Dexdiffuser: Generating dexterous grasps with diffusion models. Robotics and Automation Letters (RAL), 2024.

Thadd¨aus Wiedemer, Prasanna Mayilvahanan, Matthias Bethge, and Wieland Brendel. Compositional generalization from first principles. Advances in Neural Information Processing Systems (NeurIPS), 36:6941–6960, 2023.

Rosa Petra Wolf, Yitian Shi, Sheng Liu, and Rania Rayyes. Diffusion models for robotic manipulation: A survey. Frontiers in Robotics and AI, 12:1606247, 2025.

Lik Hang Kenny Wong, Xueyang Kang, Kaixin Bai, and Jianwei Zhang. A survey of robotic navigation and manipulation with physics simulators in the era of embodied ai. arXiv preprint arXiv:2505.01458, 2025.

Zhou Xian and Nikolaos Gkanatsios. Chaineddiffuser: Unifying trajectory diffusion and keypose prediction for robotic manipulation. In Conference on Robot Learning (CoRL). Proceedings of Machine Learning Research, 2023.

Tian-Yu Xiang, Ao-Qun Jin, Xiao-Hu Zhou, Mei-Jiang Gui, Xiao-Liang Xie, Shi-Qi Liu, ShuangYi Wang, Sheng-Bin Duan, Fu-Chao Xie, Wen-Kai Wang, et al. Parallels between vla model post-training and human motor learning: Progress, challenges, and trends. arXiv preprint arXiv:2506.20966, 2025.

Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Computer Vision and Pattern Recognition (CVPR), pp. 4818–4829, 2024.

Han Xue, Jieji Ren, Wendi Chen, Gu Zhang, Yuan Fang, Guoying Gu, Huazhe Xu, and Cewu Lu. Reactive diffusion policy: Slow-fast visual-tactile policy learning for contact-rich manipulation. arXiv preprint arXiv:2503.02881, 2025.

Ge Yan, Jiyue Zhu, Yuquan Deng, Shiqi Yang, Ri-Zhao Qiu, Xuxin Cheng, Marius Memmel, Ranjay Krishna, Ankit Goyal, Xiaolong Wang, et al. Maniflow: A general robot manipulation policy via consistency flow training. arXiv preprint arXiv:2509.01819, 2025.

Mengjiao Yang, Yilun Du, Bo Dai, Dale Schuurmans, Joshua B Tenenbaum, and Pieter Abbeel. Probabilistic adaptation of text-to-video models. arXiv preprint arXiv:2306.01872, 2023a.

Quantao Yang, Michael C Welle, Danica Kragic, and Olov Andersson. S2-diffusion: Generalizing from instance-level to category-level skills in robot manipulation. arXiv preprint arXiv:2502.09389, 2025.

Zhutian Yang, Jiayuan Mao, Yilun Du, Jiajun Wu, Joshua B Tenenbaum, Tom´as Lozano-P´erez, and Leslie Pack Kaelbling. Compositional diffusion-based continuous constraint solvers. In Conference on Robot Learning (CoRL), pp. 3242–3265. PMLR, 2023b.

Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024.

Tianhe Yu, Ted Xiao, Austin Stone, Jonathan Tompson, Anthony Brohan, Su Wang, Jaspiar Singh, Clayton Tan, Jodilyn Peralta, Brian Ichter, et al. Scaling robot learning with semantically imagined experience. arXiv preprint arXiv:2302.11550, 2023.

Yanjie Ze, Zixuan Chen, Wenhao Wang, Tianyi Chen, Xialin He, Ying Yuan, Xue Bin Peng, and Jiajun Wu. Generalizable humanoid manipulation with 3d diffusion policies. arXiv preprint arXiv:2410.10803, 2024a.

Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu. 3d diffusion policy. Robotics: Science and Systems (RSS), 2024b.

Andy Zhai, Brae Liu, Bruno Fang, Chalse Cai, Ellie Ma, Ethan Yin, Hao Wang, Hugo Zhou, James Wang, Lights Shi, Lucy Liang, Make Wang, Qian Wang, Roy Gan, Ryan Yu, Shalfun Li, Starrick Liu, Sylas Chen, Vincent Chen, and Zach Xu. Igniting vlms toward the embodied space. arXiv preprint arXiv:2509.11766, 2025.

Gengyu Zhang, Hao Tang, and Yan Yan. Versatile navigation under partial observability via valueguided diffusion policy. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17943–17951, 2024a.

Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, and He Wang. Navid: Video-based vlm plans the next step for vision-andlanguage navigation. arXiv preprint arXiv:2402.15852, 2024b.

Kun Zhang, Peng Yun, Jun Cen, Junhao Cai, Didi Zhu, Hangjie Yuan, Chao Zhao, Tao Feng, Michael Yu Wang, Qifeng Chen, et al. Generative artificial intelligence in robotic manipulation: A survey. arXiv preprint arXiv:2503.03464, 2025a.

Qinsheng Zhang, Jiaming Song, Xun Huang, Yongxin Chen, and Ming-Yu Liu. Diffcollage: Parallel generation of large content with diffusion models. In Computer Vision and Pattern Recognition (CVPR), pp. 10188–10198. IEEE, 2023.

Wenbo Zhang, Tianrun Hu, Yanyuan Qiao, Hanbo Zhang, Yuchu Qin, Yang Li, Jiajun Liu, Tao Kong, Lingqiao Liu, and Xiao Ma. Chain-of-action: Trajectory autoregressive modeling for robotic manipulation. arXiv preprint arXiv:2506.09990, 2025b.

Xinyu Zhang, Yuhan Liu, Haonan Chang, Liam Schramm, and Abdeslam Boularias. Autoregressive

action sequence learning for robotic manipulation. IEEE Robotics and Automation Letters, 2025c. Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual

manipulation with low-cost hardware. Robotics: Science and Systems (RSS), 2023.

Tony Z Zhao, Jonathan Tompson, Danny Driess, Pete Florence, Seyed Kamyar Seyed Ghasemipour, Chelsea Finn, and Ayzaan Wahid. Aloha unleashed: A simple recipe for robot dexterity. In Conference on Robot Learning (CoRL), pp. 1910–1924. PMLR, 2025.

Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla: a 3d vision-language-action generative world model. In International Conference on Machine Learning (ICML), pp. 61229–61245, 2024.

Ying Zheng, Lei Yao, Yuejiao Su, Yi Zhang, Yi Wang, Sicheng Zhao, Yiyi Zhang, and Lap-Pui Chau. A survey of embodied learning for object-centric robotic manipulation. Machine Intelligence Research, pp. 1–39, 2025.

Yifan Zhong, Fengshuo Bai, Shaofei Cai, Xuchuan Huang, Zhang Chen, Xiaowei Zhang, Yuanfei Wang, Shaoyang Guo, Tianrui Guan, Ka Nam Lui, et al. A survey on vision-language-action models: An action tokenization perspective. arXiv preprint arXiv:2507.01925, 2025.

Zhongyi Zhou, Yichen Zhu, Minjie Zhu, Junjie Wen, Ning Liu, Zhiyuan Xu, Weibin Meng, Ran Cheng, Yaxin Peng, Chaomin Shen, et al. Chatvla: Unified multimodal understanding and robot control with vision-language-action model. arXiv preprint arXiv:2502.14420, 2025.

Minjie Zhu, Yichen Zhu, Jinming Li, Junjie Wen, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, et al. Scaling diffusion policy in transformer to 1 billion parameters for robotic manipulation. arXiv preprint arXiv:2409.14411, 2024.

Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL), pp. 2165–2183. PMLR, 2023.

APPENDIX

- 1. Assumptions, Notation, and Preliminary Facts – Basic assumptions, notation, and preliminaries.
- 2. Proof of Proposition 4.1 (Single-step Convex Improvement) – Formal proof of the singlestep convex improvement result.
- 3. Proof of Proposition 4.2 (Score-to-Sample Stability) – Detailed stability analysis of scoreto-sample mapping.
- 4. Proof of Corollary 4.1 (GPC Tightens the Terminal Bound) – Proof that GPC improves the overall error bound.
- 5. Detailed Tools: Norm Derivative, Integrating Factor, and Inequalities – Mathematical tools used in the proofs.
- 6. How Propositions and Corollary Fit Together – Logical connections between the main results.
- 7. The Flexibility of GPC with Any Prediction Types – Extensions of GPC to different prediction types.
- 8. In-Depth Analysis on What Leads to the Success of GPC – Analysis on GPC’s success.
- 9. Experiment Details – Experimental setup, architectures, and hyperparameters.
- 10. Additional Experimental Results – Extended quantitative comparisons.
- 11. Visualization on Robot Tasks – Visualizations of robot task rollouts.
- 12. Detailed Literature Review – Extended survey of related work.
- 13. Future Work – Open challenges and future research directions.

- A ASSUMPTIONS, NOTATION, AND PRELIMINARY FACTS

Dynamics. Sampling is modeled as an ODE/SDE:

`

t, xptq, spt,xptqq˘

, t P r0,Ts, xp0q „ µ0, (8)

x9ptq “ F

where xptq P Rd, the score s : r0,Ts ˆ Rd Ñ Rd, and F : r0,Ts ˆ Rd ˆ Rd Ñ Rd represents the transformed score due to noise schedule, parameterization, or solver.

Oracle vs. realized flow. We compare the oracle trajectory x˚ptq solving equation 8 with s˚, and the realized trajectory xsˆptq solving equation 8 with sˆ; both share the same initial conditions.

#### Score error.

∆spt,xq :“ sˆpt,xq ´ s˚pt,xq.

Score error bound assumption. We assume: There exists a nonnegative function κptq such that for all x,

}∆spt,xq} ď κptq. Regularity assumptions. We assume:

- (A1) Lipschitz of F. (Sohrab, 2003; Thomson et al., 2008) For a.e. t P r0,Ts, there exist integrable Lx,Ls ě 0 such that, for all x,y,s,r,

}Fpt,x,sq ´ Fpt,y,rq} ď Lxptq}x ´ y} ` Lsptq}s ´ r}.

- (A2) Score regularity. s˚pt,¨q and sˆpt,¨q are locally Lipschitz on the tube visited by the flows, with (time-dependent) moduli Λ˚ptq,Λˆptq P L1pr0,Tsq and at most linear growth.

Absolute continuity and norm derivative. If e : r0,Ts Ñ Rd is absolutely continuous, then e1ptq exists a.e., eptq “ ep0q ` ş0t e1pτqdτ, and ϕptq :“ }eptq} is absolutely continuous with

ϕ1ptq ď }e1ptq} for a.e. t. (9) We give a complete proof in §E.

Why Lipschitz is reasonable. For probability-flow ODEs of score-based models, F is affine in s (e.g., reverse ODE (Song et al., 2020b): F “ fpt,xq ´ gptq2spt,xq), so Lsptq “ gptq2 (known, bounded on finite T). The dependence on x comes via fpt,xq (often smooth) and through s; with networks on compact tubes, local Lipschitz holds and yields finite Lipschitz moduli Λ˚ptq,Λˆptq. These are standard in stability analyses of neural ODEs and probability-flow ODEs.

- B PROOF OF PROPOSITION 4.1 (SINGLE-STEP CONVEX IMPROVEMENT)

Statement restated. Conditioning on pt,xtq, suppose we have two estimators

##### εi “ s˚pt,xtq ` bipt,xtq ` ηi, i “ t1,2u,

where s˚pt,xtq is the true score, bi are deterministic biases, and ηi are zero-mean random noises. For w P r0,1s, define the convex combination

εpwq “ wε1 ` p1 ´ wqε2, Qpwq :“ Eη}εpwq ´ s˚pt,xtq}2. (10) Notation. This abstraction (equation 10) unifies different modeling paradigms:

- • When noise is present, the estimator can be viewed as the output of a diffusion model, and the residual term η plays the role of the diffusion component in the time-reversed stochastic dynamics (e.g., a reverse-time ODE).
- • When the noise term vanishes (i.e., η “ 0), the formulation reduces to a deterministic transport setting, which is the flow matching case.

The role of ηi is analogous to the stochastic noise introduced in the diffusion forward process (e.g., Gaussian perturbations of the clean sample), ensuring that each estimator εi remains random even when pt,xtq is fixed. All expectations in Proposition 4.1 are therefore taken with respect to the joint distribution of pη1,η2q. and the randomness is solely due to pη1,η2q. We write expectations as

##### “¨ | xt,t

‰

Er¨s ” Eηr¨s ” Eη

##### .

1,η2

Goal. Show that Qpwq is a convex quadratic, derive its coefficients, the minimizer w‹, the minimum value, and conditions for improvement over the endpoints w “ 0,1.

Decomposition. Subtracting s˚pt,xtq, we write

εpwq ´ s˚ “ upwq ` vpwq, where

upwq “ wb1 ` p1 ´ wqb2, vpwq “ wη1 ` p1 ´ wqη2. Hence

Qpwq “ E}upwq ` vpwq}2

“ }upwq}2 ` 2Exupwq,vpwqy ` E}vpwq}2. (11) Since Erηis “ 0, the cross term vanishes, leaving

Qpwq “ }upwq}2 ` E}vpwq}2. Bias contribution. Expanding }upwq}2 gives

}upwq}2 “ }b2 ` wpb1 ´ b2q}2 “ }b2}2 ` 2wxb2,b1 ´ b2y ` w2}b1 ´ b2}2. Noise contribution. Expanding E}vpwq}2 gives

##### E}vpwq}2 “ E}wη1 ` p1 ´ wqη2}2

“ w2 E}η1}2 ` p1 ´ wq2 E}η2}2 ` 2wp1 ´ wqExη1,η2y. Quadratic form. Combining the two contributions, Qpwq is a quadratic function

Qpwq “ Aw2 ` Bw ` C, where

- A “ }b1 ´ b2}2 ` E}η1}2 ` E}η2}2 ´ 2Exη1,η2y,
- B “ 2xb2,b1 ´ b2y ´ 2E}η2}2 ` 2Exη1,η2y, C “ }b2}2 ` E}η2}2.

Convexity. By Cauchy–Schwarz,

Exη1,η2y ď aE}η1}2 E}η2}2, so

A ě paE}η1}2 ´ aE}η2}2q2 ` }b1 ´ b2}2 ě 0. Thus Qpwq is convex, and strictly convex unless both biases and noises coincide. Minimizer. If A ą 0, the unique minimizer is

E}η2}2 ´ Exη1,η2y ´ xb2,b1 ´ b2y }b1 ´ b2}2 ` E}η1}2 ` E}η2}2 ´ 2Exη1,η2y

B 2A “

w‹ “ ´

. The minimum value is

B2 4A

Qpw‹q “ C ´

.

#### Endpoint comparison. At w “ 0,1, we have

Qp0q “ C, Qp1q “ A ` B ` C. The gaps are

2

4A ě 0, Qp1q ´ Qpw‹q “ p2A`Bq

Qp0q ´ Qpw‹q “ B

2

4A ě 0, with strict inequality unless B “ 0 or 2A ` B “ 0.

- Special Case I: Unbiased case. If b1 “ b2 “ 0, then A “ E}η1}2 ` E}η2}2 ´ 2Exη1,η2y, B “ ´2E}η2}2 ` 2Exη1,η2y, C “ E}η2}2,

with A ě 0 and A ą 0 unless η1,η2 are perfectly correlated with identical second moments. For A ą 0, the unique minimizer is

w‹ “

E}η2}2 ´ Exη1,η2y E}η1}2 ` E}η2}2 ´ 2Exη1,η2y

, and the endpoint gaps are

Qp0q ´ Qpw‹q “

B2 4A “ `

E}η2}2 ´ Exη1,η2y˘2 E}η1}2 ` E}η2}2 ´ 2Exη1,η2y

ą 0 whenever Exη1,η2y ‰ E}η2}2, and

Qp1q ´ Qpw‹q “

p2A ` Bq2

4A “ `

E}η1}2 ´ Exη1,η2y˘2 E}η1}2 ` E}η2}2 ´ 2Exη1,η2y

ą 0

whenever Exη1,η2y ‰ E}η1}2. Thus Qpw‹q ă mintQp0q,Qp1qu whenever η1,η2 are not perfectly correlated.

- Special Case II: No-noise (deterministic) case with bias Assume η1 “ η2 “ 0 (deterministic estimators with bias). Then

Qpwq “ }wb1 ` p1 ´ wqb2}2 “ }b2 ` wpb1 ´ b2q}2, w P r0,1s. Write Qpwq “ αw2 ` 2βw ` γ with

α “ }b1 ´ b2}2, β “ xb2, b1 ´ b2y, γ “ }b2}2. If b1 ‰ b2 then α ą 0 and the unconstrained minimizer is wR‹ “ ´β{α, giving

β2 α

minwPR Qpwq “ γ ´

. Hence the endpoint gaps are

xb2, b1 ´ b2y2 }b1 ´ b2}2

pα ` βq2

β2 α “

β2 α “

Qp0q´minQ “

α ě 0 .

ě 0 , Qp1q´minQ “ α`2β`

Therefore, if wR‹ P p0,1q (so the constrained minimizer over r0,1s equals the unconstrained one), we have

Qpw‹q “ minQ ă mintQp0q,Qp1qu,

with strict inequalities unless β “ 0 (for Qp0q) or α ` β “ 0 (for Qp1q). If wR‹ R p0,1q, the constrained minimizer lies at an endpoint and no strict improvement over both endpoints is possible.

If b1 “ b2, then α “ 0 and Qpwq ” }b1}2 for all w.

Remark on strict improvement. Strict improvement requires the unconstrained optimum to lie within p0,1q. If one estimator significantly outperforms the other, the optimal weight lies on the boundary, recovering the best single-estimator performance.

- C PROOF OF PROPOSITION 4.2 (SCORE-TO-SAMPLE STABILITY)

Overview before proof. The goal of the score-to-sample stability result is to show how errors in the score estimation translates into deviations of the generated trajectory. Formally, we view sampling as an ODE of the form x9ptq “ Fpt,xptq,spt,xptqqq, where the score s directly drives the dynamics, and F represents the transformed output after scheduler, parameterization, or solver. Replacing the oracle score s˚ with an estimator sˆ perturbs this vector field, and the resulting trajectory deviation can be quantified.

The proof proceeds by analyzing the trajectory difference eptq “ xsˆptq ´ x˚ptq. Its derivative naturally splits into two terms: a Lipschitz growth component proportional to }eptq}, and a forcing component proportional to the score error }sˆ´ s˚}. This reduces the problem to a standard stability inequality for ODEs. Applying Gr¨onwall’s inequality (Gronwall, 1919; Bellman, 1943) then yields a trajectory-level bound expressed in terms of the integrated score error.

Finally, this stability guarantee connects back to Proposition 4.1: since convex composition strictly improves score estimation at the single-step level, the bound implies that the composed policy inherits a strictly tighter trajectory deviation bound. This prepares the ground for Corollary 4.1, which consolidates the results into a trajectory-level performance guarantee for GPC.

#### Statement restated. Let x˚ptq and xsˆptq solve

x9˚ptq “ Fpt,x˚ptq,s˚pt,x˚ptqqq, x9sˆptq “ Fpt,xsˆptq,sˆpt,xsˆptqqq, with the same xp0q. Under (A1)–(A2), with

L˜ptq :“ Lxptq ` LsptqΛˆptq, we have for all T P r0,Ts:

}xsˆpTq ´ x˚pTq} ď ż T

exp´ż T

L˜pτqdτ¯Lsptq}∆spt,x˚ptqq}dt. (12)

0

t

Taking expectation, applying Cauchy–Schwarz and Jensen and using the assumption of score error bound, we obtain

˜ż T

L˜pτqdτ Lsptq2 dt¸1{2˜ż T

κptq2 dt¸1{2, (13)

e2ştT

E}xsˆpTq ´ x˚pTq} ď

0

0

where κptq is a nonnegative function κptq such that for all x, }∆spt,xq} ď κptq. (i.e., Score error bound assumption). Please see the detailed proof as follows.

- A. ABSOLUTE CONTINUITY AND THE ERROR DIFFERENTIAL INEQUALITY

Let eptq :“ xsˆptq ´ x˚ptq. we have

e1ptq “ Fpt,xsˆptq,sˆpt,xsˆptqqq ´ Fpt,x˚ptq,s˚pt,x˚ptqqq for a.e. t. (14) Insert and subtract two intermediate terms to separate x and s contributions:

`

t,xsˆ,sˆpt,xsˆq˘ ´ F

`

t,xsˆ,sˆpt,x˚q˘

}e1ptq} ď ›F

› (15) ` ›F

`

t,xsˆ,sˆpt,x˚q˘ ´ F

`

t,xsˆ,s˚pt,x˚q˘

› (16) ` ›F

`

t,xsˆ,s˚pt,x˚q˘ ´ F

`

t,x˚,s˚pt,x˚q˘

›. (17)

By (A1), the first (15) and second (16) equations are bounded by Lsptq}sˆpt,xsˆq ´ sˆpt,x˚q} and Lsptq}sˆpt,x˚q ´ s˚pt,x˚q}, respectively; the third equation 17 is bounded by Lxptq}xsˆ ´ x˚} “ Lxptq}eptq}. Using the x-Lipschitzness of sˆ from (A2),

}sˆpt,xsˆq ´ sˆpt,x˚q} ď Λˆptq}eptq}. Therefore,

}e1ptq} ď ´Lxptq ` LsptqΛˆptq¯

}eptq} ` Lsptq}∆spt,x˚ptqq}. (18)

:“L˜ptq

- B. FROM }e1ptq} TO ϕ1ptq

Let ϕptq :“ }eptq}. By equation 9, ϕ is absolutely continuous and

ϕ1ptq ď }e1ptq} for a.e. t.

Combining with equation 18 gives the scalar differential inequality

ϕ1ptq ď L˜ptqϕptq ` Lsptq}∆spt,x˚ptqq} for a.e. t, ϕp0q “ 0. (19)

- C. GRONWALL¨ (INTEGRATING FACTOR) AND THE PATHWISE BOUND

Define Aptq :“ ş0t

L˜pτqdτ and gptq :“ e´Aptqϕptq. Then a.e. g1ptq “ e´Aptq

`

ϕ1ptq ´ L˜ptqϕptq˘ ď e´AptqLsptq}∆spt,x˚ptqq}. Integrate from 0 to T; since ϕp0q “ 0 (same initial conditions) we have gp0q “ 0:

gpTq ď ż T

0

e´AptqLsptq}∆spt,x˚ptqq}dt. Multiply by eApTq:

ϕpTq “ eApTqgpTq ď ż T

0

eApTq´AptqLsptq}∆spt,x˚ptqq}dt.

Since ApTq ´ Aptq “ ştT

L˜pτqdτ, the bound

ϕpTq “ }epTq} ď ż T

0

exp´ż T

t

L˜pτqdτ¯Lsptq}∆spt,x˚ptqq}dt (20) follows, which is exactly equation 12.

- D. EXPECTATION AND A READABLE UPPER BOUND We first take expectations of the pathwise bound equation 20:

E}epTq} “ E«ż T

L˜pτqdτ Lsptq}∆spt,x˚ptqq}dtff.

eştT

0

Notation. The expectation Er¨s is taken over the randomness of the initial conditions. This already provides a valid (and tight) expected bound. In the following we present a slightly looser but cleaner form by applying classical inequalities, which is easier to read and to apply in practice.

By Tonelli’s theorem (non-negative integrand) (Fubini, 1907; Tonelli, 1909): “ ż T

eştT

L˜pτqdτ LsptqE}∆spt,x˚ptqq}dt.

0

Apply Cauchy–Schwarz (Cauchy, 1821) in L2pr0,Tsq: ż T

WptqE}∆spt,x˚ptqq}dt ď ´ż T

Wptq2 dt¯1{2´ż T

pE}∆spt,x˚ptqq}q2 dt¯1{2,

0

0

0

where

##### Wptq :“ eştT

L˜pτqdτ Lsptq. Use Jensen (Jensen, 1906) on pE}∆s}q2 ď E}∆s}2 to obtain

pE}∆spt,x˚ptqq}q2 ď E}∆spt,x˚ptqq}2. Readable expected bound. Combining the above yields

L˜pτqdτ Lsptq2 dt¸1{2˜ż T

E}∆spt,x˚ptqq}2 dt¸1{2 (21)

˜ż T

e2ştT

E}epTq} ď

0

0

Assumption on score error. Using the Assumption of score error bound, which guarantees }∆spt,xq} ď κptq for all x, then

L˜pτqdτ Lsptq2 dt¸1{2˜ż T

κptq2 dt¸1{2.

˜ż T

e2ştT

E}epTq} ď

0

0

This is exactly the equation 13 and the result of Proposition 4.2.

| |
|---|

- D PROOF OF COROLLARY 4.1 (GPC TIGHTENS THE TERMINAL BOUND)

Statement restated. Let Bpsˆq denote the upper bound on the expected sampling error derived in Proposition 4.2. If a convex combination scomp “ ws1 ` p1 ´ wqs2 satisfies

ż T

0

E}scomp ´ s˚}2dt ă min

i

ż T

0

E}si ´ s˚}2dt, then the corresponding theoretical error bound is strictly reduced:

Bpscompq ă min

i

Bpsiq.

Proof. From Proposition 4.2, the expected trajectory error for any estimator sˆ is bounded by:

E}xsˆpTq ´ x˚pTq} ď Bpsˆq :“

˜ż T

0

e2ştT

L˜pτqdτ Lsptq2 dt¸1{2

¨

˜ż T

0

E}sˆ´ s˚}2dt¸1{2 ,

where ˜

ş0T e2ştT

L˜pτqdτ Lsptq2 dt¸1{2

ą 0 is a system-dependent constant independent of the score

estimator. The bound function Bp¨q is strictly increasing with respect to the integrated mean-squared score error (MSE). Since the premise states that the integrated MSE of the composite score scomp is strictly smaller than that of the individual estimators si, it follows immediately that:

Bpscompq ă min

i

Bpsiq.

Thus, convex score composition strictly tightens the theoretical guarantee on the trajectory simulation error.

| |
|---|

- E DETAILED TOOLS: NORM DERIVATIVE, INTEGRATING FACTOR, AND INEQUALITIES

- E.1 NORM DERIVATIVE INEQUALITY

Let e : r0,Ts Ñ Rd be absolutely continuous. Define ϕptq “ }eptq}. We show ϕ is absolutely continuous and ϕ1ptq ď }e1ptq} for a.e. t.

Absolute continuity. Since eptq “ ep0q ` ş0t e1pτqdτ with e1 P L1, and the norm is 1-Lipschitz, ϕ is absolutely continuous.

Difference-quotient proof. Fix a point where e1 exists. Then

ϕpt ` hq ´ ϕptq h “

}ept ` hq} ´ }eptq}

}ept ` hq ´ eptq} h

h ď

##### .

Taking h Ñ 0 gives ϕ1ptq ď }e1ptq}. This holds for a.e. t. Chain-rule proof (when eptq ‰ 0). For gpxq “ }x}, ∇gpxq “ x{}x} when x ‰ 0. Then

ϕ1ptq “ x∇gpeptqq,e1ptqy “ A eptq

, e1ptqE ď }e1ptq}.

}eptq}

At points with eptq “ 0, use the difference-quotient argument above.

- E.2 INTEGRATING FACTOR

Starting from ϕ1ptq ď aptqϕptq ` bptq with ϕp0q “ 0 and a,b P L1, define Aptq “ ş0t apτqdτ and gptq “ e´Aptqϕptq. Then

g1ptq “ e´Aptq

`

ϕ1ptq ´ aptqϕptq˘ ď e´Aptqbptq. Integrate:

gpTq ď ż T

0

e´Aptqbptqdt ñ ϕpTq ď ż T

0

eApTq´Aptqbptqdt.

Since ApTq ´ Aptq ď ş0T a, a looser bound is ϕpTq ď eş0T a ş0T bptqdt.

- E.3 TONELLI, CAUCHY–SCHWARZ, AND JENSEN Given a nonnegative integrand Hpω,tq on Ω ˆ r0,Ts, Tonelli implies

E«ż T

Hpω,tqdtff

“ ż T

ErHpω,tqsdt.

0

0

ş0T fg ď }f}2 }g}2. For a random variable Z, Jensen yields pE}Z}q2 ď E}Z}2.

For functions f,g P L2pr0,Tsq,

- F HOW PROPOSITIONS AND COROLLARY FIT TOGETHER

Prop. 4.1 guarantees the existence of a convex weight (often interior) that lowers the score MSE under mild, testable conditions (heterogeneous models reduce cross-correlation and diversify biases). Prop. 4.2 translates any reduction in (time-integrated) score MSE into a reduction of a nonasymptotic terminal error bound. Cor. 4.1 merely combines the two: once the functional-level inequality is strict, the certified sampling bound tightens accordingly.

- G THE FLEXIBILITY OF GPC WITH ANY PREDICTION TYPES

A key strength of General Policy Composition (GPC) is its flexibility and independence from the specific parameterization used to train the underlying diffusion or flow-matching policies. The fundamental principle of GPC is the composition of the underlying score functions of the data distributions, sθpτt,tq “ ∇τt

log ptpτtq. Common parameterizations, such as noise prediction, data prediction, and v-prediction, are all mathematically inter-convertible and represent this same underlying score function. This ensures that GPC can seamlessly compose policies trained with different prediction objectives without requiring extra training.

Let’s formalize the relationship between these parameterizations. The diffusion forward process defines a noisy trajectory τt at time t from an initial trajectory τ0 and a Gaussian noise sample ϵ „ Np0,Iq as:

τt “ αtτ0 ` σtϵ, (22) where αt and σt are schedule-dependent coefficients. Score Prediction (s-prediction). This parameterization directly models the score function. The score is related to the noise ϵ by the following identity (Song et al., 2020b):

ϵ σt

. (23)

spτt,tq “ ∇τt

log ptpτtq “ ´

Composing scores is the core of GPC. Any other parameterization can be converted to a score before composition.

Noise Prediction (ϵ-prediction). This is the most common parameterization, used in the original DDPM (Ho et al., 2020). The model ϵθpτt,tq is trained to predict the noise ϵ. A model trained on noise prediction can be converted to a score prediction model:

ϵθpτt,tq σt

. (24)

sθpτt,tq “ ´

Since the relationship is linear, composing predicted noises with weights wi is equivalent to composing the scores with the same weights.

Data Prediction (τ0-prediction). This parameterization trains the model pτ0qθpτt,tq to predict the original clean data τ0 from the noisy input τt. The predicted noise ϵ can be recovered from the predicted data using the forward process definition:

τt ´ αtpτ0qθpτt,tq σt

. (25)

ϵθpτt,tq “

This allows a data-prediction policy to be converted to the score or noise representation for composition.

Velocity Prediction (v-prediction). Introduced by (Salimans & Ho, 2022), v-prediction offers improved numerical stability. The target, v, is defined as v “ αtϵ ´ σtτ0. A model vθpτt,tq is trained to predict this target. We can recover the noise ϵ from a v-prediction model’s output using:

ϵθpτt,tq “ αtvθpτt,tq ` σtτt. (26) From there, the equivalent score can be calculated.

Implications for GPC. The interchangeability of these parameterizations is what makes GPC “solver-agnostic.” Suppose we want to compose two policies, π1 and π2. If π1 was trained using noise prediction (outputting ϵ1θ) and π2 was trained using v-prediction (outputting vθ2), we can perform composition by first converting their outputs to a common representation.

For example, we can convert both to the score representation:

- s1θpτt,tq “ ´

ϵ1θpτt,tq σt

(27)

- s2θpτt,tq “ ´

αtvθ2pτt,tq ` σtτt σt

(28)

Then, we can perform the convex composition in the score space:

##### scomp “ w1s1θ ` w2s2θ. (29)

This composed score scomp can then be used in any standard ODE/SDE solver step to generate the next state τt´1.

Alternatively, and often more direct in practice, one can convert all outputs to the noise (ϵ) representation before composition, which yields an equivalent result due to the linear relationship between score and noise. This flexibility allows GPC to serve as a universal, plug-and-play module for combining a wide variety of pre-trained diffusion-based or flow-based policies, regardless of their specific training objective or parameterization.

- H IN-DEPTH ANALYSIS ON WHAT LEADS TO THE SUCCESS OF GPC

Intuitively, GPC can be viewed as forming a product-of-experts distribution: the composed score is a convex combination of individual scores, corresponding to a (re-weighted) product of their probability densities. A higher weight on one policy simply means its density contributes more strongly to the final target distribution, concentrating probability mass on trajectories that both experts consider likely.

- H.1 THEORETICAL PERSPECTIVE: WHY CONVEX SCORE COMPOSITION CAN BE BETTER THAN INDIVIDUAL POLICIES

Based on Prop. 4.1, we show that at each diffusion timestep there exists a convex weight w˚ such that the composed score has a smaller error with respect to the ideal score s˚ than any individual policy. Prop. 4.2 extends this from a single step to the entire denoising trajectory: if at each step the composed score is closer to s˚, then the error along the whole trajectory accumulates more favorably, leading to a trajectory distribution closer to the ideal one. In other words, once perstep improvement in score quality is established, the stability of the generative process allows this advantage to propagate along the full trajectory, making convex score composition provably superior to relying on a single policy.

- H.2 EMPIRICAL PERSPECTIVE: WHAT DRIVES LARGER GAINS FOR SOME COMBINATIONS

At a high level, we view GPC as a way to aggregate the “good knowledge” learned by different base policies into a single, higher-likelihood target distribution, so that sampling from the composed score produces higher-quality trajectories than sampling from any individual policy alone. In practice, what counts as “good knowledge” is shaped by several factors:

Complementary modalities provide richer information. When GPC combines policies trained on different modalities (e.g., RGB vs. point cloud), the composed score effectively leverages complementary views of the scene: RGB captures appearance, texture, and color cues, while point clouds provide precise 3D geometry and depth structure. This multimodal fusion reduces perceptual ambiguity compared to using either modality alone, and can be viewed as increasing the “informational richness” available to the composed policy. The DP+DP3 combination is a good example of this, and the visualizations of their sample distributions in Fig. 5(a) support this interpretation.

Diverse architectures capture different inductive biases. Even when policies share the same input modality, different architectures can encode different inductive biases and error patterns. For instance, π0 is DiT-based, whereas Flow Policy uses a U-Net-style backbone. They may model similar underlying action distributions but emphasize different structures in the data. GPC can exploit this diversity by aggregating their strengths while averaging out idiosyncratic weaknesses, which we see reflected in improved performance and the qualitative analysis in Fig. 5(b).

Task-specific strengths shape composition benefits. Task choice also matters: in some tasks, one base policy may be significantly weaker. In such cases, naive averaging (or an unbalanced weight) can let the weaker policy drag down the overall performance. Our analysis in Sec. 6.3 (Finding 3) shows that GPC performs best when the *better-performing base policy receives a larger weight*, which is especially important when task difficulty or mismatch affects one policy more than the other. Thus, task-specific policy strengths and the ability to adjust weights are key factors behind the observed gains.

Weight selection matters in the composed distribution. Our theory guarantees the existence of an optimal weight w˚, but in practice GPC still requires choosing concrete weights. These weights control how much each base distribution contributes to the final composed distribution and therefore directly influence how much probability mass is placed on high-quality trajectories. This is why we performed extensive weight-sweep studies: to show how different weights affect performance, and to provide empirical guidance on weight selection for real deployments. We view designing better, possibly adaptive, strategies to approximate w˚ as an important direction for future work.

- I EXPERIMENT DETAILS

- I.1 ROBOMIMIC

The Robomimic benchmark (Mandlekar et al., 2022) includes three manipulation tasks: Can, Lift, and Square. We train all baselines with batch size 1024 for 1000 epochs. Training uses DDIM sampling with the scaled linear beta scheduler and prediction with epsilon. Diffusion steps are set to 100 during training and 10 at inference. Each model is trained with observation horizon = 2 and chunk size = 16. Evaluation is performed across 20 parallel environments, each running 10 episodes, giving a total of 200 rollouts. The original code of Robomimic is from https: //github.com/ARISE-Initiative/robomimic. We reproduce the baselines based on the codes from https://github.com/EDiRobotics/mimictest.

- I.2 PUSHT

The PushT benchmark (Florence et al., 2021) involves planar pushing in a 2D workspace. Here, training uses batch size 256 and runs for 500 epochs, with all other parameters kept identical to Robomimic. Evaluation follows the same protocol of 200 rollouts. The original code of Robomimic is from https://github.com/real-stanford/diffusion_policy and https:// github.com/google-research/ibc. We reproduce the baselines based on the codes from https://github.com/EDiRobotics/mimictest.

- I.3 ROBOTWIN

RoboTwin (Mu et al., 2025) is a dual-arm manipulation benchmark that combines real-world teleoperated demonstrations with high-fidelity synthetic data, offering a standardized platform for studying large-scale manipulation learning. The extended RoboTwin 2.0 (Chen et al., 2025) release covers more than 50 tasks, supporting diverse and complex scenarios. Baselines are reproduced based on the codes from https://github.com/RoboTwin-Platform/ RoboTwin/tree/RoboTwin-1.0 and https://github.com/RoboTwin-Platform/ RoboTwin/tree/main. The success rate of each task is determined with 100 rollouts.

For our experiments, we evaluate on a curated subset of tasks:

- • RoboTwin 1.0: Empty Cup Place, Dual Bottles Pick (Hard), Dual Bottles Pick (Easy), Shoe Place, Dual Shoes Place, Pick Apple Messy, Block Hammer Beat.
- • RoboTwin 2.0: Hanging Mug, Open Laptop, Place Burger Fries, Put Object Cabinet, Stack Bowls, Three Turn Switch.

The DPimg and DPpcd correspond to the diffusion policy based on RGB images (i.e., DP (Chi et al., the DPimg and DPpcd (without using point cloud color) with random seed 0. Since the diffusion scores from different policies are composed at each denoising step (Alg. 1), we unify the training settings of both DPpcd and DPimg. In particular, they are trained with DDPM with 100 training and inference steps. In RoboTwin 2.0 experiments, we train DPs with the same settings as RDT to ensure compatibility so that our GPC can be applied consistently. For example, RDT employs sample prediction, we align our diffusion models accordingly by training both DPimg and DPpcd under the same prediction setting.

- 2023)) and point cloud (i.e., DP3 (Ze et al., 2024b)), respectively. In RoboTwin 1.0, we reproduce

Task Prompt in RoboTwin. Here we present the detailed text description and its corresponding text prompt for VLAs. For example, for the Place Burger task, the task description and schema are:

- • Full description: “Use dual arm to pick the hamburg and frenchfries and put them onto the tray.”
- • Schema: “A denotes the hamburg, B denotes the tray, C denotes the frenchfries”

During training, the VLA receives diverse paraphrased prompts for this task, such as:

- • “Use both arms to move A and C to B.”

[Figure 88]

Figure 7: Illustration of Experimental Setup.

[Figure 89]

[Figure 90]

[Figure 91]

### 

## GPC  DPimg DPpcd

[Figure 92]

[Figure 93]

#### Figure 8: Tracking Results of Real-world Experiment in Place Bottles.

- • “Lift A and C, placing them neatly on B”

At test time, we use unseen paraphrases with the same schema, e.g..:

- • “Pick A and C, then place them on B.”
- • “Grab A and C together, setting them on B.”

- I.4 REAL-WORLD EXPERIMENTS

We choose DPimg and DPpcd as our base policies for real-world experiments. For DPimg, we use an Intel RealSense D435 RGB camera at 640 ˆ 480 resolution (primary view and wrist view) to get

the RGB images. For DPpcd, we use an Intel RealSense L515 depth camera at 640 ˆ 480, where we obtain point clouds by using depth images together with camera intrinsics. The robot platform is Piper, operated in a master–slave teleoperation setup. The illustration of the real-world experimental setup is shown in Fig. 7. Our GPC achieves superior performance compared with the base policies,

presenting better trajectories in Fig. 8. Training follows official configurations: DPpcd is trained for 600 epochs with batch size 256 (official code), while DPimg is trained for 20k steps with batch size 64 (Lerobot (Cadene et al., 2024) diffusion implementation).

- I.5 NOTATION FOR GPC FLEXIBILITY

Notably, the prediction types in diffusion models are not strictly restricted to a single formulation (e.g., ϵ-prediction, x0-prediction, or v-prediction), but can be freely combined within our framework. When heterogeneous prediction types are adopted simultaneously, the denoising process requires proper alignment to ensure consistency. We provide detailed guidance in Sec. G on how to reconcile different prediction types in GPC from a theoretical perspective, further demonstrating the flexibility of our proposed method.

- J ADDITIONAL EXPERIMENTAL RESULTS

In this section, we provide the complete set of experimental results to complement the main text. These results include all weight configurations for convex score composition, as well as the outcomes under logical AND and OR operators.

Robomimic and PushT. We report detailed results on the Robomimic (Can, Lift, Square) and PushT tasks. In addition to the average performance reported in the main paper, we include (i) the full tables for convex score combination, logical AND, and logical OR composition in Tab. 10, and (ii) the breakdown of performance under different convex weights w for each task: Can (Tab. 11 & Fig. 9), Square (Tab. 13 & Fig. 10), Lift (Tab. 12 & Fig. 11) and PushT (Tab. 14 & Fig. 12). These results illustrate how GPC adapts across weighting configurations and provide insight into the trade-offs between modalities and model backbones.

RoboTwin. We also provide the complete results on RoboTwin 2.0 across all tasks. In particular, we include full tables comparing base policies and their compositions (e.g., DPimg + DPpcd, RDT + DPpcd), with all tested weight settings: Open Labtop (Tab. 15), Place Burger (Tab. 16 & Fig. 21), Put Object Cabinet (Tab. 17 & Fig. 24), Hanging Mug (Tab. 18 & Fig. 23), Stack Bowls Three (Tab. 19 & Fig. 14) and Turn Switch (Tab. 20 & Fig. 20). These detailed numbers confirm the robustness of GPC across diverse manipulation tasks and further validate the findings in Sec. 6.

Real-world Experiments. We further report complete results for the four real-world tasks: Place Bottles (Tab. 21 & Fig. 27), Hang Mug (Tab. 22 & Fig. 26), Clean Table (Tab. 23 & Fig. 25), and Punch Holes (Tab. 24 & Fig. 28). Similar to the simulation benchmarks, we provide full comparisons between base policies and their GPC compositions under all tested weight settings. These results consistently show that GPC achieves higher success rates than individual policies, thereby confirming its effectiveness in practical robotic scenarios.

Summary. Together, these extended results give a comprehensive view of GPC’s empirical behavior across different operators and weightings. They serve as a reference for understanding not only the average improvements but also the sensitivity of performance to the choice of weights and composition strategies.

#### Table 10: Experiments on Robomimic and PushT with GPC under convex score combination, Logical AND and Logical OR.

Robomimic PushT

Method Generative Mode Model Type

Can Lift Square PushT Average Base Policies

Diffusion Policy (DP) Diffusion VA 34.50 98.50 2.00 21.75 39.19 Mamba Policy (MP) Flow Matching VA 5.00 98.50 3.00 12.06 29.64 Flow Policy (FP) Diffusion VA 95.00 13.00 77.50 54.25 59.94 Florence Policy-D Diffusion VLA 61.50 97.00 46.50 40.00 61.25 Florence Policy-F Flow Matching VLA 89.00 98.50 88.50 39.38 78.84 π0 Flow Matching VLA 96.50 99.00 92.50 57.69 86.42

Composed Policies via Convex Score Combination DP+MP Diffusion VA & VA 34.50 99.50 8.00 23.63 41.41 +2.22 Florence-Policy-D+DP Diffusion VLA & VA 62.50 100.00 61.50 43.06 66.76 +5.51 Florence-Policy-D+MP Diffusion VLA & VA 63.00 100.00 54.50 40.88 64.60 +3.35 Florence-Policy-F+FP Flow Matching VLA & VA 98.50 98.50 92.50 56.06 86.39 +7.55 π0+FP Flow Matching VLA & VA 99.50 100.00 94.00 62.25 88.94 +2.52 Composed Policies via Logical AND Composition DP+MP Diffusion VA & VA 84.00 99.50 48.00 28.18 64.92 +25.73 Florence-Policy-D+DP Diffusion VLA & VA 90.50 100.00 90.00 36.31 79.20 +17.95 Florence-Policy-D+MP Diffusion VLA & VA 83.00 100.00 90.00 37.38 77.60 +16.35 Composed Policies via Logical OR Composition DP+MP Diffusion VA & VA 82.50 99.50 44.00 29.13 63.78 +24.59 Florence-Policy-D+DP Diffusion VLA & VA 83.50 100.00 89.00 37.87 77.59 +16.34 Florence-Policy-D+MP Diffusion VLA & VA 86.50 100.00 86.50 38.44 77.86 +16.61

#### Table 11: Experiments on Robomimic Can with GPC under different weighting.

|Method Generative Mode Model Type<br><br>|Can|
|---|---|
| |0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0|
|DP+Mamba Policy Diffusion VA & VA Florence DiT + DP Diffusion VLA & VA Florence DiT + MambaP Diffusion VLA & VA Florence Flow+FlowP Flow Matching VLA & VA π0+FlowP Flow Matching VLA & VA|5.00 10.00 10.50 10.50 16.50 20.00 20.00 23.00 25.00 29.00 34.50<br><br>34.50 34.50 42.50 48.00 56.00 62.50 60.50 63.50 58.00 62.50 61.50<br><br>5.00 11.50 21.50 30.50 39.00 44.50 47.50 46.50 56.50 63.00 61.50<br><br><br>95.00 98.50 98.50 96.00 96.50 97.00 93.00 92.00 90.00 90.50 89.00 95.00 96.00 99.00 98.00 97.50 98.50 99.50 98.00 96.00 96.00 96.50<br><br>|

#### Table 12: Experiments on Robomimic Lift with GPC under different weighting.

|Method Generative Mode Model Type|Lift|
|---|---|
| |0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0|
|DP+Mamba Policy Diffusion VA & VA Florence DiT + DP Diffusion VLA & VA Florence DiT + MambaP Diffusion VLA & VA Florence Flow+FlowP Flow Matching VLA & VA π0+FlowP Flow Matching VLA & VA<br><br>|98.50 99.00 99.50 96.50 99.00 98.50 98.50 98.50 98.50 98.50 98.50<br><br>98.50 99.50 99.00 100.00 99.50 99.50 99.50 99.50 99.50 98.00 97.00<br>98.50 100.00 99.50 99.00 99.50 99.00 99.00 97.50 98.50 97.00 97.00 13.00 10.50 12.50 23.50 55.00 81.50 93.00 98.00 100.00 98.50 98.50 13.00 12.50 17.00 32.50 67.50 92.50 98.50 100.00 100.00 99.50 99.00<br>|

#### Table 13: Experiments on Robomimic Square with GPC under different weighting.

|Method Generative Mode Model Type<br><br>|Square|
|---|---|
| |0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0|
|DP+Mamba Policy Diffusion VA & VA Florence DiT + DP Diffusion VLA & VA Florence DiT + MambaP Diffusion VLA & VA Florence Flow+FlowP Flow Matching VLA & VA π0+FlowP Flow Matching VLA & VA<br><br>|3.00 3.00 4.00 1.50 8.00 4.50 6.00 6.00 3.50 7.50 2.00<br><br>2.00 12.50 20.00 34.00 44.00 49.00 61.50 57.00 59.50 54.50 46.50<br>3.00 8.00 8.50 17.00 22.00 34.00 45.00 45.50 50.00 54.50 46.50<br><br><br>77.50 79.00 85.00 92.00 92.00 92.00 91.00 88.00 88.50 92.50 88.50<br>77.50 80.50 84.50 94.00 93.50 94.00 93.50 93.00 90.50 93.50 92.50<br>|

Table 14: Experiments on PushT with GPC under different weighting.

|Method Generative Mode Model Type<br><br>|PushT|
|---|---|
| |0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0|
|DP+Mamba Policy Diffusion VA & VA Florence DiT + DP Diffusion VLA & VA Florence DiT + MambaP Diffusion VLA & VA Florence Flow+FlowP Flow Matching VLA & VA π0+FlowP Flow Matching VLA & VA<br><br>|12.06 19.81 18.31 18.87 19.94 19.88 18.13 21.50 23.63 22.38 21.75 21.75 26.75 29.38 32.75 36.06 39.69 41.13 43.06 40.50 40.56 40.00 12.06 22.88 25.81 30.62 33.94 37.00 38.44 40.50 40.75 40.88 40.00 54.25 56.06 54.50 50.81 47.38 48.31 47.69 50.50 46.19 40.75 39.38 54.25 54.31 56.81 56.37 53.31 57.69 59.12 61.50 62.25 61.50 57.69|

- Table 15: Experiments on RoboTwin Open Laptop with GPC under different weighting.

|Method Generative Mode Model Type|RoboTwin: Open Laptop|
|---|---|
| |0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0<br><br>|
|DP+DP3 Diffusion VA & VA RDT + DP Diffusion VLA & VA RDT + DP3 Diffusion VLA & VA|0.93 0.93 0.92 0.93 0.93 0.87 0.84 0.79 0.77 0.74 0.74 0.74 0.74 0.77 0.78 0.79 0.80 0.75 0.76 0.73 0.68 0.69 0.93 0.92 0.92 0.91 0.92 0.94 0.91 0.86 0.77 0.67 0.69<br><br>|

- Table 16: Experiments on RoboTwin Place Burger Fries with GPC under different weighting.

|Method Generative Mode Model Type<br><br>|RoboTwin: Place Burger Fries|
|---|---|
| |0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0|
|DP+DP3 Diffusion VA & VA RDT + DP Diffusion VLA & VA RDT + DP3 Diffusion VLA & VA<br><br>|0.72 0.73 0.78 0.74 0.72 0.65 0.68 0.64 0.66 0.54 0.49 0.49 0.54 0.56 0.57 0.53 0.49 0.50 0.48 0.45 0.45 0.46 0.72 0.79 0.83 0.83 0.77 0.78 0.72 0.67 0.67 0.55 0.46|

- Table 17: Experiments on RoboTwin Put Object Cabinet with GPC under different weighting.

|Method Generative Mode Model Type<br><br>|RoboTwin: Put Object Cabinet|
|---|---|
| |0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0|
|DP+DP3 Diffusion VA & VA RDT + DP Diffusion VLA & VA RDT + DP3 Diffusion VLA & VA<br><br>|0.71 0.80 0.82 0.73 0.66 0.73 0.63 0.67 0.67 0.55 0.56 0.56 0.54 0.59 0.54 0.51 0.52 0.40 0.35 0.27 0.32 0.32 0.71 0.71 0.78 0.71 0.69 0.61 0.58 0.46 0.37 0.40 0.32|

- Table 18: Experiments on RoboTwin Hanging Mug with GPC under different weighting.

|Method Generative Mode Model Type|RoboTwin: Hanging Mug|
|---|---|
| |0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0<br><br>|
|DP+DP3 Diffusion VA & VA RDT + DP Diffusion VLA & VA RDT + DP3 Diffusion VLA & VA<br><br>|0.21 0.23 0.20 0.22 0.18 0.17 0.16 0.11 0.15 0.11 0.10 0.10 0.13 0.09 0.15 0.11 0.18 0.18 0.15 0.14 0.12 0.13 0.21 0.26 0.31 0.30 0.36 0.25 0.25 0.22 0.24 0.15 0.13|

- Table 19: Experiments on RoboTwin Stack Bowls Three with GPC under different weighting.

|Method Generative Mode Model Type<br><br>|RoboTwin: Stack Bowls Three|
|---|---|
| |0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0|
|DP+DP3 Diffusion VA & VA RDT + DP Diffusion VLA & VA RDT + DP3 Diffusion VLA & VA<br><br>|0.64 0.70 0.66 0.71 0.60 0.53 0.63 0.56 0.59 0.49 0.52 0.52 0.65 0.66 0.57 0.66 0.59 0.58 0.50 0.40 0.32 0.47<br>0.64 0.71 0.73 0.55 0.71 0.70 0.60 0.59 0.48 0.42 0.47<br>|

- Table 20: Experiments on RoboTwin Turn Switch with GPC under different weighting.

|Method Generative Mode Model Type<br><br>|RoboTwin: Turn Switch|
|---|---|
| |0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0|
|DP+DP3 Diffusion VA & VA RDT + DP Diffusion VLA & VA RDT + DP3 Diffusion VLA & VA<br><br>|0.71 0.68 0.60 0.63 0.67 0.56 0.50 0.45 0.41 0.42 0.38 0.38 0.28 0.31 0.28 0.36 0.37 0.34 0.30 0.38 0.35 0.30 0.71 0.52 0.54 0.48 0.51 0.59 0.51 0.43 0.42 0.45 0.30|

#### Table 21: Experiments on Real-world Place Bottle with GPC under different weighting.

|Method Generative Mode Model Type<br><br>|Real-world: Place Bottle|
|---|---|
| |0.0 0.2 0.4 0.6 0.8 1.0<br><br>|
|DP+DP3 Diffusion VA & VA|11/20 13/20 11/20 12/20 10/20 7/20|

#### Table 22: Experiments on Real-world Hang Mug with GPC under different weighting.

|Method Generative Mode Model Type<br><br>|Real-world: Hang Mug|
|---|---|
| |0.0 0.2 0.4 0.6 0.8 1.0|
|DP+DP3 Diffusion VA & VA|6/20 7/20 5/20 7/20 6/20 5/20|

#### Table 23: Experiments on Real-world Clean Table with GPC under different weighting.

|Method Generative Mode Model Type|Real-world: Clean Table|
|---|---|
| |0.0 0.2 0.4 0.6 0.8 1.0|
|DP+DP3 Diffusion VA & VA|7/20 7/20 14/20 10/20 12/20 12/20|

#### Table 24: Experiments on Real-world Punch Holes with GPC under different weighting.

|Method Generative Mode Model Type<br><br>|Real-world: Punch Holes|
|---|---|
| |0.0 0.2 0.4 0.6 0.8 1.0|
|DP+DP3 Diffusion VA & VA<br><br>|6/20 6/20 5/20 7/20 9/20 7/20|

- K VISUALIZATION ON ROBOT TASKS

[Figure 94]

[Figure 95]

[Figure 96]

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

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

#### Figure 9: Illustration of Robomimic Can.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

#### Figure 10: Illustration of Robomimic Square.

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

#### Figure 11: Illustration of Robomimic Lift.

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Figure 12: Illustration of PushT.

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Figure 13: Illustration of RoboTwin 1.0 Blocks Stack (Hard).

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

Figure 14: Illustration of RoboTwin 1.0 Bowl Stack.

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

- Figure 15: Illustration of RoboTwin 1.0 Dual Bottle Pick Hard.

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

- Figure 16: Illustration of RoboTwin 1.0 Dual Shoes Place.

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

#### Figure 17: Illustration of RoboTwin 1.0 Empty Cup Place.

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

#### Figure 18: Illustration of RoboTwin 1.0 Pick Apple Messy.

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

#### Figure 19: Illustration of RoboTwin 1.0 Put Bottles Dustbin.

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

Figure 20: Illustration of RoboTwin 2.0 Turn Switch.

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

- Figure 21: Illustration of RoboTwin 2.0 Place Burger Fries.

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

#### Figure 22: Illustration of RoboTwin 2.0 Open Laptop.

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

#### Figure 23: Illustration of RoboTwin 2.0 Hanging Mug.

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

#### Figure 24: Illustration of RoboTwin 2.0 Put Object Cabinet.

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

#### Figure 25: Illustration of Real-world Experiment Clean Table.

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

#### Figure 26: Illustration of Real-world Experiment Hang Mug.

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

#### Figure 27: Illustration of Real-world Experiment Place Bottles.

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

#### Figure 28: Illustration of Real-world Experiment Punch Holes.

- L DETAILED LITERATURE REVIEW

Owing to space constraints in the main text, we expand the related work section with a detailed literature review. This appendix aims to provide a more comprehensive overview of prior work, highlighting additional studies and applications that could not be discussed in detail in the main body of the paper. All these works have made significant contributions to the robotics community.

- L.1 COMPOSITIONAL GENERATIVE MODELING

Compositional generative modeling has recently emerged as a compelling alternative to monolithic large-scale models, emphasizing the idea that complex data distributions can be captured more effectively by composing simpler factors. Instead of relying on a single, over-parameterized model, researchers argue that distributions can be factorized and modeled in a modular fashion, thereby reducing data requirements and improving interpretability (Koller & Friedman, 2009; Murphy, 2022). This line of work draws inspiration from probabilistic graphical models and energy-based formulations, but has been extended to modern deep generative architectures.

A key motivation for compositional modeling is data efficiency. By factorizing distributions into manageable components, one can achieve accurate modeling even under limited training data. For instance, Janner et al. (2022) and Ajay et al. (2022) showed that trajectory generation benefits from decomposing the sequence into components, leading to faster training and improved generalization. In natural language processing, Du et al. (2023b) demonstrated that reasoning can be improved by combining multiple large language models, effectively composing factors across models in a “multi-agent” framework. Similarly, Liu et al. (Liu et al., 2021) introduced composable diffusion for text-to-image generation, where local sentence-level factors combine to synthesize complex global scenes.

Beyond efficiency, compositionality provides a natural mechanism for generalization to novel tasks and distributions. In decision-making and planning, Ajay et al. (2023) proposed hierarchical foundation models that integrate language, video, and action policies, enabling flexible recombination for zero-shot planning (Wang et al., 2025c). In robotic manipulation, Yang et al. (2023b) and Mishra et al. (2023) showed that object rearrangement tasks can be solved by composing local constraint factors, while Wang et al. (2024c) extended this idea to heterogeneous policy composition (Wang

- et al., 2024b; Wang, 2025). For visual domains, Du et al. (2023a) and Zhang et al. (2023) developed methods to assemble image collages via factorized regional conditionals, while Yang et al.

- (2023a) demonstrated that video style transfer can be achieved by composing a pretrained prior with a lightweight style model.

Another strand of work investigates how compositional structure can be discovered automatically. Du et al. (2021) and Su et al. (2024) showed that autoencoders trained with product-of-experts likelihoods naturally uncover object-level factors, which can later be recombined to generate hybrid scenes. In dynamical systems, Comas et al. (2023) inferred relational potentials between particles, enabling recombination of discovered interaction rules. Similarly, Liu et al. (2023) found that compositional components learned on ImageNet correspond to semantic classes, making it possible to synthesize images of unseen multi-class combinations.

At the methodological level, energy-based models (EBMs) provide a natural framework for composition, since energies are additive by construction (Hinton, 2002; Du & Mordatch, 2019; Grathwohl et al., 2021). This perspective has been adapted to diffusion models, where each time step defines an implicit EBM and compositions are realized by combining energies across models (Song & Ermon, 2019; Ho et al., 2020; Du et al., 2023a). Extensions to discrete domains employ Metropolis– Hastings with learned proposals (Li et al., 2022; Verkuil et al., 2022), while Garipov et al. (2023) demonstrated how constraint energies can “sculpt” generative trajectories at inference time.

Despite these advances, challenges remain. Current approaches often assume a fixed structure of composition, limiting adaptability. Efforts by Wiedemer et al. (2023), Lachapelle et al. (2023), Misino et al. (2022), and Sehgal et al. (2023) highlight the need for robust theoretical frameworks that explain compositional generalization and offer methods to automatically infer appropriate factorization structures. Addressing these open problems will be crucial for compositional models to scale and integrate seamlessly into real-world generative systems.

- L.2 DIFFUSION MODELS IN ROBOT LEARNING

Diffusion models have become a central paradigm for robot learning, offering a probabilistic framework for efficient trajectory generation and planning. Building on the recent surveys (Barreiros

- et al., 2025; Shao et al., 2025; Zhong et al., 2025; Xiang et al., 2025; Firoozi et al., 2025; Song et al., 2025; Sapkota et al., 2025; Wong et al., 2025; Wolf et al., 2025; An et al., 2025; Zhang

- et al., 2025a; Adilkhanov et al., 2025; Lin et al., 2024b; Ma et al., 2024b; Zheng et al., 2025; Liu
- et al., 2025b; Urain et al., 2024), we group diffusion-based robot policies into two categories (Song et al., 2025): (i) small-size diffusion-based policies, which integrate CNN or Transformer backbones with diffusion heads and are trained on task-specific datasets for efficient visuomotor control, and (ii) large-scale diffusion policies, which couple diffusion modules with pre-trained foundation models or large robot datasets to achieve broader semantic grounding and cross-embodiment generalization. Together, these developments demonstrate how diffusion can serve both as a lightweight control primitive in specialized tasks and as a scalable component in foundation-style robot policies, bridging the gap between low-level stochastic control and high-level semantic reasoning.

Small-size CNN/Transformer-based diffusion policies. A growing body of visuomotor research couples compact CNN or Transformer encoders with diffusion heads, showing that stochastic denoising can serve as an effective control primitive across diverse manipulation settings. Within this line, several works directly map observations to actions using diffusion: Diffusion Policy (Chi et al.,

- 2023) established the basic recipe for action diffusion with both CNN and Transformer backbones, and DP3 (Ze et al., 2024b) extends the paradigm to point-cloud inputs to strengthen 3D spatial generalization. iDP3 (Ze et al., 2024a) extends DP3 for humanoid robots to learn from noisy human data. Mamba Policy (Cao et al., 2025b) improves DP3 by introducing a linear-complexity architecture Mamba (Gu & Dao, 2024; Dao & Gu, 2024). H3DP (Lu et al., 2025) explicitly incorporates hierarchical structures to strengthen the integration between visual features and action generation. MCDP (Cao et al., 2025a) integrates DP and DP3 via compositional diffusion to achieve enhanced performances. Temporal architectures tailored for planning appear in Motion Planning Diffusion (Carvalho et al., 2023), while design ablations highlight effective DiffusionTransformer components (DiT-Block Policy (Dasari et al., 2025)) and introduce attention-based conditioning for guided control (MTDP (Wang et al., 2025a)). Dexterous grasp synthesis is treated by DexDiffuser (Weng et al., 2024), which progressively denoises grasp configurations for multifingered hands. Moving beyond pooled embeddings, 3D Diffuser Actor (Ke et al., 2025) conditions on tokenized 3D scene representations, and R&D (Vosylius et al., 2024) presents a unified image–action formulation using ViT encoders. For long-horizon or structured problems, ALOHA Unleashed (Zhao et al., 2025) trains a Transformer with a diffusion loss for bi-manual skills; ChainedDiffuser (Xian & Gkanatsios, 2023) first predicts end-effector keyposes and then connects them with feasible trajectories; and HDP (Ma et al., 2024a) injects kinematics-aware priors to improve physical realism. Category-level robustness is pursued by S2-Diffusion (Yang et al., 2025) via visual foundation priors for spatial semantics and by C3DM (Saxena et al., 2023) through constrainedcontext conditioning with a fixation step to resist distractors. Diffusion has also been adopted as a behavior prior for planning: Diffuser (Janner et al., 2022) frames sampling-based planning as probabilistic behavior synthesis, while DTP (Fan et al., 2025) adds 2D trajectory guidance in a two-stage pipeline. Language- and object-centric formulations include StructDiffusion (Liu et al.,

- 2022), which fuses object-centric Transformers with diffusion under language goals, and PlayFusion (Chen et al., 2023), which uses discrete bottlenecks to acquire language-annotated skills. From the data side, ROSIE (Yu et al., 2023) exploits state-of-the-art text-to-image diffusion for aggressive augmentation, and Ha et al. (2023) broadens language conditioning across tasks. Finally, capacity can be increased without full monolithic scaling through expert routing: GSC (Mishra et al.,
- 2023) probabilistically chains learned skills with classifier-based guidance to satisfy constraints, and DBC (Chen et al., 2024) stabilizes learning by casting behavioral cloning of expert state–action pairs as diffusion-based modeling.

Beyond the core diffusion-based frameworks, a number of extensions have been proposed to improve efficiency, generalization, and adaptability in visuomotor policy learning. Streaming Diffusion Policy (Høeg et al., 2024) accelerates policy synthesis by producing partially denoised trajectories where each action may retain a different noise level, while Bidirectional Decoding (BID) (Liu et al.,

- 2024b) enables test-time inference through a combination of action chunking and closed-loop adaptation. Crossway Diffusion (Li et al., 2024b) introduces a specialized state decoder together with

an auxiliary self-supervised learning objective to reinforce policy robustness, and Equivariant Diffusion Policy (Wang et al., 2024a) exploits domain symmetries to achieve higher sample efficiency and better generalization in the denoising process. Complementary empirical work by Lin et al.

- (2024a) investigates data scaling effects in imitation learning at scale. Other approaches extend the representational capacity of policies, such as Imagination Policy (Huang et al., 2025a), which generates point cloud predictions of target states before converting them into executable actions, and Consistency Policy (Prasad et al., 2024), which distills faster visuomotor policies through a consistency regularization process. For fine-tuning, DPPO (Ren et al., 2024) provides a unified framework that integrates policy gradient techniques with diffusion policies in continuous control domains. Extensions to tactile-rich scenarios include the Reactive Diffusion Policy (Xue et al., 2025), which combines slow-fast visual–tactile imitation learning for contact-rich manipulation.

Several recent efforts also integrate reasoning and sequence modeling into the diffusion paradigm. The Unified Video Action Model (Li et al., 2025) jointly optimizes video prediction and action inference for accurate and efficient trajectory generation, while Chain-of-Action (CoA) (Zhang et al.,

- 2025b) explicitly reasons backward from task goals, producing coherent trajectories through an action-level chain-of-thought mechanism. Beyond diffusion, flow matching has emerged as a strong alternative: ManiFlow (Yan et al., 2025) combines flow matching with consistency training to synthesize dexterous actions in just one or two steps; Flow Matching Policy Gradients (McAllister et al.,

- 2025) embed flow matching directly into policy gradient algorithms for reinforcement learning; and VITA evolves latent visual states into latent actions under a flow matching framework; Steering Your Diffusion Policy (Wagenmaker et al., 2025) adapts behavior-cloning policies by performing reinforcement learning over the latent noise space, offering a flexible way to guide policy behavior without retraining from scratch. In addition to these extensions, recent works also explore safety and dynamics-aware guidance. DynaGuide (Du & Song, 2025) introduces a steering mechanism for diffusion policies by incorporating feedback from an external dynamics model directly into the denoising process, enabling more physically consistent action generation. Latent Policy Barrier (LPB) (Sun & Song, 2025), inspired by control barrier functions, formulates expert latent embeddings as implicit safety boundaries that distinguish in-distribution states from out-of-distribution ones, thereby enhancing robustness in visuomotor policy learning.

Together, these results indicate that lightweight diffusion policies augmented by stronger scene encoders, subgoal scaffolding, data augmentation, or MoE-style (Jiang et al., 2024) routing are competitive and data-efficient when embodiment and task distributions are relatively constrained.

Large-size LLM–based diffusion policies. At larger scales, diffusion modules are integrated with pre-trained vision–language-model (VLM) or LLM backbones or trained atop broad crossembodiment corpora, marrying semantic understanding with probabilistic action generation. Methods leveraging general data pre-training use foundation models to inject world knowledge and linguistic structure: MDT (Reuss et al., 2024) builds on CLIP (Radford et al., 2021) and Voltron (Karamcheti et al., 2023) for long-horizon manipulation with sparse language, enriching instructions via GPT-4 (Achiam et al., 2023); Ha et al. (2023) employs LLMs for high-level plan synthesis and success inference while delegating low-level control to diffusion policies; ROSIE (Yu et al., 2023) uses LLM-authored prompts to drive text-to-image diffusion for targeted data augmentation; and TinyVLA (Wen et al., 2025a) freezes a multimodal backbone and applies parameterefficient tuning („5% trainable) to produce actions efficiently. Compositional planning stacks further tighten the loop between language and diffusion: HiP (Ajay et al., 2023) composes expert models LLMs for task planning, video diffusion for trajectory proposals, and an inverse model for action mapping, while Plan Diffuser (Sharan et al., 2024) autoregressively emits textual subgoals with an LLM and translates them into visual subgoals via diffusion for downstream control. In parallel, robot data pre-training focuses on large, heterogeneous robot datasets to strengthen embodiment transfer. Octo (Team et al., 2024) aggregates 25 datasets from Open X-Embodiment (O’Neill et al.,

- 2024) and trains a Transformer with a diffusion head to map observation/task tokens to action tokens across embodiments; Diffusion-VLA (Wen et al., 2025b) pre-trains on Open X-Embodiment and DROID (Khazatsky et al., 2024) and adapts to tasks via LoRA (Hu et al., 2022); ChatVLA (Zhou et al., 2025) co-trains on robot and reasoning data with staged alignment and MoE routing to reduce task interference; RDT-1B (Liu et al., 2024a) specializes in fine-grained (including bimanual) skills by standardizing a unified action space over heterogeneous robots; and LAPA (Ye et al.,

- 2024) systematically studies cross-embodiment pre-training using BridgeV2 (Walke et al., 2023) and Open X-Embodiment (O’Neill et al., 2024). π0 (Black et al., 2024) couples a pre-trained

vision–language backbone with a flow-matching action expert for precise, smooth manipulation; Based on π0, π0.5 (Intelligence et al., 2025) uses co-training on heterogeneous tasks to enable broad generalization; Pertsch et al. (2025) and Driess et al. (2025) focused on accelerating the VLAs based on empirical studies; Enerverse (Huang et al., 2025b) and Video Prediction Policy (Hu et al.,

- 2024) use diffusion models for learning visual representations to improve scene understanding and subsequently enhance policy performance; HybridVLA (Liu et al., 2025a) unifies both the continuous nature of diffusion-based actions and the contextual reasoning of autoregression within a single LLM; GR00T N1 (Bjorck et al., 2025) is a dual-system (Figure, 2024; Cui et al., 2025) VLA for generalist humanoid robots, achieving state-of-the-art performances across multiple robot embodiments; Galaxea G0 (Jiang et al., 2025) also adpots the dual-sytem, coupling a VLM for multimodal planning and a VLA for low-level robot control; Agibot GO-1 (Bu et al., 2025) is a generalist policy that leverages latent action representations to maximize data utilization; Gemini Robotics family (Team et al., 2025) achieves generalized abilities in diverse tasks, including robot control, object detection, pointing, trajectory, and grasp prediction. In addition to using the diffusion policy as the action head, there are many excellent works that employ flow matching (Lipman et al., 2023; Liu, 2022) for action prediction: GraspVLA integrates autoregressive perception tasks and flow matching-based action generation into a unified Chain-of-Thought process; GR-3 (Cheang et al.,
- 2025) excels in understanding complex instructions with abstract concepts, generalizes effectively to novel objects and environments; WALL-OSS (Zhai et al., 2025) presents a coupled architecture, unifying instruction reasoning, subgoal decomposition, and fine-grained action synthesis.

Overall, these large-scale policies suggest a convergent recipe in which language models structure objectives and subgoals, diffusion processes synthesize trajectories and actions, and pre-training (on general or robot-centric corpora) supplies the semantic and embodiment priors required for robust generalization.

- L.3 NON-DIFFUSION-BASED MODELS IN ROBOT LEARNING

While diffusion-based approaches have recently attracted significant attention, a wide range of nondiffusion architectures continue to play a pivotal role in robot learning. These models typically leverage sequence modeling, spatial reasoning, and large-scale VLA systems to build versatile manipulation and navigation capabilities. Unlike diffusion policies, which rely on iterative denoising, non-diffusion frameworks often emphasize direct policy learning through MLP or transformers (Vaswani et al., 2017), hierarchical control, or data scaling strategies. In what follows, we summarize representative advances in manipulation and navigation, highlighting how non-diffusion models complement and extend the landscape of robot learning.

Manipulation. A major thread in non-diffusion visuomotor learning frames control as sequence modeling. ACT (Zhao et al., 2023) introduces a conditional encoder–decoder Transformer that predicts action sequences rather than single steps, attenuating compounding error over long horizons. Building on this idea, MT-ACT (Bharadhwaj et al., 2024) augments training with task semantics to learn a universal multi-task manipulator, while CogACT (Li et al., 2024a) couples a VLA backbone so that language-guided cognition and low-level motor control are optimized in concert. Chunking Causal Transformer (Zhang et al., 2025c) retains the ACT-style autoregressive policy but segments trajectories into chunks, improving stability and sample efficiency for long sequences. Beyond pure sequence decoding, several works enrich spatial grounding and 3D control: Act3D (Gervet & Xiao, 2023) is a language-conditioned Transformer for 6-DoF manipulation that outputs continuousresolution 3D action maps via adaptive 3D computation; ICRT (Fu et al., 2024) performs genuine in-context learning on a physical robot, leveraging a handful of contextual trajectories to execute unseen tasks without additional training; and Spatial Policy (Liu et al., 2025c) explicitly models scene geometry so that visual predictions align with executable end-effector motions. A complementary line scales VLA systems with data and hierarchy: RT-1 (Brohan et al., 2023) demonstrates that Transformers trained on large, diverse robot datasets yield strong generalists; RT-2 (Zitkovich et al., 2023) transfers web-derived vision–language knowledge into control; RT-X (O’Neill et al.,

- 2024) shows that pretraining on large-scale OXE data can set new performance bars, underscoring the value of data scale; and RT-H (Belkhale et al., 2024) inserts a language-motion layer that bridges high-level instructions and low-level actions through an explicit hierarchy. Practical systemization and modality breadth are advanced by Beyond Sight (Jones et al., 2025) (Octo-style finetuning to adapt generalist visuomotor policies to heterogeneous sensors), OpenVLA (Kim et al., 2025) (fully

- released training/testing recipes), and RoboVLM (Li et al., 2024c) (a design study distilling the most consequential choices in VLA pipelines). Finally, emerging embodied models lift perception and coordination to 3D and dexterous settings: 3D-VLA (Zhen et al., 2024) links 3D perception, reasoning, and action via a generative world model; Bi-VLA (Gbagbe et al., 2024) targets coordinated bimanual manipulation; LEO (Huang et al., 2024) acts as a multimodal generalist capable of perceiving, grounding, reasoning, planning, and acting in 3D environments; SpatialBot (Cai et al.,
- 2025) strengthens spatial understanding by fusing RGB with depth; Lift3D (Jia et al., 2024) elevates 2D foundation features into robust 3D manipulation representations; and RoboDual (Bu et al., 2024) unifies generalist breadth with specialist precision in a synergistic dual-policy framework.

Further advances focus on constraint-driven representations for manipulation. Relational Keypoint Constraints (ReKep) (Huang et al., 2025c) define visually grounded constraints as Python functions that map sets of 3D keypoints in the scene to numerical costs, providing a flexible interface for encoding task-specific relations. VosPoser (Huang et al., 2023) leverages large language models to extract affordances and constraints from natural language, composing 3D value maps in the observation space that guide robotic interactions in a structured manner.

Together, these works illustrate that non-diffusion architectures, particularly sequence models and VLA systems, achieve strong manipulation generalization through long-horizon decoding, explicit spatial grounding, data scaling, and modular hierarchy.

Navigation. For locomotion and navigation, non-diffusion approaches similarly exploit hierarchy, distillation, and language grounding. Cheng et al. (2024b) develop extreme legged parkour by first training a teacher with reinforcement learning and then distilling its competence into a student policy that runs purely on onboard depth, enabling agile behaviors in the wild. Mobility VLA (Chiang et al., 2024) adopts a hierarchical design: long-context VLMs provide scene understanding and commonsense reasoning at the high level, while a robust low-level navigator follows a topological graph to execute the plan. NaVid (Zhang et al., 2024b) turns streaming RGB video and a natural-language instruction into a sequence of textual action directives that a robot can carry out, emphasizing language-as-action for purely visual inputs. NaVILA (Cheng et al., 2024a) extends this idea to legged visual–language-navigation (VLN) with two levels of control: a finetuned VLM outputs mid-level language actions (e.g.,“turn right 30”), and a learned visual locomotion controller faithfully executes those commands. These systems highlight a recurring pattern in non-diffusion navigation: decompose high-level intent into compact linguistic subgoals, pair them with robust low-level policies for accurate robot control.

- M FUTURE WORK

This work opens several avenues for future exploration. On the methodological side, a key direction is to move beyond fixed test-time weight discretization. More adaptive weighting strategies could be developed, such as reinforcement learning or gradient-based meta-optimization, to automatically adjust convex weights across tasks and environments. Another natural extension is to scale from dual-policy to multi-policy composition. Since na¨ıvely increasing the number of composed policies incurs a high computational cost, future work may explore feature sharing mechanisms or compact latent representations to enable efficient integration. Finally, the design of stronger composition operators remains an open challenge. Our initial results with superdiffusion highlight its potential, but more efficient variants, as well as extensions that integrate with flow-based models, could further amplify policy performance.

At a broader level, the principle of policy composition can potentially extend beyond diffusion-based policies. The same compositional framework could be applied to diverse policy classes and architectures, enabling modular integration of heterogeneous skills. Moreover, while our experiments focus on robotic VA and VLA in manipulation tasks, we anticipate a broader impact in related domains. For instance, vision-language-navigation (VLN) tasks, such as some successful state-ofthe-art methods TrackVLA (Wang et al., 2025b) and LOVON (Peng et al., 2025), may also benefit from compositional strategies to enhance generalization and robustness. Exploring these directions would further validate GPC as a general paradigm for leveraging pre-trained models in complex sequential decision-making domains.

