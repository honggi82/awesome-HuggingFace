# Learning Dynamics in Continual Pre-Training for Large Language Models

Xingjin Wang12 Howe Tissue Lu Wang3 Linjing Li12 Daniel Dajun Zeng12

arXiv:2505.07796v2[cs.CL]19Jun2025

## Abstract

Continual Pre-Training (CPT) is a popular and effective method for applying strong foundation models to specific downstream tasks. In this work, we explore the learning dynamics throughout the CPT process for large language models. We specifically focus on how general and downstream domain performance evolves at each training step, with performance measured by validation losses. We observe that the CPT loss curve fundamentally characterizes a transition from an initial pre-training trajectory to a new, domainspecific one, conceptualized as a shift between two hidden loss curves. This transition can be described by decoupling the effects of distribution shift and learning rate annealing. We derive a CPT scaling law that combines these two factors, enabling the prediction of loss at any (continual) training step and across various learning rate schedules. Our formulation presents a comprehensive understanding of several critical factors in CPT, including loss potential, peak learning rate, training steps, and replay ratio. Moreover, our approach can be adapted to optimize training hyper-parameters for different CPT goals, such as balancing general and domain-specific performance. Extensive experiments demonstrate that our scaling law holds across various CPT datasets and hyper-parameters.

## 1. Introduction

In recent years, large language models (LLMs) have exhibited versatile abilities and garnered significant academic and industrial attention (Dubey et al., 2024; OpenAI, 2023).

1School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China 2State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences, Beijing, China 3RitzzAI. lwzzfzl@gmail.com. Correspondence to: Howe Tissue (project lead) <h-sun20@tsinghua.org.cn>.

Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

Continual Pre-Training (CPT) of LLMs aims to enhance their abilities in specific downstream domains (e.g. coding, finance, math) while mitigating the substantial costs associated with re-training (Chen et al., 2023a; C¸a˘gatay Yıldız et al., 2024; Ibrahim et al., 2024).

CPT primarily involves a trade-off between performance on general and downstream domains. It is widely observed that improvements on downstream tasks may come at the expense of degrading performance on general domain tasks, a phenomenon known as catastrophic forgetting (French, 1999; Gupta et al., 2023). Recently, some scaling laws have been proposed for CPT scenarios. For example, Hernandez et al. (2021b) and Barnett (2024) discovered a law describing how data transfer effectiveness scales with fine-tuning dataset size and model size. Que et al. (2024) and Gu et al. (2024) proposed a law to find the optimal replay ratio to balance general and downstream performances.

However, very few studies have attempted to quantitatively describe the learning dynamics of CPT, particularly how performance varies on general and downstream domains throughout the CPT process. We have two primary research questions (RQs): (1) Can we derive an accurate law describing the influence of as many variables as possible on the final CPT performance? (2) Can we trace the performance of LLMs throughout the entire CPT process, rather than only the final performance? Studying the first RQ will help researchers investigate various factors that affect CPT performance and facilitate hyper-parameters optimization through prediction; studying the second RQ will help the community understand the learning dynamics of LLMs at each step of the CPT process, providing deeper insights and theoretical guidance for subsequent CPT research.

Following previous works (Gupta et al., 2023; Ibrahim et al., 2024; Que et al., 2024), we trace performance changes using validation losses on corresponding domains. We find that the CPT loss curve acts as a transfer curve and can be described by decoupling the effects of distribution shift and learning rate (LR) annealing. Specifically, the distribution shift between the pre-training (PT) and CPT data leads to a deviation in the loss curve, while LR annealing results in a loss decrease in both the PT and CPT phases. By analyzing various loss curves, we discover a CPT scaling law that integrates these two factors, enabling accurate prediction of

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

Transfer Point

0.00

0 10000 20000 30000 40000 50000 60000

Step

(a) Constant PT and CPT LRS.

- 0.25

- 0.50

0.75

1.00

- 1.25

1.50

1.75

2.00

4LearningRate×10

Pre-Training LRS

Continual Pre-Training LRS

Annealing Point

Transfer Point

(d) WSD PT and CPT LRS.

0 10000 20000 30000 40000 50000 60000

Step

3.1

3.2

3.3

3.4

3.5

3.6

3.7

ValidationLossDpt

Transfer Point

Distribution Shift

Pre-Traning with Dpt

Hidden Pre-Traning with Dpt

Continual Pre-Training

Pre-Traning with Dcpt

(e) Dpt (FineWeb) Validation Loss.

0 10000 20000 30000 40000 50000 60000

Step

2.4

2.5

2.6

2.7

2.8

2.9

3.0

3.1

3.2

3.3

3.4

ValidationLossDcpt

Transfer Point

Distribution Shift

Pre-Traning with Dpt

Hidden Pre-Traning with Dpt

Continual Pre-Training

Pre-Traning with Dcpt

(f) Dcpt (Knowledge Pile) Validation Loss.

- Figure 1. CPT loss curves under different learning rate schedules (LRS): constant (a-c) and warmup-stable-decay (WSD) (Hu et al., 2024) (d-f). The CPT loss curve acts as a transfer curve from the hidden PT curve trained on Dpt (Blue dashed) to the hidden PT curve trained on Dcpt (Orange dashed). The transfer curve converges to the hidden PT curve trained on Dcpt.

losses throughout the entire CPT phase.

Our proposed scaling law provides a comprehensive model of how key variables affect the training dynamics of CPT, such as loss potential (defined in section 3.3), peak LR, training steps, and replay ratio. We demonstrate how these variables jointly affect model performance at each CPT step, and how to optimize these hyper-parameters for better CPT performance. By applying our scaling law, several valuable conclusions emerge. For example: (1) PT models with higher loss potential can better adapt to downstream domains in CPT; (2) The performance degradation on the PT domain during the CPT phase is inevitable if the turning length is infinitely large, which implies that the PT model is adequately trained or the distribution shift between the PT and CPT data is very large; (3) For specific CPT goals, like balancing performance between the PT and CPT domains, or optimizing out-of-domain performance, our scaling law can predict the optimal training hyper-parameters such as the loss potential, peak LR, and PT dataset replay ratio.

2. Pilot Observation

- 2.1. Task Formulation

0.00

0 10000 20000 30000 40000 50000 60000

Step

3.7

Pre-Traning with Dpt

Hidden Pre-Traning with Dpt

ValidationLossDpt

3.6

Continual Pre-Training

Pre-Traning with Dcpt

3.5

3.4

Distribution Shift

3.3

Transfer Point

3.2

3.1

0 10000 20000 30000 40000 50000 60000

Step

(b) Dpt (FineWeb) Validation Loss.

Pre-Traning with Dpt

3.4

Hidden Pre-Traning with Dpt

ValidationLossDcpt

Continual Pre-Training

3.3

Pre-Traning with Dcpt

Transfer Point

3.2

3.1

3.0

2.9

Distribution Shift

2.8

2.7

2.6

2.5

0 10000 20000 30000 40000 50000 60000

Step

(c) Dcpt (Knowledge Pile) Validation Loss.

Experimental Setup. Our main experiments employ LLaMA-like models (Dubey et al., 2024) with 106M to

- 1.7B non-embedding parameters. We use FineWeb (Penedo et al., 2024) as Dpt and Knowledge-Pile (Fei et al., 2024) as Dcpt. We leverage different LRS in the PT and CPT phases (see Fig. 1). More details are provided in Appendix B.

Observation. As observed in previous studies (Ibrahim et al., 2024; Gupta et al., 2023), during the CPT process, the Dpt validation loss tends to increase (Fig. 1b and Fig. 1e), whereas the Dcpt validation loss decreases (Fig. 1c and Fig. 1f). Moreover, in both PT and CPT phases, the loss curve is significantly influenced by the LRS. For example, the loss decreases rapidly when the LR anneals, which is observed in our prior work (Tissue et al., 2024).

- 2.2. CPT Transfer Loss Curve

To enhance our understanding of the CPT training dynamics, we train two additional loss curves: the hidden PT curve trained on Dpt and the hidden PT curve trained on Dcpt.

We investigate the dynamics of performance in both general and downstream domains during the CPT process. Following previous works (Ibrahim et al., 2024; Que et al., 2024; Gu et al., 2024; Hernandez et al., 2021a), we assess model performance by examining the validation loss on the PT dataset Dpt and the CPT dataset Dcpt.

Hidden PT Curve trained on Dpt. This curve represents the loss when the model is consistently pre-trained using Dpt with the same LRS as used in the CPT phase.

Hidden PT Curve trained on Dcpt. This curve depicts the loss when the model is trained from scratch on Dcpt, while adhering to the same training setups (such as LRS) as those applied in the PT and CPT phases.

Pre-training

3.6

Hidden Pre-training

ValidationLossDpt

CPT in 10K Steps CPT in 20K Steps CPT in 30K Steps

3.4

Distribution Shift

3.2

0.3

DistributionShift

0.2

Power: 0.368(1 (0.008t + 1) 0.173) R2=0.994 Exp: 0.195(1 e0.001t) R2=0.926

3.0

0.1

0.0

CPT Step (t)

2.8

0 10000 20000 30000 40000 50000 60000

Step

(a) Dpt (FineWeb) validation loss shift.

4.0

0.6

Pre-training

DistributionShift

Hidden Pre-training

3.8

ValidationLossDcpt

0.4

CPT in 10K Steps CPT in 20K Steps CPT in 30K Steps

Power: 0.702(1 (1.345t + 1) 0.131) R2=0.985 Exp: 0.485(1 e0.007t) R2=0.801

3.6

0.2

0.0

3.4

CPT Step (t)

3.2

3.0

Distribution Shift

2.8

2.6

0 10000 20000 30000 40000 50000 60000

Step

(b) Dcpt (Knowledge Pile) validation loss shift.

- Figure 2. The transfer loss curve in Dpt and Dcpt validation sets for different transfer starting points with constant LRS. We find that the distribution shift term is independent of the transfer starting points and adheres to a power-law form.

Transfer Curve. As shown in Fig. 1, the CPT loss curve acts as a transfer curve between these two hidden PT curves; i.e., the CPT loss deviates from the hidden PT curve trained on Dpt and converges towards the hidden PT curve trained on Dcpt. The discrepancy between the transfer loss curve and the hidden PT curve trained on Dpt is called distribution shift. As the number of CPT steps approaches infinity, the CPT loss is expected to converge to the hidden PT curve trained on Dcpt.

Finding 1. The process of CPT is how the loss curve transitions from the hidden PT curve trained on Dpt to the hidden PT curve trained on Dcpt.

- 3. Continual Learning Dynamics Law

where t denotes the CPT step, and S1pt(S2pt) and S1cpt(S2cpt) are the forward (annealing) areas at the PT and CPT stages, respectively.

#### 3.2. Distribution Shift Term

The distribution shift term describes the deviations from the hidden PT curve trained on Dpt. This shift reflects the distributional distance between Dpt and Dcpt. Many studies (Ibrahim et al., 2024; Wang et al., 2024; Parmar et al., 2024) have highlighted the impact of LRS at the CPT stage, implying that this shift should be also affected by the LRS. We first analyze the form of the distribution shift term with a constant LR to isolate the effects of LRS, then we incorporate the forward area into the equation to accurately describe the distribution shift term for different LRS.

We quantitatively analyze the transfer curve by modeling the effects of LR annealing and distribution shift.

#### 3.1. LR Annealing

Without data transfer, the CPT loss curve would follow the trajectory of the hidden PT curve trained on Dpt. Tissue et al. (2024) introduced a scaling law to describe the loss dynamics at each step t as affected by LR annealing:

L(t) = L0 + A · S1−α − C · S2, (1)

where the forward area S1 = ti=1 ηi is the summed LR, and the annealing area S2 = ti=1 ik=1 (ηk−1 − ηk) · λi−k is a term affected by LR annealing. L0,A,C,α are constant positive parameters to be fitted. λ = 0.999 is a hyper-parameter related to the momentum term.

The loss in the CPT process without distribution shift (denoted as Lbase(t)) follows this law, i.e.,

Lbase(t) = L0+A·(S1pt+S1cpt)−α−C·(S2pt+S2cpt), (2)

Constant LRS. We first use a constant LR in both PT and CPT phases. To study the relationship between distribution shift and the PT model state, we continually pre-train the model starting from different transfer points. As shown in Fig. 2, these distribution shift terms tend to overlap regardless of the transfer starting point. This overlap suggests that the distribution shift term is independent of transfer starting points or PT model checkpoints.

We compare to fit the distribution shift term using exponential and power-law forms, and find the best fit to be ∆L(t) = B · (1 − (E · t + 1)−β). We do not adopt the simple power-law form ∆L(t) = B · t−β to ensure that ∆L(0) = 0. We leverage this equation to fit the transfer loss curve of both Dpt and Dcpt validation sets, as shown in Fig. 2.

Other LRS. When considering the effect of LRS, we find that the LR values, i.e., the forward area in Eq. 1, significantly affects the distribution shift term. The smaller forward area in the CPT results in a smaller distribution

3.8

WSD Pre-Training Truth Loss

WSD Pre-Training Truth Loss

2.00

3.4

4LearningRate×10

WSD Continual Pre-Training Truth Loss

WSD Continual Pre-Training Truth Loss

ValidationLossDcpt

3.7

ValidationLossDpt

WSD Pre-Training Fitted Loss

WSD Pre-Training Fitted Loss

3.3

1.75

WSD Continual Pre-Training Fitted Loss

WSD Continual Pre-Training Fitted Loss

3.2

3.6

1.50

3.1

L = 3.057 + 0.490(S1pt + S1cpt) 0.497 + 0.366(1 (28S1cpt + 1) 0.189) 0.299S2pt 0.298S2cpt

1.25

3.5

L = 3.024 + 0.412(S1pt + S1cpt) 0.677 0.666(1 (5662S1cpt + 1) 0.146) 0.278S2pt 0.324S2cpt Distribution Shift

3.0

Peak Point

1.00

3.4

2.9

+ LR Re-Warmup

0.75

2.8

3.3

Distribution Shift + LR Re-Warmup

0.50

2.7

Pre-Training LRS

3.2

0.25

2.6

Continual Pre-Training LRS

0.00

3.1

2.5

0 10000 20000 30000 40000 50000 60000

10000 20000 30000 40000 50000 60000

10000 20000 30000 40000 50000 60000

Step

Step

Step

(a) WSD PT and CPT LRS.

(b) Dpt (FineWeb) Validation Loss.

(c) Dcpt (Knowledge Pile) Validation Loss.

- 0.00

0.25

0.50

0.75

1.00

1.25

1.50

- 1.75

- 2.00

4LearningRate×10

Pre-Training LRS

Continual Pre-Training LRS

(d) Cosine PT and CPT LRS.

10000 20000 30000 40000 50000 60000

Step

3.1

3.2

3.3

3.4

3.5

3.6

3.7

3.8

ValidationLossDpt

L = 3.057 + 0.490(S1pt + S1cpt) 0.497 + 0.366(1 (28S1cpt + 1) 0.189) 0.299S2pt 0.298S2cpt Peak Point

Distribution Shift + LR Re-Warmup

Cosine Pre-Training Truth Loss

Cosine Continual Pre-Training Truth Loss

Cosine Pre-Training Fitted Loss

Cosine Continual Pre-Training Fitted Loss

(e) Dpt (FineWeb) Validation Loss.

10000 20000 30000 40000 50000 60000

Step

2.5

2.6

2.7

2.8

2.9

3.0

3.1

3.2

3.3

3.4

ValidationLossDcpt

L = 3.024 + 0.412(S1pt + S1cpt) 0.677 0.666(1 (5662S1cpt + 1) 0.146) 0.278S2pt 0.324S2cpt

Distribution Shift + LR Re-Warmup

Cosine Pre-Training Truth Loss

Cosine Continual Pre-Training Truth Loss

Cosine Pre-Training Fitted Loss

Cosine Continual Pre-Training Fitted Loss

(f) Dcpt (Knowledge Pile) Validation Loss.

- Figure 3. Using Eq. 4 to fit all PT and CPT loss curves with different LRS (WSD and Cosine). For Dpt validation sets, all loss curves (b and e) are described by the same equation; similarly, for Dcpt validation sets, all loss curves (c and f) follow the same equation.

shift, as shown in different transfer curves in Fig. 1b vs. 1e (or Fig. 1c vs. 1f). Hence, following Tissue et al. (2024), we replace the training steps t with the forward area S1cpt in the CPT phase:

∆L(t) = B · (1 − (1 + E · S1cpt)−β), (3)

which instead adopts S1cpt to represent the training amount in CPT stage, considering the impact of LR values.

- 3.3. Final Transfer Curve

0 10000 20000 30000 40000 50000 60000

Step

across different LRS throughout the training process. We also use the fitted equation to predict loss curves of other LRS, and the prediction accurately matches the observation (see Fig. 10). Furthermore, the batch size and sequence length may change in the CPT phase. However, our scaling law equation remains adaptable to these hyper-parameter changes, as demonstrated in Appendix F.

Finding 2. The CPT loss curve can be decomposed into a hidden PT curve trained on Dpt and a distribution shift term. The hidden PT curve trained on Dpt is formalized as a scaling law with LR annealing, whereas the distribution shift term is independent of transfer starting points and adheres to a power-law form.

We combine the effect of LR annealing (Eq. 2) and distribution shift (Eq. 3) to get the equation for the CPT loss:

L(t) = Lbase(t) + ∆L(t)

= L0 + A · S1pt + S1cpt −α −C1 · S2pt − C2 · S2cpt

(4)

Scaling law with LR annealing

Transfer Loss Surface. To better understand our formulation, we follow Tissue et al. (2024) to view the loss surface of LLMs as a slide-like transition between surfaces in Fig. 4. The CPT process transitions from one surface to another following a power-law form. A larger distributional distance between Dpt and Dcpt leads to a steeper slope of the transfer surface, and thus a sharper increase in the Dpt loss. When the LR anneals, the amplitude of the oscillation on the loss surface decreases, and thus the loss also decreases. In the annealing view, we term the “height” of the current model state as its loss potential. We use this concept to capture the potential for future loss drop via LR annealing. Quantitatively, we can define loss potential as the ratio of the final annealed LR of the PT phase to the peak learning rate in the PT phase.

##### + B · 1 − 1 + E · S1cpt −β

Power-law distribution shift

We adopt different coefficients C1 and C2 for S2pt and S2cpt because the distributions of Dpt and Dcpt are different, and thus result in different annealing effects.

Our equation can predict the loss at any step with any LRS during both the PT and CPT phases. We conduct experiments utilizing the widely adopted WSD (Hu et al., 2024) and cosine (Loshchilov & Hutter, 2016) LRS in the PT and CPT phases (see Fig. 3a and Fig. 3d). We use Eq. 4 to fit all loss curves on the Dpt and Dcpt validation sets. As illustrated in the middle and right panels of Fig. 3, our equation successfully captures the trends in loss variations

the concept of loss potential introduced in section 3.3 to describe the degree of annealing for PT models. Specifically, a PT model trained without annealing has a high loss potential, while a PT model that anneals to a zero LR value has a low loss potential. We investigate the impact of loss potential on CPT under two different experimental settings: without or with LR re-warmup.

e

c

urfa

S

s

s

o

L

Dpt

Annealing Area Direction

W/o Re-warmup. In this setting, we set the initial LR for CPT as the final LR in PT and linearly anneal the LR for CPT to zero. We conduct experiments using PT models with different loss potentials (Fig. 5a). As shown in Fig. 5b, models with higher loss potential achieve lower final losses on Dcpt. This observation matches the prediction made by our CPT scaling law (Fig. 5c). We also utilize our equation to predict the final loss across various CPT steps, confirming that this trend persists in different settings.

Pre-Training Constant LR Continual Pre-Training

Annealing LR

FowardAreaDirection

(a) Loss surface of Dpt as a transfer slide.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

LossPotential

FowardLoss

Foward Area Direction

Annealing Area Direction

With Re-warmup. A common practice for CPT is to linearly re-warmup the LR from zero to a certain value, such as 10% of the peak LR in PT, before annealing it to zero (Fig. 5d). As shown in Fig. 5e and Fig. 5f, models with high loss potential consistently achieve lower final losses.

(b) Forward view.

(c) Annealing view.

- Figure 4. The loss surface of the CPT process and two directional views.

#### 3.4. Extension to Model Size and Replay Ratio

We attempt to incorporate model size N into our CPT scaling law by analyzing the effect of N on both the LR annealing and distribution shift term. Our experiments show that the distribution shift terms remains unchanged across different model sizes when other settings are fixed (see details in Appendix E). Therefore, we can directly follow Tissue et al. (2024) to integrate an N-related term and use our scaling law to fit and predict CPT loss curves for different model sizes. More discussion is provided in Appendix E.

We also integrate the replay ratio into our scaling law since replaying some data from Dpt is a common practice in CPT. Our experiments show that the replay ratio influences the distribution shift term in an exponential manner. By adding a single replay ratio related term, our scaling law can predict the entire training dynamic for different replay ratios, while previous studies (Que et al., 2024) can only predict the final loss. More details are given in Appendix H.

We can use our CPT scaling law (Eq. 4) to analyze the impact of loss potentials. Specifically, as the annealing coefficient C2 > C1 often holds for Dcpt, then allocating a larger annealing area in the CPT phase, i.e., a larger S2cpt, facilitates a lower loss. Moreover, models with higher loss potential have larger forward areas S1pt and S1cpt, which further contribute to a lower loss. Therefore, PT models with high loss potential usually lead to lower Dcpt loss. This conclusion is also validated in previous works (Wang et al., 2024).

Finding 3. PT models with higher loss potential consistently achieve lower Dcpt validation losses. Hence, we advocate that when releasing opensource models, it is beneficial to release a high loss potential version to facilitate downstream tasks.

#### 4.2. Replay Ratio

## 4. Factor Analyses and Applications

In this section, we analyze various factors for CPT and apply our scaling law to provide insights into these factors.

#### 4.1. Loss Potential

Most PT models are trained by annealing to a minimum LR for lower PT losses. However, the optimal PT model for CPT is not necessarily a fully annealed model. We use

The distributional distance between Dpt and Dcpt significantly influences the distribution shift term in Eq. 4. As shown in Fig. 6a, a more distinct Dcpt, Pile of Law (Henderson* et al., 2022), leads to a sharper transfer curve than a more similar Dcpt (Knowledge Pile (Fei et al., 2024)).

In CPT, it is a common practice to mix Dpt into Dcpt based on a certain replay ratio to mitigate the increase of validation loss on Dpt. The replay ratio plays a critical role in adjusting the distributional distance between Dpt and Dcpt since Dcpt

10% Loss Potential 30% Loss Potential 50% Loss Potential 100% Loss Potential

4LearningRate×10

2.0

PT End

1.5

1.0

0.5

0.0

PT Total

Step

(a) CPT with different loss potentials (w/o re-warmup setting).

0% Loss Potential

2.00

4LearningRate×10

10% Loss Potential 50% Loss Potential 100% Loss Potential

1.75

PT End

1.50

1.25

1.00

0.75

0.50

0.25

0.00

PT Total

Step

(d) CPT with different loss potentials (w/ re-warmup).

2.70

10% Loss Potenial Truth Loss 30% Loss Potenial Truth Loss 50% Loss Potenial Truth Loss 100% Loss Potenial Truth Loss

2.68

ValidationLossDcpt

2.66

2.64

2.62

2.60

2.58

2.56

10000 12000 14000 16000 18000 20000

CPT Step

(b) Dcpt true loss vs. CPT step for different loss potentials (w/o re-warmup setting).

2.67

0% Loss Potenial Truth Loss

10% Loss Potenial Truth Loss 50% Loss Potenial Truth Loss 100% Loss Potenial Truth Loss

ValidationLossDcpt

2.66

2.65

2.64

2.63

10000 12000 14000 16000 18000 20000

CPT Step

(e) Dcpt true loss vs. CPT step for different loss potentials (w/ re-warmup setting) .

2.66

20K CPT Steps 30K CPT Steps 40K CPT Steps 50K CPT Steps

2.64

ValidationLossDcpt

2.62

2.60

2.58

2.56

2.54

2.52

20 40 60 80 100

Loss Potential %

(c) Dcpt predicted loss vs. loss potentials for different CPT steps (w/o re-warmup setting).

2.67

20K CPT Steps 30K CPT Steps 40K CPT Steps 50K CPT Steps

2.66

ValidationLossDcpt

2.65

2.64

2.63

2.62

2.61

2.60

2.59

0 20 40 60 80 100

Loss Potential %

(f) Dcpt predicted loss vs. loss potentials for different CPT steps (w/ re-warmup setting).

- Figure 5. The impact of the loss potential of PT models. We illustrate the true loss of models with different loss potential in the middle panel. We utilize Eq. 4 to predict the losses of these models across different training steps in the right panel. The red star (⋆) refers to the models that achieve the lowest Dcpt validation loss given the number of CPT steps.

is modified to approach Dpt 1. Results in Fig. 6b and Fig. 6c indicate that higher replay ratios lead to smaller distribution shifts and thus effectively decelerate the deviation from Dpt. Quantitatively, we find that the replay ratio influences the distribution shift term based on an exponential form, which is elaborated in Appendix H.

#### 4.3. Peak LR

In real scenarios, choosing an appropriate peak LR for rewarmup is important for CPT. Different peak LRs affect the Dpt and Dcpt validation loss. We leverage Eq. 4 to predict the final loss of different peak LRs. Specifically, we assume the PT model is trained using the WSD LRS. As shown in Fig. 7a and Fig. 7b, a high peak LR in the CPT phase accelerates the decrease of the Dcpt validation loss while leading to an increase of the Dpt validation loss.

#### 4.4. CPT Training Steps

The number of CPT training steps is also an important hyperparameter. A general observation is that more training steps lead to lower Dcpt validation loss. However, the Dpt validation loss may exhibit three different patterns based on the state of the PT model and the distributional distance between Dpt and Dcpt: (1) a continuous rise; (2) an initial

1It is important to note that the Dcpt undergoes modifications in the presence of replays. For instance, if we mix 0.1 FineWeb with 0.9 KP, the actual Dcpt distribution is 0.1 FineWeb with 0.9 KP, not 1.0 KP.

rise followed by a decline that does not return to the original loss value; or (3) an initial rise followed by a decline that goes below the original loss value.

As shown in Fig. 7c, we define the critical point (indicated by the blue dashed line) as the convergence value of the Dpt loss on the hidden PT curve trained on Dcpt. When CPT occurs before this critical point, the Dpt loss will first rise and then decline. The final loss may or may not be lower than the original loss. The minimum training steps required to return to the initial loss value are designated as the turning length. Conversely, if CPT occurs after the critical point, achieving a lower Dpt loss than the initial value becomes unattainable, regardless of how many steps we train.

Finding 4. Inadequate pre-training or weak distribution shift can result in lower Dpt loss values after sufficient CPT steps compared to the PT model.

Otherwise, we are unlikely to achieve a lower Dpt loss than the PT model, regardless of how many CPT steps we train. In this situation, more training often leads to degraded general performance.

## 5. Balance Between Dpt and Dcpt Loss

Validation losses on Dpt and Dcpt typically exhibit a tradeoff in the CPT process. Balancing these losses is critical for optimizing the overall performance of the model during CPT. We define the increase in Dpt loss as ∆LD

and the

pt

- 3.4

3.6

3.8

4.0

4.2

- 4.4

FineWebLoss

Weak Distribution Shift

Strong Distribution Shift

DCPT dataset = Knowledge Pile

DCPT dataset = Pile of Law

(a) The difference in distribution shift for different Dcpt datasets.

10000 20000 30000 40000 50000 60000

Step

3.20

3.25

3.30

3.35

3.40

3.45

3.50

3.55

FineWebLoss

Strong Distribution Shift

Weak Distribution Shift

- KP:Fineweb=1:0 (0% Replay)

- KP:Fineweb=2:1 (33% Replay)

- KP:Fineweb=1:1 (50% Replay)

- KP:Fineweb=1:2 (67% Replay)

KP:Fineweb=0:1 (100% Replay)

(b) The distribution shift on the Dpt validation set for different replay ratios.

10000 20000 30000 40000 50000 60000

Step

2.6

2.8

3.0

3.2

3.4

KnowledgePileLoss

Strong Distribution Shift

Weak Distribution Shift

- KP:Fineweb=1:0 (0% Replay)

- KP:Fineweb=2:1 (33% Replay)

- KP:Fineweb=1:1 (50% Replay)

- KP:Fineweb=1:2 (67% Replay)

KP:Fineweb=0:1 (100% Replay)

(c) The distribution shift on the Dcpt validation set for different replay ratios.

Figure 6. We compare the distribution shift for different distributional distances between the Dcpt and Dpt datasets. Additionally, we examine the impact of different replay ratios on the distribution shifts within both the Dcpt and Dpt validation sets.

0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

Max learning rate 1e 4

3.30

3.31

3.32

3.33

3.34

3.35

3.36

3.37

ValidationLossDpt

20K CPT Steps 40K CPT Steps 60K CPT Steps 80K CPT Steps

(a) Dpt predicted loss vs. peak LRs for different CPT steps.

0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

Max learning rate 1e 4

2.52

2.54

2.56

2.58

2.60

2.62

2.64

2.66

ValidationLossDcpt

20K CPT Steps 40K CPT Steps 60K CPT Steps 80K CPT Steps

(b) Dcpt predicted loss vs. peak LRs for different CPT steps.

Step

ValidationLossDpt

Turning Length

Critical Point

Pre-Critical

Post-Critical

Pre-Training Line

Hidden PT Line on Dcpt

CPT from Pre-Critical

CPT from Post-Critical

L0 of Hidden PT Line on Dcpt

(c) Critical point and turning length in Dpt validation loss.

Figure 7. (a)-(b) The effect of the peak LR. We utilize Eq. 4 to predict the final loss for different peak LRs. (c) The effect of different CPT steps. We show the critical point and turning length in the Dpt validation loss.

decrease in Dcpt loss as ∆LD

cpt

. To balance the loss of Dpt and Dcpt validation sets, we assign normalized balance coefficient to different validation sets:

min

S1cpt,S2cpt

λ1∆LD

pt

+ λ2∆LD

cpt

s.t. λ1 + λ2 = 1

(5)

where λ1 and λ2 are coefficients that should be set based on our prior knowledge of the relative importance of general and downstream performance.

- 5.1. Optimal Hyper-Parameters

3.2

3.0

10000 20000 30000 40000 50000 60000

Step

Given the different coefficients λ1 and λ2, there exist some optimal CPT hyper-parameters.

Loss Potential. Fig. 8a shows the optimal loss potential for different values of λ1. It can be observed that a small λ1 corresponds to a large optimal loss potential. This makes sense since a small λ1 means that the final loss is dominated by the Dcpt loss, and thus it is necessary to reserve sufficient loss potential for downstream domains.

Peak LR. We can also predict the optimal peak LR in the CPT process when λ1 is given (Fig. 8b). A larger λ1 suggests a preference for minimizing the increase in Dpt loss, thereby necessitating a lower peak LR.

Replay Ratio. Based on our scaling law with replay ratio Eq. 8, we can determine the optimal replay ratio for each λ1 (Fig. 8c). The same distribution line (dashed line) in Fig. 8c indicates that the optimal replay ratio should be the same as the target weight λ1 if we initialize the CPT model randomly rather than from a pre-trained model. Instead, in practice, the optimal replay ratio shifts because the PT model has already been trained on Dpt, which causes the curve to deviate and exhibit a wave pattern.

CPT Training Steps. As shown in Fig. 13, we can get different turning lengths for different values of λ1. When λ1 is small, the Dcpt loss predominates the composite loss λ1LD

, which consistently remains below the initial value. Conversely, with a moderate λ1, there exists a specific step that makes the composite loss equals the inital loss. For a large λ1, the composite loss is always higher than the initial loss, which means that CPT is not suitable any more in this situation.

+ λ2LD

pt

cpt

#### 5.2. Out-of-Domain Validation Set

Note that our CPT scaling law is designated to predict losses on Dpt and Dcpt validation sets, while it is not directly applicable to the out-of-domain (OOD) validation set Dood. Inspired by previous works (Ye et al., 2024; Zhang et al., 2025) that the OOD validation loss can be represented as a

OptimalLossPotential%

100

Strong Distribution Shift

Moderate Distribution Shift

Weak Distribution Shift

80

60

40

20

0

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

1

(a) The optimal loss potential.

2.00

Strong Distribution Shift

4OptimalMaxLR×10

Moderate Distribution Shift

1.75

Weak Distribution Shift

1.50

1.25

1.00

0.75

0.50

0.25

0.00

0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

1

(b) The optimal peak LR.

1.0

20K CPT Steps 30K CPT Steps 40K CPT Steps Same Distribution

OptimalReplayRatio

0.8

0.6

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

1

(c) The optimal replay ratio.

- Figure 8. Optimizing hyper-parameters for CPT based on different coefficients to balance general and downstream performance. Strong, moderate, weak distribution shift in (a) and (b) denote different CPT datasets as Pile of Law, Knowledge Pile, and a mixture of 67%

FineWeb and 33% Knowledge Pile, respectively. The “same distribution” in (c) represents a reference line where the target weight (λ1) is the same as replay ratio. See Appendix I for more details.

42500 45000 47500 50000 52500 55000 57500 60000

Step

2.8

3.0

3.2

3.4

3.6

3.8

Out-of-DomainLoss

4 = 0.152Dcpt + 0.86 Dpt

S im a ama = 0. 85Dcpt + 0.622Dpt

St i = 0.516Dcpt + 1.40 Dpt

4 = 0.152Dcpt + 0.86 Dpt

S im a ama = 0. 85Dcpt + 0.622Dpt

St i = 0.516Dcpt + 1.40 Dpt

C4 = 0.152Dcpt + 0.867Dpt

SlimPajama = 0.385Dcpt + 0.622Dpt

Stories = -0.516Dcpt + 1.409Dpt

C4 Loss

SlimPajama Loss

Stories Loss

(a) Dood dataset truth and predicted loss similar to Dpt.

42500 45000 47500 50000 52500 55000 57500 60000

Step

2.0

2.2

2.4

2.6

2.8

3.0

3.2

3.4

Out-of-DomainLoss

tackexc a e = 1. 53Dcpt - 0.546Dpt

x = 1.631Dcpt - 0.60 Dpt

ook = 0.4 5Dcpt 0.611Dpt

tackexc a e = 1. 53Dcpt - 0.546Dpt

x = 1.631Dcpt - 0.60 Dpt

ook = 0.4 5Dcpt 0.611Dpt

Stackexchange = 1.853Dcpt - 0.546Dpt

Arxiv = 1.631Dcpt - 0.609Dpt

Books = 0.475Dcpt + 0.611Dpt

Stackexchange Loss

Arxiv Loss

Books Loss

(b) Dood dataset truth and predicted loss similar to Dcpt.

- Figure 9. The predicted loss curve of Dood validation set, predicted by leveraging a linear combination of Dpt and Dcpt validation losses. We show some different shapes of rising curves similar to Dpt on the left and some different shapes of falling curves similar to Dcpt on the right.

linear combination of losses on several base domains, we hypothesize that the loss on Dood can be represented by a linear combination based on Dpt and Dcpt validation losses:

##### LD

ood

= λ′1LD

pt

+ λ′2LD

cpt

(6)

We verify this hypothesis and calculate λ′1 and λ′2 for several example OOD datasets in Appendix K. Note that the

coefficients λ′1 and λ′2 are related only to datasets and not to other training hyper-parameters.

Loss Prediction of Dood. The Dood validation loss does not adhere to the formulation described in Eq. 4. However,

by calculating and specifying the coefficients λ′1 and λ′2, it becomes feasible to predict the Dood loss curve using a linear combination of the Dpt and Dcpt loss curves. It is interesting that this problem reduces to the balance between Dpt and Dcpt loss (Eq. 5). The optimal hyper-parameters such as LR and replay ratio for this setting have been adequately discussed in the previous section.

As shown in Fig. 9, we first calculate the coefficients in Eq. 6 for several OOD datasets and then predict the corresponding validation losses. The almost perfect prediction suggests

that our approach are quite effective and practical in real scenarios. Moreover, the calculated coefficient represents the “similarity” between OOD datasets and Dpt or Dcpt. As Fig. 9 shows, there are two kinds of OOD datasets: (1) Dpt-like one (larger λ′1) with loss curve upward, and (2) Dcpt-like one (larger λ′2) with loss curve downward.

Finding 5. There exists an optimal loss potential, peak LR and replay ratio designated to balance Dpt and Dcpt losses. Besides, the turning lengths vary depending on the different balance weights. Predicting LD

is equivalent to balancing Dpt and Dcpt losses by utilization of linear combination tricks.

ood

## 6. Open-Source PT Models

For the majority of LLM communities, the PT models we use are usually not trained by ourselves, but from opensource models. Most training details are not reported for those open-source PT models, i.e., the distribution of Dpt, the loss potential, and the PT training hyper-parameters are usually unknown. This inhibits the direct application

of our CPT scaling law. To solve this issue, we propose the following methods to make our scaling law become applicable again.

(a) Firstly, for the unknown PT dataset distribution, some methods based on probing (Hayase et al., 2024) have been proposed. Instead, we simply utilize an open-source Common Crawl dataset as a proxy Dpt to approximate the distribution of Dpt. (b) Secondly, when fitting our scaling law, we regard some variables as unknown parameters to fit. For example, we treat S1pt as a parameter that requires fitting to be close to the undisclosed real S1pt. (c) Thirdly, as most open-source PT models anneal to a minimal LR to get a better performance nowadays, we assume all open-source models anneal their LR to zero when calculating S2cpt. Refer to Appendix G for more details.

To verify our solutions for open-source PT models, we continually pre-train LLaMA3.2-1B (Dubey et al., 2024) and select the RedPajama (Weber et al., 2024) dataset as an proxy Dpt. As Fig. 18 in Appendix G shows, the almost perfect fitting and prediction for CPT loss curve of LLaMA3.2-1B suggests the effectiveness of our proposed methods. Moreover, this result also indicates that our scaling law can be easily extended to CPT scenarios with unknown PT model information, demonstrating the superiority of our scaling law to capture the learning dynamics of CPT.

## 7. Discussion

Laws Formulation. The formulation of S2 in Eq. 1 can have other forms. For example, S2 could also be a multipower form (Luo et al., 2025), which is proposed following the work of Tissue et al. (2024). We adopt the equation form in Eq. 1 because it has fewer parameters and it works more effectively in practice. We also compare some format variates including adding a LR-weighted coefficient and adding a power term to S2 (see more details in Appendix J). The experiments show that all formats lead to similar results while our formulation has superiority in simplicity (i.e. fewer parameters).

Laws Fitting. In our experiments, we predominantly employ constant, cosine, and WSD LRS to fit data, which are widely used in practical applications. It is worth noting that many other LR schedules could be also modeled. To apply our scaling law, we use common LRS (e.g. constant and cosine) to train a few steps to collect loss values. After parameters are fitted based on these values, our scaling law is also capable of predicting the loss curve under other specialized LRS for much longer training durations. Our scaling law shares the similar idea of fitting cost conservation with Tissue et al. (2024), thanks to our scaling law being able to describe the whole dynamics in CPT rather than only final loss.

Limitations. One main limitation of our work is that our laws are primarily based on empirical analyses and experimental verifications. We acknowledge that there is a lack of rigorous theoretical analysis and proof because it is difficult to build theoretical deduction in a non-toy environment with thousands of LLM training factors. However, our scaling law can reasonably reflect the learning dynamics of the CPT process, which can be applied in practical CPT scenarios.

## 8. Conclusion

In this study, we explore the learning dynamics in continual pre-training of large language models. We focus on the evolution of performance across general and downstream domains, with domain performance assessed with validation loss. By observations and analyses, we propose a CPT scaling law that integrates distribution shift and learning rate annealing to predict the validation loss at any intermediate training step under common learning rate schedules. Our scaling law provides a comprehensive understanding of key CPT factors and helps optimize the hyper-parameters in CPT for different training goals. Further experiments demonstrate that the law can also be extended to more complicated scenarios such as out-of-domain datasets and pretrained models with unknown training details. We believe that our CPT scaling law is promising to reshape the understanding of researchers for LLM continual pre-training and scaling laws.

## Impact Statement

CPT is a effective method to enhance the foundation large language models to specific downstream domains or tasks. Our work provides a scaling law to quantitatively describe the learning dynamics of CPT processes. Our results can be used to optimize the training hyper-parameters for balancing the general and downstream performance.

While there will be important impacts resulting from the use of CPT in general, here we focus on the impact of using our scaling law to provide explanations for CPT process. There are many benefits for using our method, such as predicting the loss curve dynamics and optimizing hyper-parameters. The work presented in this aims to advance the field of Large Language Models. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgments

This work was supported by the Strategic Priority Research Program of Chinese Academy of Sciences under Grant XDA0480301 and Fujian Provincial Natural Science Foundation of China (No. 2024J08371).

## References

Balandat, M., Karrer, B., Jiang, D., Daulton, S., Letham, B., Wilson, A. G., and Bakshy, E. Botorch: A framework for efficient monte-carlo bayesian optimization. Advances in neural information processing systems, 33:21524–21538, 2020.

Barnett, M. An empirical study of scaling laws for transfer, 2024. URL https://arxiv.org/abs/2408. 16947.

Bergstra, J. and Bengio, Y. Random search for hyperparameter optimization. Journal of machine learning research, 13(2), 2012.

Chen, W., Zhou, Y., Du, N., Huang, Y., Laudon, J., Chen, Z., and Cui, C. Lifelong language pretraining with distribution-specialized experts. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023a.

Chen, Z., Cano, A. H., Romanou, A., Bonnet, A., Matoba, K., Salvi, F., Pagliardini, M., Fan, S., K¨opf, A., Mohtashami, A., Sallinen, A., Sakhaeirad, A., Swamy, V., Krawczuk, I., Bayazit, D., Marmet, A., Montariol, S., Hartley, M.-A., Jaggi, M., and Bosselut, A. Meditron70b: Scaling medical pretraining for large language models, 2023b. URL https://arxiv.org/abs/2311.

16079.

Colombo, P., Pires, T. P., Boudiaf, M., Culver, D., Melo, R., Corro, C., Martins, A. F. T., Esposito, F., Raposo, V. L., Morgado, S., and Desa, M. Saullm-7b: A pioneering large language model for law, 2024. URL https:// arxiv.org/abs/2403.03883.

DeepSeek-AI, Zhu, Q., Guo, D., Shao, Z., Yang, D., Wang, P., Xu, R., Wu, Y., Li, Y., Gao, H., Ma, S., Zeng, W., Bi, X., Gu, Z., Xu, H., Dai, D., Dong, K., Zhang, L., Piao, Y., Gou, Z., Xie, Z., Hao, Z., Wang, B., Song, J., Chen, D., Xie, X., Guan, K., You, Y., Liu, A., Du, Q., Gao, W., Lu, X., Chen, Q., Wang, Y., Deng, C., Li, J., Zhao, C., Ruan, C., Luo, F., and Liang, W. Deepseekcoder-v2: Breaking the barrier of closed-source models in code intelligence, 2024. URL https://arxiv.org/ abs/2406.11931.

Dou, L., Liu, Q., Zeng, G., Guo, J., Zhou, J., Lu, W., and Lin, M. Sailor: Open language models for south-east asia. arXiv preprint arXiv:2404.03608, 2024.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv preprint arXiv: 2407.21783, 2024.

Fei, Z., Shao, Y., Li, L., Zeng, Z., Yan, H., Qiu, X., and Lin, D. Query of cc: Unearthing large scale domainspecific knowledge from public corpora. arXiv preprint arXiv:2401.14624, 2024.

French, R. M. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128–135, 1999.

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., Presser, S., and Leahy, C. The Pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Gu, J., Yang, Z., Ding, C., Zhao, R., and Tan, F. CMR scaling law: Predicting critical mixture ratios for continual pre-training of language models. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 16143–16162, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main. 903. URL https://aclanthology.org/2024.

emnlp-main.903.

Gupta, K., Th´erien, B., Ibrahim, A., Richter, M. L., Anthony, Q., Belilovsky, E., Rish, I., and Lesort, T. Continual pre-training of large language models: How to (re)warm your model?, 2023. URL https://arxiv.

org/abs/2308.04014.

Hayase, J., Liu, A., Choi, Y., Oh, S., and Smith, N. A. Data mixture inference: What do bpe tokenizers reveal about their training data? arXiv preprint arXiv:2407.16607, 2024.

Henderson*, P., Krass*, M. S., Zheng, L., Guha, N., Manning, C. D., Jurafsky, D., and Ho, D. E. Pile of law: Learning responsible data filtering from the law and a 256gb open-source legal dataset, 2022. URL https:

//arxiv.org/abs/2207.00220.

Hernandez, D., Kaplan, J., Henighan, T., and McCandlish, S. Scaling laws for transfer. arXiv preprint arXiv:2102.01293, 2021a.

Hernandez, D., Kaplan, J., Henighan, T., and McCandlish, S. Scaling laws for transfer, 2021b. URL https:// arxiv.org/abs/2102.01293.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., and Sifre, L. Training compute-optimal large language models. arXiv preprint arXiv: 2203.15556, 2022.

Hu, S., Tu, Y., Han, X., He, C., Cui, G., Long, X., Zheng, Z., Fang, Y., Huang, Y., Zhao, W., Zhang, X., Thai, Z. L., Zhang, K., Wang, C., Yao, Y., Zhao, C., Zhou, J., Cai, J., Zhai, Z., Ding, N., Jia, C., Zeng, G., Li, D., Liu, Z., and Sun, M. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv: 2404.06395, 2024.

Huber, P. J. Robust Estimation of a Location Parameter. The Annals of Mathematical Statistics, 35(1):73 – 101, 1964. doi: 10.1214/aoms/1177703732. URL https: //doi.org/10.1214/aoms/1177703732.

Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, T., Zhang, J., Yu, B., Lu, K., Dang, K., Fan, Y., Zhang, Y., Yang, A., Men, R., Huang, F., Zheng, B., Miao, Y., Quan, S., Feng, Y., Ren, X., Ren, X., Zhou, J., and Lin, J. Qwen2.5-coder technical report, 2024. URL https://arxiv.org/abs/2409.12186.

Ibrahim, A., Th´erien, B., Gupta, K., Richter, M. L., Anthony, Q. G., Belilovsky, E., Lesort, T., and Rish, I. Simple and scalable strategies to continually pre-train large language models. Trans. Mach. Learn. Res., 2024, 2024. URL https://openreview.net/forum? id=DimPeeCxKO.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv: 2001.08361, 2020.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In Bengio, Y. and LeCun, Y. (eds.), 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http: //arxiv.org/abs/1412.6980.

Lange, M. D., van de Ven, G. M., and Tuytelaars, T. Continual evaluation for lifelong learning: Identifying the stability gap. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=Zy350cRstc6.

Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.

Loshchilov, I. and Hutter, F. Sgdr: Stochastic gradient descent with warm restarts. International Conference on Learning Representations, 2016.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv: 1711.05101, 2017.

Luo, K., Wen, H., Hu, S., Sun, Z., Sun, M., Liu, Z., Lyu, K., and Chen, W. A multi-power law for loss curve prediction across learning rate schedules. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=KnoS9XxIlK.

Nocedal, J. Updating quasi newton matrices with limited storage. Mathematics of Computation, 35(151): 951–958, July 1980. ISSN 0025-5718. doi: 10.1090/ S0025-5718-1980-0572855-7.

OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Parmar, J., Satheesh, S., Patwary, M., Shoeybi, M., and Catanzaro, B. Reuse, don’t retrain: A recipe for continued pretraining of language models. arXiv preprint arXiv: 2407.07263, 2024.

Paster, K., Santos, M. D., Azerbayev, Z., and Ba, J. Openwebmath: An open dataset of high-quality mathematical web text, 2023.

Penedo, G., Kydl´ıˇcek, H., allal, L. B., Lozhkov, A., Mitchell, M., Raffel, C., Werra, L. V., and Wolf, T. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum? id=n6SCkn2QaG.

Que, H., Liu, J., Zhang, G., Zhang, C., Qu, X., Ma, Y., Duan, F., ZhiqiBai, JiakaiWang, Zhang, Y., Tan, X., Fu, J., Wang, J., Qu, L., Su, W., and Zheng, B. D-CPT law: Domain-specific continual pre-training scaling law for large language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum? id=JzKFN5fWOk.

Shi, H., Xu, Z., Wang, H., Qin, W., Wang, W., Wang, Y., Wang, Z., Ebrahimi, S., and Wang, H. Continual learning of large language models: A comprehensive survey, 2024. URL https://arxiv.org/abs/2404.16789.

Snoek, J., Larochelle, H., and Adams, R. P. Practical bayesian optimization of machine learning algorithms. Advances in neural information processing systems, 25, 2012.

Soboleva, D., Al-Khateeb, F., Myers, R., Steeves, J. R., Hestness, J., and Dey, N. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama, June 2023. URL https://huggingface.co/ datasets/cerebras/SlimPajama-627B.

Tissue, H., Wang, V., and Wang, L. Scaling law with learning rate annealing, 2024. URL https://arxiv. org/abs/2408.11029.

Wang, Z., Zhang, Z., Lee, C.-Y., Zhang, H., Sun, R., Ren, X., Su, G., Perot, V., Dy, J., and Pfister, T. Learning to prompt for continual learning. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 139–149, 2022. doi: 10.1109/CVPR52688.2022.00024.

Wang, Z., Liu, S., Huang, J., Zheng, W., Liao, Y., Chen, X., Yao, J., and Su, J. A learning rate path switching training paradigm for version updates of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 13581– 13594, 2024.

Weber, M., Fu, D. Y., Anthony, Q., Oren, Y., Adams, S., Alexandrov, A., Lyu, X., Nguyen, H., Yao, X., Adams, V., Athiwaratkun, B., Chalamala, R., Chen, K., Ryabinin, M., Dao, T., Liang, P., R´e, C., Rish, I., and Zhang, C. Redpajama: an open dataset for training large language models. NeurIPS Datasets and Benchmarks Track, 2024.

Xie, X., Ding, K., Yan, S., Toh, K.-C., and Wei, T. Optimization hyper-parameter laws for large language models, 2025. URL https://arxiv.org/abs/2409. 04777.

Ye, J., Liu, P., Sun, T., Zhou, Y., Zhan, J., and Qiu, X. Data mixing laws: Optimizing data mixtures by predicting language modeling performance, 2024. URL https: //arxiv.org/abs/2403.16952.

Zhang, M., Tissue, H., Wang, L., and Qiu, X. Domain2vec: Vectorizing datasets to find the optimal data mixture without training, 2025. URL https://openreview. net/forum?id=sF8jmiD8Bq.

C¸a˘gatay Yıldız, Ravichandran, N. K., Punia, P., Bethge, M., and Ermis, B. Investigating continual pretraining in large language models: Insights and implications, 2024. URL https://arxiv.org/abs/2402.17400.

## A. Related Work

Continual Pre-Training. Continual Pre-Training (CPT) aims to continuously pre-train LLMs to adapt to new domains, such as code (Hui et al., 2024; DeepSeek-AI et al., 2024), medicine (Chen et al., 2023b) and law (Colombo et al., 2024), while avoiding the need to train domain-specific LLMs from scratch (Shi et al., 2024). The primary objective of CPT is to enhance downstream performance while avoiding catastrophic forgetting (Lange et al., 2023; French, 1999; Gupta et al., 2023; Ibrahim et al., 2024). Most existing CPT methods primarily leverage the replay to mix appropriate pre-training data (Que et al., 2024; Gu et al., 2024) or introduce extra model parameters (Wang et al., 2022) to assimilate new domain knowledge. In this work, we comprehensively study the learning dynamics of CPT and propose a CPT scaling law to describe both general and downstream validation loss.

Scaling Laws. Kaplan et al. (2020) empirically discovers a power-law relationship between validation loss L and there factors: model size N, dataset size D, and training compute. Hoffmann et al. (2022) develops Chinchilla, a compute-optimal LLM to balance model size and dataset size. Tissue et al. (2024) introduces a scaling law to describe the learning dynamics affected by the learning rate annealing, which can predict loss at any training steps under various LRS. However, these scaling laws are limited to pre-training scenarios and do not apply when the training dataset changes.

Regarding continual pre-training, Hernandez et al. (2021b) study scaling law for transfer with respect to model size and CPT data. Barnett (2024) proposes an empirical scaling law that incorporates a transfer gap term to indicate the distribution difference between two datasets. Some methods like D-CPT (Que et al., 2024) and CMR (Gu et al., 2024) introduce scaling laws that account for data replay or mixture ratio in the CPT process. Dou et al. (2024) proposes a quadratic function that considers both learning rate and replay ratio. Nevertheless, these existing scaling laws primarily describe the final loss of given LRS and do not account for all CPT-related factors. Our proposed CPT scaling law integrates all relevant factors and can predict loss at each CPT step, thereby providing a comprehensive description of the complete learning dynamics.

Hyper-Parameter Optimization. Identifying optimal hyper-parameter settings is crucial for achieving robust performance in machine learning (Bergstra & Bengio, 2012; Snoek et al., 2012). The principal hyper-parameters in large language models include peak learning rate, learning rate schedules, batch size, and training steps (Kaplan et al., 2020; Hu et al., 2024; Xie et al., 2025). Initial approaches to hyper-parameter optimization primarily utilize model-free techniques such as grid and random search (Bergstra & Bengio, 2012). Subsequently, some methods have employed Bayesian Optimization (Balandat et al., 2020) to predict the performance of various hyper-parameters and select the most effective ones accordingly. Our research focuses on the hyper-parameter in the continual pre-training of larger language models using our proposed CPT scaling law. The hyper-parameter we optimize include the learning rate schedules, peak learning rate, and replay ratio.

## B. Experiment Setups

In this work, we employ multiple experimental setups to validate the effectiveness of our equation. We summarize all experimental setups in Table 1. The majority of our experiments utilize Setting A. Experiments with different replay ratios, batch size, or sequence length are conducted by directly modifying corresponding setups.

## C. Fitting Details

We set λ = 0.999 in our all experiments. Given the LRS of PT and CPT, we can compute out S1pt, S2pt, S1cpt, and S2cpt in advance. We adopt a similar fitting method as Chinchilla scaling law (Hoffmann et al., 2022). We minimize the Huber loss (Huber, 1964) between the predicted and the observed log loss using the L-BFGS algorithm (Nocedal, 1980). We implement this by the utilization of minimize in scipy library. We mitigate the potential issue of local minima of fitting by choosing the optimal fit from a range of initial conditions.

## D. Additional Continual Pre-Training Results

Prediction of Other LRS. We use the fitted parameters in Fig. 3 to predict the loss of other LRS (Fig. 10a and Fig. 10d). Our equation could effectively predict the loss of other LRS as shown in Fig. 10.

Other Dcpt Dataset. Besides Knowledge-Pile (Fei et al., 2024), we also use Eq. 4 to fit transfer loss curves of other Dcpt dataset Pile-of-Law (Henderson* et al., 2022) in the Fig. 11.

Table 1. Experimental settings adopted in this work. Model size denotes the number of nonembedding parameters. We use AdamW Optimizer (Kingma & Ba, 2015; Loshchilov & Hutter, 2017). Most experiments adopt LLaMA-3’s tokenizer (Dubey et al., 2024).

Setups Setting A (main) Setting B Setting C Setting D Model Size 106M 106M 594M 1720M PT dataset FineWeb FineWeb FineWeb FineWeb CPT dataset Knowledge-Pile Pile-of-Law Knowledge-Pile Knowledge-Pile

- Peak LR 2 × 10−4 2 × 10−4 2 × 10−4 2 × 10−4 PT Batch Size (Tokens) 4M 4M 4M 4M CPT Batch Size (Tokens) 4M 4M 4M 4M PT Sequence Length 4096 4096 4096 4096 CPT Sequence Length 4096 4096 4096 4096 Tokenizer LLaMA-3’s LLaMA-3’s LLaMA-3’s LLaMA-3’s

β1,β2 in AdamW 0.9, 0.95 0.9, 0.95 0.9, 0.95 0.9, 0.95

Weight Decay 0.1 0.1 0.1 0.1 Gradient Clip 1.0 1.0 1.0 1.0

Setups LLaMA3.2-1B Model Size 1B PT dataset Unknown CPT dataset Pile-of-Law

- Peak LR 2 × 10−5 PT Batch Size (Tokens) Unknown CPT Batch Size (Tokens) 4M PT Sequence Length Unknown CPT Sequence Length 4096 Tokenizer LLaMA-3’s

β1,β2 in AdamW 0.9, 0.95

Weight Decay 0.1 Gradient Clip 1.0

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

0.00

0 10000 20000 30000 40000 50000 60000

Step

(a) Learning Rate Schedule.

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

0.00

0 10000 20000 30000 40000 50000 60000

Step

(d) Learning Rate Schedule.

WSD Pre-Training Truth Loss

Linear Continual Pre-Training Truth Loss

3.7

WSD Pre-Training Predicted Loss

ValidationLossDpt

Linear Continual Pre-Training Predicted Loss

3.6

3.5

3.4

3.3

3.2

3.1

10000 20000 30000 40000 50000 60000

Step

(b) Predicted Dpt (FineWeb) Loss.

WSD Pre-Training Truth Loss

Cosine Continual Pre-Training Truth Loss

3.7

WSD Pre-Training Predicted Loss

ValidationLossDpt

Cosine Continual Pre-Training Predicted Loss

3.6

3.5

3.4

3.3

3.2

3.1

10000 20000 30000 40000 50000 60000

Step

(e) Predicted Dpt (FineWeb) Loss.

WSD Pre-Training Truth Loss

3.4

Linear Continual Pre-Training Truth Loss

WSD Pre-Training Predicted Loss

ValidationLossDcpt

3.3

Linear Continual Pre-Training Predicted Loss

3.2

3.1

3.0

2.9

2.8

2.7

2.6

2.5

10000 20000 30000 40000 50000 60000

Step

(c) Predicted Dcpt (Knowledge Pile) Loss.

WSD Pre-Training Truth Loss

3.4

Cosine Continual Pre-Training Truth Loss

WSD Pre-Training Predicted Loss

ValidationLossDcpt

3.3

Cosine Continual Pre-Training Predicted Loss

3.2

3.1

3.0

2.9

2.8

2.7

2.6

2.5

10000 20000 30000 40000 50000 60000

Step

(f) Predicted Dcpt (Knowledge Pile) Loss.

Figure 10. Using the fitted parameters in the Fig. 3 to predict all pre-training and CPT loss curve of other LRS. (a) is one kind of without re-warmup method and (b) is a more realistic LRS that the learning rate re-warmup to 10% peak PT learning rate and then annealing to zero with cosine method.

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

0.00

5000 100001500020000250003000035000400004500050000

Step

(a) Learning Rate Schedule.

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

0.00

10000 20000 30000 40000 50000 60000 70000

Step

(d) Learning Rate Schedule.

4.2

WSD Pre-Training Truth Loss

Cosine Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

4.0

ValidationLossDpt

Cosine Continual Pre-Training Fitted Loss

3.8

L(s) = 3.064 + 0.482(S1pt + S1cpt) 0.511

+1.353(1 (37S1cpt + 1) 0.262) 0.294S2pt 0.264S2cpt

3.6

3.4

3.2

5000 10000 15000 20000 25000 30000 35000 40000 45000 50000

Step

(b) Dpt (FineWeb) Validation Loss.

4.4

WSD Pre-Training Truth Loss

Cosine Continual Pre-Training Truth Loss

4.2

WSD Pre-Training Fitted Loss

ValidationLossDpt

Cosine Continual Pre-Training Fitted Loss

4.0

L(s) = 3.064 + 0.482(S1pt + S1cpt) 0.511

+1.353(1 (37S1cpt + 1) 0.262) 0.294S2pt 0.264S2cpt

3.8

3.6

3.4

3.2

10000 20000 30000 40000 50000 60000 70000

Step

(e) Dpt (FineWeb) Validation Loss.

WSD Pre-Training Truth Loss

3.50

Cosine Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

ValidationLossDcpt

Cosine Continual Pre-Training Fitted Loss

3.25

3.00

L(s) = 2.978 + 0.485(S1pt + S1cpt) 0.701 1.409(1 (383S1cpt + 1) 0.368) 0.274S2pt 0.298S2cpt

2.75

2.50

2.25

2.00

1.75

5000 100001500020000250003000035000400004500050000

Step

(c) Dcpt (Pile-of-Law) Validation Loss.

WSD Pre-Training Truth Loss

3.50

Cosine Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

ValidationLossDcpt

Cosine Continual Pre-Training Fitted Loss

3.25

3.00

L(s) = 2.978 + 0.485(S1pt + S1cpt) 0.701 1.409(1 (383S1cpt + 1) 0.368) 0.274S2pt 0.298S2cpt

2.75

2.50

2.25

2.00

1.75

10000 20000 30000 40000 50000 60000 70000

Step

(f) Dcpt (Pile-of-Law) Validation Loss.

Figure 11. Using Eq. 4 to fit all loss curves which are pre-trained with FineWeb and continual pre-trained with law.

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

0.00

0 10000 20000 30000 40000 50000 60000

Step

(a) Learning Rate Schedule.

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

0.00

0 10000 20000 30000 40000 50000 60000

Step

(d) Learning Rate Schedule.

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

0.00

0 10000 20000 30000 40000 50000 60000

Step

(g) Learning Rate Schedule.

WSD Pre-Training Truth Loss

3.4

WSD Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

ValidationLossDcpt

WSD Continual Pre-Training Fitted Loss

3.2

L(s) = 3.020 + 0.441(S1pt + S1cpt) 0.696 0.632(1 (5046S1cpt + 1) 0.145) 0.273S2pt 0.275S2cpt

3.0

2.8

2.6

10000 20000 30000 40000 50000 60000

Step

(b) Dpt (FineWeb) Loss of 33% Replay.

3.7

WSD Pre-Training Truth Loss

WSD Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

3.6

ValidationLossDpt

WSD Continual Pre-Training Fitted Loss

3.5

L(s) = 3.068 + 0.480(S1pt + S1cpt) 0.518 + 0.046(1 (3S1cpt + 1) 0.915) 0.294S2pt 0.269S2cpt

3.4

3.3

3.2

3.1

10000 20000 30000 40000 50000 60000

Step

(e) Dpt (FineWeb) Loss of 50% Replay.

3.7

WSD Pre-Training Truth Loss

WSD Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

3.6

ValidationLossDpt

WSD Continual Pre-Training Fitted Loss

3.5

L(s) = 3.069 + 0.479(S1pt + S1cpt) 0.519 + 0.105(1 (11S1cpt + 1) 0.069) 0.295S2pt 0.280S2cpt

3.4

3.3

3.2

3.1

10000 20000 30000 40000 50000 60000

Step

(h) Dpt (FineWeb) Loss of 67% Replay.

3.7

WSD Pre-Training Truth Loss

WSD Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

3.6

ValidationLossDpt

WSD Continual Pre-Training Fitted Loss

3.5

L(s) = 3.067 + 0.481(S1pt + S1cpt) 0.515 + 0.077(1 (6S1cpt + 1) 0.608) 0.298S2pt 0.284S2cpt

3.4

3.3

3.2

3.1

10000 20000 30000 40000 50000 60000

Step

(c) Dcpt (KP) Loss of 33% Replay.

WSD Pre-Training Truth Loss

3.4

WSD Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

ValidationLossDcpt

WSD Continual Pre-Training Fitted Loss

3.2

L(s) = 3.017 + 0.442(S1pt + S1cpt) 0.686 0.599(1 (6484S1cpt + 1) 0.144) 0.277S2pt 0.277S2cpt

3.0

2.8

2.6

10000 20000 30000 40000 50000 60000

Step

(f) Dcpt (KP) Loss of 50% Replay.

WSD Pre-Training Truth Loss

3.4

WSD Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

ValidationLossDcpt

WSD Continual Pre-Training Fitted Loss

3.2

L(s) = 3.017 + 0.442(S1pt + S1cpt) 0.685 0.532(1 (5693S1cpt + 1) 0.166) 0.274S2pt 0.276S2cpt

3.0

2.8

2.6

10000 20000 30000 40000 50000 60000

Step

(i) Dcpt (KP) Loss of 67% Replay.

Figure 12. Using Eq. 4 to fit different Dpt replay ratio models independently.

>10K

Strong Distribution Shift

Moderate Distribution Shift

Weak Distribution Shift

80K

TurningLength

60K

40K

20K

0

0.4 0.5 0.6 0.7 0.8 0.9 1.0

1

Figure 13. The CPT turning lengths of different coefficients for balancing general and downstream domain performance.

0.3

4.2

N=106M N=594M N=1720M

DistributionShift

0.2

4.0

0.627(1 (0.009s + 1) 0.081)

0.1

FineWebLoss

3.8

0.0

3.6

CPT Step (s)

3.4

3.2

3.0

Distribution Shift

2.8

2.6

0 10000 20000 30000 40000 50000 60000

Step

(a) Dpt Distribution Shift of different model size N.

4.00

0.6

N=106M N=594M N=1720M

DistributionShift

KnowledgePileLoss

3.75

0.4

1.607(1 (107.439s + 1) 0.025)

0.2

3.50

0.0

3.25

CPT Step (s)

3.00

2.75

2.50

Distribution Shift

2.25

0 10000 20000 30000 40000 50000 60000

Step

(b) Dcpt Distribution Shift of different model size N.

Figure 14. The distribution shift across different model size with constant LRS.

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

0.00

0 10000 20000 30000 40000 50000 60000

Step

(a) Learning Rate Schedule.

4.0

N=106M Ground Truth Loss N=594M Ground Truth Loss N=1720M Ground Truth Loss

N=106M Predict Loss N=594M Predict Loss N=1720M Predict Loss

3.8

3.6

FineWebLoss

L(s) = 1.829 + 0.504(S1pt + S1cpt) 0.537 + 0.278(1 (12S1cpt + 1) 0.359) 0.090S2ptN0.235 0.103S2cptN0.199 + 3.072N 0.195

3.4

3.2

3.0

2.8

2.6

0 10000 20000 30000 40000 50000 60000

Step

(b) FineWeb (Dpt) Validation Loss.

3.8

N=106M Ground Truth Loss N=594M Ground Truth Loss N=1720M Ground Truth Loss

N=106M Predict Loss N=594M Predict Loss N=1720M Predict Loss

3.6

KnowledgePileLoss

3.4

3.2

3.0

2.8

2.6

L(s) = 1.860 + 0.438(S1pt + S1cpt) 0.585 0.552(1 (187S1cpt + 1) 0.326) 0.333S2ptN0.012 0.140S2cptN0.145

2.4

+2.718N 0.189

2.2

2.0

0 10000 20000 30000 40000 50000 60000

Step

(c) Knowledge Pile (Dcpt) Validation Loss.

Figure 15. Using Eq. 7 to fit the loss curve of all model size in both Dpt and Dcpt validation set.

Different Replay Ratio. We use Eq. 4 to fit all loss curves of different Dpt replay ratio independently in Fig. 12.

## E. Extension To Model Size Scaling

Distribution Shift Term of Different Model Sizes. We first explore the effect of model size N on the distribution shift term. CPT experiments are conducted across various model sizes—106M, 594M, and 1.7B without embedding—using a constant learning rate. As shown in Fig. 14a, the shift terms for different model sizes N nearly coincide. Based on these observations, we hypothesize that the distribution shift term is independent of both model size N and transfer starting points. This implies that data transfer results in a consistent loss difference across different models sizes.

Model Size Scaling. Meanwhile, scaling law with LR annealing (Tissue et al., 2024) has demonstrated that the learning rate annealing scales with model sizes N, that S2 ∝ Nγ. Building on the experiments and analysis above, we extend our proposed Eq. 4 to incorporate model size scaling:

− C2 · S2cpt·Nγ

L(Spt,Scpt) =L0 + A · (S1pt + S1cpt)−α − C1 · S2pt·Nγ

2

1

##### + B · (1 − (1 + E · S1cpt)−β) + F · N−γ

3

(7)

where F,γ1,γ2,γ3 is the constant parameters. The F · N−γ

3 is the model size term in traditional Chinchilla scaling law (Hoffmann et al., 2022). We use Eq. 7 to fit the transfer curves of all model sizes as shown in Fig. 15. Furthermore, we apply Eq. 4 to fit the CPT loss curves of larger model sizes independently, as illustrated in Fig. 16. This demonstrates the adaptability of our equation across various model sizes.

However, it should be emphasized that the model size in our experiments have not yet reached the scale of mainstream LLMs today, so our experimental conclusions regarding model size are based on the assumptions derived from existing

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

0.00

0 10000 20000 30000 40000 50000 60000

Step

(a) Learning Rate Schedule.

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

0.00

0 10000 20000 30000 40000 50000 60000

Step

(d) Learning Rate Schedule.

WSD Pre-Training Truth Loss

WSD Continual Pre-Training Truth Loss

3.3

WSD Pre-Training Fitted Loss

ValidationLossDpt

WSD Continual Pre-Training Fitted Loss

3.2

L(s) = 2.688 + 0.526(S1pt + S1cpt) 0.483 + 29.121(1 (100S1cpt + 1) 0.001) 0.437S2pt 0.395S2cpt

3.1

3.0

2.9

2.8

2.7

10000 20000 30000 40000 50000 60000

Step

(b) Dpt (FineWeb) Loss of 594M.

WSD Pre-Training Truth Loss

WSD Continual Pre-Training Truth Loss

3.2

WSD Pre-Training Fitted Loss

ValidationLossDpt

WSD Continual Pre-Training Fitted Loss

3.1

L(s) = 2.496 + 0.569(S1pt + S1cpt) 0.474 + 8.710(1 (100S1cpt + 1) 0.004) 0.485S2pt 0.438S2cpt

3.0

2.9

2.8

2.7

2.6

10000 20000 30000 40000 50000 60000

Step

(e) Dpt (FineWeb) Loss of 1720M.

WSD Pre-Training Truth Loss

3.1

WSD Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

ValidationLossDcpt

3.0

WSD Continual Pre-Training Fitted Loss

2.9

2.8

L(s) = 2.657 + 0.443(S1pt + S1cpt) 0.566 0.518(1 (100S1cpt + 1) 0.424) 0.360S2pt 0.358S2cpt

2.7

2.6

2.5

2.4

2.3

2.2

10000 20000 30000 40000 50000 60000

Step

(c) Dcpt (Knowledge Pile) Loss of 594M.

WSD Pre-Training Truth Loss

3.0

WSD Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

ValidationLossDcpt

2.9

WSD Continual Pre-Training Fitted Loss

2.8

2.7

2.6

L(s) = 2.557 + 0.407(S1pt + S1cpt) 0.682 0.527(1 (100S1cpt + 1) 0.390) 0.440S2pt 0.412S2cpt

2.5

2.4

2.3

2.2

2.1

10000 20000 30000 40000 50000 60000

Step

(f) Dcpt (Knowledge Pile) Loss of 1720M.

Figure 16. Using Eq. 4 to fit all PT and CPT loss curve of 594M and 1720M model size respectively.

results. If the assumption hold for larger model size (e.g., 7B, 70B), we can conclude that influenced by the same absolute distribution shift value, larger models exhibit greater vulnerability in the general domain but demonstrate better adaptability to downstream domains.

## F. Batch Size and Sequence Length

In the above experiments, we maintain the same batch size for both PT and CPT phases. However, in the real situation, when computational resources and datasets are limited, practitioners may keep a smaller CPT global batch size than PT phase. Additionally, in other cases, CPT aims to increase the context length of LLMs, requiring increases in both sequence length and RoPE base. We conduct CPT experiments with larger and smaller batch sizes, as shown in Fig. 17. When the sequence length is 8K, we increase RoPE base from 10,000 to 500,000.

Distribution Shift of Different Batch Size We leverage the constant LR to examine whether the distribution shift term for different batch sizes satisfies the same functional form. We using Eq. 4 to fit the loss curves of different batch sizes. As shown in Fig. 17a and Fig. 17b, all loss curves with different transfer steps for both larger and smaller batch sizes can be fitted with a single distribution shift term, which demonstrates that Eq. 4 can also accommodate changes in batch size and sequence length.

## G. Open-Source Pre-Training Models

A more realistic scenario posits that the PT model is an open-source model, and we do not have access to the exact PT process. Therefore, the distribution of PT dataset, the loss potential, and the PT training amount usually remain unknown.

Unknown PT Training Amount and Loss Potential For the open-source models, we do not know the PT training amount and loss potential to get the PT forward area S1pt and the final LR to calculate the CPT annealing area S2cpt. For forward area S1pt, we treat it as a parameter to be fitted. Typically, most open source models will anneal the LR to zero or a minium LR to get a better benchmark performance. We assume that the final LR of all open-source models is zero, which facilitates the computation of the CPT annealing area S2cpt. The learning rate of CPT of open-source models is consider to re-warmup

4.0

PT Seqlen=4K GBS=1024

Seqlen=8K GBS=512 Seqlen=4K GBS=256 Seqlen=2K GBS=4096

ValidationLossDpt

LDT(S) = 0.29(1 (88S1cpt + 1) 0.20)

3.8

LDT(S) = 0.45(1 (142S1cpt + 1) 0.18)

3.6

3.4

3.2

LDT(S) = 0.36(1 (37S1cpt + 1) 0.15)

3.0

0 5000 10000 15000 20000 25000 30000 35000 40000

Step

(a) Dpt (FineWeb) loss curve of different batch size.

3.8

PT Seqlen=4K GBS=1024

Seqlen=8K GBS=512 Seqlen=4K GBS=256 Seqlen=2K GBS=4096

ValidationLossDcpt

3.6

LDT(S) = 0.58(1 (1762S1cpt + 1) 0.22)

3.4

LDT(S) = 1.00(1 (14426S1cpt + 1) 0.06)

3.2

3.0

2.8

2.6

LDT(S) = 0.81(1 (17920S1cpt + 1) 0.10)

0 5000 10000 15000 20000 25000 30000 35000 40000

Step

(b) Dcpt (Knowledge Pile) loss curve of different batch size.

- Figure 17. Using Eq. 4 to fitted loss curve of different batch size in the continual pre-traininig. The smaller batch size is 1M tokens with 4K sequence length and the larger batch size is 8M tokens with 8K sequence length. We annotate the different distribution shift terms in the figure.

from zero to the specific peak learning rate and then anneal with specific LRS.

Unknown PT Dataset Distribution The aforementioned equation holds only in the loss curve of Dpt and Dcpt validation dataset. However, we do not know the exact Dpt dataset distribution of open-source models. In this case, we could select an open-source common crawl validation set as proxy Dpt.

To verify the two hypotheses mentioned above are reasonable, We conduct the experiments that continual pre-train the LLaMA3.2-1B (Dubey et al., 2024) with Pile-of-Law dataset (Henderson* et al., 2022). We consider the C4 portion of the RedPajama (Weber et al., 2024) dataset as a proxy Dpt. We use fewer training steps to fit the parameters and then predict the loss of longer steps for the proxy Dpt and true Dcpt. As shown in Fig. 18, Eq. 4 could effectively predict the further loss of proxy Dpt and Dcpt. Based on the proxy Dpt and true Dcpt, we could describe the performance dynamics and complete the above hyper-parameters optimization for open-source models.

We also conduct experiments with our model pre-trained with FineWeb and continual pre-trained with Pile-of-Law dataset. We treat it as a model with unknown PT information. We still use the portion of RedPajama as the proxy Dpt dataset to predict the loss of longer training steps as shown in Fig. 18.

## H. Adding Replay Ratio to Our Formulation

D-CPT law (Que et al., 2024) proposed a scaling law integrating with Dpt and Dcpt data mixture ratio. We have also integrated this data mixture ratio into our formulation. Appendix D demonstrates that loss curves for different data ratios can be individually fitted using distinct equations. However, we are currently exploring a unified formulation that incorporates the data mixture ratio to represent all loss curves. Both the distribution shift term and the LR annealing term are influenced by the replay ratio. A higher Dpt ratio leads to a weaker distribution shift, and results in a smaller LR annealing term in the Dcpt validation loss, while increasing the LR annealing term in the Dpt validation loss. We find that the exponential form, which is consistent with the Data Mixing Law (Ye et al., 2024), best fits these effects and subsequently incorporate it into both the distribution shift term and LR annealing term:

1rpt + B · 1 − 1 + E · S1cpt −β (1 − e−a

Lpt = L0 + A · S1pt + S1cpt −α − C1 · S2pt − C2 · S2cptea

2rcpt)

1rcpt + B · 1 − 1 + E · S1cpt −β (ea

Lcpt = L0 + A · S1pt + S1cpt −α − C1 · S2pt − C2 · S2cptea

2rcpt − 1)

(8)

where rpt and rcpt are the data mixture ratio of PT and CPT data respectively, such that rpt + rcpt = 1, and a1 and a2 are the additional parameters. To ensure that the distribution shift term is zero when rcpt equals zero, we have modified the exponential formulation in the distribution shift term accordingly. The effectiveness of this equation is illustrated in Fig. 19. While the D-CPT law predicts only the final loss across different replay ratios, our method is capable of describing the entire training dynamics for various replay ratios.

2.00

4LearningRate×10

1.75

1.50

1.25

Fitted LRS

Predict LRS

1.00

0.75

0.50

0.25

0.00

0 10000 20000 30000 40000 50000

Step

(a) Learning rate schedule of fitted and pre-

- dict of Model I.

0 1000 2000 3000 4000 5000

Step

2.70

2.75

2.80

2.85

2.90

RedPajama-C4Loss

L(s) = 2.554 + 11.123(S1cpt + 1.907) 7.122 6.949(1 (11S1cpt + 1) 0.022) 0.001S2cpt

Ground Truth Loss

Fitted loss curve

(b) Fitted RedPajama-C4 dataset loss curve of Model I.

0 2500 5000 7500 10000 12500 15000 17500 20000

Step

2.7

2.8

2.9

3.0

3.1

RedPajama-C4Loss

L(s) = 2.554 + 11.123(S1cpt + 1.907) 7.122 6.949(1 (11S1cpt + 1) 0.022) 0.001S2cpt

Ground Truth Loss

Predict loss curve

(c) Predicted RedPajama-C4 dataset loss curve of Model I.

0 2500 5000 7500 10000 12500 15000 17500 20000

Step

0.00

0.25

0.50

0.75

1.00

1.25

1.50

1.75

2.00

4LearningRate×10

Fitted LRS

Predict LRS

(d) Learning rate schedule of fitted and pre-

- dict of Model II.

2.00

4LearningRate×10

1.75

1.50

1.25

Fitted LRS

Predict LRS

1.00

0.75

0.50

0.25

0.00

0 10000 20000 30000 40000 50000

Step

(g) Learning rate schedule of fitted and pre-

- dict of Model I.

0 1000 2000 3000 4000 5000

Step

1.50

1.55

1.60

1.65

1.70

1.75

1.80

PileofLawLoss

L(s) = 2.101 + 0.326(S1cpt + 0.102) 0.241 0.932(1 (12S1cpt + 1) 12.195) 0.099S2cpt

Ground Truth Loss

Fitted loss curve

(h) Fitted Pile-of-Law dataset loss curve of Model I.

0 2500 5000 7500 10000 12500 15000 17500 20000

Step

1.40

1.45

1.50

1.55

1.60

1.65

1.70

1.75

1.80

PileofLawLoss

L(s) = 2.101 + 0.326(S1cpt + 0.102) 0.241 0.932(1 (12S1cpt + 1) 12.195) 0.099S2cpt

Ground Truth Loss

Predict loss curve

(i) Predicted Pile-of-Law dataset loss curve of Model I.

0 2500 5000 7500 10000 12500 15000 17500 20000

Step

0.00

0.25

0.50

0.75

1.00

1.25

1.50

1.75

2.00

4LearningRate×10

Fitted LRS

Predict LRS

(j) Learning rate schedule of fitted and pre-

- dict of Model II.

4.2

4.1

RedPajama-C4Loss

4.0

3.9

L(s) = 2.742 + 0.887(S1cpt + 5.157) 0.414 3.112(1 (100S1cpt + 1) 0.072) 0.279S2cpt

3.8

3.7

3.6

Ground Truth Loss

3.5

Fitted Loss Curve

3.4

0 2000 4000 6000 8000 10000

Step

(e) Fitted RedPajama-C4 dataset loss curve of Model II.

2.2

Ground Truth Loss

Fitted Loss Curve

2.1

PileofLawLoss

L(s) = 2.414 + 0.171(S1cpt + 0.085) 0.531 0.790(1 (17S1cpt + 1) 35.333) 0.311S2cpt

2.0

1.9

1.8

1.7

0 2000 4000 6000 8000 10000

Step

(k) Fitted Pile-of-Law dataset loss curve of Model II.

4.4

4.3

RedPajama-C4Loss

4.2

4.1

L(s) = 2.742 + 0.887(S1cpt + 5.157) 0.414 3.112(1 (100S1cpt + 1) 0.072) 0.279S2cpt

4.0

3.9

3.8

Ground Truth Loss Predict Loss Curve

3.7

3.6

0 10000 20000 30000 40000 50000

Step

(f) Predicted RedPajama-C4 dataset loss curve of Model II.

2.2

Ground Truth Loss Predict Loss Curve

2.1

PileofLawLoss

2.0

L(s) = 2.414 + 0.171(S1cpt + 0.085) 0.531 0.790(1 (17S1cpt + 1) 35.333) 0.311S2cpt

1.9

1.8

1.7

1.6

0 10000 20000 30000 40000 50000

Step

(l) Predicted Pile-of-Law dataset loss curve of Model II.

- Figure 18. Using Eq. 4 to fit and predict the proxy Dpt and true Dcpt dataset of open-source PT models. The Model I refers to LLaMA3.21B. Model II refers to our model pre-trained with FineWeb but we regard it as an unknown model and use proxy Dpt rather than FineWeb. The Dcpt dataset are both Pile-of-Law, and the proxy Dpt is RedPajama-C4.

## I. Optimal Hyper-Parameters

In the Fig. 8, we show the optimal hyper-parameters based on different coefficients. For optimal loss potential and peak LR, we use FineWeb as Dpt and and explore three different Dcpt dataset: (1) Pile of Law, (2) Knowledge Pile, and (3) a mixture of 67% FineWeb and 33% Knowledge Pile. These three Dcpt datasets have strong, moderate, and weak distribution shifts comparing to Dpt. The training setting for each Dcpt dataset is consistent with the Setting A and Setting B in Table 1. The LRS used for fitting these three Dcpt datastes and the fitted equation coefficients are shown in Fig. 11, Fig. 3 and Fig. 12. We directly use these three sets of coefficients to search the optimal hyper-parameters. In the Fig. 8a and Fig. 8b, we assume that the LRS for the PT phase follows a WSD schedule with 40k steps, while the continual CPT phase employs a cosine schedule with 10k steps, consistent with the configuration shown in Fig. 11a.

For optimal replay ratio, we use FineWeb as Dpt and Knowledge Pile as Dcpt. In Fig. 8c, we maintain the assumption that the PT phase employs WSD scheduling while the CPT phase uses cosine scheduling with varying CPT step counts. The blue dashed reference line represents the scenario where the target weight (λ1) equals the replay ratio. It can be reasonably assumed that if the model were initialized from scratch (rather than from a pre-trained model), the optimal replay ratio curve would follow the blue dashed line. However, since our model is pre-trained, this causes the curve to deviate and exhibit a wavy pattern. We directly apply the fitted coefficients presented in Fig. 19 to determine the optimal replay ratio.

### J. Other Formats with LR Annealing Similar with scaling law with LR annealing (Tissue et al., 2024), we also try the other possible forms of LR annealing.

Adding a LR-weighted Coefficient To solve that when LR anneals to nearly 0, S2 still has historical momentum, making the loss continue to decrease. A revision is that adding a LR-weighted coefficient to S2:

s

mi · ηiϵ (9)

S2 =

i=1

We test the coefficient ϵ is 0.1 and 0.2, showing the fitted result in the Fig. 20.

S2 Power Formats Considering that the annealing loss and S2 have a positive correlation, L ∝ S2ζ might be a more reasonable format than L ∝ S2. We revise our formulation:

L = L0 + A · S1pt + S1cpt −α − C1 · (S2pt)ζ

− C2 · (S2cpt)ζ

1

2

(10)

+ B · 1 − 1 + E · S1cpt −β

We add two other fitted parameters in the function for different annealing area of PT and CPT. We also show the fitted effect in the Fig. 20. We show the huber loss and R2 of all possible formats in the Table 2. All the fitting effect are really good, but the original format has the fewest parameters which is more effective.

## K. Out-of-Domain Validation Set

Data mixing law (Ye et al., 2024) shows that validation loss for some domains can be represented by a combination of other domains. In scenarios where the CPT dataset, such as Knowledge-Pile, is not highly domain-specific, employing a linear combination of Dpt and Dcpt can serve as a reasonable approximation for certain downstream validation sets. However, it is important to note that this approach may not be universally applicable across all CPT datasets and all Dood validation loss scenarios. We test the validity of using a linear combination of Dpt (FineWeb) and Dcpt (Knowledge-Pile) to estimate the validation loss for certain out-of-domain sets in the Fig. 21 and Fig. 22. These out-of-domain validation sets include StackExchange, arXiv, and C4 in RedPajama (Weber et al., 2024), as well as PhilPapers and Books in Pile (Gao et al., 2020), SlimPajama (Soboleva et al., 2023), and Open-Web-Math (Paster et al., 2023).

3.6

CPT Truth Loss

KP(100%) Fitted Loss

ValidationLossDpt

KP(67%) Fitted Loss KP(50%) Fitted Loss KP(33%) Fitted Loss KP(0%) Fitted Loss

3.5

3.4

3.3

3.2

3.1

30000 35000 40000 45000 50000 55000 60000

Step

(a) Dpt loss curve of different replay ratios.

3.6

CPT Truth Loss

KP(100%) Fitted Loss

ValidationLossDcpt

3.4

KP(67%) Fitted Loss KP(50%) Fitted Loss KP(33%) Fitted Loss KP(0%) Fitted Loss

3.2

3.0

2.8

2.6

30000 35000 40000 45000 50000 55000 60000

Step

(b) Dcpt loss curve of different replay ratios.

###### Figure 19. Using Eq. 8 to fitted all loss curves of different replay ratio in the continual pre-training. The fitted equation is Lpt = 3.067+0.480· S1pt + S1cpt −0.510 −0.280·S2pt −0.263·S2cpte0.055rpt +0.276· 1 − 1 + 99.35 · S1cpt −β (1 − e−3.238rcpt) and Lcpt = 2.992+0.456· S1pt + S1cpt −0.510−0.285·S2pt−0.279·S2cpte0.037rpt−0.526· 1 − 1 + 100.34 · S1cpt −β (e5.696rcpt − 1).

Table 2. The fitting effect of different possible equation formats.

Dpt Huber Loss ↓ R2 ↑

Original 0.0016 0.9944

- Adding LR Cofficient (ϵ = 0.1) 0.0016 0.9950
- Adding LR Cofficient (ϵ = 0.2) 0.0017 0.9950 Adding S2 Power 0.0016 0.9952

Dcpt Huber Loss ↓ R2 ↑ Original 0.0021 0.9993

- Adding LR Cofficient (ϵ = 0.1) 0.0025 0.9984
- Adding LR Cofficient (ϵ = 0.2) 0.0024 0.9983 Adding S2 Power 0.0025 0.9984

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

0.00

0 10000 20000 30000 40000 50000 60000

Step

(a) Learning Rate Schedule used in fitting

- adding LR-weighted coefficient ϵ = 0.1.

10000 20000 30000 40000 50000 60000

Step

3.1

3.2

3.3

3.4

3.5

3.6

3.7

3.8

ValidationLossDpt

L(s) = 3.037 + 0.510(S1pt + S1cpt) 0.464 + 4.542(1 (100S1cpt + 1) 0.008) 0.779S2pt 0.728S2cpt

Peak Point

Distribution Shift + LR Re-Warmup

WSD Pre-Training Truth Loss

WSD Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

WSD Continual Pre-Training Fitted Loss

(b) Dpt Validation Loss for adding LRweighted coefficient ϵ = 0.1.

10000 20000 30000 40000 50000 60000

Step

2.5

2.6

2.7

2.8

2.9

3.0

3.1

3.2

3.3

3.4

ValidationLossDcpt

L(s) = 2.982 + 0.451(S1pt + S1cpt) 0.563 0.528(1 (100S1cpt + 1) 0.511) 0.773S2pt 0.798S2cpt Distribution Shift

+ LR Re-Warmup

WSD Pre-Training Truth Loss

WSD Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

WSD Continual Pre-Training Fitted Loss

(c) Dcpt Validation Loss for adding LRweighted cofficient ϵ = 0.1.

0 10000 20000 30000 40000 50000 60000

Step

0.00

0.25

0.50

0.75

1.00

1.25

1.50

1.75

2.00

4LearningRate×10

Pre-Training LRS

Continual Pre-Training LRS

(d) Learning Rate Schedule used in fitting

- adding LR-weighted coefficient ϵ = 0.2.

2.00

4LearningRate×10

1.75

1.50

1.25

1.00

0.75

0.50

Pre-Training LRS

0.25

Continual Pre-Training LRS

0.00

0 10000 20000 30000 40000 50000 60000

Step

(g) Learning Rate Schedule used in fitting adding adding S2 power format.

3.8

WSD Pre-Training Truth Loss

WSD Continual Pre-Training Truth Loss

3.7

WSD Pre-Training Fitted Loss

ValidationLossDpt

WSD Continual Pre-Training Fitted Loss

3.6

L(s) = 3.036 + 0.512(S1pt + S1cpt) 0.463 + 1.567(1 (100S1cpt + 1) 0.024) 2.004S2pt 1.738S2cpt

3.5

Peak Point

3.4

3.3

Distribution Shift + LR Re-Warmup

3.2

3.1

10000 20000 30000 40000 50000 60000

Step

(e) Dpt Validation Loss for adding LRweighted cofficient ϵ = 0.2.

3.8

WSD Pre-Training Truth Loss

WSD Continual Pre-Training Truth Loss

3.7

WSD Pre-Training Fitted Loss

ValidationLossDpt

WSD Continual Pre-Training Fitted Loss

3.6

L(s) = 3.053 + 0.498(S1pt + S1cpt) 0.493 + 1.455(1 (100S1cpt + 1) 0.027) 0.249(S2pt)0.906 0.240(S2cpt)0.901

3.5

Peak Point

3.4

3.3

Distribution Shift + LR Re-Warmup

3.2

3.1

10000 20000 30000 40000 50000 60000

Step

(h) Dpt Validation Loss for adding S2 power format.

WSD Pre-Training Truth Loss

3.4

WSD Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

ValidationLossDcpt

3.3

WSD Continual Pre-Training Fitted Loss

3.2

3.1

L(s) = 2.971 + 0.461(S1pt + S1cpt) 0.535 0.534(1 (100S1cpt + 1) 0.498) 2.022S2pt 1.972S2cpt Distribution Shift

3.0

2.9

+ LR Re-Warmup

2.8

2.7

2.6

2.5

10000 20000 30000 40000 50000 60000

Step

(f) Dcpt Validation Loss for adding LRweighted cofficient ϵ = 0.2.

WSD Pre-Training Truth Loss

3.4

WSD Continual Pre-Training Truth Loss

WSD Pre-Training Fitted Loss

ValidationLossDcpt

3.3

WSD Continual Pre-Training Fitted Loss

3.2

3.1

L(s) = 2.988 + 0.452(S1pt + S1cpt) 0.581 0.530(1 (100S1cpt + 1) 0.506) 0.171(S2pt) 0.693 0.296(S2cpt)0.962 Distribution Shift

3.0

2.9

+ LR Re-Warmup

2.8

2.7

2.6

2.5

10000 20000 30000 40000 50000 60000

Step

(i) Dcpt Validation Loss for adding S2 power format.

- Figure 20. Using other possible S2 formats Eq. 4 to fit all PT and CPT loss curve with different LRS.

|1.853Dcpt - 0.546Dpt|
|---|

|1.631Dcpt - 0.609Dpt|
|---|

|0.475Dcpt + 0.611Dpt|
|---|

|1.2Dcpt - 0.078Dpt|
|---|

Open-web-math

StackExchange

###### Books

###### Arxiv

|0.152Dcpt + 0.867Dpt|
|---|

|0.385Dcpt + 0.622Dpt|
|---|

|-0.516Dcpt + 1.409Dpt|
|---|

|1.505Dcpt - 0.178Dpt|
|---|

Pile-PhilPapers

SlimPajama

Stories

###### C4

- Figure 21. The linear combination of Dpt and Dcpt to represent the out-of-doamin Dood validation loss.

0.0030

0.0025

AbsoluteErrors

0.0020

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

0.0015

0.0010

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

0.0005

0.0000

StackExchange Arxiv Books Math C4 SlimPajama Stories PhilPapers

Figure 22. The absolute errors of linear combination of Dpt and Dcpt to represent the out-of-doamin Dood validation loss.

