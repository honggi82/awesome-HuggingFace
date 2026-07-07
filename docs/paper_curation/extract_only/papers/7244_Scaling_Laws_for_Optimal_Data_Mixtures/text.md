## arXiv:2507.09404v2[cs.LG]2Oct2025

# Scaling Laws for Optimal Data Mixtures

Mustafa Shukor∗, Louis Bethune†, Dan Busbridge†, David Grangier†, Enrico Fini†, Alaaeldin El-Nouby†, Pierre Ablin†

∗Sorbonne University , †Apple

Large foundationmodelsare typicallytrainedondatafrommultipledomains, with the data mixture–the proportion ofeach domain used–playing a critical role in model performance. The standard approach to selecting this mixture relies on trial and error, which becomes impractical for large-scale pretraining. We propose a systematic method to determine the optimal data mixture for any target domain using scaling laws. Our approach accurately predicts the loss of a model of size N trained with D tokens and a specific domain weight vector h. We validate the universality of these scaling laws by demonstrating their predictive power in three distinct and large-scale settings: large language model (LLM), native multimodal model (NMM), and large vision models (LVM) pretraining. We further show that these scaling laws can extrapolate to new data mixtures and across scales: their parameters can be accurately estimated using a few small-scale training runs, and used to estimate the performance at larger scales and unseen domain weights. The scaling laws allow to derive the optimal domain weights for any target domain under a given training budget (N,D), providing a principled alternative to costly trial-and-error methods.

Correspondence: Pierre Ablin: p_ablin@apple.com Date: October 3, 2025

Model sizes: 106M 275M 616M 2.3B 8B

3.6

Training runs

h∗Add

3.6

Evaluation runs

h∗Joint

3.4

Scaling law prediction

3.4

Extrapolation

Loss

Loss

3.2

3.2

[Figure 1]

3.0

3.0

2.8

2.8

1020 1021

1020 1021

FLOPs

FLOPs

Figure1 Scaling Laws for Optimal Data Mixtures. Left: We derive scaling laws that predict the loss of a model as a function of model size N, number of training tokens D, and the domain weights used to train the model (represented by the color of each point). The scaling law is fitted with small-scale runs with different domain weights, and used to predict accurately the loss of large-scale models trained with new, unseen domain weights. Right: We find the data mixture scaling law based on small-scale experiments (e.g., below 1B parameters) and use it to predict the optimal data mixture at larger scales (e.g., 8B parameters). Both our additive (equation (2.4)) and joint (equation (2.5)) laws lead to similar performance, and better than other mixtures (in the gray area). FLOPs are computed as 6ND.

Apple and the Apple logo are trademarks of Apple Inc., registered in the U.S. and other countries and regions.

### 1 Introduction

Modern machine learning models (Brown et al., 2020; Dubey et al., 2024; Fini et al., 2024) are pre-trained on diverse data domains, such as text for LLMs, images for vision models, and mixed modalities for multimodal models. For LLMs, these domains encompass general knowledge, code, reasoning, multilingual content, and more (Grangier et al., 2024; Dubey et al., 2024; Team et al., 2023; Bai et al., 2023; Team et al., 2024). Multimodal models (McKinzie et al., 2024; Zhang et al., 2024b; Laurençon et al., 2024; Abdin et al., 2024; Shukor et al., 2025) are trained on a mix of text, paired, and interleaved multimodal data, and finally, large vision models are trained on image domains of different qualities, containing or not images paired with text (Oquab et al., 2023; El-Nouby et al., 2024; Fini et al., 2024).

The domain weights determine the proportion of each domain used during training, significantly impacting model performance. However, these weights are typically chosen through ad-hoc trial and error, involving training with different domain weights and selecting what works best (Dubey et al., 2024; Shukor et al., 2025; McKinzie et al., 2024). Despite their critical role, a principled method for selecting domain weights is largely absent.

Scaling laws provide a theoretical framework to predict model performance. Initially developed for LLMs (Kaplan et al., 2020; Hernandez et al., 2021; Hoffmann et al., 2022), these laws model the loss of a model as a function of the number of parameters N and training tokens D. This framework has been extended to other domains and modalities (Shukor et al., 2025; Aghajanyan et al., 2023) and to account for factors such as the number of experts in mixture-of-experts models (Krajewski et al., 2024), sparsity (Abnar et al., 2025), data repetitions (Muennighoff et al., 2023), fine-tuning tokens (Zhang et al., 2024a; Bethune et al., 2025), and learning rate schedules (Luo et al., 2025).

In this work, we extend scaling laws to model the effect of domain weights on model performance. We show that the model loss depends in a predictable way on the domain weights, interacting with the number of training tokens and model parameters. We extensively validate our scaling laws in three large-scale settings: large language models (LLMs), native multimodal models (NMMs), and large vision models (LVMs) pretraining. We train large models - up to 7B parameters and 150B tokens for LLMs, 8B parameters and 160B tokens for NMMs, and 1B parameters with 330B tokens for LVMs. across multiple text, multimodal, and image domains. The key takeaways from our work are:

Scaling laws that extrapolate. We demonstrate that our scaling laws can be fitted using small-scale runs, and then provide an accurate prediction of the loss of large-scale models trained with new, unseen domain weights. This is illustrated by figure 1, left, where we report the loss on text when training NMMs. As expected, more text data helps reduce that loss. Our scaling laws precisely quantify this phenomenon.

Optimal domain weights estimation. Once fitted, these scaling laws give an accurate estimation of the loss as a function of the domain weights. Minimizing this estimation gives us optimal domain weights. This approach provides a principled alternative to the costly practice of trying different domain weights and selecting the best one. This is illustrated by figure 1, right, where we report the average loss of NMMs.

This paper is organized as follows. In section 2, we introduce the problem of domain weight selection and describe our scaling law formulations. In section 3, we detail the model architectures and data domains for LLM, NMM, and LVM pretraining. section 4 demonstrates that our scaling laws accurately extrapolate to new domain weights, larger model sizes and number of tokens. section 5 shows how the fitted laws can be used to estimate the optimal domain weights. with a few small-scale runs. Finally, section 6 explores various aspects of our scaling laws, including showing that we need a small number of different domain weights to get a satisfying estimation, how the best domain weights change when scaling flops, as well as alternative scaling laws formulations. Finally, we discuss related works in section 7.

### 2 Data mixture scaling laws

#### 2.1 Problem setup

We consider training models with data coming from k data domains D1,...,Dk; we can query random samples x from any domain Di. Consequently, we can sample from the law mix(h) = ki=1 hiDi for any domain weights h, following the law p(x|mix(h)) = ki=1 hip(x|Di). Here, h is a k−dimensional vector of positive entries that sum to one, that is, an element of the simplex ∆k. In plain words, data is sampled from the domain i with probability hi. We have a target domain DT, which can be one of the training domains. We consider a model with N parameters, represented with the vector of parameters θ ∈ RN. Finally, we have a loss function ℓ(x,θ) defined for any x in the data domains Di or the target domain DT. Note that the target DT does not have to be one of the training domains Di. This allows us to define the loss for any domain weights h, as well as the target loss, as the expectations

Lh(θ) = Ex∼mix(h) [ℓ(x,θ)] and LT(θ) = Ex∼D

T

[ℓ(x,θ)] (2.1)

The training of the model, with fixed domain weights h, is done by running an optimization algorithm such as Adam to approximately minimize Lh. In the course of its execution, the optimization algorithm processes D tokens and outputs trained parameters θ∗(h,D) of size N. The goal of this paper is to predict the loss on the target domain DT after training a model of size N with D tokens with domain weights h; a quantity denoted as L(N,D,h) defined as LT(θ∗(h,D)). In practice, we can, of course, have several target domains that capture different aspects of a model’s capabilities. In that case, we estimate the target loss on all of the target domains by fitting multiple scaling laws.

This framework is sufficiently general to encompass various model architectures and modalities. In this work, we consider different domains to be either various text domain datasets, various image domains, different modalities (e.g., image and text), or different data types (e.g., paired or interleaved).

#### 2.2 Scaling laws derivation

In their original form, scaling laws allow us to predict the training loss of a model of a given size N after having been trained on D tokens (Kaplan et al., 2020). The Chinchilla scaling law models training loss as an additive power law (Hoffmann et al., 2022):

B Dβ

A Nα

, (2.2)

L(N,D) = E +

+

where E,A,α and β are parameters that depend on the training set, model’s architecture, and optimization algorithm. We depart from these original scaling laws in two ways: i) we consider the loss of a model on a target domain that need not be the training domain, and more importantly, ii) we quantify the impact of the domain weights h on the loss. Regarding i), as already been shown in several works (Mikami et al., 2022; Hernandez et al., 2021; Ghorbani et al., 2021; Shukor et al., 2025), the loss on a target domain can still be modeled by a scaling law of the form equation (2.2). Hence, for every domain weights h used for training, we expect the loss on the target domain to follow a Chinchilla power law, where the coefficients depend on h. In other words, the loss on the target domain can be expressed as:

Ah Nαh

L(N,D,h) = Eh +

Bh Dβh

. (2.3)

+

The question now is, how do the parameters Eh,Ah,αh,Bh and βh depend on h? We propose two different formulas that use simple parametric representations for these parameters. We first study the additive scaling law, in which only Eh is modeled as a function of h, while the other parameters Ah,αh,Bh and βh are taken as constants:

A Nα

1

B Dβ

. (2.4)

L = E +

+

+

k i=1 Cihiγ

i

The parameters of the scaling law are Z = (E,A,B,α,β,(Ci)ki=1,(γi)ki=1), which depend on the model architecture, the target and the source domains. This scaling law has 5 + 2k parameters. Since this scaling

law is additive, the optimal domain weights h∗ that minimize it are independent of the model size N and the number of tokens D.

In order to capture the interaction between scale and mixture, we also propose the joint scaling law:

1

L = E +

k i=1 Cihγ

i

i

Ah Nα

+

k

Bh Dβ

with Ah = (

+

i=1

A

CiAhi)γ

k

and Bh = (

i=1

B

CiBhi)γ

(2.5)

In that scaling law, we consider the same dependency in h for the bias term E as in the equation (2.4), and we additionally model the terms Ah and Bh as simple functions of h. The parameters of the law are Z = (E,α,β,(Ci)ki=1,(γi)ki=1,(CiA)ki=1,γA,(CiB)ki=1,γB), which gives 5 + 4k parameters. In this law, there is an interaction between N, D and h, in the sense that ∂

2L

2L

∂D∂h ̸= 0, while these partial derivatives are 0 for the additive scaling law. This law predicts that the contribution of N and D to the loss depends on the domain weights, and as such, the optimal domain weights are compute-dependent. The joint scaling law is more expressive than the additive scaling law, since we can recover equation (2.4) by taking γA = γB = 1 and CiA = A, CiB = B for all domains i. As such, if the scaling laws are fitted properly, the error on the training runs is always lower for the joint scaling law than for the additive scaling law. The joint scaling law still models the terms αh and βh as constants. We tried modelling these terms using the same simple parametric form depending on h, but it never yielded any significant improvement (see section 6). On the other hand, going from the additive to the joint law often decreased the estimation error significantly. Hence, we restrict the bulk of our study to these two laws.

∂N∂h ̸= 0 and ∂

#### 2.3 Fitting the scaling laws

In order to fit the scaling laws, we launch several training runs with different domain weights h, model sizes N, and number of tokens D, and record the loss on the target domain LT. We train model sizes and number of tokens that are evenly spaced. We chose the training domain weights by taking a grid of evenly spaced points in the simplex, where each domain weight is above a minimal value (i.e., 0.1). We have p input-target pairs (Nj,Dj,hj),LjT for j = 1,...,p where p is the number of training runs.

The optimal parameters Z∗ are obtained by minimizing the standard Huber loss:

10−5

Scalinglawﬁttingloss

Basin-hopping

p

1 p

L-BFGS

Huber LjT − L(Nj,Dj,hj;Z) , (2.6)

H(Z) =

j=1

2

where Huber(x) = x

2 if |x| < δ and δ(|x| − 2δ) otherwise, where δ is a hyperparameter which we take as δ = 1e-3.

10−6

The standard technique to fit scaling laws consists of using LBFGS (Liu and Nocedal, 1989) to minimize the loss starting from an evenly-spaced grid of initial parameters Z and then retaining the smallest local minimum. Contrary to most scaling laws that involve only 5 parameters, our scaling laws have 5+2k or 5+4k parameters, where k is the number of domains. In our experiments, we use up to k = 8 domains, which gives 37 parameters to fit. This increased dimensionality makes the standard technique to fit scaling laws cumbersome. We propose two changes that lead to good fit. Firstly, we use a random search, to sample the initial parameters Z. Secondly, we use the Basin-hopping algorithm (Wales and Doye, 1997) instead of L-BFGS to explore the minimizers of the loss function. The Basin-hopping algorithm itself uses L-BFGS as an inner routine to minimize the loss function, but it also uses a random walk to explore the space of local minima. figure 2 gives an example of the performance of the algorithm: to reach a low fitting loss, the Basin-hopping algorithm requires far fewer calls to L-BFGS than doing a random search over the L-BFGS initializations.

102 103 104

# Calls to L-BFGS

Figure 2 Value of the Huber loss (2.6) as a function of the number of L-BFGS calls to fit equation (2.5) on the Interleaved domain from the multimodal experiment (p = 1062 input-target pairs, k = 3 domains). We repeat 100 random trials, the bold line is the median, and the shaded regions are the 25-75% quantiles. The Basin-hopping method with L-BFGS subroutine converges faster than repeated calls to L-BFGS.

In order to evaluate the scaling law, we take a new set of runs that give different values of (N,D,h), and compare the loss on those runs predicted by the scaling law against the actual loss achieved by the model. We quantify this with the Mean Relative Error (MRE), computed as |prediction − observation|/observation, and we report it as a percentage.

### 3 Experimental setup

We give an overview of the models and domains used in our experiments. Detailed architectures and hyperparameters are given in section A.

#### 3.1 Pretraining of large language models (LLMs)

Models. We use transformers (Vaswani, 2017) for autoregressive language modelling. We use the same setup as llama (Touvron et al., 2023), with rotary positional embeddings, SwiGLU activations, and RMSNorm. The models are scaled by changing the latent dimension, with model sizes ranging from 186M to 7B parameters.

For some smaller-scale analyses, we also use GPT2-style transformers (Radford et al., 2019) to perform autoregressive language modeling with model sizes ranging from 90M to 3B parameters.

Datasets. For the main experiments, we use the k = 7 domains from slimpajama (Soboleva et al., 2023). We use these domains as distributed by the authors, without any additional data filtering. For some smaller-scale analyses, we use up to k = 8 domains coming from the Pile dataset (Gao et al., 2020): Wikipedia, StackExchange, GitHub, pg19, arxiv, free law, openwebtext, and PubMed Central.

#### 3.2 Pretraining of native multimodal models (NMMs)

Models. We pretrain native multimodal models (NMMs), based on an early-fusion architecture (Bavishi et al., 2023; Shukor et al., 2025) and follow the design and implementation proposed in (Shukor et al., 2025). The model consists of a single transformer (Vaswani, 2017) without a separate vision encoder, resulting in the same architecture used for LLMs. The model processes a sequence of interleaved text and image tokens. Text tokens are obtained using a standard LLM tokenizer, while image tokens are obtained by patchifying the image and applying a linear projection. Images are resized to 224×224 resolution with a 14×14 patch size. The overall model architecture is aligned with (Li et al., 2024), incorporating SwiGLU FFNs (Shazeer, 2020) and QK-Norm (Dehghani et al., 2023).

Datasets. Following previous works (Shukor et al., 2025; Laurençon et al., 2024; Lin et al., 2024) we train on a mixture of multimodal datasets, covering k = 3 data types: (1) text-only data from DCLM (Li et al., 2024), (2) interleaved multimodal documents from Obelics (Laurençon et al., 2024), and (3) paired image-caption datasets from DFN (Fang et al., 2023), COYO (Byeon et al., 2022), and a private collection of High-Quality Image-Text Pairs (HQITP).

##### 3.3 Pretraining of large vision models (LVMs) Models. We pretrain large vision models with a multimodal objective, following the AIMv2 recipe (Fini

- et al., 2024). Unlike traditional language modeling or multimodal models described above that focus on text decoding, AIMv2 trains a vision encoder using an autoregressive objective on both image and text tokens. The model architecture is composed of a vision encoder and a multimodal decoder stitched together in a late-fusion fashion. Datasets. We train on a mixture of paired of image-caption datasets drawn from four domains (k = 4):

- (1) noisy alt-text sourced from the Internet, including COYO-700M (Byeon et al., 2022) and DFN2B (Fang et al., 2023), which provide large-scale real-world image-text pairs with varying levels of noise and quality;
- (2) HQ-ITP-1, a high-quality dataset containing 134 million samples; (3) HQ-ITP-2, another high-quality dataset comprising 400 million samples; and (4) synthetic data, consisting of recaptioned versions of DFN2B and HQ-ITP-2.

N≤1.4B (train) N: 3B (eval) N≤1B (train) N: 2B (eval) N: 8B (eval) N≤530M (train) N: 1B (eval)

LLM, Additive Law

NMM, Additive Law

LVM, Additive Law

3.6

[Figure 2]

[Figure 3]

[Figure 4]

PredictedLoss

2.75

3.4

3.0

2.50

3.2

2.5

2.25

3.0

2.00

2.8

2.0

2.00 2.25 2.50 2.75

2.8 3.0 3.2 3.4 3.6

2.0 2.5 3.0

Observed Loss

Observed Loss

Observed Loss

LLM, Joint Law

NMM, Joint Law

LVM, Joint Law

3.6

[Figure 5]

[Figure 6]

[Figure 7]

PredictedLoss

2.75

3.4

3.0

2.50

3.2

2.5

2.25

3.0

2.00

2.8

2.0

2.00 2.25 2.50 2.75

2.8 3.0 3.2 3.4 3.6

2.0 2.5 3.0

Observed Loss

Observed Loss

Observed Loss

- Figure 3 Observed vs predicted loss for LLM pretraining on domains from the slimpajama dataset, NMM pretraining with multimodal domains, and LVM pretraining with image-caption domains. The scaling laws are fitted on small-scale models (blue points in the figure) and extrapolated to larger models. We display here the average loss over all domains for each modality. The MRE% for each domain is reported in table 2.

#### 3.4 Implementation details

In order to scale models, we change the hidden dimension size in the transformers d, keeping a fixed number of layers. To reduce the experimental cost, most of the experiments are done with a constant learning rate scheduler. This allows us to collect many points with varying numbers of tokens D for each run, instead of one per run, which means that we can run more experiments and explore the space of domain weights more thoroughly. We also validate our findings when using cosine learning scheduler in section 6, where we show that the scaling laws also extrapolate from small-scale to large-scale behavior in that case.

### 4 Predicting large-scale performance from small-scale experiments

In this section, we demonstrate that (a) our scaling laws accurately capture the training data, and (b) generalize effectively to larger scales with significantly increased values of N and D. To this end, we fit the laws using small models trained with a small number of tokens, and we validate them on larger models with a large number of tokens. We experiment with LLMs, trained with a mixture of text domains, NMMs, trained with a mixture of multimodal domains, and LVMs, trained with images of different qualities and paired or not with text. For LLMs, we consider the k = 7 domains from slimpajama, which are different text domains. For multimodal pretraining, and similar to previous works, (Laurençon et al., 2024; Zhang et al., 2024b; Shukor

- et al., 2025), the data mixture spans k = 3 different domains: text, paired (image-captions), and interleaved multimodal data. For large vision model pretraining, we use k = 4 domains.

table 1 displays the different model sizes, number of training tokens, and number of different domains weights that we use to train and evaluate the scaling laws.

Results. figure 3 presents a comparison between the actual loss achieved by our trained models and the loss predicted by our scaling laws. We summarize the results on each domain by showing the average predicted loss (full results in section B). Remarkably, the predicted losses align closely with the observed values for both

Table 1 Experimental setup for the extrapolation experiment.

N D # domain weights FLOPs

412M 4-20B 60

834M 7-26B 40 1.1B 7-36B 40 1.4B 13-46B 20

Train

2.5 × 1022

LLM

3B 20-100B 4 7.2 × 1021 7B 150B 1 6.3 × 1021

Eval

106M 20-100B 32

275M 20-100B 31 616M 20-100B 30 932M 20-100B 34

Train

3.7 × 1022

NMM

2.3B 100-160B 8 1.8 × 1022 8B 100-160B 4 3.1 × 1022

Eval

89M 17-50B 32

157M 17-50B 32 306M 17-100B 15 531M 17-170B 16

Train

1.4 × 1022

LVM

Eval 1.1B 17-334B 8 1.8 × 1022

###### Table 2 Scaling laws MRE

MRE% (Additive / Joint law) Train Val

Modality Target domain

Arxiv 0.50 / 0.39 2.09 / 1.62 book 0.29 / 0.24 0.80 / 1.19

C4 0.29 / 0.24 0.31 / 0.34

Github 0.65 / 0.54 1.17 / 2.51 Commoncrawl 0.29 / 0.24 0.58 / 0.90 Stackexchange 0.51 / 0.38 0.36 / 0.47

Language

Wikipedia 0.92 / 0.57 4.45 / 2.09

Image-Caption 0.47 / 0.43 0.95 / 0.95 Interleaved 0.14 / 0.10 0.48 / 0.42 Text 0.12 / 0.10 0.44 / 0.37

Multimodal

Noisy image-text 0.35 / 0.23 1.20 / 0.63 Synthetic 1.89 / 0.83 6.19 / 5.94

Vision

- High-quality 1 1.19 / 0.34 2.02 / 1.18
- High-quality 2 0.64 / 0.31 0.79 / 1.06

the joint and additive laws. In addition, the laws show good extrapolation to larger model sizes. To further quantify this alignment, we report the mean relative error (MRE%) in table 2, which reveals a consistently low MRE% for both laws, with an improvement of the joint law over the additive one. These results demonstrate that we can fit the scaling laws on small scales and extrapolate to larger scales. We remark that the MREs have some degree of variability across domains; for instance, on the LLM experiment, we get at the same time an extremely low MRE of 0.31% on the C4 dataset and a high MRE of 4.45% on Wikipedia.

FLOPs count. We report the approximate computational cost of running the small-scale experiments to fit the scaling laws, and contrast it with the cost of large-scale runs in table 1. We compute the FLOPs required for one run as 6ND. We observe that the cost of large-scale runs is comparable to that of small-scale runs. We also note that our large-scale runs could be trained with far more tokens, which would increase their FLOPs.

### 5 Optimal data mixtures

Optimal domain weights estimation Once the scaling law is fitted, we can derive the optimal domain weights h∗ that minimize it, by solving the following optimization problem on the simplex:

L(N,D,h) (5.1)

min

h∈∆k

This is an optimization problem on the simplex, which we solve using mirror descent, i.e. iterating ht+1 =

- hˆt

- i

hˆti where hˆt = ht × exp(−η∇hL(N,D,h)), with η a small step size. In practice, we may want to obtain a model that works well on several tasks at once, with weights w. In that case, we have m different target domains DT1 ,...,DTm. We can estimate the scaling law for each target domain DTi and obtain m different scaling laws Li(N,D,h). We obtain the optimal domain weights h∗ that are good on average by solving the following optimization problem:

h∗(N,D) ∈ arg min h∈∆k

m

Li(N,D,h). (5.2)

i=1

The behavior of h∗ depends on the scaling law that we consider. Since equation (2.4) assumes an additive relationship, the minimizer of equation (5.2) is independent of N,D; in other words, it does not depend on scale. On the other hand, equation (2.5) takes into account a multiplicative interaction between N,D and h. Therefore, the optimal h is scale-dependent. If one task is more important than another, we can incorporate importance weights in the sum in equation (5.2).

Average Loss on the training domains

OpenHermes Loss

| |Optimal for Average<br><br>Optimal for OpenHermes<br><br>Uniform distribution<br><br>Base slimpajama distribution| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Optimal for Average

2.1

1.5

Optimal for OpenHermes

Uniform distribution

2.0

1.4

Base slimpajama distribution

Loss

Loss

1.9

1.3

1.8

1.2

1.7

20 40 60 80 100 120 140

20 40 60 80 100 120 140

Billion Training Tokens

Billion Training Tokens

- Figure 4 Losses of the 7B models. After fitting the scaling laws on the small scale runs, we estimate the optimal

domain weights h∗avg that minimize the average loss over the training domains (left), and h∗OH that minimizes the loss on the OpenHermes dataset (right). We then train 7B models with these optimal weights, and compare them to two baselines: one with uniform weights, and one with the standard distribution of slimpajama. The losses are averaged over all training domains, and also reported on the OpenHermes dataset. As expected, the model trained with h∗OH performs best on OpenHermes, while the model trained with h∗avg performs best on the training domains.

The main practical takeaway of our paper is this simplified approach to optimal mixture estimation. Indeed, as demonstrated in section 4, we can accurately fit our scaling laws with small-scale runs. Using these scaling laws, we can then solve equation (5.2) for various targets (N,D), which gives a principled way of choosing domain weights, rather than using ad-hoc methods as usually done in practice. To demonstrate our point, we do this for different modalities considered in the paper

LLM results. Since the additive scaling law gave us the lowest MREs, we use it to estimate the optimal data mixture that minimizes the average loss over the k = 7 training domains, which we denote h∗avg We then train a 7B model with 150B tokens with that optimal data mixture.

For all the runs, we also monitor the loss on the OpenHermes dataset, which is a small high-quality dataset used for model alignment. We fit the scaling laws for that domain as well, even though this domain is not part of the pre-training domains. The rationale is that we want to estimate weights that lead to the best performance on this high quality dataset, which should be a proxy of performance on downstream tasks. We then find the optimal domain weights for that scaling law, which we denote h∗OH, and train another 7B model with 150B tokens. As baselines, we train two more 7B models with that many tokens, one with the standard distribution of slimpajama, proportional to the number of tokens in each domain, and one with a uniform distribution over domains.

Since we want the best models possible, we use a cosine learning rate schedule, which makes the scaling law extrapolation impossible to conduct. We report the average loss of these models on the training domains, on the OpenHermes dataset in Figure 4. We also evaluate them on many downstream tasks and report the results in Table 3. The model trained with weights h∗OH is overal better than the other models in terms of evaluations. We believe that the pipeline demonstrated in this paper — estimate the scaling law for the loss on a high quality domain with small scale runs, find the minimizer, train a large scale model with it — is a promising avenue to get better models.

NMM results. We fit both the additive and joint scaling laws on three multimodal data domains, using only small models. For the joint scaling laws, we predict the best training mixture h∗ that minimizes the average of the domain losses for each model size while fixing the number of tokens at 100B for practicality. We then train models with these optimized mixtures. figure 1 compares the performance of these best mixtures against uniform mixtures, those used in prior works (Shukor et al., 2025; McKinzie et al., 2024), and randomly sampled mixtures that cover an important area of mixture grid. Models trained with our estimated mixtures consistently outperform alternatives. Notably, both additive and joint laws perform similarly well, making additive laws a strong and more practical baseline, since it takes the same optimal mixture for all runs. Remarkably, the optimized mixtures generalize effectively to larger model sizes, which validates the possibility

Table 3 Evaluations of the 7B models. We report the median score on the CORE tasks (Li et al., 2024) as well as other accuracies on standard benchmarks. The model trained with weights that attempt to minimize the loss on OpenHermes performs best.

Weights CORE MMLU Arc-easy Arc-challenge Boolq Piqa Siqa Hellaswag Winogrande

h∗avg 56 32 71 40 60 79 52 72 65 h∗OH 58 37 70 40 67 79 53 72 65 Uniform 53 30 70 40 46 78 51 70 65 Base 52 25 70 43 51 79 53 71 65

Multimodal, 3 domains

LLM, 4 domains

LLM, 6 domains

LLM, 8 domains

- 100

- 101

Additive law

Additive law

Additive law

Additive law

EvalMRE%

Joint law

Joint law

Joint law

Joint law

100

100

100

10−1

5 10 15

5 10 15

0 20 40

0 20 40

# training domain weights

# training domain weights

# training domain weights

# training domain weights

Figure 6 Evaluation of the scaling law as a function of the number of training runs. We randomly select q different domain weights htrain = [h1, . . . , hq], and only use the runs that use these histograms to fit the scaling laws. We then evaluate the MRE on all the domain weights htest that are not part of htrain. For the multimodal and LLM with 4 domains, we compute the eval MRE on the large-scale (resp. 1B and 8B) models. For the LLM with 6 and 8 domains, we compute the eval MRE on same size models.

of choosing the optimal mixture based on small-scale experiments, and then extrapolating to larger scales.

LVM results. We fit the scaling laws on the AIMv2 data mixture, which consists of 4 domains, and we estimate the optimal domain weights that minimize the average loss over these domains. We then train a 1B model with these optimal weights, and compare it to a model trained with uniform weights. We find that the model trained with the optimal weights performs better than the one trained with uniform weights, which validates our approach of estimating the optimal domain weights from small-scale runs.

Image-Caption

### 6 Scaling laws analysis

Model size N

- 1011

- 1012

- 1013

30M 300M

[Figure 8]

In that section, we conduct LLM experiments in a different setting, using the Pile dataset (Gao et al., 2020), which allows us to have a variable number of domains, between k = 4 and k = 8.

| |
|---|

NumberoftokensD

3B

1/3

30B

300B

| |
|---|

Only 10-20 runs are needed to fit the scaling laws. We investigate the number of runs needed for an accurate scaling law fitting. We randomly partition the domain weights into htrain = [h1,...,hq], of size q, and put the other domain weights into htest. We fit the scaling law on htrain and report the MRE on the test domain weights. Since the number of parameters of the scaling law depends linearly on the number of domains k, we expect the number of runs necessary to fit the scaling law to increase when we consider more domains. To verify this hypothesis, we consider the NMM pretraining experiments, with k = 3 domains and LLM pretraining with k = 4,6,8 domains. For the NMM with 3 domains and LLM with 4 domains as considered so far, we fit the laws on small-scale models

1/3

Additive scaling law 1010

Text Interleaved

1/3

Figure 5 Evolution of optimal domain weights h∗ with compute budget (N, D) on the multimodal data, as predicted by the joint scaling law (equation (2.5)).

###### Table 4 Other Scaling laws average MRE%

Simple Additive Joint Full

NMM 0.70 0.62 0.58 0.60 LLM 1.70 1.39 1.30 1.21 LVM 2.31 2.55 2.21 2.04

and compute the MRE on large-scale models as in section 4. For the LLM training with k = 6,8 domains, because of the very large search space with a high number of domains, we take a single model size and skip the dependency on N in the scaling law, only considering the dependency on h and the number of training tokens D. We report the MRE as a function of the number of training histograms q in figure 6. We observe that we need about 10 runs for the NMM and LLM with 4 domains to get to an optimal MRE, while we need about 20 for LLM with 6 and 8 domains. Interestingly, we observe that when the number of training runs is very low, the additive law has a slightly lower eval MRE, due to its lower number of parameters.

Optimal domain weights behavior when scaling FLOPs. We study how the optimal mixture h∗(N,D) for the average loss evolves as a function of the compute-budget (N,D) on the multimodal models, as predicted by the joint scaling law. We report results in figure 5. We see that interleaved data gets less important as we increase D, whereas bigger models tend to rely more on text. The additive law captures the average behavior across all scales.

Cosinelearningratescheduler. The bulk of our experiments use a constant learning rate, which helps us gather many D points for each run, but this departs from what is done in practice when training competitive models, where a cosine learning rate is typically used. In order to validate that our scaling laws are still valid when training with a cosine learning rate, we repeat the LLM experiments with k = 4 pile domains with cosine learning rate decays, with fewer runs, training for 5 different values of D, with 25 different domain weights. We use 90M,200M,350M and 700M models for training, and extrapolate to 1.3B. We observe that our scaling law fit is similar to those in the main experiments: on the 1B model, we get to an average MRE of 0.76% for the additive law and 0.54% for the joint law. We report detailed results in section B. Interestingly, the estimated optimal domain weights for the average loss are very similar to those estimated with the constant learning rate: for the additive law, we have h∗cos = [0.35,0.18,0.30,0.17] and h∗const = [0.34,0.17,0.32,0.17].

Other scaling laws formulas. We investigate alternative scaling laws, and validate our proposed scaling laws, by evaluating other formulas in the same setup as in section 4. First, we want to understand whether we could use a simpler form for the dependency on the domain weights h. To do so, we use the “simple additive” scaling law

k

A Dα

B Nβ

Cihi)γ +

, (6.1)

L = E + (

+

i=1

where the dependency in h is simpler compared to the additive and joint scaling laws. This law has k−1 fewer parameters than the additive scaling law. The joint scaling law models the terms Ah and Bh as functions of domain weights. We want to understand whether also taking a dependency of α and β on h helps capture more information about models’ behavior. To do so, we consider the “full” scaling law:

k

Ah = (

i=1

A

CiAhi)γ

Ah Nαh

1

L = E +

+

k i=1 Cihγ

i

i

k

k

B

, Bh =(

CiBhi)γ

, αh = (

i=1

i=1

Bh Dβh

+

α

Ciαhi)γ

, with (6.2)

k

β

Ciβhi)γ

and βh = (

i=1

(6.3)

This law is more expressive than the joint scaling law, and it adds 2k +1 parameters. We give the full results of those laws in section B, and report the average MRE in table 4. We see that the additive law reduces the MRE a lot compared to the simple law, especially in the LLM experiment. Despite having a slightly smaller train MRE, the full law does not extrapolate as well as the joint law. For the LVM experiment, the

picture is different: the simple law works better than the additive law, and the full scaling law brings some improvement over the joint law.

Asymptotic behavior. We can get an information-theoretic explanation of the bias term in the scaling laws. Let p be the true data distribution of the target domain. Let q(h) be the output distribution of a model of size N → ∞ trained for D → ∞ tokens with domain weights h. Let h∗ be the optimal domain weights, which minimizes the scaling law L(+∞,+∞,h) = E + ( ki=1 Cihγ

i )−1, i.e the cross-entropy term CE(p,q(h)) = H(p) + KL(p∥q(h)), with H(·) the Shannon entropy and KL(·∥·) the Kullback-Leibler divergence. We have the following decomposition:

i

+KL(p∥q(h)) − KL(p∥q(h∗))

CE(p,q(h)) = H(p) + KL(p∥q(h∗)) constant, independant of h

≥0 by hypothesis on h∗

k

k

k

Cih∗i γ

Cih∗i γ

Cihiγ

)−1 +(

)−1 − (

)−1.

i

i

= E + (

i

i=1

i=1

i=1

(6.4)

We can identify both terms since they are of the form “constant plus non-negative function that cancels”. We see that E + ( ki=1 Cih∗i γ

)−1 captures both the intrinsic entropy H(p) of the target distribution, and the shift KL(p∥q(h∗)) induced by training on the optimal mixture h∗, while the right-hand term is the expected log-likelihood ratio Ep[log q(h

i

∗)

q(h) ], which measures how far the model trained on h is from the optimal one. If p is one of the training domains Di, for disjoint domains we can assume h∗i ≈ 1 (see section C for justification) and simply bound its entropy H(p) ≤ E + Ci−1.

### 7 Related works

Scaling laws. Scaling laws research investigates how model performance varies with training compute. Foundational studies (Hestness et al., 2017; Kaplan et al., 2020; Hoffmann et al., 2022) established that language models follow a predictable power-law relationship between performance and compute, allowing for the optimal allocation of parameters and training tokens within a specified budget. Scaling behavior has since been explored across a wide range of domains, including vision models (Fini et al., 2024; Rajasegaran et al., 2025), diffusion transformers (Liang et al., 2024), and other fields (Cheng et al., 2024; Pearce et al., 2024). While typical scaling laws consider the number of total parameters, other studies have examined the influence of both width and depth (McLeish et al., 2025), or the number of parameters allocated to the teacher and student in cse of model distillation (Busbridge et al., 2025). Sparse Mixture of Experts (MoE) models have been another focus, with investigations into how factors like sparsity, the number of experts, and routing strategies affect scaling (Krajewski et al., 2024; Clark et al., 2022; Wang et al., 2024; Abnar et al., 2025). For multimodal models, scaling laws have been explored in studies such as (Aghajanyan et al., 2023; Shukor et al., 2025). Of particular relevance is (Shukor et al., 2025), which examines native multimodal models. However, their analysis is constrained by a fixed pretraining mixture.

Scaling laws for data mixtures Optimizing data mixtures for model training is a critical challenge, often requiring extensive experimentation. Recent studies have begun exploring systematic approaches to identify optimal mixtures more efficiently. For instance, Goyal et al. (2024) investigated scaling laws for data filtering in CLIP models, emphasizing data quality and repetition. Gu et al. (2024) examined scaling laws for continual pretraining of language models, predicting the optimal balance between pretraining and domain-specific data, while Bethune et al. (2025) followed the same approach, focused on forgetting in finetuning. Similarly, Chang et al. (2024) derived scaling laws that account for data quality factors such as diversity. Closer to our work, Ye et al. (2024) and Ge et al. (2024) propose scaling laws that model the loss as a function of h for fixed (N, D), but they do not consider a joint law for (N,D,h) as we do here. We also find that, in our experiments, for a fixed (N, D), our scaling laws extrapolate better to unseen mixtures (see section B). Further, these approaches are generally constrained to a single modality, and they consider relatively small models, below 1B parameters.

Data mixture selection. The standard approach to selecting training data mixtures relies on trial and error, where different combinations are tested to determine the best-performing mixture (Soldaini et al., 2024;

Lin et al., 2024; Laurençon et al., 2024; McKinzie et al., 2024; Zhang et al., 2024b; Fini et al., 2024; Shukor et al., 2023). However, this method is costly, leading to recent efforts exploring alternative strategies. Some studies adopt heuristic methods, adjusting mixture ratios based on data sizes for each domain (Rae et al., 2021; Groeneveld et al., 2024; Chung et al., 2023) or to match a target task’s distribution (Grangier et al.,

- 2024). Others predict model performance using small models that take the mixture as input (Xie et al., 2023; Albalak et al., 2023; Liu et al., 2024; Fan et al., 2023). A third approach employs auxiliary models to rank and select high-quality training data, which has been popularized recently by large foundation models (Wenzek et al., 2019; Brown et al., 2020; Wettig et al., 2024; Penedo et al., 2024; Dubey et al., 2024).

8 Discussion

Limitations. Our current study is focused on pretraining, but continual pre-training and finetuning are also scenarios in which the mixture is important. Our scaling law predicts a generic target loss (Kaplan et al., 2020), which is known to correlate with downstream task performance (Hoffmann et al., 2022; Mayilvahanan et al.,

- 2025). Future work may involve predicting this performance directly, like Isik et al. (2025). Furthermore, assuming no data repetition (i.e., an infinite stream of data from each domain), as is typical for LLM pretraining, is unrealistic when training with very scarce, high-quality sources. Finally, we assume that the mixture is fixed throughout training, but future works may consider a dynamic evolution of the weights (e.g., curriculum learning).

Broader impact. Mixture coefficients have a tremenduous impact on the performance on downstream tasks. Modern training corpora are typically a combination of dozens of sub-domains, striking a balance between diversity and quality. Giving the cost of pre-training, finding the optimal mixture through extensive trials and errors can be prohibitively expensive. Our scaling law only require a few runs at small scale to yield meaningful coefficients for larger models. Our work also has environmental benefits, as it significantly reduces the cost of pre-training, including the amount of CO2 emission and the energy needed. Moreover, it may yield better models in the long run.

Conclusion. We propose a data mixing law that predicts the loss on an arbitrary target domain, from both mixture coefficients and the compute budget (N,D). Our laws hold for language, multimodal, and vision pretraining. The optimal mixture coefficients found from a small scale can be used for much larger models and training budgets, demonstrating significant improvement over domain weights found by naive grid search. This work paves the way to a principled theory of data mixture selection.

### 9 Acknowledgement

We thank Victor Guilherme Turrisi da Costa for technical support. We thank Marco Cuturi, Jason Ramapuram, Denise Hui, Samy Bengio, and the MLR team at Apple for their support throughout the project.

### References

Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.

Samira Abnar, Harshay Shah, Dan Busbridge, Alaaeldin Mohamed Elnouby Ali, Josh Susskind, and Vimal Thilak. Parameters vs flops: Scaling laws for optimal sparsity for mixture-of-experts language models. arXiv preprint arXiv:2501.12370, 2025.

Armen Aghajanyan, Lili Yu, Alexis Conneau, Wei-Ning Hsu, Karen Hambardzumyan, Susan Zhang, Stephen Roller, Naman Goyal, Omer Levy, and Luke Zettlemoyer. Scaling laws for generative mixed-modal language models. In International Conference on Machine Learning, pages 265–279. PMLR, 2023.

Alon Albalak, Liangming Pan, Colin Raffel, and William Yang Wang. Efficient online data mixing for language model pre-training. arXiv preprint arXiv:2312.02406, 2023.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sağnak Taşırlar. Introducing our multimodal models, 2023. URL https://www.adept.ai/blog/fuyu-8b.

Louis Bethune, David Grangier, Dan Busbridge, Eleonora Gualdoni, Marco Cuturi, and Pierre Ablin. Scaling laws for forgetting during finetuning with pretraining data injection. arXiv preprint arXiv:2502.06042, 2025.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, and Russ Webb. Distillation scaling laws. arXiv preprint arXiv:2502.08606, 2025.

Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Imagetext pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.

Ernie Chang, Matteo Paltenghi, Yang Li, Pin-Jie Lin, Changsheng Zhao, Patrick Huber, Zechun Liu, Rastislav Rabatin, Yangyang Shi, and Vikas Chandra. Scaling parameter-constrained language models with quality data. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 80–97, 2024.

Xingyi Cheng, Bo Chen, Pan Li, Jing Gong, Jie Tang, and Le Song. Training compute-optimal protein language models. bioRxiv, 2024. doi: 10.1101/2024.06.06.597716. URL https://www.biorxiv.org/content/early/2024/06/09/2024.06.06.597716.

Hyung Won Chung, Noah Constant, Xavier Garcia, Adam Roberts, Yi Tay, Sharan Narang, and Orhan Firat. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining. arXiv preprint arXiv:2304.09151, 2023.

Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. Unified scaling laws for routed language models. In International conference on machine learning, pages 4057–4086. PMLR, 2022.

Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In International Conference on Machine Learning, pages 7480–7512. PMLR, 2023.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar, Joshua M Susskind, and Armand Joulin. Scalable pre-training of large autoregressive image models. arXiv preprint arXiv:2401.08541, 2024.

Alaaeldin El-Nouby*, Victor Guilherme Turrisi da Costa*, Enrico Fini*, Michal Klein*, Mustafa Shukor*, Jason Ramapuram, Jesse Allardic, Roman Bachmann, David Mizrahi, Vishnu Banna, Chun-Liang Li, Samira Abnar, Vimal Thilak, and Joshua M Susskind. Large multi-modal models (l3m) pre-training, 2025. URL https://github.com/ apple/ml-l3m.

Simin Fan, Matteo Pagliardini, and Martin Jaggi. Doge: Domain reweighting with generalization estimation. arXiv preprint arXiv:2310.15393, 2023.

Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023.

Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor Guilherme Turrisi da Costa, Louis Béthune, Zhe Gan, Alexander T Toshev, Marcin Eichner, Moin Nabi, Yinfei Yang, Joshua M. Susskind, and Alaaeldin El-Nouby. Multimodal autoregressive pre-training of large vision encoders, 2024.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Ce Ge, Zhijian Ma, Daoyuan Chen, Yaliang Li, and Bolin Ding. Bimix: Bivariate data mixing law for language model pretraining. arXiv preprint arXiv:2405.14908, 2024.

Behrooz Ghorbani, Orhan Firat, Markus Freitag, Ankur Bapna, Maxim Krikun, Xavier Garcia, Ciprian Chelba, and Colin Cherry. Scaling laws for neural machine translation. arXiv preprint arXiv:2109.07740, 2021.

Sachin Goyal, Pratyush Maini, Zachary C Lipton, Aditi Raghunathan, and J Zico Kolter. Scaling laws for data filtering–data curation cannot be compute agnostic. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22702–22711, 2024.

David Grangier, Simin Fan, Skyler Seto, and Pierre Ablin. Task-adaptive pretrained language models via clusteredimportance sampling. arXiv preprint arXiv:2410.03735, 2024.

Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. Olmo: Accelerating the science of language models. arXiv preprint arXiv:2402.00838, 2024.

Jiawei Gu, Zacc Yang, Chuanghao Ding, Rui Zhao, and Fei Tan. Cmr scaling law: Predicting critical mixture ratios for continual pre-training of language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 16143–16162, 2024.

Danny Hernandez, Jared Kaplan, Tom Henighan, and Sam McCandlish. Scaling laws for transfer. arXiv preprint arXiv:2102.01293, 2021.

Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, pages 30016–30030, 2022.

Berivan Isik, Natalia Ponomareva, Hussein Hazimeh, Dimitris Paparas, Sergei Vassilvitskii, and Sanmi Koyejo. Scaling laws for downstream task performance in machine translation. In The Thirteenth International Conference on Learning Representations, 2025.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Pióro, Michał Krutul, Szymon Antoniak, Kamil Ciebiera, Krystian Król, Tomasz Odrzygóźdź, Piotr Sankowski, et al. Scaling laws for fine-grained mixture of experts. arXiv preprint arXiv:2402.07871, 2024.

Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Advances in Neural Information Processing Systems, 36, 2024.

Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, et al. Datacomp-lm: In search of the next generation of training sets for language models. arXiv preprint arXiv:2406.11794, 2024.

Zhengyang Liang, Hao He, Ceyuan Yang, and Bo Dai. Scaling laws for diffusion transformers. arXiv preprint arXiv:2410.08184, 2024.

Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689–26699, 2024.

Dong C Liu and Jorge Nocedal. On the limited memory bfgs method for large scale optimization. Mathematical programming, 45(1):503–528, 1989.

Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, and Min Lin.

Regmix: Data mixture as regression for language model pre-training. arXiv preprint arXiv:2407.01492, 2024. Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Kairong Luo, Haodong Wen, Shengding Hu, Zhenbo Sun, Zhiyuan Liu, Maosong Sun, Kaifeng Lyu, and Wenguang Chen. A multi-power law for loss curve prediction across learning rate schedules. arXiv preprint arXiv:2503.12811, 2025.

Prasanna Mayilvahanan, Thaddäus Wiedemer, Sayak Mallick, Matthias Bethge, and Wieland Brendel. Llms on the line: Data determines loss-to-loss scaling laws. arXiv preprint arXiv:2502.12120, 2025.

Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Anton Belyi, et al. Mm1: methods, analysis and insights from multimodal llm pretraining. In European Conference on Computer Vision, pages 304–323. Springer, 2024.

Sean McLeish, John Kirchenbauer, David Yu Miller, Siddharth Singh, Abhinav Bhatele, Micah Goldblum, Ashwinee Panda, and Tom Goldstein. Gemstones: A model suite for multi-faceted scaling laws. arXiv preprint arXiv:2502.06857, 2025.

Hiroaki Mikami, Kenji Fukumizu, Shogo Murai, Shuji Suzuki, Yuta Kikuchi, Taiji Suzuki, Shin-ichi Maeda, and Kohei Hayashi. A scaling law for syn2real transfer: How much is your pre-training effective? In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 477–492. Springer, 2022.

Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel. Scaling data-constrained language models. Advances in Neural Information Processing Systems, 36:50358–50376, 2023.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Tim Pearce, Tabish Rashid, Dave Bignell, Raluca Georgescu, Sam Devlin, and Katja Hofmann. Scaling laws for pre-training agents and world models. arXiv preprint arXiv:2411.04434, 2024.

Guilherme Penedo, Hynek Kydlíček, Anton Lozhkov, Margaret Mitchell, Colin A Raffel, Leandro Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 37:30811–30849, 2024.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021.

Jathushan Rajasegaran, Ilija Radosavovic, Rahul Ravishankar, Yossi Gandelsman, Christoph Feichtenhofer, and Jitendra Malik. An empirical study of autoregressive pre-training from videos. arXiv preprint arXiv:2501.05453, 2025.

Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. Mustafa Shukor, Corentin Dancette, Alexandre Rame, and Matthieu Cord. UnIVAL: Unified model for image, video,

audio and language tasks. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview. net/forum id=4uflhObpcp.

Mustafa Shukor, Enrico Fini, Victor Guilherme Turrisi da Costa, Matthieu Cord, Joshua Susskind, and Alaaeldin El-Nouby. Scaling laws for native multimodal models. arXiv preprint arXiv:2504.07951, 2025.

Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. https://www.cerebras.net/blog/ slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama, 2023. URL https://huggingface.co/datasets/cerebras/ SlimPajama-627B.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al. Dolma: An open corpus of three trillion tokens for language model pretraining research. arXiv preprint arXiv:2402.00159, 2024.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Params 412M 834M 1.1B 1.4B 3B 7B width 1024 1536 1792 2048 3072 4864 depth 24 Learning rate 6e-4 4e-3 3.4e-4 3e-4 2e-4 1.2e-4 Optimizer Fully decoupled AdamW (Loshchilov and Hutter, 2017) Optimizer Momentum β1 = 0.9,β2 = 0.95 Minimum Learning rate 0 Warm up iterations 2k Weight decay 1e-4 Batch size 2M

Table 5 Pre-training hyperparameters used for pre-training of LLM to conduct the main scaling laws study.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. David J Wales and Jonathan PK Doye. Global optimization by basin-hopping and the lowest energy structures of

lennard-jones clusters containing up to 110 atoms. The Journal of Physical Chemistry A, 101(28):5111–5116, 1997.

Siqi Wang, Zhengyu Chen, Bei Li, Keqing He, Min Zhang, and Jingang Wang. Scaling laws across model architectures: A comparative analysis of dense and MoE models in large language models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5583–5595, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.319. URL https://aclanthology.org/2024.emnlp-main.319/.

Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave. Ccnet: Extracting high quality monolingual datasets from web crawl data. arXiv preprint arXiv:1911.00359, 2019.

Alexander Wettig, Aatmik Gupta, Saumya Malik, and Danqi Chen. Qurating: Selecting high-quality data for training language models. arXiv preprint arXiv:2402.09739, 2024.

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy S Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. Doremi: Optimizing data mixtures speeds up language model pretraining. Advances in Neural Information Processing Systems, 36:69798–69818, 2023.

Jiasheng Ye, Peiju Liu, Tianxiang Sun, Jun Zhan, Yunhua Zhou, and Xipeng Qiu. Data mixing laws: Optimizing data mixtures by predicting language modeling performance. arXiv preprint arXiv:2403.16952, 2024.

Biao Zhang, Zhongtao Liu, Colin Cherry, and Orhan Firat. When scaling meets llm finetuning: The effect of data, model and finetuning method. arXiv preprint arXiv:2402.17193, 2024a.

Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, et al. Mm1. 5: Methods, analysis & insights from multimodal llm fine-tuning. arXiv preprint arXiv:2409.20566, 2024b.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

Params 275M 468M 932M 1.63B 2.28B 3.35B 8.13B width 800 1088 1632 2208 2624 3232 5120 depth 24 Learning rate 1.5e-3 1.5e-3 5e-4 4.2e-4 4e-4 3.5e-4 2.4e-4 Training tokens 2.5B-600B Optimizer Fully decoupled AdamW (Loshchilov and Hutter, 2017) Optimizer Momentum β1 = 0.9,β2 = 0.95 Minimum Learning rate 0 Weight decay 1e-4 Batch size 2k Patch size (14, 14) Gradient clipping 1.0 Warmup iterations 1k Augmentations:

RandomResizedCrop size 224px scale [0.4, 1.0]

RandomHorizontalFlip p = 0.5

Table 6 Pre-training hyperparameters used for pre-training of NMM to conduct the scaling laws study.

### A Implementation details

#### A.1 LLM pretraining

For the main experiments, we use LLAMA style architectures. For the analyses with the Pile domains, we borrow architectures from (Brown et al., 2020). We use a fixed depth of 24 and change the latent dimension of the network to obtain different model scales. All hyperparameters are described in table 5 and table 7.

#### A.2 Multimodal pretraining

Implementation details. We closely follow the implementation of (Shukor et al., 2025) and present in table 6 the pre-training hyperparameters for the model configurations used in our scaling laws study. Training is conducted in the L3M code base (El-Nouby* et al., 2025). Our models range from 100M to 8B parameters, with width scaling accordingly while maintaining a constant depth of 24 layers. We use causal attention for text tokens and bidirectional attention for image tokens. Learning rates are adjusted based on model size, generally decreasing for larger models. These values were determined through empirical testing. Optimization is handled using a fully decoupled AdamW optimizer with momentum values set to β_1 = 0.9, β_2 = 0.95, and a weight decay of 1×10−4. Each training batch consists of 2,000 samples, totaling 2 million tokens with a 1,024-token context length. Gradients are clipped at 1.0, and training begins with a warmup phase of 1,000 iterations, followed by a constant learning rate schedule to reduce the number of experiments.

For vision inputs, we process images as (14,14) patches with augmentations including Random Resized Crop (224px, scale range of [0.4, 1.0]) and Random Horizontal Flip with a 50% chance. Model training benefits from efficiency techniques such as bfloat16 precision, Fully Sharded Data Parallel (FSDP) (Zhao et al., 2023), activation checkpointing, gradient accumulation, and sequence packing to minimize padding in the image-captioning dataset.

We assess model performance on three held-out data subsets: interleaved data (Obelics), image-caption data (HQITP), and text-only data (DCLM), following prior works (Hoffmann et al., 2022; Aghajanyan et al., 2023; Abnar et al., 2025). This setup provides a robust evaluation of model generalization across diverse data types.

Params 90M 200M 350M 700m 1.3B 3B Width 512 768 1024 1536 2048 3072 Depth 24 Learning rate constant after warmup: 1e-4 Training tokens 8.4B 8.4B 8.4B 16.8B 33.6B 67.2B Optimizer AdamW (Loshchilov and Hutter, 2017) Optimizer Momentum β1 = 0.9,β2 = 0.95 Batch size 128 Sequence length 1024 Gradient clipping 1.0 Warmup iterations 1k

Table7 Pre-training hyperparameters used for the pre-training of LLM with PILE dataset to conduct the analyses

### B Detailed extrapolation results

We report the detailed per-domain and per-model size MRE corresponding to each experiment and scaling law in the paper.

#### B.1 Comparison to the laws of Ye et al. (2024) and Liu et al. (2024)

Ye et al. (2024) propose four laws to model the behavior of the loss on a domain as a function of h only, that is, for a fixed N,D budget. They propose the following formulas, rewritten with notations consistent with our notation:

k

Ci exp(γihi) (Ye M1)

L(h) = E +

i=1

L(h) = E + C

k

exp(γihi) (Ye M2)

i=1

k

L(h) = E + C exp(

i=1

γihi) (Ye M3)

k

γihi) (Ye M4)

L(h) = E + C exp(

i=1

where the parameters of the law are the E,C,Ci,γi. Liu et al. (2024) propose to model the loss as a linear function of h. We compare this with the form of our scaling law equation (2.4) when N and D are fixed:

1

L(h) = E +

k i=1 Cihγ

i

i

(Additive, fixed (N, D))

We fit all those scaling laws on the LLM training data, keeping only one value for N,D (we take N = 200m, D = 8B tokens). We only keep 25 training mixtures to fit the laws, and report the MRE on the 84−25 = 59 remaining in table 13. Note that we had trouble fitting the M3 law, which is also reported to underperform in (Ye et al., 2024). Overall, our formula gives systematically better training errors, and it most of the time translates to better testing errors on unseen mixtures. We stress once again that one of the core contributions of our work is to explain how scale interacts with data mixtures, by proposing scaling laws that take N,D, and h as inputs. (Ye et al., 2024) only consider scaling laws with respect to h.

Scaling Law Domain Train MRE(%) 3B MRE(%)

Simple Arxiv 0.55 2.40 Additive Arxiv 0.50 2.09 Joint Arxiv 0.39 1.62 Full Arxiv 0.39 1.83

Simple Book 0.38 1.11 Additive Book 0.29 0.80 Joint Book 0.24 1.19 Full Book 0.24 1.14

Simple C4 0.35 0.47 Additive C4 0.29 0.31 Joint C4 0.24 0.34 Full C4 0.23 0.41

Simple GitHub 0.81 2.05 Additive GitHub 0.65 1.17 Joint GitHub 0.54 2.51 Full GitHub 0.52 1.97

Simple Commoncrawl 0.34 0.65 Additive Commoncrawl 0.29 0.58 Joint Commoncrawl 0.24 0.90 Full Commoncrawl 0.23 0.74

Simple Stackexchange 0.57 0.68 Additive Stackexchange 0.51 0.36 Joint Stackexchange 0.38 0.47 Full Stackexchange 0.36 0.42

Simple Wikipedia 0.97 4.53 Additive Wikipedia 0.92 4.45 Joint Wikipedia 0.57 2.09 Full Wikipedia 0.56 1.98

Table 8 Full results of experiments in section 4 for the LLM experiment.

Scaling Law Domain Train MRE(%) 2B MRE(%) 8B MRE(%)

Simple Text 0.15 0.44 0.50 Additive Text 0.12 0.40 0.51 Joint Text 0.10 0.39 0.32 Full Text 0.09 0.38 0.33

Simple Image-Captions 0.52 0.89 1.36 Additive Image-Captions 0.47 0.83 1.23 Joint Image-Captions 0.43 0.85 1.17 Full Image-Captions 0.43 0.90 1.33

Simple Interleaved 0.22 0.65 0.80 Additive Interleaved 0.14 0.44 0.58 Joint Interleaved 0.10 0.41 0.45 Full Interleaved 0.10 0.40 0.45

Table 9 Full results of experiments in section 4 for the multimodal experiment.

Scaling Law Domain Train MRE(%) 1B MRE(%)

Simple Noisy image-text 0.34 1.05 Additive Noisy image-text 0.35 1.20 Joint Noisy image-text 0.23 0.63 Full Noisy image-text 0.21 0.58

Simple Synthetic 1.85 5.96 Additive Synthetic 1.89 6.19 Joint Synthetic 0.83 5.94 Full Synthetic 0.70 5.56

- Simple High quality 1 0.70 0.99

- Additive High quality 1 1.19 2.02

- Joint High quality 1 0.34 1.18

- Full High quality 1 0.32 1.08

Simple High quality 2 0.64 1.22 Additive High quality 2 0.64 0.79 Joint High quality 2 0.31 1.06

- Full High quality 2 0.31 0.93

- Table 10 Full results of experiments in section 4 for the LVM experiment.

Scaling Law Domain Train MRE(%) 700m MRE(%) 1B MRE(%) Simple Wikipedia 0.28 0.75 1.13 Additive Wikipedia 0.24 0.77 1.11 Joint Wikipedia 0.13 0.24 0.39 Full Wikipedia 0.12 0.23 0.39 Simple GitHub 0.60 1.38 3.28 Additive GitHub 0.42 1.10 1.69 Joint GitHub 0.23 0.38 1.46 Full GitHub 0.22 0.49 1.91 Simple StackExchange 0.40 0.90 1.50 Additive StackExchange 0.33 0.88 1.31 Joint StackExchange 0.17 0.26 1.05 Full StackExchange 0.16 0.30 1.17 Simple PG-19 0.21 0.53 0.91 Additive PG-19 0.16 0.55 0.94 Joint PG-19 0.15 0.40 0.71 Full PG-19 0.14 0.35 0.54

- Table 11 Full results of experiments in section 6 for the LLM experiment.

Scaling Law Domain Train MRE(%) 700m MRE(%) 1B MRE(%) Additive Wikipedia 0.45 0.53 0.53 Joint Wikipedia 0.22 0.18 0.19 Additive GitHub 0.70 0.88 1.49 Joint GitHub 0.39 0.31 1.09 Additive StackExchange 0.50 0.55 0.50 Joint StackExchange 0.29 0.23 0.44 Additive PG-19 0.24 0.37 0.53 Joint PG-19 0.18 0.25 0.44

Table 12 Full results of experiments in section 4 for the cosine schedule experiment.

#### B.2 Comparison to Ge et al. (2024)

Ge et al. (2024) propose a scaling law to evaluate the loss the i − th training domains, as a function of both h and number of tokens D:

C hγi

B Dβ

(Ge 24)

Li(h,D) =

+ E

where the coefficients B,β,E,C and γ have to be fitted. Here, the loss must be on domain i, and it only involves the proportion of that domain hi, not that of the other domains. In our view, this is a caveat since it implies that all other domains would have the same impact on the loss for domain i, while one other training domain might be very useful for that task, and another might be useless.

Since this law does not take into account model scale, we compare it to our additive law for a fixed model scale:

1

B Dβ

L(h,D) = E +

+

k i=1 Cihγ

i

i

(Additive, fixed N)

We consider a fixed size of model (N=200M) on the LLM training experiment, take only 25 mixtures to fit the scaling laws, and report the MRE on the remaining testing set in table 14. We see that our proposed scaling law achieves a significantly lower MRE on both train and testing data, highlighting the importance of taking all other domains into account.

### C Additional analysis

#### C.1 Optimal domain weights

Optimal weights for the additive scaling law. We report the optimal domain weights h∗ for the additive scaling law in table 15.

Recall that section 5 defines the optimal domain weights h∗(·) as function of compute budget (N,D). In the main text, we assumed uniform weights of the target domains Di. However, we can also consider a weighted average scenario, with an arbitrary weight vector wi.

h∗(w,N,D) ∈ arg min h∈∆k

m

wiLi(N,D,h). (C.1)

i=1

Once again, this objective is seamlessly optimizable with mirror descent. When training domains and target domains are the same, w and h∗ are a probability distribution over the same simplex. Therefore, we can study the mapping w  → h∗(w,N,D).

Scaling law Domain Train (MRE%) Test (MRE%) Additive, fixed (N, D) Wikipedia 0.07 0.18 Ye M1 Wikipedia 0.08 0.14 Ye M2 Wikipedia 0.09 0.17 Ye M3 Wikipedia 2.54 4.20 Ye M4 Wikipedia 0.17 0.31 Regmix Wikipedia 0.92 1.31 Additive, fixed (N, D) GitHub 0.10 0.19 Ye M1 GitHub 0.20 0.61 Ye M2 GitHub 0.22 0.40 Ye M3 GitHub 5.45 5.26 Ye M4 GitHub 0.36 0.44 Regmix GitHub 0.65 1.35 Additive, fixed (N, D) StackExchange 0.07 0.18 Ye M1 StackExchange 0.14 0.32 Ye M2 StackExchange 0.14 0.21 Ye M3 StackExchange 4.11 3.20 Ye M4 StackExchange 0.22 0.34 Regmix StackExchange 0.78 0.92 Additive, fixed (N, D) PG-19 0.08 0.12 Ye M1 PG-19 0.09 0.17 Ye M2 PG-19 0.13 0.18 Ye M3 PG-19 2.21 3.14 Ye M4 PG-19 0.16 0.21 Regmix PG-19 0.64 0.89

Table 13 Comparison of our scaling laws for a fixed (N, D) budget with those of (Ye et al., 2024) and (Liu et al., 2024) on the LLM data.

Behavior at the corners We predict the optimal domain weights for both laws, chosing each training domain j as the target, that is, putting wi = 1 if i = j and wi = 0 otherwise. The results are given in figure 7.

We see that the data mixture law predicts that the optimal domain weights are typically located in the corresponding corner of the simplex - which is not too surprising when there is little domain overlap. This justifies the rule of thumb h∗i ≈ 1 when the target domain is Di.

Fixed-points For a given (N,D) pair, the optimization problem of equation (C.1) defines a function w  → h∗(w) that maps the simplex onto itself. The fact that h∗(w) ̸= w indicates the surprising phenomenon that, in order to minimize the loss LERM = mi=1 wiLi(θ), it is faster to instead minimize L∗ = mi=1 h∗(w)iLi(θ), rather than minimizing directly LERM.

Fixed points of the map w  → h∗(w) correspond to target mixtures w that are minimized by training on w itself: this is the empirical risk minimizer.

To find these points, we compute the Jensen-Shannon distance, defined as JS(w,h∗) = 1/2(KL(w∥m) + KL(h∗∥m)) with m = (w + h∗)/2, and we look for near-zero values, in figure 8 and figure 9.

Accumulation point of asymptotes. In mirror of figure 5 we can monitore how the optimal mixture h∗ evolves as we scale parameters N and data D. For example, we can keep D constant and scale N, or the opposite, or scale both simultaneously with D ∝ N. Results are highlighted in Figure 10. These asymptotes reach an accumulation point when D → ∞ or N → ∞. They depend on the speed at which N and D grow.

Scaling law Domain Train (MRE%) Test (MRE%)

Additive, fixed N Wikipedia 0.13 0.17 Ge 24 Wikipedia 0.51 0.62

Additive, fixed N GitHub 0.20 0.26 Ge 24 GitHub 1.99 2.26

Additive, fixed N StackExchange 0.14 0.19 Ge 24 StackExchange 0.94 1.21

Additive, fixed N PG-19 0.13 0.15 Ge 24 PG-19 0.49 0.54

- Table 14 Comparison of our scaling laws for a fixed model size N with that of (Ge et al., 2024) on the LLM data.
- Table 15 Optimal domain weights h∗ for LLMs. Note that for the OpenHermes optimal weights, the law predicted a weight of 0 for wikipedia, which we artificially set to 1%.

Model Arxiv Book C4 GitHub Commoncrawl Stackexchange Wikipedia For average loss h∗avg 9.6 9.5 25.1 8.0 12.1 17.9 17.0

For OpenHermes h∗OH 9.4 4.8 27.8 6.6 36.9 13.5 1.0 Base 4.6 4.2 26.7 5.2 52.2 3.3 3.8

Additive law

Joint law

StackExchange

StackExchange

Wikipedia GitHub

Wikipedia GitHub

PG19

PG19

100

|[Figure 9]| |
|---|---|
| | |
| | |
| | |
| | |

[Figure 10]

[Figure 11]

99.7% 0.0% 0.3% 0.0%

99.2% 0.0% 0.0% 0.8%

Wikipedia

80

Targetdomains

0.0% 90.4% 9.6% 0.0%

0.0% 96.1% 3.9% 0.0%

GitHub

60

0.0% 6.4% 93.6% 0.0%

- 0.9% 0.5% 98.6% 0.0%
- 1.9% 0.3% 0.0% 97.8%

StackExchange

40

20

5.4% 0.0% 0.0% 94.6%

PG19

Training domains

Training domains

0

###### Figure7 Optimal domain weights for a single (pure) domain, typically lies at the corner of the probability simplex. Scaling law predictions for a 1.3B model trained on 10B tokens.

JS(w,h∗(w)) with 25% of PG19

Stackexchange

0.22

[Figure 12]

Jensen-Shannondistance

0.20

0.18

0.16

0.14

Wikipedia Github

- Figure8 The optimal domain weights are always different from the target mixture, except at the corners of the simplex 1.3B model with 10B tokens, on 4 domains of The Pile.

Jensen-Shannon metric JS(w,h∗(w))

|[Figure 13]| |
|---|---|
| | |
| | |
| | |

PG19

0.20

Stackexchange

0.15

Github

0.10

Wikipedia

- Figure 9 Jensen-Shannon distance between target mixture h and its optimum training mixture h∗(w) on the 4D simplex. 1.3B model with 10B tokens, on 4 domains of the Pile. No fixed-point are visible, except at the corners. This suggests that in general, it is better not to train on the mixture you want to be good on.

Image-Caption

Optimum weights h∗(w) Small model, few tokens

N = O(1),D = O(t) N = O(t),D = O(1)

N = O(t),D = O(√t) N = O(√t),D = O(t) N = O(t),D = O(log t) N = O(log t),D = O(t) Isocurve: N = O(t),D = O(t)

Accumulation point t → ∞

Text Interleaved

- Figure 10 Asymptotes of optimal mixture when increasing N and D at different speeds on multimodal data. Surprisingly, there is little diffenrence between proportional scaling O(t) and square-root scaling O(√t): both are fast enough. However, when one quantity is held constant, or only grow at logarithmic speed, the accumulation point changes.

