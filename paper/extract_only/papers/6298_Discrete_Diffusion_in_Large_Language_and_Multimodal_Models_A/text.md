# Discrete Diffusion in Large Language and Multimodal Models: A Survey

Runpeng Yu*, Qi Li*, Xinchao Wang† National Univerisity of Singapore {r.yu, liqi}@u.nus.edu, xinchao@nus.edu.sg

## arXiv:2506.13759v5[cs.LG]19Sep2025

Abstract—In this work, we provide a systematic survey of Discrete Diffusion Language Models (dLLMs) and Discrete Diffusion Multimodal Language Models (dMLLMs). Unlike autoregressive (AR) models, dLLMs and dMLLMs adopt a multi-token, parallel decoding paradigm using full attention and a denoising-based generation strategy. This paradigm naturally enables parallel generation, fine-grained output control, and dynamic perception. These capabilities are previously difficult to achieve with AR models. A growing number of industrial-scale proprietary d(M)LLMs, as well as a large number of open-source academic d(M)LLMs, have demonstrated performance comparable to their autoregressive counterparts, while achieving up to 10× acceleration in inference speed. These developments position discrete diffusion models as a promising alternative to intelligence based on the traditional autoregressive approach. In this work, we present a comprehensive overview of the research in the dLLM and dMLLM domains. We trace the historical development of dLLMs and dMLLMs, formalize the underlying mathematical frameworks, list commonly-used modeling methods, and categorize representative models. We further analyze key techniques for training, inference, quantization. We also discuss the trustworthy issues and summarize emerging applications across language, vision-language, and biological domains and etc.. We conclude by discussing future directions for research and deployment.

Relative papers are collected in this repo.

Index Terms—Discrete Diffusion, Large Language Model, Multimodal Large Language Model, Diffusion Large Language Model, Diffusion Multimodal Large Language Model, Language Model, Unified Model

I INTRODUCTION

In recent years, Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have demonstrated remarkable advances, exhibiting capabilities that increasingly resemble, or even surpass, human-level performance in domains traditionally associated with intelligence. Modern LLMs and MLLMs achieve superior scores on standard benchmarks designed for general knowledge, comprehension, and reasoning, suggesting that these systems are no longer merely text completion engines but competent general-purpose agents.

To date, the predominant paradigm for both LLMs and MLLMs has been autoregressive (AR) [1, 2, 3, 4, 5]. Despite their successes, autoregressive (AR) models face intrinsic limitations. Their left-to-right decoding hinders parallel inference, reducing efficiency. They also struggle with precise structural control (e.g., length or format), making natural language inefficient for fine-grained orchestration of tools and agentic tasks.

————————————————————

* Equal contribution, random order. † Corresponding author.

Moreover, causal attention forces one-pass static perception of inputs, limiting dynamic task-aware and dynamic perception without costly chain-of-thought or multi-round processing.

Discrete Diffusion Large Language Models (dLLMs) and Discrete Diffusion Multimodal Large Language Models (dMLLMs) [6, 7, 8, 9, 10, 11] have recently emerged as a promising direction. In tasks such as code generation [12, 13], planning [7], and Sudoku [7], dLLMs have been widely shown to achieve better performance than autoregressive models. Moreover, [14] demonstrates that in a data-constrained setting, increasing the number of training FLOPs for dLLMs enables it to outperform AR.

In contrast to AR generation, discrete diffusion models treat generation as an iterative denoising process over discrete token sequences. This paradigm eliminates the left-to-right constraint and enables generation that is parallel and structurally controllable with bidirectional attention mechanism. Here are some unique properties of discrete diffusion.

- • Parallel Decoding: Unlike AR models that decode one token at a time, discrete diffusion models generate multiple tokens simultaneously in each denoising step. This parallel decoding significantly accelerates inference speed.
- • Better Controllability: Discrete diffusion treats generation as a denoising or infilling task rather than unbounded leftto-right generation. This allows for precise control over output properties such as response length, format, and even the reasoning structure—by conditioning the generation on predefined templates.
- • Dynamic Perception: Enabled by bidirectional attention, discrete diffusion models continuously revise their perception of visual and linguistic context throughout the generation process. This facilitates adaptive comprehension that evolves with the response, overcoming the static, onepass limitations of AR models.

Early efforts in discrete diffusion established the foundational mathematical formulations of discrete diffusion, introducing a token corruption schemes specifically designed for categorical data [15, 16]. These models demonstrated the feasibility of diffusion-based generation on various types of discrete data, including natural language [15, 17], images [15], and biological sequences such as DNA [18]. In this early stage, experiments were limited to models with around or fewer than 1 billion parameters. Through simplifications and reparameterizations [17, 18, 19], along with practical engineering efforts, the absorbingstate discrete diffusion formulation has gradually become the predominant mathematical framework adopted by open-source models and is termed as the masked diffusion model.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

D3PM Plaid

MDLM

UniDisc

MDLM RDM Diffusion

Models ≤ 1B

[Figure 5]

[Figure 6]

DiffusionBERT

TESS

SEDD MD4

[Figure 7]

[Figure 8]

[Figure 9]

-NAT

[Figure 10]

[Figure 11]

DreamOn DreamCoder

DiffuGPT DiffuLLaMA LLaDA

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Diffusion-LLMs

LLaDA 1.5 TESS2

Language Models

Gemini Diffusion

DiffuCoder

Seed Diffusion

DREAM

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Dimple

[Figure 21]

[Figure 22]

[Figure 23]

LLaDA-V

Mercury

[Figure 24]

MultiModal Models

LaViDa

[Figure 25]

Muddit

[Figure 26]

Unified Models

| | | | |FUDOKI| |
|---|---|---|---|---|---|
| | | | | | |

[Figure 27]

MMaDA

* Closed source dLLMs / dMLLMs are underlined.

2021

2022 2023 2024

2025

- Fig. 1. A timeline of existing dLLMs and dMLLMs in recent years. The timeline is established mainly according to the release date (e.g., the submission date to arXiv) of the technical paper for a model. The affiliation marked in the figure is based on the first affiliation listed in each paper.

II MATHEMATICAL FORMULATIONS

With the masked diffusion formulation, recent advances have significantly improved the scalability and effectiveness of discrete diffusion models [20, 21]. A major breakthrough on the industrial front came with the presentation of discrete diffusionbased large language models by Inception Labs and Google, namely Mercury [13] and Gemini Diffusion [12]. These models reports comparable performance on code and mathematics benchmarks with their AR counterpart, while also achieving 10× speedups in decoding, with about 1000 tokens per second.

In this section, we discuss the mathematical formulation of the discrete diffusion.

II.A Discrete Diffusion Model and Transition Matrix

Diffusion models with discrete state spaces were initially introduced in [22] for binary variables, and later extended to categorical variables in [16]. Based on the previous works, Discrete Denoising Diffusion Probabilistic Models (D3PMs) [15] provides a general and flexible framework.

In parallel, the research community has developed and opensourced an increasing number of discrete diffusion-based language and multimodal models. The development began with dLLM models trained on large-scale text corpora, such as LLaDA [6] and Dream [7]. Later, using the public available dLLMs as the backbones, dMLLMs, such as Dimple [8], LaViDa [10], and LLaDA-V [9], are developed through multimodal alignment, instruction tuning, preference learning, and then reasoning enhancement.

Let x0 ∼ q(x0) denote the data distribution over sequences composed of K categorical values. The D3PM framework defines a forward Markov process q(x1:T | x0) that gradually corrupts the data into noise, and a parameterized reverse process pθ(x0:T) that learns to denoise:

T

q(xt | xt−1), (1)

q(x1:T | x0) =

To provide a comprehensive framework for understanding discrete diffusion large language models (dLLMs) and discrete diffusion multimodal large language models (dMLLMs), this survey systematically explores recent advances in modeling, training, generation and applications of discrete diffusion techniques.

t=1

T

pθ(xt−1 | xt). (2)

pθ(x0:T) = p(xT)

t=1

Each xt is a discrete random variable and q(xt | xt−1) is defined via a time-dependent transition matrix Qt, with categorical transition probabilities:

In the rest of this paper, Sec. II presents the mathematical foundations of discrete diffusion models. Sec. III lists several modeling techniques for the discrete diffusion task. These techniques build upon mathematical formulations, primarily aiming to enhance model flexibility or introduce additional capabilities. Sec. IV surveys representative discrete diffusion language models across varying scales. This includes early-stage models, scaled dLLMs, dMLLMs, and unified models. Sec. V discusses the key training strategies used in dLLMs and dMLLMs. Sec. VI lists various inference techniques used in dLLMs and dMLLMs. Sec. VIII discusses trustworthy issues in dLLMs. Sec. VII discusses the quantization techniques of dLLMs. Sec. IX reviews the broad range of applications powered by discrete diffusion models. Finally, Appendix Sec.F summarize potential directions for future research. The organization of this survey is illustrated in Fig. 2.

q(xt | xt−1) = Cat(xt;p = xt−1Qt), (3)

where xt−1 is a one-hot vector and Qt ∈ RK×K is a rowstochastic matrix. The marginal distribution q(xt | x0) and the posterior q(xt−1 | xt,x0) are:

q(xt | x0) = Cat(xt;p = x0Q1:t), (4)

Q1:t = Q1Q2 ...Qt, (5) q(xt−1 | xt,x0) = Cat xt−1;p =

xtQ⊤t ◦ x0Q1:t−1 x0Q1:tx⊤

. (6)

t

D3PM framework support various types of transition matrices Qt, each inducing a different corruption behavior. Here, we present the two most commonly used transition matrices: uniform and absorbing. Additional types, including hybrid,

Discrete Diffusion Model & Transition Matrix (Sec. II.A)

Discrete Diffusion for Binary [22] and Categorical [16] Variable, D3PM [15], Roulette Diffusion [23], GIDD [24]

Simplified Masked Diffusion Model (Sec. II.B) MLDM [18], MD4 [19] Continuous Time Discrete Denoising Models [25] (Sec. II.C) Concrete Score (Sec. II.D)

Mathematical Formulations (Sec. II)

CSM [26], DWDSE [18], Categorical Ratio Matching Loss [27], RADD [28], TCSM [29], CEDD [23]

Discrete Flow Matching [30] (Sec. II.E) Reparameterized Discrete Diffusion Model [17] (Sec. Apdx.A.I)

Block Diffusion Models [31](Sec. III.A), Flexible-Length Masked Diffusion [32](Sec. III.B), Partial Masking [33](Sec. III.C), DDOT [34](Sec. III.D)

Modeling Language Diffusion (Sec. III)

D3PM [15], DiffusionBERT [35], RDM [17],Masked-Diffuse LM [36], Diffusion-NAT [37], TESS [38], Plaid [39], SEDD [40], MDLM [18], MD4 [19], UniDisc [41]

Discrete Diffusion Models around 1B (Sec. IV.A)

Large Diffusion Language Models

- (Sec. IV.B)

LLaDA [6],DIFFUSION-LLMs[42],DiffuGPT&DiffuLLaMA [20], DREAM[7], LLaDA 1.5 [43], TESS 2[44], DreamOn [45], DreamCoder [46], DiffuCoder [47], Seed Diffusion [48]

Large Diffusion Multimodal Models

- (Sec. IV.C)

Representative Models (Sec. IV)

Dimple [8], LaViDa [10], LLaDA-V [9]

Large Unified Models (Sec. IV.D) MMaDA [11], FUDOKI [49], Muddit [50]

BERT Initialization [35], Autoregressive Initialization[20, 7], Autoregressive-then Diffusion Training [8]

Initialization Technique (Sec. V.B)

Complementary Masking[10](Sec. V.C) Masking Scheduling Technique (Sec. V.D)

Linear [15],Geometric [40, 19],Cosine [51],Polynomial [19], Token-wise Masking Scheduling [35], Blockwise SFT [52]

Training Techniques (Sec. V)

Reweighting Technique (Sec. V.E) MGDM [53]

Distillation (Sec. V.F) Di4C [54]

Reinforcement Learning (Sec. V.G) UniGRPO [11],VRPO [43],SDPO [55],wd1 [56],DCoLT [57]

Training-Testing Input Discrepancy (Appendix Sec.C.III) Two-Step Strategy [58]

Discrete Diffusion (M)LLMs

Metrics Used in Unmasking: Confidence, Margin, Entropy [7, 59], EB-Sampler [60], PC-Sampler. [61]; Confident Decoding [8], Block-wise [9], DUS [62], Continuous Time (Flow Matching) Unmasking [30]

Unmasking Techniques (Sec. VI.A)

Discrete Time Remasking [63],Wide In Narrow Out [64] Continuous Time (Flow Matching) Remasking [30]

Remasking Techniques (Sec. VI.B)

Prefilling and Caching Technique (Sec. VI.C)

Prefilling [8, 10], dKV-Cache [65], dLLM-Cache [66], DualCache [67]

Classifier-Free Guidance [21], Classifier Guidance [68], Reward Guidance [44], EDLM [69]

Inference Techniques (Sec. VI)

Guidance Technique (Sec. VI.D)

Early Stopping [70, 71], Particle Gibbs Sampling [72], Temporal Self-Consistency [73]

Sampling Technique (Sec. VI.E)

Context Length Extension (Sec. VI.F) LongLLaDA [74]

Sparse Computation (Sec. VI.G) Sparse-dLLM [75], DPad [76],Pipelined Parallel [77]

Response Length Control (Sec. VI.H) DAEDAL [78] Quantization (Sec. VII) Quantization Meets dLLMs[79], DLLMQuant [80] Privacy and Safety (Sec. VIII) DIJA [81], PAD [82], MOSA [83]

Language Related Applications

- (Sec. IX.A)

DiffuSeq StylePTB [84], DiffEmbed [85], SLD [86], DiffusPoll [87], PoetryDiffusion [88], SVDD [89], EdiText [90], CrossMamba [91], DiffETM [92], TermDiffuSum [93], CDA2 [94], DiffusionCLS [95], GDP [96], Layout-Corrector [97]

Knowledge and Reasoning

- (Sec. IX.B)

DoT [98], DiffuCOMET [99], DPCL-Diff [100], d1 [101], MGDM [53], NeSyDM [102]

Vision and Multimodal (Sec. IX.C) UDAN-CLIP [103], M2D2M [104], AR-Diffusion [105]

Applications (Sec. IX)

Robotics and Autonomous Driving

- (Sec. IX.D)

DiffVLA [106], DDVLA [107], ViLaD [108], VPDD [109], DGD [110]

Graph and Structured Predictions

- (Sec. IX.E)

LO-ARM [111], ReDiSC [112], Scaffold Diffusion [113]

Biological and Drug Discovery

- (Sec. IX.F)

MolEditRL [114], CFP-Gen [115], TransDLM [116], GenMol [117], DPLM-2 [118], PepTune [119], CMCM-DLM [120]

- Fig. 2. Overview of our survey. We begin by introducing the mathematical foundations (Sec. II) and modeling methods (Sec. III) of discrete diffusion language models . Next, we present a high-level overview of representative base models (Sec. IV), followed by discussions on training strategies (Sec. V), inference techniques (Sec. VI) and quantization (Sec. VII). Furthermore, we discuss the privacy and safety studies of discrete diffusion language models (Sec. VIII). In addition, we also review a wide range of applications (Sec. IX) that adopt discrete diffusion language models as their core model.

discretized Gaussian, band-diagonal, and embedding-based, are described in Appendix Section A.I.

###### • Uniform: Qt = (1 − βt)I + β

K 11⊤ yields a uniform stationary distribution. The uniform transition matrix looks like

t





1 − KK−1βt β

K ··· β

t

t

K

βt K 1 − KK−1βt ··· β

t

K

Quniformt =

.

 

 

... .

. .

βt K ··· 1 − KK−1βt

βt K

(7)

###### • Absorbing: Qt = (1−βt)I +βt1e⊤m, where em is a vector with a one on the absorbing state and zeros elsewhere. Tokens either remain unchanged or are mapped to a special

[MASK] token with probability βt. The absorbing transition matrix looks like





1 − βt 0 ··· 0 βt 0 1 − βt ··· 0 βt

... . .

Qabsorbt =

. (8)

. .

 

 

0 0 ··· 1 − βt βt 0 0 ··· 0 1

Following the x0-parameterization, the model predicts

- pθ(xt−1 | xt) using:

pθ(xt−1 | xt) ∝

x ˜0

q(xt−1,xt | x˜0)˜pθ(˜x0 | xt), (9)

where p˜θ(˜x0 | xt) is a network predicting logits over x0. This parameterization ensures the reverse distribution respects the sparsity pattern of Qt.

The loss function combines the variational lower bound Lvb with an auxiliary cross-entropy denoising term Lce:

Lλ = Lvb + λLce, (10) Lvb = Eq(x

0) KL(q(xT | x0)∥p(xT))

LT

+

T

t=2

Eq(x

t|x0) [KL(q(xt−1 | xt,x0)∥pθ(xt−1 | xt))]

Lt−1 − Eq(x

1|x0) [log pθ(x0 | x1)]

L0

, (11)

Lce = Eq(x

0)Eq(x

t|x0)[−log p˜θ(x0 | xt)]. (12) In Lvb,

- • LT is the KL divergence between the forward terminal distribution q(xT | x0) and the prior p(xT),
- • Lt−1 is the KL divergence between the forward posterior q(xt−1 | xt,x0) and the learned reverse model pθ(xt−1 | xt) at each intermediate step,
- • L0 is the cross-entropy loss for reconstructing x0 from x1 using the reverse model.

Such decomposition enables the model to be trained efficiently by sampling time steps t uniformly and estimating each term using stochastic gradient descent. The forward posterior

- q(xt−1 | xt,x0) has a closed-form expression under categorical diffusion, and the model is typically parameterized to predict

pθ(x0 | xt), from which pθ(xt−1 | xt) is derived analytically. The auxiliary Lce is added to encourage accurate prediction of the original data x0 from corrupted samples.

Besides a flexible representation of discrete diffusion, D3PM also unifies various paradigms such as BERT, autoregressive models, and masked language models within the discrete diffusion framework.

II.B Simplified Masked Diffusion Model

A widely adopted class of discrete diffusion models is based on the absorbing state and is often referred to as the Masked Diffusion Model. Both [19, 18] introduced simplifications to the diffusion process and the corresponding training objective for masked diffusion, yielding improved performance and computational efficiency. For the simplification of general discrete diffusion, we discuss Reparameterized Discrete Diffusion Models (RDMs) [17] in Appendix Sec.A.II.

In masked diffusion, the forward process progressively replaces input tokens with a special [MASK] token. Once a token is masked, it remains in that state throughout the remainder of the process, making the [MASK] token an absorbing state. At each time step t, the forward transition for a token x is defined as:

q(xt | x0) = Cat(xt;αtx0 + (1 − αt)m), (13) where m is the one-hot vector corresponding to the [MASK] token, and αt ∈ [0,1] is the monotonically decreasing schedule such that α0 ≈ 1 and αT = 0.

The reverse process aims to denoise the masked sequence by substituting [MASK] tokens with predicted tokens. Importantly, unmasked tokens are carried out unchanged throughout the denoising process. The reverse posterior at a previous time s conditioned on xt and x0 is given by:

q(xs | xt,x0) =

 

Cat(xs;xt), if xt ̸= m, Cat xs;

(14)

(1 − αs)m + (αs − αt)x0 1 − αt

, if xt = m.



This formulation reflects two key properties of the masking process: (1) If the current token xt is not masked, the posterior is deterministic: xs = xt. (2) If the token is masked, then the posterior is a linear interpolation between the mask vector m and the clean token x0, scaled by the noise schedule parameters αs and αt.

Let fθ(xt) be the neural network output predicting the original token x0 from the noisy input xt. The above reverse transition is rewritten as:

Cat(xs;xt), if xt ̸= m, Cat xs; (1−α

pθ(xs | xt) =

s)m+(αs−αt)fθ(xt)

1−αt , if xt = m. (15)

The variational lower bound for discrete diffusion can be simplified using the above formulation, leading to an final loss in the form of

T

N

αt−1 − αt 1 − αt

Ex

L =

0,x1:T −

δm(xt,n)x0,n log[fθ(xt)]n ,

t=2

n=1

(16)

where δm(xt,n) denotes the indicator function, δm(xt,n) = 1, if the n-th token of xt is a masked token, otherwise, δm(xt,n) = 0.

By defining the discrete time series as 0, T1 ,...,1− T1 ,1 and letting T → ∞, both works [18, 19] extend the above loss Eq. (16) to the continuous-time setting:

L =

1

Ex

0:1

0

αt′ 1 − αt

N

δm(xt,n)x0,n log[fθ(xt)]n dt.

n=1

(17)

This loss corresponds to a reweighted cross-entropy loss evaluated only over masked tokens. Such loss formulation is significantly simpler than the original variational bound and has become the standard training objective for subsequent large discrete diffusion models.

- II.C Continuous Time Discrete Denoising Models

D3PM operates in discrete time, i.e., with time steps t = 0,1,2,...,T. [25] describes a continuous-time framework for discrete denoising models, formulated as a Continuous-Time Markov Chain (CTMC), where t ∈ [0,T]. This approach generalizes discrete diffusion models by allowing transitions at arbitrary time points. The infinitesimal transition probabilities are given by:

###### qt|t−∆t(x′ | x) = δx,x′ + Rt(x,x′)∆t + o(∆t). (18)

This process converges to a tractable reference distribution as t → T. The time-reversed generative process is another CTMC with reverse rate matrix Rˆt, expressed as:

qt|0(x′ | x0) qt|0(x | x0)

Rˆt(x,x′) = Rt(x′,x)

pθ(x0 | x), (19)

x0

where pθ(x0 | x) is a learnable denoising model approximating q0|t(x0 | x).

The training of pθ(x0 | x) is guided by a continuoustime version of the variational lower bound. Let Zt(x) =

x′̸=x Rt(x,x′) be the total outgoing rate and rt(x′ | x) = Rt(x,x′)/Zt(x) the normalized jump probability. The continuous-time variational lower bound is:

Lvb(θ) =T Et∼U(0,T)Ex∼q

Ex′∼rt(·|x)

t

Rˆtθ(x,x′′) − Zt(x) log Rˆtθ(x′,x) + C, (20)

x′′̸=x

where C is constant with respect to θ. This objective can be efficiently optimized using stochastic gradient descent by sampling (x,x′) pairs according to the forward process.

During inference, however, the exact simulation of the reverse CTMC can be computationally prohibitive. Instead, the tauleaping algorithm [reference] approximates the reverse process by applying multiple transitions within a time interval τ simultaneously. For a current state xt, the number of transitions to each x′ during [t − τ,t] is modeled as:

###### Px′ ∼ Poisson(τ · Rˆtθ(xt,x′)). (21)

The next state xt−τ is obtained by aggregating the sampled transitions. This method supports parallel decoding by allowing simultaneous updates across multiple dimensions.

To further improve sample fidelity, predictor-corrector steps is used. After a tau-leaping step, corrector transitions with rate matrix Rc = Rt + Rˆtθ are applied to refine the sample distribution toward the target marginal qt(x). This approach is analogous to Langevin correctors in continuous diffusion models.

- II.D Concrete Score

In the continuous-time discrete diffusion framework, as formulated in previous [25], the reverse process can also be analytically expressed in terms of the forward transition rate matrix and a function known as the concrete score [26]. This construction enables training via score matching, analogous to score-based models in continuous diffusion model.

Let Rt(x,x′) be the forward transition rate matrix of a continuous-time Markov chain (CTMC) over a discrete state space X. The reverse-time transition rate Rˆt(x,x′) can be formulated as:

Rˆt(x,x′) =

 



pt(x′) pt(x)

Rt(x′,x), x′ ̸= x, −

k̸=x

Rˆt(x,k), x′ = x.

(22)

Here, the scalar ratio p

t(x′)

pt(x) is referred to as the concrete score. It quantifies the relative likelihood of two discrete states at time t and modulates the reverse transition rate accordingly. Thus, instead of learning the full reverse transition distribution, one can train a model sθ(x,t) to estimate the concrete score:

sθ(x,t) ≈

pt(x′) pt(x) x′∈X

. (23)

In the Appendix Sec.A.III, we discuss the commonly-used training loss under concrete score formulation, the connection of concrete score with traditional cross-entropy loss and the time independency simplification of concrete score.

- II.E Discrete Flow Matching (DFM)

Build upon the continuous-time Markov chain (CTMC) framework in Continuous Time Discrete Denoising Models [25], Discrete Flow Matching (DFM) [30] extends the Flow Matching paradigm to categorical sequence data. The model defines a probability path pt interpolating between a source distribution p (e.g., all-mask sequences) and a target distribution q (e.g., the data distribution), such that p0 = p and p1 = q.

Given a coupling distribution π(x0,x1) between source and target sequences, the marginal probability at time t is defined as:

pt(x | x0,x1)π(x0,x1), (24)

pt(x) =

x0,x1∈D

where the conditional path pt(x | x0,x1) factorizes over positions:

pt(x | x0,x1) =

N

pt(xi | x0,x1), (25)

i=1

with token-level conditional paths defined as a convex combination of basis distributions:

m

pt(xi | x0,x1) =

κit,jwj(xi | x0,x1), (26)

j=1

where κit,j ≥ 0, j κit,j = 1 form a scheduler controlling the path dynamics.

The generative process is defined via probability velocity

fields {uit}Ni=1 guiding transitions between states. The update rule for sampling is:

(·) + huit(·,xt), (27) where ut is said to generate pt if the process satisfies:

xit+h ∼ δxi

t

pt+h(x) = pt(x) − hdivx(ptut) + o(h), (28) with the discrete divergence operator:

[v(z,x) − v(x,z)]. (29)

divx(v) =

z∈D

For the convex interpolation path: pt(xi | x0,x1) = (1 − κt)δx

(xi), (30) the corresponding generating velocity takes the closed form:

###### (xi) + κtδx

0

1

κ˙t 1 − κt

uit(xi,z) =

p1|t(xi | z) − δz(xi) , (31)

where p1|t(xi | z) is the probability denoiser: the conditional probability of the target token xi1 given the current state z.

To estimate the posteriors required in the generative process, the model minimizes the cross-entropy loss:

N

0,x1),xt log p1|t(xi1 | xt;θ) , (32)

Et,(x

L(θ) = −

i=1

where xt ∼ pt(· | x0,x1). III MODELING LANGUAGE DIFFUSION

Beyond the mathematical formulation introduced in the previous section, in this section, we present several techniques used when modeling the language diffusion task, such as specialized neural network designs, and extra sub-tasks. These techniques are introduced to enhance the controllability and flexibility.

- III.A Block Diffusion Models

Block Diffusion models (BD3-LMs) [31] provide a hybrid framework that interpolates between autoregressive language models and fully parallel diffusion models. Instead of denoising all tokens simultaneously, BD3-LMs segment the sequence into blocks and perform discrete denoising diffusion within each block, while conditioning autoregressively on all preceding blocks. The mathematical formualtion of BD3-LMs are provided in Appendix Sec.B.I.

- III.B Flexible-Length Masked Diffusion

Flexible-Length Masked Diffusion (FlexMDM) [32] extends masked diffusion models to handle variable-length sequences by introducing a third token state, called empty, in addition to the usual masked and ground truth states. This allows the model not

only to recover missing tokens (unmasking) but also to insert new tokens into the sequence. During inference, the process alternates between two steps: first, the model predicts how many new mask tokens should be inserted before each existing token, effectively expanding the sequence length; then, it replaces all mask tokens with actual content tokens. By repeating this insertion–unmasking cycle, FlexMDM can gradually generate longer sequences, enabling it to produce variable-length outputs while maintaining the flexibility of generating tokens in any order. In Appendix Sec. B.II, we provide the mathematical formulation of FlexMDM along with a more detailed description of its generation process.

III.C Partial Masking

In masked diffusion model, each token has only two states. Partial Masking [33] introduces an additional intermediate state by decomposing each token into a sequence of sub-tokens.

Each original token xi0 ∈ X, where X = {0,...,C − 1}, is

mapped into a sub-token sequence y0i = [y0i,1,...,y0i,ℓ] ∈ Yℓ through an invertible base-b encoding function f: f : X → Yℓ,

√

C⌉. The inverse mapping f−1 reconstructs tokens from sub-tokens, ensuring lossless transformation: xi0 = f−1(y0i). Consequently, the forward pass, backward pass, training, and inference of diffusion with partial masking all occur at the sub-token level.

y0i = f(xi0), where Y = {0,...,b − 1} and b = ⌈ ℓ

- III.D Diffusion with Optimal Transport Position Coupling

Standard discrete diffusion models relies on fixed token positions during generation, which prevents them from handling flexible-length or flexible-position text infilling. Discrete Diffusion with Optimal Transport Position Coupling (DDOT) [34] jointly denoises both token values and token positions, enabling dynamic sequence restructuring while preserving relative ordering. During sampling, DDOT alternates between token denoising and position refinement:

- 1) Predict token distributions sθ(xt,t) and replace masks accordingly.
- 2) Predict position velocities vθ(zt,t) and update the position variable zt with Euler steps.

In Appendix Section B.III, we present additional mathematical formulations of DDOT.

IV REPRESENTATIVE MODELS

In this section, we provide a high-level overview of the representative works. In the following sections, we give detailed discussions on the training paradigms and inference-time decoding strategies of the models scaled to sizes comparable to LLMs. An evolutionary diagram of representative dLLM and dMLLM models is shown in Fig. 1.

- IV.A Discrete Diffusion Models around 1B

[22] first introduces a diffusion process over binary variables. This idea is generalized to categorical variables by [16], who demonstrates its effectiveness on image generation. Building on these foundations, D3PM [15] proposes a more flexible family of noising schedules that extends discrete diffusion to a broader class of discrete spaces (see Sec. II.A). DiffusionBERT [35]

Autoregressive (M)LLM Discrete Diffusion (M)LLM

(image) text

(image) text response

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

| | | | | |
|---|---|---|---|---|

Architecture

Model

Vision Encoder Text

Vision Encoder Text

Embedding Projector Layer

Embedding Projector Layer

Autoregressive LLM

Discrete Diffusion LLM

Key

Key

|𝑙4|
|---|

|𝑙4|
|---|

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

#### AttentionMask

|𝑙3|
|---|

|𝑙3|
|---|

Query

Query

|𝑙2|
|---|

|𝑙2|
|---|

𝑙1

𝑙1

|𝑥[1]|
|---|

|𝑥[2]|
|---|

|𝑥[3]|
|---|

|𝑥[1]|
|---|

|𝑥[2]|
|---|

|𝑥[3]|
|---|

𝑥[0]

𝑥[0]

Causal Attention Mask

Layer-wise Information Flow

Full Attention Mask

Layer-wise Information Flow

[BoS]

[M] [M] [M] [M] [M]

|𝑡 = 0 InitializationBegin of|
|---|

|𝑡 = 3 InitializationMask|
|---|

###### on:

###### on:

| |
|---|

| | | | | |
|---|---|---|---|---|

Sentence

Sequence

[BoS] Hi

[M] [M] a pen [M]

Inference

|𝑡 = 1|
|---|

|𝑡 = 2|
|---|

| |
|---|

| | |
|---|---|

| |
|---|

###### Decoding:

###### Decoding:

- •Left-to-Right
- •One-token per step #steps = #tokens

- •Free Order
- •Multi-token per step #steps =? #tokens

[BoS] Hi !

This is a pen [M]

|𝑡 = 2 •|
|---|

|𝑡 = 1 •|
|---|

| |
|---|

| |
|---|

[BoS] Hi ! [EoS]

This is a pen .

|𝑡 = 3|
|---|

|𝑡 = 0|
|---|

| |
|---|

| |
|---|

This is a pen .

This is a pen .

|Training label 𝒙𝒈𝒕|
|---|

|Training label 𝒙𝒈𝒕|
|---|

#### Training

This [M] a [M] .

This is a pen .

|Training Input 𝒙 = 𝒙𝒈𝒕|
|---|

Training Input 𝒙𝒕：Random Masked 𝒙𝒈𝒕

Evaluate Loss on All Tokens

Evaluate Loss only on Masked Tokens

𝑤(𝑡,𝑛) ∙ 𝑥𝑔𝑡𝑛 log(𝑥𝑡(𝑛))

𝑥𝑔𝑡𝑛+1 log(𝑥(𝑛))

ℒ = −෍

ℒ𝑡 = −෍

Loss：Cross Entropy Loss

Loss：Weighted Cross Entropy Loss

𝑛

𝑛

- Fig. 3. This figure compares autoregressive models and discrete diffusion models from four perspectives. First, regarding model architecture, both models share the same network structure; the key difference lies in their generation mechanisms. In addition to the LLM, both MLLM and dMLLM require an additional vision encoder. In terms of the attention mask, the autoregressive model uses a causal attention mask, whereas the discrete diffusion model adopts a full (bidirectional) attention mask. During inference, the autoregressive model starts from a BoS token and generates tokens one by one from left to right. In contrast, the discrete diffusion model begins with a sequence of mask tokens and denoises all tokens in parallel. At each step, a subset of tokens is selected and replaced with non-mask tokens, continuing until no mask tokens remain. For training, the autoregressive model directly takes the input sequence and applies next-token prediction loss. The discrete diffusion model first randomly masks the input tokens and then computes a weighted cross-entropy loss over the masked positions.

explores training BERT [121] to reverse a discrete diffusion process with an absorbing state, introducing a token-aware noise schedule for the forward pass and methods to embed time-step information into BERT.

LM (MDLM) [36] propose to leverage the inherit linguistic features of texts to encourage the model to recover the text following an easy-first-generation nature, and directly predict the discrete token with cross-entropy loss to stabilize the intermediate diffusion steps. Diffusion-NAT [37] uses the pretrained BART [122] as the language backbone and unifies the inference process of pretrained language models and the denoising process of discrete diffusion models in a non-autoregressive manner, thus the BART model plays the role of the parameterized denoiser in discrete diffusion models. Furthermore, TESS [38] leverages a new form of self-conditioning and applies diffusion on the logit simplex instead of the learned embedding space. Plaid [39] takes the first step towards closing the likelihood gap

[17] simplifies the formulation of discrete diffusion process in D3PM and introduces a new model family named Reparameterized Discrete Diffusion Models (RDMs). It reformulate the backward process of discrete diffusion in D3PM [15] into a two-stage sampling procedure and yield a greatly simplified training objective and enable more flexible decoding algorithms. MLDM [18] and MD4 [19] further simplified the discrete diffusion model specifically for cases with an absorbing state, also referred to as masked diffusion models. Masked-Diffuse

between autoregressive and diffusion-based language models through SEDD [40] generalize the idea of score matching into the discrete spaces by proposing a novel loss named score entropy, which can be integrated seamlessly to build discrete diffusion models and can significantly boost model performance.

For unified models, UniDisc [41] is a prior work on unified discrete diffusion model for text and image modalities. Conceptually, UniDisc treats an image and a caption as two token sequences (from discrete codebooks) and denoises them together with a decoder-only Transformer and bidirectional attention.

- IV.B Large Diffusion Language Models

IV.B.1 DIFFUSION-LLMs

DIFFUSION-LLMs [42] is the first work we identified that scales discrete diffusion language models to 3B and 10B parameters, showing that their performance improves consistently with model size. Its training involves two stages. First, the model is pretrained with a masked LLM objective (similar to BERT [121]) to acquire world knowledge. Next, the pretrained masked LLM is reprogrammed into a dLLM using the RDM loss and training strategy on either specific downstream task datasets or instruction-following datasets. Experiments demonstrate that scaled-up discrete diffusion models, like autoregressive models, possess zero-shot generation, in-context learning, and even reasoning capabilities. Moreover, DIFFUSION-LLMs show that dLLMs can outperform autoregressive models on reasoning tasks requiring implicit planning, such as Path-Finding on PathStar Graphs [123].

IV.B.2 LLaDA Series

The LLaDA series represents the pioneering line of discrete diffusion-based alternatives to autoregressive LLMs. LLaDA [6], the first work in this line of research, is a discrete diffusion large language model. It follows the standard masked diffusion model framework, employing a Transformer with bidirectional attention. Its training objective is the variational likelihood bound (ELBO) [124], rather than the exact log-likelihood. While LLaDA demonstrates strong performance after supervised finetuning, aligning a dLLM with human preferences (akin to RLHF [125] for AR models) remains challenging. LLaDA 1.5 [43] specifically addresses this by introducing VarianceReduced Preference Optimization (VRPO) for dLLMs. The main challenge addressed by LLaDA 1.5 is the high variance that arises when estimating log-probabilities with ELBO during reinforcement learning. Based on both theoretical and empirical analysis, LLaDA 1.5 proposes three solutions: increasing the number of Monte Carlo samples, using optimal allocation of samples (many timesteps but one mask per step), and applying antithetic sampling in preference comparisons to reduce variance.

IV.B.3 DiffuGPT & DiffuLLaMA

DiffuGPT and DiffuLLaMA [20] propose converting a pretrained autoregressive Transformer (such as GPT-2 [126] or LLaMA [127]) into a dLLM, thereby avoiding the cost of training large models entirely from scratch. Crucially, [20] establish theoretical connections between autoregressive nexttoken prediction and the diffusion denoising objective, enabling alignment between the two paradigms during adaptation. Leveraging the autoregressive model’s knowledge as initialization, the

diffusion model can be scaled up with significantly less data (under 200 billion tokens, compared to trillions for training from scratch). Experiments span models from 127M to 7B parameters, and the resulting series achieves performance comparable to autoregressive LLMs. Notably, due to their bidirectional nature, DiffuGPT and DiffuLLaMA can perform infilling natively without prompt engineering or token reordering, a challenge for autoregressive models. During inference, they also support a trade-off between generation speed and quality by adjusting the number of diffusion iterations, often requiring fewer refinement steps to achieve fluent outputs than other dLLMs.

IV.B.4 DiffuCoder

DiffuCoder [47] is a 7B-parameter dLLM on a large code corpus, the authors reveal how dLLMs dynamically shift between autoregressive-like and parallel generation patterns. To stabilize preference optimization under diffusion training, they propose coupled-GRPO, a reinforcement learning algorithm that reduces variance via complementary mask sampling, yielding measurable improvements in code generation benchmarks.

IV.B.5 DREAM Series

DREAM 7B [7] is one of the most powerful open-source dLLMs to date. DREAM 7B achieves performance on par with, or exceeding, autoregressive models of similar size (e.g. it matches LLaMA3 8B [128] and Qwen 2.5-7B [129] on many benchmarks). A key to DREAM’s success is an optimized training recipe distilled from extensive experiments at smaller scales. [7] carefully explores the design choices on a 1B model and identify two especially valuable components: (1) AR weight initialization as in [20] and (2) context adaptive noise scheduling as in [53]. The AR initialization of DREAM is chosen to be Qwen2.5 [129].

The Dream series has several follow-up works. Dream-Coder 7B [46] introduces a diffusion-based code model, adapted from Qwen2.5-Coder [130] and trained on 322B tokens. It supports multiple decoding styles, including sketch-first generation, leftto-right decoding, and interleaved reasoning. DreamOn [45] resolves the fixed-canvas limitation of diffusion decoding by introducing special tokens (<|expand|>, <|delete|>) for variablelength generation. During inference, a masked token can be predicted as one of these operations: the <|expand|> token splits the current mask into two masks, while the <|delete|> token removes the current mask. This innovation enables true flexible infilling and significantly improves performance on benchmarks such as HumanEval-Infilling.

IV.B.6 TESS 2

TESS 2 [44] is another dLLM that is not only large-scale but also instruction-following and general-purpose. The training recipe for TESS 2 is a culmination of ideas from prior works, such as AR initialization [20] and reward guidance [38]. TESS 2 starts by adapting a powerful AR base model via continued pretraining on the diffusion objective, and then applies thorough instruction-tuning to that adapted model. TESS 2 finds that both the adaptation procedure and the choice of base model are crucial for a good dLLM. For reward guidance, TESS 2 shows that the choice of reward model exhibits some robustness.

IV.B.7 Seed Diffusion

Seed Diffusion (Preview) [48] is a powerful discrete diffusion model primarily designed for code generation, achieving an inference speed of 2,146 tokens per second on H20 GPUs.

Seed Diffusion employs two types of data noising: 80% masked language modeling and 20% random deletion, insertion, or substitution, with edit distance used to measure differences between sequences. The training loss combines a mask prediction term and an overall reconstruction term. To enable better sequential generation, Seed Diffusion leverages a pretrained diffusion model to generate partial trajectory data, which is then filtered by overall log-likelihood to construct a refined trajectory dataset. To further reduce decoding steps, it introduces a reinforcement learning paradigm that maximizes the edit distance between consecutive steps, encouraging the model to decode as many

- tokens as possible per step while maintaining correctness.

- IV.C Large Diffusion Multimodal Models

- IV.C.1 Dimple Dimple [8] is one of the first Discrete Diffusion Multimodal

Large Language Models (dMLLMs). Its base architecture (vision encoder + transformer LLM) resembles existing visionlanguage models (e.g. Qwen-VL [131], LLaVA [132, 133, 134]). One of the Dimple’s key innovations is its two-stage hybrid training. In Stage 1, with the weights of Dream 7B [7] as an initialization, it is fine-tuned autoregressively on visionlanguage instruction data (for alignment and instruction following). In Stage 2, it is then further fine-tuned with a discrete diffusion objective. This hybrid approach is devised because pure diffusion training is found to be unstable (leading to length bias and performance drop). By warming up with autoregressive training first, Dimple-7B achieves stable training and eventually surpasses the fully-autoregressive models.

During inference, Dimple introduces a confident decoding strategy for efficiency: the model dynamically chooses how many tokens to fill in at each step based on model confidence (see Sec. VI.A). Empirically, this reduces the number of iterations to about response3 length. Dimple also re-implements an autoregressive prefilling trick: by filling in some tokens from the existing context, it speeds up inference by about 1.5×–7× with minimal impact on quality. Under the same training budget and dataset as LLaVA-NEXT [134], Dimple7B achieves higher aggregate scores on multimodal benchmarks than LLaVA-NEXT-7B. This result shows that with a proper hybrid training recipe, a discrete dMLLM can match or exceed strong autoregressive baselines on vision-language tasks.

- IV.C.2 LaViDa LaViDa [10] is among the first models to extend discrete

diffusion into the multimodal large language model (dMLLM) setting. It consists of a vision encoder (e.g. SigLIP-400M [135]) and a discrete diffusion Transformer. Its language model is a standard discrete dLLM (either LLaDA-8B or Dream-7B). LaViDa’s key innovation is complementary masking in training: for each training sample, two distinct mask patterns are created so that each token is masked in one of the two versions. This ensures that even short or rare answer tokens (e.g. object names in vision tasks) contribute to the loss and all tokens are learned efficiently, improving alignment between the visual encoder and the language model. During inference, LaViDa employs a special Prefix-DLM [10] attention mask so that the encoded image and prompt tokens can be cached and reused. The model also uses a timestep-shifting schedule to improve sample quality.

IV.C.3 LLaDA-V

LLaDA-V [9] is one of the pioneering efforts in developing Discrete Diffusion Multimodal Large Language Models (dMLLMs). Built on LLaDA [6], LLaDA-V undergoes three training phases. In the first stage, language–image alignment, the MLP projector is trained to align visual features with LLaDA’s word embeddings. The second stage, visual instruction tuning, fine-tunes the model on large-scale multimodal instruction data to build instruction-following abilities. Finally, the third stage, multimodal reasoning enhancement, focuses on strengthening reasoning capabilities by training on reasoning-focused multimodal QA data with detailed reasoning chains, and further balancing direct answering and explicit reasoning through mixed training with “no think” and “think” tags.

IV.D Large Unified Model IV.D.1 MMaDA

MMaDA [11] employs a unified diffusion architecture with a shared probabilistic formulation across image and text modalities. It uses a single diffusion-based transformer for all data types (text, images, etc.), rather than separate encoders for each modality. During training, MMaDA is fine-tuned with a mixed long chain-of-thought strategy. Reasoning steps from both text and vision tasks are converted into a unified CoT format so that the model learns aligned reasoning across modalities. For example, the rationale for answering a visual question is interleaved into the textual input. This CoT alignment provides a form of cold-start for the final reinforcement learning (RL) stage, allowing complex multi-step reasoning from the outset. Finally, MMaDA proposes a unified policy-gradient-based RL algorithm named UniGRPO. By incorporating diversified reward modeling, UniGRPO unifies the post-training across both reasoning and generation tasks, improving the performance.

IV.D.2 FUDOKI

FUDOKI [49] is a unified multimodal model built on discrete flow matching [136]. It uses a metric-induced probability path with kinetic-optimal velocities [136], which significantly improves over simple masking by enabling continuous selfcorrection during generation. For efficiency, FUDOKI is initialized from a pretrained AR-based multimodal LLM (Janus1.5B [137]) and then transferred to the discrete flow matching framework. For input modalities, text is tokenized normally, while images are handled by separate pipelines: a semantic encoder (SigLIP [135]) extracts features for image understanding, and a pixel encoder/decoder [138] converts images to/from discrete image tokens for generation. At the output stage, FUDOKI has two output heads—one predicting text tokens and one predicting image tokens—and selects the appropriate head depending on the target modality during inference.

IV.D.3 Muddit

Muddit [50] is another unified model that uses purely discrete diffusion to handle text and images under one framework. The architecture of Muddit comprises a single multimodal diffusion transformer (MM-DiT) [50], plus encoders/decoders for each modality. The MM-DiT follows a dual-/single-stream design (similar to FLUX [139]) and is initialized from the pretrained Meissonic [140]. Inputs are quantized into a shared token space: images are encoded by a pretrained VQ-VAE [141] into

discrete codebook indices, and text is encoded by a CLIP text encoder [142]. During training and inference, the MM-DiT predicts masked tokens in this joint space. A linear head maps the predicted tokens to actual text tokens for text output, while the VQ-VAE decoder reconstructs pixels from image tokens.

V TRAINING TECHNIQUES

In this section, we summarize the techniques employed during the training of diffusion language models (dLLMs) and diffusion multimodal language models (dMLLMs).

- V.A Challenges

First, we summarize several challenges encountered in the training of discrete diffusion models. These challenges stem from low corpus utilization and high variance due to stochastic masking.

V.A.1 Low Corpus Utilization

Unlike autoregressive training, where each token in the answer sequence contributes to the learning signal, discrete diffusion training applies supervision only to a randomly selected subset of tokens at each time step. Given an input sequence x of length L = Lprompt + Lanswer, diffusion training samples a timestep t ∈ [1,T] and computes loss only over the masked

- tokens at that timestep. This leads to sparse supervision across training samples, resulting in inefficient utilization of the corpus.

V.A.2 Random Sampling of Time Index

In diffusion training, the time index t is randomly sampled for each training instance. As a result, only a single generation step is supervised per sample, while the decoding process at inference time typically involves multiple time steps. This mismatch introduces a coverage gap between training and inference: although decoding requires refinement over many steps, training provides gradient signals for only one of those steps per instance.

- V.B Initialization Techniques

To address the inefficiencies and instabilities in training dLLMs and dMLLMs, several works adopt advanced initialization strategies that convert the full diffusion training process into a fine-tuning task. This approach accelerates convergence and enhances final model performance.

Because the diffusion generation process can be interpreted as a multi-step masked language modeling (MLM) procedure. [35] initializes diffusion models from pretrained BERT. Moreover, [20, 7] explored direct adaptation from autoregressive language models by aligning the training objectives of the two paradigms. A key technique enabling this transition is the shift operation. In standard diffusion training, the model predicts the original token x0 from its corrupted version xt at each timestep. However, this formulation differs from AR training, where each hidden state hi is trained to predict the next token xi+1 in a left-toright fashion. To bridge this gap, [20, 7] propose shifting the output logits of the diffusion model by one position, such that the model’s prediction at position i corresponds to token xi+1. This allows the diffusion model to be initialized with pretrained autoregressive models.

Another approach similar to initialization is Autoregressivethen-Diffusion Training, making the Diffusion Training as

posting-training of autoregressive training. Dimple [8] uses an autoregressive-then-diffusion training approach, demonstrating notable performance improvements for DMLLM. The Dimple training pipeline is divided into two distinct phases:

- • Phase I: Autoregressive Training. In the first phase, Dimple is treated as an autoregressive model using the causal attention mask and next-token prediction loss.
- • Phase II: Diffusion Fine-tuning. After autoregressive training, Dimple is treated as an diffusion model using the full attention masks and timestep-dependent masked language modeling losses.

V.C Complementary Masking Technique

To improve the utilization of the corpus, in [10], to ensure that all tokens participate in training, complementary masking constructs two complementary masked versions of each input sequence: Xt and XtC. Xt and XtC have non-overlapping masked spans. For example, consider the sentence:

‘‘The answer is dog.’’ One masked variant might be:

‘‘The [M] [M] dog.’’ and its complement:

‘‘[M] answer is [M].’’

This setup ensures that all tokens are eventually masked and optimized over the course of training.

V.D Masking Scheduling Technique

Masking scheduling governs the corruption process in the forward diffusion formulation. Specifically, the schedule defines the corruption level αt at each timestep t, thereby determining the proportion of tokens masked during training. An effective schedule balances learning stability and generation quality by controlling the signal-to-noise ratio across timesteps.

V.D.1 Uniform Masking Scheduling

Masking scheduling can either apply the same scheduling function to all tokens, referred to as uniform masking scheduling, or assign different scheduling functions to individual tokens, referred to as token-wise masking scheduling. We first introduce two commonly used uniform masking scheduling methods.

Given a timestep t ∈ [0,1], the forward process defines the corruption as:

q(xt | x0) = αtx0 + (1 − αt)m, (33)

where m is the one-hot [MASK] token. The loss at each step is reweighted according to:

αt′ 1 − αt

, (34)

wt =

where αt′ = dα

dt is the derivative of αt with respect to time. Linear Schedule and Cosine Schedule are two commonlyused scheduling strategies are as follows. Their corresponding schedule functions are plotted in Fig. 4.

t

##### • Linear Schedule [15]: αt = 1 − t, wt = −

1 t

. (35)

- 0.5

0.75

- 1

0

- -20

- -15

- -10

- -5

wt

t

Linear

Linear

Cosine

Cosine

Geometric

Geometric

0.25

Polynomial r = 2.0 Polynomial r = 0.5 Spindle = 0.1

Polynomial r = 2.0 Polynomial r = 0.5 Spindle = 0.1

Spindle = 0.1

Spindle = 0.1

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

t

t

(a) αt Schedules

(b) wt Schedules

- Fig. 4. Different schedules for αt and wt. To unify notation, we also transformed the spindle schedule from the discrete-time format used in the original paper to a continuous-time format. The revised formulation is as

follows: αt = 1−t−τ sin(πt) and wt = −1+t+τπτ sin(cos(πtπt)), where τ corresponds to the original λH˜.

##### • Cosine Schedule [51]: αt = 1 − cos

π 2

π 2

(1 − t) , wt = −

tan

π 2

(1 − t) .

(36)

The theoretical analyses of the optimal masking scheduling function remain scarce. One pioneering work [143] theoretically proves the optimality of cosine scheduling under the Fisher–Rao geometry. In Appendix Sec.C.I, we include another two uniform scheduling functions.

- V.D.2 Token-wise Masking Scheduling Uniform masking scheduling uses the same scheduling for all

tokens, which ignores the inherent variation in informativeness among different tokens. [35] introduces a token-wise masking schedule, which defines a Spindle-shaped Noise Schedule, where the entropy-aware corruption pattern resembles a spindle curve: more informative tokens are masked earlier and less informative ones later. This design ensures that low-entropy, easier-topredict tokens are decoded first, leading to more coherent and stable sequence generation. The math of the Spindle-shaped Noise Schedule is discussed in Appendix Sec.C.I.

- V.D.3 Block-wise Masking Scheduling For reasoning tasks, the prevailing approach adopts a block-

wise semi-autoregressive decoding strategy. However, previous random masking across the full response, does not reflect this procedure. Let the response be partitioned into M contiguous blocks b(1),...,b(M) of size B. For a given active block a, Blockwise SFT [52] defines:

- • Prefix Iprefix(a) : tokens before block a, kept clean and fixed.
- • Active block I(a): tokens of block a, subject to stochastic masking.
- • Suffix Isuffix(a) : tokens after block a, fully hidden.

In other words, the masking rule is:

 

- 0, i ∈ Iprefix(a) (clean prefix), Bernoulli(π), i ∈ I(a) (masked active block),
- 1, i ∈ Isuffix(a) (fully hidden suffix),

mi =



with π ∼ Uniform(10−3,1). Loss and gradient updates are computed only for the active block.

- V.E Reweighting Technique Multi-Granularity Diffusion Modeling (MGDM) [53] intro-

duces an additional token-level reweighting factor v(xt,n) in

the loss function, yielding:

N

T

w(t) · v(xt,n) · ℓ(x0,xt,n;θ), (37)

LMGDM =

n=1

t=1

where ℓ(x0,xt,n;θ) is the CE loss on the n-th token, and the adaptive token-level weight is defined as:

v(xt,n) = α(1 − exp(−ℓ(x0,xt,n;θ)))β, (38)

with hyperparameters α > 0, β > 0. This reweighting assigns larger weights to harder tokens (i.e., those with higher loss), thereby prioritizing difficult subgoals during training and accelerating convergence.

- V.F Distillation through Dimensional Correlations

To enable efficient few-step or even one-step generation while maintaining performance, Di4C [54] introduces a distillation strategy that compresses a multi-step dLLM into a fewer-step counterpart. It employs two loss functions: *Distillation Loss* and *Consistency Loss*, grounded in distributional matching and multi-path coherence. The distillation loss transfers probabilistic knowledge from a teacher model performing multi-step denoising to a student model designed for fewer-step generation, while the consistency loss enforces stable behavior of the student model across predictions from different intermediate noise levels. The detailed mathematical formulations of these two losses are provided in Appendix Sec. C.II.

- V.G Reinforcement Learning

Reinforcement learning has been extensively applied to discrete diffusion language models [11, 43, 57, 56, 55, 144, 73]. Reinforcement learning for diffusion language models largely inherits the paradigm from autoregressive models; however, it also presents several unique challenges, such as the estimation of likelihood. The following provides a brief summary of several reinforcement learning techniques, with detailed descriptions presented in Appendix Sec. C.IV.

Diffusion-based GRPO (UniGRPO) extends clipped policy optimization by integrating structured noising and KLregularized surrogate rewards [11]. Variance-Reduced Preference Optimization (VRPO) improves stability by replacing intractable log-likelihoods in DPO with ELBO estimates and applying advanced variance reduction techniques [43]. To address reward propagation across trajectories, Stepwise Decomposition Preference Optimization (SDPO) reformulates alignment into tractable per-step KL-regularized objectives [55]. Weighted Policy Optimization (wd1) recasts the objective as a weighted likelihood maximization, where weights derived from centered rewards ensure better sample efficiency [56]. Finally, Diffusion Chain of Lateral Thought (DCoLT) introduces a reinforcement-learned Unmask Policy Module that adaptively controls the token unmasking order during generation [57]. Together, these methods highlight complementary directions for improving stability, efficiency, and controllability in diffusionbased reinforcement learning.

VI INFERENCE TECHNIQUES

In this section, we summarize the techniques involved in the inference phase of dLLMs and dMLLMs. These techniques

affect both performance and efficiency, typically requiring a trade-off between the two. Ideally, the goal is to improve decoding efficiency without compromising performance.

- VI.A Unmasking Techniques

In dLLMs and dMLLMs, the model predicts all the response tokens at each step. However, only a subset of the masked tokens are selected to be unmasked at every iteration, while the remainder remain masked. The main challenges are deciding which and how many tokens to unmask per iteration. Figure 5 provides a detailed illustration of the unmasking strategies. In this section, we discuss each category in detail and describe the specific unmasking strategies proposed in each work.

VI.A.1 Discrete-Time Unmasking

- VI.A.1.(a) Random Unmasking The simplest strategy is to randomly select st masked tokens to unmask at step t. The value of st can be fixed across steps or controlled by a scheduling function as discussed in the training techniques, such as cosine scheduling [9].

- VI.A.1.(b) Metric-Based Unmasking Rather than relying on random selection, metric-based strategies assign a metric value to each token prediction and select tokens to be unmasked based on the metric value.

Let p(i) ∈ RK be the predicted probability distribution over the vocabulary for the i-th token, where K is the vocabulary size. The following are some commonly used metrics.

##### • Maximum Probability (Confidence) [7, 59]: c(i) = max(p(i)), (39)

indicating the model’s certainty about the most likely token. [67] provides a theoretical analysis of the equivalence between the parallel decoding using confidence and sequential decoding using confidence.

###### • Margin [7, 59]: c(i) = p(top1i) − p(top2i) , (40)

where p(top1i) and p(top2i) are the first and second highest probabilities, respectively. This measures how dominant the

top prediction is.

##### • Negative Entropy [7, 59]:

K

p(ki) log(pi + ϵ), (41)

c(i) = −

k=1

with a small constant ϵ for numerical stability. This captures the peakedness of the distribution.

###### • In EB-Sampler [60], instead of performing filtering at the token level, the metric is defined over a set of tokens, aiming to directly decide on unmasking a subset U of tokens.

c(U) =

H(p(l)) − max

H(p(l)) (42)

l∈U

l∈U

with H(p(l)) is the entropy of p(l).

###### • In PC-Sampler. [61], the metric is defined as

c(i) = w(i) · p(xi)

(i) · [−log pD(x(i))], w(i) = e−λ·i, (43)

where x(i) is the predicted token at position i. In this metric, w(i) controls the degree of left-to-right decoding by applying an exponentially decaying weight based on token positions. The parameter λ serves as a hyperparameter.

p(xi)

denotes the model’s predicted probability for the current token, reflecting its confidence. pD represents the token frequency distribution estimated from a publicly available corpus pD, used to downweight trivial words such as “the” and “a”.

(i)

VI.A.1.(c) Selection Policies After obtaining the metric value for each token, the diffusion model performs selection based on different policies.

- • Top-st Strategy [15]: Select the st tokens with the highest confidence scores for unmasking. The value of st follows the same scheduling principles as in random unmasking.
- • Confident Decoding: As introduced in Dimple [8], this strategy dynamically selects the number of tokens to unmask based on a fixed confidence threshold γ ∈ (0,1). The motivation of this approach is that decoding should adapt to the semantic structure of the text: some steps may allow many tokens to be confidently predicted, while others may necessitate more caution. Thus, the number of decoded token should be adaptively adjusted at each step. At step t, the set of positions to decode is defined as:

It = {i | c(ti) ≥ γ}, (44)

where c(ti) is the confidence score at position i. If It is non-empty, all tokens in It are unmasked. Otherwise, only the token with the highest confidence is selected. This approach enables:

- – decoding multiple tokens in parallel when the model is confident, improving efficiency;
- – avoiding low-confidence predictions, preserving generation quality.

- • Block-wise Unmasking: Block-wise semi-autoregressive decoding strategy [6] divides the full response into multiple blocks, similar to block diffusion. During each forward pass, predictions are generated for all blocks simultaneously. However, the unmasking of tokens follows a leftto-right, block-by-block order. Tokens in the next block are only allowed to be unmasked once all tokens in the previous block have been unmasked.
- • Dilated Unmasking Schedule (DUS) [62]: DUS considers the relative distances between tokens during decoding and requires the distances among decoded tokens to decrease

gradually from large to small. Specifically, let Ut denote the selected tokens at time t, DUS selects equally spaced tokens as following:

U0 = ∅, (45) Pt = {k | (k − 1) mod st = 0}, (46) Ut = Ut−1 ∪ Pt, (47)

where st is a dilation coefficient that decreases over time.

VI.B Remasking Techniques

For masked discrete diffusion model, once a token is unmasked, it remains unchanged in subsequent steps. This static

Vocab Size

response response

[Figure 33]

(image) text

(image) text

max(p)

token ...

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

t = 1

t = 1

max(p)

###### >

max(p)

token ...

...

...

max(p)

[Figure 40]

[Figure 41]

1. Maximum Probability (Confidence)

ptop2

ptop1

-

...

[Figure 42]

ptop2 ptop1

ptop1- ptop2

###### >

Discrete Diffusion (M)LLM

Discrete Diffusion (M)LLM

...

[Figure 43]

[Figure 44]

ptop1 ptop2

all confidence below threshold γ < γ

###### < γ < γ

###### > γ

> γ < γ

some confidence above threshold γ

2. Margin

... ...

× × × × × × × ×

unmask the token with the highest confidence

unmask tokens with confidence above threshold γ

log( )

>

highest conf.

× × × × × × × ×

...

...

log( )

t = 0

t = 0

3. Negative Entropy

(a) Metric-Based Unmasking.

(b) Confident Decoding.

confidence decoding

θ t = 1

(image) text

response

(image) text

response

(image) text

response

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Dimple

t = 1

t = 1

...

...

...

[Figure 54]

[Figure 55]

[Figure 56]

current block

threshold-based (dimple)

Discrete Diffusion (M)LLM

Discrete Diffusion (M)LLM

Discrete Diffusion (M)LLM

current timestep s

[Figure 57]

time-dependent probabilistic

Unmask = 2 tokens with the highest confidence

unmask within current block

current block

...

...

...

t = 0

t = 0

t = 0

diffullama, fudoki

(c) Top-st Strategy.

(d) Local Unmasking.

(e) Continuous-Time Unmasking.

top-st strategy

Local Unmasking

- Fig. 5. Unmasking strategies. We divide the unmasking strategies used in dLLMs and dMLLMs into two categories: Discrete-Time Unmasking (a,b,c,d) and Continuous-Time Unmasking (e). In discrete-time unmasking, besides random unmasking, there are other two unmasking strategies: Metric-Based Unmasking

mmada, llada,

(Maximum Probability (Confidence), Margin and Negative Entropy, see (a)) and Selection Policies (Top-st Strategy (see (c)), Confident Decoding (see (b)), and Local Unmasking (see (d))).

diffusion-llms

threshold-based (dimple)

threshold-based (dimple)

behavior limits the model’s capacity to revise or refine earlier predictions. To address this, the remasking technique reintroduces masked tokens at previously unmasked positions, enabling iterative refinement of generated outputs.

with the decoding process of the original response sequence. If some decoded tokens are found to have low confidence, they are replaced with mask tokens.

VI.C Prefilling and Caching Techniques

VI.B.1 Remasking in General Masked Diffusion Models. [63] formulates the reversal diffusion process with remasking

Prefilling and Key-Value Cache (KV-Cache) are standard inference acceleration techniques widely adopted in autoregressive language models. Intuitively, Prefilling and KV-Cache avoids redundant computation by storing the key and value representations from previous decoding steps, enabling the model to reuse them instead of recalculating at each new time step. In autoregressive models, the use of causal attention masks ensures that caching is theoretically lossless. In contrast, dLLMs and dMLLMs employ full (bidirectional) attention mechanisms, wherein each token can attend to all other positions. As a result, even tokens that have already been decoded and unmasked may have their key and value vectors influenced by updates to other tokens during subsequent diffusion iterations. Thus, caching in dLLM and dMLLM are not theoretically lossless.

- as

qσ(zs | zt,x) = Cat(zs;(1 − σt)x + σtm), zt ̸= m, Cat zs; α

(48)

s−(1−σt)αt

1−αt x + 1−α

s−σtαt

1−αt m , zt = m,

where σt is used to control the ratio of remasked tokens. When zt ̸= m, the token has already been decoded. The model samples the next token zs from a distribution that mixes the input x and the mask token m, controlled by σt. This enables remasking by reintroducing uncertainty into already decoded tokens. When zt = m, the token is still masked. The sampling distribution is a weighted combination of x and m, adjusted by both αt and σt, allowing flexible control over how much information from the input or the mask dominates.

For dLLMs, dKV-Cache [65] and dLLM-Cache [66] develops the naive KV-Cache techniques. Their observations are that, with small update intervals (e.g., 2 to 8), caching in dLLMs leads to minimal performance degradation and achieves 10x˜ speedup. For dMLLMs, Dimple [8] and LaViDa [10] empirically verify that the use of prefilling incurs negligible performance degradation on the majority of vision-language benchmarks and provides a 2× to 7× speed-up.

VI.B.2 Wide-In Narrow-Out

Under the block-wise decoding setting, [64] introduces an alternative verification-based remasking approach. Specifically, A appends an additional set of shadow tokens after the current response, with the number of shadow tokens equal to the current block size and each shadow token corresponding to one token in the block. By adjusting the attention mask, the original token sequence is prevented from attending to the shadow tokens, while each shadow token can attend to all tokens except its corresponding token in the current block. Consequently, the shadow tokens verify the decoded tokens without interfering

Following are some representative designs for KV-Cache.

• dKV-Cache [65]. The core idea of dKV-Cache is to cache the key-value pairs when the tokens are unmasked and reuse the cached key-value pairs with an interval hyperparameter to periodically update the cached key-value pairs.

VI.E Sampling Techniques

(image) text

response

[Figure 58]

[Figure 59]

[Figure 60]

t = 1

The generation process of dLLMs relies on a multi-round iterative denoising procedure. Beyond strictly following the mathematical definition of the inverse transition in discrete diffusion, many techniques introduce additional control over the iterative denoising process to achieve better performance or efficiency, such as, majority voting based on decoding history and early stopping. In this subsection, we provide a summary of these techniques.

...

[Figure 61]

Discrete Diffusion (M)LLM

- Step 1: Unmask

- Step 2: Remask

Step 2: Unmask

VI.E.1 Temporal Self-Consistency

[73] identifies a phenomenon termed temporal oscillation, where correct intermediate predictions emerge during the iterative decoding steps but are later overwritten by incorrect outputs in subsequent steps.

...

t = 0

To mitigate the performance degradation caused by temporal oscillation, rather than relying exclusively on the final denoising step, [73] utilizes the predictions at all timesteps and predicts through a weighted voting scheme:

- Fig. 6. Remasking in General Discrete Diffusion Models.

unisisc, llada, llada-v,

- • dLLM-Cache [66]. In addition to key-value (KV) pairs, dLLM-Cache stores attention (AttnOut) and feed-forward (FFNOut) outputs, reusing them in later steps. It applies different update intervals for prompt and response segments, with the prompt cache updated far less frequently. For responses, beyond periodic full updates, an Adaptive Partial Update strategy selectively refreshes tokens: cosine similarity between current and cached values identifies significant changes, and after each forward pass, a subset of tokens is proportionally updated.
- • DualCache [67]. DualCache adopts a block-wise caching strategy, decoding text block by block. The block currently being decoded is not cached, while the surrounding blocks are cached.

T

answer∗ = arg max

f(t) · 1 semantic meaning(xt0) = a ,

a

t=1

(49) where xt0 denotes the prediction at time step t, and semantic meaning(xt0) represents the extracted answer from xt0, 1(·) is the indicator function and f(t) is a weighting function (e.g., exponential decay) assigning more importance to later steps. [73] also introduces Temporal Semantic Entropy (TSE), a measure of semantic uncertainty across intermediate predictions. The computation of TSE is based on the semantic clustering of historical answers. This metric can be incorporated into the reinforcement learning advantage function, thereby mitigating the issue of temporal oscillation in the model.

VI.E.2 Particle Gibbs Sampling for Discrete Diffusion Models Instead of sampling only once to obtain a single trajectory,

VI.D Guidance

[72] introduces a reference trajectory that is iteratively refined across multiple rounds of sampling.

In dLLMs and dMLLMs, post-processing on the predicted logits or sampling probabilities is commonly referred to as guidance, following terminology from image diffusion models. Guidance methods are used to influence the generation process toward desired characteristics, such as improved diversity or controllability. Fig. 7 provides an illustration of several guidance techniques. The detailed mathematical formulations are included in Appendix Sec. D.II. Classifier-free guidance adjusts conditional predictions with unconditional ones to mitigate promptindependent bias and enhance text diversity [145, 21], though excessive guidance may degrade quality [146, 147]. In contrast, classifier guidance explicitly integrates class-conditional signals via an auxiliary classifier, enabling controllable generation across diffusion blocks [145, 68]. Beyond classifier-based methods, reward guidance [44] leverages a reward model to adjust logits at inference by gradient ascent, thus steering outputs toward high-quality responses. Finally, energy-based diffusion augments the denoising distribution with an energy function Eϕ and reweights candidate samples via importance sampling, providing a theoretically grounded correction mechanism [69]. Together, these techniques highlight the trade-offs between diversity, controllability, quality, and theoretical interpretability in guided discrete diffusion.

The method begins by drawing an initial reference trajectory x′0:T using Sequential Monte Carlo (SMC) with a single particle. At each subsequent iteration, a set of k particles is generated, where one particle is deterministically fixed to the reference trajectory while the remaining k −1 particles are defined as the candidate particles and are initialized randomly. At each step t, candidate particles are propagated according to the model’s denoising distribution, x¯(t−i)1 ∼ pθ(xt−1 | c,x(ti)), with the reference particle x¯(t−k)1 kept fixed. The importance weight are for each candidate particles at time t is defined based on a reward function r:

r(c,x¯(t−i)1) − r(c,x(ti)) β

wt(−i)1 = exp

, (50)

and normalized before resampling. After resampling, the reference trajectory is updated by selecting one trajectory from the set with probability proportional to its importance weight. Iterating this procedure m times yields a refined trajectory distribution that converges to the reward-weighted posterior.

VI.E.3 Early Stopping [70] finds that diffusion models can often generate the correct

answer before completing the entire decoding process. There-

θ

θ

(image) text

response

response t = 1

(image) text

response t = 1

(image) text

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

y

t = 1

...

...

...

.........

.........

t =

t =

[Figure 71]

weights

×

Classifier

Discrete Diffusion (M)LLM

Discrete Diffusion (M)LLM

Reward Model

...

...

Discrete Diffusion (M)LLM

unmask all

unmask all

unmask all

unmask all

t =  −1

t =  −1

###### ×

weights

×

t =

t =

weights

×

...

...

...

Reward

Δ

t = 0

t = 0

t = 0

(a) Classifier-Free Guidance.

(b) Classifier Guidance.

(c) Reward Guidance.

classifier-guidance

- Fig. 7. Guidance Techniques. We divide the guidance techniques (i.e., the post-processing on the predicted logits or sampling probabilities) into three categories: (a) Classifier-Free Guidance, (b) Classifier Guidance, (c) and Reward Guidance.

classifier-free guidance

reward model

fore, discrete diffusion can leverage early stopping to improve decoding efficiency. Both [70] and [71] employ either predefined response templates or in-place prompts to divide the response space into two parts: the final answer region and the reasoning region. Once the confidence in the final answer region reaches a sufficient level, the subsequent decoding is terminated.

tokens in the last active block exceeds a predefined threshold. This gradual inclusion of additional blocks reduces computation on the right-side tokens and thereby improves efficiency.

VI.H Response Length Control

Adaptive control of response length is an essential capability in discrete diffusion models, and it can be achieved through either training-based or training-free approaches. In trainingbased methods, DreamOn [45] enables dynamic adjustment of token counts by introducing two special tokens, expand and delete. Similarly, FlexMDM [32] employs an auxiliary network to predict how many tokens should be inserted before each existing token, thereby allowing the response length to grow dynamically. In contrast, the following section introduces training-free techniques for adaptive response length control.

- VI.F Context Length Extension

A notable phenomenon observed in dLLMs is that, with bidriectional attention, their robustness in handling extended contexts. Unlike autoregressive models that fail beyond the pretraining context window, diffusion LLMs can still retrieve information from the most recent segment of the input, even at depths far beyond the training context length.

To further extend the context window, LongLLaDA [74] applies the NTK-aware scaling method to RoPE within diffusion LLMs. Following [148], the critical dimension dextra is:

dextra = 2 ·

d 2 · logβ

0

Ttrain 2π

, (51)

where d is the hidden dimension, β0 is the rotary base, and Ttrain is the pretraining context length. Given a target context length t, the NTK scaling factor λ is chosen as:

λ = β0−1 ·

t 2π

d/dextra

. (52)

- VI.G Sparse Computation

DAEDAL [78] introduces a two-stage adaptive mechanism for response length control. It uses the confidence in predicting the End-of-Sequence (EOS) token as a signal of length sufficiency. First, it starts with a short sequence of length Linit and repeatedly appends the response length until the EOS confidence exceeds τeos or the maximum length Lmax is reached. Second, while denoising, low-confidence tokens (pθ(xt[i]) < τlow) trigger expansion: the least confident position is replaced by Efactor mask tokens, allowing the sequence to grow adaptively.

VII QUANTIZATION

- [79] presents the first systematic study of post-training quan-

tization (PTQ) on dLLMs, benchmarking mainstream quantization methods across multiple models. Their analysis highlights severe activation outliers in dLLMs, sensitivity differences across tasks, and the relative robustness of instruction-tuned variants compared to base models.

- [80] also analyzes why conventional PTQ methods degrade

The forward computation of diffusion-based large language models exhibits inherent sparsity, allowing inference efficiency to be improved by eliminating part of the computations.

Under the block-wise decoding setup, many techniques have been explored. Sparse-dLLM [75] indicates that it is unnecessary to cache and reuse all key–value pairs of tokens outside the current block. By applying an average attention score between non-current tokens and tokens in the current block, a subset of tokens can be filtered out, thereby reducing the memory consumption of the KV cache and improving inference throughput. DPad [76] demonstrates that tokens in subsequent blocks do not need to attend to all tokens in the current block. Instead, a random dropout can be applied, with a higher dropout rate assigned to tokens that are farther from the current block. Pipelined Parallel decoding [77] adopts a “soft” block-wise decoding strategy. Instead of waiting for the current block to be fully decoded before moving to the next, it begins decoding tokens in the subsequent block once the proportion of decoded

severely on dLLMs, identifying three key challenges: (1) quantization errors accumulate across iterations, (2) distinct token distributions across decoding steps, and (3) significant disparities in feature distributions across both token and channel dimensions. To address these, they introduce DLLMQuant, a framework featuring Temporal-Mask Adaptive Sampling (TMAS), Interaction-Aware Activation Quantization (IA-AQ), and Certainty-Guided Quantization (CGQ). The details of these techniques are discussed in Appendix Sec.E.I.

VIII PRIVACY AND SAFETY

dLLMs introduce unique safety vulnerabilities stemming from their bidirectional context modeling and parallel decoding

mechanisms. These features, while enabling efficient infilling and interactive generation, also weaken the defenses that are effective in autoregressive (AR) LLMs.

Diffusion-based LLMs Jailbreak Attack (DIJA) [81] and PArallel Decoding jailbreak (PAD)[82] reformulates conventional jailbreak prompts into an interleaved mask-text format, compelling the model to generate unsafe outputs while maintaining contextual consistency. Formally, let a = (a1,...,aR) denote a vanilla jailbreak prompt, and let m = ([MASK],...,[MASK])Q represent Q consecutive mask tokens. DIJA and PAD construct an interleaved prompt

###### pi = a ⊕ (m ⊗ w), (53)

where ⊕ denotes concatenation, ⊗ denotes interleaving, and w represents benign separator text, such as, “Step 1” and ”Step 2”. Importantly, the hazardous intent contained in a is preserved, while critical instructions are forced into masked positions.

[83] shows that dLLMs are more vulnerable to manipulation

- at the middle of the response than at the initial tokens and previous optimization-based jailbreak methods are effective at manipulating initial tokens but largely fail to optimize middle tokens. Thus, [83] proposes the Middle tOken Safety Alignment (MOSA), a reinforcement learning alignment strategy, which requires the model’s generated middle tokens to align with a set of predefined safe tokens. Experimental evidence shows that optimizing alignment in the middle portion of the sequence is more effective than optimizing alignment at the beginning.

IX APPLICATIONS IX.A Language Related Applications

Recent work applies diffusion language models to language tasks. [84] introduce a diffusion-based model for fine-grained style transfer. [85] leverage the bidirectional nature of diffusion for text embeddings, achieving strong reasoning performance. To enable controllable generation, [86] propose Segment-Level Diffusion, decoding segments sequentially. In applications, [87] use diffusion with task-specific masking and attribute tags to generate poll options; [88] enforce both semantics and meter in poetry; and [89] apply masked diffusion with verifier guidance for style control. EdiText [90] extends controllable generation to text editing, operating at both coarse and fine levels to achieve target attributes. Moving from editing to long-form generation, [91] design a discrete diffusion model for abstractive summarization. Complementing this direction, DiffETM [92] inject diffusion into the embedded topic model, enabling document–topic distributions to be sampled through a more realistic stochastic process. The CDA2 framework [94] uses counterfactual diffusion augmentation to improve cross-domain sentiment adaptation. DiffusionCLS [95] enhances low-resource classification by generating label-consistent pseudo-samples via sentiment-relevant token reconstruction. For aspect sentiment quad prediction, GDP [96] employs a template-guided diffusion strategy. In layout generation, [97] mitigate layout sticking in discrete diffusion models by introducing Layout-Corrector, which scores and resets misplaced tokens for improved positioning. TermDiffuSum [93] integrates term-aware attention and a re-ranking loss during diffusion, effectively emphasizing legally legally salient sentences.

- IX.B Reasoning

Diffusion-of-Thought (DoT) [98] firstly integrates chain-ofthought reasoning into dLLMs to enhance reasoning capabilities. DiffuCOMET [99] develops a series of models that leverage diffusion to infer contextually relevant commonsense knowledge from narratives. DPCL-Diff [100] combines graph node diffusion with dual-domain periodic contrastive learning for temporal knowledge graph reasoning. The d1 framework [101] adapts pretrained dLLMs into reasoning models via a combination of supervised fine-tuning and reinforcement learning. It introduces a novel critic-free policy-gradient algorithm called diffu-GRPO and employs masked SFT to distill reasoning knowledge directly from existing datasets. [53] provides an insight into dLLM reasoning: the difficulty of decoding individual tokens in a response varies, and it is not necessarily the case that tokens on the left are easier to decode than those on the right. NeSyDM [102] introduces a discrete diffusion process in the symbolic space. The framework first extracts symbolic representations from perceptual inputs and then performs a multi-step denoising diffusion process over these symbols.

- IX.C Vision and Multimodal

UDAN-CLIP [103] proposes an image-to-image diffusion framework for underwater enhancement. Turning to motion synthesis, M2D2M [104] employs a discrete diffusion model to generate continuous human-motion sequences from textual descriptions of multiple actions. AR-Diffusion [105] introduces a novel architecture that blends autoregressive and diffusion techniques for flexible, asynchronous video generation. Besides the understanding tasks, discrete diffusion is also largely used in the vision generation tasks [51, 140].

- IX.D Robotics and Autonomous Driving

DiffVLA [106] introduces a vision-language-guided diffusion policy for autonomous-driving planning. ViLaD [108] introduces a large vision–language diffusion framework for endto-end autonomous driving, addressing the latency and unidirectional limitations of autoregressive VLM-based decision models. Meanwhile, Discrete Diffusion VLA [107] unifies vision, language, and action decoding in a single transformer. It discretizes actions into tokens and employs iterative re-masking for adaptive, parallel action generation. Extending diffusion policies to robotics, VPDD [109] first pre-trains on large-scale actionless human videos and then transfers the learned discrete diffusion policy to robot-control tasks. Discrete-Guided Diffusion (DGD) [110] integrates discrete multi-agent path finding (MAPF) with diffusion models to address scalable and safe multi-robot motion planning.

- IX.E Graph and Structured Predictions

LO-ARM [111] introduces a learning-order autoregressive model that dynamically adapts generation order for molecular graphs, improving flexibility and validity. ReDiSC [112] proposes a reparameterized discrete masked diffusion model for node classification and achieves scalable and interpretable predictions on large graphs. Scaffold Diffusion [113] formulates sparse multi-category voxels as discrete token sequences and applies a dLLM for 3D sparse structure generation.

Trends of Discrete Diffusion Research on arXiv

| |Discrete Diffusion Model<br><br>Discrete Diffusion Language<br><br>Discrete Diffusion Large Language<br><br><<br><br><<br><br><<br><br>Predicted Trend| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

600

500

NumberofPapers

400

300

200

100

0

2021 2022 2023 2024 2025 (to date) Year

- Fig. 8. Number of arXiv publications retrieved via keyword-based search (Discrete Diffusion Model, Discrete Diffusion Language, and Discrete Diffusion Large Language) under the Computer Science (cs) category using the All fields search option, which scans across all metadata including titles, abstracts, and author information. The results show a consistent year-over-year increase, reflecting the growing research interest in this area.

IX.F Biological and Drug Discovery

A molecular-editing framework named MolEditRL [114] combines a discrete graph-diffusion model with reinforcement learning to optimize molecular properties while preserving structural similarity. CFP-Gen [115] adopts a diffusion language model for combinatorial functional protein generation. TransDLM [116] proposes a text-guided, multi-property molecularoptimization method that leverages a diffusion language model. GenMol [117] presents a single, discrete diffusion model that serves as a versatile generator across diverse pharmaceutical tasks. DPLM-2 [118] is a multimodal protein language model capable of understanding and generating both protein sequences and their 3D structures. PepTune [119] targets therapeuticpeptide design with a multi-objective, discrete diffusion framework built on a masked language model backbone. [120] propose CMCM-DLM, which integrates structure-control and property-control modules into a pretrained dLLM for molecules.

X CONCLUSION

In summary, this survey provides a comprehensive overview of Discrete Diffusion Large Language Models (dLLMs) and Discrete Diffusion Large Multimodal Models (dMLLMs). We present a detailed exposition of their mathematical foundations and landmark developments. We further detail the training and inference strategies behind them, and summarize the current application domains and potential future directions of them. As a promising alternative to autoregressive LLMs, dLLMs have attracted growing attention (see Figure 8) and show great potential in a variety of real-world scenarios. We hope this survey will serve as a valuable foundation for future research and development in this fast-evolving and important field. At the end of the Appendix, we also discuss some future directions of dLLM and dMLLM.

REFERENCES

- [1] OpenAI , “Gpt-4o system card,” 2024.
- [2] OpenAI, “Gpt-4 technical report,” 2024.
- [3] DeepSeek-AI, “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” 2025.
- [4] Gemini Team , “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” 2024.
- [5] Gemini Team, “Gemini: A family of highly capable multimodal models,” 2025.
- [6] S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. Zhou, Y. Lin, J.R. Wen, and C. Li, “Large language diffusion models,” arXiv preprint arXiv:2502.09992, 2025.
- [7] J. Ye, Z. Xie, L. Zheng, J. Gao, Z. Wu, X. Jiang, Z. Li, and L. Kong, “Dream 7b: Diffusion large language models,” 2025.
- [8] R. Yu, X. Ma, and X. Wang, “Dimple: Discrete diffusion multimodal large language model with parallel decoding,” arXiv preprint arXiv:2505.16990, 2025.
- [9] Z. You, S. Nie, X. Zhang, J. Hu, J. Zhou, Z. Lu, J.-R. Wen, and C. Li, “Llada-v: Large language diffusion models with visual instruction tuning,” arXiv preprint arXiv:2505.16933, 2025.
- [10] S. Li, K. Kallidromitis, H. Bansal, A. Gokul, Y. Kato, K. Kozuka, J. Kuen, Z. Lin, K.-W. Chang, and A. Grover, “Lavida: A large diffusion language model for multimodal understanding,” arXiv preprint arXiv:2505.16839, 2025.
- [11] L. Yang, Y. Tian, B. Li, X. Zhang, K. Shen, Y. Tong, and M. Wang, “Mmada: Multimodal large diffusion language models,” arXiv preprint arXiv:2505.15809, 2025.
- [12] DeepMind, “Gemini diffusion,” https://deepmind.google/models/ gemini-diffusion/, 2025, accessed: 2025-06-16.
- [13] Inception Labs, “Mercury,” https://www.inceptionlabs.ai/ introducing-mercury, 2025, accessed: 2025-06-16.
- [14] M. Prabhudesai, M. Wu, A. Zadeh, K. Fragkiadaki, and D. Pathak, “Diffusion beats autoregressive in data-constrained settings,” 2025.
- [15] J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. Van Den Berg, “Structured denoising diffusion models in discrete state-spaces,” Advances in neural information processing systems, vol. 34, pp. 17981–17993, 2021.
- [16] E. Hoogeboom, D. Nielsen, P. Jaini, P. Forr´e, and M. Welling, “Argmax flows and multinomial diffusion: Learning categorical distributions,” Advances in neural information processing systems, vol. 34, pp. 12454– 12465, 2021.
- [17] L. Zheng, J. Yuan, L. Yu, and L. Kong, “A reparameterized discrete diffusion model for text generation,” arXiv preprint arXiv:2302.05737, 2023.
- [18] S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. Chiu, A. Rush, and V. Kuleshov, “Simple and effective masked diffusion language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 130136–130184, 2024.
- [19] J. Shi, K. Han, Z. Wang, A. Doucet, and M. Titsias, “Simplified and generalized masked diffusion for discrete data,” Advances in neural information processing systems, vol. 37, pp. 103131–103167, 2024.
- [20] S. Gong, S. Agarwal, Y. Zhang, J. Ye, L. Zheng, M. Li, C. An, P. Zhao, W. Bi, J. Han et al., “Scaling diffusion language models via adaptation from autoregressive models,” arXiv preprint arXiv:2410.17891, 2024.
- [21] S. Nie, F. Zhu, C. Du, T. Pang, Q. Liu, G. Zeng, M. Lin, and C. Li, “Scaling up masked diffusion models on text,” 2025.
- [22] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in International conference on machine learning. pmlr, 2015, pp. 2256– 2265.
- [23] E. Haxholli, Y. Z. Gurbuz, O. Can, and E. Waxman, “Efficient perplexity bound and ratio matching in discrete diffusion language models,” in The Thirteenth International Conference on Learning Representations, 2025.
- [24] D. von R¨utte, J. Fluri, Y. Ding, A. Orvieto, B. Sch¨olkopf, and T. Hofmann, “Generalized interpolating discrete diffusion,” in Forty-second International Conference on Machine Learning, 2025.
- [25] A. Campbell, J. Benton, V. De Bortoli, T. Rainforth, G. Deligiannidis, and A. Doucet, “A continuous time framework for discrete denoising models,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 28266–28279.
- [26] C. Meng, K. Choi, J. Song, and S. Ermon, “Concrete score matching: Generalized score matching for discrete data,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 34532–34545.
- [27] H. Sun, L. Yu, B. Dai, D. Schuurmans, and H. Dai, “Score-based continuous-time discrete diffusion models,” in Proceedings of the International Conference on Learning Representations (ICLR), 2023.

- [28] J. Ou, S. Nie, K. Xue, F. Zhu, J. Sun, Z. Li, and C. Li, “Your absorbing discrete diffusion secretly models the conditional distributions of clean data,” 2025.
- [29] R. ZHANG, S. Zhai, Y. Zhang, J. Thornton, Z. Ou, J. M. Susskind, and N. Jaitly, “Target concrete score matching: A holistic framework for discrete diffusion,” in Forty-second International Conference on Machine Learning, 2025.
- [30] I. Gat, T. Remez, N. Shaul, F. Kreuk, R. T. Q. Chen, G. Synnaeve, Y. Adi, and Y. Lipman, “Discrete flow matching,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 133345–133385.
- [31] M. Arriola, A. Gokaslan, J. T. Chiu, Z. Yang, Z. Qi, J. Han, S. S. Sahoo, and V. Kuleshov, “Block diffusion: Interpolating between autoregressive and diffusion language models,” in The Thirteenth International Conference on Learning Representations, 2025.
- [32] J. Kim, L. Cheuk-Kit, C. Domingo-Enrich, Y. Du, S. Kakade, T. Ngotiaoco, S. Chen, and M. Albergo, “Any-order flexible length masked diffusion,” 2025.
- [33] C.-H. Chao, W.-F. Sun, H. Liang, C.-Y. Lee, and R. G. Krishnan, “Beyond masked and unmasked: Discrete diffusion models via partial masking,” 2025.
- [34] A. Zhang, A. Sivakumar, C. Tang, and C. Thomas, “Flexible-length text infilling for discrete diffusion models,” 2025.
- [35] Z. He, T. Sun, K. Wang, X. Huang, and X. Qiu, “Diffusionbert: Improving generative masked language models with diffusion models,” arXiv preprint arXiv:2211.15029, 2022.
- [36] J. Chen, A. Zhang, M. Li, A. Smola, and D. Yang, “A cheaper and better diffusion language model with soft-masked noise,” arXiv preprint arXiv:2304.04746, 2023.
- [37] K. Zhou, Y. Li, W. X. Zhao, and J.-R. Wen, “Diffusion-nat: Selfprompting discrete diffusion for non-autoregressive text generation,” arXiv preprint arXiv:2305.04044, 2023.
- [38] R. K. Mahabadi, H. Ivison, J. Tae, J. Henderson, I. Beltagy, M. E. Peters, and A. Cohan, “Tess: Text-to-text self-conditioned simplex diffusion,” arXiv preprint arXiv:2305.08379, 2023.
- [39] I. Gulrajani and T. B. Hashimoto, “Likelihood-based diffusion language models,” Advances in Neural Information Processing Systems, vol. 36, pp. 16693–16715, 2023.
- [40] A. Lou, C. Meng, and S. Ermon, “Discrete diffusion modeling by estimating the ratios of the data distribution,” arXiv preprint arXiv:2310.16834, 2023.
- [41] A. Swerdlow, M. Prabhudesai, S. Gandhi, D. Pathak, and K. Fragkiadaki, “Unified multimodal discrete diffusion,” arXiv preprint arXiv:2503.20853, 2025.
- [42] J. Ye, Z. Zheng, Y. Bao, L. Qian, and Q. Gu, “Diffusion language models can perform many tasks with scaling and instruction-finetuning,” arXiv preprint arXiv:2308.12219, 2023.
- [43] F. Zhu, R. Wang, S. Nie, X. Zhang, C. Wu, J. Hu, J. Zhou, J. Chen, Y. Lin, J.-R. Wen et al., “Llada 1.5: Variance-reduced preference optimization for large language diffusion models,” arXiv preprint arXiv:2505.19223, 2025.
- [44] J. Tae, H. Ivison, S. Kumar, and A. Cohan, “Tess 2: A large-scale generalist diffusion language model,” arXiv preprint arXiv:2502.13917, 2025.
- [45] Z. Wu, L. Zheng, Z. Xie, J. Ye, J. Gao, Y. Feng, Z. Li, V. W., G. Zhou, and L. Kong, “Dreamon: Diffusion language models for code infilling beyond fixed-size canvas,” 2025. [Online]. Available: https://hkunlp.github.io/blog/2025/dreamon
- [46] Z. Xie, J. Ye, L. Zheng, J. Gao, J. Dong, Z. Wu, X. Zhao, S. Gong, X. Jiang, Z. Li, and L. Kong, “Dream-coder 7b,” 2025. [Online]. Available: https://hkunlp.github.io/blog/2025/dream-coder
- [47] S. Gong, R. Zhang, H. Zheng, J. Gu, N. Jaitly, L. Kong, and Y. Zhang, “Diffucoder: Understanding and improving masked diffusion models for code generation,” arXiv preprint arXiv:2506.20639, 2025.
- [48] Y. Song, Z. Zhang, C. Luo, P. Gao, F. Xia, H. Luo, Z. Li, Y. Yang, H. Yu, X. Qu et al., “Seed diffusion: A large-scale diffusion language model with high-speed inference,” arXiv preprint arXiv:2508.02193, 2025.
- [49] J. Wang, Y. Lai, A. Li, S. Zhang, J. Sun, N. Kang, C. Wu, Z. Li, and P. Luo, “Fudoki: Discrete flow-based unified understanding and generation via kinetic-optimal velocities,” arXiv preprint arXiv:2505.20147, 2025.
- [50] Q. Shi, J. Bai, Z. Zhao, W. Chai, K. Yu, J. Wu, S. Song, Y. Tong, X. Li, X. Li et al., “Muddit: Liberating generation beyond text-to-image with a unified discrete diffusion model,” arXiv preprint arXiv:2505.23606, 2025.
- [51] H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman, “Maskgit: Masked generative image transformer,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 11315–

- 11325.
- [52] B. Sun, Y. Cai, M.-H. Yang, and Y. Wang, “Blockwise sft for diffusion language models: Reconciling bidirectional attention and autoregressive decoding,” 2025.
- [53] J. Ye, J. Gao, S. Gong, L. Zheng, X. Jiang, Z. Li, and L. Kong, “Beyond autoregression: Discrete diffusion for complex reasoning and planning,” in The Thirteenth International Conference on Learning Representations, 2025.
- [54] S. Hayakawa, Y. Takida, M. Imaizumi, H. Wakaki, and Y. Mitsufuji, “Distillation of discrete diffusion through dimensional correlations,” arXiv preprint arXiv:2410.08709, 2024.
- [55] J. Han, A. Wang, M. Xu, W. Chu, M. Dang, Y. Yue, and S. Ermon, “Discrete diffusion trajectory alignment via stepwise decomposition,” 2025.
- [56] X. Tang, R. Dolga, S. Yoon, and I. Bogunovic, “wd1: Weighted policy optimization for reasoning in diffusion language models,” 2025.
- [57] Z. Huang, Z. Chen, Z. Wang, T. Li, and G.-J. Qi, “Reinforcing the diffusion chain of lateral thought with diffusion language models,” 2025.
- [58] M. Asada and M. Miwa, “Addressing the training-inference discrepancy in discrete diffusion for text generation,” in Proceedings of the 31st International Conference on Computational Linguistics. Association for Computational Linguistics, Jan. 2025, pp. 7156–7164.
- [59] J. Kim, K. Shah, V. Kontonis, S. M. Kakade, and S. Chen, “Train for the worst, plan for the best: Understanding token ordering in masked diffusions,” in Forty-second International Conference on Machine Learning, 2025.
- [60] H. Ben-Hamu, I. Gat, D. Severo, N. Nolte, and B. Karrer, “Accelerated sampling from masked diffusion models via entropy bounded unmasking,” 2025.
- [61] P. Huang, S. Liu, Z. Liu, Y. Yan, S. Wang, Z. Chen, and T. Xiao, “Pcsampler: Position-aware calibration of decoding bias in masked diffusion models,” 2025.
- [62] O. Luxembourg, H. Permuter, and E. Nachmani, “Plan for speed: Dilated scheduling for masked diffusion language models,” 2025.
- [63] G. Wang, Y. Schiff, S. Sahoo, and V. Kuleshov, “Remasking discrete diffusion models with inference-time scaling,” arXiv preprint arXiv:2503.00307, 2025.
- [64] F. Hong, G. Yu, Y. Ye, H. Huang, H. Zheng, Y. Zhang, Y. Wang, and J. Yao, “Wide-in, narrow-out: Revokable decoding for efficient and effective dllms,” 2025.
- [65] X. Ma, R. Yu, G. Fang, and X. Wang, “dkv-cache: The cache for diffusion language models,” arXiv preprint arXiv:2505.15781, 2025.
- [66] Z. Liu, Y. Yang, Y. Zhang, J. Chen, C. Zou, Q. Wei, S. Wang, and L. Zhang, “dllm-cache: Accelerating diffusion large language models with adaptive caching,” arXiv preprint arXiv:2506.06295, 2025.
- [67] C. Wu, H. Zhang, S. Xue, Z. Liu, S. Diao, L. Zhu, P. Luo, S. Han, and E. Xie, “Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding,” 2025.
- [68] C. Huang and H. Tang, “Ctrldiff: Boosting large diffusion language models with dynamic block prediction and controllable generation,” 2025.
- [69] M. Xu, T. Geffner, K. Kreis, W. Nie, Y. Xu, J. Leskovec, S. Ermon, and A. Vahdat, “Energy-based diffusion language models for text generation,” in The Thirteenth International Conference on Learning Representations, 2025.
- [70] P. Li, Y. Zhou, D. Muhtar, L. Yin, S. Yan, L. Shen, Y. Liang, S. Vosoughi, and S. Liu, “Diffusion language models know the answer before decoding,” 2025.
- [71] X. Jin, Y. Wang, Y. Gao, Z. Wen, B. Qi, D. Liu, and L. Zhang, “Thinking inside the mask: In-place prompting in diffusion llms,” 2025.
- [72] M. Dang, J. Han, M. Xu, K. Xu, A. Srivastava, and S. Ermon, “Inferencetime scaling of diffusion language models with particle gibbs sampling,” 2025.
- [73] W. Wang, B. Fang, C. Jing, Y. Shen, Y. Shen, Q. Wang, H. Ouyang, H. Chen, and C. Shen, “Time is a feature: Exploiting temporal dynamics in diffusion language models,” 2025.
- [74] X. Liu, Z. Liu, Z. Huang, Q. Guo, Z. He, and X. Qiu, “Longllada: Unlocking long context capabilities in diffusion llms,” 2025.
- [75] Y. Song, X. Liu, R. Li, Z. Liu, Z. Huang, Q. Guo, Z. He, and X. Qiu, “Sparse-dllm: Accelerating diffusion llms with dynamic cache eviction,” 2025.
- [76] X. Chen, S. Huang, C. Guo, C. Wei, Y. He, J. Zhang, H. H. Li, and Y. Chen, “Dpad: Efficient diffusion language models with suffix dropout,” 2025.
- [77] X. Wang, C. Xu, Y. Jin, J. Jin, H. Zhang, and Z. Deng, “Diffusion llms can do faster-than-ar inference via discrete diffusion forcing,” arXiv preprint arXiv:2508.09192, 2025.
- [78] J. Li, X. Dong, Y. Zang, Y. Cao, J. Wang, and D. Lin, “Beyond

- fixed: Training-free variable-length denoising for diffusion large language models,” 2025.
- [79] H. Lin, H. Xu, Y. Wu, Z. Guo, R. Zhang, Z. Lu, Y. Wei, Q. Zhang, and Z. Sun, “Quantization meets dllms: A systematic study of post-training quantization for diffusion llms,” arXiv preprint arXiv:2508.14896, 2025.
- [80] C. Xu and D. Yang, “Dllmquant: Quantizing diffusion-based large language models,” arXiv preprint arXiv:2508.14090, 2025.
- [81] Z. Wen, J. Qu, D. Liu, Z. Liu, R. Wu, Y. Yang, X. Jin, H. Xu, X. Liu, W. Li, C. Lu, J. Shao, C. He, and L. Zhang, “The devil behind the mask: An emergent safety vulnerability of diffusion llms,” 2025.
- [82] Y. Zhang, F. Xie, Z. Zhou, Z. Li, H. Chen, K. Wang, and Y. Guo, “Jailbreaking large language diffusion models: Revealing hidden safety flaws in diffusion-based text generation,” arXiv preprint arXiv:2507.19227, 2025.
- [83] Z. Xie, X. Song, and J. Luo, “Where to start alignment? diffusion large language model may demand a distinct position,” arXiv preprint arXiv:2508.12398, 2025.
- [84] Y. Lyu, T. Luo, J. Shi, T. C. Hollon, and H. Lee, “Fine-grained text style transfer with diffusion-based language models,” arXiv preprint arXiv:2305.19512, 2023.
- [85] S. Zhang, Y. Zhao, L. Geng, A. Cohan, A. T. Luu, and C. Zhao, “Diffusion vs. autoregressive language models: A text embedding perspective,” arXiv preprint arXiv:2505.15045, 2025.
- [86] X. Zhu, G. Karadzhov, C. Whitehouse, and A. Vlachos, “Segmentlevel diffusion: A framework for controllable long-form generation with diffusion language models,” arXiv preprint arXiv:2412.11333, 2024.
- [87] L. Cheng and S. Li, “Diffuspoll: Conditional text diffusion model for poll generation,” in Findings of the Association for Computational Linguistics ACL 2024, 2024, pp. 925–935.
- [88] Z. Hu, C. Liu, Y. Feng, A. T. Luu, and B. Hooi, “Poetrydiffusion: Towards joint semantic and metrical manipulation in poetry generation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 16, 2024, pp. 18279–18288.
- [89] T. K. Padole, S. P. Awate, and P. Bhattacharyya, “Improving text style transfer using masked diffusion language models with inference-time scaling,” arXiv preprint arXiv:2508.10995, 2025.
- [90] C. H. Lee, H. Kim, J. Yeom, and S. Yoon, “Editext: Controllable coarseto-fine text editing with diffusion language models,” arXiv preprint arXiv:2502.19765, 2025.
- [91] D. A. Do, L. A. Tuan, W. Buntine et al., “Discrete diffusion language model for efficient text summarization,” in Findings of the Association for Computational Linguistics: NAACL 2025, 2025, pp. 6278–6290.
- [92] W. Shao, M. Liu, and L. Song, “Diffetm: Diffusion process enhanced embedded topic model,” arXiv preprint arXiv:2501.00862, 2025.
- [93] X. Dong, W. Li, Y. Le, Z. Jiang, J. Zhong, and Z. Wang, “Termdiffusum: A term-guided diffusion model for extractive summarization of legal documents,” in Proceedings of the 31st International Conference on Computational Linguistics, 2025, pp. 3222–3235.
- [94] D. Xin, K. Zhao, J. Sun, and Y. Li, “Cdaˆ2: Counterfactual diffusion augmentation for cross-domain adaptation in low-resource sentiment analysis,” in Proceedings of the 31st International Conference on Computational Linguistics, 2025, pp. 61–72.
- [95] Z. Chen, L. Wang, Y. Wu, X. Liao, Y. Tian, and J. Zhong, “An effective deployment of diffusion lm for data augmentation in low-resource sentiment classification,” arXiv preprint arXiv:2409.03203, 2024.
- [96] L. Zhu, X. Chen, X. Guo, C. Zhang, Z. Zhu, Z. Zhou, and X. Kong, “Pinpointing diffusion grid noise to enhance aspect sentiment quad prediction,” in Findings of the Association for Computational Linguistics ACL 2024, 2024, pp. 3717–3726.
- [97] S. Iwai, A. Osanai, S. Kitada, and S. Omachi, “Layout-corrector: Alleviating layout sticking phenomenon in discrete diffusion model,” in European Conference on Computer Vision. Springer, 2024, pp. 92–110.
- [98] J. Ye, S. Gong, L. Chen, L. Zheng, J. Gao, H. Shi, C. Wu, X. Jiang, Z. Li, W. Bi et al., “Diffusion of thoughts: Chain-of-thought reasoning in diffusion language models,” arXiv preprint arXiv:2402.07754, 2024.
- [99] S. Gao, M. Ismayilzada, M. Zhao, H. Wakaki, Y. Mitsufuji, and A. Bosselut, “Diffucomet: Contextual commonsense knowledge diffusion,” arXiv preprint arXiv:2402.17011, 2024.
- [100] Y. Cao, L. Wang, and L. Huang, “Dpcl-diff: Temporal knowledge graph reasoning based on graph node diffusion model with dual-domain periodic contrastive learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 14, 2025, pp. 14806–14814.
- [101] S. Zhao, D. Gupta, Q. Zheng, and A. Grover, “d1: Scaling reasoning in diffusion large language models via reinforcement learning,” arXiv preprint arXiv:2504.12216, 2025.
- [102] E. van Krieken, P. Minervini, E. Ponti, and A. Vergari, “Neurosymbolic diffusion models,” arXiv preprint arXiv:2505.13138, 2025.
- [103] A. Shaahid and M. Behzad, “Underwater diffusion attention network

- with contrastive language-image joint learning for underwater image enhancement,” arXiv preprint arXiv:2505.19895, 2025.
- [104] S. Chi, H.-g. Chi, H. Ma, N. Agarwal, F. Siddiqui, K. Ramani, and K. Lee, “M2d2m: Multi-motion generation from text with discrete diffusion models,” in European Conference on Computer Vision. Springer, 2024, pp. 18–36.
- [105] M. Sun, W. Wang, G. Li, J. Liu, J. Sun, W. Feng, S. Lao, S. Zhou, Q. He, and J. Liu, “Ar-diffusion: Asynchronous video generation with auto-regressive diffusion,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 7364–7373.
- [106] A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Z. Zhang et al., “Diffvla: Vision-language guided diffusion planning for autonomous driving,” arXiv preprint arXiv:2505.19381, 2025.
- [107] Z. Liang, Y. Li, T. Yang, C. Wu, S. Mao, L. Pei, X. Yang, J. Pang, Y. Mu, and P. Luo, “Discrete diffusion vla: Bringing discrete diffusion to action decoding in vision-language-action policies,” arXiv preprint arXiv:2508.20072, 2025.
- [108] C. Cui, Y. Zhou, J. Peng, S.-Y. Park, Z. Yang, P. Sankaranarayanan, J. Zhang, R. Zhang, and Z. Wang, “Vilad: A large vision language diffusion framework for end-to-end autonomous driving,” arXiv preprint arXiv:2508.12603, 2025.
- [109] H. He, C. Bai, L. Pan, W. Zhang, B. Zhao, and X. Li, “Learning an actionable discrete diffusion policy via large-scale actionless video pretraining,” arXiv preprint arXiv:2402.14407, 2024.
- [110] J. Liang, S. Koenig, and F. Fioretto, “Discrete-guided diffusion for scalable and safe multi-robot motion planning,” arXiv preprint arXiv:2508.20095, 2025.
- [111] Z. Wang, J. Shi, N. Heess, A. Gretton, and M. K. Titsias, “Learning-order autoregressive models with application to molecular graph generation,” arXiv preprint arXiv:2503.05979, 2025.
- [112] Y. Li, Y. Lu, Z. Wang, Z. Wei, Y. Li, and B. Ding, “Redisc: A reparameterized masked diffusion model for scalable node classification with structured predictions,” arXiv preprint arXiv:2507.14484, 2025.
- [113] J. Jung, “Scaffold diffusion: Sparse multi-category voxel structure generation with discrete diffusion,” 2025.
- [114] Y. Zhuang, D. Shen, and Y. Sun, “Moleditrl: Structure-preserving molecular editing via discrete diffusion and reinforcement learning,” arXiv preprint arXiv:2505.20131, 2025.
- [115] J. Yin, C. Zha, W. He, C. Xu, and X. Gao, “Cfp-gen: Combinatorial functional protein generation via diffusion language models,” arXiv preprint arXiv:2505.22869, 2025.
- [116] Y. Xiong, K. Li, W. Liu, J. Wu, B. Du, S. Pan, and W. Hu, “Text-guided multi-property molecular optimization with a diffusion language model,” arXiv preprint arXiv:2410.13597, 2024.
- [117] S. Lee, K. Kreis, S. P. Veccham, M. Liu, D. Reidenbach, Y. Peng, S. Paliwal, W. Nie, and A. Vahdat, “Genmol: A drug discovery generalist with discrete diffusion,” arXiv preprint arXiv:2501.06158, 2025.
- [118] X. Wang, Z. Zheng, F. Ye, D. Xue, S. Huang, and Q. Gu, “Dplm2: A multimodal diffusion protein language model,” arXiv preprint arXiv:2410.13782, 2024.
- [119] S. Tang, Y. Zhang, and P. Chatterjee, “Peptune: De novo generation of therapeutic peptides with multi-objective-guided discrete diffusion,” ArXiv, pp. arXiv–2412, 2025.
- [120] Y. Zhang, Y. Wang, K. V. Nguyen, and P. Hong, “Cross-modality controlled molecule generation with diffusion language model,” arXiv preprint arXiv:2508.14748, 2025.
- [121] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.
- [122] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, “Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension,” arXiv preprint arXiv:1910.13461, 2019.
- [123] G. Bachmann and V. Nagarajan, “The pitfalls of next-token prediction,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, Eds., vol. 235. PMLR, 21–27 Jul 2024, pp. 2296–2318.
- [124] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” arXiv preprint arXiv:2011.13456, 2020.
- [125] T. Kaufmann, P. Weng, V. Bengs, and E. H¨ullermeier, “A survey of reinforcement learning from human feedback,” arXiv preprint arXiv:2312.14925, vol. 10, 2023.
- [126] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al.,

- “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.
- [127] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.
- [128] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.
- [129] Qwen, :, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu, “Qwen2.5 technical report,” 2025.
- [130] B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu, J. Zhang, B. Yu, K. Lu, K. Dang, Y. Fan, Y. Zhang, A. Yang, R. Men, F. Huang, B. Zheng, Y. Miao, S. Quan, Y. Feng, X. Ren, X. Ren, J. Zhou, and J. Lin, “Qwen2.5-coder technical report,” 2024.
- [131] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, “Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond,” arXiv preprint arXiv:2308.12966, 2023.
- [132] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” 2023.
- [133] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” 2023.
- [134] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee, “Llava-next: Improved reasoning, ocr, and world knowledge,” January 2024. [Online]. Available: https://llava-vl.github.io/blog/2024-01-30-llava-next/
- [135] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11975–11986.
- [136] N. Shaul, I. Gat, M. Havasi, D. Severo, A. Sriram, P. Holderrieth, B. Karrer, Y. Lipman, and R. T. Chen, “Flow matching with general discrete paths: A kinetic-optimal perspective,” arXiv preprint arXiv:2412.03487, 2024.
- [137] C. Wu, X. Chen, Z. Wu, Y. Ma, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, C. Ruan et al., “Janus: Decoupling visual encoding for unified multimodal understanding and generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 12966–12977.
- [138] P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan, “Autoregressive model beats diffusion: Llama for scalable image generation,” arXiv preprint arXiv:2406.06525, 2024.
- [139] B. F. Labs, “Flux,” https://github.com/black-forest-labs/flux, 2024.
- [140] J. Bai, T. Ye, W. Chow, E. Song, Q.-G. Chen, X. Li, Z. Dong, L. Zhu, and S. Yan, “Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis,” in The Thirteenth International Conference on Learning Representations, 2024.
- [141] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” Advances in neural information processing systems, vol. 30, 2017.
- [142] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.
- [143] L. Zhang, “The cosine schedule is fisher-rao-optimal for masked discrete diffusion models,” 2025.
- [144] H. He, K. Renz, Y. Cao, and A. Geiger, “Mdpo: Overcoming the traininginference divide of masked diffusion language models,” 2025.
- [145] Y. Schiff, S. S. Sahoo, H. Phung, G. Wang, S. Boshar, H. Dalla-torre, B. P. de Almeida, A. M. Rush, T. PIERROT, and V. Kuleshov, “Simple guidance mechanisms for discrete diffusion models,” in The Thirteenth International Conference on Learning Representations, 2025.
- [146] H. Nisonoff, J. Xiong, S. Allenspach, and J. Listgarten, “Unlocking guidance for discrete state-space diffusion and flow models,” in The Thirteenth International Conference on Learning Representations, 2025.
- [147] K. Rojas, Y. He, C.-H. Lai, Y. Takida, Y. Mitsufuji, and M. Tao, “Theoryinformed improvements to classifier-free guidance for discrete diffusion models,” 2025.
- [148] u/emozilla (Reddit user), “Dynamically Scaled RoPE further increases performance of long context LLaMA with zero fine-tuning,” 2023, post on r/LocalLLaMA, Reddit. Available at: https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/ dynamically scaled rope further increases/.

- [149] T. Salimans and J. Ho, “Progressive distillation for fast sampling of diffusion models,” arXiv preprint arXiv:2202.00512, 2022.
- [150] D. Watson, J. Ho, M. Norouzi, and W. Chan, “Learning to efficiently sample from diffusion probabilistic models,” arXiv preprint

- arXiv:2106.03802, 2021.
- [151] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale,” Advances in neural information processing systems, vol. 35, pp. 30318–30332, 2022.
- [152] C. Zhang, J. X. Morris, and V. Shmatikov, “Extracting prompts by inverting llm outputs,” arXiv preprint arXiv:2405.15012, 2024.
- [153] Y. Chen, H. Lent, and J. Bjerva, “Text embedding inversion security for multilingual language models,” arXiv preprint arXiv:2401.12192, 2024.
- [154] J. X. Morris, W. Zhao, J. T. Chiu, V. Shmatikov, and A. M. Rush, “Language model inversion,” arXiv preprint arXiv:2311.13647, 2023.
- [155] Q. Li, R. Yu, and X. Wang, “Vid-sme: Membership inference attacks against large video understanding models,” arXiv preprint arXiv:2506.03179, 2025.
- [156] Z. Li, Y. Wu, Y. Chen, F. Tonin, E. Abad Rocamora, and V. Cevher, “Membership inference attacks against large vision-language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 98645– 98674, 2024.
- [157] Y. He, B. Li, L. Liu, Z. Ba, W. Dong, Y. Li, Z. Qin, K. Ren, and C. Chen, “Towards label-only membership inference attack against pretrained large language models,” in USENIX Security, 2025.
- [158] J. Zhang, J. Sun, E. Yeats, Y. Ouyang, M. Kuo, J. Zhang, H. F. Yang, and H. Li, “Min-k%++: Improved baseline for detecting pre-training data from large language models,” arXiv preprint arXiv:2404.02936, 2024.
- [159] C.-L. Wang, Q. Li, Z. Xiang, Y. Cao, and D. Wang, “Towards lifecycle unlearning commitment management: Measuring sample-level unlearning completeness,” arXiv preprint arXiv:2506.06112, 2025.
- [160] N. Carlini, J. Hayes, M. Nasr, M. Jagielski, V. Sehwag, F. Tramer, B. Balle, D. Ippolito, and E. Wallace, “Extracting training data from diffusion models,” in 32nd USENIX Security Symposium (USENIX Security 23), 2023, pp. 5253–5270.
- [161] Y. Qu, X. Shen, X. He, M. Backes, S. Zannettou, and Y. Zhang, “Unsafe diffusion: On the generation of unsafe images and hateful memes from text-to-image models,” in Proceedings of the 2023 ACM SIGSAC conference on computer and communications security, 2023, pp. 3403– 3417.
- [162] Y. Zhang, J. Jia, X. Chen, A. Chen, Y. Zhang, J. Liu, K. Ding, and S. Liu, “To generate or not? safety-driven unlearned diffusion models are still easy to generate unsafe images... for now,” in European Conference on Computer Vision. Springer, 2024, pp. 385–403.
- [163] X. Shen, Z. Chen, M. Backes, Y. Shen, and Y. Zhang, “”do anything now”: Characterizing and evaluating in-the-wild jailbreak prompts on large language models,” in Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security, 2024, pp. 1671– 1685.
- [164] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, “Scaling laws for neural language models,” arXiv preprint arXiv:2001.08361, 2020.
- [165] Y. Bahri, E. Dyer, J. Kaplan, J. Lee, and U. Sharma, “Explaining neural scaling laws,” Proceedings of the National Academy of Sciences, vol. 121, no. 27, p. e2311878121, 2024.
- [166] L. Floridi and M. Chiriatti, “Gpt-3: Its nature, scope, limits, and consequences,” Minds and Machines, vol. 30, pp. 681–694, 2020.
- [167] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in neural information processing systems, vol. 35, pp. 27730–27744, 2022.
- [168] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang et al., “Qwen technical report,” arXiv preprint arXiv:2309.16609, 2023.
- [169] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.
- [170] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. R´e, “Flashattention: Fast and memory-efficient exact attention with io-awareness,” Advances in neural information processing systems, vol. 35, pp. 16344–16359, 2022.
- [171] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, “Deep learning with differential privacy,” in Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, 2016, pp. 308–318.
- [172] C. F. G. D. Santos and J. P. Papa, “Avoiding overfitting: A survey on regularization methods for convolutional neural networks,” ACM Computing Surveys (Csur), vol. 54, no. 10s, pp. 1–25, 2022.
- [173] X. Ying, “An overview of overfitting and its solutions,” in Journal of physics: Conference series, vol. 1168. IOP Publishing, 2019, p. 022022.
- [174] D. Chen, Y. Huang, Z. Ma, H. Chen, X. Pan, C. Ge, D. Gao, Y. Xie, Z. Liu, J. Gao et al., “Data-juicer: A one-stop data processing system for large language models,” in Companion of the 2024 International Conference on Management of Data, 2024, pp. 120–134.

[175] M. Li, Y. Zhang, S. He, Z. Li, H. Zhao, J. Wang, N. Cheng, and T. Zhou, “Superfiltering: Weak-to-strong data filtering for fast instruction-tuning,” arXiv preprint arXiv:2402.00530, 2024.

### - Appendix -

In this appendix, we provide additional details on the works and techniques introduced in the main text. Specifically, we present more comprehensive mathematical formulations for certain methods and include several related techniques not covered earlier. The organization of the appendix largely follows the structure of the main-text. Moreover, at the end of the appendix, we also discuss potential future research directions in dLLM and dMLLM.

defined as

∞

Qt = exp(αtR) =

n=0

αtn n!

Rn, (54)

Rij = − l̸=i Ail i = j Aij i ̸= j

. (55)

This transition matrix promotes a diffusion process where tokens are more likely to transition to others that are semantically or syntactically similar in the embedding space.

APPENDIX A MATHEMATICAL FORMULATIONS

Apdx.A.I More Transition Matrices

- • Hybrid: Hybrid transition was initially discussed in [15], where multiple types of transition are combined to create more expressive diffusion processes. For example, the Routlette Diffusion in [23] and the Generalized Interpolating Discrete Diffusion (GIDD) [24] both study the linear combination of the absorbing transition and the uniform transition. A key motivation for such “Linear + Absorbing” transitions is to overcome a fundamental limitation of absorbing diffusion: once a token is denoised, it cannot be revised. Such “Linear + Absorbing” transitions address this by allowing previously denoised tokens to re-enter the diffusion process via the uniform component. This grants the model the ability to correct earlier generation errors during inference.
- • Discretized Gaussian:

[Qt]ij =

 



exp − (K4|−i−1)j2|2βt K−1

n=−(K−1)

exp − (K−4n1)22βt

if i ̸= j

1 −

l̸=i

[Qt]il if i = j

This matrix simulates Gaussian diffusion, suitable for ordinal data. Each state i is most likely to transition to nearby states j with probabilities resembling a Gaussian kernel. Closer states receive higher probabilities, while distant states receive lower ones. Diagonal values are chosen to ensure that each row sums to 1, yielding a uniform stationary distribution.

- • Band-diagonal:

[Qt]ij =

 



1

Kβt 0 < |i − j| ≤ v

- 0 |i − j| > v
- 1 − l̸=i

[Qt]il i = j

Band-diagonal imposes local transitions only: state i can only transition to its v-nearest neighbors on the ordinal scale. It biases the forward process toward small, local perturbations.

- • Embedding-based: Let A be an adjacent matrix build from based on the similarity between tokens in the embedding space, in [15] the Embedding-based transition matrix is

Apdx.A.II Reparameterized Discrete Diffusion Model

Reparameterized Discrete Diffusion Models (RDMs) [17] reformulate the backward process of discrete diffusion in D3PM into a two-stage sampling procedure. The core idea is to introduce a latent routing variable vt that determines decoding behavior for each token at every step.

Given a forward transition of the form:

q(xt | xt−1) = βtxt−1 + (1 − βt)qnoise, (56) q(xt | x0) = αtxt−1 + (1 − αt)qnoise, (57)

where qnoise is the noise distribution, and αt := ti=1 βi. The backward posterior q(xt−1 | xt,x0) can be expressed as a mixture of two cases, depending on whether xt = x0:

- λ(1)t−1xt + (1 − λ(1)t−1)qnoise, xt = x0,
- λ(2)t−1x0 + (1 − λ(2)t−1)qnoise(xt), xt ̸= x0,

q(xt−1 | xt,x0) =

(58) where qnoise(xt) is the interpolation between xt and qnoise, and λ(1)t−1, λ(2)t−1 are scalar coefficients derived from βt, αt and qnoise(xt).

To reparameterize the backward transition, RDM introduces Bernoulli latent variables:

- vt(1)−1 ∼ Bernoulli(λ(1)t−1), u(1)t ∼ qnoise, (59)
- vt(2)−1 ∼ Bernoulli(λ(2)t−1), u(2)t ∼ qnoise(xt). (60)

Let bt = 1[xt = x0], the reparameterized sample is computed as:

xt−1 = bt vt(1)−1xt + (1 − vt(1)−1)u(1)t

+ (1 − bt) vt(2)−1x0 + (1 − vt(2)−1)u(2)t . (61)

This formulation allows the model to explicitly route tokens through different behaviors: either retaining the current token, resetting to noise, or denoising back to the ground truth.

Based on Eq. (61), RDMs define a joint generative model:

T

pθ(xt−1,vt−1 | xt), (62)

pθ(x0,x1:T,v1:T) = pθ(xT)

t=1

and the evidence lower bound (ELBO) becomes:

T

log pθ(x0) ≥ L1 −

t=2

LT

where C is a constant.

###### +C, (63)

Lt

For t > 1, the loss term decomposes as:

Lt = Ex

1:T,v1:T|x0 KL(q(vt−1) ∥ pθ(vt−1 | xt))+ KL(q(xt−1 | vt−1,xt,x0) ∥ pθ(xt−1 | vt−1,xt)) . (64)

By aligning pθ(vt−1 | xt) with q(vt−1) and using the x0parameterization, the loss can be simplified into

0,x1:T − λ(2)t−1

Lt = Ex

N

(1 − bt,n)x0,n log[fθ(xt)]n , (65)

n=1

where λ(2)t−1 = α

t−1−αt 1−αt .

A central issue of RDM lies in its dependence on the ground truth x0 to compute the backward transition probabilities Eq. (58). However, in the inference stage, x0 is unknown, making it infeasible to directly evaluate the indicator function bt required for determining the appropriate transition path. To overcome this limitation, the authors propose a recursive approximation for computing bt by utilizing the Bernoulli routing variables v. Beginning with bT = 0, which assumes the initial sequence is fully noisy, the clean token set is recursively updated via:

bt−1,n = bt,n ∧ vt(1)−1,n ∨ vt(2)−1,n, (66)

where ∧ and ∨ denote logical conjunction and disjunction, respectively.

Apdx.A.III Concrete Score

Apdx.A.III.1 Training Loss Training is typically done by minimizing divergence-based

losses that compare the predicted ratio to the true ratio.

- Apdx.A.III.1.(a) Concrete Score Matching (CSM) [26]

The most direct approach is Concrete Score Matching (CSM) [26], which uses a squared-error loss to match the predicted and true ratios:

LCSM(t) =

- 1

- 2

Ex∼p

t

 

x′̸=x

sθ(x,t)x′ −

pt(x′) pt(x)

2

  (67)

While theoretically sound, this ℓ2 loss does not penalize invalid (e.g., negative or zero) predictions sufficiently, potentially leading to instability.

- Apdx.A.III.1.(b) Diffusion-Weighted Denoising Score Entropy [40] Leveraging Bregman divergence, score entropy is formulated

- as another score matching loss. Score entropy is non-negative, symmetric, and convex. It is also an extension of the conventional cross-entropy to general positive-valued functions beyond the probability simplex. Score entropy enables the construction of an ELBO for discrete diffusion models, resulting in the Diffusion-Weighted Denoising Score Entropy (DWDSE)

loss [40]

T

Rt(xt,x′)

Ex

LDWDSE(x0) =

t∼pt|0(·|x0)

0

x′̸=xt

pt|0(x′|x0) pt|0(xt|x0)

sθ(xt,t)x′ −

log sθ(xt,t)x′

pt|0(x′|x0) pt|0(xt|x0)

dt, (68)

+ K

where K is a normalizing constant function.

Apdx.A.III.1.(c) Target Concrete Score Matching [29]

Target Concrete Score Matching (TCSM) introduces two score matching losses: the score-based and distribution-based losses. The score-based objective operates directly on the concrete score vectors:

L

ℓiscore, (69)

Lscore(θ) = Eω(t)p(x

t)h(x1|xt)

i=1

  ,

  

 

 

V

V

pθ1|t([yi,x\1i]|xt) p1θ|t(x1|xt)

p1|t([yi,x\1i]|xt) p1|t(x1|xt)

ℓiscore = D

,

yi=1

yi=1

(70) where V is the vocabulary size, D(·,·) denotes a divergence measure, ω(t) is the distribution used to sample time index t, and h(x1|xt) is a proposal distribution such as the ground-truth conditional distribution p1|t(x1|xt). In the equation, [yi,x\i] := [x1,...,xi−1,yi,xi+1,...,xL] is used to define a new sequence of the same length as x1, where the token at position i is replaced by yi and all other tokens remain identical to those in x1. x\1i is used to indicate all tokens in x1 except for the token at position i.

The distribution-based objective aligns the model and true conditional distributions:

L

1 |xt)ℓidistrib, (71) ℓidistrib = D p1|t(xi1|x\1i,xt)∥pθ1|t(xi1|x\1i,xt) , (72)

Ldistrib(θ) = Eω(t)p(x

Eh(x\i

t)

i=1

where D(·,·) is a statistical divergence measures the differences between probability distributions. Shown in Proposition 3 of [29], with h(x1|xt) = p1|t(x1|xt), the two objectives are equivalent:

###### Lscore(θ;D = DGKL) ≡ Ldistrib(θ;D = V DKL + DIS), (73)

where DGKL, DKL and DIS refer to the generalized Kullback–Leibler divergence, the Kullback–Leibler divergence and Itakura–Saito divergence, respectively.

By selecting different discrepancy measures and proposal distributions, there are different instantiations of the TCSM loss. For instance, choosing the generalized KL divergence as the discrepancy D and the true conditional distribution p1|t(x1|xt)

- as the proposal h(x1|xt), the score-based TCSM becomes

ℓiscore = −log pθ1|t(xi1|xt) +

1 V pθ1|t(xi1|xt)

+

1 V y

log pθ1|t(y|xt); (74)

choosing the KL divergence as the discrepancy D and the true conditional distribution p1|t(x1|xt) as the proposal h(x1|xt), the distribution-based TCSM becomes

ℓidistrib = −Ep

1|t

log pθ1|t(xi1|xt) + C, (75)

where C is a constant. The distribution-based objective reduces to a cross-entropy loss, maximizing the pseudo-likelihood of pθ1|t under the true denoising distribution.

- Apdx.A.III.2 Connection with CE Loss

[27] leverages the theorem stating that two probability distributions are identical if and only if their conditional distributions are equal for all values of the condition. Based on this, the original marginal probability distributions in the concrete score are transformed into conditional probability distributions. Since both the numerator and denominator of the concrete score share the same functional form, they can be represented by a single neural network. That is, the concrete score can be expressed as:

pt(Xd | x\d;θ) ≈ qt(Xd | x\d) ⇒

qt(yd,x\d) qt(xd,x\d)

=

qt(Xtd = yd | x\d) qt(Xtd = xd | x\d) ≈

pt(Xtd = yd | x\d;θ) pt(Xtd = xd | x\d;θ)

. (76)

From this perspective, [27] propose the categorical ratio matching loss, which is a cross-entropy loss to fit conditional probabilities. Thus, [27] shows that training a neural network with a cross-entropy loss is also able to fit the concrete score.

[23] also connects the concrete score matching with the CE loss. Consider two discrete sequences x and y that only differ

- at position i. By Bayesian Theorem, the probability ratio p

t(y) pt(x)

can be rewritten as

pt|0(yi | h) pt|0(xi | h) · pi0|t(h | xt), (77)

pt(y) pt(x)

=

h∈V

where pt|0(· | h) is the known conditional transition probability in the forward process, and pi0|t(h | xt) is the posterior of the clean token h at position i given the noised sequence xt. Thus, the concrete score can be parameterized as

pt|0(yi | h) pt|0(xi | h) · fθi(xt,t)[h], (78)

siθ(xt,t) =

h∈V

where fθi(xt,t)[h] models the likelihood that token h was the original clean token at position i. This allows training to proceed

via a simple cross-entropy loss over the posterior p0|t, rather than requiring explicit score estimation:

L

w(t) · log fθi(xt,t)[xi0], (79)

LCEDD = EtEx

0,xt

i=1

where w(t) is a timestep-dependent weighting function.

- Apdx.A.III.3 Time Independency An issue with the Concrete Score is its dependency on time,

which prevents the use of caching techniques for inference and results in lower efficiency. [28] shows that the concrete score in absorbing discrete diffusion can be reparameterized as a product of a time-dependent scalar and a conditional distribution over clean data. Specifically, in practice, Rt can be parameterized

- as the multiplication between a scalar function and a constant rate, i.e., Rt = σ(t)R. let xi = [M] denote a masked token
- at position i. Then, the concrete score for replacing xi with a token x′i ̸= [M] is defined as:

pt(x′) pt(x)

=

pt(x1,...,x′i,...) pt(x1,...,xi,...)

=

e−σ¯(t) 1 − e−σ¯(t) · p0(x′i | xUM),

(80) where:

- • σ¯(t) = 0 t σ(s)ds is the cumulative noise schedule,
- • xUM consists of all unmasked tokens of x,
- • p0(x′i | xUM) is the conditional distribution of the clean data.

This reparameterization removes the dependence on t from the model output, enabling the training of a time-independent network cθ:

cθ(x)[i,xˆ(i)] ≈ p0(ˆx(i) | xUM). (81)

Such a model, termed RADD (Reparameterized Absorbing Discrete Diffusion), permits caching of the network outputs across steps when tokens remain unchanged, reducing the number of function evaluations (NFEs) during inference.

APPENDIX B MODELING LANGUAGE DIFFUSION

- Apdx.B.I Block Diffusion Models

Given a sequence x = (x1,...,xL), BD3-LMs partition it into B blocks of length L′, denoted x = (x(1),...,x(B)). The joint likelihood under a Block Diffusion model is factorized autoregressively across blocks:

log pθ(x) =

B

log pθ(x(b) | x(<b)), (82)

b=1

where each conditional distribution pθ(x(b) | x(<b)) is modeled by a discrete diffusion process applied within block b:

pθ(x(sb) | x(tb),x(<b))

###### q(x(sb) | x(tb),x(b))pθ(x(b) | x(tb),x(<b)), (83)

=

x(b)

where q(· | ·) is the forward noise process, and pθ(x(b) | x(tb),x(<b)) is the learned denoising model.

The model is parameterized by a transformer fθ with a blockcausal attention mask. For each block x(b), the model predicts:

###### fθ x(tb),x(<b) → xˆ(0b). (84)

During inference, block sampling proceeds sequentially over blocks, but parallel sampling is used within blocks. Block-wise KV caching can be used to avoid recomputing transformer states for previously generated blocks.

The training loss for the full sequence is obtained by summing the variational lower bound over all blocks:

B

L(x(b),x(<b);θ), (85)

−log pθ(x) ≤ LBD(x;θ) :=

b=1

where each L(x(b),x(<b);θ) follows the discrete diffusion loss, optionally adapted to continuous time or masking processes.

- Apdx.B.II Flexible-Length Masked Diffusion

In a standard Masked Diffusion Models, each token can only exist in one of two states: masked or unmasked. In FlexMDM, each token may reside in one of three states: empty, masked, or ground truth (gt). In the forward process, a token first transitions from the ground truth state to the masked state, and finally to the empty state. Let x1 = (x11,...,xn1) ∼ p1 be a target sequence of length n. FlexMDM defines two smooth monotone schedules α,β : [0,1] → [0,1] with boundary conditions (α0,β0) = (0,0) and (α1,β1) = (1,1). For each token i, FlexMDM independently sample an insertion time T1i and an unmasking time T2i:

β˙t 1 − βTi

T1i ∼ α˙t dt, T2i ∼ 1{t ≥ T1i}

###### dt,

1

where α˙t and β˙t are the derivatives of αt and βt. The token state xit at time t ∈ [0,1] evolves as:

 

∅, 0 < t < T1i, m, T1i ≤ t < T2i, xi1, T2i ≤ t ≤ 1,

xit =



where m denotes the special mask symbol and ∅ represents token removal. The sequence xt is obtained by concatenating all non-empty xit.

Thus, in FlexMDM, there are two complementary tasks:

- 1) Unmasking Task. Given a partially masked sequence

xt, the model predicts the ground-truth token for each masked position. This is identical to the original MDM formulation. Specifically, a diffusion language model fθ is trained to approximate the posterior distribution over clean tokens.

- 2) Insertion Task. Beyond unmasking, FlexMDM introduces the additional task of inserting tokens into the

sequence. For this purpose, an auxiliary network gθ is trained to predict the expected number of mask tokens that should be inserted before each existing token. During inference, this prediction is combined with a Poisson distribution to determine the actual number of insertions:

###### ki ∼ Poisson(r · gθ(xt,t)[i]),

where r is a scaling factor. The sampled k tokens are inserted as mask symbols m, thereby extending the sequence length.

At inference time, sequence generation alternates between the two tasks:

- 1) Insertion: For each position i, sample ki ∼ Poisson(r · gθ(xt,t)[i]) and insert ki mask tokens into the sequence.
- 2) Unmasking: Apply fθ to predict clean tokens at masked positions and replace them accordingly.

This insertion–unmasking cycle is repeated iteratively until the sequence converges to a fully unmasked form, thereby producing variable-length outputs while preserving the anyorder property of diffusion-based inference.

- Apdx.B.III Diffusion with Optimal Transport Position Coupling

At each timestep t, DDOT can output both (i) the predicted token value distribution and (ii) the velocity of token positions:

sθ(xt,t) and vθ(zt,t), (86)

where xt are token values and zt ∈ [−1,1]L are continuous position variables. The extra velocity of token position is predicted by and additional linear head. The token positions are diffused in continuous space from an initial distribution zT ∼ U(−1,1)L to the ground-truth positions z0. The token denoising objective follows the score entropy loss from SEDD [40], while the position denoising is learned via a weighted mean-squared error loss:

Lpos(θ) = E(z,t) Qt(xt,y)∥vθ(zt,t) − (z0 − zT)∥2 . (87)

APPENDIX C TRAINING TECHNIQUES

- Apdx.C.I Masking Scheduling Technique

- Apdx.C.I.1 Additional Uniform Masking Scheduling strategies

- • Geometric Schedule [40, 19]: αt = exp −β¯min1−tβ¯maxt , (88)

wt =

exp −β¯min1−tβ¯maxt 1 − exp −β¯min1−tβ¯maxt

β ¯min1−tβ¯maxt log

β ¯min β¯max

. (89)

- • Polynomial Schedule [19]: αt = 1 − tr, (90)

wt = −

r t

. (91)

- Apdx.C.I.2 Token-wise Masking Scheduling The Spindle-shaped Noise Schedule defines a custom cor-

ruption probability αti for each token position i at timestep t, determined by:

t T − S(t) · H˜(xi0), (92)

αti = 1 −

tπ T

, (93)

S(t) = λsin

n j=1 H(xj0)

H˜(xi0) = 1 −

, (94)

n · H(xi0)

where H(xi0) denotes the entropy of the i-th token, measuring its information content, S(t) is a sinusoidal scaling function that ensures zero informativeness contribution at t = 0 and t = T, λ is a hyperparameter controlling the strength of entropy influence.

Apdx.C.II Distillation through Dimensional Correlations

The distillation loss conveys the probabilistic knowledge of a teacher model that performs multi-step denoising to a student

model designed for fewer-step generation. This is achieved by aligning the student’s predicted posterior with that of the teacher

- at an intermediate noise level δ. Formally, the loss is defined as:

δ∼rδ DKL pψ0|δ(· | xδ)pθ0|δ(· | xδ) (95)

Ldistil(θ) = Ex

where pψ0|δ(·|xδ) is the posterior distribution over clean data x0 given intermediate noisy input xδ under the teacher model;

pθ0|δ(·|xδ) is the student model’s predicted posterior; rδ is a reference distribution over noisy states (typically chosen to

match the forward diffusion at timestep δ). This loss encourages to transfer the full-step generative knowledge from teacher to student by matching posteriors.

The consistency loss enforces that the student model produces stable predictions across varying intermediate noise levels. Specifically, if xt is a noisy sample at step t, there should be agreement between: 1). First denoising xt → xu via the teacher, then xu → xs via the student; 2). Directly predicting xt → xs via the student. This intuition is captured by the following KL divergence:

t∼rt DKL (pθs|u ◦ pψu|t)(· | xt)∥pθs|t(· | xt)

Lconsis(θ) = Ex

(96)

where pψu|t(xu|xt) is the teacher’s distribution from timestep t to u; pθs|u(xs|xu) is the student’s distribution from u to s; pθs|t(xs|xt) is the direct prediction by the student; (pθs|u ◦ pψu|t)(·|xt) denotes the composite distribution over xs via intermediate xu. This loss enforces functional coherence across generation paths, capturing multi-dimensional correlations without assuming independence.

- Apdx.C.III Input Discrepancy Between the Training and Testing

[58] mentioned a discrepancy between training and inference of dLLM. During training, the model receives ground-truth noisy tokens as input, while at inference time, the inputs of the model are the previously predicted tokens. To address this, [58] propose the two-step loss.

Let x0 denote the ground truth sequence. During training, a time step t is randomly selected, and a noised version xt is generated from x0 via a diffusion process. The model then predicts the original sequence xˆ0 = fθ(xt). Subsequently, a second input xˆt−1 is generated by applying the forward diffusion transition matrixto the predicted sequence xˆ0 and the model again attempts to recover the ground truth xˆ0 = fθ(ˆxt−1). The two-step loss is calculated between the ground truth x0 and the twice-denoised output xˆ0.

To ease training in early stages, the model does not always use this two-step strategy. Instead, a mixed strategy is adopted. With probability 1 − pk, the two-step strategy is used and the loss is evaluated between xˆ0 and x0. With probability pk, the conventional one-step strategy is used and the loss is evaluated between xˆ0 and x0. pk is set to be linearly increasing along the training step k.

Apdx.C.IV Reinforcement Learning

Apdx.C.IV.1 Diffusion-based GRPO

UniGRPO in [11] adapts policy optimization to the diffusion setting by combining structured noising, likelihood approximation, and clipped policy gradients. Let q and {oi}Gi=1 denote a query and a batch of responses, respectively. For each response oi, UniGRPO samples a masking ratio pi ∼ U[0,1] and create a perturbed version o˜i,p by masking tokens. The token-level likelihood and the sequence-level likelihood are approximated as

###### πθ′ (oi,t | q,o˜i,p,pi) = Ep

1[˜oi,t,p = [M]]log pθ(oi,t,p | q) ,

i

(97) πi′ =

1 M o

log pθ(oi,t | q). (98)

i,t∈M

Following GRPO, UniGRPO loss is an integration of the clipped surrogate rewards Ri,t with the KL regularization:

 .

  1

|oi|

G

1 |oi|

Ri,t − βDKL(πθ′ ∥πref′ )

JUniGRPO(θ) = E

G

t=1

i=1

(99)

Apdx.C.IV.2 Variance-Reduced Preference Optimization LLaDA 1.5 [43] introduces the Variance-Reduced Prefer-

ence Optimization (VRPO) which replaces the intractable loglikelihoods in DPO with Evidence Lower Bounds (ELBOs)

Bπ(y | x) ≜ Et∼U[0,1]Ey

t∼q(yt|t,y,x)ℓπ(yt,t,y | x) (100) ≤ log π(y | x), (101)

where ℓπ is the mask prediction loss. To reduce estimator variance, VRPO introduces (i) increased sampling budgets, (ii) full sampling budget over timesteps, and (iii) antithetic sampling that shares noise between πθ and πref.

Apdx.C.IV.3 Stepwise Decomposition Preference Optimization

A central challenge in reinforcement learning alignment for discrete diffusion models is how to propagate reward information through the entire denoising trajectory. Stepwise Decomposition Preference Optimization (SDPO) [55] reformulates trajectory alignment into a collection of tractable per-step alignment objectives.

The standard KL-regularized reward optimization objective for diffusion trajectories is: max

Ep

θ(x0:T|c) r(x0,c) −βDKL pθ(x0:T | c) ∥ pref(x0:T | c) ,

pθ

(102) where c is the conditioning context, r(x0,c) is a reward function on the clean sequence, and pref is the reference model. However, this requires sampling and scoring the entire trajectory, which is computationally prohibitive. Instead of optimizing the whole trajectory, SDPO aligns the per-step posterior:

pθ(x0 | xt,c), (103) which admits exact likelihood evaluation under masked diffu-

sion models. The stepwise alignment objective is:

- 2) Compute a total transition rate λ(i) =

x̸=x(ti)

u(ti)(x,x(ti) | xˆ(1i));

- 3) Draw a random variable Z ∼ U(0,1);
- 4) If Z ≤ 1 − e−hλ

Ep

θ(x0|xt,c)[r(x0,c)]−βtDKL pθ(x0 | xt,c)||pref(x0 | xt,c) ,

- max

pθ

(104) with step-dependent regularization βt = β/w(t). The work shows that this step-wise optimization is equivalent to a distribution matching problem, which can be used to further simplify the loss function.

, update x(ti) by sampling from:

(i)

u(ti)(x,x(ti) | xˆ(1i)) λ(i)

x(t+i)h ∼

.

Apdx.C.IV.4 Weighted Policy Optimization Weighted Policy Optimization (wd1) [56] reformulates the

This process dynamically determines which tokens to update (i.e., unmask) based on local transition rates. The higher the rate λ(i), the more likely token i will jump to a new value, allowing the model to continuously refine its predictions in a semantically meaningful way.

reinforcement learning objective as a weighted likelihood objective.

Under the reverse-KL–regularized policy optimization, the target solution π∗ has the closed form. wd1 minimizes the KL divergence DKL(π∗ ∥πθ), which reduces to a weighted negative log-likelihood:

Apdx.D.I.2 Remasking under Discrete Flow Matching. In the discrete flow matching framework [30], remasking

is incorporated via a velocity field that interpolates between forward and backward update directions:

G

###### Lwd1(θ) = −Eq∼D,{o

w(q,oi) · log πθ(oi | q) .

i}∼πref-old

i=1

(105) To address the issue where samples with small advantages receive disproportionately low weights, the weights are defined as:

###### u¯it(xi,z) = αtuˆit(xi,z) − βtuˇit(xi,z), (110)

where uˆit and uˇit denote the forward-time and backward-time velocity fields, respectively. This combined velocity u¯it is valid as long as αt,βt > 0 and satisfies the probability flow condition for t ∈ (0,1). When αt − βt = 1, each step progresses forward in time with remasking (corrector sampling), enabling iterative refinement. When αt−βt = 0, the model operates in a stationary regime (corrector iteration), reintroducing noise and adjusting tokens within a fixed diffusion step.

w(q,oi) = −w+(q,oi) + w−(q,oi), (106) w+(q,oi) =

exp(ψAi)

, (107)

G j=1 exp(ψAj)

exp(−ψAi)

w−(q,oi) =

, (108)

G j=1 exp(−ψAj)

Apdx.D.II Guidance Techniques Apdx.D.II.1 Classifier-Free Guidance [145, 21] introduce the unsupervised classifier-free guidance

where Ai is the centered reward and ψ is a scaling factor. Apdx.C.IV.5 Diffusion Chain of Lateral Thought

Diffusion Chain of Lateral Thought (DCoLT) [57] introduces a Unmask Policy Module (UPM), which is trained via reinforcement learning to control the decoding order. The UPM learns a ranking-based policy over masked tokens:

strategy for discrete diffusion generation. The method performs two forward predictions at each diffusion timestep t:

- • A conditional prediction, conditioned on both the prompt p0 and the noisy response rt;
- • An unconditional prediction, conditioned on a sequence of mask token M and the same response rt.

###### hiθ,t = UPM(xt−1,t,mi), (109)

where hiθ,t is the predicted score for token i at step t, and mi is its mask indicator. A Plackett–Luce distribution is then used to sample a top-K unmasking set Ut. Once Ut is selected, the model predicts token values for those positions using the standard diffusion langauge model. The UPM is implemented as a lightweight transformer block attached to diffusion langauge model.

The unconditional prediction captures the model’s inherent bias in the absence of instruction signals. The final guided prediction is adjusted as:

pθ(r0 | p0,rt)1+w pθ(r0 | m,rt)w

, (111)

p˜θ(r0 | p0,rt) ∝

where w is a tunable hyperparameter controlling the strength of the guidance. This guidance promotes text diversity by reducing the dominance of generic, encouraging prompt-independent responses.

APPENDIX D DECODING TECHNIQUES

- Apdx.D.I Unmasking and Remasking for Continuous-Time Models

In the continuous-time setting, [146] provides the formulation of classifier-free guidance applied to the reverse transition matrix

Apdx.D.I.1 Continuous-Time Unmasking (Flow Matching) In continuous-time inference under the discrete flow matching

1−w

w qt(x) qt(x′)

pt(x) pt(x′)

framework (e.g., FUDOKI [49]), unmasking is modeled as a stochastic jump process along a probability path.

Rˆt(x,x′) = Rt(x′,x)

, (112)

Let xt denote the current sequence state at time t ∈ [0,1], and let x1 be the target sequence. For each token position i, the update from xt to xt+h is governed by:

where distribution qt is the reference distribution. [147] further demonstrate that applying strong guidance during early decoding can severely degrade sample quality. This degradation arises because high guidance accelerates the unmasking rate,

1) Sample a predicted target xˆ(1i) ∼ pθ(x(1i) | xt);

leading to overly confident and premature token predictions. To mitigate this issue, the authors suggests applying a column-wise normalization of the guided transition matrix.

Apdx.D.II.2 Classifier Guidance To improve controllability, for block diffusion model, [145,

68] introduce the classifier guidance framework that integrates class-conditional preferences into the sampling process.

At each diffusion step t for block b, the guided reverse process modifies the original sampling distribution pθ by incorporating the signal from a classifier pξ, yielding:

pγ(xsb | xtb,x<b,y) ∝ pθ(xsb | xtb,x<b) · pξ(y | xsb,xtb,x<b)γ, (113)

where y is the desired class label and γ controls the influence of the classifier. To reduce computational complexity, the method assumes intra-block independence and approximates the classifier as:

L

pξ(y | xsb,xtb,x<b) ≈

pξ(y | xˆℓb,t|s,x<b), (114)

ℓ=1

where xˆℓb,t|s denotes the sequence with the ℓ-th token in xtb replaced by the candidate token xsb,ℓ. This allows the guided probability to be reformulated as:

pγ(xsb | xtb,x<b,y) =

L

pθ(xsb,ℓ | xtb,x<b) · pξ(y | xˆℓb,t|s,x<b)γ x′ pθ(x′ | xtb,x<b) · pξ(y | xˆ′ℓ b,t|s,x<b)γ

. (115)

ℓ=1

By integrating classifier predictions with the model’s native probabilities, this approach enables fine-grained, attributeconditioned generation across blocks while maintaining computational feasibility.

Apdx.D.II.3 Reward Guidance

TESS 2 [44] represents a unique approach under the extramodel guidance category by leveraging an external reward model to guide token prediction. The main purpose of this method is to improve the quality of the generated response. Specifically, at each diffusion step, the model output Sˆθ is first transformed into a token probability distribution:

pt = softmax(Sˆθ), (116) cw = Ept, (117)

where E is the embedding matrix. The resulting continuous representation cw is fed into the reward model, which outputs a scalar reward R ∈ R.

To maximize this reward, TESS 2 performs gradient ascent on the logits by computing ∇θR and updates the model output:

###### Sˆθ := Sˆθ + η · ∇θR, (118)

where η is a tunable guidance coefficient. This process is performed during inference only, and no guidance is applied during training. By incorporating gradient signals from the reward model, TESS 2 can steer the generation towards more desirable outputs without modifying the base diffusion model.

Apdx.D.II.4 Energy-Based Diffusion [69] proposes the Energy-based Diffusion Language Model

(EDLM), which augments a pretrained diffusion model pθ(x0 |

xt) with an unnormalized energy model Eϕ(x0,xt,t), yielding a joint denoising distribution:

exp(−Eϕ(x0,xt,t)) Zϕ(xt)

, (119)

pθ,ϕ(x0 | xt) = pθ(x0 | xt) ·

where Zϕ(xt) is the intractable partition function required for normalization. This residual formulation corrects the denoising distribution by reweighting samples from the diffusion model using the energy function. The energy function can be derived from either a pretrained autoregressive (AR) model or a finetuned diffusion model. To implement this energy functionbased guidance, [69] adopts an importance sampling strategy. The decoding process at each time step can be summarized as follows.

- 1) Generate k candidate samples {x(0i)}ki=1 ∼ pθ(x0 | xt) using the diffusion model.
- 2) Compute the unnormalized energy scores e(i) = Eϕ(x(0i),xt,t) for all samples in {x(0i)}ki=1.
- 3) Sample one x0 from the candidate pool according to importance weights:

w(i) =

exp(−e(i))

k j=1 exp(−e(j))

. (120)

- 4) Use the sampled x0 to perform one denoising step via the backward posterior:

###### xt−1 ∼ q(xt−1 | xt,x0). (121)

APPENDIX E QUANTIZATION Apdx.E.I DLLMQuant

[80] introduces DLLMQuant, a framework includes Temporal-Mask Adaptive Sampling (TMAS), Interaction-Aware Activation Quantization (IA-AQ), and Certainty-Guided Quantization (CGQ).

TMAS addresses the calibration challenge in dLLMs by accounting for variations across time steps and masking ratios. Specifically, it divides the iterative generation into blocks and selects calibration inputs at specific time intervals, ensuring coverage of diverse mask ratios across all timesteps.

Quantization errors L(xt) at time step t in dLLMs accumulate geometrically across denoising steps. Formally, the error propagation can be expressed as:

L(xt) = xt − Deq(Q(xt + L(xt+1)))

(122)

= QModel(xt+1) − QModel(Deq(Q(xt+1)))

where Q(·) is the quantization process, Deq is the dequantization operation, and QModel denotes the quantized model.

A key source of error is the matrix multiplication between softmax outputs and the value matrix in attention. IA-AQ redefines the quantization loss for value matrix V as:

2 F

L(s) = ⌊V −s z⌋ − V · Deq(Osoftmax)

, (123)

where Osoftmax is the softmax output, z is zero-point, and s is the scale factor. The optimal scaling factor is chosen by:

Vmax − Vmin Qmax − Qmin

. (124)

L(α ⊙ sˆ), sˆ =

s = arg min

α∈{1.0,0.8}

In addition, not all tokens equally affect subsequent iterations. Errors on unmasked or low-confidence tokens are small, while masked tokens with high confidence dominate the next step. To account for this, CGQ incorporates token certainty into the Hessian used for weight quantization:

###### H = X⊙(1[Xt = M]+√sct) · X⊙(1[Xt = M]+√sct) ⊤,

(125) where 1[Xt = M] indicates a custom weighted indicator function, and sct is the final confidence score to each token in model output. This certainty-weighted Hessian prioritizes minimizing quantization error on critical masked tokens.

[163], diffusion models generate content in a more holistic way. This makes real-time content moderation non-trivial—dLLMs might only reveal problematic content once the final denoised text is produced. These areas remain critical future directions to address before dLLMs can be responsibly deployed at scale.

APPENDIX F FUTURE DIRECTIONS

- Apdx.F.I Infrastructure

The infrastructure for dLLMs remains relatively underdeveloped compared to their autoregressive counterparts. In the autoregressive paradigm, the community has benefited from mature open-source models, standardized training frameworks, and reproducible pipelines that facilitate rapid iteration and deployment at scale. Therefore, establishing standardized modular and scalable training frameworks, and open-sourced pretrained models will be critical directions for the community. Building a robust infrastructure will not only promote fair comparisons and accelerate innovation, but also enable practical deployment across a wide range of real-world applications.

- Apdx.F.II Inference Efficiency

Despite their recent successes, dLLMs still face substantial limitations in inference efficiency and system scalability [6, 10, 65, 66]. Future work can explore several key directions to improve their deployability and performance. At the architectural level, incorporating efficient attention mechanisms and multi-scale token representations may help reduce the compute burden during inference. In terms of the denoising process, advancing fast sampling techniques—such as progressive distillation [149] and adaptive timestep scheduling [150, 51]—could accelerate generation without compromising quality. In addition, integration with quantized inference (e.g., INT8 or INT4) [151] may yield high-throughput, low-latency generation pipelines.

- Apdx.F.III Security and Privacy

The security and privacy implications of dLLMs are an emerging concern as these models become more widely used. Diffusion models share similar risks with other large generative models [152, 153, 154, 155, 156, 157, 158, 159]: they can inadvertently memorize and regurgitate sensitive training data, raising the possibility of privacy leakage or copyright violations. Recent studies have demonstrated that diffusion models trained on vast internet data can reproduce portions of their training examples [160], much like LLMs. In addition, security in terms of model misuse and alignment is another crucial aspect. Like any powerful language model, dLLMs could be misused to generate harmful, false, or biased content [161, 162]. One challenge is that controlling a diffusion model’s output may require new methods: unlike AR models that can be guided token-by-token or halted upon generating disallowed tokens

