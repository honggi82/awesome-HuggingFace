## Probabilistic Conceptual Explainers: Trustworthy Conceptual Explanations for Vision Foundation Models

Hengyi Wang*1 Shiwei Tan*1 Hao Wang1

# arXiv:2406.12649v3[cs.LG]31Oct2024

### Abstract

Vision transformers (ViTs) have emerged as a significant area of focus, particularly for their capacity to be jointly trained with large language models and to serve as robust vision foundation models. Yet, the development of trustworthy explanation methods for ViTs has lagged, particularly in the context of post-hoc interpretations of ViT predictions. Existing sub-image selection approaches, such as feature-attribution and conceptual models, fall short in this regard. This paper proposes five desiderata for explaining ViTs – faithfulness, stability, sparsity, multi-level structure, and parsimony – and demonstrates the inadequacy of current methods in meeting these criteria comprehensively. We introduce a variational Bayesian explanation framework, dubbed ProbAbilistic Concept Explainers (PACE), which models the distributions of patch embeddings to provide trustworthy post-hoc conceptual explanations. Our qualitative analysis reveals the distributions of patch-level concepts, elucidating the effectiveness of ViTs by modeling the joint distribution of patch embeddings and ViT’s predictions. Moreover, these patch-level explanations bridge the gap between image-level and dataset-level explanations, thus completing the multi-level structure of PACE. Through extensive experiments on both synthetic and real-world datasets, we demonstrate that PACE surpasses state-of-the-art methods in terms of the defined desiderata1.

*Equal contribution 1Department of Computer Science, Rutgers University, New Jersey, USA. Correspondence to: Hengyi Wang <hengyi.wang@rutgers.edu>.

Proceedings of the 41st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).

1Code will soon be available at https://github.com/Wang-MLLab/interpretable-foundation-models

### 1. Introduction

Vision Transformers (ViTs) (Dosovitskiy et al., 2020) and their variants (Liu et al., 2021; Touvron et al., 2021; Radford et al., 2021) have emerged as pivotal models in computer vision, leveraging stacked self-attention blocks to encode raw inputs and produce patch-wise embeddings as contextual representations. Given their increasing application in highrisk domains such as autonomous driving, explainability has become a critical concern.

To date, post-hoc explanations in computer vision often involve attributing predictions to specific image regions. However, we identify two primary limitations in current methods: (1) Existing conceptual explanation methods (Fel et al., 2023a;b; Ghorbani et al., 2019; Zhang et al., 2021; Wang et al., 2020; Novello et al.; Chen et al., 2024; Li et al., 2021a; Wang et al., 2023) are not fully compatible with transformer-based models like vision transformers (ViTs), and they also fall short in offering a cohesive structure for dataset-image-patch analysis of input images. (2) Current state-of-the-art methods (Li et al., 2020; Pan et al., 2021; Agarwal et al., 2022; Colin et al., 2022; Xie et al., 2022; Wang et al., 2022; Fel et al., 2023b; Chen et al., 2023) evaluate visual concepts through subjective human utility scores or limited quantitative analysis, lacking a fair and consistent comparison framework. To address this, we propose a comprehensive set of desiderata for post-hoc conceptual explanations for ViTs, namely (see formal definitions in Sec. 3.2):

- • Faithfulness: The explanation should be faithful to the explained ViT and able to recover its prediction.
- • Stability: The explanation should be stable for different perturbed versions of the same image.
- • Sparsity: For each prediction’s explanation, only a small subset of concepts are relevant.
- • Multi-Level Structure: There should be dataset-level, image-level, and patch-level explanations.
- • Parsimony: There are a small number of concepts in total (see Appendix B for more details).

While previous research (Kim et al., 2018; Fel et al., 2023b; Oikarinen et al., 2023; Gilpin et al., 2018; Murdoch et al., 2019) has proposed and met different dimensions of the learned concepts, these studies often lack a comprehensive

evaluation. In this paper, propose ProbAbilistic Concept Explainers (PACE) to provide trustworthy conceptual explanations aligned with these desiderata, drawing inspiration from hierarchical Bayesian deep learning (Wang & Yeung, 2016;

- 2020; Wang et al., 2016). For example, to enable multilevel explanations, we (1) model K concepts as a mixture of K Gaussian patch-embedding distributions, (2) treat the explained ViT’s patch-level embeddings as observed variables, (3) learn a hierarchical Bayesian model that generates these embeddings in a top-down manner, from dataset-level concepts through image-level concepts to patch-level embeddings, and (4) infer these multi-level concepts as our multi-level conceptual explanations; to enhance stability, our hierarchical Bayesian model ensures that the inferred concepts from two different perturbed versions of the same image remain similar to each other. Our contributions are:

- 1. We comprehensively study a systematic set of five desiderata faithfulness, stability, sparsity, multi-level structure, and parsimony when generating trustworthy concept-level explanations for ViTs.
- 2. We develop the first general method, dubbed ProbAbilistic Concept Explainers (PACE), as a variational Bayesian framework that satisfies these desiderata.
- 3. Through both quantitative and qualitative evaluations, our method demonstrates superior performance in explaining post-hoc ViT predictions via visual concepts, outperforming state-of-the-art methods across various synthetic and real-world datasets.

- 2. Related Work

Vision Transformers. Vision Transformers (ViTs) have revolutionized computer vision by adapting the Transformer architecture for image recognition. The pioneering ViT model processes images as sequences of patches, surpassing traditional convolutional networks in efficiency and performance (Dosovitskiy et al., 2020). Subsequent innovations include the Swin Transformer (Liu et al., 2021), which introduces a hierarchical structure with shifted windows, and the Data-efficient Image Transformers (DeiT) (Touvron et al., 2021), which optimize training with a distillation token and teacher-student strategy. The CLIP model (Radford et al., 2021) extends ViT’s applicability by learning from natural language supervision, showcasing the architecture’s versatility and robustness in visual representation learning.

Visual Explanation Methods. The landscape of visual explanation methods (Gilpin et al., 2018; Langer et al.,

- 2021; Schwalbe & Finzel, 2023) in computer vision is diverse, encompassing both feature attribution and conceptbased approaches. Prominent methods such as LIME and SHAP (Ribeiro et al., 2016; Lundberg & Lee, 2017; Simonyan et al., 2013; Li et al., 2021b; Shrikumar et al.,

2017) provide insights by assigning importance scores to

input features, enhancing the understanding of model decisions. Alongside these, concept-based explanations are also gaining popularity. Inherent methods (Chen et al., 2019; Alvarez Melis & Jaakkola, 2018; Koh et al., 2020; Kim et al., 2018; Chattopadhyay et al., 2024; Xu et al., 2024) learn and deduce concepts alongside the prediction model. These methods necessitate modifications to the models for explanations, posing challenges in scalability to new model architectures and increased computational demands.

To address these challenges, post-hoc methods (Yuksekgonul et al., 2022; Pan et al., 2021; Fel et al., 2023b; Sundararajan et al., 2017; Bach et al., 2015; Kindermans et al., 2016; Rohekar et al., 2024; Xie et al., 2022; Covert et al.,

- 2022; Bennetot et al., 2022) deduce concepts from the existing prediction model without additional modifications. Given such advantages, our work focuses on the post-hoc setting. These methods are pivotal in image-level explanation for ViTs, providing deeper insights into ViTs’ visual data processing. Nevertheless, their focus remains on imagelevel explanations, overlooking the multi-level structure within ViTs. They also fall short in other desiderata such as faithfulness/stability. Some methods require additional text supervision or human-annotated labels, such as (Yang et al.,
- 2023; Ben Melech Stan et al., 2024; Menon & Vondrick,

- 2022; Chefer et al., 2021b;a; Ming et al., 2022; Kim et al.,
- 2023; Losch et al., 2019). Therefore, these approaches are not applicable to our unsupervised learning setting.

In contrast, our PACE provides multi-level conceptual explanations that are faithful, stable, sparse, and parsimonious; this is verified by our empirical results in Sec. 4.

### 3. Methodology

In this section, we formalize the definition of five desiderata for post-hoc conceptual explanations of ViTs and describe our PACE for achieving these desiderata.

###### 3.1. Problem Setting and Notations

Consider a dataset comprising M images, each dissected into J patches as per the model in (Dosovitskiy et al., 2020). We analyze a vision transformer, denoted as f(·), which processes image m (represented by Im) and yields: (1) the predicted label ym with N classes, (2) patch embeddings em ≜ [emj]Jj=1 with emj ∈ Rd (d is the hidden dimension), and (3) the attention weights a(mh) ≜ [a(mjh)]Jj=1 (am(h) ∈ RJ) for each patch relative to the final layer’s ‘CLS’ token, where h signifies the attention head h. We define the mean attention weight across H heads as amj = H1 Hh=1 a(mjh), and consequently am ≜ [amj]Jj=1 (refer to the ViT shown at the bottom of Fig. 1). A typical post-hoc explainer, denoted as g(·), processes the contextual representation em, predicted label ym, and optionally ViT parameters P, producing a

∥θm−θ′

|𝐲m|
|---|

m∥

∥θm∥ , where ∥ · ∥ denotes the L2 norm.

- (3) Sparsity, which involves the concept vector having a sparse representation, measured by the fraction of values nearing zero. Specifically, sparsity is defined

as the proportion of θm’s entries nearing zero, i.e.,

1 K

K k=1 1(|θmk| < ϵ), with a small threshold ϵ > 0.

- (4) Multi-Level Structure, which means that an ideal explainer should yield K dataset-level variables

{Ωk}Kk=1 = {µk,Σk}Kk=1 representing the mean and covariance of each concept in the dataset, an image-level

variable θm ∈ RK for each image m, and a patch-level variable ϕmj ∈ RK for each patch j in image m.

- (5) Parsimony, which involves using the minimal number of concepts K to produce clear and simple explanations for humans. Methods with flexible concept counts could use fewer concepts while maintaining other properties’ performance. In contrast, too many concepts usually lead to redundancy in conceptual explanations and a lack of compact representation.

|PACE|
|---|

|𝐌𝐋𝐏 𝐇𝐞𝐚𝐝|
|---|

patch embeddings

attentions

| | |
|---|---|
| | |

||𝒆mn|𝒆mn|𝒆 …<br><br>mn|
|---|---|---|
<br><br>|𝒆mJ|
|---|
<br><br>…| |
|---|---|
|…<br><br>…| |

…

|𝒆mn|𝒆mn|𝒆 …<br><br>mn|
|---|---|---|

|𝒆𝑪𝑳𝑺|𝒆𝐦𝟏|
|---|---|

am0 am1 amJ

…

| | |
|---|---|

| | | |
|---|---|---|

ViT

[Figure 1]

…

…

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

|[𝑪𝑳𝑺]|
|---|

… …

- Figure 1. Overview of PACE framework. PACE utilizes patch embeddings em, model predictions ym, and multi-head attentions am as observations to infer hidden parameters.

concept activation θm ∈ RK (K is the number of concepts), that is, g(em, ym,P) = θm. See Appendix D for details. Note that while some methods do not inherently provide explanations with authentic concepts, the explanation activation θm (or its suitably adapted version) can still be interpreted as a quasi-concept vector.

In Definition 3.1, Property (1) ensures the learned concepts convey essential information for predicting image labels from hidden embeddings. Property (2) guarantees robustness and generalization in face of perturbations. Property (3) reflects that each prediction usually only involves a small number of relevant concepts. Property (4) offers diverse and comprehensive multi-level conceptual explanations. Finally, Property (5) facilitates learning concepts efficiently, restricts the number of redundant concepts for meaningful explanations, and reduces humans’ cognitive load reading explanations. Theorem 3.2 in Sec. 3.7 provides the theoretical guarantees for PACE in terms of these properties.

In contrast to typical post-hoc explainers that only provide image-level explanations θm, our PACE provides multilevel conceptual explanations; for an image m, PACE provides K dataset-level variables {Ωk}Kk=1 = {µk,Σk}Kk=1 (µk ∈ Rd is the Gaussian mean, and Σk ∈ Rd×d the Covariance), an image-level variable θm, and J patch-level variables ϕm ≜ {ϕmj}Jj=1 (see details in Sec. 3.2 and Sec. 4).

###### 3.2. Definition of Trustworthy Conceptual Explanations

###### 3.3. ProbAbilistic Conceptual Explainers (PACE)

We formally define the five desiderata for trustworthy conceptual explanations for ViTs as follows (see Sec. 3.1).

Drawing inspiration from hierarchical Bayesian deep learning (Wang & Yeung, 2016; 2020; Wang et al., 2016; ?; Wang & Yan, 2023; Xu et al., 2023), we introduce a variational Bayesian framework, dubbed ProbAbilistic Concept Explainers (PACE), for post-hoc conceptual explanation of Vision Transformers (ViTs). To ensure PACE produces trustworthy concepts as defined in Definition 3.1, PACE treats the explained ViT’s patch-level embeddings as observed variable and design a hierarchical Bayesian model that generates these embeddings in a top-down manner, from datasetlevel concepts through image-level concepts to patch-level embeddings.

Definition 3.1 (Trustworthy Conceptual Explanations). Consider a dataset D with M images Im (m ∈ 1,...,M), each consisting of J patches. For a given number of concepts K, a trustworthy conceptual explanation for an image m consists of K dataset-level variables {Ωk}Kk=1 = {µk,Σk}Kk=1, an image-level variable θm, and J patchlevel variables {ϕmj}Jj=1 with the following properties:

- (1) Faithfulness, which implies a strong relation between

the concept activation θm and the post-hoc label ym derived from ViT predictions. In this paper, we measure linear faithfulness score by applying a logistic regression model LR(·), i.e., ym = LR(θm)(1 ≤ m ≤ M), and evaluating its accuracy (details in Sec. 4).

- (2) Stability, which is the consistency of explanations across different perturbed versions of the same image. For an image Im with the inferred θm and its perturbed version I′m with inferred θ′m, stability is quantified by

Fig. 1 shows an overview of our PACE, where patch embeddings em and ViT’s predicted label ym are treated as observable variables. Attention weights am are considered as the virtual count for each patch; for example, amj = 0.2 means patch j is considered as 0.2J patches, where J is the total number of patches in image m (see details below).PACE models the patch embedding distribution using a mixture

F = {1,2,..,M}\{m}, and we have p(ys|z¯m,z¯′m,β) as p(ys = 1|z¯1:M,z¯′m,β) = exp(

𝜶

𝑯

βT(z¯m◦z¯′m))

, (3)

𝜷

𝜽𝒎

𝜽𝒎

f∈F exp(βT(z¯m◦z¯f))

𝒚𝒎 𝒛𝒎𝒋

𝒛𝒎𝒋

𝒚𝒔

where β ∈ RK is a learnable parameter, and ◦ denotes the element-wise product.

|𝝁𝒌<br><br>𝚺𝒌<br><br>K|
|---|

|ViT|
|---|

|ViT|
|---|

𝒆𝒎𝒋 𝒆𝒎𝒋

Given this generative process, learning the latent concept structures in ViT across the dataset involves learning the dataset-level parameters {µk,Σk}Kk=1 for the K concepts. Similarly, explaining ViT for each image is equivalent to inferring the distributions of the image-level and patch-level latent variables θm and {zmj}Jj=1, respectively.

Ja𝒎𝒋 J𝒂𝒎𝒋 M M

- Figure 2. Graphical model of our PACE. We sample each original patch embedding emj for J · amj times, and each perturbed patch embedding e′mj for J · a′mj times (ViT is shared for both).

of K Gaussians (K concepts), characterized by parameters µk ∈ Rd and Σk ∈ Rd×d (1 ≤ k ≤ K). For image m, PACE provides three levels of conceptual explanations: (1) K dataset-level variables {Ωk}Kk=1 = {µk,Σk}Kk=1 representing the mean and covariance of each concept k in the dataset, (2) an image-level variable θm ∈ RK for each image m, and (3) J patch-level variable ϕm ≜ {ϕmj}Jj=1 for each patch j in image m, where ϕmj ∈ RK.

###### 3.4. Inferring Conceptual Explanations using PACE

We begin by detailing the inference of image-level and patch-level explanations (i.e., θm and {zmj}Jj=1) given the dataset-level concept parameters {µk,Σk}Kk=1. We then discuss learning {µk,Σk}Kk=1 later in Sec. 3.5.

Inferring Patch-Level and Image-Level Concepts. Given the dataset-level concept parameters {µk,Σk}Kk=1, the patch embeddings em ≜ [emj]Jj=1, and the associated attention weights am ≜ [amj]Jj=1, as well as the predicted label ym produced by the ViT, for each image Im, PACE infers the posterior distribution of the image-level concept explanation θm, i.e., p(θm|em,am,{µk,Σk}Kk=1, ym), and the posterior distribution of the patch-level concept explanation zmj, i.e., p(zmj|em,am,{µk,Σk}Kk=1, ym). Fig. 1 describes the inference process of PACE.

Generative Process. Below we describe the generative process of PACE (Fig. 2 shows the corresponding PGM):

- • Draw the image-level concept distribution vector θm ∼ Dirichlet(α) for either the original image Im or the perturbed image I′m.
- • For each patch j in either Im or I′m (1 ≤ j ≤ J):

- – Draw the patch-level one-hot concept index zmj ∼ Categorical(θm).
- – Given the ViT attentions amj, for J · amj times,

* Draw patch j’s embedding, i.e., emj, from concept zmj’s Gaussian component emj ∼ N(µz

mj

,Σz

mj

).

- • Draw the predicted label ym ∼ GLM(z¯m,H).
- • For each pair of images Im and I′m, draw a binary variable ys ∼ p(ys|z¯m,z¯′m,β), which indicates whether Im and I′m come from the same image.

Variational Distributions. The aforementioned posterior distributions are intractable; hence, we employ variational inference (Jordan et al., 1998; Blei et al., 2003; Chang & Blei, 2009), using variational distributions q(θm|γm) and q(zmj|ϕmj) to approximate them. This results in the following joint variational distribution:

q(θm,{zmj}Jj=1|γm,{ϕmj}Jj=1)

J j=1

q(zmj|ϕmj), (4)

= q(θm|γm) ·

Here α ∈ RK is the parameter for the Dirichlet distribution Dirichlet(·), and we define

where the variational parameters γm ∈ RK and ϕmj ∈ RK are estimated by maximizing Eq. 5 (more details below).

J j=1

zmj. (1)

z¯m = 1/J

Objective Function. In line with the generative process outlined in Sec. 3.3, for each image m sampled from the dataset, the optimal variational distributions are found by maximizing the following evidence lower bound (ELBO):

GLM(·) denotes a categorical distribution from a generalized linear model (GLM), given by

N n=1

###### η

T nz¯m)

[ exp(

L(emj, γm, ϕm, ϕ′m, ym, ys; α, {µk, Σk}Kk=1, H, β)

n′z¯m)] y

, (2)

p( ym|H,z¯m) =

mn

n′ exp(ηT

= Le + Lf + Ls, (5) Le = L(emj, γm, ϕm; α, {µk, Σk}Kk=1), Lf = L( ym, ϕm; H), Ls = L(ys, ϕm, ϕ′m; β). (6)

where N is the number of classes, and H = [η1,...,ηN] are the learnable parameters (H ∈ RK×N). The function

p(ys|z¯m,z¯′m,β) defines a distribution over whether I′m is the perturbation of Im, where ys is a binary label. Let

This equation can be derived using log likelihood factorization of the variables in Fig. 2 (details provided in Appendix F.1). Below we describe each term’s intuition:

- 1. Le = L(emj,γm,ϕm;α,{µk,Σk}Kk=1) is the expected log likelihood of the joint distribution of patch embeddings emj and the variational parameters γm,ϕm. This term models the generation of patch embeddings emj in the ViT.
- 2. Lf = L( ym,ϕm;H) is the expected log likelihood

of the predicted label ym given explanation ϕm. This term reflects the faithfulness property in Definition 3.1.

- 3. Ls = L(ys,ϕm,ϕ′m;β) is the expected log likelihood of the binary label ys, which indicates whether image Im (with its inferred concepts ϕm) and I′m (with its in-

ferred concepts ϕ′m) comes from the same image. This term reflects the stability property in Definition 3.1.

Computing Le. We compute Le as:

Le = Eq[log p(emj, γm, ϕm|α, {µk, Σk}Kk=1)]

ϕmjkamj log N(emj|µk, Σk) + Eq[log p(γm, ϕm|α)]

=

k

ϕmjkamj{− 12 (emj − µk)T Σ−k 1(emj − µk) − log[(2π)d/2|Σk|1/2]} + Eq[log p(γm, ϕm|α)], (7)

=

k

where the expectation is over the joint variational distribution in Eq. 4. d is the dimension of the embedding emj.

Computing Lf. We compute Lf according to Eq. 2:

Lf =Eq[log p( ym|z¯m, H)]

= N

ymn(ηTnϕ¯m) − Eq[log( N

exp(ηTnz¯m))] ≈ N

n=1

n=1

ymn(ηTnϕ¯m) − log( N

exp(ηTnϕ¯m)), (8)

n=1

n=1

where N is the number of classes for classification, n the class index. We approximate z¯m by taking the average of ϕm: z¯m ≈ ϕ¯m = 1/J

J j=1 ϕmj. See Appendix F.1 for details on how to approximate z¯m. Eq. 8 implies that maximizing the log likelihood of the predicted class ym encourages a correlation between ym and the inferred patchlevel concepts ϕm, thereby enhancing PACE’s faithfulness.

Computing Ls. Inspired by contrastive learning (Chen et al., 2020), for each original image Im, we first generate its perturbed image I′m. Then, with their associated patchlevel concepts z¯1:M and z¯′m from Eq. 1, the stability term Ls is defined as the expected likelihood of the binary label ys in Eq. 3. Let F = {1,..,M}\{m}. We compute Ls as:

Ls = Eq[log p(ys = 1|z¯1:M, z¯′m, β)]

= βT (z¯m ◦ z¯′m) − Eq[log(

exp(βT (z¯m ◦ z¯f)))]

f∈F

≈ βT (ϕ¯m ◦ ϕ¯′m) − log(

exp(βT (ϕ¯m ◦ ϕ¯f))), (9)

f∈F

where ◦ is the element-wise product. Eq. 9 indicates that maximizing the log likelihood of ys encourages the inferred

Algorithm 1 Learning and Inference of PACE

Input: Initialized α,β,H,{γm}Mm=1, {ϕm}Mm=1, {Ωk}Kk=1, images {Im}Mm=1, perturbed images {I′m}Mm=1, predicted labels { ym}Mm=1, and number of epochs T.

for t = 1 : T do for m = 1 : M do

Update ϕm and γm using Eq. 10 and Eq. 11, respectively.

Update {Ωk}Kk=1 using Eq. 12 and Eq. 13. Output: {Ωk}Kk=1 as dataset-level, q(θm|γm) as imagelevel, and q(zm|ϕm) as patch-level concept explanations.

patch-level concepts from the original and perturbed patches (ϕm from Im and ϕ′m from I′m) to be similar, thus enhancing PACE’s stability against perturbations. See detailed derivations of Le, Lf, and Ls in Appendix F.1.

Update Rules for ϕmjk and γmk. Inferring the conceptual explanations using PACE involves learning the variational parameters, ϕmjk and γmk, in Eq. 4. This is done by iteratively updating ϕmjk and γmk to maximize Eq. 5.

Specifically, taking the derivative of the ELBO in Eq. 5 with respect to ϕmjk (see Appendix F.2.1 for details) and setting it to zero, we obtain the update rule for ϕmjk:

K k′=1

ϕmjk ∝ |Σk1|1/2 exp[Ψ(γmk) − Ψ(

γk′) −12amj(emj − µk)T Σ−k 1(emj − µk)

N n=1 exp(ηTnϕ¯m)ηn

N n=1

+J1 [(

ymnηn −

)k

N n=1 exp(ηTnϕ¯m)

βT (ϕ¯m◦ϕ¯f))(βT ϕ¯f) f∈F exp(βT (ϕ¯m◦ϕ¯f))

+βT ϕ¯′m − f∈F exp(

)], (10)

with the normalization constraint Kk=1 ϕmjk = 1. Here Ψ(·) is the digamma function (the first derivative of the

logarithm of the Gamma function Γ(z) = 0 ∞ tz−1e−tdt). Similarly, the update rule for γmk is:

γmk = αk +

J j=1

ϕmjkamj. (11)

In summary, the inference algorithm alternates between updating ϕmjk for all (m,j,k) tuples and updating γmk for all (m,k) tuples until convergence.

Image- and Patch-Level Explanations: θm and ϕmj. We then use γm = {γmk}Kk=1 with q(θm|γm) to obtain the image-level conceptual explanation θm and use ϕmj = {ϕmjk}Kk=1 as the patch-level explanation.

###### 3.5. Learning of PACE

Learning Dataset-Level Explanations: {µk,Σk}Kk=1. The inference algorithm in Sec. 3.4 assumes the availabil-

ity of the dataset-level concept parameters {µk,Σk}Kk=1.

To learn these parameters, one needs to iterate between (1) inferring image-level and patch-level variational parameters γm and ϕmj in Sec. 3.4, respectively, and (2) learning dataset-level concept parameters {µk,Σk}Kk=1. Alg. 1 summarizes the learning of PACE.

Update Rules for µk and Σk. Similar to Sec. 3.4, we expand the ELBO in Eq. 7 (see Appendix F.2.2 for details)

and set its derivative with respect to µk and Σk to zero, yielding the update rule for learning µk and Σk:

µk = m,j ϕmjkamjemj

m,j ϕmjkamj , (12) Σk = m,j ϕmjkamj(emj−

###### µ

k)(emj−µ

k)T

m,j ϕmjkamj . (13)

###### 3.6. Summary of Learning and Inference of PACE

In summary, PACE is a variational Bayesian framework that consists of (1) the learning stage to train on the training set, and (2) the inference stage to explain on the test set.

For example, given a finetuned ViT classifier on a dataset, PACE explains it by (1) training the global parameters, i.e., the dataset-level concept centers µk and covariance matrices Σk (where k = 1,...,K) as dataset-level explanations, on the training set (these are called global parameters because they are shared across all data points, e.g. images), and (2) inferring the local parameters, i.e., the image-level concepts (explanations) θm and patch-level concepts (explanations) ϕm, on the test set (these are called local parameters because each image has its own θm and ϕm).

Below, we discuss the learning and inference processes, respectively.

The Learning Stage. In Sec. 3.5, we describe the process of learning the global parameters µk and Σk (where k = 1,...,K). As described in Alg. 1, in each epoch t (t = 1,...,T):

- (1) PACE first infers the local parameters γm and ϕmj for each document m using Eq. 10 and Eq. 11;
- (2) PACE then updates the global parameters µk and Σk (where k = 1,...,K) for the entire dataset using Eq. 12 and Eq. 13.

The learning stage concludes at the Tth epoch. Note that the process above is iterative; it alternates between (1) updating the local parameters and (2) updating the global parameters.

The Inference Stage. In Sec. 3.4, we describe the process of inferring the local parameters θm and ϕmj after the PACE learns the global parameters in the learning stage. Specifically, given the global parameters µk and Σk (where k = 1,...,K),

- (1) PACE initializes local parameters γm and ϕmj;
- (2) Given the current γm and ϕmj, PACE updates the

local parameters γm using Eq. 10;

- (3) Given the current γm and ϕmj, PACE updates the local parameters ϕmj using Eq. 11;
- (4) PACE repeats (2) and (3) until γm and ϕmj converge;
- (5) PACE then infers the image-level concept θm using

the learned variational distribution q(θm|γm), which is a Dirichlet distribution. One can (roughly) think of

θm as a normalized version of γm. 3.7. Discussion and Theoretical Analysis Our PACE addresses all five desiderata in Definition 3.1:

- • Faithfulness is encouraged by maximizing the prediction ym’s likelihood, i.e., Lf in Eq. 8.
- • Stability against perturbations is enhanced by maximizing the binary label ys’s likelihood Ls in Eq. 9.
- • Sparsity is encouraged by the Dirichlet prior p(θm|α) that regularizes the inference of θm.
- • Multi-Level Structure is intrinsically supported by our multi-level generative process in Sec. 3.3.
- • Parsimony is ensured by the flexibility in choosing the number of concepts K in PACE (see Appendix A).

Theorem 3.2 below further demonstrates that PACE’s inferred image-level and patch-level explanations, θm and {ϕmj}Jj=1, align with the properties in Definition 3.1.

Theorem 3.2 (Mutual Information Maximization). The ELBO in Eq. 5 is upper-bounded by the sum of (1) mutual information between contextual embeddings em and multilevel explanation θm,{ϕmj}Jj=1 in Definition 3.1, (2) mutual information between the predicted label ym and patchlevel concept ϕm, and (3) mutual information between the patch-level original concept ϕm and perturbed concept ϕ′m. Formally, with approximate posteriors q(θm|γm) and q(zmj|ϕmj), we have

L(emj, γm, ϕm, ϕ′m, ym, ys)

≤ I(em;θm,ϕm) + I( ym;ϕm) + I(ϕm;ϕ′m) + C, (14)

where the C is a constant.

The proof of Theorem 3.2 is provided in Appendix E. Theorem 3.2 implies that maximizing the ELBO in Eq. 5 is equivalent to maximizing the sum of (1) mutual information between the contextual embeddings em and multi-level conceptual explanations defined in Definition 3.1, thereby ensuring the generated explanations are informative, (2) mutual information between the ViT prediction ym and patch-level concept ϕm, thereby enhancing PACE’s faithfulness, and (3) mutual information between the patch-level original ϕm and perturbed ϕ′m, thereby enhancing PACE’s stability against perturbations.

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

dicates stronger (linear) faithfulness. Note that one can fit more complex models (e.g., nonlinear models such as neural networks) to evaluate nonlinear faithfulness; for simplicity, we focus on linear faithfulness. Stability: For each input image Im in the test set, we generate an augmented version I′m (following Chen et al. (2020)), and compute the normalized difference

- Class 0
- Class 1

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Figure 3. Example images from our Color dataset.

between inferred concepts θm and θ′m, i.e., ∥θm−θ′

m∥

∥θm∥ . Lower values indicate stronger stability.

### 4. Experiments

In this section, we compare PACE with existing methods on one synthetic dataset and three real-world datasets.

- • Sparsity: We compute sparsity (with the threshold ϵ = 0.1/K) as K1 Kk=1 1(|θmk| < ϵ), where K is the number of concepts. For many concept-based explanation methods, including ours, the inferred activation typically normalizes to sum up to 1. If this is not the case, we normalize the explanation activation before calculating the sparsity score, ensuring a fair comparison.

- • Multi-Level Structure: As highlighted in Sec. 1, baseline models do not account for dataset-level and/or patch-level concepts, thereby possess No or Partial multi-level structure. In contrast, PACE is specifically designed to offer Full conceptual explanations at three levels: dataset, image, and patch. We will demonstrate in Sec. 4.5 that modeling embeddings’ distribution is instrumental in bridging three levels of ViT concepts.
- • Parsimony: For the conceptual explanation methods PACE and CRAFT (Fel et al., 2023b), we set the number of concepts K = 100, to facilitate a fair comparison. Note that other baseline models’ number of concept K is constrained to the hidden dimension of ViT embeddings, i.e., K = 768; they therefore fall short in parsimony.

###### 4.1. Datasets

We constructed Color as a synthetic dataset with clear definition of 4 concepts (red/yellow/green/blue). It contains two image classes: Class 0 (images with red/yellow colors) and Class 1 (green/blue colors), both against a black background (see example images in Fig. 3 and more details in Appendix A). We use three real-world datasets, Oxford 102 Flower (Flower) (Nilsback & Zisserman, 2008), Stanford Cars (Cars) (Krause et al., 2013), and CUB-200-2011 (CUB) (Wah et al., 2011) (for dataset statistics, see Appendix A). For real-world datasets, we follow the preprocessing steps from (Dosovitskiy et al., 2020) and use the same train-test split. For the Color dataset, we adopt an 8:2 train/test split among 2,000 images (1,000 per class).

- 4.2. Baselines We compare PACE with state-of-the-art methods, including:

- • SHAP (Lundberg & Lee, 2017) is an explanation method that assigns importance scores to input features using Shapley values.
- • LIME (Ribeiro et al., 2016) explains the model by approximates it with a local surrogate model via data perturbation.
- • SALIENCY (Simonyan et al., 2013) uses the saliency map of an image to explain the model prediction.
- • AGI (Pan et al., 2021) produces explanations via adversarial gradient integration.
- • CRAFT (Fel et al., 2023b) employs recursive low-rank matrix factorization to obtain concepts from intermediate layers.

For details and the three other desiderata, see Appendix A.

###### 4.4. Quantitative Results

Table 1 shows the quantitative results for our PACE and the baselines for the desiderata in Definition 3.1 across one synthetic dataset (Color) and three real-world datasets (Flower, Cars, and CUB). For a detailed discussion on Multi-level Structure and Parsimony, please refer to Appendix A. Below we discuss Faithfulness, Stability, and Sparsity in detail. Color. On the Color dataset, PACE distinctly surpasses other leading models, as detailed in Table 1. PACE achieves perfect faithfulness (1.00) and the best stability score (0.20), demonstrating consistency in its explanations. It leads in sparsity (0.97), delivering focused and clear explanations.

###### 4.3. Evaluation Metrics

With ViT-Base (Dosovitskiy et al., 2020) as the prediction model, we evaluate different methods against the five desiderata defined in Definition 3.1:

Flower, Cars, and CUB. Our evaluation on three real-world datasets – Flower, Cars, and CUB – reveals PACE’s significant advantages over established baselines across various desiderata. As shown in Table 1, PACE consistently registers the highest faithfulness scores (0.80 on Flower, 0.50 on Cars, and 0.56 on CUB), reflecting its superior precision in

• (Linear) Faithfulness: We fit a logistic regression (LR) model ym = LR(θm) to each dataset’s training set, with ym as the prediction of the ViT, and test the model’s accuracy on the test set. Higher accuracy in-

##### Input Image-Level Dataset-Level Patch-Level

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

- Cpt 1
- Cpt 2
- Cpt 3
- Cpt 4

- Concept 1
- Concept 2
- Concept 3
- Concept 4

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

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

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 169]

- Cpt 1
- Cpt 2
- Cpt 3
- Cpt 4

- Concept 1
- Concept 2
- Concept 3
- Concept 4

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

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

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

[Figure 225]

[Figure 226]

[Figure 227]

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

[Figure 251]

[Figure 252]

[Figure 253]

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

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

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

[Figure 291]

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

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

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

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

- Figure 4. PACE’s three-level conceptual explanations on the Color dataset. Dataset-Level: PACE’s top 4 dataset-level concepts; for

each concept k, we plot the top 3 patches with emj closest to µk. Image-Level: Given an input image m, we show PACE’s generated image-level explanation θm for the 4 selected concepts. For example, for the top-left input image, PACE’s generated image-level explanation θm indicates a strong association with Concept 3 (green) and Concept 4 (blue). Patch-Level: Given an input image m, PACE’s ϕmj identifies the top concepts for the selected patches. For example, for the top-left input image, the blue patch is associated with Concept 4 (containing similar blue patches across the dataset) while the green patch is linked to Concept 3 (containing similar green patches across the dataset).

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

[Figure 351]

[Figure 352]

[Figure 353]

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

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

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

- Table 1. Results for the five desiderata in Definition 3.1 for different methods on four datasets. ‘All’ denotes results on all four datasets. We mark the best results with bold face and the second best results with underline. ↑ / ↓ indicates higher/lower is better, respectively.

Desiderata (Linear) Faithfulness ↑ Stability ↓ Sparsity ↑ Multi-Level Parsimony ↓ Datasets Color Flower Cars CUB Color Flower Cars CUB Color Flower Cars CUB All All SHAP 0.47 0.52 0.44 0.34 4.39 0.92 1.21 1.55 0.54 0.12 0.13 0.11 No 768 LIME 0.54 0.06 0.02 0.03 1.50 1.54 1.45 1.80 0.59 0.54 0.52 0.55 No 768 Saliency 1.00 0.57 0.50 0.49 0.35 0.47 0.43 0.48 0.01 0.00 0.00 0.00 No 768 AGI 1.00 0.54 0.34 0.49 1.40 1.21 1.83 2.53 0.01 0.07 0.04 0.03 No 768 CRAFT 0.59 0.01 0.01 0.00 4.49 0.52 1.76 0.64 0.26 0.29 0.11 0.25 Partial 100 PACE 1.00 0.80 0.50 0.56 0.20 0.12 0.05 0.05 0.97 0.48 0.49 0.63 Full 100

- Table 2. Average results across all four datasets in terms of faithfulness, stability, and sparsity for different methods. The best results are marked with bold face. ↑ / ↓ indicates higher/lower is better, respectively.

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

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

Average Performance Across Datasets. Table 2 shows the average performance across all four datasets in terms of faithfulness, stability, and sparsity. PACE consistently leads in faithfulness (0.72), stability (0.11), and sparsity (0.64). Compared to other models, its improvements are substantial, enhancing faithfulness by 0.08 ∼ 0.57, stability by 0.32 ∼ 1.91, and sparsity by 0.09 ∼ 0.64, verifying PACE’s effectiveness in terms of trustworthy explanations.

Desiderata SHAP LIME Saliency AGI CRAFT PACE

Faithfulness ↑ 0.44 0.16 0.64 0.59 0.15 0.72 Stability ↓ 2.02 1.57 0.43 1.74 1.85 0.11 Sparsity ↑ 0.23 0.55 0.00 0.04 0.23 0.64

mirroring the model’s decision-making process; note that both Cars and CUB contain a large number of classes (196 and 200); therefore, a linear faithfulness score of 0.50 is already very high. As mentioned in Sec. 4.3, one can always fit more complex models (e.g., nonlinear models such as a two-layer neural networks) to evaluate nonlinear faithfulness; for simplicity, we focus on linear faithfulness in this paper. Its stability scores (lower is better) on these three datasets are 0.12, 0.05, and 0.05, respectively, illustrating its resilience to input perturbations. In terms of sparsity, PACE is highly competitive, achieving second-best results and thus providing succinct, pertinent explanations.

###### 4.5. Qualitative Analysis

Color. Fig. 4 illustrates PACE’s three-level explanations on the Color dataset, where the ViT correctly predicts the top and bottom input images as Class 1 (green/blue colors) and Class 0 (images with red/yellow colors), respectively.

• Dataset-Level Explanation: The dataset-level column shows PACE’s top 4 dataset-level concepts (for each concept, we plot the top 3 patches with emj closest to µk); they are consistent with the 4 primary colors in the dataset (see Sec. 4.1), verifying the PACE’s effec-

[Figure 551]

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

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

[Figure 1008]

Probabilistic Conceptual Explainers: Trustworthy Conceptual Explanations for Vision Foundation Models

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

[Figure 1095]

[Figure 1096]

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

[Figure 1104]

[Figure 1105]

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

[Figure 1114]

[Figure 1115]

[Figure 1116]

[Figure 1117]

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

[Figure 1122]

[Figure 1123]

[Figure 1124]

[Figure 1125]

[Figure 1126]

[Figure 1127]

[Figure 1131]

[Figure 1132]

[Figure 1133]

[Figure 1134]

[Figure 1135]

[Figure 1136]

[Figure 1137]

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

[Figure 1147]

[Figure 1148]

[Figure 1149]

[Figure 1151]

[Figure 1152]

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

[Figure 1159]

[Figure 1160]

[Figure 1161]

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

[Figure 1166]

[Figure 1167]

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

[Figure 1172]

[Figure 1173]

[Figure 1174]

[Figure 1175]

[Figure 1176]

[Figure 1177]

[Figure 1178]

[Figure 1179]

[Figure 1180]

[Figure 1181]

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

[Figure 1186]

[Figure 1187]

[Figure 1188]

[Figure 1189]

[Figure 1190]

[Figure 1191]

[Figure 1192]

[Figure 1193]

[Figure 1194]

[Figure 1195]

[Figure 1196]

[Figure 1197]

[Figure 1198]

[Figure 1199]

[Figure 1200]

[Figure 1201]

[Figure 1202]

[Figure 1203]

[Figure 1204]

[Figure 1205]

[Figure 1206]

[Figure 1207]

[Figure 1208]

[Figure 1209]

[Figure 1210]

[Figure 1211]

[Figure 1212]

[Figure 1213]

[Figure 1214]

[Figure 1215]

[Figure 1216]

[Figure 1217]

[Figure 1218]

[Figure 1219]

[Figure 1220]

[Figure 1221]

[Figure 1222]

[Figure 1223]

[Figure 1224]

[Figure 1225]

[Figure 1226]

[Figure 1227]

[Figure 1228]

###### Input Image-Level Dataset-Level Patch-Level

[Figure 1231]

[Figure 1232]

[Figure 1233]

[Figure 1234]

[Figure 1235]

[Figure 1236]

[Figure 1237]

[Figure 1238]

[Figure 1239]

[Figure 1240]

[Figure 1241]

[Figure 1242]

[Figure 1243]

[Figure 1244]

[Figure 1245]

[Figure 1246]

[Figure 1251]

[Figure 1252]

[Figure 1253]

[Figure 1254]

[Figure 1255]

[Figure 1256]

[Figure 1257]

[Figure 1258]

[Figure 1259]

[Figure 1260]

[Figure 1261]

[Figure 1262]

[Figure 1263]

[Figure 1264]

[Figure 1265]

[Figure 1266]

[Figure 1267]

[Figure 1268]

[Figure 1269]

[Figure 1270]

[Figure 1271]

[Figure 1272]

[Figure 1273]

[Figure 1274]

[Figure 1275]

[Figure 1276]

[Figure 1277]

[Figure 1278]

[Figure 1279]

[Figure 1280]

[Figure 1281]

[Figure 1282]

[Figure 1283]

[Figure 1284]

[Figure 1285]

[Figure 1286]

[Figure 1287]

[Figure 1288]

[Figure 1289]

[Figure 1290]

[Figure 1291]

[Figure 1292]

[Figure 1293]

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

[Figure 1298]

[Figure 1299]

[Figure 1300]

[Figure 1301]

[Figure 1302]

[Figure 1303]

[Figure 1304]

[Figure 1305]

[Figure 1306]

[Figure 1307]

[Figure 1308]

[Figure 1309]

[Figure 1310]

[Figure 1311]

[Figure 1312]

[Figure 1313]

[Figure 1314]

[Figure 1315]

[Figure 1316]

[Figure 1317]

[Figure 1318]

[Figure 1319]

[Figure 1320]

[Figure 1321]

[Figure 1322]

[Figure 1323]

[Figure 1324]

[Figure 1325]

[Figure 1326]

[Figure 1327]

[Figure 1328]

[Figure 1329]

[Figure 1330]

[Figure 1331]

[Figure 1332]

[Figure 1335]

[Figure 1336]

[Figure 1337]

[Figure 1338]

[Figure 1339]

[Figure 1340]

[Figure 1341]

[Figure 1342]

[Figure 1343]

[Figure 1344]

[Figure 1345]

[Figure 1346]

[Figure 1347]

[Figure 1348]

[Figure 1349]

[Figure 1350]

[Figure 1351]

[Figure 1352]

[Figure 1355]

[Figure 1356]

[Figure 1357]

[Figure 1358]

[Figure 1359]

[Figure 1360]

[Figure 1361]

[Figure 1362]

[Figure 1363]

[Figure 1364]

[Figure 1365]

[Figure 1366]

[Figure 1367]

[Figure 1368]

[Figure 1369]

[Figure 1370]

[Figure 1371]

[Figure 1372]

[Figure 1373]

[Figure 1374]

[Figure 1375]

[Figure 1376]

[Figure 1377]

[Figure 1378]

[Figure 1379]

[Figure 1380]

[Figure 1381]

[Figure 1382]

[Figure 1383]

[Figure 1384]

[Figure 1385]

[Figure 1386]

[Figure 1387]

[Figure 1395]

[Figure 1396]

[Figure 1397]

[Figure 1398]

[Figure 1399]

[Figure 1400]

[Figure 1401]

[Figure 1402]

[Figure 1403]

[Figure 1404]

[Figure 1405]

[Figure 1406]

[Figure 1407]

[Figure 1408]

[Figure 1409]

[Figure 1410]

[Figure 1411]

[Figure 1412]

[Figure 1413]

[Figure 1414]

[Figure 1415]

[Figure 1416]

[Figure 1417]

[Figure 1418]

[Figure 1419]

[Figure 1420]

[Figure 1421]

[Figure 1422]

[Figure 1423]

[Figure 1424]

[Figure 1425]

[Figure 1426]

[Figure 1427]

[Figure 1428]

[Figure 1429]

[Figure 1430]

[Figure 1431]

[Figure 1435]

[Figure 1436]

[Figure 1437]

[Figure 1438]

[Figure 1439]

[Figure 1440]

[Figure 1441]

[Figure 1442]

[Figure 1443]

[Figure 1444]

[Figure 1445]

[Figure 1446]

[Figure 1447]

[Figure 1448]

[Figure 1449]

[Figure 1450]

[Figure 1451]

[Figure 1452]

[Figure 1453]

[Figure 1454]

[Figure 1455]

[Figure 1456]

[Figure 1457]

[Figure 1458]

[Figure 1459]

[Figure 1460]

[Figure 1461]

[Figure 1462]

[Figure 1463]

[Figure 1464]

[Figure 1465]

[Figure 1466]

[Figure 1467]

[Figure 1468]

[Figure 1469]

[Figure 1470]

[Figure 1471]

[Figure 1472]

[Figure 1473]

[Figure 1475]

[Figure 1476]

[Figure 1477]

[Figure 1478]

[Figure 1479]

[Figure 1480]

[Figure 1481]

[Figure 1482]

[Figure 1483]

[Figure 1484]

[Figure 1485]

[Figure 1486]

[Figure 1487]

[Figure 1488]

[Figure 1489]

[Figure 1490]

[Figure 1491]

[Figure 1492]

[Figure 1493]

[Figure 1494]

[Figure 1495]

[Figure 1496]

[Figure 1497]

[Figure 1498]

[Figure 1499]

[Figure 1500]

[Figure 1501]

[Figure 1502]

[Figure 1503]

[Figure 1504]

[Figure 1505]

[Figure 1506]

[Figure 1507]

[Figure 1508]

[Figure 1509]

[Figure 1510]

[Figure 1511]

[Figure 1512]

[Figure 1513]

[Figure 1514]

[Figure 1515]

[Figure 1516]

[Figure 1517]

[Figure 1518]

[Figure 1519]

[Figure 1520]

[Figure 1521]

[Figure 1522]

[Figure 1523]

[Figure 1524]

[Figure 1525]

[Figure 1526]

[Figure 1527]

[Figure 1528]

[Figure 1529]

[Figure 1530]

[Figure 1531]

[Figure 1532]

[Figure 1535]

[Figure 1536]

[Figure 1537]

[Figure 1538]

[Figure 1539]

[Figure 1540]

[Figure 1541]

[Figure 1542]

[Figure 1543]

[Figure 1544]

[Figure 1545]

[Figure 1546]

[Figure 1547]

[Figure 1548]

[Figure 1549]

[Figure 1550]

- Concept 1
- Concept 2
- Concept 3
- Concept 4

- Cpt 1
- Cpt 2
- Cpt 3
- Cpt 4

- Concept 1
- Concept 2
- Concept 3
- Concept 4

[Figure 1555]

[Figure 1556]

[Figure 1557]

[Figure 1558]

[Figure 1559]

[Figure 1560]

[Figure 1561]

[Figure 1562]

[Figure 1563]

[Figure 1564]

[Figure 1565]

[Figure 1566]

[Figure 1567]

[Figure 1568]

[Figure 1569]

[Figure 1570]

[Figure 1571]

[Figure 1572]

[Figure 1573]

[Figure 1574]

[Figure 1575]

[Figure 1576]

[Figure 1577]

[Figure 1578]

[Figure 1579]

[Figure 1580]

[Figure 1581]

[Figure 1582]

[Figure 1583]

[Figure 1584]

[Figure 1585]

[Figure 1586]

[Figure 1587]

[Figure 1588]

[Figure 1589]

[Figure 1590]

[Figure 1591]

[Figure 1592]

[Figure 1596]

[Figure 1597]

[Figure 1598]

[Figure 1599]

[Figure 1600]

[Figure 1601]

[Figure 1602]

[Figure 1603]

[Figure 1604]

[Figure 1605]

[Figure 1606]

[Figure 1607]

[Figure 1608]

[Figure 1609]

[Figure 1610]

[Figure 1611]

[Figure 1612]

[Figure 1613]

[Figure 1614]

[Figure 1615]

[Figure 1616]

[Figure 1617]

[Figure 1618]

[Figure 1619]

[Figure 1620]

[Figure 1621]

[Figure 1622]

[Figure 1623]

[Figure 1624]

[Figure 1625]

[Figure 1626]

[Figure 1627]

[Figure 1628]

[Figure 1629]

[Figure 1630]

[Figure 1631]

[Figure 1632]

[Figure 1633]

[Figure 1634]

[Figure 1635]

[Figure 1636]

[Figure 1637]

[Figure 1638]

[Figure 1639]

[Figure 1640]

[Figure 1641]

[Figure 1642]

[Figure 1643]

[Figure 1644]

[Figure 1645]

[Figure 1646]

[Figure 1647]

[Figure 1648]

[Figure 1649]

[Figure 1650]

[Figure 1651]

[Figure 1652]

[Figure 1653]

[Figure 1656]

[Figure 1657]

[Figure 1658]

[Figure 1659]

[Figure 1660]

[Figure 1661]

[Figure 1662]

[Figure 1663]

[Figure 1664]

[Figure 1665]

[Figure 1666]

[Figure 1667]

[Figure 1668]

[Figure 1669]

[Figure 1670]

[Figure 1671]

[Figure 1676]

[Figure 1677]

[Figure 1678]

[Figure 1679]

[Figure 1680]

[Figure 1681]

[Figure 1682]

[Figure 1683]

[Figure 1684]

[Figure 1685]

[Figure 1686]

[Figure 1687]

[Figure 1688]

[Figure 1689]

[Figure 1690]

[Figure 1691]

[Figure 1692]

[Figure 1693]

[Figure 1694]

[Figure 1695]

[Figure 1696]

[Figure 1697]

[Figure 1698]

[Figure 1699]

[Figure 1700]

[Figure 1701]

[Figure 1702]

[Figure 1703]

[Figure 1704]

[Figure 1705]

[Figure 1706]

[Figure 1707]

[Figure 1708]

[Figure 1709]

[Figure 1710]

[Figure 1711]

[Figure 1712]

[Figure 1713]

[Figure 1714]

[Figure 1716]

[Figure 1717]

[Figure 1718]

[Figure 1719]

[Figure 1720]

[Figure 1721]

[Figure 1722]

[Figure 1723]

[Figure 1724]

[Figure 1725]

[Figure 1726]

[Figure 1727]

[Figure 1728]

[Figure 1729]

[Figure 1730]

[Figure 1731]

[Figure 1732]

[Figure 1736]

[Figure 1737]

[Figure 1738]

[Figure 1739]

[Figure 1740]

[Figure 1741]

[Figure 1742]

[Figure 1743]

[Figure 1744]

[Figure 1745]

[Figure 1746]

[Figure 1747]

[Figure 1748]

[Figure 1749]

[Figure 1750]

[Figure 1751]

[Figure 1752]

[Figure 1753]

[Figure 1756]

[Figure 1757]

[Figure 1758]

[Figure 1759]

[Figure 1760]

[Figure 1761]

[Figure 1762]

[Figure 1763]

[Figure 1764]

[Figure 1765]

[Figure 1766]

[Figure 1767]

[Figure 1768]

[Figure 1769]

[Figure 1770]

[Figure 1771]

[Figure 1772]

[Figure 1773]

[Figure 1774]

[Figure 1775]

[Figure 1776]

[Figure 1777]

[Figure 1778]

[Figure 1779]

[Figure 1780]

[Figure 1781]

[Figure 1782]

[Figure 1783]

[Figure 1784]

[Figure 1785]

[Figure 1786]

[Figure 1787]

[Figure 1788]

[Figure 1796]

[Figure 1797]

[Figure 1798]

[Figure 1799]

[Figure 1800]

[Figure 1801]

[Figure 1802]

[Figure 1803]

[Figure 1804]

[Figure 1805]

[Figure 1806]

[Figure 1807]

[Figure 1808]

[Figure 1809]

[Figure 1810]

[Figure 1811]

[Figure 1812]

[Figure 1813]

[Figure 1814]

[Figure 1815]

[Figure 1816]

[Figure 1817]

[Figure 1818]

[Figure 1819]

[Figure 1820]

[Figure 1821]

[Figure 1822]

[Figure 1823]

[Figure 1824]

[Figure 1825]

[Figure 1826]

[Figure 1827]

[Figure 1828]

[Figure 1829]

[Figure 1830]

[Figure 1831]

[Figure 1832]

[Figure 1833]

[Figure 1836]

[Figure 1837]

[Figure 1838]

[Figure 1839]

[Figure 1840]

[Figure 1841]

[Figure 1842]

[Figure 1843]

[Figure 1844]

[Figure 1845]

[Figure 1846]

[Figure 1847]

[Figure 1848]

[Figure 1849]

[Figure 1850]

[Figure 1851]

[Figure 1852]

[Figure 1853]

[Figure 1854]

[Figure 1855]

[Figure 1856]

[Figure 1857]

[Figure 1858]

[Figure 1859]

[Figure 1860]

[Figure 1861]

[Figure 1862]

[Figure 1863]

[Figure 1864]

[Figure 1865]

[Figure 1866]

[Figure 1867]

[Figure 1868]

[Figure 1869]

[Figure 1870]

[Figure 1871]

[Figure 1872]

[Figure 1873]

[Figure 1874]

[Figure 1875]

[Figure 1876]

[Figure 1877]

[Figure 1878]

[Figure 1879]

[Figure 1880]

[Figure 1881]

[Figure 1882]

[Figure 1883]

[Figure 1884]

[Figure 1885]

[Figure 1886]

[Figure 1887]

[Figure 1888]

[Figure 1889]

[Figure 1890]

[Figure 1891]

[Figure 1892]

[Figure 1893]

[Figure 1896]

[Figure 1897]

[Figure 1898]

[Figure 1899]

[Figure 1900]

[Figure 1901]

[Figure 1902]

[Figure 1903]

[Figure 1904]

[Figure 1905]

[Figure 1906]

[Figure 1907]

[Figure 1908]

[Figure 1909]

[Figure 1910]

[Figure 1911]

[Figure 1916]

[Figure 1917]

[Figure 1918]

[Figure 1919]

[Figure 1920]

[Figure 1921]

[Figure 1922]

[Figure 1923]

[Figure 1924]

[Figure 1925]

[Figure 1926]

[Figure 1927]

[Figure 1928]

[Figure 1929]

[Figure 1930]

[Figure 1931]

[Figure 1932]

[Figure 1933]

[Figure 1934]

[Figure 1935]

[Figure 1936]

[Figure 1937]

[Figure 1938]

[Figure 1939]

[Figure 1940]

[Figure 1941]

[Figure 1942]

[Figure 1943]

[Figure 1944]

[Figure 1945]

[Figure 1946]

[Figure 1947]

[Figure 1948]

[Figure 1949]

[Figure 1950]

[Figure 1951]

[Figure 1952]

[Figure 1956]

[Figure 1957]

[Figure 1958]

[Figure 1959]

[Figure 1960]

[Figure 1961]

[Figure 1962]

[Figure 1963]

[Figure 1964]

[Figure 1965]

[Figure 1966]

[Figure 1967]

[Figure 1968]

[Figure 1969]

[Figure 1970]

[Figure 1971]

[Figure 1972]

[Figure 1973]

[Figure 1974]

[Figure 1975]

[Figure 1976]

[Figure 1977]

[Figure 1978]

[Figure 1979]

[Figure 1980]

[Figure 1981]

[Figure 1982]

[Figure 1983]

[Figure 1984]

[Figure 1985]

[Figure 1986]

[Figure 1987]

[Figure 1988]

[Figure 1989]

[Figure 1990]

[Figure 1991]

[Figure 1992]

[Figure 1993]

[Figure 1994]

[Figure 1995]

[Figure 1996]

[Figure 1997]

[Figure 1998]

[Figure 1999]

[Figure 2000]

[Figure 2001]

[Figure 2002]

[Figure 2003]

[Figure 2004]

[Figure 2005]

[Figure 2006]

[Figure 2007]

[Figure 2008]

[Figure 2009]

[Figure 2010]

[Figure 2011]

[Figure 2012]

[Figure 2013]

[Figure 2016]

[Figure 2017]

[Figure 2018]

[Figure 2019]

[Figure 2020]

[Figure 2021]

[Figure 2022]

[Figure 2023]

[Figure 2024]

[Figure 2025]

[Figure 2026]

[Figure 2027]

[Figure 2028]

[Figure 2029]

[Figure 2030]

[Figure 2031]

[Figure 2036]

[Figure 2037]

[Figure 2038]

[Figure 2039]

[Figure 2040]

[Figure 2041]

[Figure 2042]

[Figure 2043]

[Figure 2044]

[Figure 2045]

[Figure 2046]

[Figure 2047]

[Figure 2048]

[Figure 2049]

[Figure 2050]

[Figure 2051]

[Figure 2052]

[Figure 2053]

[Figure 2054]

[Figure 2055]

[Figure 2056]

[Figure 2057]

[Figure 2058]

[Figure 2059]

[Figure 2060]

[Figure 2061]

[Figure 2062]

[Figure 2063]

[Figure 2064]

[Figure 2065]

[Figure 2066]

[Figure 2067]

[Figure 2068]

[Figure 2069]

[Figure 2070]

[Figure 2071]

[Figure 2072]

[Figure 2073]

[Figure 2074]

[Figure 2076]

[Figure 2077]

[Figure 2078]

[Figure 2079]

[Figure 2080]

[Figure 2081]

[Figure 2082]

[Figure 2083]

[Figure 2084]

[Figure 2085]

[Figure 2086]

[Figure 2087]

[Figure 2088]

[Figure 2089]

[Figure 2090]

[Figure 2091]

[Figure 2092]

[Figure 2096]

[Figure 2097]

[Figure 2098]

[Figure 2099]

[Figure 2100]

[Figure 2101]

[Figure 2102]

[Figure 2103]

[Figure 2104]

[Figure 2105]

[Figure 2106]

[Figure 2107]

[Figure 2108]

[Figure 2109]

[Figure 2110]

[Figure 2111]

[Figure 2112]

[Figure 2113]

[Figure 2114]

[Figure 2115]

[Figure 2116]

[Figure 2117]

[Figure 2118]

[Figure 2119]

[Figure 2120]

[Figure 2121]

[Figure 2122]

[Figure 2123]

[Figure 2124]

[Figure 2125]

[Figure 2126]

[Figure 2127]

[Figure 2128]

[Figure 2129]

[Figure 2130]

[Figure 2131]

[Figure 2132]

[Figure 2133]

[Figure 2134]

[Figure 2135]

[Figure 2136]

[Figure 2137]

[Figure 2138]

[Figure 2139]

[Figure 2140]

[Figure 2141]

[Figure 2142]

[Figure 2143]

[Figure 2144]

[Figure 2145]

[Figure 2146]

[Figure 2147]

[Figure 2148]

[Figure 2149]

[Figure 2150]

[Figure 2151]

[Figure 2152]

[Figure 2153]

[Figure 2154]

[Figure 2155]

[Figure 2156]

[Figure 2157]

[Figure 2158]

[Figure 2159]

[Figure 2160]

[Figure 2161]

[Figure 2162]

[Figure 2163]

[Figure 2164]

[Figure 2165]

[Figure 2166]

[Figure 2167]

[Figure 2168]

[Figure 2169]

[Figure 2170]

[Figure 2171]

[Figure 2172]

[Figure 2173]

[Figure 2176]

[Figure 2177]

[Figure 2178]

[Figure 2179]

[Figure 2180]

[Figure 2181]

[Figure 2182]

[Figure 2183]

[Figure 2184]

[Figure 2185]

[Figure 2186]

[Figure 2187]

[Figure 2188]

[Figure 2189]

[Figure 2190]

[Figure 2191]

[Figure 2192]

[Figure 2193]

[Figure 2196]

[Figure 2197]

[Figure 2198]

[Figure 2199]

[Figure 2200]

[Figure 2201]

[Figure 2202]

[Figure 2203]

[Figure 2204]

[Figure 2205]

[Figure 2206]

[Figure 2207]

[Figure 2208]

[Figure 2209]

[Figure 2210]

[Figure 2211]

[Figure 2212]

[Figure 2213]

[Figure 2214]

[Figure 2215]

[Figure 2216]

[Figure 2217]

[Figure 2218]

[Figure 2219]

[Figure 2220]

[Figure 2221]

[Figure 2222]

[Figure 2223]

[Figure 2224]

[Figure 2225]

[Figure 2226]

[Figure 2227]

[Figure 2228]

[Figure 2236]

[Figure 2237]

[Figure 2238]

[Figure 2239]

[Figure 2240]

[Figure 2241]

[Figure 2242]

[Figure 2243]

[Figure 2244]

[Figure 2245]

[Figure 2246]

[Figure 2247]

[Figure 2248]

[Figure 2249]

[Figure 2250]

[Figure 2251]

[Figure 2256]

[Figure 2257]

[Figure 2258]

[Figure 2259]

[Figure 2260]

[Figure 2261]

[Figure 2262]

[Figure 2263]

[Figure 2264]

[Figure 2265]

[Figure 2266]

[Figure 2267]

[Figure 2268]

[Figure 2269]

[Figure 2270]

[Figure 2271]

[Figure 2272]

[Figure 2273]

[Figure 2276]

[Figure 2277]

[Figure 2278]

[Figure 2279]

[Figure 2280]

[Figure 2281]

[Figure 2282]

[Figure 2283]

[Figure 2284]

[Figure 2285]

[Figure 2286]

[Figure 2287]

[Figure 2288]

[Figure 2289]

[Figure 2290]

[Figure 2291]

[Figure 2292]

[Figure 2293]

[Figure 2294]

[Figure 2295]

[Figure 2296]

[Figure 2297]

[Figure 2298]

[Figure 2299]

[Figure 2300]

[Figure 2301]

[Figure 2302]

[Figure 2303]

[Figure 2304]

[Figure 2305]

[Figure 2306]

[Figure 2307]

[Figure 2308]

[Figure 2309]

[Figure 2310]

[Figure 2311]

[Figure 2312]

[Figure 2313]

[Figure 2317]

[Figure 2318]

[Figure 2319]

[Figure 2320]

[Figure 2321]

[Figure 2322]

[Figure 2323]

[Figure 2324]

[Figure 2325]

[Figure 2326]

[Figure 2327]

[Figure 2328]

[Figure 2329]

[Figure 2330]

[Figure 2331]

[Figure 2332]

[Figure 2333]

[Figure 2334]

[Figure 2335]

[Figure 2336]

[Figure 2337]

[Figure 2338]

[Figure 2339]

[Figure 2340]

[Figure 2341]

[Figure 2342]

[Figure 2343]

[Figure 2344]

[Figure 2345]

[Figure 2346]

[Figure 2347]

[Figure 2348]

[Figure 2349]

[Figure 2350]

[Figure 2351]

[Figure 2352]

[Figure 2353]

[Figure 2354]

[Figure 2355]

[Figure 2356]

[Figure 2357]

[Figure 2358]

[Figure 2359]

[Figure 2360]

[Figure 2361]

[Figure 2362]

[Figure 2363]

[Figure 2364]

[Figure 2365]

[Figure 2366]

[Figure 2367]

[Figure 2368]

[Figure 2369]

[Figure 2370]

[Figure 2371]

[Figure 2372]

[Figure 2373]

[Figure 2374]

[Figure 2377]

[Figure 2378]

[Figure 2379]

[Figure 2380]

[Figure 2381]

[Figure 2382]

[Figure 2383]

[Figure 2384]

[Figure 2385]

[Figure 2386]

[Figure 2387]

[Figure 2388]

[Figure 2389]

[Figure 2390]

[Figure 2391]

[Figure 2392]

[Figure 2397]

[Figure 2398]

[Figure 2399]

[Figure 2400]

[Figure 2401]

[Figure 2402]

[Figure 2403]

[Figure 2404]

[Figure 2405]

[Figure 2406]

[Figure 2407]

[Figure 2408]

[Figure 2409]

[Figure 2410]

[Figure 2411]

[Figure 2412]

[Figure 2413]

[Figure 2414]

[Figure 2415]

[Figure 2416]

[Figure 2417]

[Figure 2418]

[Figure 2419]

[Figure 2420]

[Figure 2421]

[Figure 2422]

[Figure 2423]

[Figure 2424]

[Figure 2425]

[Figure 2426]

[Figure 2427]

[Figure 2428]

[Figure 2429]

[Figure 2430]

[Figure 2431]

[Figure 2432]

[Figure 2433]

[Figure 2434]

[Figure 2435]

[Figure 2437]

[Figure 2438]

[Figure 2439]

[Figure 2440]

[Figure 2441]

[Figure 2442]

[Figure 2443]

[Figure 2444]

[Figure 2445]

[Figure 2446]

[Figure 2447]

[Figure 2448]

[Figure 2449]

[Figure 2450]

[Figure 2451]

[Figure 2452]

[Figure 2457]

[Figure 2458]

[Figure 2459]

[Figure 2460]

[Figure 2461]

[Figure 2462]

[Figure 2463]

[Figure 2464]

[Figure 2465]

[Figure 2466]

[Figure 2467]

[Figure 2468]

[Figure 2469]

[Figure 2470]

[Figure 2471]

[Figure 2472]

[Figure 2473]

[Figure 2477]

[Figure 2478]

[Figure 2479]

[Figure 2480]

[Figure 2481]

[Figure 2482]

[Figure 2483]

[Figure 2484]

[Figure 2485]

[Figure 2486]

[Figure 2487]

[Figure 2488]

[Figure 2489]

[Figure 2490]

[Figure 2491]

[Figure 2492]

[Figure 2493]

[Figure 2494]

[Figure 2497]

[Figure 2498]

[Figure 2499]

[Figure 2500]

[Figure 2501]

[Figure 2502]

[Figure 2503]

[Figure 2504]

[Figure 2505]

[Figure 2506]

[Figure 2507]

[Figure 2508]

[Figure 2509]

[Figure 2510]

[Figure 2511]

[Figure 2512]

[Figure 2513]

[Figure 2514]

[Figure 2517]

[Figure 2518]

[Figure 2519]

[Figure 2520]

[Figure 2521]

[Figure 2522]

[Figure 2523]

[Figure 2524]

[Figure 2525]

[Figure 2526]

[Figure 2527]

[Figure 2528]

[Figure 2529]

[Figure 2530]

[Figure 2531]

[Figure 2532]

[Figure 2533]

[Figure 2534]

[Figure 2535]

[Figure 2536]

[Figure 2537]

[Figure 2538]

[Figure 2539]

[Figure 2540]

[Figure 2541]

[Figure 2542]

[Figure 2543]

[Figure 2544]

[Figure 2545]

[Figure 2546]

[Figure 2547]

[Figure 2548]

[Figure 2549]

[Figure 2557]

[Figure 2558]

[Figure 2559]

[Figure 2560]

[Figure 2561]

[Figure 2562]

[Figure 2563]

[Figure 2564]

[Figure 2565]

[Figure 2566]

[Figure 2567]

[Figure 2568]

[Figure 2569]

[Figure 2570]

[Figure 2571]

[Figure 2572]

[Figure 2573]

[Figure 2574]

[Figure 2577]

[Figure 2578]

[Figure 2579]

[Figure 2580]

[Figure 2581]

[Figure 2582]

[Figure 2583]

[Figure 2584]

[Figure 2585]

[Figure 2586]

[Figure 2587]

[Figure 2588]

[Figure 2589]

[Figure 2590]

[Figure 2591]

[Figure 2592]

[Figure 2593]

[Figure 2594]

[Figure 2595]

[Figure 2596]

[Figure 2597]

[Figure 2598]

[Figure 2599]

[Figure 2600]

[Figure 2601]

[Figure 2602]

[Figure 2603]

[Figure 2604]

[Figure 2605]

[Figure 2606]

[Figure 2607]

[Figure 2608]

[Figure 2609]

[Figure 2610]

[Figure 2611]

[Figure 2612]

[Figure 2613]

[Figure 2617]

[Figure 2618]

[Figure 2619]

[Figure 2620]

[Figure 2621]

[Figure 2622]

[Figure 2623]

[Figure 2624]

[Figure 2625]

[Figure 2626]

[Figure 2627]

[Figure 2628]

[Figure 2629]

[Figure 2630]

[Figure 2631]

[Figure 2632]

[Figure 2633]

[Figure 2634]

[Figure 2635]

[Figure 2637]

[Figure 2638]

[Figure 2639]

[Figure 2640]

[Figure 2641]

[Figure 2642]

[Figure 2643]

[Figure 2644]

[Figure 2645]

[Figure 2646]

[Figure 2647]

[Figure 2648]

[Figure 2649]

[Figure 2650]

[Figure 2651]

[Figure 2652]

[Figure 2653]

[Figure 2654]

[Figure 2655]

[Figure 2656]

[Figure 2657]

[Figure 2658]

[Figure 2659]

[Figure 2660]

[Figure 2661]

[Figure 2662]

[Figure 2663]

[Figure 2664]

[Figure 2665]

[Figure 2666]

[Figure 2667]

[Figure 2668]

[Figure 2669]

[Figure 2670]

[Figure 2671]

[Figure 2672]

[Figure 2673]

[Figure 2674]

[Figure 2675]

[Figure 2676]

[Figure 2677]

[Figure 2678]

[Figure 2679]

[Figure 2680]

[Figure 2681]

[Figure 2682]

[Figure 2683]

[Figure 2684]

[Figure 2685]

[Figure 2686]

[Figure 2687]

[Figure 2688]

[Figure 2689]

[Figure 2690]

[Figure 2691]

[Figure 2692]

[Figure 2693]

[Figure 2694]

[Figure 2695]

[Figure 2696]

[Figure 2697]

[Figure 2698]

[Figure 2699]

[Figure 2700]

[Figure 2701]

[Figure 2702]

[Figure 2703]

[Figure 2704]

[Figure 2705]

[Figure 2706]

[Figure 2707]

[Figure 2708]

[Figure 2709]

[Figure 2710]

[Figure 2711]

[Figure 2712]

[Figure 2713]

[Figure 2714]

[Figure 2717]

[Figure 2718]

[Figure 2719]

[Figure 2720]

[Figure 2721]

[Figure 2722]

[Figure 2723]

[Figure 2724]

[Figure 2725]

[Figure 2726]

[Figure 2727]

[Figure 2728]

[Figure 2729]

[Figure 2730]

[Figure 2731]

[Figure 2732]

[Figure 2737]

[Figure 2738]

[Figure 2739]

[Figure 2740]

[Figure 2741]

[Figure 2742]

[Figure 2743]

[Figure 2744]

[Figure 2745]

[Figure 2746]

[Figure 2747]

[Figure 2748]

[Figure 2749]

[Figure 2750]

[Figure 2751]

[Figure 2752]

[Figure 2753]

[Figure 2754]

[Figure 2755]

[Figure 2756]

[Figure 2757]

[Figure 2758]

[Figure 2759]

[Figure 2760]

[Figure 2761]

[Figure 2762]

[Figure 2763]

[Figure 2764]

[Figure 2765]

[Figure 2766]

[Figure 2767]

[Figure 2768]

[Figure 2769]

[Figure 2770]

[Figure 2771]

[Figure 2772]

[Figure 2773]

[Figure 2774]

[Figure 2775]

[Figure 2776]

[Figure 2777]

[Figure 2778]

[Figure 2780]

[Figure 2781]

[Figure 2782]

[Figure 2783]

[Figure 2784]

[Figure 2785]

[Figure 2786]

[Figure 2787]

[Figure 2788]

[Figure 2789]

[Figure 2790]

[Figure 2791]

[Figure 2792]

[Figure 2793]

[Figure 2794]

[Figure 2795]

[Figure 2800]

[Figure 2801]

[Figure 2802]

[Figure 2803]

[Figure 2804]

[Figure 2805]

[Figure 2806]

[Figure 2807]

[Figure 2808]

[Figure 2809]

[Figure 2810]

[Figure 2811]

[Figure 2812]

[Figure 2813]

[Figure 2814]

[Figure 2815]

[Figure 2816]

[Figure 2817]

[Figure 2820]

[Figure 2821]

[Figure 2822]

[Figure 2823]

[Figure 2824]

[Figure 2825]

[Figure 2826]

[Figure 2827]

[Figure 2828]

[Figure 2829]

[Figure 2830]

[Figure 2831]

[Figure 2832]

[Figure 2833]

[Figure 2834]

[Figure 2835]

[Figure 2836]

[Figure 2837]

[Figure 2841]

[Figure 2842]

[Figure 2843]

[Figure 2844]

[Figure 2845]

[Figure 2846]

[Figure 2847]

[Figure 2848]

[Figure 2849]

[Figure 2850]

[Figure 2851]

[Figure 2852]

[Figure 2853]

[Figure 2854]

[Figure 2855]

[Figure 2856]

[Figure 2857]

[Figure 2858]

[Figure 2861]

[Figure 2862]

[Figure 2863]

[Figure 2864]

[Figure 2865]

[Figure 2866]

[Figure 2867]

[Figure 2868]

[Figure 2869]

[Figure 2870]

[Figure 2871]

[Figure 2872]

[Figure 2873]

[Figure 2874]

[Figure 2875]

[Figure 2876]

[Figure 2877]

[Figure 2878]

[Figure 2879]

[Figure 2880]

[Figure 2881]

[Figure 2882]

[Figure 2883]

[Figure 2884]

[Figure 2885]

[Figure 2886]

[Figure 2887]

[Figure 2888]

[Figure 2889]

[Figure 2890]

[Figure 2891]

[Figure 2892]

[Figure 2893]

[Figure 2901]

[Figure 2902]

[Figure 2903]

[Figure 2904]

[Figure 2905]

[Figure 2906]

[Figure 2907]

[Figure 2908]

[Figure 2909]

[Figure 2910]

[Figure 2911]

[Figure 2912]

[Figure 2913]

[Figure 2914]

[Figure 2915]

[Figure 2916]

[Figure 2917]

[Figure 2918]

[Figure 2921]

[Figure 2922]

[Figure 2923]

[Figure 2924]

[Figure 2925]

[Figure 2926]

[Figure 2927]

[Figure 2928]

[Figure 2929]

[Figure 2930]

[Figure 2931]

[Figure 2932]

[Figure 2933]

[Figure 2934]

[Figure 2935]

[Figure 2936]

[Figure 2937]

[Figure 2938]

[Figure 2939]

[Figure 2942]

[Figure 2943]

[Figure 2944]

[Figure 2945]

[Figure 2946]

[Figure 2947]

[Figure 2948]

[Figure 2949]

[Figure 2950]

[Figure 2951]

[Figure 2952]

[Figure 2953]

[Figure 2954]

[Figure 2955]

[Figure 2956]

[Figure 2957]

[Figure 2958]

[Figure 2959]

[Figure 2962]

[Figure 2963]

[Figure 2964]

[Figure 2965]

[Figure 2966]

[Figure 2967]

[Figure 2968]

[Figure 2969]

[Figure 2970]

[Figure 2971]

[Figure 2972]

[Figure 2973]

[Figure 2974]

[Figure 2975]

[Figure 2976]

[Figure 2977]

[Figure 2978]

[Figure 2979]

[Figure 2980]

[Figure 2981]

[Figure 2982]

[Figure 2983]

[Figure 2984]

[Figure 2985]

[Figure 2986]

[Figure 2987]

[Figure 2988]

[Figure 2989]

[Figure 2990]

[Figure 2991]

[Figure 2992]

[Figure 2993]

[Figure 2994]

[Figure 3002]

[Figure 3003]

[Figure 3004]

[Figure 3005]

[Figure 3006]

[Figure 3007]

[Figure 3008]

[Figure 3009]

[Figure 3010]

[Figure 3011]

[Figure 3012]

[Figure 3013]

[Figure 3014]

[Figure 3015]

[Figure 3016]

[Figure 3017]

[Figure 3018]

[Figure 3019]

[Figure 3020]

[Figure 3021]

[Figure 3022]

[Figure 3023]

[Figure 3024]

[Figure 3025]

[Figure 3026]

[Figure 3027]

[Figure 3028]

[Figure 3029]

[Figure 3030]

[Figure 3031]

[Figure 3032]

[Figure 3033]

[Figure 3034]

[Figure 3035]

[Figure 3036]

[Figure 3037]

[Figure 3038]

[Figure 3042]

[Figure 3043]

[Figure 3044]

[Figure 3045]

[Figure 3046]

[Figure 3047]

[Figure 3048]

[Figure 3049]

[Figure 3050]

[Figure 3051]

[Figure 3052]

[Figure 3053]

[Figure 3054]

[Figure 3055]

[Figure 3056]

[Figure 3057]

[Figure 3058]

[Figure 3059]

[Figure 3060]

[Figure 3061]

[Figure 3062]

[Figure 3063]

[Figure 3064]

[Figure 3065]

[Figure 3066]

[Figure 3067]

[Figure 3068]

[Figure 3069]

[Figure 3070]

[Figure 3071]

[Figure 3072]

[Figure 3073]

[Figure 3074]

[Figure 3075]

[Figure 3076]

[Figure 3077]

[Figure 3078]

[Figure 3079]

[Figure 3080]

[Figure 3082]

[Figure 3083]

[Figure 3084]

[Figure 3085]

[Figure 3086]

[Figure 3087]

[Figure 3088]

[Figure 3089]

[Figure 3090]

[Figure 3091]

[Figure 3092]

[Figure 3093]

[Figure 3094]

[Figure 3095]

[Figure 3096]

[Figure 3097]

[Figure 3098]

[Figure 3099]

[Figure 3100]

[Figure 3101]

[Figure 3102]

[Figure 3103]

[Figure 3104]

[Figure 3105]

[Figure 3106]

[Figure 3107]

[Figure 3108]

[Figure 3109]

[Figure 3110]

[Figure 3111]

[Figure 3112]

[Figure 3113]

[Figure 3114]

[Figure 3115]

[Figure 3116]

[Figure 3117]

[Figure 3118]

[Figure 3119]

[Figure 3120]

[Figure 3121]

[Figure 3122]

[Figure 3123]

[Figure 3124]

[Figure 3125]

[Figure 3126]

[Figure 3127]

[Figure 3128]

[Figure 3129]

[Figure 3130]

[Figure 3131]

[Figure 3132]

[Figure 3133]

[Figure 3134]

[Figure 3135]

[Figure 3136]

[Figure 3137]

[Figure 3138]

[Figure 3139]

[Figure 3142]

[Figure 3143]

[Figure 3144]

[Figure 3145]

[Figure 3146]

[Figure 3147]

[Figure 3148]

[Figure 3149]

[Figure 3150]

[Figure 3151]

[Figure 3152]

[Figure 3153]

[Figure 3154]

[Figure 3155]

[Figure 3156]

[Figure 3157]

[Figure 3162]

- Cpt 1
- Cpt 2
- Cpt 3
- Cpt 4

- Concept 1
- Concept 2
- Concept 3
- Concept 4

[Figure 3163]

[Figure 3164]

[Figure 3165]

[Figure 3166]

[Figure 3167]

[Figure 3168]

[Figure 3169]

[Figure 3170]

[Figure 3171]

[Figure 3172]

[Figure 3173]

[Figure 3174]

[Figure 3175]

[Figure 3176]

[Figure 3177]

[Figure 3178]

[Figure 3179]

[Figure 3180]

[Figure 3181]

[Figure 3182]

[Figure 3183]

[Figure 3184]

[Figure 3185]

[Figure 3186]

[Figure 3187]

[Figure 3188]

[Figure 3189]

[Figure 3190]

[Figure 3191]

[Figure 3192]

[Figure 3193]

[Figure 3194]

[Figure 3195]

[Figure 3196]

[Figure 3197]

[Figure 3198]

[Figure 3203]

[Figure 3204]

[Figure 3205]

[Figure 3206]

[Figure 3207]

[Figure 3208]

[Figure 3209]

[Figure 3210]

[Figure 3211]

[Figure 3212]

[Figure 3213]

[Figure 3214]

[Figure 3215]

[Figure 3216]

[Figure 3217]

[Figure 3218]

[Figure 3219]

[Figure 3220]

[Figure 3221]

[Figure 3223]

[Figure 3224]

[Figure 3225]

[Figure 3226]

[Figure 3227]

[Figure 3228]

[Figure 3229]

[Figure 3230]

[Figure 3231]

[Figure 3232]

[Figure 3233]

[Figure 3234]

[Figure 3235]

[Figure 3236]

[Figure 3237]

[Figure 3238]

[Figure 3239]

[Figure 3240]

[Figure 3243]

[Figure 3244]

[Figure 3245]

[Figure 3246]

[Figure 3247]

[Figure 3248]

[Figure 3249]

[Figure 3250]

[Figure 3251]

[Figure 3252]

[Figure 3253]

[Figure 3254]

[Figure 3255]

[Figure 3256]

[Figure 3257]

[Figure 3258]

[Figure 3259]

[Figure 3263]

[Figure 3264]

[Figure 3265]

[Figure 3266]

[Figure 3267]

[Figure 3268]

[Figure 3269]

[Figure 3270]

[Figure 3271]

[Figure 3272]

[Figure 3273]

[Figure 3274]

[Figure 3275]

[Figure 3276]

[Figure 3277]

[Figure 3278]

[Figure 3279]

[Figure 3280]

[Figure 3283]

[Figure 3284]

[Figure 3285]

[Figure 3286]

[Figure 3287]

[Figure 3288]

[Figure 3289]

[Figure 3290]

[Figure 3291]

[Figure 3292]

[Figure 3293]

[Figure 3294]

[Figure 3295]

[Figure 3296]

[Figure 3297]

[Figure 3298]

[Figure 3299]

[Figure 3300]

[Figure 3301]

[Figure 3302]

[Figure 3303]

[Figure 3304]

[Figure 3305]

[Figure 3306]

[Figure 3307]

[Figure 3308]

[Figure 3309]

[Figure 3310]

[Figure 3311]

[Figure 3312]

[Figure 3313]

[Figure 3314]

[Figure 3315]

[Figure 3323]

[Figure 3324]

[Figure 3325]

[Figure 3326]

[Figure 3327]

[Figure 3328]

[Figure 3329]

[Figure 3330]

[Figure 3331]

[Figure 3332]

[Figure 3333]

[Figure 3334]

[Figure 3335]

[Figure 3336]

[Figure 3337]

[Figure 3338]

[Figure 3339]

[Figure 3340]

[Figure 3343]

[Figure 3344]

[Figure 3345]

[Figure 3346]

[Figure 3347]

[Figure 3348]

[Figure 3349]

[Figure 3350]

[Figure 3351]

[Figure 3352]

[Figure 3353]

[Figure 3354]

[Figure 3355]

[Figure 3356]

[Figure 3357]

[Figure 3358]

[Figure 3359]

[Figure 3360]

[Figure 3361]

[Figure 3362]

[Figure 3363]

[Figure 3364]

[Figure 3365]

[Figure 3366]

[Figure 3367]

[Figure 3368]

[Figure 3369]

[Figure 3370]

[Figure 3371]

[Figure 3372]

[Figure 3373]

[Figure 3374]

[Figure 3375]

[Figure 3376]

[Figure 3377]

[Figure 3378]

[Figure 3379]

[Figure 3380]

[Figure 3381]

[Figure 3383]

[Figure 3384]

[Figure 3385]

[Figure 3386]

[Figure 3387]

[Figure 3388]

[Figure 3389]

[Figure 3390]

[Figure 3391]

[Figure 3392]

[Figure 3393]

[Figure 3394]

[Figure 3395]

[Figure 3396]

[Figure 3397]

[Figure 3398]

[Figure 3399]

[Figure 3403]

[Figure 3404]

[Figure 3405]

[Figure 3406]

[Figure 3407]

[Figure 3408]

[Figure 3409]

[Figure 3410]

[Figure 3411]

[Figure 3412]

[Figure 3413]

[Figure 3414]

[Figure 3415]

[Figure 3416]

[Figure 3417]

[Figure 3418]

[Figure 3419]

[Figure 3420]

[Figure 3423]

[Figure 3424]

[Figure 3425]

[Figure 3426]

[Figure 3427]

[Figure 3428]

[Figure 3429]

[Figure 3430]

[Figure 3431]

[Figure 3432]

[Figure 3433]

[Figure 3434]

[Figure 3435]

[Figure 3436]

[Figure 3437]

[Figure 3438]

[Figure 3439]

[Figure 3440]

[Figure 3441]

[Figure 3442]

[Figure 3443]

[Figure 3444]

[Figure 3445]

[Figure 3446]

[Figure 3447]

[Figure 3448]

[Figure 3449]

[Figure 3450]

[Figure 3451]

[Figure 3452]

[Figure 3453]

[Figure 3454]

[Figure 3455]

[Figure 3463]

[Figure 3464]

[Figure 3465]

[Figure 3466]

[Figure 3467]

[Figure 3468]

[Figure 3469]

[Figure 3470]

[Figure 3471]

[Figure 3472]

[Figure 3473]

[Figure 3474]

[Figure 3475]

[Figure 3476]

[Figure 3477]

[Figure 3478]

[Figure 3479]

[Figure 3480]

[Figure 3481]

[Figure 3482]

[Figure 3483]

[Figure 3484]

[Figure 3485]

[Figure 3486]

[Figure 3487]

[Figure 3488]

[Figure 3489]

[Figure 3490]

[Figure 3491]

[Figure 3492]

[Figure 3493]

[Figure 3494]

[Figure 3495]

[Figure 3496]

[Figure 3497]

[Figure 3498]

[Figure 3499]

[Figure 3500]

[Figure 3503]

[Figure 3504]

[Figure 3505]

[Figure 3506]

[Figure 3507]

[Figure 3508]

[Figure 3509]

[Figure 3510]

[Figure 3511]

[Figure 3512]

[Figure 3513]

[Figure 3514]

[Figure 3515]

[Figure 3516]

[Figure 3517]

[Figure 3518]

[Figure 3519]

[Figure 3520]

[Figure 3521]

[Figure 3522]

[Figure 3523]

[Figure 3524]

[Figure 3525]

[Figure 3526]

[Figure 3527]

[Figure 3528]

[Figure 3529]

[Figure 3530]

[Figure 3531]

[Figure 3532]

[Figure 3533]

[Figure 3534]

[Figure 3535]

[Figure 3536]

[Figure 3537]

[Figure 3538]

[Figure 3539]

[Figure 3540]

[Figure 3541]

[Figure 3542]

[Figure 3543]

[Figure 3544]

[Figure 3545]

[Figure 3546]

[Figure 3547]

[Figure 3548]

[Figure 3549]

[Figure 3550]

[Figure 3551]

[Figure 3552]

[Figure 3553]

[Figure 3554]

[Figure 3555]

[Figure 3556]

[Figure 3557]

[Figure 3558]

[Figure 3559]

[Figure 3560]

[Figure 3563]

[Figure 3564]

[Figure 3565]

[Figure 3566]

[Figure 3567]

[Figure 3568]

[Figure 3569]

[Figure 3570]

[Figure 3571]

[Figure 3572]

[Figure 3573]

[Figure 3574]

[Figure 3575]

[Figure 3576]

[Figure 3577]

[Figure 3578]

[Figure 3583]

[Figure 3584]

[Figure 3585]

[Figure 3586]

[Figure 3587]

[Figure 3588]

[Figure 3589]

[Figure 3590]

[Figure 3591]

[Figure 3592]

[Figure 3593]

[Figure 3594]

[Figure 3595]

[Figure 3596]

[Figure 3597]

[Figure 3598]

[Figure 3599]

[Figure 3600]

[Figure 3601]

[Figure 3602]

[Figure 3603]

[Figure 3604]

[Figure 3605]

[Figure 3606]

[Figure 3607]

[Figure 3608]

[Figure 3609]

[Figure 3610]

[Figure 3611]

[Figure 3612]

[Figure 3613]

[Figure 3614]

[Figure 3615]

[Figure 3616]

[Figure 3617]

[Figure 3618]

[Figure 3619]

[Figure 3620]

[Figure 3621]

[Figure 3623]

[Figure 3624]

[Figure 3625]

[Figure 3626]

[Figure 3627]

[Figure 3628]

[Figure 3629]

[Figure 3630]

[Figure 3631]

[Figure 3632]

[Figure 3633]

[Figure 3634]

[Figure 3635]

[Figure 3636]

[Figure 3637]

[Figure 3638]

[Figure 3639]

[Figure 3640]

[Figure 3643]

[Figure 3644]

[Figure 3645]

[Figure 3646]

[Figure 3647]

[Figure 3648]

[Figure 3649]

[Figure 3650]

[Figure 3651]

[Figure 3652]

[Figure 3653]

[Figure 3654]

[Figure 3655]

[Figure 3656]

[Figure 3657]

[Figure 3658]

[Figure 3659]

[Figure 3660]

[Figure 3661]

[Figure 3662]

[Figure 3663]

[Figure 3664]

[Figure 3665]

[Figure 3666]

[Figure 3667]

[Figure 3668]

[Figure 3669]

[Figure 3670]

[Figure 3671]

[Figure 3672]

[Figure 3673]

[Figure 3674]

[Figure 3675]

[Figure 3683]

[Figure 3684]

[Figure 3685]

[Figure 3686]

[Figure 3687]

[Figure 3688]

[Figure 3689]

[Figure 3690]

[Figure 3691]

[Figure 3692]

[Figure 3693]

[Figure 3694]

[Figure 3695]

[Figure 3696]

[Figure 3697]

[Figure 3698]

[Figure 3699]

[Figure 3700]

[Figure 3703]

[Figure 3704]

[Figure 3705]

[Figure 3706]

[Figure 3707]

[Figure 3708]

[Figure 3709]

[Figure 3710]

[Figure 3711]

[Figure 3712]

[Figure 3713]

[Figure 3714]

[Figure 3715]

[Figure 3716]

[Figure 3717]

[Figure 3718]

[Figure 3719]

[Figure 3720]

[Figure 3721]

[Figure 3722]

[Figure 3723]

[Figure 3724]

[Figure 3725]

[Figure 3726]

[Figure 3727]

[Figure 3728]

[Figure 3729]

[Figure 3730]

[Figure 3731]

[Figure 3732]

[Figure 3733]

[Figure 3734]

[Figure 3735]

[Figure 3736]

[Figure 3737]

[Figure 3738]

[Figure 3739]

[Figure 3740]

[Figure 3741]

[Figure 3743]

[Figure 3744]

[Figure 3745]

[Figure 3746]

[Figure 3747]

[Figure 3748]

[Figure 3749]

[Figure 3750]

[Figure 3751]

[Figure 3752]

[Figure 3753]

[Figure 3754]

[Figure 3755]

[Figure 3756]

[Figure 3757]

[Figure 3758]

[Figure 3759]

[Figure 3763]

[Figure 3764]

[Figure 3765]

[Figure 3766]

[Figure 3767]

[Figure 3768]

[Figure 3769]

[Figure 3770]

[Figure 3771]

[Figure 3772]

[Figure 3773]

[Figure 3774]

[Figure 3775]

[Figure 3776]

[Figure 3777]

[Figure 3778]

[Figure 3779]

[Figure 3780]

[Figure 3781]

[Figure 3782]

[Figure 3783]

[Figure 3784]

[Figure 3785]

[Figure 3786]

[Figure 3787]

[Figure 3788]

[Figure 3789]

[Figure 3790]

[Figure 3791]

[Figure 3792]

[Figure 3793]

[Figure 3794]

[Figure 3795]

[Figure 3796]

[Figure 3797]

[Figure 3798]

[Figure 3799]

[Figure 3800]

[Figure 3801]

[Figure 3802]

[Figure 3803]

[Figure 3804]

[Figure 3805]

[Figure 3806]

[Figure 3807]

[Figure 3808]

[Figure 3809]

[Figure 3810]

[Figure 3811]

[Figure 3812]

[Figure 3813]

[Figure 3814]

[Figure 3815]

[Figure 3816]

[Figure 3817]

[Figure 3818]

[Figure 3819]

[Figure 3820]

[Figure 3821]

[Figure 3822]

[Figure 3823]

[Figure 3824]

[Figure 3825]

[Figure 3826]

[Figure 3827]

[Figure 3828]

[Figure 3829]

[Figure 3830]

[Figure 3831]

[Figure 3832]

[Figure 3833]

[Figure 3834]

[Figure 3835]

[Figure 3836]

[Figure 3837]

[Figure 3838]

[Figure 3839]

[Figure 3840]

[Figure 3843]

[Figure 3844]

[Figure 3845]

[Figure 3846]

[Figure 3847]

[Figure 3848]

[Figure 3849]

[Figure 3850]

[Figure 3851]

[Figure 3852]

[Figure 3853]

[Figure 3854]

[Figure 3855]

[Figure 3856]

[Figure 3857]

[Figure 3858]

[Figure 3859]

[Figure 3860]

[Figure 3863]

[Figure 3864]

[Figure 3865]

[Figure 3866]

[Figure 3867]

[Figure 3868]

[Figure 3869]

[Figure 3870]

[Figure 3871]

[Figure 3872]

[Figure 3873]

[Figure 3874]

[Figure 3875]

[Figure 3876]

[Figure 3877]

[Figure 3878]

[Figure 3879]

[Figure 3880]

[Figure 3881]

[Figure 3882]

[Figure 3883]

[Figure 3884]

[Figure 3885]

[Figure 3886]

[Figure 3887]

[Figure 3888]

[Figure 3889]

[Figure 3890]

[Figure 3891]

[Figure 3892]

[Figure 3893]

[Figure 3894]

[Figure 3895]

[Figure 3903]

[Figure 3904]

[Figure 3905]

[Figure 3906]

[Figure 3907]

[Figure 3908]

[Figure 3909]

[Figure 3910]

[Figure 3911]

[Figure 3912]

[Figure 3913]

[Figure 3914]

[Figure 3915]

[Figure 3916]

[Figure 3917]

[Figure 3918]

[Figure 3923]

[Figure 3924]

[Figure 3925]

[Figure 3926]

[Figure 3927]

[Figure 3928]

[Figure 3929]

[Figure 3930]

[Figure 3931]

[Figure 3932]

[Figure 3933]

[Figure 3934]

[Figure 3935]

[Figure 3936]

[Figure 3937]

[Figure 3938]

[Figure 3939]

[Figure 3940]

[Figure 3943]

[Figure 3944]

[Figure 3945]

[Figure 3946]

[Figure 3947]

[Figure 3948]

[Figure 3949]

[Figure 3950]

[Figure 3951]

[Figure 3952]

[Figure 3953]

[Figure 3954]

[Figure 3955]

[Figure 3956]

[Figure 3957]

[Figure 3958]

[Figure 3959]

[Figure 3960]

[Figure 3961]

[Figure 3962]

[Figure 3963]

[Figure 3964]

[Figure 3965]

[Figure 3966]

[Figure 3967]

[Figure 3968]

[Figure 3969]

[Figure 3970]

[Figure 3971]

[Figure 3972]

[Figure 3973]

[Figure 3974]

[Figure 3975]

[Figure 3976]

[Figure 3977]

[Figure 3978]

[Figure 3979]

[Figure 3980]

[Figure 3981]

[Figure 3982]

[Figure 3984]

[Figure 3985]

[Figure 3986]

[Figure 3987]

[Figure 3988]

[Figure 3989]

[Figure 3990]

[Figure 3991]

[Figure 3992]

[Figure 3993]

[Figure 3994]

[Figure 3995]

[Figure 3996]

[Figure 3997]

[Figure 3998]

[Figure 3999]

[Figure 4000]

[Figure 4001]

[Figure 4004]

[Figure 4005]

[Figure 4006]

[Figure 4007]

[Figure 4008]

[Figure 4009]

[Figure 4010]

[Figure 4011]

[Figure 4012]

[Figure 4013]

[Figure 4014]

[Figure 4015]

[Figure 4016]

[Figure 4017]

[Figure 4018]

[Figure 4019]

[Figure 4020]

[Figure 4021]

[Figure 4022]

[Figure 4023]

[Figure 4024]

[Figure 4025]

[Figure 4026]

[Figure 4027]

[Figure 4028]

[Figure 4029]

[Figure 4030]

[Figure 4031]

[Figure 4032]

[Figure 4033]

[Figure 4034]

[Figure 4035]

[Figure 4036]

[Figure 4044]

[Figure 4045]

[Figure 4046]

[Figure 4047]

[Figure 4048]

[Figure 4049]

[Figure 4050]

[Figure 4051]

[Figure 4052]

[Figure 4053]

[Figure 4054]

[Figure 4055]

[Figure 4056]

[Figure 4057]

[Figure 4058]

[Figure 4059]

[Figure 4060]

[Figure 4061]

[Figure 4064]

[Figure 4065]

[Figure 4066]

[Figure 4067]

[Figure 4068]

[Figure 4069]

[Figure 4070]

[Figure 4071]

[Figure 4072]

[Figure 4073]

[Figure 4074]

[Figure 4075]

[Figure 4076]

[Figure 4077]

[Figure 4078]

[Figure 4079]

[Figure 4080]

[Figure 4081]

[Figure 4082]

[Figure 4083]

[Figure 4084]

[Figure 4085]

[Figure 4086]

[Figure 4087]

[Figure 4088]

[Figure 4089]

[Figure 4090]

[Figure 4091]

[Figure 4092]

[Figure 4093]

[Figure 4094]

[Figure 4095]

[Figure 4096]

[Figure 4097]

[Figure 4098]

[Figure 4099]

[Figure 4100]

[Figure 4101]

[Figure 4102]

[Figure 4104]

[Figure 4105]

[Figure 4106]

[Figure 4107]

[Figure 4108]

[Figure 4109]

[Figure 4110]

[Figure 4111]

[Figure 4112]

[Figure 4113]

[Figure 4114]

[Figure 4115]

[Figure 4116]

[Figure 4117]

[Figure 4118]

[Figure 4119]

[Figure 4124]

[Figure 4125]

[Figure 4126]

[Figure 4127]

[Figure 4128]

[Figure 4129]

[Figure 4130]

[Figure 4131]

[Figure 4132]

[Figure 4133]

[Figure 4134]

[Figure 4135]

[Figure 4136]

[Figure 4137]

[Figure 4138]

[Figure 4139]

[Figure 4140]

[Figure 4144]

[Figure 4145]

[Figure 4146]

[Figure 4147]

[Figure 4148]

[Figure 4149]

[Figure 4150]

[Figure 4151]

[Figure 4152]

[Figure 4153]

[Figure 4154]

[Figure 4155]

[Figure 4156]

[Figure 4157]

[Figure 4158]

[Figure 4159]

[Figure 4160]

[Figure 4161]

[Figure 4164]

[Figure 4165]

[Figure 4166]

[Figure 4167]

[Figure 4168]

[Figure 4169]

[Figure 4170]

[Figure 4171]

[Figure 4172]

[Figure 4173]

[Figure 4174]

[Figure 4175]

[Figure 4176]

[Figure 4177]

[Figure 4178]

[Figure 4179]

[Figure 4180]

[Figure 4181]

[Figure 4184]

[Figure 4185]

[Figure 4186]

[Figure 4187]

[Figure 4188]

[Figure 4189]

[Figure 4190]

[Figure 4191]

[Figure 4192]

[Figure 4193]

[Figure 4194]

[Figure 4195]

[Figure 4196]

[Figure 4197]

[Figure 4198]

[Figure 4199]

[Figure 4200]

[Figure 4201]

[Figure 4202]

[Figure 4203]

[Figure 4204]

[Figure 4205]

[Figure 4206]

[Figure 4207]

[Figure 4208]

[Figure 4209]

[Figure 4210]

[Figure 4211]

[Figure 4212]

[Figure 4213]

[Figure 4214]

[Figure 4215]

[Figure 4216]

[Figure 4224]

[Figure 4225]

[Figure 4226]

[Figure 4227]

[Figure 4228]

[Figure 4229]

[Figure 4230]

[Figure 4231]

[Figure 4232]

[Figure 4233]

[Figure 4234]

[Figure 4235]

[Figure 4236]

[Figure 4237]

[Figure 4238]

[Figure 4239]

[Figure 4240]

[Figure 4241]

[Figure 4244]

[Figure 4245]

[Figure 4246]

[Figure 4247]

[Figure 4248]

[Figure 4249]

[Figure 4250]

[Figure 4251]

[Figure 4252]

[Figure 4253]

[Figure 4254]

[Figure 4255]

[Figure 4256]

[Figure 4257]

[Figure 4258]

[Figure 4259]

[Figure 4260]

[Figure 4261]

[Figure 4262]

[Figure 4263]

[Figure 4264]

[Figure 4265]

[Figure 4266]

[Figure 4267]

[Figure 4268]

[Figure 4269]

[Figure 4270]

[Figure 4271]

[Figure 4272]

[Figure 4273]

[Figure 4274]

[Figure 4275]

[Figure 4276]

[Figure 4277]

[Figure 4278]

[Figure 4279]

[Figure 4280]

[Figure 4281]

[Figure 4282]

[Figure 4284]

[Figure 4285]

[Figure 4286]

[Figure 4287]

[Figure 4288]

[Figure 4289]

[Figure 4290]

[Figure 4291]

[Figure 4292]

[Figure 4293]

[Figure 4294]

[Figure 4295]

[Figure 4296]

[Figure 4297]

[Figure 4298]

[Figure 4299]

[Figure 4300]

[Figure 4301]

[Figure 4304]

[Figure 4305]

[Figure 4306]

[Figure 4307]

[Figure 4308]

[Figure 4309]

[Figure 4310]

[Figure 4311]

[Figure 4312]

[Figure 4313]

[Figure 4314]

[Figure 4315]

[Figure 4316]

[Figure 4317]

[Figure 4318]

[Figure 4319]

[Figure 4320]

[Figure 4321]

[Figure 4322]

[Figure 4323]

[Figure 4324]

[Figure 4325]

[Figure 4326]

[Figure 4327]

[Figure 4328]

[Figure 4329]

[Figure 4330]

[Figure 4331]

[Figure 4332]

[Figure 4333]

[Figure 4334]

[Figure 4335]

[Figure 4336]

[Figure 4344]

[Figure 4345]

[Figure 4346]

[Figure 4347]

[Figure 4348]

[Figure 4349]

[Figure 4350]

[Figure 4351]

[Figure 4352]

[Figure 4353]

[Figure 4354]

[Figure 4355]

[Figure 4356]

[Figure 4357]

[Figure 4358]

[Figure 4359]

[Figure 4360]

[Figure 4361]

[Figure 4364]

[Figure 4365]

[Figure 4366]

[Figure 4367]

[Figure 4368]

[Figure 4369]

[Figure 4370]

[Figure 4371]

[Figure 4372]

[Figure 4373]

[Figure 4374]

[Figure 4375]

[Figure 4376]

[Figure 4377]

[Figure 4378]

[Figure 4379]

[Figure 4380]

[Figure 4381]

[Figure 4382]

[Figure 4383]

[Figure 4384]

[Figure 4385]

[Figure 4386]

[Figure 4387]

[Figure 4388]

[Figure 4389]

[Figure 4390]

[Figure 4391]

[Figure 4392]

[Figure 4393]

[Figure 4394]

[Figure 4395]

[Figure 4396]

[Figure 4397]

[Figure 4398]

[Figure 4399]

[Figure 4400]

[Figure 4401]

[Figure 4402]

[Figure 4404]

[Figure 4405]

[Figure 4406]

[Figure 4407]

[Figure 4408]

[Figure 4409]

[Figure 4410]

[Figure 4411]

[Figure 4412]

[Figure 4413]

[Figure 4414]

[Figure 4415]

[Figure 4416]

[Figure 4417]

[Figure 4418]

[Figure 4419]

[Figure 4424]

[Figure 4425]

[Figure 4426]

[Figure 4427]

[Figure 4428]

[Figure 4429]

[Figure 4430]

[Figure 4431]

[Figure 4432]

[Figure 4433]

[Figure 4434]

[Figure 4435]

[Figure 4436]

[Figure 4437]

[Figure 4438]

[Figure 4439]

[Figure 4440]

[Figure 4441]

[Figure 4444]

[Figure 4445]

[Figure 4446]

[Figure 4447]

[Figure 4448]

[Figure 4449]

[Figure 4450]

[Figure 4451]

[Figure 4452]

[Figure 4453]

[Figure 4454]

[Figure 4455]

[Figure 4456]

[Figure 4457]

[Figure 4458]

[Figure 4459]

[Figure 4460]

[Figure 4464]

[Figure 4465]

[Figure 4466]

[Figure 4467]

[Figure 4468]

[Figure 4469]

[Figure 4470]

[Figure 4471]

[Figure 4472]

[Figure 4473]

[Figure 4474]

[Figure 4475]

[Figure 4476]

[Figure 4477]

[Figure 4478]

[Figure 4479]

[Figure 4480]

[Figure 4481]

[Figure 4484]

[Figure 4485]

[Figure 4486]

[Figure 4487]

[Figure 4488]

[Figure 4489]

[Figure 4490]

[Figure 4491]

[Figure 4492]

[Figure 4493]

[Figure 4494]

[Figure 4495]

[Figure 4496]

[Figure 4497]

[Figure 4498]

[Figure 4499]

[Figure 4500]

[Figure 4501]

[Figure 4502]

[Figure 4503]

[Figure 4504]

[Figure 4505]

[Figure 4506]

[Figure 4507]

[Figure 4508]

[Figure 4509]

[Figure 4510]

[Figure 4511]

[Figure 4512]

[Figure 4513]

[Figure 4514]

[Figure 4515]

[Figure 4516]

[Figure 4524]

[Figure 4525]

[Figure 4526]

[Figure 4527]

[Figure 4528]

[Figure 4529]

[Figure 4530]

[Figure 4531]

[Figure 4532]

[Figure 4533]

[Figure 4534]

[Figure 4535]

[Figure 4536]

[Figure 4537]

[Figure 4538]

[Figure 4539]

[Figure 4540]

[Figure 4541]

- Figure 5. PACE’s three-level conceptual explanations on the Flower dataset. Dataset-Level: PACE’s top 4 dataset-level concepts (i.e.,

[Figure 4544]

[Figure 4545]

[Figure 4546]

[Figure 4547]

[Figure 4548]

[Figure 4549]

[Figure 4550]

[Figure 4551]

[Figure 4552]

[Figure 4553]

[Figure 4554]

[Figure 4555]

[Figure 4556]

[Figure 4557]

[Figure 4558]

[Figure 4559]

[Figure 4560]

[Figure 4561]

[Figure 4564]

[Figure 4565]

[Figure 4566]

[Figure 4567]

[Figure 4568]

[Figure 4569]

[Figure 4570]

[Figure 4571]

[Figure 4572]

[Figure 4573]

[Figure 4574]

[Figure 4575]

[Figure 4576]

[Figure 4577]

[Figure 4578]

[Figure 4579]

[Figure 4580]

[Figure 4581]

[Figure 4582]

[Figure 4583]

[Figure 4584]

[Figure 4585]

[Figure 4586]

[Figure 4587]

[Figure 4588]

[Figure 4589]

[Figure 4590]

[Figure 4591]

[Figure 4592]

[Figure 4593]

[Figure 4594]

[Figure 4595]

[Figure 4596]

[Figure 4604]

[Figure 4605]

[Figure 4606]

[Figure 4607]

[Figure 4608]

[Figure 4609]

[Figure 4610]

[Figure 4611]

[Figure 4612]

[Figure 4613]

[Figure 4614]

[Figure 4615]

[Figure 4616]

[Figure 4617]

[Figure 4618]

[Figure 4619]

[Figure 4620]

[Figure 4621]

[Figure 4624]

[Figure 4625]

[Figure 4626]

[Figure 4627]

[Figure 4628]

[Figure 4629]

[Figure 4630]

[Figure 4631]

[Figure 4632]

[Figure 4633]

[Figure 4634]

[Figure 4635]

[Figure 4636]

[Figure 4637]

[Figure 4638]

[Figure 4639]

[Figure 4640]

[Figure 4641]

[Figure 4642]

[Figure 4643]

[Figure 4644]

[Figure 4645]

[Figure 4646]

[Figure 4647]

[Figure 4648]

[Figure 4649]

[Figure 4650]

[Figure 4651]

[Figure 4652]

[Figure 4653]

[Figure 4654]

[Figure 4655]

[Figure 4656]

[Figure 4657]

[Figure 4658]

[Figure 4659]

[Figure 4664]

[Figure 4665]

[Figure 4666]

[Figure 4667]

[Figure 4668]

[Figure 4669]

[Figure 4670]

[Figure 4671]

[Figure 4672]

[Figure 4673]

[Figure 4674]

[Figure 4675]

[Figure 4676]

[Figure 4677]

[Figure 4678]

[Figure 4679]

[Figure 4680]

[Figure 4681]

[Figure 4682]

[Figure 4684]

[Figure 4685]

[Figure 4686]

[Figure 4687]

[Figure 4688]

[Figure 4689]

[Figure 4690]

[Figure 4691]

[Figure 4692]

[Figure 4693]

[Figure 4694]

[Figure 4695]

[Figure 4696]

[Figure 4697]

[Figure 4698]

[Figure 4699]

[Figure 4700]

[Figure 4701]

[Figure 4704]

[Figure 4705]

[Figure 4706]

[Figure 4707]

[Figure 4708]

[Figure 4709]

[Figure 4710]

[Figure 4711]

[Figure 4712]

[Figure 4713]

[Figure 4714]

[Figure 4715]

[Figure 4716]

[Figure 4717]

[Figure 4718]

[Figure 4719]

[Figure 4720]

‘Cpt 1’ to ‘Cpt 4’); for each concept, we plot the top 5 patches with emj closest to µk. Image-Level: Given an input image m, e.g., the top-left image, PACE’s generated image-level explanation θm indicates a strong association with Concepts 1 (green stem/leaves) and

[Figure 4724]

[Figure 4725]

[Figure 4726]

[Figure 4727]

[Figure 4728]

[Figure 4729]

[Figure 4730]

[Figure 4731]

[Figure 4732]

[Figure 4733]

[Figure 4734]

[Figure 4735]

[Figure 4736]

[Figure 4737]

[Figure 4738]

[Figure 4739]

[Figure 4740]

[Figure 4741]

[Figure 4744]

[Figure 4745]

[Figure 4746]

[Figure 4747]

[Figure 4748]

[Figure 4749]

[Figure 4750]

[Figure 4751]

[Figure 4752]

[Figure 4753]

[Figure 4754]

[Figure 4755]

[Figure 4756]

[Figure 4757]

[Figure 4758]

[Figure 4759]

[Figure 4760]

[Figure 4761]

[Figure 4762]

[Figure 4763]

[Figure 4764]

[Figure 4765]

[Figure 4766]

[Figure 4767]

[Figure 4768]

[Figure 4769]

[Figure 4770]

[Figure 4771]

[Figure 4772]

[Figure 4773]

[Figure 4774]

[Figure 4775]

[Figure 4776]

[Figure 4784]

[Figure 4785]

[Figure 4786]

[Figure 4787]

[Figure 4788]

[Figure 4789]

[Figure 4790]

[Figure 4791]

[Figure 4792]

[Figure 4793]

[Figure 4794]

[Figure 4795]

[Figure 4796]

[Figure 4797]

[Figure 4798]

[Figure 4799]

[Figure 4800]

[Figure 4801]

[Figure 4804]

[Figure 4805]

[Figure 4806]

[Figure 4807]

[Figure 4808]

[Figure 4809]

[Figure 4810]

[Figure 4811]

[Figure 4812]

[Figure 4813]

[Figure 4814]

[Figure 4815]

[Figure 4816]

[Figure 4817]

[Figure 4818]

[Figure 4819]

[Figure 4820]

[Figure 4821]

[Figure 4822]

[Figure 4823]

[Figure 4824]

[Figure 4825]

[Figure 4826]

[Figure 4827]

[Figure 4828]

[Figure 4829]

[Figure 4830]

[Figure 4831]

[Figure 4832]

[Figure 4833]

[Figure 4834]

[Figure 4835]

[Figure 4836]

[Figure 4837]

[Figure 4838]

[Figure 4839]

[Figure 4840]

[Figure 4841]

[Figure 4842]

[Figure 4844]

[Figure 4845]

[Figure 4846]

[Figure 4847]

[Figure 4848]

[Figure 4849]

[Figure 4850]

[Figure 4851]

[Figure 4852]

[Figure 4853]

[Figure 4854]

[Figure 4855]

[Figure 4856]

[Figure 4857]

[Figure 4858]

[Figure 4859]

[Figure 4860]

[Figure 4861]

[Figure 4864]

[Figure 4865]

[Figure 4866]

[Figure 4867]

[Figure 4868]

[Figure 4869]

[Figure 4870]

[Figure 4871]

[Figure 4872]

[Figure 4873]

[Figure 4874]

[Figure 4875]

[Figure 4876]

[Figure 4877]

[Figure 4878]

[Figure 4879]

[Figure 4880]

[Figure 4881]

[Figure 4882]

[Figure 4883]

[Figure 4884]

[Figure 4885]

[Figure 4886]

[Figure 4887]

[Figure 4888]

[Figure 4889]

[Figure 4890]

[Figure 4891]

[Figure 4892]

[Figure 4893]

[Figure 4894]

[Figure 4895]

[Figure 4896]

[Figure 4904]

[Figure 4905]

[Figure 4906]

[Figure 4907]

[Figure 4908]

[Figure 4909]

[Figure 4910]

[Figure 4911]

[Figure 4912]

[Figure 4913]

[Figure 4914]

[Figure 4915]

[Figure 4916]

[Figure 4917]

[Figure 4918]

[Figure 4919]

[Figure 4920]

[Figure 4921]

- 3 (purple petal). This is consistent with the image’s petal as the foreground and stem/leaves as the background. Patch-Level: For an

input image m, e.g., the top-left image, PACE’s ϕmj identifies the top concepts for patch j. The patch at the top (green stem/leaves) is associated with Concept 1 (containing similar appearance patches across the dataset); the middle patch (purple petal) is linked to Concept

[Figure 4924]

[Figure 4925]

[Figure 4926]

[Figure 4927]

[Figure 4928]

[Figure 4929]

[Figure 4930]

[Figure 4931]

[Figure 4932]

[Figure 4933]

[Figure 4934]

[Figure 4935]

[Figure 4936]

[Figure 4937]

[Figure 4938]

[Figure 4939]

[Figure 4940]

[Figure 4941]

[Figure 4942]

[Figure 4943]

[Figure 4944]

[Figure 4945]

[Figure 4946]

[Figure 4947]

[Figure 4948]

[Figure 4949]

[Figure 4950]

[Figure 4951]

[Figure 4952]

[Figure 4953]

[Figure 4954]

[Figure 4955]

[Figure 4956]

[Figure 4957]

[Figure 4958]

[Figure 4959]

[Figure 4960]

[Figure 4961]

[Figure 4962]

[Figure 4964]

[Figure 4965]

[Figure 4966]

[Figure 4967]

[Figure 4968]

[Figure 4969]

[Figure 4970]

[Figure 4971]

[Figure 4972]

[Figure 4973]

[Figure 4974]

[Figure 4975]

[Figure 4976]

[Figure 4977]

[Figure 4978]

[Figure 4979]

[Figure 4980]

[Figure 4981]

[Figure 4984]

[Figure 4985]

[Figure 4986]

[Figure 4987]

[Figure 4988]

[Figure 4989]

[Figure 4990]

[Figure 4991]

[Figure 4992]

[Figure 4993]

[Figure 4994]

[Figure 4995]

[Figure 4996]

[Figure 4997]

[Figure 4998]

[Figure 4999]

[Figure 5000]

[Figure 5001]

[Figure 5002]

[Figure 5003]

[Figure 5004]

[Figure 5005]

[Figure 5006]

[Figure 5007]

[Figure 5008]

[Figure 5009]

[Figure 5010]

[Figure 5011]

[Figure 5012]

[Figure 5013]

[Figure 5014]

[Figure 5015]

[Figure 5016]

[Figure 5024]

[Figure 5025]

[Figure 5026]

[Figure 5027]

[Figure 5028]

[Figure 5029]

[Figure 5030]

[Figure 5031]

[Figure 5032]

[Figure 5033]

[Figure 5034]

[Figure 5035]

[Figure 5036]

[Figure 5037]

[Figure 5038]

[Figure 5039]

[Figure 5040]

[Figure 5041]

- 3 (containing patches of other pincushion flower petal across the dataset). tiveness.

[Figure 5044]

[Figure 5045]

[Figure 5046]

[Figure 5047]

[Figure 5048]

[Figure 5049]

[Figure 5050]

[Figure 5051]

[Figure 5052]

[Figure 5053]

[Figure 5054]

[Figure 5055]

[Figure 5056]

[Figure 5057]

[Figure 5058]

[Figure 5059]

[Figure 5060]

[Figure 5061]

[Figure 5062]

[Figure 5063]

[Figure 5064]

[Figure 5065]

[Figure 5066]

[Figure 5067]

[Figure 5068]

[Figure 5069]

[Figure 5070]

[Figure 5071]

[Figure 5072]

[Figure 5073]

[Figure 5074]

[Figure 5075]

[Figure 5076]

[Figure 5077]

[Figure 5078]

[Figure 5079]

[Figure 5080]

[Figure 5081]

[Figure 5082]

[Figure 5084]

[Figure 5085]

[Figure 5086]

[Figure 5087]

[Figure 5088]

[Figure 5089]

[Figure 5090]

[Figure 5091]

[Figure 5092]

[Figure 5093]

[Figure 5094]

[Figure 5095]

[Figure 5096]

[Figure 5097]

[Figure 5098]

[Figure 5099]

[Figure 5100]

[Figure 5101]

[Figure 5104]

[Figure 5105]

[Figure 5106]

[Figure 5107]

[Figure 5108]

[Figure 5109]

[Figure 5110]

[Figure 5111]

[Figure 5112]

[Figure 5113]

[Figure 5114]

[Figure 5115]

[Figure 5116]

[Figure 5117]

[Figure 5118]

[Figure 5119]

[Figure 5120]

[Figure 5121]

[Figure 5122]

[Figure 5123]

[Figure 5124]

[Figure 5125]

[Figure 5126]

[Figure 5127]

[Figure 5128]

[Figure 5129]

[Figure 5130]

[Figure 5131]

[Figure 5132]

[Figure 5133]

[Figure 5134]

[Figure 5135]

[Figure 5136]

[Figure 5144]

[Figure 5145]

[Figure 5146]

[Figure 5147]

[Figure 5148]

[Figure 5149]

[Figure 5150]

[Figure 5151]

[Figure 5152]

[Figure 5153]

[Figure 5154]

[Figure 5155]

[Figure 5156]

[Figure 5157]

[Figure 5158]

[Figure 5159]

[Figure 5160]

[Figure 5161]

patch-level explanation. For the same image m above, PACE’s ϕmj identifies the top concepts for patch j. The patch at the top (green stem/leaves) is associated with Concept 1, comprising similar appearance patches across the dataset; the middle patch (purple petal) is linked to Concept 3, which includes patches of other pincushion flower petal across the dataset.

[Figure 5164]

[Figure 5165]

[Figure 5166]

[Figure 5167]

[Figure 5168]

[Figure 5169]

[Figure 5170]

[Figure 5171]

[Figure 5172]

[Figure 5173]

[Figure 5174]

[Figure 5175]

[Figure 5176]

[Figure 5177]

[Figure 5178]

[Figure 5179]

[Figure 5180]

[Figure 5181]

[Figure 5184]

[Figure 5185]

[Figure 5186]

[Figure 5187]

[Figure 5188]

[Figure 5189]

[Figure 5190]

[Figure 5191]

[Figure 5192]

[Figure 5193]

[Figure 5194]

[Figure 5195]

[Figure 5196]

[Figure 5197]

[Figure 5198]

[Figure 5199]

[Figure 5200]

[Figure 5201]

[Figure 5202]

[Figure 5203]

[Figure 5204]

[Figure 5205]

[Figure 5206]

[Figure 5207]

[Figure 5208]

[Figure 5209]

[Figure 5210]

[Figure 5211]

[Figure 5212]

[Figure 5213]

[Figure 5214]

[Figure 5215]

[Figure 5216]

[Figure 5224]

[Figure 5225]

[Figure 5226]

[Figure 5227]

[Figure 5228]

[Figure 5229]

[Figure 5230]

[Figure 5231]

[Figure 5232]

[Figure 5233]

[Figure 5234]

[Figure 5235]

[Figure 5236]

[Figure 5237]

[Figure 5238]

[Figure 5239]

[Figure 5240]

[Figure 5241]

- • Image-Level Explanation: Given an input image m, PACE infers the image-level concepts. For example, given the input image in the top row of Fig. 4,

PACE’s generated image-level explanation θm indicates a strong association with Concept 3 (green) and Concept 4 (blue). This is consistent with the color distribution in the image m, predominantly blue and less green.

- • Patch-Level Explanation: PACE also generated patch-level explanation. For the same image above,

See Appendix C for further qualitative analysis on more real-world datasets.

### 5. Conclusion

PACE’s ϕmj identifies the top concepts for the selected patches; the blue patch is associated with Concept 4 (containing similar blue patches across the dataset), while the green patch is linked to Concept 3 (containing green patches across the dataset).

In this paper, we identify five desiderata faithfulness, stability, sparsity, multi-level structure, and parsimony when generating trustworthy concept-level explanations for ViTs. We develop the first general method, PACE, that is compatible with any transformer variants and satisfies these desiderata. Through both quantitative and qualitative evaluations, our method demonstrates superior performance in explaining post-hoc ViT predictions via visual concepts, outperforming state-of-the-art methods across various datasets. As a limitation, our approach assumes a fixed number of concepts (Similar to existing methods). Therefore future work could focus on developing PACE into a non-parametric explainer that automatically determines the number of concepts. Another limitation is that our approach requires access to hidden states and attention weights from the layers inside ViTs; we argue that this is an advantage because it allows our PACE to interpret vision foundation models’ internals thoroughly rather than simply their output superficially.

Flower. Fig. 5 demonstrates PACE’s three-level explanations on the Flower dataset.

- • Dataset-Level Explanation: The dataset-level column shows the top 4 dataset-level concepts from our PACE, each with unique shapes, texture, and colors.
- • Image-Level Explanation: Given an input image m, PACE infers the image-level concepts. For example, given the input image in the top row of Fig. 5,

PACE’s generated image-level explanation θm indicates a strong association with Concepts 1 (green stem/leaves) and 3 (purple petal). This is consistent with the image’s petal as the foreground and stem/leaves as the background.

- • Patch-Level Explanation: PACE also generated

### Acknowledgements

We extend our heartfelt thanks to Akshay Nambi and Tanuja Ganu from Microsoft Research for their invaluable suggestions, which greatly improved the presentation of this paper. We are grateful for the support from Microsoft Research AI & Society Fellowship, NSF Grant IIS-2127918, and Amazon Faculty Research Award. Additionally, we thank the reviewers and the area chair/senior area chair for their insightful feedback and for recognizing the novelty and contributions of our work. We thank the Center for AI Safety (CAIS) for making computing resources available for this research.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

networks. In Artificial intelligence and statistics, pp. 81–

88. PMLR, 2009.

Chattopadhyay, A., Chan, K. H. R., and Vidal, R. Bootstrapping variational information pursuit with large language and vision models for interpretable image classification. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.

net/forum?id=9bmTbVaA2A.

Chefer, H., Gur, S., and Wolf, L. Generic attention-model explainability for interpreting bi-modal and encoderdecoder transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 397– 406, 2021a.

Chefer, H., Gur, S., and Wolf, L. Transformer interpretability beyond attention visualization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 782–791, 2021b.

### References

Agarwal, C., Krishna, S., Saxena, E., Pawelczyk, M., Johnson, N., Puri, I., Zitnik, M., and Lakkaraju, H. Openxai: Towards a transparent evaluation of model explanations. Advances in Neural Information Processing Systems, 35: 15784–15799, 2022.

Alvarez Melis, D. and Jaakkola, T. Towards robust interpretability with self-explaining neural networks. Advances in neural information processing systems, 31, 2018.

Bach, S., Binder, A., Montavon, G., Klauschen, F., M¨uller, K.-R., and Samek, W. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PloS one, 10(7):e0130140, 2015.

Ben Melech Stan, G., Yehezkel Rohekar, R., Gurwicz, Y., Olson, M. L., Bhiwandiwalla, A., Aflalo, E., Wu, C., Duan, N., Tseng, S.-Y., and Lal, V. Lvlm-intrepret: An interpretability tool for large vision-language models. arXiv e-prints, pp. arXiv–2404, 2024.

Bennetot, A., Franchi, G., Del Ser, J., Chatila, R., and DiazRodriguez, N. Greybox xai: A neural-symbolic learning framework to produce interpretable predictions for image classification. Knowledge-Based Systems, 258:109947, 2022.

Blei, D. M., Ng, A. Y., and Jordan, M. I. Latent dirichlet allocation. the Journal of machine Learning research, 3: 993–1022, 2003.

Chang, J. and Blei, D. Relational topic models for document

Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., and Su, J. K. This looks like that: deep learning for interpretable image recognition. Advances in neural information processing systems, 32, 2019.

Chen, R., Li, J., Zhang, H., Sheng, C., Liu, L., and Cao, X. Sim2word: Explaining similarity with representative attribute words via counterfactual explanations. ACM Transactions on Multimedia Computing, Communications and Applications, 19(6):1–22, 2023.

Chen, R., Zhang, H., Liang, S., Li, J., and Cao, X. Less is more: Fewer interpretable region via submodular subset selection. arXiv preprint arXiv:2402.09164, 2024.

Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597–1607. PMLR, 2020.

Colin, J., Fel, T., Cad`ene, R., and Serre, T. What i cannot predict, i do not understand: A human-centered evaluation framework for explainability methods. Advances in Neural Information Processing Systems, 35:2832–2845, 2022.

Covert, I., Kim, C., and Lee, S.-I. Learning to estimate shapley values with vision transformers. arXiv preprint arXiv:2206.05282, 2022.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Fel, T., Boutin, V., Moayeri, M., Cad`ene, R., Bethune, L., Chalvidal, M., Serre, T., et al. A holistic approach to unifying automatic concept extraction and concept importance estimation. arXiv preprint arXiv:2306.07304, 2023a.

Fel, T., Picard, A., Bethune, L., Boissin, T., Vigouroux, D., Colin, J., Cad`ene, R., and Serre, T. Craft: Concept recursive activation factorization for explainability. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2711–2721, 2023b.

Ghorbani, A., Wexler, J., Zou, J. Y., and Kim, B. Towards automatic concept-based explanations. Advances in neural information processing systems, 32, 2019.

Gilpin, L. H., Bau, D., Yuan, B. Z., Bajwa, A., Specter, M., and Kagal, L. Explaining explanations: An overview of interpretability of machine learning. In 2018 IEEE 5th International Conference on data science and advanced analytics (DSAA), pp. 80–89. IEEE, 2018.

Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., and Saul, L. K. An introduction to variational methods for graphical models. In Learning in graphical models, pp. 105–161. Springer, 1998.

Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In International conference on machine learning, pp. 2668–2677. PMLR, 2018.

Kim, S., Oh, J., Lee, S., Yu, S., Do, J., and Taghavi, T. Grounding counterfactual explanation of image classifiers to textual concept space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10942–10950, 2023.

Kindermans, P.-J., Sch¨utt, K., M¨uller, K.-R., and D¨ahne, S. Investigating the influence of noise and distractors on the interpretation of neural networks. arXiv preprint arXiv:1611.07270, 2016.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson,

- E., Kim, B., and Liang, P. Concept bottleneck models. In International conference on machine learning, pp. 5338–

5348. PMLR, 2020.

Krause, J., Stark, M., Deng, J., and Fei-Fei, L. 3d object representations for fine-grained categorization. In Proceedings of the IEEE international conference on computer vision workshops, pp. 554–561, 2013.

Langer, M., Oster, D., Speith, T., Hermanns, H., K¨astner, L., Schmidt, E., Sesing, A., and Baum, K. What do we want from explainable artificial intelligence (xai)?–a stakeholder perspective on xai and a conceptual model guiding interdisciplinary xai research. Artificial Intelligence, 296:103473, 2021.

Li, J., Kuang, K., Li, L., Chen, L., Zhang, S., Shao, J., and Xiao, J. Instance-wise or class-wise? a tale of neighbor shapley for concept-based explanation. In Proceedings of the 29th ACM International Conference on Multimedia, pp. 3664–3672, 2021a.

Li, J., Zhang, C., Zhou, J. T., Fu, H., Xia, S., and Hu, Q. Deep-lift: Deep label-specific feature learning for image annotation. IEEE transactions on Cybernetics, 52(8): 7732–7741, 2021b.

Li, X.-H., Shi, Y., Li, H., Bai, W., Song, Y., Cao, C. C., and Chen, L. Quantitative evaluations on saliency methods: An experimental study. arXiv preprint arXiv:2012.15616, 2020.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.

Losch, M., Fritz, M., and Schiele, B. Interpretability beyond classification output: Semantic bottleneck networks. arXiv preprint arXiv:1907.10882, 2019.

Lundberg, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30, 2017.

Menon, S. and Vondrick, C. Visual classification via description from large language models. arXiv preprint arXiv:2210.07183, 2022.

Ming, Y., Cai, Z., Gu, J., Sun, Y., Li, W., and Li, Y. Delving into out-of-distribution detection with vision-language representations. Advances in neural information processing systems, 35:35087–35102, 2022.

Murdoch, W. J., Singh, C., Kumbier, K., Abbasi-Asl, R., and Yu, B. Definitions, methods, and applications in interpretable machine learning. Proceedings of the National Academy of Sciences, 116(44):22071–22080, 2019.

Nilsback, M.-E. and Zisserman, A. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics & image processing, pp. 722–729. IEEE, 2008.

Novello, P., Fel, T., and Vigouroux, D. Making sense of dependence: Efficient black-box explanations using

dependence measure, september 2022. arXiv preprint arXiv:2206.06219.

Oikarinen, T., Das, S., Nguyen, L. M., and Weng, T.-W. Label-free concept bottleneck models. arXiv preprint arXiv:2304.06129, 2023.

Pan, D., Li, X., and Zhu, D. Explaining deep neural network models with adversarial gradient integration. In Thirtieth International Joint Conference on Artificial Intelligence (IJCAI), 2021.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, pp. 8024–8035, 2019.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Ribeiro, M. T., Singh, S., and Guestrin, C. ” why should i trust you?” explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 1135–1144, 2016.

Rohekar, R. Y., Gurwicz, Y., and Nisimov, S. Causal interpretation of self-attention in pre-trained transformers. Advances in Neural Information Processing Systems, 36, 2024.

Schwalbe, G. and Finzel, B. A comprehensive taxonomy for explainable artificial intelligence: a systematic survey of surveys on methods and concepts. Data Mining and Knowledge Discovery, pp. 1–59, 2023.

Shrikumar, A., Greenside, P., and Kundaje, A. Learning important features through propagating activation differences. In International conference on machine learning, pp. 3145–3153. PMLR, 2017.

Simonyan, K., Vedaldi, A., and Zisserman, A. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.

Sundararajan, M., Taly, A., and Yan, Q. Axiomatic attribution for deep networks. In International conference on machine learning, pp. 3319–3328. PMLR, 2017.

Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., and J´egou, H. Training data-efficient image transformers & distillation through attention. In International conference on machine learning, pp. 10347–10357. PMLR, 2021.

Wah, C., Branson, S., Welinder, P., Perona, P., and Belongie, S. The caltech-ucsd birds-200-2011 dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.

- Wang, A., Lee, W.-N., and Qi, X. Hint: Hierarchical neuron concept explainer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10254–10264, 2022.
- Wang, B., Li, L., Nakashima, Y., and Nagahara, H. Learning bottleneck concepts in image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10962–10971, 2023.

Wang, H. and Yan, J. Self-interpretable time series prediction with counterfactual explanations. In ICML, 2023.

Wang, H. and Yeung, D.-Y. Towards bayesian deep learning: A framework and some existing methods. TDKE, 28(12): 3395–3408, 2016.

Wang, H. and Yeung, D.-Y. A survey on bayesian deep learning. CSUR, 53(5):1–37, 2020.

Wang, H., Xingjian, S., and Yeung, D.-Y. Natural-parameter networks: A class of probabilistic neural networks. In NIPS, pp. 118–126, 2016.

Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., Mardziel, P., and Hu, X. Score-cam: Score-weighted visual explanations for convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pp. 24–25, 2020.

Xie, W., Li, X.-H., Cao, C. C., and Zhang, N. L. Vit-cx: Causal explanation of vision transformers. arXiv preprint arXiv:2211.03064, 2022.

Xu, X., Qin, Y., Mi, L., Wang, H., and Li, X. Energy-based concept bottleneck models: Unifying prediction, concept intervention, and probabilistic interpretations. In ICLR, 2024.

Xu, Z., Hao, G., He, H., and Wang, H. Domain indexing variational bayes: Interpretable domain index for domain adaptation. In ICLR, 2023.

Yang, Y., Panagopoulou, A., Zhou, S., Jin, D., CallisonBurch, C., and Yatskar, M. Language in a bottle: Language model guided concept bottlenecks for interpretable image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19187–19197, 2023.

Yuksekgonul, M., Wang, M., and Zou, J. Post-hoc concept bottleneck models. arXiv preprint arXiv:2205.15480, 2022.

Zhang, R., Madumal, P., Miller, T., Ehinger, K. A., and Rubinstein, B. I. Invertible concept-based explanations for cnn models with non-negative concept activation vectors. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 11682–11690, 2021.

- A. Implementation Details In this section, we provide implementation details of PACE.

Color Dataset Generation. We constructed Color as a synthetic dataset with clear definition of 4 concepts (red/yellow/green/blue). It contains two image classes: Class 0 (images with red/yellow colors) and Class 1 (green/blue colors), both against a black background (see example images in Fig. 3). Images are initially created at a 2 × 2 resolution, where each pixel samples color from (red/yellow,green/blue,black), and are subsequently up-sampled with gradual color shift to 224 × 224 for ViT inputs. We introduce random Gaussian noise to each image. The dataset includes 1000 images per class, with a split of 800 training and 200 test samples.

Table 3. Dataset statistics, i.e., the number of train/test images (Mtrain/Mtest), the number of classes N, and the number of patches J per image.

###### Dataset Mtrain Mtest N J

Color 1,600 400 2 197 Flower 7,169 1,020 102 197 Cars 8,144 8,041 196 197 CUB 5,994 5,794 200 197

Table 3 provides statistics for the COLOR dataset along with three additional real-world datasets. Experimental Setup. Following the approach outlined in (Chen et al., 2020), we implement perturbation described in Definition 3.1 based on their augmentation algorithms, as demonstrated by the following code snippet:

contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(size=size), transforms.RandomApply([

transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)

], p=0.8), transforms.RandomGrayscale(p=0.2), transforms.GaussianBlur(kernel_size=9), transforms.Normalize((0.5,), (0.5,))

])

We also utilize in-batch negative examples according to (Chen et al., 2020). We implemented and trained using PyTorch (Paszke et al., 2019) on an A5000 GPU with 24GB of memory. The training duration does not exceed one day for all four datasets. We employ the Adam optimizer (Kingma & Ba, 2014) with initial learning rates ranging from 10−5 to 10−3, depending on the dataset.

From preliminary results, we observed that a smaller value for K is inadequate for effectively learning significant image concepts. Conversely, a larger value for K tends to lead to redundancy among the population of all concepts. Consequently, we adhered to the baseline methodologies (e.g., (Fel et al., 2023b)) by setting K to 100 across all datasets. This number was chosen as it strikes an effective balance between capturing adequate detail and avoiding model overfitting.

Baselines Methods. For the implementation of the baseline methods, we either utilize the original packages provided by the authors (Lundberg & Lee, 2017; Ribeiro et al., 2016; Fel et al., 2023b), or implement their methods by referencing the authors’ code (Simonyan et al., 2013; Pan et al., 2021). To account for the stochastic nature of methods like those in (Ribeiro et al., 2016; Lundberg & Lee, 2017), we perform multiple executions of these baseline methods, averaging scores across all runs. The frequency of repetition is contingent upon the time required to generate explanations. To balance efficiency and effectiveness, SHAP is executed 100 times, and LIME 10 times. In contrast, our PACE, being fully deterministic post-training, requires only a single inference on the test set.

Details on the Quantitative Analysis. In the results shown in Table 1 in Sec. 4.4, PACE offers Full three-level conceptual explanations, encompassing dataset, image, and patch levels. In contrast, CRAFT (Fel et al., 2023b) is limited to providing explanations at only the patch and image levels, lacking dataset-level insights and thereby exhibiting a Partial multi-level structure. Other baseline models are unable to achieve this multi-level conceptual explanation. Both the baseline CRAFT and our PACE are inherently compatible with arbitrary number of concepts K, therefore enjoying better parsimony by setting K = 100. This choice of K is driven by the goal of maintaining a moderate dimension size while ensuring that

[Figure 5244]

[Figure 5245]

[Figure 5246]

[Figure 5247]

[Figure 5248]

[Figure 5249]

[Figure 5250]

[Figure 5251]

[Figure 5252]

[Figure 5253]

[Figure 5254]

[Figure 5264]

[Figure 5265]

[Figure 5266]

[Figure 5267]

[Figure 5268]

[Figure 5269]

[Figure 5270]

[Figure 5271]

[Figure 5272]

[Figure 5273]

[Figure 5274]

[Figure 5284]

[Figure 5285]

[Figure 5286]

[Figure 5287]

[Figure 5288]

[Figure 5289]

[Figure 5290]

[Figure 5291]

[Figure 5292]

[Figure 5293]

[Figure 5294]

[Figure 5295]

[Figure 5304]

[Figure 5305]

[Figure 5306]

[Figure 5307]

[Figure 5308]

[Figure 5309]

[Figure 5310]

[Figure 5311]

[Figure 5312]

[Figure 5313]

[Figure 5314]

[Figure 5324]

[Figure 5325]

[Figure 5326]

[Figure 5327]

[Figure 5328]

[Figure 5329]

[Figure 5330]

[Figure 5331]

[Figure 5332]

[Figure 5333]

[Figure 5334]

[Figure 5344]

[Figure 5345]

[Figure 5346]

[Figure 5347]

[Figure 5348]

[Figure 5349]

[Figure 5350]

[Figure 5351]

[Figure 5352]

[Figure 5353]

[Figure 5354]

[Figure 5355]

[Figure 5364]

[Figure 5365]

[Figure 5366]

[Figure 5367]

[Figure 5368]

[Figure 5369]

[Figure 5370]

[Figure 5371]

[Figure 5372]

[Figure 5373]

[Figure 5374]

[Figure 5375]

[Figure 5376]

[Figure 5377]

[Figure 5378]

[Figure 5379]

[Figure 5380]

[Figure 5381]

[Figure 5382]

[Figure 5383]

[Figure 5384]

[Figure 5385]

[Figure 5386]

[Figure 5387]

[Figure 5388]

[Figure 5404]

[Figure 5405]

[Figure 5406]

[Figure 5407]

[Figure 5408]

[Figure 5409]

[Figure 5410]

[Figure 5411]

[Figure 5412]

[Figure 5413]

[Figure 5414]

[Figure 5424]

[Figure 5425]

[Figure 5426]

[Figure 5427]

[Figure 5428]

[Figure 5429]

[Figure 5430]

[Figure 5431]

[Figure 5432]

[Figure 5433]

[Figure 5434]

[Figure 5444]

[Figure 5445]

[Figure 5446]

[Figure 5447]

[Figure 5448]

[Figure 5449]

[Figure 5450]

[Figure 5451]

[Figure 5452]

[Figure 5453]

[Figure 5454]

[Figure 5464]

[Figure 5465]

[Figure 5466]

[Figure 5467]

[Figure 5468]

[Figure 5469]

[Figure 5470]

[Figure 5471]

[Figure 5472]

[Figure 5473]

[Figure 5474]

[Figure 5475]

[Figure 5484]

[Figure 5485]

[Figure 5486]

[Figure 5487]

[Figure 5488]

[Figure 5489]

[Figure 5490]

[Figure 5491]

[Figure 5492]

[Figure 5493]

[Figure 5494]

[Figure 5495]

[Figure 5496]

[Figure 5497]

[Figure 5498]

[Figure 5499]

[Figure 5500]

[Figure 5501]

[Figure 5502]

[Figure 5503]

[Figure 5504]

[Figure 5505]

[Figure 5506]

[Figure 5507]

[Figure 5508]

[Figure 5509]

[Figure 5510]

[Figure 5511]

[Figure 5512]

[Figure 5513]

[Figure 5514]

[Figure 5515]

[Figure 5516]

[Figure 5517]

[Figure 5518]

[Figure 5519]

###### Probabilistic Conceptual Explainers: Trustworthy Conceptual Explanations for Vision Foundation Models

[Figure 5544]

[Figure 5545]

[Figure 5546]

[Figure 5547]

[Figure 5548]

[Figure 5549]

[Figure 5550]

[Figure 5551]

[Figure 5552]

[Figure 5553]

[Figure 5554]

[Figure 5564]

[Figure 5565]

[Figure 5566]

[Figure 5567]

[Figure 5568]

[Figure 5569]

[Figure 5570]

[Figure 5571]

[Figure 5572]

[Figure 5573]

[Figure 5574]

[Figure 5584]

[Figure 5585]

[Figure 5586]

[Figure 5587]

[Figure 5588]

[Figure 5589]

[Figure 5590]

[Figure 5591]

[Figure 5592]

[Figure 5593]

[Figure 5594]

[Figure 5595]

[Figure 5604]

[Figure 5605]

[Figure 5606]

[Figure 5607]

[Figure 5608]

[Figure 5609]

[Figure 5610]

[Figure 5611]

[Figure 5612]

[Figure 5613]

[Figure 5614]

[Figure 5615]

[Figure 5616]

[Figure 5617]

[Figure 5618]

[Figure 5619]

[Figure 5620]

[Figure 5621]

[Figure 5622]

[Figure 5623]

[Figure 5624]

[Figure 5625]

[Figure 5626]

[Figure 5627]

[Figure 5628]

[Figure 5629]

[Figure 5630]

[Figure 5631]

[Figure 5632]

[Figure 5633]

[Figure 5634]

[Figure 5635]

[Figure 5636]

[Figure 5637]

[Figure 5638]

[Figure 5639]

[Figure 5664]

[Figure 5665]

[Figure 5666]

[Figure 5667]

[Figure 5668]

[Figure 5669]

[Figure 5670]

[Figure 5671]

[Figure 5672]

[Figure 5673]

###### Example Images Top Concepts

[Figure 5684]

[Figure 5685]

[Figure 5686]

[Figure 5687]

[Figure 5688]

[Figure 5689]

[Figure 5690]

[Figure 5691]

[Figure 5692]

[Figure 5693]

[Figure 5694]

[Figure 5695]

[Figure 5704]

[Figure 5705]

[Figure 5706]

[Figure 5707]

[Figure 5708]

[Figure 5709]

[Figure 5710]

[Figure 5711]

[Figure 5712]

[Figure 5713]

[Figure 5714]

[Figure 5724]

[Figure 5725]

[Figure 5726]

[Figure 5727]

[Figure 5728]

[Figure 5729]

[Figure 5730]

[Figure 5731]

[Figure 5732]

[Figure 5733]

[Figure 5734]

[Figure 5744]

[Figure 5745]

[Figure 5746]

[Figure 5747]

[Figure 5748]

[Figure 5749]

[Figure 5750]

[Figure 5751]

[Figure 5752]

[Figure 5753]

[Figure 5754]

[Figure 5755]

[Figure 5764]

[Figure 5765]

[Figure 5766]

[Figure 5767]

[Figure 5768]

[Figure 5769]

[Figure 5770]

[Figure 5771]

[Figure 5772]

[Figure 5773]

[Figure 5774]

[Figure 5775]

[Figure 5776]

[Figure 5777]

[Figure 5778]

[Figure 5779]

[Figure 5780]

[Figure 5781]

[Figure 5782]

[Figure 5783]

[Figure 5784]

[Figure 5785]

[Figure 5786]

[Figure 5787]

[Figure 5788]

[Figure 5789]

[Figure 5790]

[Figure 5791]

[Figure 5792]

[Figure 5793]

[Figure 5809]

[Figure 5810]

[Figure 5811]

[Figure 5812]

[Figure 5813]

[Figure 5814]

[Figure 5815]

[Figure 5816]

[Figure 5817]

[Figure 5818]

[Figure 5819]

[Figure 5829]

[Figure 5830]

[Figure 5831]

[Figure 5832]

[Figure 5833]

[Figure 5834]

[Figure 5835]

[Figure 5836]

[Figure 5837]

[Figure 5838]

[Figure 5849]

[Figure 5850]

[Figure 5851]

[Figure 5852]

[Figure 5853]

[Figure 5854]

[Figure 5855]

[Figure 5856]

[Figure 5857]

[Figure 5858]

[Figure 5859]

[Figure 5860]

[Figure 5861]

[Figure 5869]

[Figure 5870]

[Figure 5871]

[Figure 5872]

[Figure 5873]

[Figure 5874]

[Figure 5875]

[Figure 5876]

[Figure 5877]

[Figure 5878]

[Figure 5879]

[Figure 5880]

[Figure 5889]

[Figure 5890]

[Figure 5891]

[Figure 5892]

[Figure 5893]

[Figure 5894]

[Figure 5895]

[Figure 5896]

[Figure 5897]

[Figure 5898]

[Figure 5899]

[Figure 5900]

[Figure 5909]

[Figure 5910]

[Figure 5911]

[Figure 5912]

[Figure 5913]

[Figure 5914]

[Figure 5915]

[Figure 5916]

[Figure 5917]

[Figure 5918]

[Figure 5919]

[Figure 5929]

[Figure 5930]

[Figure 5931]

[Figure 5932]

[Figure 5933]

[Figure 5934]

[Figure 5935]

[Figure 5936]

[Figure 5937]

[Figure 5938]

[Figure 5939]

[Figure 5940]

[Figure 5941]

[Figure 5942]

[Figure 5943]

[Figure 5944]

[Figure 5945]

[Figure 5946]

[Figure 5947]

[Figure 5948]

[Figure 5949]

[Figure 5950]

[Figure 5951]

[Figure 5952]

[Figure 5953]

[Figure 5954]

[Figure 5955]

[Figure 5969]

[Figure 5970]

[Figure 5971]

[Figure 5972]

[Figure 5973]

[Figure 5974]

[Figure 5975]

[Figure 5976]

[Figure 5977]

[Figure 5978]

[Figure 5979]

[Figure 5980]

- Concept 1
- Concept 2
- Concept 3

[Figure 5989]

[Figure 5990]

[Figure 5991]

[Figure 5992]

[Figure 5993]

[Figure 5994]

[Figure 5995]

[Figure 5996]

[Figure 5997]

[Figure 5998]

[Figure 5999]

[Figure 6009]

[Figure 6010]

[Figure 6011]

[Figure 6012]

[Figure 6013]

[Figure 6014]

[Figure 6015]

[Figure 6016]

[Figure 6017]

[Figure 6018]

[Figure 6019]

[Figure 6029]

[Figure 6030]

[Figure 6031]

[Figure 6032]

[Figure 6033]

[Figure 6034]

[Figure 6035]

[Figure 6036]

[Figure 6037]

[Figure 6038]

[Figure 6039]

[Figure 6040]

[Figure 6049]

[Figure 6050]

[Figure 6051]

[Figure 6052]

[Figure 6053]

[Figure 6054]

[Figure 6055]

[Figure 6056]

[Figure 6057]

[Figure 6058]

[Figure 6059]

[Figure 6060]

[Figure 6061]

[Figure 6062]

[Figure 6063]

[Figure 6064]

[Figure 6065]

[Figure 6066]

[Figure 6067]

[Figure 6068]

[Figure 6069]

[Figure 6070]

[Figure 6071]

[Figure 6072]

[Figure 6073]

[Figure 6089]

[Figure 6090]

[Figure 6091]

[Figure 6092]

[Figure 6093]

[Figure 6094]

[Figure 6095]

[Figure 6096]

[Figure 6097]

[Figure 6098]

[Figure 6099]

[Figure 6109]

[Figure 6110]

[Figure 6111]

[Figure 6112]

[Figure 6113]

[Figure 6114]

[Figure 6115]

[Figure 6116]

[Figure 6117]

[Figure 6118]

[Figure 6119]

[Figure 6120]

[Figure 6129]

[Figure 6130]

[Figure 6131]

[Figure 6132]

[Figure 6133]

[Figure 6134]

[Figure 6135]

[Figure 6136]

[Figure 6137]

[Figure 6138]

[Figure 6149]

[Figure 6150]

[Figure 6151]

[Figure 6152]

[Figure 6153]

[Figure 6154]

[Figure 6155]

[Figure 6156]

[Figure 6157]

[Figure 6158]

[Figure 6169]

[Figure 6170]

[Figure 6171]

[Figure 6172]

[Figure 6173]

[Figure 6174]

[Figure 6175]

[Figure 6176]

[Figure 6177]

[Figure 6178]

[Figure 6179]

[Figure 6180]

[Figure 6181]

[Figure 6189]

[Figure 6190]

[Figure 6191]

[Figure 6192]

[Figure 6193]

[Figure 6194]

[Figure 6195]

[Figure 6196]

[Figure 6197]

[Figure 6198]

[Figure 6199]

[Figure 6200]

[Figure 6209]

[Figure 6210]

[Figure 6211]

[Figure 6212]

[Figure 6213]

[Figure 6214]

[Figure 6215]

[Figure 6216]

[Figure 6217]

[Figure 6218]

[Figure 6219]

[Figure 6229]

[Figure 6230]

[Figure 6231]

[Figure 6232]

[Figure 6233]

[Figure 6234]

[Figure 6235]

[Figure 6236]

[Figure 6237]

[Figure 6238]

[Figure 6239]

[Figure 6240]

[Figure 6249]

[Figure 6250]

[Figure 6251]

[Figure 6252]

[Figure 6253]

[Figure 6254]

[Figure 6255]

[Figure 6256]

[Figure 6257]

[Figure 6258]

[Figure 6259]

[Figure 6260]

[Figure 6261]

[Figure 6262]

[Figure 6263]

[Figure 6264]

[Figure 6265]

[Figure 6266]

[Figure 6267]

[Figure 6268]

[Figure 6269]

[Figure 6270]

[Figure 6271]

[Figure 6272]

[Figure 6273]

[Figure 6274]

[Figure 6275]

Black-footed Albatross

[Figure 6289]

[Figure 6290]

[Figure 6291]

[Figure 6292]

[Figure 6293]

[Figure 6294]

[Figure 6295]

[Figure 6296]

[Figure 6297]

[Figure 6298]

[Figure 6299]

[Figure 6300]

[Figure 6309]

[Figure 6310]

[Figure 6311]

[Figure 6312]

[Figure 6313]

[Figure 6314]

[Figure 6315]

[Figure 6316]

[Figure 6317]

[Figure 6318]

[Figure 6319]

[Figure 6329]

[Figure 6330]

[Figure 6331]

[Figure 6332]

[Figure 6333]

[Figure 6334]

[Figure 6335]

[Figure 6336]

[Figure 6337]

[Figure 6338]

[Figure 6339]

[Figure 6349]

[Figure 6350]

[Figure 6351]

[Figure 6352]

[Figure 6353]

[Figure 6354]

[Figure 6355]

[Figure 6356]

[Figure 6357]

[Figure 6358]

[Figure 6359]

[Figure 6360]

[Figure 6369]

[Figure 6370]

[Figure 6371]

[Figure 6372]

[Figure 6373]

[Figure 6374]

[Figure 6375]

[Figure 6376]

[Figure 6377]

[Figure 6378]

[Figure 6379]

[Figure 6380]

[Figure 6381]

[Figure 6382]

[Figure 6383]

[Figure 6384]

[Figure 6385]

[Figure 6386]

[Figure 6387]

[Figure 6388]

[Figure 6389]

[Figure 6390]

[Figure 6391]

[Figure 6392]

[Figure 6393]

[Figure 6409]

[Figure 6410]

[Figure 6411]

[Figure 6412]

[Figure 6413]

[Figure 6414]

[Figure 6415]

[Figure 6416]

[Figure 6417]

[Figure 6418]

[Figure 6419]

[Figure 6429]

[Figure 6430]

[Figure 6431]

[Figure 6432]

[Figure 6433]

[Figure 6434]

[Figure 6435]

[Figure 6436]

[Figure 6437]

[Figure 6438]

[Figure 6439]

[Figure 6440]

[Figure 6449]

[Figure 6450]

[Figure 6451]

[Figure 6452]

[Figure 6453]

[Figure 6454]

[Figure 6455]

[Figure 6456]

[Figure 6457]

[Figure 6458]

[Figure 6469]

[Figure 6470]

[Figure 6471]

[Figure 6472]

[Figure 6473]

[Figure 6474]

[Figure 6475]

[Figure 6476]

[Figure 6477]

[Figure 6478]

[Figure 6489]

[Figure 6490]

[Figure 6491]

[Figure 6492]

[Figure 6493]

[Figure 6494]

[Figure 6495]

[Figure 6496]

[Figure 6497]

[Figure 6498]

[Figure 6499]

[Figure 6500]

[Figure 6509]

[Figure 6510]

[Figure 6511]

[Figure 6512]

[Figure 6513]

[Figure 6514]

[Figure 6515]

[Figure 6516]

[Figure 6517]

[Figure 6518]

[Figure 6519]

[Figure 6520]

[Figure 6521]

[Figure 6529]

[Figure 6530]

[Figure 6531]

[Figure 6532]

[Figure 6533]

[Figure 6534]

[Figure 6535]

[Figure 6536]

[Figure 6537]

[Figure 6538]

[Figure 6539]

[Figure 6540]

[Figure 6541]

[Figure 6542]

[Figure 6543]

[Figure 6544]

[Figure 6545]

[Figure 6546]

[Figure 6547]

[Figure 6548]

[Figure 6549]

[Figure 6550]

[Figure 6551]

[Figure 6552]

[Figure 6553]

[Figure 6554]

[Figure 6555]

[Figure 6569]

[Figure 6570]

[Figure 6571]

[Figure 6572]

[Figure 6573]

[Figure 6574]

[Figure 6575]

[Figure 6576]

[Figure 6577]

[Figure 6578]

[Figure 6579]

[Figure 6589]

[Figure 6590]

[Figure 6591]

[Figure 6592]

[Figure 6593]

[Figure 6594]

[Figure 6595]

[Figure 6596]

[Figure 6597]

[Figure 6598]

[Figure 6599]

[Figure 6600]

[Figure 6609]

[Figure 6610]

[Figure 6611]

[Figure 6612]

[Figure 6613]

[Figure 6614]

[Figure 6615]

[Figure 6616]

[Figure 6617]

[Figure 6618]

[Figure 6619]

[Figure 6620]

[Figure 6629]

[Figure 6630]

[Figure 6631]

[Figure 6632]

[Figure 6633]

[Figure 6634]

[Figure 6635]

[Figure 6636]

[Figure 6637]

[Figure 6638]

[Figure 6639]

[Figure 6649]

[Figure 6650]

[Figure 6651]

[Figure 6652]

[Figure 6653]

[Figure 6654]

[Figure 6655]

[Figure 6656]

[Figure 6657]

[Figure 6658]

[Figure 6659]

[Figure 6669]

[Figure 6670]

[Figure 6671]

[Figure 6672]

[Figure 6673]

[Figure 6674]

[Figure 6675]

[Figure 6676]

[Figure 6677]

[Figure 6678]

[Figure 6679]

[Figure 6680]

[Figure 6689]

[Figure 6690]

[Figure 6691]

[Figure 6692]

[Figure 6693]

[Figure 6694]

[Figure 6695]

[Figure 6696]

[Figure 6697]

[Figure 6698]

[Figure 6699]

[Figure 6700]

[Figure 6701]

[Figure 6702]

[Figure 6703]

[Figure 6704]

[Figure 6705]

[Figure 6706]

[Figure 6707]

[Figure 6708]

[Figure 6709]

[Figure 6710]

[Figure 6711]

[Figure 6712]

[Figure 6713]

[Figure 6729]

[Figure 6730]

[Figure 6731]

[Figure 6732]

[Figure 6733]

[Figure 6734]

[Figure 6735]

[Figure 6736]

[Figure 6737]

[Figure 6738]

[Figure 6739]

[Figure 6749]

[Figure 6750]

[Figure 6751]

[Figure 6752]

[Figure 6753]

[Figure 6754]

[Figure 6755]

[Figure 6756]

[Figure 6757]

[Figure 6758]

[Figure 6759]

[Figure 6760]

[Figure 6769]

[Figure 6770]

[Figure 6771]

[Figure 6772]

[Figure 6773]

[Figure 6774]

[Figure 6775]

[Figure 6776]

[Figure 6777]

[Figure 6778]

[Figure 6789]

[Figure 6790]

[Figure 6791]

[Figure 6792]

[Figure 6793]

[Figure 6794]

[Figure 6795]

[Figure 6796]

[Figure 6797]

[Figure 6798]

[Figure 6809]

[Figure 6810]

[Figure 6811]

[Figure 6812]

[Figure 6813]

[Figure 6814]

[Figure 6815]

[Figure 6816]

[Figure 6817]

[Figure 6818]

[Figure 6819]

[Figure 6820]

[Figure 6821]

[Figure 6822]

[Figure 6823]

[Figure 6824]

[Figure 6825]

[Figure 6826]

[Figure 6827]

[Figure 6828]

[Figure 6829]

[Figure 6830]

[Figure 6831]

[Figure 6832]

[Figure 6833]

[Figure 6834]

[Figure 6835]

[Figure 6849]

[Figure 6850]

[Figure 6851]

[Figure 6852]

[Figure 6853]

[Figure 6854]

[Figure 6855]

[Figure 6856]

[Figure 6857]

[Figure 6858]

[Figure 6859]

[Figure 6860]

[Figure 6869]

[Figure 6870]

[Figure 6871]

[Figure 6872]

[Figure 6873]

[Figure 6874]

[Figure 6875]

[Figure 6876]

[Figure 6877]

[Figure 6878]

[Figure 6879]

[Figure 6880]

[Figure 6881]

[Figure 6889]

[Figure 6890]

[Figure 6891]

[Figure 6892]

[Figure 6893]

[Figure 6894]

[Figure 6895]

[Figure 6896]

[Figure 6897]

[Figure 6898]

[Figure 6899]

[Figure 6909]

[Figure 6910]

[Figure 6911]

[Figure 6912]

[Figure 6913]

[Figure 6914]

[Figure 6915]

[Figure 6916]

[Figure 6917]

[Figure 6918]

[Figure 6919]

[Figure 6920]

[Figure 6929]

[Figure 6930]

[Figure 6931]

[Figure 6932]

[Figure 6933]

[Figure 6934]

[Figure 6935]

[Figure 6936]

[Figure 6937]

[Figure 6938]

[Figure 6939]

[Figure 6940]

[Figure 6941]

[Figure 6950]

[Figure 6951]

[Figure 6952]

[Figure 6953]

[Figure 6954]

[Figure 6955]

[Figure 6956]

[Figure 6957]

[Figure 6958]

[Figure 6959]

[Figure 6960]

[Figure 6961]

[Figure 6962]

[Figure 6963]

[Figure 6964]

[Figure 6974]

[Figure 6975]

[Figure 6976]

[Figure 6977]

[Figure 6978]

[Figure 6979]

[Figure 6980]

[Figure 6981]

[Figure 6982]

[Figure 6983]

[Figure 6984]

[Figure 6994]

[Figure 6995]

[Figure 6996]

[Figure 6997]

[Figure 6998]

[Figure 6999]

[Figure 7000]

[Figure 7001]

[Figure 7002]

[Figure 7003]

[Figure 7004]

[Figure 7005]

[Figure 7014]

[Figure 7015]

[Figure 7016]

[Figure 7017]

[Figure 7018]

[Figure 7019]

[Figure 7020]

[Figure 7021]

[Figure 7022]

[Figure 7023]

[Figure 7024]

[Figure 7025]

[Figure 7026]

[Figure 7027]

[Figure 7028]

[Figure 7029]

[Figure 7030]

[Figure 7031]

[Figure 7032]

[Figure 7033]

[Figure 7034]

[Figure 7035]

[Figure 7036]

[Figure 7037]

[Figure 7038]

[Figure 7054]

[Figure 7055]

[Figure 7056]

[Figure 7057]

[Figure 7058]

[Figure 7059]

[Figure 7060]

[Figure 7061]

[Figure 7062]

[Figure 7063]

[Figure 7064]

- Concept 1
- Concept 2
- Concept 3

[Figure 7074]

[Figure 7075]

[Figure 7076]

[Figure 7077]

[Figure 7078]

[Figure 7079]

[Figure 7080]

[Figure 7081]

[Figure 7082]

[Figure 7083]

[Figure 7084]

[Figure 7085]

[Figure 7094]

[Figure 7095]

[Figure 7096]

[Figure 7097]

[Figure 7098]

[Figure 7099]

[Figure 7100]

[Figure 7101]

[Figure 7102]

[Figure 7103]

[Figure 7114]

[Figure 7115]

[Figure 7116]

[Figure 7117]

[Figure 7118]

[Figure 7119]

[Figure 7120]

[Figure 7121]

[Figure 7122]

[Figure 7123]

[Figure 7134]

[Figure 7135]

[Figure 7136]

[Figure 7137]

[Figure 7138]

[Figure 7139]

[Figure 7140]

[Figure 7141]

[Figure 7142]

[Figure 7143]

[Figure 7144]

[Figure 7145]

[Figure 7154]

[Figure 7155]

[Figure 7156]

[Figure 7157]

[Figure 7158]

[Figure 7159]

[Figure 7160]

[Figure 7161]

[Figure 7162]

[Figure 7163]

[Figure 7164]

[Figure 7165]

[Figure 7166]

[Figure 7167]

[Figure 7168]

[Figure 7169]

[Figure 7170]

[Figure 7171]

[Figure 7172]

[Figure 7173]

[Figure 7174]

[Figure 7175]

[Figure 7176]

[Figure 7177]

[Figure 7178]

[Figure 7179]

[Figure 7180]

[Figure 7194]

[Figure 7195]

[Figure 7196]

[Figure 7197]

[Figure 7198]

[Figure 7199]

[Figure 7200]

[Figure 7201]

[Figure 7202]

[Figure 7203]

[Figure 7204]

[Figure 7205]

[Figure 7214]

[Figure 7215]

[Figure 7216]

[Figure 7217]

[Figure 7218]

[Figure 7219]

[Figure 7220]

[Figure 7221]

[Figure 7222]

[Figure 7223]

[Figure 7224]

[Figure 7225]

[Figure 7226]

[Figure 7234]

[Figure 7235]

[Figure 7236]

[Figure 7237]

[Figure 7238]

[Figure 7239]

[Figure 7240]

[Figure 7241]

[Figure 7242]

[Figure 7243]

[Figure 7244]

[Figure 7254]

[Figure 7255]

[Figure 7256]

[Figure 7257]

[Figure 7258]

[Figure 7259]

[Figure 7260]

[Figure 7261]

[Figure 7262]

[Figure 7263]

[Figure 7264]

[Figure 7265]

[Figure 7274]

[Figure 7275]

[Figure 7276]

[Figure 7277]

[Figure 7278]

[Figure 7279]

[Figure 7280]

[Figure 7281]

[Figure 7282]

[Figure 7283]

[Figure 7284]

[Figure 7294]

[Figure 7295]

[Figure 7296]

[Figure 7297]

[Figure 7298]

[Figure 7299]

[Figure 7300]

[Figure 7301]

[Figure 7302]

[Figure 7303]

[Figure 7304]

[Figure 7305]

[Figure 7306]

[Figure 7307]

[Figure 7308]

[Figure 7309]

[Figure 7310]

[Figure 7311]

[Figure 7312]

[Figure 7313]

[Figure 7314]

[Figure 7315]

[Figure 7316]

[Figure 7317]

[Figure 7318]

[Figure 7334]

[Figure 7335]

[Figure 7336]

[Figure 7337]

[Figure 7338]

[Figure 7339]

[Figure 7340]

[Figure 7341]

[Figure 7342]

[Figure 7343]

[Figure 7344]

[Figure 7354]

[Figure 7355]

[Figure 7356]

[Figure 7357]

[Figure 7358]

[Figure 7359]

[Figure 7360]

[Figure 7361]

[Figure 7362]

[Figure 7363]

[Figure 7364]

[Figure 7365]

[Figure 7374]

[Figure 7375]

[Figure 7376]

[Figure 7377]

[Figure 7378]

[Figure 7379]

[Figure 7380]

[Figure 7381]

[Figure 7382]

[Figure 7383]

[Figure 7384]

Crested Auklet

[Figure 7394]

[Figure 7395]

[Figure 7396]

[Figure 7397]

[Figure 7398]

[Figure 7399]

[Figure 7400]

[Figure 7401]

[Figure 7402]

[Figure 7403]

[Figure 7404]

[Figure 7405]

[Figure 7414]

[Figure 7415]

[Figure 7416]

[Figure 7417]

[Figure 7418]

[Figure 7419]

[Figure 7420]

[Figure 7421]

[Figure 7422]

[Figure 7423]

[Figure 7434]

[Figure 7435]

[Figure 7436]

[Figure 7437]

[Figure 7438]

[Figure 7439]

[Figure 7440]

[Figure 7441]

[Figure 7442]

[Figure 7443]

[Figure 7454]

[Figure 7455]

[Figure 7456]

[Figure 7457]

[Figure 7458]

[Figure 7459]

[Figure 7460]

[Figure 7461]

[Figure 7462]

[Figure 7463]

[Figure 7464]

[Figure 7465]

[Figure 7474]

[Figure 7475]

[Figure 7476]

[Figure 7477]

[Figure 7478]

[Figure 7479]

[Figure 7480]

[Figure 7481]

[Figure 7482]

[Figure 7483]

[Figure 7484]

[Figure 7485]

[Figure 7486]

[Figure 7487]

[Figure 7488]

[Figure 7489]

[Figure 7490]

[Figure 7491]

[Figure 7492]

[Figure 7493]

[Figure 7494]

[Figure 7495]

[Figure 7496]

[Figure 7497]

[Figure 7498]

[Figure 7499]

[Figure 7500]

[Figure 7514]

[Figure 7515]

[Figure 7516]

[Figure 7517]

[Figure 7518]

[Figure 7519]

[Figure 7520]

[Figure 7521]

[Figure 7522]

[Figure 7523]

[Figure 7524]

[Figure 7525]

[Figure 7534]

[Figure 7535]

[Figure 7536]

[Figure 7537]

[Figure 7538]

[Figure 7539]

[Figure 7540]

[Figure 7541]

[Figure 7542]

[Figure 7543]

[Figure 7544]

[Figure 7545]

[Figure 7546]

[Figure 7554]

[Figure 7555]

[Figure 7556]

[Figure 7557]

[Figure 7558]

[Figure 7559]

[Figure 7560]

[Figure 7561]

[Figure 7562]

[Figure 7563]

[Figure 7564]

[Figure 7574]

[Figure 7575]

[Figure 7576]

[Figure 7577]

[Figure 7578]

[Figure 7579]

[Figure 7580]

[Figure 7581]

[Figure 7582]

[Figure 7583]

[Figure 7584]

[Figure 7585]

[Figure 7594]

[Figure 7595]

[Figure 7596]

[Figure 7597]

[Figure 7598]

[Figure 7599]

[Figure 7600]

[Figure 7601]

[Figure 7602]

[Figure 7603]

[Figure 7604]

[Figure 7605]

[Figure 7606]

[Figure 7607]

[Figure 7608]

[Figure 7609]

[Figure 7610]

[Figure 7611]

[Figure 7612]

[Figure 7613]

[Figure 7614]

[Figure 7615]

[Figure 7616]

[Figure 7617]

[Figure 7618]

[Figure 7619]

[Figure 7620]

[Figure 7621]

[Figure 7622]

[Figure 7623]

[Figure 7624]

[Figure 7625]

[Figure 7626]

[Figure 7627]

[Figure 7628]

[Figure 7629]

[Figure 7654]

[Figure 7655]

[Figure 7656]

[Figure 7657]

[Figure 7658]

[Figure 7659]

[Figure 7660]

[Figure 7661]

[Figure 7662]

[Figure 7663]

[Figure 7664]

[Figure 7665]

[Figure 7674]

[Figure 7675]

[Figure 7676]

[Figure 7677]

[Figure 7678]

[Figure 7679]

[Figure 7680]

[Figure 7681]

[Figure 7682]

[Figure 7683]

[Figure 7694]

[Figure 7695]

[Figure 7696]

[Figure 7697]

[Figure 7698]

[Figure 7699]

[Figure 7700]

[Figure 7701]

[Figure 7702]

[Figure 7703]

[Figure 7714]

[Figure 7715]

[Figure 7716]

[Figure 7717]

[Figure 7718]

[Figure 7719]

[Figure 7720]

[Figure 7721]

[Figure 7722]

[Figure 7723]

[Figure 7724]

[Figure 7725]

[Figure 7734]

[Figure 7735]

[Figure 7736]

[Figure 7737]

[Figure 7738]

[Figure 7739]

[Figure 7740]

[Figure 7741]

[Figure 7742]

[Figure 7743]

[Figure 7744]

[Figure 7745]

[Figure 7746]

[Figure 7747]

[Figure 7748]

[Figure 7749]

[Figure 7750]

[Figure 7751]

[Figure 7752]

[Figure 7753]

[Figure 7754]

[Figure 7755]

[Figure 7756]

[Figure 7757]

[Figure 7758]

[Figure 7759]

[Figure 7760]

[Figure 7774]

[Figure 7775]

[Figure 7776]

[Figure 7777]

[Figure 7778]

[Figure 7779]

[Figure 7780]

[Figure 7781]

[Figure 7782]

[Figure 7783]

[Figure 7784]

[Figure 7785]

[Figure 7794]

[Figure 7795]

[Figure 7796]

[Figure 7797]

[Figure 7798]

[Figure 7799]

[Figure 7800]

[Figure 7801]

[Figure 7802]

[Figure 7803]

[Figure 7804]

[Figure 7805]

[Figure 7806]

[Figure 7814]

[Figure 7815]

[Figure 7816]

[Figure 7817]

[Figure 7818]

[Figure 7819]

[Figure 7820]

[Figure 7821]

[Figure 7822]

[Figure 7823]

[Figure 7824]

[Figure 7834]

[Figure 7835]

[Figure 7836]

[Figure 7837]

[Figure 7838]

[Figure 7839]

[Figure 7840]

[Figure 7841]

[Figure 7842]

[Figure 7843]

[Figure 7844]

[Figure 7845]

[Figure 7854]

[Figure 7855]

[Figure 7856]

[Figure 7857]

[Figure 7858]

[Figure 7859]

[Figure 7860]

[Figure 7861]

[Figure 7862]

[Figure 7863]

[Figure 7864]

[Figure 7865]

[Figure 7866]

[Figure 7867]

[Figure 7868]

[Figure 7869]

[Figure 7870]

[Figure 7879]

[Figure 7880]

[Figure 7881]

[Figure 7882]

[Figure 7883]

[Figure 7884]

[Figure 7885]

[Figure 7886]

[Figure 7887]

[Figure 7888]

[Figure 7899]

[Figure 7900]

[Figure 7901]

[Figure 7902]

[Figure 7903]

[Figure 7904]

[Figure 7905]

[Figure 7906]

[Figure 7907]

[Figure 7908]

[Figure 7909]

[Figure 7910]

[Figure 7911]

[Figure 7919]

[Figure 7920]

[Figure 7921]

[Figure 7922]

[Figure 7923]

[Figure 7924]

[Figure 7925]

[Figure 7926]

[Figure 7927]

[Figure 7928]

[Figure 7929]

[Figure 7939]

[Figure 7940]

[Figure 7941]

[Figure 7942]

[Figure 7943]

[Figure 7944]

[Figure 7945]

[Figure 7946]

[Figure 7947]

[Figure 7948]

[Figure 7949]

[Figure 7950]

[Figure 7959]

[Figure 7960]

[Figure 7961]

[Figure 7962]

[Figure 7963]

[Figure 7964]

[Figure 7965]

[Figure 7966]

[Figure 7967]

[Figure 7968]

- Concept 1
- Concept 2
- Concept 3

[Figure 7979]

[Figure 7980]

[Figure 7981]

[Figure 7982]

[Figure 7983]

[Figure 7984]

[Figure 7985]

[Figure 7986]

[Figure 7987]

[Figure 7988]

[Figure 7989]

[Figure 7990]

[Figure 7991]

[Figure 7992]

[Figure 7993]

[Figure 7994]

[Figure 7995]

[Figure 7996]

[Figure 7997]

[Figure 7998]

[Figure 7999]

[Figure 8000]

[Figure 8001]

[Figure 8002]

[Figure 8003]

[Figure 8004]

[Figure 8005]

[Figure 8019]

[Figure 8020]

[Figure 8021]

[Figure 8022]

[Figure 8023]

[Figure 8024]

[Figure 8025]

[Figure 8026]

[Figure 8027]

[Figure 8028]

[Figure 8029]

[Figure 8030]

[Figure 8039]

[Figure 8040]

[Figure 8041]

[Figure 8042]

[Figure 8043]

[Figure 8044]

[Figure 8045]

[Figure 8046]

[Figure 8047]

[Figure 8048]

[Figure 8049]

[Figure 8050]

[Figure 8051]

[Figure 8059]

[Figure 8060]

[Figure 8061]

[Figure 8062]

[Figure 8063]

[Figure 8064]

[Figure 8065]

[Figure 8066]

[Figure 8067]

[Figure 8068]

[Figure 8069]

[Figure 8079]

[Figure 8080]

[Figure 8081]

[Figure 8082]

[Figure 8083]

[Figure 8084]

[Figure 8085]

[Figure 8086]

[Figure 8087]

[Figure 8088]

[Figure 8089]

[Figure 8090]

[Figure 8099]

[Figure 8100]

[Figure 8101]

[Figure 8102]

[Figure 8103]

[Figure 8104]

[Figure 8105]

[Figure 8106]

[Figure 8107]

[Figure 8108]

[Figure 8109]

[Figure 8110]

[Figure 8119]

[Figure 8120]

[Figure 8121]

[Figure 8122]

[Figure 8123]

[Figure 8124]

[Figure 8125]

[Figure 8126]

[Figure 8127]

[Figure 8128]

[Figure 8139]

[Figure 8140]

[Figure 8141]

[Figure 8142]

[Figure 8143]

[Figure 8144]

[Figure 8145]

[Figure 8146]

[Figure 8147]

[Figure 8148]

[Figure 8149]

[Figure 8150]

[Figure 8159]

[Figure 8160]

[Figure 8161]

[Figure 8162]

[Figure 8163]

[Figure 8164]

[Figure 8165]

[Figure 8166]

[Figure 8167]

[Figure 8168]

[Figure 8169]

[Figure 8170]

[Figure 8171]

[Figure 8179]

[Figure 8180]

[Figure 8181]

[Figure 8182]

[Figure 8183]

[Figure 8184]

[Figure 8185]

[Figure 8186]

[Figure 8187]

[Figure 8188]

[Figure 8189]

[Figure 8199]

[Figure 8200]

[Figure 8201]

[Figure 8202]

[Figure 8203]

[Figure 8204]

[Figure 8205]

[Figure 8206]

[Figure 8207]

[Figure 8208]

[Figure 8219]

[Figure 8220]

[Figure 8221]

[Figure 8222]

[Figure 8223]

[Figure 8224]

[Figure 8225]

[Figure 8226]

[Figure 8227]

[Figure 8228]

[Figure 8229]

[Figure 8230]

[Figure 8231]

[Figure 8232]

[Figure 8233]

[Figure 8234]

[Figure 8235]

[Figure 8236]

[Figure 8237]

[Figure 8238]

[Figure 8239]

[Figure 8240]

[Figure 8241]

[Figure 8242]

[Figure 8243]

[Figure 8244]

[Figure 8245]

###### Rusty Blackbird

[Figure 8259]

[Figure 8260]

[Figure 8261]

[Figure 8262]

[Figure 8263]

[Figure 8264]

[Figure 8265]

[Figure 8266]

[Figure 8267]

[Figure 8268]

[Figure 8269]

[Figure 8270]

[Figure 8279]

[Figure 8280]

[Figure 8281]

[Figure 8282]

[Figure 8283]

[Figure 8284]

[Figure 8285]

[Figure 8286]

[Figure 8287]

[Figure 8288]

[Figure 8289]

[Figure 8290]

[Figure 8291]

[Figure 8299]

[Figure 8300]

[Figure 8301]

[Figure 8302]

[Figure 8303]

[Figure 8304]

[Figure 8305]

[Figure 8306]

[Figure 8307]

[Figure 8308]

[Figure 8309]

[Figure 8319]

[Figure 8320]

[Figure 8321]

[Figure 8322]

[Figure 8323]

[Figure 8324]

[Figure 8325]

[Figure 8326]

[Figure 8327]

[Figure 8328]

[Figure 8329]

[Figure 8330]

[Figure 8339]

[Figure 8340]

[Figure 8341]

[Figure 8342]

[Figure 8343]

[Figure 8344]

[Figure 8345]

[Figure 8346]

[Figure 8347]

[Figure 8348]

[Figure 8349]

[Figure 8350]

[Figure 8351]

[Figure 8359]

[Figure 8360]

[Figure 8361]

[Figure 8362]

[Figure 8363]

[Figure 8364]

[Figure 8365]

[Figure 8366]

[Figure 8367]

[Figure 8368]

[Figure 8369]

[Figure 8370]

[Figure 8379]

[Figure 8380]

[Figure 8381]

[Figure 8382]

[Figure 8383]

[Figure 8384]

[Figure 8385]

[Figure 8386]

[Figure 8387]

[Figure 8388]

[Figure 8389]

[Figure 8399]

[Figure 8400]

[Figure 8401]

[Figure 8402]

[Figure 8403]

[Figure 8404]

[Figure 8405]

[Figure 8406]

[Figure 8407]

[Figure 8408]

[Figure 8419]

[Figure 8420]

[Figure 8421]

[Figure 8422]

[Figure 8423]

[Figure 8424]

[Figure 8425]

[Figure 8426]

[Figure 8427]

[Figure 8428]

[Figure 8429]

[Figure 8430]

[Figure 8431]

[Figure 8432]

[Figure 8433]

[Figure 8434]

[Figure 8435]

[Figure 8436]

[Figure 8437]

[Figure 8438]

[Figure 8439]

[Figure 8440]

[Figure 8441]

[Figure 8442]

[Figure 8443]

[Figure 8444]

[Figure 8445]

[Figure 8459]

[Figure 8460]

[Figure 8461]

[Figure 8462]

[Figure 8463]

[Figure 8464]

[Figure 8465]

[Figure 8466]

[Figure 8467]

[Figure 8468]

[Figure 8469]

[Figure 8470]

[Figure 8479]

[Figure 8480]

[Figure 8481]

[Figure 8482]

[Figure 8483]

[Figure 8484]

[Figure 8485]

[Figure 8486]

[Figure 8487]

[Figure 8488]

[Figure 8489]

[Figure 8490]

[Figure 8491]

[Figure 8499]

[Figure 8500]

[Figure 8501]

[Figure 8502]

[Figure 8503]

[Figure 8504]

[Figure 8505]

[Figure 8506]

[Figure 8507]

[Figure 8508]

[Figure 8509]

[Figure 8519]

[Figure 8520]

[Figure 8521]

[Figure 8522]

[Figure 8523]

[Figure 8524]

[Figure 8525]

[Figure 8526]

[Figure 8527]

[Figure 8528]

[Figure 8529]

[Figure 8530]

[Figure 8539]

[Figure 8540]

[Figure 8541]

[Figure 8542]

[Figure 8543]

[Figure 8544]

[Figure 8545]

[Figure 8546]

[Figure 8547]

[Figure 8548]

[Figure 8549]

[Figure 8550]

[Figure 8551]

[Figure 8559]

[Figure 8560]

[Figure 8561]

[Figure 8562]

[Figure 8563]

[Figure 8564]

[Figure 8565]

[Figure 8566]

[Figure 8567]

[Figure 8568]

[Figure 8569]

[Figure 8579]

[Figure 8580]

[Figure 8581]

[Figure 8582]

[Figure 8583]

[Figure 8584]

[Figure 8585]

[Figure 8586]

[Figure 8587]

[Figure 8588]

[Figure 8599]

[Figure 8600]

[Figure 8601]

[Figure 8602]

[Figure 8603]

[Figure 8604]

[Figure 8605]

[Figure 8606]

[Figure 8607]

[Figure 8608]

[Figure 8609]

[Figure 8610]

- Figure 6. PACE’s dataset-level conceptual explanations for classes Black-footed Albatross, Crested Auklet, and Rusty Blackbird in the CUB dataset. For each class, we show PACE’s top 3 dataset-level concepts; for each Concept k, we show the top 5 patches with their associated embeddings emj closest to the concept center µk.

[Figure 8619]

[Figure 8620]

[Figure 8621]

[Figure 8622]

[Figure 8623]

[Figure 8624]

[Figure 8625]

[Figure 8626]

[Figure 8627]

[Figure 8628]

[Figure 8629]

[Figure 8630]

[Figure 8631]

[Figure 8639]

[Figure 8640]

[Figure 8641]

[Figure 8642]

[Figure 8643]

[Figure 8644]

[Figure 8645]

[Figure 8646]

[Figure 8647]

[Figure 8648]

[Figure 8649]

[Figure 8659]

[Figure 8660]

[Figure 8661]

[Figure 8662]

[Figure 8663]

[Figure 8664]

[Figure 8665]

[Figure 8666]

[Figure 8667]

[Figure 8668]

[Figure 8669]

[Figure 8670]

[Figure 8671]

[Figure 8679]

[Figure 8680]

[Figure 8681]

[Figure 8682]

[Figure 8683]

[Figure 8684]

[Figure 8685]

[Figure 8686]

[Figure 8687]

[Figure 8688]

[Figure 8689]

[Figure 8699]

[Figure 8700]

[Figure 8701]

[Figure 8702]

[Figure 8703]

[Figure 8704]

[Figure 8705]

[Figure 8706]

[Figure 8707]

[Figure 8708]

[Figure 8709]

[Figure 8710]

[Figure 8711]

[Figure 8719]

[Figure 8720]

[Figure 8721]

[Figure 8722]

[Figure 8723]

[Figure 8724]

[Figure 8725]

[Figure 8726]

[Figure 8727]

[Figure 8728]

[Figure 8729]

[Figure 8739]

[Figure 8740]

[Figure 8741]

[Figure 8742]

[Figure 8743]

[Figure 8744]

[Figure 8745]

[Figure 8746]

[Figure 8747]

[Figure 8748]

[Figure 8749]

[Figure 8750]

[Figure 8751]

[Figure 8759]

[Figure 8760]

[Figure 8761]

[Figure 8762]

[Figure 8763]

[Figure 8764]

[Figure 8765]

[Figure 8766]

[Figure 8767]

[Figure 8768]

[Figure 8769]

[Figure 8779]

[Figure 8780]

[Figure 8781]

[Figure 8782]

[Figure 8783]

[Figure 8784]

[Figure 8785]

[Figure 8786]

[Figure 8787]

[Figure 8788]

[Figure 8789]

[Figure 8790]

[Figure 8791]

[Figure 8799]

[Figure 8800]

[Figure 8801]

[Figure 8802]

[Figure 8803]

[Figure 8804]

[Figure 8805]

[Figure 8806]

[Figure 8807]

[Figure 8808]

[Figure 8809]

concept activation possesses meaningful semantics. In contrast, other baselines’ number of concepts K is constrained to the hidden dimension of ViT embeddings, i.e., K = 768; they are therefore lacking in parsimony.

[Figure 8819]

[Figure 8820]

[Figure 8821]

[Figure 8822]

[Figure 8823]

[Figure 8824]

[Figure 8825]

[Figure 8826]

[Figure 8827]

[Figure 8828]

[Figure 8829]

[Figure 8830]

[Figure 8831]

[Figure 8839]

[Figure 8840]

[Figure 8841]

[Figure 8842]

[Figure 8843]

[Figure 8844]

[Figure 8845]

[Figure 8846]

[Figure 8847]

[Figure 8848]

[Figure 8849]

Details on the Qualitative Analysis. For qualitative analysis, we visualize 2 × 2 aggregated patches, chosen for their visibility and robustness against random noise. The aggregation only affects patch-level concepts, computed similarly to z¯m in Eq. 1. The mean of ϕmj approximates the patch-level concept for each aggregated patch ϕmj, computed as follows:

2 +v = 14(ϕm,2u·S+2v + ϕm,2u·S+2v+1 + ϕm,(2u+1)·S+2v + ϕm,(2u+1)·S+2v+1), (15)

ϕm,u·S

where S is the number of rows and columns. Note that this aggregation is for visualization purposes only during qualitative analysis and does not affect the quantitative results, dataset-level and image-level concepts, or the learning process.

### B. Sparsity versus Parsimony

As we discuss in Definition 3.1, sparsity is defined as the proportion of θm’s entries nearing zero, i.e., K1 Kk=1 1(|θmk| < ϵ), with a small threshold ϵ > 0; parsimony is defined as the minimal number of concepts K to produce clear and simple

explanations. While sparsity is an image-level property, parsimony is dataset-level.

- Example 1. For example, first consider a dataset with four concepts and three images, I1, I2, and I3:

- • On the image level, I1: θ1 = (1,0,0,0). I2: θ2 = (0,1,0,0). I3: θ1 = (0,0,1,0).
- • On the dataset level, µm = em(1 ≤ m ≤ 3), µ4 = 31(µ1 + µ2 + µ3).

According to Definition 3.1, these image-level concepts satisfies sparsity; however, the dataset-level concept does not satisfy parsimony, since the last concept center µ4 is redundant.

- Example 2. Next, we consider a dataset with three concepts and three images, I1, I2, and I3:

- • On the image level, I1: θ1 = (0.5,0.5,0). I2: θ2 = (0.5,0,0.5). I3: θ1 = (0,0.5,0.5).
- • On the dataset level, µ1 = 12(e1 + e2), µ2 = 12(e1 + e3), and µ1 = 21(e2 + e3).

According to Definition 3.1, These image-level concepts apparently does not satisfy sparsity; however, the dataset-level concept satisfies parsimony, because there are no redundant concepts. Therefore, in our paper, sparsity and parsimony, though related, are distinct and non-interchangeable properties.

[Figure 9179]

[Figure 9180]

[Figure 9181]

[Figure 9182]

[Figure 9183]

[Figure 9184]

[Figure 9185]

[Figure 9186]

[Figure 9187]

[Figure 9188]

[Figure 9189]

[Figure 9190]

[Figure 9191]

[Figure 9192]

[Figure 9193]

[Figure 9194]

[Figure 9195]

[Figure 9196]

[Figure 9197]

[Figure 9198]

[Figure 9199]

[Figure 9200]

[Figure 9201]

[Figure 9202]

[Figure 9203]

[Figure 9204]

[Figure 9205]

[Figure 9206]

[Figure 9207]

[Figure 9208]

[Figure 9209]

[Figure 9210]

[Figure 9211]

[Figure 9212]

[Figure 9213]

[Figure 9214]

[Figure 9215]

[Figure 9216]

[Figure 9217]

[Figure 9218]

[Figure 9219]

[Figure 9220]

[Figure 9221]

[Figure 9222]

[Figure 9223]

[Figure 9224]

[Figure 9225]

[Figure 9226]

[Figure 9227]

[Figure 9228]

[Figure 9229]

[Figure 9230]

[Figure 9231]

[Figure 9232]

[Figure 9233]

[Figure 9234]

[Figure 9235]

[Figure 9236]

[Figure 9237]

[Figure 9238]

[Figure 9239]

[Figure 9240]

[Figure 9241]

[Figure 9242]

[Figure 9243]

[Figure 9244]

[Figure 9245]

[Figure 9246]

[Figure 9247]

[Figure 9248]

[Figure 9249]

[Figure 9250]

[Figure 9251]

[Figure 9252]

[Figure 9253]

[Figure 9254]

[Figure 9255]

[Figure 9256]

[Figure 9257]

[Figure 9279]

[Figure 9280]

[Figure 9281]

[Figure 9282]

[Figure 9283]

[Figure 9284]

[Figure 9285]

[Figure 9286]

[Figure 9287]

[Figure 9288]

[Figure 9289]

[Figure 9290]

[Figure 9291]

[Figure 9292]

[Figure 9293]

[Figure 9294]

[Figure 9295]

[Figure 9296]

[Figure 9297]

[Figure 9298]

[Figure 9299]

[Figure 9300]

[Figure 9301]

[Figure 9302]

[Figure 9303]

[Figure 9304]

[Figure 9319]

[Figure 9320]

[Figure 9321]

[Figure 9322]

[Figure 9323]

[Figure 9324]

[Figure 9325]

[Figure 9326]

[Figure 9327]

[Figure 9328]

[Figure 9329]

[Figure 9339]

[Figure 9340]

[Figure 9341]

[Figure 9342]

[Figure 9343]

[Figure 9344]

[Figure 9345]

[Figure 9346]

[Figure 9347]

[Figure 9348]

[Figure 9349]

[Figure 9359]

[Figure 9360]

[Figure 9361]

[Figure 9362]

[Figure 9363]

[Figure 9364]

[Figure 9365]

[Figure 9366]

[Figure 9367]

[Figure 9368]

[Figure 9369]

[Figure 9379]

[Figure 9380]

[Figure 9381]

[Figure 9382]

[Figure 9383]

[Figure 9384]

[Figure 9385]

[Figure 9386]

[Figure 9387]

[Figure 9388]

[Figure 9389]

[Figure 9390]

[Figure 9399]

[Figure 9400]

[Figure 9401]

[Figure 9402]

[Figure 9403]

[Figure 9404]

[Figure 9405]

[Figure 9406]

[Figure 9407]

[Figure 9408]

[Figure 9409]

[Figure 9410]

[Figure 9411]

[Figure 9412]

[Figure 9413]

[Figure 9414]

[Figure 9415]

[Figure 9416]

[Figure 9417]

[Figure 9418]

[Figure 9419]

[Figure 9420]

[Figure 9421]

[Figure 9422]

[Figure 9423]

[Figure 9424]

[Figure 9425]

[Figure 9426]

[Figure 9427]

[Figure 9428]

[Figure 9429]

[Figure 9430]

[Figure 9431]

[Figure 9432]

[Figure 9433]

[Figure 9434]

[Figure 9435]

[Figure 9436]

[Figure 9437]

[Figure 9438]

[Figure 9439]

[Figure 9440]

[Figure 9441]

[Figure 9442]

[Figure 9443]

[Figure 9444]

[Figure 9445]

[Figure 9446]

[Figure 9447]

[Figure 9448]

[Figure 9449]

[Figure 9450]

[Figure 9451]

[Figure 9452]

[Figure 9453]

[Figure 9454]

[Figure 9455]

[Figure 9456]

[Figure 9457]

[Figure 9458]

[Figure 9459]

[Figure 9460]

[Figure 9461]

[Figure 9462]

[Figure 9463]

[Figure 9464]

[Figure 9465]

[Figure 9466]

[Figure 9467]

[Figure 9468]

[Figure 9469]

[Figure 9470]

[Figure 9471]

[Figure 9472]

[Figure 9473]

[Figure 9474]

[Figure 9475]

[Figure 9476]

[Figure 9477]

[Figure 9499]

[Figure 9500]

[Figure 9501]

[Figure 9502]

[Figure 9503]

[Figure 9504]

[Figure 9505]

[Figure 9506]

[Figure 9507]

[Figure 9508]

[Figure 9509]

[Figure 9510]

[Figure 9511]

[Figure 9512]

[Figure 9513]

[Figure 9514]

[Figure 9515]

[Figure 9516]

[Figure 9517]

[Figure 9518]

[Figure 9519]

[Figure 9520]

[Figure 9521]

[Figure 9522]

[Figure 9523]

[Figure 9524]

[Figure 9539]

[Figure 9540]

[Figure 9541]

[Figure 9542]

[Figure 9543]

[Figure 9544]

[Figure 9545]

[Figure 9546]

[Figure 9547]

[Figure 9548]

[Figure 9549]

[Figure 9559]

[Figure 9560]

[Figure 9561]

[Figure 9562]

[Figure 9563]

[Figure 9564]

[Figure 9565]

[Figure 9566]

[Figure 9567]

[Figure 9568]

[Figure 9569]

[Figure 9570]

[Figure 9571]

[Figure 9579]

[Figure 9580]

[Figure 9581]

[Figure 9582]

[Figure 9583]

[Figure 9584]

[Figure 9585]

[Figure 9586]

[Figure 9587]

[Figure 9588]

[Figure 9589]

[Figure 9590]

[Figure 9599]

[Figure 9600]

[Figure 9601]

[Figure 9602]

[Figure 9603]

[Figure 9604]

[Figure 9605]

[Figure 9606]

[Figure 9607]

[Figure 9608]

[Figure 9609]

[Figure 9619]

[Figure 9620]

[Figure 9621]

[Figure 9622]

[Figure 9623]

[Figure 9624]

[Figure 9625]

[Figure 9626]

[Figure 9627]

[Figure 9628]

[Figure 9629]

[Figure 9639]

[Figure 9640]

[Figure 9641]

[Figure 9642]

[Figure 9643]

[Figure 9644]

[Figure 9645]

[Figure 9646]

[Figure 9647]

[Figure 9648]

[Figure 9649]

[Figure 9659]

[Figure 9660]

[Figure 9661]

[Figure 9662]

[Figure 9663]

[Figure 9664]

[Figure 9665]

[Figure 9666]

[Figure 9667]

[Figure 9668]

[Figure 9669]

[Figure 9670]

[Figure 9679]

[Figure 9680]

[Figure 9681]

[Figure 9682]

[Figure 9683]

[Figure 9684]

[Figure 9685]

[Figure 9686]

[Figure 9687]

[Figure 9688]

[Figure 9689]

[Figure 9690]

[Figure 9691]

[Figure 9692]

[Figure 9693]

[Figure 9694]

[Figure 9695]

[Figure 9696]

[Figure 9697]

[Figure 9698]

[Figure 9699]

[Figure 9700]

[Figure 9701]

[Figure 9702]

[Figure 9703]

[Figure 9704]

[Figure 9719]

[Figure 9720]

[Figure 9721]

[Figure 9722]

[Figure 9723]

[Figure 9724]

[Figure 9725]

[Figure 9726]

[Figure 9727]

[Figure 9728]

[Figure 9729]

[Figure 9739]

[Figure 9740]

[Figure 9741]

[Figure 9742]

[Figure 9743]

[Figure 9744]

[Figure 9745]

[Figure 9746]

[Figure 9747]

[Figure 9748]

[Figure 9749]

[Figure 9750]

[Figure 9751]

[Figure 9752]

[Figure 9753]

[Figure 9754]

[Figure 9755]

[Figure 9756]

[Figure 9757]

[Figure 9758]

[Figure 9759]

[Figure 9760]

[Figure 9761]

[Figure 9762]

[Figure 9763]

[Figure 9764]

[Figure 9765]

[Figure 9766]

[Figure 9767]

[Figure 9768]

[Figure 9769]

[Figure 9770]

[Figure 9771]

[Figure 9772]

[Figure 9773]

[Figure 9774]

[Figure 9775]

[Figure 9776]

[Figure 9777]

[Figure 9778]

[Figure 9779]

[Figure 9780]

[Figure 9781]

[Figure 9782]

[Figure 9783]

[Figure 9784]

[Figure 9785]

[Figure 9786]

[Figure 9787]

[Figure 9788]

[Figure 9789]

[Figure 9790]

[Figure 9791]

[Figure 9792]

[Figure 9793]

[Figure 9794]

[Figure 9795]

[Figure 9796]

[Figure 9797]

[Figure 9798]

[Figure 9799]

[Figure 9800]

[Figure 9801]

[Figure 9802]

[Figure 9803]

[Figure 9804]

[Figure 9805]

[Figure 9806]

[Figure 9807]

[Figure 9808]

[Figure 9809]

[Figure 9810]

[Figure 9811]

[Figure 9812]

[Figure 9813]

[Figure 9814]

[Figure 9815]

[Figure 9816]

[Figure 9817]

[Figure 9839]

[Figure 9840]

[Figure 9841]

[Figure 9842]

[Figure 9843]

[Figure 9844]

[Figure 9845]

[Figure 9846]

[Figure 9847]

[Figure 9848]

[Figure 9849]

[Figure 9850]

[Figure 9851]

[Figure 9852]

[Figure 9853]

[Figure 9854]

[Figure 9855]

[Figure 9856]

[Figure 9857]

[Figure 9858]

[Figure 9859]

[Figure 9860]

[Figure 9861]

[Figure 9862]

[Figure 9863]

[Figure 9864]

[Figure 9879]

[Figure 9880]

[Figure 9881]

[Figure 9882]

[Figure 9883]

[Figure 9884]

[Figure 9885]

[Figure 9886]

[Figure 9887]

[Figure 9888]

[Figure 9889]

[Figure 9899]

[Figure 9900]

[Figure 9901]

[Figure 9902]

[Figure 9903]

[Figure 9904]

[Figure 9905]

[Figure 9906]

[Figure 9907]

[Figure 9908]

[Figure 9909]

[Figure 9910]

[Figure 9911]

[Figure 9912]

[Figure 9913]

[Figure 9919]

[Figure 9920]

[Figure 9921]

[Figure 9922]

[Figure 9923]

[Figure 9924]

[Figure 9925]

[Figure 9926]

[Figure 9927]

[Figure 9928]

[Figure 9929]

[Figure 9930]

[Figure 9931]

[Figure 9939]

[Figure 9940]

[Figure 9941]

[Figure 9942]

[Figure 9943]

[Figure 9944]

[Figure 9945]

[Figure 9946]

[Figure 9947]

[Figure 9948]

[Figure 9949]

[Figure 9950]

[Figure 9959]

[Figure 9960]

[Figure 9961]

[Figure 9962]

[Figure 9963]

[Figure 9964]

[Figure 9965]

[Figure 9966]

[Figure 9967]

[Figure 9968]

[Figure 9969]

[Figure 9979]

[Figure 9980]

[Figure 9981]

[Figure 9982]

[Figure 9983]

[Figure 9984]

[Figure 9985]

[Figure 9986]

[Figure 9987]

[Figure 9988]

[Figure 9989]

[Figure 9990]

[Figure 9991]

[Figure 9992]

[Figure 9993]

[Figure 9994]

[Figure 9995]

[Figure 9996]

[Figure 9997]

[Figure 9999]

[Figure 10000]

[Figure 10001]

[Figure 10002]

[Figure 10003]

[Figure 10004]

[Figure 10005]

[Figure 10006]

[Figure 10007]

[Figure 10008]

[Figure 10009]

[Figure 10010]

[Figure 10011]

[Figure 10012]

[Figure 10019]

[Figure 10020]

[Figure 10021]

[Figure 10022]

[Figure 10023]

[Figure 10024]

[Figure 10025]

[Figure 10026]

[Figure 10027]

[Figure 10028]

[Figure 10029]

[Figure 10039]

[Figure 10040]

[Figure 10041]

[Figure 10042]

[Figure 10043]

[Figure 10044]

[Figure 10045]

[Figure 10046]

[Figure 10047]

[Figure 10048]

[Figure 10049]

[Figure 10059]

[Figure 10060]

[Figure 10061]

[Figure 10062]

[Figure 10063]

[Figure 10064]

[Figure 10065]

[Figure 10066]

[Figure 10067]

[Figure 10068]

[Figure 10069]

[Figure 10070]

[Figure 10079]

[Figure 10080]

[Figure 10081]

[Figure 10082]

[Figure 10083]

[Figure 10084]

[Figure 10085]

[Figure 10086]

[Figure 10087]

[Figure 10088]

[Figure 10089]

[Figure 10090]

[Figure 10091]

[Figure 10092]

[Figure 10093]

[Figure 10094]

[Figure 10095]

[Figure 10096]

[Figure 10097]

[Figure 10098]

[Figure 10099]

[Figure 10100]

[Figure 10101]

[Figure 10102]

[Figure 10103]

[Figure 10104]

[Figure 10119]

[Figure 10120]

[Figure 10121]

[Figure 10122]

[Figure 10123]

[Figure 10124]

[Figure 10125]

[Figure 10126]

[Figure 10127]

[Figure 10128]

[Figure 10129]

###### Probabilistic Conceptual Explainers: Trustworthy Conceptual Explanations for Vision Foundation Models

[Figure 10139]

[Figure 10140]

[Figure 10141]

[Figure 10142]

[Figure 10143]

[Figure 10144]

[Figure 10145]

[Figure 10146]

[Figure 10147]

[Figure 10148]

[Figure 10149]

[Figure 10150]

[Figure 10151]

[Figure 10152]

[Figure 10153]

[Figure 10154]

[Figure 10155]

[Figure 10156]

[Figure 10157]

[Figure 10158]

[Figure 10159]

[Figure 10160]

[Figure 10161]

[Figure 10162]

[Figure 10163]

[Figure 10164]

[Figure 10165]

[Figure 10166]

[Figure 10167]

[Figure 10168]

[Figure 10169]

[Figure 10170]

[Figure 10171]

[Figure 10172]

[Figure 10173]

[Figure 10174]

[Figure 10175]

[Figure 10176]

[Figure 10177]

[Figure 10178]

[Figure 10179]

[Figure 10180]

[Figure 10181]

[Figure 10182]

[Figure 10183]

[Figure 10184]

[Figure 10185]

[Figure 10186]

[Figure 10187]

[Figure 10188]

[Figure 10189]

[Figure 10190]

[Figure 10191]

[Figure 10192]

[Figure 10193]

[Figure 10194]

[Figure 10195]

[Figure 10196]

[Figure 10197]

[Figure 10198]

[Figure 10199]

[Figure 10200]

[Figure 10201]

[Figure 10202]

[Figure 10203]

[Figure 10204]

[Figure 10205]

[Figure 10206]

[Figure 10207]

[Figure 10208]

[Figure 10209]

[Figure 10210]

[Figure 10211]

[Figure 10212]

[Figure 10213]

[Figure 10214]

[Figure 10215]

[Figure 10216]

[Figure 10217]

[Figure 10239]

[Figure 10240]

[Figure 10241]

[Figure 10242]

[Figure 10243]

[Figure 10244]

[Figure 10245]

[Figure 10246]

[Figure 10247]

[Figure 10248]

[Figure 10249]

[Figure 10250]

[Figure 10251]

[Figure 10252]

[Figure 10253]

[Figure 10254]

[Figure 10255]

[Figure 10256]

[Figure 10257]

[Figure 10258]

[Figure 10259]

[Figure 10260]

[Figure 10261]

[Figure 10262]

[Figure 10263]

[Figure 10264]

[Figure 10279]

[Figure 10280]

[Figure 10281]

[Figure 10282]

[Figure 10283]

[Figure 10284]

[Figure 10285]

[Figure 10286]

[Figure 10287]

[Figure 10288]

[Figure 10289]

[Figure 10299]

[Figure 10300]

[Figure 10301]

[Figure 10302]

[Figure 10303]

[Figure 10304]

[Figure 10305]

[Figure 10306]

[Figure 10307]

[Figure 10308]

[Figure 10309]

[Figure 10310]

[Figure 10311]

[Figure 10312]

[Figure 10313]

[Figure 10319]

[Figure 10320]

[Figure 10321]

[Figure 10322]

[Figure 10323]

[Figure 10324]

[Figure 10325]

[Figure 10326]

[Figure 10327]

[Figure 10328]

[Figure 10329]

[Figure 10330]

[Figure 10331]

[Figure 10332]

[Figure 10333]

[Figure 10334]

[Figure 10335]

[Figure 10336]

[Figure 10337]

[Figure 10339]

[Figure 10340]

[Figure 10341]

[Figure 10342]

[Figure 10343]

[Figure 10344]

[Figure 10345]

[Figure 10346]

[Figure 10347]

[Figure 10348]

[Figure 10349]

[Figure 10350]

[Figure 10351]

[Figure 10359]

[Figure 10360]

[Figure 10361]

[Figure 10362]

[Figure 10363]

[Figure 10364]

[Figure 10365]

[Figure 10366]

[Figure 10367]

[Figure 10368]

[Figure 10369]

[Figure 10370]

[Figure 10379]

[Figure 10380]

[Figure 10381]

[Figure 10382]

[Figure 10383]

[Figure 10384]

[Figure 10385]

[Figure 10386]

[Figure 10387]

[Figure 10388]

[Figure 10389]

[Figure 10399]

[Figure 10400]

[Figure 10401]

[Figure 10402]

[Figure 10403]

[Figure 10404]

[Figure 10405]

[Figure 10406]

[Figure 10407]

[Figure 10408]

[Figure 10409]

[Figure 10410]

[Figure 10411]

[Figure 10412]

[Figure 10419]

[Figure 10420]

[Figure 10421]

[Figure 10422]

[Figure 10423]

[Figure 10424]

[Figure 10425]

[Figure 10426]

[Figure 10427]

[Figure 10428]

[Figure 10429]

[Figure 10439]

[Figure 10440]

[Figure 10441]

[Figure 10442]

[Figure 10443]

[Figure 10444]

[Figure 10445]

[Figure 10446]

[Figure 10447]

[Figure 10448]

[Figure 10449]

[Figure 10459]

[Figure 10460]

[Figure 10461]

[Figure 10462]

[Figure 10463]

[Figure 10464]

[Figure 10465]

[Figure 10466]

[Figure 10467]

[Figure 10468]

[Figure 10469]

[Figure 10470]

[Figure 10479]

[Figure 10480]

[Figure 10481]

[Figure 10482]

[Figure 10483]

[Figure 10484]

[Figure 10485]

[Figure 10486]

[Figure 10487]

[Figure 10488]

[Figure 10489]

[Figure 10490]

[Figure 10491]

[Figure 10492]

[Figure 10493]

[Figure 10494]

[Figure 10495]

[Figure 10496]

[Figure 10497]

[Figure 10498]

[Figure 10499]

[Figure 10500]

[Figure 10501]

[Figure 10502]

[Figure 10503]

[Figure 10504]

[Figure 10519]

[Figure 10520]

[Figure 10521]

[Figure 10522]

[Figure 10523]

[Figure 10524]

[Figure 10525]

[Figure 10526]

[Figure 10527]

[Figure 10528]

[Figure 10529]

[Figure 10539]

[Figure 10540]

[Figure 10541]

[Figure 10542]

[Figure 10543]

[Figure 10544]

[Figure 10545]

[Figure 10546]

[Figure 10547]

[Figure 10548]

[Figure 10549]

[Figure 10550]

[Figure 10551]

[Figure 10552]

[Figure 10553]

[Figure 10554]

[Figure 10555]

[Figure 10556]

[Figure 10557]

[Figure 10558]

[Figure 10559]

[Figure 10560]

[Figure 10561]

[Figure 10562]

[Figure 10563]

[Figure 10564]

[Figure 10565]

[Figure 10566]

[Figure 10567]

[Figure 10568]

[Figure 10569]

[Figure 10570]

[Figure 10571]

[Figure 10572]

[Figure 10573]

[Figure 10574]

[Figure 10575]

[Figure 10576]

[Figure 10577]

[Figure 10578]

[Figure 10579]

[Figure 10580]

[Figure 10581]

[Figure 10582]

[Figure 10583]

[Figure 10584]

[Figure 10585]

[Figure 10586]

[Figure 10587]

[Figure 10588]

[Figure 10589]

[Figure 10590]

[Figure 10591]

[Figure 10592]

[Figure 10593]

[Figure 10594]

[Figure 10595]

[Figure 10596]

[Figure 10597]

[Figure 10598]

[Figure 10599]

[Figure 10600]

[Figure 10601]

[Figure 10602]

[Figure 10603]

[Figure 10604]

[Figure 10605]

[Figure 10606]

[Figure 10607]

[Figure 10608]

[Figure 10609]

[Figure 10610]

[Figure 10611]

[Figure 10612]

[Figure 10613]

[Figure 10614]

[Figure 10615]

[Figure 10616]

[Figure 10617]

[Figure 10639]

[Figure 10640]

[Figure 10641]

[Figure 10642]

[Figure 10643]

[Figure 10644]

[Figure 10645]

[Figure 10646]

[Figure 10647]

[Figure 10648]

[Figure 10649]

[Figure 10650]

[Figure 10651]

[Figure 10652]

[Figure 10653]

[Figure 10654]

[Figure 10655]

[Figure 10656]

[Figure 10657]

[Figure 10658]

[Figure 10659]

[Figure 10660]

[Figure 10661]

[Figure 10662]

[Figure 10663]

[Figure 10664]

[Figure 10679]

[Figure 10680]

[Figure 10681]

[Figure 10682]

[Figure 10683]

[Figure 10684]

[Figure 10685]

[Figure 10686]

[Figure 10687]

[Figure 10688]

[Figure 10689]

[Figure 10699]

[Figure 10700]

[Figure 10701]

[Figure 10702]

[Figure 10703]

[Figure 10704]

[Figure 10705]

[Figure 10706]

[Figure 10707]

[Figure 10708]

[Figure 10709]

[Figure 10710]

[Figure 10711]

[Figure 10712]

[Figure 10713]

[Figure 10719]

[Figure 10720]

[Figure 10721]

[Figure 10722]

[Figure 10723]

[Figure 10724]

[Figure 10725]

[Figure 10726]

[Figure 10727]

[Figure 10728]

[Figure 10729]

[Figure 10730]

[Figure 10731]

[Figure 10732]

[Figure 10733]

[Figure 10734]

[Figure 10735]

[Figure 10736]

[Figure 10737]

[Figure 10739]

[Figure 10740]

[Figure 10741]

[Figure 10742]

[Figure 10743]

[Figure 10744]

[Figure 10745]

[Figure 10746]

[Figure 10747]

[Figure 10748]

[Figure 10749]

[Figure 10750]

[Figure 10751]

[Figure 10752]

[Figure 10759]

[Figure 10760]

[Figure 10761]

[Figure 10762]

[Figure 10763]

[Figure 10764]

[Figure 10765]

[Figure 10766]

[Figure 10767]

[Figure 10768]

[Figure 10769]

[Figure 10770]

[Figure 10771]

[Figure 10779]

[Figure 10780]

[Figure 10781]

[Figure 10782]

[Figure 10783]

[Figure 10784]

[Figure 10785]

[Figure 10786]

[Figure 10787]

[Figure 10788]

[Figure 10789]

[Figure 10790]

[Figure 10799]

[Figure 10800]

[Figure 10801]

[Figure 10802]

[Figure 10803]

[Figure 10804]

[Figure 10805]

[Figure 10806]

[Figure 10807]

[Figure 10808]

[Figure 10809]

###### Example Images Top Concepts

[Figure 10819]

[Figure 10820]

[Figure 10821]

[Figure 10822]

[Figure 10823]

[Figure 10824]

[Figure 10825]

[Figure 10826]

[Figure 10827]

[Figure 10828]

[Figure 10829]

[Figure 10830]

[Figure 10831]

[Figure 10841]

[Figure 10842]

[Figure 10843]

[Figure 10844]

[Figure 10845]

[Figure 10846]

[Figure 10847]

[Figure 10848]

[Figure 10849]

[Figure 10850]

[Figure 10851]

[Figure 10861]

[Figure 10862]

[Figure 10863]

[Figure 10864]

[Figure 10865]

[Figure 10866]

[Figure 10867]

[Figure 10868]

[Figure 10869]

[Figure 10870]

[Figure 10871]

[Figure 10872]

[Figure 10881]

[Figure 10882]

[Figure 10883]

[Figure 10884]

[Figure 10885]

[Figure 10886]

[Figure 10887]

[Figure 10888]

[Figure 10889]

[Figure 10890]

[Figure 10891]

[Figure 10892]

[Figure 10893]

[Figure 10894]

[Figure 10895]

[Figure 10896]

[Figure 10897]

[Figure 10898]

[Figure 10899]

[Figure 10900]

[Figure 10901]

[Figure 10902]

[Figure 10903]

[Figure 10904]

[Figure 10905]

[Figure 10906]

[Figure 10907]

[Figure 10908]

[Figure 10909]

[Figure 10924]

[Figure 10925]

[Figure 10926]

[Figure 10927]

[Figure 10928]

[Figure 10929]

[Figure 10930]

[Figure 10931]

[Figure 10932]

[Figure 10933]

[Figure 10934]

[Figure 10944]

[Figure 10945]

[Figure 10946]

[Figure 10947]

[Figure 10948]

[Figure 10949]

[Figure 10950]

[Figure 10951]

[Figure 10952]

[Figure 10953]

[Figure 10954]

[Figure 10955]

[Figure 10956]

[Figure 10957]

[Figure 10958]

[Figure 10959]

[Figure 10960]

[Figure 10961]

[Figure 10962]

[Figure 10963]

[Figure 10964]

[Figure 10965]

[Figure 10966]

[Figure 10967]

[Figure 10968]

[Figure 10969]

[Figure 10970]

[Figure 10971]

[Figure 10972]

[Figure 10973]

[Figure 10974]

[Figure 10975]

[Figure 10976]

[Figure 10977]

[Figure 10978]

[Figure 10979]

[Figure 10980]

[Figure 10981]

[Figure 10982]

[Figure 10983]

[Figure 10984]

[Figure 10985]

[Figure 10986]

[Figure 10987]

[Figure 10988]

[Figure 10989]

[Figure 10990]

[Figure 10991]

[Figure 10992]

[Figure 10993]

[Figure 10994]

[Figure 10995]

[Figure 10996]

[Figure 10997]

[Figure 10998]

[Figure 10999]

[Figure 11000]

[Figure 11001]

[Figure 11002]

[Figure 11003]

[Figure 11004]

[Figure 11005]

[Figure 11006]

[Figure 11007]

[Figure 11008]

[Figure 11009]

[Figure 11010]

[Figure 11011]

[Figure 11012]

[Figure 11013]

[Figure 11014]

[Figure 11015]

[Figure 11016]

[Figure 11017]

[Figure 11018]

[Figure 11019]

[Figure 11020]

[Figure 11021]

[Figure 11022]

[Figure 11044]

[Figure 11045]

[Figure 11046]

[Figure 11047]

[Figure 11048]

[Figure 11049]

[Figure 11050]

[Figure 11051]

[Figure 11052]

[Figure 11053]

[Figure 11054]

[Figure 11055]

[Figure 11056]

[Figure 11057]

[Figure 11058]

[Figure 11059]

[Figure 11060]

[Figure 11061]

[Figure 11062]

[Figure 11063]

[Figure 11064]

[Figure 11065]

[Figure 11066]

[Figure 11067]

[Figure 11068]

[Figure 11069]

[Figure 11084]

[Figure 11085]

[Figure 11086]

[Figure 11087]

[Figure 11088]

[Figure 11089]

[Figure 11090]

[Figure 11091]

[Figure 11092]

[Figure 11093]

[Figure 11094]

[Figure 11104]

[Figure 11105]

[Figure 11106]

[Figure 11107]

[Figure 11108]

[Figure 11109]

[Figure 11110]

[Figure 11111]

[Figure 11112]

[Figure 11113]

[Figure 11114]

[Figure 11115]

[Figure 11116]

[Figure 11117]

[Figure 11124]

[Figure 11125]

[Figure 11126]

[Figure 11127]

[Figure 11128]

[Figure 11129]

[Figure 11130]

[Figure 11131]

[Figure 11132]

[Figure 11133]

[Figure 11134]

[Figure 11135]

[Figure 11136]

[Figure 11144]

[Figure 11145]

[Figure 11146]

[Figure 11147]

[Figure 11148]

[Figure 11149]

[Figure 11150]

[Figure 11151]

[Figure 11152]

[Figure 11153]

[Figure 11154]

[Figure 11155]

[Figure 11156]

[Figure 11157]

[Figure 11158]

[Figure 11164]

[Figure 11165]

[Figure 11166]

[Figure 11167]

[Figure 11168]

[Figure 11169]

[Figure 11170]

[Figure 11171]

[Figure 11172]

[Figure 11173]

[Figure 11174]

[Figure 11175]

[Figure 11184]

[Figure 11185]

[Figure 11186]

[Figure 11187]

[Figure 11188]

[Figure 11189]

[Figure 11190]

[Figure 11191]

[Figure 11192]

[Figure 11193]

[Figure 11194]

[Figure 11195]

[Figure 11204]

[Figure 11205]

[Figure 11206]

[Figure 11207]

[Figure 11208]

[Figure 11209]

[Figure 11210]

[Figure 11211]

[Figure 11212]

[Figure 11213]

[Figure 11214]

[Figure 11224]

[Figure 11225]

[Figure 11226]

[Figure 11227]

[Figure 11228]

[Figure 11229]

[Figure 11230]

[Figure 11231]

[Figure 11232]

[Figure 11233]

[Figure 11234]

[Figure 11235]

[Figure 11236]

[Figure 11237]

[Figure 11238]

[Figure 11239]

[Figure 11240]

[Figure 11241]

[Figure 11242]

[Figure 11244]

[Figure 11245]

[Figure 11246]

[Figure 11247]

[Figure 11248]

[Figure 11249]

[Figure 11250]

[Figure 11251]

[Figure 11252]

[Figure 11253]

[Figure 11254]

[Figure 11255]

[Figure 11256]

[Figure 11257]

[Figure 11258]

[Figure 11259]

[Figure 11260]

[Figure 11261]

[Figure 11262]

[Figure 11263]

[Figure 11264]

[Figure 11265]

[Figure 11266]

[Figure 11267]

[Figure 11268]

[Figure 11269]

[Figure 11270]

[Figure 11271]

[Figure 11284]

[Figure 11285]

[Figure 11286]

[Figure 11287]

[Figure 11288]

[Figure 11289]

[Figure 11290]

[Figure 11291]

[Figure 11292]

[Figure 11293]

[Figure 11294]

[Figure 11295]

[Figure 11296]

[Figure 11304]

[Figure 11305]

[Figure 11306]

[Figure 11307]

[Figure 11308]

[Figure 11309]

[Figure 11310]

[Figure 11311]

[Figure 11312]

[Figure 11313]

[Figure 11314]

[Figure 11315]

[Figure 11324]

[Figure 11325]

[Figure 11326]

[Figure 11327]

[Figure 11328]

[Figure 11329]

[Figure 11330]

[Figure 11331]

[Figure 11332]

[Figure 11333]

[Figure 11334]

- Concept 1
- Concept 2
- Concept 3

[Figure 11344]

[Figure 11345]

[Figure 11346]

[Figure 11347]

[Figure 11348]

[Figure 11349]

[Figure 11350]

[Figure 11351]

[Figure 11352]

[Figure 11353]

[Figure 11354]

[Figure 11355]

[Figure 11364]

[Figure 11365]

[Figure 11366]

[Figure 11367]

[Figure 11368]

[Figure 11369]

[Figure 11370]

[Figure 11371]

[Figure 11372]

[Figure 11373]

[Figure 11374]

[Figure 11375]

[Figure 11376]

[Figure 11377]

[Figure 11378]

[Figure 11379]

[Figure 11380]

[Figure 11381]

[Figure 11382]

[Figure 11383]

[Figure 11384]

[Figure 11385]

[Figure 11386]

[Figure 11387]

[Figure 11388]

[Figure 11389]

[Figure 11390]

[Figure 11391]

[Figure 11392]

[Figure 11393]

[Figure 11394]

[Figure 11395]

[Figure 11396]

[Figure 11397]

[Figure 11398]

[Figure 11399]

[Figure 11424]

[Figure 11425]

[Figure 11426]

[Figure 11427]

[Figure 11428]

[Figure 11429]

[Figure 11430]

[Figure 11431]

[Figure 11432]

[Figure 11433]

[Figure 11434]

[Figure 11444]

[Figure 11445]

[Figure 11446]

[Figure 11447]

[Figure 11448]

[Figure 11449]

[Figure 11450]

[Figure 11451]

[Figure 11452]

[Figure 11453]

[Figure 11454]

[Figure 11464]

[Figure 11465]

[Figure 11466]

[Figure 11467]

[Figure 11468]

[Figure 11469]

[Figure 11470]

[Figure 11471]

[Figure 11472]

[Figure 11473]

[Figure 11474]

[Figure 11475]

[Figure 11484]

[Figure 11485]

[Figure 11486]

[Figure 11487]

[Figure 11488]

[Figure 11489]

[Figure 11490]

[Figure 11491]

[Figure 11492]

[Figure 11493]

[Figure 11494]

[Figure 11495]

[Figure 11496]

[Figure 11497]

[Figure 11498]

[Figure 11499]

[Figure 11500]

[Figure 11501]

[Figure 11502]

[Figure 11503]

[Figure 11504]

[Figure 11505]

[Figure 11506]

[Figure 11507]

[Figure 11508]

[Figure 11509]

[Figure 11524]

[Figure 11525]

[Figure 11526]

[Figure 11527]

[Figure 11528]

[Figure 11529]

[Figure 11530]

[Figure 11531]

[Figure 11532]

[Figure 11533]

[Figure 11534]

[Figure 11544]

[Figure 11545]

[Figure 11546]

[Figure 11547]

[Figure 11548]

[Figure 11549]

[Figure 11550]

[Figure 11551]

[Figure 11552]

[Figure 11553]

[Figure 11554]

[Figure 11555]

[Figure 11556]

[Figure 11557]

[Figure 11564]

[Figure 11565]

[Figure 11566]

[Figure 11567]

[Figure 11568]

[Figure 11569]

[Figure 11570]

[Figure 11571]

[Figure 11572]

[Figure 11573]

[Figure 11574]

[Figure 11575]

[Figure 11576]

[Figure 11584]

[Figure 11585]

[Figure 11586]

[Figure 11587]

[Figure 11588]

[Figure 11589]

[Figure 11590]

[Figure 11591]

[Figure 11592]

[Figure 11593]

[Figure 11594]

[Figure 11595]

[Figure 11604]

[Figure 11605]

[Figure 11606]

[Figure 11607]

[Figure 11608]

[Figure 11609]

[Figure 11610]

[Figure 11611]

[Figure 11612]

[Figure 11613]

[Figure 11614]

[Figure 11615]

[Figure 11616]

[Figure 11617]

[Figure 11618]

[Figure 11619]

[Figure 11620]

[Figure 11621]

[Figure 11622]

[Figure 11623]

[Figure 11624]

[Figure 11625]

[Figure 11626]

[Figure 11627]

[Figure 11628]

[Figure 11629]

[Figure 11630]

[Figure 11631]

[Figure 11632]

[Figure 11633]

[Figure 11634]

[Figure 11635]

[Figure 11636]

[Figure 11637]

[Figure 11638]

[Figure 11639]

[Figure 11640]

[Figure 11641]

[Figure 11642]

[Figure 11643]

[Figure 11644]

[Figure 11645]

[Figure 11646]

[Figure 11647]

[Figure 11648]

[Figure 11649]

[Figure 11650]

[Figure 11651]

[Figure 11652]

[Figure 11653]

[Figure 11654]

[Figure 11655]

[Figure 11656]

[Figure 11657]

[Figure 11658]

[Figure 11659]

[Figure 11660]

[Figure 11661]

[Figure 11662]

[Figure 11663]

[Figure 11664]

[Figure 11665]

[Figure 11666]

[Figure 11667]

[Figure 11668]

[Figure 11669]

[Figure 11670]

[Figure 11671]

[Figure 11672]

[Figure 11673]

[Figure 11674]

[Figure 11675]

[Figure 11676]

[Figure 11677]

[Figure 11678]

[Figure 11679]

[Figure 11680]

[Figure 11681]

[Figure 11682]

[Figure 11704]

[Figure 11705]

[Figure 11706]

[Figure 11707]

[Figure 11708]

[Figure 11709]

[Figure 11710]

[Figure 11711]

[Figure 11712]

[Figure 11713]

[Figure 11714]

[Figure 11724]

[Figure 11725]

[Figure 11726]

[Figure 11727]

[Figure 11728]

[Figure 11729]

[Figure 11730]

[Figure 11731]

[Figure 11732]

[Figure 11733]

[Figure 11734]

[Figure 11735]

[Figure 11744]

[Figure 11745]

[Figure 11746]

[Figure 11747]

[Figure 11748]

[Figure 11749]

[Figure 11750]

[Figure 11751]

[Figure 11752]

[Figure 11753]

[Figure 11754]

[Figure 11755]

[Figure 11756]

[Figure 11757]

[Figure 11758]

[Figure 11759]

[Figure 11760]

[Figure 11761]

[Figure 11762]

[Figure 11763]

[Figure 11764]

[Figure 11765]

[Figure 11766]

[Figure 11767]

[Figure 11768]

[Figure 11769]

[Figure 11784]

[Figure 11785]

[Figure 11786]

[Figure 11787]

[Figure 11788]

[Figure 11789]

[Figure 11790]

[Figure 11791]

[Figure 11792]

[Figure 11793]

[Figure 11794]

[Figure 11804]

[Figure 11805]

[Figure 11806]

[Figure 11807]

[Figure 11808]

[Figure 11809]

[Figure 11810]

[Figure 11811]

[Figure 11812]

[Figure 11813]

[Figure 11814]

[Figure 11815]

[Figure 11816]

[Figure 11817]

[Figure 11818]

[Figure 11819]

[Figure 11820]

[Figure 11821]

[Figure 11822]

[Figure 11823]

[Figure 11824]

[Figure 11825]

[Figure 11826]

[Figure 11827]

[Figure 11828]

[Figure 11829]

[Figure 11830]

[Figure 11831]

[Figure 11844]

[Figure 11845]

[Figure 11846]

[Figure 11847]

[Figure 11848]

[Figure 11849]

[Figure 11850]

[Figure 11851]

[Figure 11852]

[Figure 11853]

[Figure 11854]

[Figure 11855]

[Figure 11856]

[Figure 11857]

[Figure 11858]

[Figure 11864]

[Figure 11865]

[Figure 11866]

[Figure 11867]

[Figure 11868]

[Figure 11869]

[Figure 11870]

[Figure 11871]

[Figure 11872]

[Figure 11873]

[Figure 11874]

[Figure 11875]

[Figure 11876]

[Figure 11877]

[Figure 11878]

[Figure 11879]

[Figure 11880]

[Figure 11881]

[Figure 11882]

Acura TL Sedan 2012

[Figure 11884]

[Figure 11885]

[Figure 11886]

[Figure 11887]

[Figure 11888]

[Figure 11889]

[Figure 11890]

[Figure 11891]

[Figure 11892]

[Figure 11893]

[Figure 11894]

[Figure 11895]

[Figure 11896]

[Figure 11904]

[Figure 11905]

[Figure 11906]

[Figure 11907]

[Figure 11908]

[Figure 11909]

[Figure 11910]

[Figure 11911]

[Figure 11912]

[Figure 11913]

[Figure 11914]

[Figure 11915]

[Figure 11924]

[Figure 11925]

[Figure 11926]

[Figure 11927]

[Figure 11928]

[Figure 11929]

[Figure 11930]

[Figure 11931]

[Figure 11932]

[Figure 11933]

[Figure 11934]

[Figure 11944]

[Figure 11945]

[Figure 11946]

[Figure 11947]

[Figure 11948]

[Figure 11949]

[Figure 11950]

[Figure 11951]

[Figure 11952]

[Figure 11953]

[Figure 11954]

[Figure 11955]

[Figure 11964]

[Figure 11965]

[Figure 11966]

[Figure 11967]

[Figure 11968]

[Figure 11969]

[Figure 11970]

[Figure 11971]

[Figure 11972]

[Figure 11973]

[Figure 11974]

[Figure 11975]

[Figure 11976]

[Figure 11977]

[Figure 11978]

[Figure 11979]

[Figure 11980]

[Figure 11981]

[Figure 11982]

[Figure 11983]

[Figure 11984]

[Figure 11985]

[Figure 11986]

[Figure 11987]

[Figure 11988]

[Figure 11989]

[Figure 11990]

[Figure 11991]

[Figure 11992]

[Figure 11993]

[Figure 11994]

[Figure 11995]

[Figure 11996]

[Figure 11997]

[Figure 11998]

[Figure 11999]

[Figure 12024]

[Figure 12025]

[Figure 12026]

[Figure 12027]

[Figure 12028]

[Figure 12029]

[Figure 12030]

[Figure 12031]

[Figure 12032]

[Figure 12033]

[Figure 12034]

[Figure 12044]

[Figure 12045]

[Figure 12046]

[Figure 12047]

[Figure 12048]

[Figure 12049]

[Figure 12050]

[Figure 12051]

[Figure 12052]

[Figure 12053]

[Figure 12054]

[Figure 12064]

[Figure 12065]

[Figure 12066]

[Figure 12067]

[Figure 12068]

[Figure 12069]

[Figure 12070]

[Figure 12071]

[Figure 12072]

[Figure 12073]

[Figure 12074]

[Figure 12075]

[Figure 12084]

[Figure 12085]

[Figure 12086]

[Figure 12087]

[Figure 12088]

[Figure 12089]

[Figure 12090]

[Figure 12091]

[Figure 12092]

[Figure 12093]

[Figure 12094]

[Figure 12095]

[Figure 12096]

[Figure 12097]

[Figure 12098]

[Figure 12099]

[Figure 12100]

[Figure 12101]

[Figure 12102]

[Figure 12103]

[Figure 12104]

[Figure 12105]

[Figure 12106]

[Figure 12107]

[Figure 12108]

[Figure 12109]

[Figure 12124]

[Figure 12125]

[Figure 12126]

[Figure 12127]

[Figure 12128]

[Figure 12129]

[Figure 12130]

[Figure 12131]

[Figure 12132]

[Figure 12133]

[Figure 12134]

[Figure 12144]

[Figure 12145]

[Figure 12146]

[Figure 12147]

[Figure 12148]

[Figure 12149]

[Figure 12150]

[Figure 12151]

[Figure 12152]

[Figure 12153]

[Figure 12154]

[Figure 12155]

[Figure 12156]

[Figure 12157]

[Figure 12164]

[Figure 12165]

[Figure 12166]

[Figure 12167]

[Figure 12168]

[Figure 12169]

[Figure 12170]

[Figure 12171]

[Figure 12172]

[Figure 12173]

[Figure 12174]

[Figure 12175]

[Figure 12184]

[Figure 12185]

[Figure 12186]

[Figure 12187]

[Figure 12188]

[Figure 12189]

[Figure 12190]

[Figure 12191]

[Figure 12192]

[Figure 12193]

[Figure 12194]

[Figure 12195]

[Figure 12196]

[Figure 12204]

[Figure 12205]

[Figure 12206]

[Figure 12207]

[Figure 12208]

[Figure 12209]

[Figure 12210]

[Figure 12211]

[Figure 12212]

[Figure 12213]

[Figure 12214]

[Figure 12215]

[Figure 12216]

[Figure 12217]

[Figure 12218]

[Figure 12219]

[Figure 12220]

[Figure 12221]

[Figure 12222]

[Figure 12223]

[Figure 12224]

[Figure 12225]

[Figure 12226]

[Figure 12227]

[Figure 12228]

[Figure 12229]

[Figure 12230]

[Figure 12231]

[Figure 12244]

[Figure 12245]

[Figure 12246]

[Figure 12247]

[Figure 12248]

[Figure 12249]

[Figure 12250]

[Figure 12251]

[Figure 12252]

[Figure 12253]

[Figure 12254]

[Figure 12264]

[Figure 12265]

[Figure 12266]

[Figure 12267]

[Figure 12268]

[Figure 12269]

[Figure 12270]

[Figure 12271]

[Figure 12272]

[Figure 12273]

[Figure 12274]

[Figure 12275]

[Figure 12284]

[Figure 12285]

[Figure 12286]

[Figure 12287]

[Figure 12288]

[Figure 12289]

[Figure 12290]

[Figure 12291]

[Figure 12292]

[Figure 12293]

[Figure 12294]

[Figure 12295]

[Figure 12296]

[Figure 12297]

[Figure 12298]

[Figure 12299]

[Figure 12300]

[Figure 12301]

[Figure 12302]

[Figure 12303]

[Figure 12304]

[Figure 12305]

[Figure 12306]

[Figure 12307]

[Figure 12308]

[Figure 12309]

[Figure 12310]

[Figure 12311]

[Figure 12312]

[Figure 12313]

[Figure 12314]

[Figure 12315]

[Figure 12316]

[Figure 12317]

[Figure 12318]

[Figure 12319]

[Figure 12320]

[Figure 12321]

[Figure 12322]

[Figure 12323]

[Figure 12324]

[Figure 12325]

[Figure 12326]

[Figure 12327]

[Figure 12328]

[Figure 12329]

[Figure 12330]

[Figure 12331]

[Figure 12332]

[Figure 12333]

[Figure 12334]

[Figure 12335]

[Figure 12336]

[Figure 12337]

[Figure 12338]

[Figure 12339]

[Figure 12340]

[Figure 12341]

[Figure 12342]

[Figure 12343]

[Figure 12344]

[Figure 12345]

[Figure 12346]

[Figure 12347]

[Figure 12348]

[Figure 12349]

[Figure 12350]

[Figure 12351]

[Figure 12352]

[Figure 12353]

[Figure 12354]

[Figure 12355]

[Figure 12356]

[Figure 12357]

[Figure 12358]

[Figure 12359]

[Figure 12360]

[Figure 12361]

[Figure 12362]

[Figure 12384]

[Figure 12385]

[Figure 12386]

[Figure 12387]

[Figure 12388]

[Figure 12389]

[Figure 12390]

[Figure 12391]

[Figure 12392]

[Figure 12393]

[Figure 12394]

[Figure 12395]

[Figure 12396]

[Figure 12397]

[Figure 12398]

[Figure 12399]

[Figure 12400]

[Figure 12401]

[Figure 12402]

[Figure 12403]

[Figure 12404]

[Figure 12405]

[Figure 12406]

[Figure 12407]

[Figure 12408]

[Figure 12409]

[Figure 12424]

[Figure 12425]

[Figure 12426]

[Figure 12427]

[Figure 12428]

[Figure 12429]

[Figure 12430]

[Figure 12431]

[Figure 12432]

[Figure 12433]

[Figure 12434]

[Figure 12444]

[Figure 12445]

[Figure 12446]

[Figure 12447]

[Figure 12448]

[Figure 12449]

[Figure 12450]

[Figure 12451]

[Figure 12452]

[Figure 12453]

[Figure 12454]

[Figure 12455]

[Figure 12456]

[Figure 12457]

[Figure 12458]

[Figure 12464]

[Figure 12465]

[Figure 12466]

[Figure 12467]

[Figure 12468]

[Figure 12469]

[Figure 12470]

[Figure 12471]

[Figure 12472]

[Figure 12473]

[Figure 12474]

[Figure 12475]

[Figure 12476]

[Figure 12477]

[Figure 12478]

[Figure 12479]

[Figure 12480]

[Figure 12481]

[Figure 12482]

[Figure 12484]

[Figure 12485]

[Figure 12486]

[Figure 12487]

[Figure 12488]

[Figure 12489]

[Figure 12490]

[Figure 12491]

[Figure 12492]

[Figure 12493]

[Figure 12494]

[Figure 12495]

[Figure 12496]

[Figure 12504]

[Figure 12505]

[Figure 12506]

[Figure 12507]

[Figure 12508]

[Figure 12509]

[Figure 12510]

[Figure 12511]

[Figure 12512]

[Figure 12513]

[Figure 12514]

[Figure 12515]

[Figure 12524]

[Figure 12525]

[Figure 12526]

[Figure 12527]

[Figure 12528]

[Figure 12529]

[Figure 12530]

[Figure 12531]

[Figure 12532]

[Figure 12533]

[Figure 12534]

[Figure 12544]

[Figure 12545]

[Figure 12546]

[Figure 12547]

[Figure 12548]

[Figure 12549]

[Figure 12550]

[Figure 12551]

[Figure 12552]

[Figure 12553]

[Figure 12554]

[Figure 12555]

[Figure 12564]

[Figure 12565]

[Figure 12566]

[Figure 12567]

[Figure 12568]

[Figure 12569]

[Figure 12570]

[Figure 12571]

[Figure 12572]

[Figure 12573]

[Figure 12574]

[Figure 12575]

[Figure 12576]

[Figure 12577]

[Figure 12578]

[Figure 12579]

[Figure 12580]

[Figure 12581]

[Figure 12582]

[Figure 12583]

[Figure 12584]

[Figure 12585]

[Figure 12586]

[Figure 12587]

[Figure 12588]

[Figure 12589]

[Figure 12590]

[Figure 12591]

[Figure 12592]

[Figure 12593]

[Figure 12594]

[Figure 12595]

[Figure 12596]

[Figure 12597]

[Figure 12598]

[Figure 12599]

[Figure 12624]

[Figure 12625]

[Figure 12626]

[Figure 12627]

[Figure 12628]

[Figure 12629]

[Figure 12630]

[Figure 12631]

[Figure 12632]

[Figure 12633]

[Figure 12634]

[Figure 12644]

[Figure 12645]

[Figure 12646]

[Figure 12647]

[Figure 12648]

[Figure 12649]

[Figure 12650]

[Figure 12651]

[Figure 12652]

[Figure 12653]

[Figure 12654]

[Figure 12664]

[Figure 12665]

[Figure 12666]

[Figure 12667]

[Figure 12668]

[Figure 12669]

[Figure 12670]

[Figure 12671]

[Figure 12672]

[Figure 12673]

[Figure 12674]

[Figure 12675]

[Figure 12684]

[Figure 12685]

[Figure 12686]

[Figure 12687]

[Figure 12688]

[Figure 12689]

[Figure 12690]

[Figure 12691]

[Figure 12692]

[Figure 12693]

[Figure 12694]

[Figure 12695]

[Figure 12696]

[Figure 12697]

[Figure 12698]

[Figure 12699]

[Figure 12700]

[Figure 12701]

[Figure 12702]

[Figure 12703]

[Figure 12704]

[Figure 12705]

[Figure 12706]

[Figure 12707]

[Figure 12708]

[Figure 12709]

[Figure 12724]

[Figure 12725]

[Figure 12726]

[Figure 12727]

[Figure 12728]

[Figure 12729]

[Figure 12730]

[Figure 12731]

[Figure 12732]

[Figure 12733]

[Figure 12734]

[Figure 12744]

[Figure 12745]

[Figure 12746]

[Figure 12747]

[Figure 12748]

[Figure 12749]

[Figure 12750]

[Figure 12751]

[Figure 12752]

[Figure 12753]

[Figure 12754]

[Figure 12755]

[Figure 12756]

[Figure 12757]

[Figure 12758]

[Figure 12759]

[Figure 12760]

[Figure 12761]

[Figure 12762]

[Figure 12769]

[Figure 12770]

[Figure 12771]

[Figure 12772]

[Figure 12773]

[Figure 12774]

[Figure 12775]

[Figure 12776]

[Figure 12777]

[Figure 12778]

[Figure 12779]

[Figure 12780]

[Figure 12781]

[Figure 12782]

[Figure 12783]

[Figure 12784]

[Figure 12785]

[Figure 12786]

[Figure 12787]

[Figure 12788]

[Figure 12789]

[Figure 12790]

[Figure 12791]

[Figure 12792]

[Figure 12793]

[Figure 12794]

[Figure 12795]

[Figure 12796]

[Figure 12809]

[Figure 12810]

[Figure 12811]

[Figure 12812]

[Figure 12813]

[Figure 12814]

[Figure 12815]

[Figure 12816]

[Figure 12817]

[Figure 12818]

[Figure 12819]

[Figure 12820]

[Figure 12829]

[Figure 12830]

[Figure 12831]

[Figure 12832]

[Figure 12833]

[Figure 12834]

[Figure 12835]

[Figure 12836]

[Figure 12837]

[Figure 12838]

[Figure 12839]

[Figure 12840]

[Figure 12841]

[Figure 12849]

[Figure 12850]

[Figure 12851]

[Figure 12852]

[Figure 12853]

[Figure 12854]

[Figure 12855]

[Figure 12856]

[Figure 12857]

[Figure 12858]

[Figure 12859]

[Figure 12869]

[Figure 12870]

[Figure 12871]

[Figure 12872]

[Figure 12873]

[Figure 12874]

[Figure 12875]

[Figure 12876]

[Figure 12877]

[Figure 12878]

[Figure 12879]

[Figure 12880]

[Figure 12889]

[Figure 12890]

[Figure 12891]

[Figure 12892]

[Figure 12893]

[Figure 12894]

[Figure 12895]

[Figure 12896]

[Figure 12897]

[Figure 12898]

[Figure 12899]

[Figure 12900]

[Figure 12901]

[Figure 12902]

[Figure 12903]

[Figure 12909]

[Figure 12910]

[Figure 12911]

[Figure 12912]

[Figure 12913]

[Figure 12914]

[Figure 12915]

[Figure 12916]

[Figure 12917]

[Figure 12918]

[Figure 12919]

[Figure 12920]

[Figure 12921]

[Figure 12922]

[Figure 12923]

[Figure 12924]

[Figure 12925]

[Figure 12926]

[Figure 12927]

[Figure 12929]

[Figure 12930]

[Figure 12931]

[Figure 12932]

[Figure 12933]

[Figure 12934]

[Figure 12935]

[Figure 12936]

[Figure 12937]

[Figure 12938]

[Figure 12939]

[Figure 12940]

[Figure 12941]

[Figure 12949]

[Figure 12950]

[Figure 12951]

[Figure 12952]

[Figure 12953]

[Figure 12954]

[Figure 12955]

[Figure 12956]

[Figure 12957]

[Figure 12958]

[Figure 12959]

[Figure 12960]

[Figure 12969]

[Figure 12970]

[Figure 12971]

[Figure 12972]

[Figure 12973]

[Figure 12974]

[Figure 12975]

[Figure 12976]

[Figure 12977]

[Figure 12978]

[Figure 12979]

[Figure 12989]

[Figure 12990]

[Figure 12991]

[Figure 12992]

[Figure 12993]

[Figure 12994]

[Figure 12995]

[Figure 12996]

[Figure 12997]

[Figure 12998]

[Figure 12999]

[Figure 13000]

[Figure 13009]

[Figure 13010]

[Figure 13011]

[Figure 13012]

[Figure 13013]

[Figure 13014]

[Figure 13015]

[Figure 13016]

[Figure 13017]

[Figure 13018]

[Figure 13019]

[Figure 13020]

[Figure 13021]

[Figure 13022]

[Figure 13023]

[Figure 13024]

[Figure 13025]

[Figure 13026]

[Figure 13027]

[Figure 13028]

[Figure 13029]

[Figure 13030]

[Figure 13031]

[Figure 13032]

[Figure 13033]

[Figure 13034]

[Figure 13035]

[Figure 13036]

[Figure 13037]

[Figure 13038]

[Figure 13039]

[Figure 13040]

[Figure 13041]

[Figure 13042]

[Figure 13043]

[Figure 13044]

[Figure 13069]

[Figure 13070]

[Figure 13071]

[Figure 13072]

[Figure 13073]

[Figure 13074]

[Figure 13075]

[Figure 13076]

[Figure 13077]

[Figure 13078]

[Figure 13079]

[Figure 13089]

[Figure 13090]

[Figure 13091]

[Figure 13092]

[Figure 13093]

[Figure 13094]

[Figure 13095]

[Figure 13096]

[Figure 13097]

[Figure 13098]

[Figure 13099]

[Figure 13109]

[Figure 13110]

[Figure 13111]

[Figure 13112]

[Figure 13113]

[Figure 13114]

[Figure 13115]

[Figure 13116]

[Figure 13117]

[Figure 13118]

[Figure 13119]

[Figure 13120]

[Figure 13129]

[Figure 13130]

[Figure 13131]

[Figure 13132]

[Figure 13133]

[Figure 13134]

[Figure 13135]

[Figure 13136]

[Figure 13137]

[Figure 13138]

[Figure 13139]

[Figure 13140]

[Figure 13141]

[Figure 13142]

[Figure 13143]

[Figure 13144]

[Figure 13145]

[Figure 13146]

[Figure 13147]

[Figure 13148]

[Figure 13149]

[Figure 13150]

[Figure 13151]

[Figure 13152]

[Figure 13153]

[Figure 13154]

[Figure 13169]

[Figure 13170]

[Figure 13171]

[Figure 13172]

[Figure 13173]

[Figure 13174]

[Figure 13175]

[Figure 13176]

[Figure 13177]

[Figure 13178]

[Figure 13179]

- Concept 1
- Concept 2
- Concept 3

[Figure 13189]

[Figure 13190]

[Figure 13191]

[Figure 13192]

[Figure 13193]

[Figure 13194]

[Figure 13195]

[Figure 13196]

[Figure 13197]

[Figure 13198]

[Figure 13199]

[Figure 13200]

[Figure 13201]

[Figure 13202]

[Figure 13203]

[Figure 13204]

[Figure 13205]

[Figure 13206]

[Figure 13207]

[Figure 13208]

[Figure 13209]

[Figure 13210]

[Figure 13211]

[Figure 13212]

[Figure 13213]

[Figure 13214]

[Figure 13215]

[Figure 13216]

[Figure 13229]

[Figure 13230]

[Figure 13231]

[Figure 13232]

[Figure 13233]

[Figure 13234]

[Figure 13235]

[Figure 13236]

[Figure 13237]

[Figure 13238]

[Figure 13239]

[Figure 13240]

[Figure 13241]

[Figure 13242]

[Figure 13249]

[Figure 13250]

[Figure 13251]

[Figure 13252]

[Figure 13253]

[Figure 13254]

[Figure 13255]

[Figure 13256]

[Figure 13257]

[Figure 13258]

[Figure 13259]

[Figure 13260]

[Figure 13269]

[Figure 13270]

[Figure 13271]

[Figure 13272]

[Figure 13273]

[Figure 13274]

[Figure 13275]

[Figure 13276]

[Figure 13277]

[Figure 13278]

[Figure 13279]

[Figure 13280]

[Figure 13281]

[Figure 13289]

[Figure 13290]

[Figure 13291]

[Figure 13292]

[Figure 13293]

[Figure 13294]

[Figure 13295]

[Figure 13296]

[Figure 13297]

[Figure 13298]

[Figure 13299]

[Figure 13309]

[Figure 13310]

[Figure 13311]

[Figure 13312]

[Figure 13313]

[Figure 13314]

[Figure 13315]

[Figure 13316]

[Figure 13317]

[Figure 13318]

[Figure 13319]

[Figure 13320]

[Figure 13329]

[Figure 13330]

[Figure 13331]

[Figure 13332]

[Figure 13333]

[Figure 13334]

[Figure 13335]

[Figure 13336]

[Figure 13337]

[Figure 13338]

[Figure 13339]

[Figure 13340]

[Figure 13341]

[Figure 13342]

[Figure 13343]

[Figure 13349]

[Figure 13350]

[Figure 13351]

[Figure 13352]

[Figure 13353]

[Figure 13354]

[Figure 13355]

[Figure 13356]

[Figure 13357]

[Figure 13358]

[Figure 13359]

[Figure 13360]

[Figure 13361]

[Figure 13362]

[Figure 13363]

[Figure 13364]

[Figure 13365]

[Figure 13366]

[Figure 13367]

[Figure 13369]

[Figure 13370]

[Figure 13371]

[Figure 13372]

[Figure 13373]

[Figure 13374]

[Figure 13375]

[Figure 13376]

[Figure 13377]

[Figure 13378]

[Figure 13379]

[Figure 13380]

[Figure 13381]

[Figure 13389]

[Figure 13390]

[Figure 13391]

[Figure 13392]

[Figure 13393]

[Figure 13394]

[Figure 13395]

[Figure 13396]

[Figure 13397]

[Figure 13398]

[Figure 13399]

[Figure 13400]

[Figure 13409]

[Figure 13410]

[Figure 13411]

[Figure 13412]

[Figure 13413]

[Figure 13414]

[Figure 13415]

[Figure 13416]

[Figure 13417]

[Figure 13418]

[Figure 13419]

[Figure 13429]

[Figure 13430]

[Figure 13431]

[Figure 13432]

[Figure 13433]

[Figure 13434]

[Figure 13435]

[Figure 13436]

[Figure 13437]

[Figure 13438]

[Figure 13439]

[Figure 13440]

[Figure 13449]

[Figure 13450]

[Figure 13451]

[Figure 13452]

[Figure 13453]

[Figure 13454]

[Figure 13455]

[Figure 13456]

[Figure 13457]

[Figure 13458]

[Figure 13459]

[Figure 13469]

[Figure 13470]

[Figure 13471]

[Figure 13472]

[Figure 13473]

[Figure 13474]

[Figure 13475]

[Figure 13476]

[Figure 13477]

[Figure 13478]

[Figure 13479]

[Figure 13480]

[Figure 13481]

[Figure 13482]

[Figure 13483]

[Figure 13484]

[Figure 13485]

[Figure 13486]

[Figure 13487]

[Figure 13488]

[Figure 13489]

[Figure 13490]

[Figure 13491]

[Figure 13492]

[Figure 13493]

[Figure 13494]

[Figure 13495]

[Figure 13496]

[Figure 13497]

[Figure 13498]

[Figure 13499]

[Figure 13500]

[Figure 13501]

[Figure 13502]

[Figure 13503]

[Figure 13504]

[Figure 13529]

[Figure 13530]

[Figure 13531]

[Figure 13532]

[Figure 13533]

[Figure 13534]

[Figure 13535]

[Figure 13536]

[Figure 13537]

[Figure 13538]

[Figure 13539]

[Figure 13540]

[Figure 13541]

[Figure 13542]

[Figure 13543]

[Figure 13544]

[Figure 13545]

[Figure 13546]

[Figure 13547]

[Figure 13548]

[Figure 13549]

[Figure 13550]

[Figure 13551]

[Figure 13552]

[Figure 13553]

[Figure 13554]

[Figure 13555]

[Figure 13556]

[Figure 13557]

[Figure 13558]

[Figure 13559]

[Figure 13560]

[Figure 13561]

[Figure 13562]

[Figure 13563]

[Figure 13564]

[Figure 13565]

[Figure 13589]

[Figure 13590]

[Figure 13591]

[Figure 13592]

[Figure 13593]

[Figure 13594]

[Figure 13595]

[Figure 13596]

[Figure 13597]

[Figure 13598]

[Figure 13599]

[Figure 13600]

[Figure 13609]

[Figure 13610]

[Figure 13611]

[Figure 13612]

[Figure 13613]

[Figure 13614]

[Figure 13615]

[Figure 13616]

[Figure 13617]

[Figure 13618]

[Figure 13619]

Audi RS 4 Convertible 2008

[Figure 13629]

[Figure 13630]

[Figure 13631]

[Figure 13632]

[Figure 13633]

[Figure 13634]

[Figure 13635]

[Figure 13636]

[Figure 13637]

[Figure 13638]

[Figure 13639]

[Figure 13640]

[Figure 13641]

[Figure 13642]

[Figure 13643]

[Figure 13644]

[Figure 13645]

[Figure 13646]

[Figure 13647]

[Figure 13648]

[Figure 13649]

[Figure 13650]

[Figure 13651]

[Figure 13652]

[Figure 13653]

[Figure 13654]

[Figure 13655]

[Figure 13656]

[Figure 13669]

[Figure 13670]

[Figure 13671]

[Figure 13672]

[Figure 13673]

[Figure 13674]

[Figure 13675]

[Figure 13676]

[Figure 13677]

[Figure 13678]

[Figure 13679]

[Figure 13680]

[Figure 13681]

[Figure 13682]

[Figure 13689]

[Figure 13690]

[Figure 13691]

[Figure 13692]

[Figure 13693]

[Figure 13694]

[Figure 13695]

[Figure 13696]

[Figure 13697]

[Figure 13698]

[Figure 13699]

[Figure 13700]

[Figure 13709]

[Figure 13710]

[Figure 13711]

[Figure 13712]

[Figure 13713]

[Figure 13714]

[Figure 13715]

[Figure 13716]

[Figure 13717]

[Figure 13718]

[Figure 13719]

[Figure 13720]

[Figure 13721]

[Figure 13729]

[Figure 13730]

[Figure 13731]

[Figure 13732]

[Figure 13733]

[Figure 13734]

[Figure 13735]

[Figure 13736]

[Figure 13737]

[Figure 13738]

[Figure 13739]

[Figure 13749]

[Figure 13750]

[Figure 13751]

[Figure 13752]

[Figure 13753]

[Figure 13754]

[Figure 13755]

[Figure 13756]

[Figure 13757]

[Figure 13758]

[Figure 13759]

[Figure 13760]

[Figure 13769]

[Figure 13770]

[Figure 13771]

[Figure 13772]

[Figure 13773]

[Figure 13774]

[Figure 13775]

[Figure 13776]

[Figure 13777]

[Figure 13778]

[Figure 13779]

[Figure 13780]

[Figure 13781]

[Figure 13782]

[Figure 13783]

[Figure 13789]

[Figure 13790]

[Figure 13791]

[Figure 13792]

[Figure 13793]

[Figure 13794]

[Figure 13795]

[Figure 13796]

[Figure 13797]

[Figure 13798]

[Figure 13799]

[Figure 13800]

[Figure 13801]

[Figure 13802]

[Figure 13803]

[Figure 13804]

[Figure 13805]

[Figure 13806]

[Figure 13807]

[Figure 13809]

[Figure 13810]

[Figure 13811]

[Figure 13812]

[Figure 13813]

[Figure 13814]

[Figure 13815]

[Figure 13816]

[Figure 13817]

[Figure 13818]

[Figure 13819]

[Figure 13820]

[Figure 13821]

[Figure 13822]

[Figure 13823]

[Figure 13824]

[Figure 13825]

[Figure 13826]

[Figure 13827]

[Figure 13828]

[Figure 13829]

[Figure 13830]

[Figure 13831]

[Figure 13832]

[Figure 13833]

[Figure 13834]

[Figure 13849]

[Figure 13850]

[Figure 13851]

[Figure 13852]

[Figure 13853]

[Figure 13854]

[Figure 13855]

[Figure 13856]

[Figure 13857]

[Figure 13858]

[Figure 13859]

[Figure 13869]

[Figure 13870]

[Figure 13871]

[Figure 13872]

[Figure 13873]

[Figure 13874]

[Figure 13875]

[Figure 13876]

[Figure 13877]

[Figure 13878]

[Figure 13879]

[Figure 13880]

[Figure 13881]

[Figure 13889]

[Figure 13890]

[Figure 13891]

[Figure 13892]

[Figure 13893]

[Figure 13894]

[Figure 13895]

[Figure 13896]

[Figure 13897]

[Figure 13898]

[Figure 13899]

[Figure 13900]

[Figure 13909]

[Figure 13910]

[Figure 13911]

[Figure 13912]

[Figure 13913]

[Figure 13914]

[Figure 13915]

[Figure 13916]

[Figure 13917]

[Figure 13918]

[Figure 13919]

[Figure 13929]

[Figure 13930]

[Figure 13931]

[Figure 13932]

[Figure 13933]

[Figure 13934]

[Figure 13935]

[Figure 13936]

[Figure 13937]

[Figure 13938]

[Figure 13939]

[Figure 13940]

[Figure 13949]

[Figure 13950]

[Figure 13951]

[Figure 13952]

[Figure 13953]

[Figure 13954]

[Figure 13955]

[Figure 13956]

[Figure 13957]

[Figure 13958]

[Figure 13959]

[Figure 13960]

[Figure 13961]

[Figure 13962]

[Figure 13963]

[Figure 13964]

[Figure 13965]

[Figure 13966]

[Figure 13967]

[Figure 13968]

[Figure 13969]

[Figure 13970]

[Figure 13971]

[Figure 13972]

[Figure 13973]

[Figure 13974]

[Figure 13975]

[Figure 13976]

[Figure 13977]

[Figure 13978]

[Figure 13979]

[Figure 13980]

[Figure 13981]

[Figure 13982]

[Figure 13983]

[Figure 13984]

[Figure 14009]

[Figure 14010]

[Figure 14011]

[Figure 14012]

[Figure 14013]

[Figure 14014]

[Figure 14015]

[Figure 14016]

[Figure 14017]

[Figure 14018]

[Figure 14019]

[Figure 14020]

[Figure 14021]

[Figure 14022]

[Figure 14023]

[Figure 14024]

[Figure 14025]

[Figure 14026]

[Figure 14027]

[Figure 14028]

[Figure 14029]

[Figure 14030]

[Figure 14031]

[Figure 14032]

[Figure 14033]

[Figure 14034]

[Figure 14035]

[Figure 14036]

[Figure 14049]

[Figure 14050]

[Figure 14051]

[Figure 14052]

[Figure 14053]

[Figure 14054]

[Figure 14055]

[Figure 14056]

[Figure 14057]

[Figure 14058]

[Figure 14059]

[Figure 14060]

[Figure 14061]

[Figure 14062]

[Figure 14069]

[Figure 14070]

[Figure 14071]

[Figure 14072]

[Figure 14073]

[Figure 14074]

[Figure 14075]

[Figure 14076]

[Figure 14077]

[Figure 14078]

[Figure 14079]

[Figure 14080]

[Figure 14089]

[Figure 14090]

[Figure 14091]

[Figure 14092]

[Figure 14093]

[Figure 14094]

[Figure 14095]

[Figure 14096]

[Figure 14097]

[Figure 14098]

[Figure 14099]

[Figure 14100]

[Figure 14101]

[Figure 14109]

[Figure 14110]

[Figure 14111]

[Figure 14112]

[Figure 14113]

[Figure 14114]

[Figure 14115]

[Figure 14116]

[Figure 14117]

[Figure 14118]

[Figure 14119]

[Figure 14129]

[Figure 14130]

[Figure 14131]

[Figure 14132]

[Figure 14133]

[Figure 14134]

[Figure 14135]

[Figure 14136]

[Figure 14137]

[Figure 14138]

[Figure 14139]

[Figure 14140]

[Figure 14149]

[Figure 14150]

[Figure 14151]

[Figure 14152]

[Figure 14153]

[Figure 14154]

[Figure 14155]

[Figure 14156]

[Figure 14157]

[Figure 14158]

[Figure 14159]

[Figure 14160]

[Figure 14161]

[Figure 14162]

[Figure 14163]

[Figure 14169]

[Figure 14170]

[Figure 14171]

[Figure 14172]

[Figure 14173]

[Figure 14174]

[Figure 14175]

[Figure 14176]

[Figure 14177]

[Figure 14178]

[Figure 14179]

[Figure 14180]

[Figure 14181]

[Figure 14182]

[Figure 14183]

[Figure 14184]

[Figure 14185]

[Figure 14186]

[Figure 14187]

[Figure 14189]

[Figure 14190]

[Figure 14191]

[Figure 14192]

[Figure 14193]

[Figure 14194]

[Figure 14195]

[Figure 14196]

[Figure 14197]

[Figure 14198]

[Figure 14199]

[Figure 14200]

[Figure 14209]

[Figure 14210]

[Figure 14211]

[Figure 14212]

[Figure 14213]

[Figure 14214]

[Figure 14215]

[Figure 14216]

[Figure 14217]

[Figure 14218]

[Figure 14219]

[Figure 14220]

[Figure 14221]

[Figure 14222]

[Figure 14223]

[Figure 14224]

[Figure 14225]

[Figure 14226]

[Figure 14227]

[Figure 14228]

[Figure 14229]

[Figure 14230]

[Figure 14231]

[Figure 14232]

[Figure 14233]

[Figure 14234]

[Figure 14235]

[Figure 14236]

[Figure 14237]

[Figure 14238]

[Figure 14239]

[Figure 14240]

[Figure 14241]

[Figure 14242]

[Figure 14243]

[Figure 14244]

[Figure 14269]

[Figure 14270]

[Figure 14271]

[Figure 14272]

[Figure 14273]

[Figure 14274]

[Figure 14275]

[Figure 14276]

[Figure 14277]

[Figure 14278]

[Figure 14279]

[Figure 14280]

[Figure 14281]

[Figure 14282]

[Figure 14283]

[Figure 14284]

[Figure 14285]

[Figure 14286]

[Figure 14287]

[Figure 14288]

[Figure 14289]

[Figure 14290]

[Figure 14291]

[Figure 14292]

[Figure 14293]

[Figure 14294]

[Figure 14295]

[Figure 14296]

[Figure 14309]

[Figure 14310]

[Figure 14311]

[Figure 14312]

[Figure 14313]

[Figure 14314]

[Figure 14315]

[Figure 14316]

[Figure 14317]

[Figure 14318]

[Figure 14319]

[Figure 14320]

[Figure 14329]

[Figure 14330]

[Figure 14331]

[Figure 14332]

[Figure 14333]

[Figure 14334]

[Figure 14335]

[Figure 14336]

[Figure 14337]

[Figure 14338]

[Figure 14339]

[Figure 14340]

[Figure 14341]

[Figure 14349]

[Figure 14350]

[Figure 14351]

[Figure 14352]

[Figure 14353]

[Figure 14354]

[Figure 14355]

[Figure 14356]

[Figure 14357]

[Figure 14358]

[Figure 14359]

[Figure 14369]

[Figure 14370]

[Figure 14371]

[Figure 14372]

[Figure 14373]

[Figure 14374]

[Figure 14375]

[Figure 14376]

[Figure 14377]

[Figure 14378]

[Figure 14379]

[Figure 14380]

- Figure 7. PACE’s dataset-level conceptual explanations for classes Acura TL Sedan 2012 and Audi RS 4 Convertible 2008 in the Cars dataset. For each class, we show PACE’s top 3 dataset-level concepts; for each Concept k, we show the top 5 patches with their associated embeddings emj closest to the concept center µk.

[Figure 14389]

[Figure 14390]

[Figure 14391]

[Figure 14392]

[Figure 14393]

[Figure 14394]

[Figure 14395]

[Figure 14396]

[Figure 14397]

[Figure 14398]

[Figure 14399]

[Figure 14400]

[Figure 14409]

[Figure 14410]

[Figure 14411]

[Figure 14412]

[Figure 14413]

[Figure 14414]

[Figure 14415]

[Figure 14416]

[Figure 14417]

[Figure 14418]

[Figure 14419]

[Figure 14420]

[Figure 14421]

[Figure 14422]

[Figure 14423]

[Figure 14424]

[Figure 14425]

[Figure 14426]

[Figure 14427]

[Figure 14428]

[Figure 14429]

[Figure 14430]

[Figure 14431]

[Figure 14432]

[Figure 14433]

[Figure 14434]

[Figure 14435]

[Figure 14436]

[Figure 14437]

[Figure 14438]

[Figure 14439]

[Figure 14440]

[Figure 14441]

[Figure 14442]

[Figure 14443]

[Figure 14444]

[Figure 14469]

[Figure 14470]

[Figure 14471]

[Figure 14472]

[Figure 14473]

[Figure 14474]

[Figure 14475]

[Figure 14476]

[Figure 14477]

[Figure 14478]

[Figure 14479]

[Figure 14480]

[Figure 14481]

[Figure 14482]

[Figure 14483]

[Figure 14484]

[Figure 14485]

[Figure 14486]

[Figure 14487]

[Figure 14488]

[Figure 14489]

[Figure 14490]

[Figure 14491]

[Figure 14492]

[Figure 14493]

[Figure 14494]

[Figure 14495]

[Figure 14496]

[Figure 14509]

[Figure 14510]

[Figure 14511]

[Figure 14512]

[Figure 14513]

[Figure 14514]

[Figure 14515]

[Figure 14516]

[Figure 14517]

[Figure 14518]

[Figure 14519]

[Figure 14520]

[Figure 14529]

[Figure 14530]

[Figure 14531]

[Figure 14532]

[Figure 14533]

[Figure 14534]

[Figure 14535]

[Figure 14536]

[Figure 14537]

[Figure 14538]

[Figure 14539]

[Figure 14540]

[Figure 14541]

[Figure 14549]

[Figure 14550]

[Figure 14551]

[Figure 14552]

[Figure 14553]

[Figure 14554]

[Figure 14555]

[Figure 14556]

[Figure 14557]

[Figure 14558]

[Figure 14559]

[Figure 14569]

[Figure 14570]

[Figure 14571]

[Figure 14572]

[Figure 14573]

[Figure 14574]

[Figure 14575]

[Figure 14576]

[Figure 14577]

[Figure 14578]

[Figure 14579]

[Figure 14580]

[Figure 14589]

[Figure 14590]

[Figure 14591]

[Figure 14592]

[Figure 14593]

[Figure 14594]

[Figure 14595]

[Figure 14596]

[Figure 14597]

[Figure 14598]

[Figure 14599]

[Figure 14600]

[Figure 14609]

[Figure 14610]

[Figure 14611]

[Figure 14612]

[Figure 14613]

[Figure 14614]

[Figure 14615]

[Figure 14616]

[Figure 14617]

[Figure 14618]

[Figure 14619]

[Figure 14620]

[Figure 14621]

[Figure 14622]

[Figure 14623]

[Figure 14624]

[Figure 14625]

[Figure 14626]

[Figure 14627]

[Figure 14628]

[Figure 14629]

[Figure 14630]

[Figure 14631]

[Figure 14632]

[Figure 14633]

[Figure 14634]

[Figure 14635]

[Figure 14636]

[Figure 14637]

[Figure 14638]

[Figure 14639]

[Figure 14640]

[Figure 14641]

[Figure 14642]

[Figure 14643]

[Figure 14644]

[Figure 14669]

[Figure 14670]

[Figure 14671]

[Figure 14672]

[Figure 14673]

[Figure 14674]

[Figure 14675]

[Figure 14676]

[Figure 14677]

[Figure 14678]

[Figure 14679]

[Figure 14680]

[Figure 14681]

[Figure 14682]

[Figure 14683]

[Figure 14684]

[Figure 14685]

[Figure 14686]

[Figure 14687]

[Figure 14688]

[Figure 14689]

[Figure 14690]

[Figure 14691]

[Figure 14692]

[Figure 14693]

[Figure 14694]

[Figure 14695]

[Figure 14696]

[Figure 14709]

[Figure 14710]

[Figure 14711]

[Figure 14712]

[Figure 14713]

[Figure 14714]

[Figure 14715]

[Figure 14716]

[Figure 14717]

[Figure 14718]

[Figure 14719]

[Figure 14720]

[Figure 14729]

[Figure 14730]

[Figure 14731]

[Figure 14732]

[Figure 14733]

[Figure 14734]

[Figure 14735]

[Figure 14736]

[Figure 14737]

[Figure 14738]

[Figure 14739]

[Figure 14740]

[Figure 14741]

[Figure 14749]

[Figure 14750]

[Figure 14751]

[Figure 14752]

[Figure 14753]

[Figure 14754]

[Figure 14755]

[Figure 14756]

[Figure 14757]

[Figure 14758]

[Figure 14759]

[Figure 14769]

[Figure 14770]

[Figure 14771]

[Figure 14772]

[Figure 14773]

[Figure 14774]

[Figure 14775]

[Figure 14776]

[Figure 14777]

[Figure 14778]

[Figure 14779]

[Figure 14780]

[Figure 14789]

[Figure 14790]

[Figure 14791]

[Figure 14792]

[Figure 14793]

[Figure 14794]

[Figure 14795]

[Figure 14796]

[Figure 14797]

[Figure 14798]

[Figure 14799]

[Figure 14800]

[Figure 14809]

[Figure 14810]

[Figure 14811]

[Figure 14812]

[Figure 14813]

[Figure 14814]

[Figure 14815]

[Figure 14816]

[Figure 14817]

[Figure 14818]

[Figure 14819]

[Figure 14820]

[Figure 14821]

[Figure 14822]

[Figure 14823]

[Figure 14824]

[Figure 14825]

[Figure 14826]

[Figure 14827]

[Figure 14828]

[Figure 14829]

[Figure 14830]

[Figure 14831]

[Figure 14832]

[Figure 14833]

[Figure 14834]

[Figure 14835]

[Figure 14836]

[Figure 14837]

[Figure 14838]

[Figure 14839]

[Figure 14840]

[Figure 14841]

[Figure 14842]

[Figure 14843]

[Figure 14844]

### C. More Qualitative Results

In Fig. 6 and Fig. 7, we present the top three concepts for several distinct classes in the CUB and Cars datasets, respectively. Each concept is illustrated with the top five patches, providing dataset-level explanations.

Results on CUB. Fig. 6 shows PACE’s dataset-level conceptual explanations for the CUB dataset’s classes Black-footed Albatross, Crested Auklet, and Rusty Blackbird. For instance, the class Black-footed Albatross encompasses three predominant concepts: Concept 1 (Ocean Background), Concept 2 (Brown Feather), and Concept 3 (Long Wing). The accompanying top five patches exemplify PACE’s conceptual explanations, highlighting critical dataset-level concepts such as the habitat (Ocean), distinctive texture (Brown Feather), and characteristic posture (Long Wing) crucial for classifying Black-footed Albatross. Similarly, the class Crested Auklet is distinguished by concepts such as Concept 1 (Orange Beak), Concept 2 (Grey Feather), and Concept 3 (Rocks/Moss); similarly, the class Rusty Blackbird is distinguished by Concept 1 (Rusty Feather), Concept 2 (Tail), and Concept 3 (Grass/Branch). These findings reveal that distinct bird classes are each linked to unique body characteristics, such as color, shape, and texture, as well as specific habitats.

Results on Cars. Fig. 7 shows PACE’s dataset-level conceptual explanations for the Cars dataset’s classes, such as Acura TL Sedan 2012 and Audi RS 4 Convertible 2008. For example, the Audi RS 4 Convertible 2008 class features three prominent concepts: Concept 1 (Front Light), Concept 2 (Grill), and Concept 3 (Rear). The top five patches representing these concepts indicate that design elements like the front light (Front Light), grill pattern (Grill), and rear features (Rear) are essential for classifying an image as class Acura TL Sedan 2012. Moreover, the Audi RS 4 Convertible 2008 class is defined by concepts such as Concept 1 (Streamline), Concept 2 (Tire/Fender), and Concept 3 (Front Light), suggesting that car classes differ in design aspects such as shape and color.

Remark. In summary, these results showcase PACE’s proficiency in identifying crucial dataset-level concepts across different classes, utilizing patch representations within the ViT framework. Notably, this process, once the model is trained, involves deducing top concepts for each class via inference, eliminating the need for retraining or finetuning. This approach is both efficient and effective compared to methods like CRAFT (Fel et al., 2023b) that require training for each individual class.

#### D. Details on Inferring Concepts In this section, we discuss g(·) defined in Sec. 3.1 in detail.

In PACE, g(·) is implemented as an inference process on our Probabilistic Graphical Model (PGM), as shown in Fig. 2. One can see g(·) as a function that

- (1) takes the observed variables em,e′m, ym,am,a′m,ys as inputs,
- (2) goes through the learning stage and the inference stage discussed in Sec. 3.5 and Sec. 3.4, and
- (3) outputs θm, the image-level concept explanations for each image m.

As shown in Fig. 1, g(·) refers to the PACE model (the gray box on the top right). It takes as inputs the patch embeddings em, the attention weights am (which can be computed given the ViT’s parameters P), and the ViT’s predicted label ym; it then outputs the image-level explanations θ, i.e., θm = g(em, ym,am).

Besides the image-level concept explanations θm, PACE also produces the dataset-level explanations µk and Σk (where k = 1,...,K) as well as patch-level explanations ϕmj for patch j of image m). g(·) is represented by the entire Fig. 2, except for the dashed box (with the text “ViT” inside). For example, during the inference stage, PACE will

- (1) be given the global parameters µk and Σk (where k = 1,...,K) obtained from the learning stage,
- (2) treats the patch embeddings em, the attention weights am (which can be computed given the ViT’s parameters P), and the ViT’s predicted label ym as observed variables,
- (3) and then, for a new image m, infer the local parameters, i.e.,

(a) the image-level concepts (explanations) θm, which is parameterized by q(θm|γm) and (b) patch-level concepts (explanations) zmj, which is parameterized by q(zmj|ϕm).

These are called local parameters because each image has its own θm and ϕm.

### E. Theoretical Analysis

We provide the following proof of Theorem 3.2. For convenience, let Ω = (µKk=1,ΣKk=1). We then introduce a helper joint distribution of the variables em and θm,ϕm, s(em,θm,ϕm) = p(em)q(θm,ϕm|em).

According to the definition of ELBO of Sec. 3.4, in Eq. 5 and Eq. 6, we only need to prove that

LHS = Le + Lf + Ls

≤ I(em;θm,ϕm) + I( ym;ϕm) + I(ϕm;ϕ′m) + C. (16) We split the proof into the following three separate part:

###### (1) The bound of Le. We have that

m)[Eq[log p(em|Ω,θm,ϕm)]] + Eq[log q(θm,ϕm|Ω)]. (17) Since Eq[log q(θm,ϕm|Ω)] ≤ 0, we are going to prove that

Le = Ep(e

m)[Eq[log p(em|Ω,θm,ϕm)]] ≤ Is(em;θm,ϕm) − H(em). (18) In fact,

Le ≤ Ep(e

Ep(em)[Eq[log p(em|θm, ϕm, Ω)]] ≤ Ep(em)[Eq[log p(em|θm, ϕm)]]

θm,ϕm) p(em)

p(em)p(em|θm,ϕm) q(em|θm,ϕm) ]]

= Ep(em)[Eq[log q(em|

θm,ϕm)

θm,ϕm)

= Ep(em)[Eq[log q(em|

p(em) ]] + Ep(em)[Eq[log p(em)]] + Ep(em)[Eq[log p(em|

q(em|θm,ϕm) ]]

= Is(em; θm, ϕm) − H(em) − Eq[KL(q(em|θm, ϕm)|p(em|θm, ϕm))] ≤ Is(em; θm, ϕm) − H(em) − 0, (19)

where H(em) is a constant.

###### (2) The bound of Lf. With the constraint −1 ≤ ηn ≤ 1(1 ≤ n ≤ N), we have that Lf = Eq[log p( ym|z¯m,H)]

N n=1

N n=1

exp(ηTnϕ¯m)) ≤

ymn(ηTnϕ¯m) − log(

≈

N n=1

ymn(ηTnϕ¯m) ≤

N n=1

K k=1

ymnϕ¯mk ≤

###### ϕ¯

p( ym,ϕ¯m)log p( ym,

m) p( ym)p(

+ C1

###### ϕ¯

m)

###### ym ϕ

m

= I( ym;ϕm) + C1, (20)

where C1 is a constant.

- (3) The bound of Ls. With the constraint 0 ≤ β ≤ 1, we have that Ls = Eq[log p(ys = 1|z¯1:M,z¯′m,β)]

≈ βT(ϕ¯m ◦ ϕ¯′m) − log(

exp(βT(ϕ¯m ◦ ϕ¯f)))

f∈F

≤ βT(ϕ¯m ◦ ϕ¯′m) ≤ ϕ¯m · ϕ¯′m ≤

m,ϕ′

ϕ

p(ϕm,ϕ′m)log p(

m) p(ϕ

m) + C2

m)p(ϕ′

###### ϕ′

###### ϕ

m

m

= I(ϕm;ϕ′m) + C2, (21)

where C2 is a constant. Combining (1 ∼ 3) above concludes the proof.

### F. Details on Learning PACE

- F.1. Derivations of ELBO Inferring zm. According to Eq. 1,

J j=1

zmj, (22) where zmj can be approximate by a variational distribution parameterized by ϕmj:

z¯m = J1

q(zmj | ϕmj) = Categorical(zmj | ϕmj), (23) which indicates that

E[zmj] = ϕmj. (24) Therefore, we have

J j=1

J j=1

ϕmj = ϕ¯m. (25) Hence, we have

E[z¯m] = J1

E[z¯mj] = J1

z¯m ≈ ϕ¯m. (26) Computing Le. We can expand the ELBO in Eq. 7 as:

K

K

K

K

αk) −

(αk − 1)(Ψ(γmk) − Ψ(

Le = log Γ(

log Γ(αk) +

γk′))

k′=1

k=1

k=1

k=1

K

K

ϕmjk(Ψ(γmk) − Ψ(

+

γmk′))

k′=1

k=1

K

ϕmjkamj{−12(emj − µk)TΣ−k 1(emj − µk) − log[(2π)d/2|Σk|1/2]}

+

k=1

K

K

K

K

− log Γ(

log Γ(γmk) −

(γmk − 1)(Ψ(γmk) − Ψ(

γmk) +

γmk′))

k′=1

k=1

k=1

k=1

K

ϕmjk log ϕmjk. (27)

−

k=1

We can interpret the meaning of each term of Le as follows:

- • The sum of the first and the fourth terms, namely Eq[log p(θm|α)]−Eq[log q(θm)], is equal to −KL(q(θm)|p(θm|α)), which is the negation of KL Divergence between the variational posterior probability q(θm) and the prior probability

- p(θm|α) of the topic proportion θm for document m. Therefore maximizing the sum of these two terms is equivalent to minimizing the KL Divergence KL(q(θm)|p(θm|α)); this serves as a regularization term to make sure the inferred
- q(θm) is close to its prior distribution p(θm|α).

- • Similarly, the sum of the second and the last terms (ignoring the summation over the word index j for simplicity), namely Eq[log p(zmj|θm)] − Eq[log q(zmj)] is equal to −KL(q(zmj)|p(zmj|θm)), which is the negation of the KL Divergence between the variational posterior probability q(zmj) and the prior probability p(zmj|θm) of the word-level topic assignment zmj for word j of document m. Therefore maximizing the sum of these two terms is equivalent to minimizing the KL Divergence KL(q(zmj)|p(zmj|θm)); this serves as a regularization term to make sure the inferred q(zmj) is close to its “prior” distribution p(zmj|θm).
- • The third term Eq[log p(emj|zmj,µz

) of every contextual embedding emj (for word j of document m) conditioned on the inferred zmj and the parameters (µz

)] is to maximize the log likelihood p(emj|zmj,µz

###### ,Σz

###### ,Σz

mj

mj

mj

mj

).

###### ,Σz

mj

mj

Computing Lf. Eq. 8 is derived from employing Taylor’s expansion to Eq. 2:

Lf = Eq[log p( ym|z¯m,H)]

N n=1

N n=1

ymn(ηTnϕ¯m) − Eq[log(

exp(ηTnz¯m))] ≈

=

N n=1

N n=1

ymn(ηTnϕ¯m) − log(

exp(ηTnϕ¯m + (1/2)ηTnSmηn)), (28)

where Sm is the covariance matrix of z¯m. We will see that for any entry of Sm, i.e. ∀x,y ∈ {1,2,...,K}, we have

0 ≤ Sm,xy ≤ J12 . (29) In our setting, the number of patches J in each image satisfies J > 100, hence Sm,xy is very close to zero. We compute Sm by definition:

Sm,xy = Cov[z¯mz¯m′]x,y

= E[(¯zmxz¯m′x − E[¯zmxz¯m′x])(¯zmyz¯m′y − E[¯zmyz¯m′y])]

= E[(¯zmxz¯m′x − E[¯zmx]E[¯zm′x]])(¯zmyz¯m′y − E[¯zmy]E[¯zm′y]])]

= E[(¯zmxz¯m′x − ϕ¯mxϕ¯m′x)(¯zmyz¯m′y − ϕ¯myϕ¯m′y)]

= E[¯zmxz¯m′xz¯myz¯m′y] − ϕ¯myϕ¯m′yE[¯zmxz¯m′x] − ϕ¯mxϕ¯m′xE[¯zmyz¯m′y] + ϕ¯mxϕ¯m′xϕ¯myϕ¯m′y

= E[¯zmxz¯my]E[¯zm′xz¯m′y] − ϕ¯mxϕ¯m′xϕ¯myϕ¯m′y. (30) We then consider two different cases:

- Case (1): x = y. Then we have that

Cov[z¯mz¯m′]x,y = E[¯zmx2 ]E[¯zm2 ′y] − ϕ¯2mxϕ¯2m′y

= ϕ¯2mxϕ¯2m′y − ϕ¯2mxϕ¯2m′y

= 0. (31)

- Case (2): x ̸= y. Note that

z¯mxz¯my = J1

j

zmjx · J1

j

zmjy. (32)

Given that zmj is a one-hot vector, we have

zmjx · zmjy = 0. (33) Hence, we have

Therefore, we have

J

E[¯zmxz¯my] = E[¯zmx]E[¯zmy] − J12 (

j=1

E[zmjx]E[zmjy])

= ϕ¯mxϕ¯my − J12

J

ϕmjxϕmjy. (34)

j=1

Cov[z¯mz¯m′]x,y = (ϕ¯mxϕ¯my − J12

J

= J14

ϕmjxϕmjy

j=1

J

ϕmjxϕmjy)(ϕ¯m′xϕ¯m′y − J12

j=1

J

ϕm′jxϕm′jy) − ϕ¯mxϕ¯m′xϕ¯myϕ¯m′y

j=1

J

ϕm′jxϕm′jy. (35)

j=1

Since 0 ≤ ϕmj,ϕ′mj ≤ 1, we have that

0 ≤ Sm,xy = Cov[z¯mz¯m′]x,y ≤ J12 . (36) In summary, we can see that in either case, Eq. 29 holds. Therefore we have

Lf ≈

N n=1

ymn(ηTnϕ¯m) − log(

N n=1

exp(ηTnϕ¯m)). (37)

Computing Ls. Similarly, by employ Taylor’s expansion of Eq. 3, as well as Eq. 29, we have that

Ls = Eq[log p(ys = 1|z¯1:M,z¯′m,β)]

= βT(z¯m ◦ z¯′m) − Eq[log(

exp(βT(z¯m ◦ z¯f)))] ≈ βTϕ¯mϕ¯′m − log(

f∈F

exp(βT(ϕ¯mϕ¯f) + (1/2)βTSmβ)) ≈ βT(ϕ¯m ◦ ϕ¯′m) − log(

f∈F

exp(βT(ϕ¯m ◦ ϕ¯f))), (38) where the parameter β is learned jointly by gradient-based optimization algorithms, such as Adam.

f∈F

###### F.2. Update Rules

- F.2.1. INFERENCE

Derivative of Le. Taking the derivative of the Le in Eq. 7 with respect to ϕmjk and setting it to zero, we obtain the update rule for ϕmjk:

K k′=1

ϕmjk ∝ |Σk1|1/2 exp[Ψ(γmk) − Ψ(

γk′) −12amj(emj − µk)TΣ−k 1(emj − µk)]. (39)

Derivative of Lf. The log-sum term of Eq. 37 is intractable. To address this, with Taylor’s Expansion, we have that

log(

N n=1

exp(ηTnϕ¯m)) ≈ log(

ϕ¯(0)

N n=1

n=1 exp(η

m )η

N

T n

exp(ηTnϕ¯(0)m )) + (ϕ¯m − ϕ¯(0)m )T

, (40)

n N

ϕ¯(0)

n=1 exp(ηT

m )

n

where ϕ¯(0)m is the value of ϕ¯m at the last iteration of ϕ − γ update discussed in Alg. 1. Taking the derivative w.r.t. ϕm, we have that

N

###### ϕ¯

n=1 exp(η

m)η

N

T n

∂Lf ∂

. (41)

≈

ymnηn −

n N

###### ϕ¯

###### ϕ¯

n=1 exp(ηT

m)

m

n

n=1

Note that by definition ϕ¯m = 1/J

= ∂Lf ∂

∂Lf ∂ϕ

###### ϕ¯

mj

J j=1 ϕmj in the main paper, we have

N

###### ϕ¯

· ∂

= N1 ∂Lf

= N1 (

m

###### ϕ¯

###### ∂ϕ

∂

mj

m

m

n=1

ymnηn −

###### ϕ¯

n=1 exp(η

m)η

N

T n

). (42)

n N

###### ϕ¯

n=1 exp(ηT

m)

n

Derivative of Ls. Taking the derivative of Eq. 38, i.e.

Ls = βT(ϕ¯m ◦ ϕ¯′m) − log(

exp(βT(ϕ¯m ◦ ϕ¯f))), (43) we have that

f∈F

###### ϕ¯

###### ϕ¯

f))βTϕ¯

βT(

m◦

≈ βTϕ¯′m − f∈F exp(

∂Ls ∂

. (44)

f

###### ϕ¯

###### ϕ¯

###### ϕ¯

f∈F exp(βT(

m◦

f))

m

In summary, the partial derivative of ELBO in Eq. 5 w.r.t. ϕmj is

+ ∂∂Lf

###### = ∂L

∂L ∂ϕ

e

###### ∂ϕ

###### ϕ

mj

mj

mj

###### + ∂L

. (45)

s

###### ∂ϕ

mj

Setting the derivative to 0, we have a closed-form update rules for ϕ as follows:

K k′=1

γk′) − 12amj(emj − µk)TΣ−k 1(emj − µk)

ϕmj ∝ |Σk1|1/2 exp[Ψ(γmk) − Ψ(

###### ϕ¯

###### ϕ¯

f))(βTϕ¯

###### ϕ¯

βT(

N n=1

n′=1 exp(η

m)η

N

T n′

m◦

+ βTϕ¯′m − f∈F exp(

f) f∈F exp(βT(

)]. (46)

+J1 (

ymlηl −

n N

###### ϕ¯

###### ϕ¯

###### ϕ¯

n=1 exp(ηT

m◦

f))

m)

n

Taking derivative of Eq. 27 and set to 0, we have

γmk = αk +

J j=1

ϕmjkamj. (47)

- F.2.2. LEARNING

Similar to Sec. F.2.1, we expand the ELBO in Eq. 7, take its derivative w.r.t. µk and set it to 0:

∂L ∂µ

k

yielding the update rule for learning µk:

ϕmjkamjΣ−k 1(emj − µk) = 0, (48)

=

m,j

µk = m,j ϕmjkamjemj

m,j ϕmjkamj , (49)

where Σ−k 1 is canceled out. Similarly, setting the derivatives w.r.t. Σk to 0, i.e.,

we have

ϕmjkamj(−Σ−k 1 + Σ−k 1(emj − µk)(emj − µk)TΣ−k 1), (50)

= 21

∂L ∂Σk

m,j

###### µ

k)(emj−µ

k)T

Σk = m,j ϕmjkamj(emj−

m,j ϕmjkamj . (51)

