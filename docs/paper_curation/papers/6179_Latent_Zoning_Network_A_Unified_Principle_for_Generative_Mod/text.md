# arXiv:2509.15591v2[cs.LG]4Nov2025

## Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification

#### Zinan Lin∗

Microsoft Research Redmond, WA, USA

Enshu Liu Tsinghua University Beijing, China

Xuefei Ning Tsinghua University Beijing, China

Junyi Zhu KU Leuven Leuven, Belgium

Wenyu Wang Redmond, WA, USA

Sergey Yekhanin Microsoft Research Redmond, WA, USA

### Abstract

Generative modeling, representation learning, and classification are three core problems in machine learning (ML), yet their state-of-the-art (SoTA) solutions remain largely disjoint. In this paper, we ask: Can a unified principle address all three? Such unification could simplify ML pipelines and foster greater synergy across tasks. We introduce Latent Zoning Network (LZN) as a step toward this goal. At its core, LZN creates a shared Gaussian latent space that encodes information across all tasks. Each data type (e.g., images, text, labels) is equipped with an encoder that maps samples to disjoint latent zones, and a decoder that maps latents back to data. ML tasks are expressed as compositions of these encoders and decoders: for example, label-conditional image generation uses a label encoder and image decoder; image embedding uses an image encoder; classification uses an image encoder and label decoder. We demonstrate the promise of LZN in three increasingly complex scenarios: (1) LZN can enhance existing models (image generation): When combined with the SoTA Rectified Flow model, LZN improves FID on CIFAR10 from 2.76 to 2.59—without modifying the training objective. (2) LZN can solve tasks independently (representation learning): LZN can implement unsupervised representation learning without auxiliary loss functions, outperforming the seminal MoCo and SimCLR methods by 9.3% and 0.2%, respectively, on downstream linear classification on ImageNet. (3) LZN can solve multiple tasks simultaneously (joint generation and classification): With image and label encoders/decoders, LZN performs both tasks jointly by design, improving FID and achieving SoTA classification accuracy on CIFAR10. The code and trained models are available at https://github.com/microsoft/ latent-zoning-networks. The project website is at https://zinanlin.me/ blogs/latent_zoning_networks.html.

### 1 Introduction

Generative modeling, representation learning, and classification are three of the most widely used machine learning (ML) tasks. Generative models like DALLE [72, 71, 5] and GPT [69, 70, 7, 1] power applications such as question answering and content creation. Representation learning, exemplified by CLIP [68], supports tasks like information retrieval. Classification is central to tasks such as object recognition [17] and sentiment analysis [19, 55].

Notably, the state-of-the-art (SoTA) techniques for these tasks differ. For example, SoTA generative modeling relies on diffusion models [32, 77, 79] and auto-regressive transformers [69, 70, 7, 1]; SoTA

∗Correspondence to: Zinan Lin (zinanlin@microsoft.com). † Main updates in arXiv V2: Improving Fig. 2, fixing Fig. 3, expanding related work discussions in § 3.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

①② ③④ ⑤⑥ ⑦⑧ ⑨⑩

Examples of tasks:

- ① Unconditional image generation
- ② Image embedding
- ③ Label-conditioned image generation
- ④ Image classification
- ⑤ Text embedding
- ⑥ Image-to-text generation (e.g., image captioning)
- ⑦ Text-to-image generation
- ⑧ Text-to-text generation (e.g., chatbot)

- ⑪
- ⑫
- ⑬
- ⑭
- ⑮
- ⑯
- ⑰

④ ② ① ⑤

⑧

⑦

③

⑥

Shared latent space

- Figure 1: Latent Zoning Network (LZN) connects multiple encoders and decoders through a shared latent space, enabling a wide range of ML tasks via different encoder-decoder combinations or standalone encoders/decoders. The figure illustrates eight example tasks, but more could be supported. Only tasks 1-4 are evaluated in this paper, while the rest are for illustration.

Latent space

Follows Gaussian

distribution

“Cat”

“Dog” “Bird”

The latent zone of label

The latent zone of image

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

- Figure 2: The latent space of LZN has two key properties: (1) Generative: It follows a simple Gaussian prior, allowing easy sampling for generation tasks. (2) Unified: It serves as a shared representation across all data types (e.g., image, text, label). Each data type induces a distinct partitioning of the latent space into latent zones, where each zone corresponds to a specific sample (e.g., an individual image or label). The latent space is shown as a closed circle for illustration, but it is unbounded in practice.

|Latent computation|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

“Cat” “Dog” “Bird”

Samples

Anchor points

Flow matching

Flow matching

|Latent alignment|
|---|

Latent zones

Latent space

Latent space

Figure 3: Training and inference in LZN rely on two atomic operations: (1) Latent computation (§ 2.2.1): Computes latent zones for a data type by encoding samples into anchor points and using flow matching (FM) [54, 51] to partition the latent space. Conversely, any latent point can be mapped to a sample via the decoder (not shown). (2) Latent alignment (§ 2.2.2): Aligns latent zones across data types by matching their FM processes. This figure also illustrates the approach for LZN in joint conditional generative modeling and classification (§ 5).

representation learning employs contrastive loss [28, 10, 26]; and SoTA classification uses dedicated models trained with cross-entropy loss and its variants [46]. Although using distinct methods for these tasks has long been established and widely accepted in the community, we revisit this methodology from first principles and question its necessity. Specifically, we ask, out of curiosity:

Can a single principle unify generative modeling, representation learning, and classification?

Part of our motivation stems from Occam’s Razor [84], which favors simpler solutions when possible. More importantly, while these tasks differ in formulation, they are fundamentally related and can benefit from one another; a unified principle could facilitate such synergy.2

In this paper, we reflect on the strengths and limitations of existing techniques and propose a new unified framework, Latent Zoning Network (LZN), illustrated in Figs. 1 and 2. At the core of our design is a shared latent space that connects a series of encoders and decoders, each corresponding to a specific data type (e.g., images, text, labels). Encoders map data into a zone in the latent space, and the corresponding decoder maps it back to the data. Different tasks can be interpreted as

2While auto-regressive (AR) transformers with large-scale pre-training provide one approach to unify these tasks [69, 70, 7, 1], SoTA transformer-based representation learning still relies on contrastive learning [45, 90]. More importantly, our approach can be viewed as an orthogonal layer on top of transformers and should be seen as complementary rather than competing. See § 2.4 for further discussion.

performing “translations” within the latent space—either using encoder–decoder pairs or leveraging a single encoder or decoder. Compared to popular representation learning approaches, which place no constraint on the latent distribution [10], LZN’s latent space is generative: it follows a simple prior distribution for easy sampling. In contrast to modern generative modeling approaches, where different conditions (e.g., class labels, text) are treated as separate inputs [88], LZN maintains a single latent space that unifies different types of conditional information. Finally, unlike standard classification, where class labels are model outputs, LZN treats “class labels” as a data type connected through the latent space like any other data type.

To train and perform inference with LZN, we rely on two atomic operations (Fig. 3 and § 2): (1) Latent computation: Given an encoder, we compute the latent zones for a batch of samples. Specifically, we first use the encoder to compute each sample’s anchor point, then apply flow matching [54, 51] to map these points to their corresponding latent zones. This procedure ensures that the resulting zones collectively follow a simple Gaussian distribution, facilitating generation tasks, while also guaranteeing that zones from different samples remain disjoint–allowing them to serve as unique latents for classification and representation learning. (2) Latent alignment: Aligning the latent zones produced by two different encoders to facilitate the tasks that require translations between encoders and decoders from different data types. This is a fundamentally challenging task due to its discrete nature. To address it, we introduce a novel “soft” approximation that performs the alignment midway through the flow matching process, enabling effective and tractable training.

We demonstrate that, despite its simplicity and reliance on just two atomic operations, LZN is capable of supporting a wide range of seemingly diverse tasks. To illustrate its versatility and practical utility, we present three levels of applications:

- • L1: Enhancing one existing task (§ 3). Because LZN latents can be computed without supervision, they can be seamlessly integrated into existing models as an additional conditioning signal–without requiring any changes to the training loss or methods. In this setup, LZN latents hopefully can learn to improve the task performance. To demonstrate this, we incorporate LZN into rectified flow models [54]–a state-of-the-art generative approach for images–and observe improved sample quality across CIFAR10, AFHQ-Cat, CelebA-HQ, LSUN-Bedroom datasets. Specifically, on CIFAR10, LZN closes the FID gap between conditional and unconditional generation by 59%.
- • L2: Solving one task independently (§ 4). LZN can also tackle tasks entirely on its own, without relying on existing methods. As a case study, we use LZN to implement unsupervised representation learning, a task traditionally addressed with contrastive loss. We find that LZN can even outperform the seminal methods such as MoCo [28] and SimCLR [10] by 9.3% and 0.2%, respectively, on downstream ImageNet linear classification.
- • L3: Solving multiple tasks simultaneously (§ 5). Pushing further, LZN is capable of handling multiple tasks at once. In particular, we employ two encoder–decoder pairs—one for images and one for labels—enabling LZN to jointly support class-conditional generation and classification within a single, unified framework. Built on rectified flow, this implementation outperforms the baseline conditional rectified flow model in generation quality, while also achieving state-of-theart classification accuracy. Notably, the performance on both generation and classification exceeds that of training each task in isolation. This supports our core motivating intuition: seemingly distinct tasks can benefit from shared representations, and LZN provides a principled framework for enabling such synergy.

While the early results are promising, many challenges, open questions, and exciting opportunities remain unexplored. In principle, as more encoder–decoder pairs are added to the shared latent space, the range of applications LZN can support should grow at least quadratically (Fig. 1). Whether LZN can scale gracefully and realize this potential remains to be seen. We hope this work opens a new line of research toward this ambitious vision. See more discussions in § 6.

2 Latent Zoning Network (LZN)

- 2.1 Overall Framework

Revisiting existing approaches. To motivate our design on a unified framework for diverse ML tasks, we first analyze the strengths and limitations of existing approaches.

- • Generative modeling. Given samples x ∼ p from an unknown distribution p, generative models aim to learn a decoder Dx such that Dx (z) approximates p, where z is random noise drawn

- from a simple distribution.3 While z can carry useful information for representation learning and classification [50], this pipeline has key limitations: (1) The mapping from z to x lacks flexibility. For example, in diffusion models, the optimal mapping is fixed once the distributions of x and z are fixed [79]. To introduce controllability, models often augment D with additional condition inputs c1,...,ck: G(z,c1,...,ck) [88]. This is suboptimal–conditions may overlap or conflict (e.g., text vs. label conditions in image generation), and the resulting representation (z,c1,...,ck) becomes fragmented. (2) Inverting D to recover z from a sample x is non-trivial for some SoTA generative models [36]. These issues limit the effectiveness of generative models for representation learning, as also observed in prior work [24].
- • Unsupervised representation learning.4 SoTA representation learning typically uses contrastive loss [10], where an encoder E maps related sample pairs–either from the same modality (e.g., image augmentations) or across modalities (e.g., image–text pairs)–to similar embeddings, while pushing unrelated samples apart. These embeddings can perform zero-shot classification by comparing pairwise cosine similarities [68] or be adapted for classification using a linear head trained with cross-entropy loss [10]. However, contrastive loss leads E to discard important details (e.g., augmentations, modalities), making the representations unsuitable for standalone generation. Moreover, with few exceptions [2], the representations lack distributional constraints, making them hard to sample from for generative tasks unless training an additional generative model [44].
- • Classification. The most common and SoTA classification approach trains a dedicated model with cross-entropy loss to map inputs to class labels [19, 55]. Intermediate layer outputs can be used as representations [81]. However, because the objective focuses solely on classification, these representations tend to discard class-irrelevant information, limiting their utility for generalpurpose representation or generation. As with contrastive learning, they also lack distributional constraints for generative tasks.

While one could combine the above objectives and methods into a single training setup [61, 43], our focus is on designing a clean, unified framework that naturally integrates all these tasks.

Desiderata. We observe that all the above tasks can be framed as learning mappings between data and a latent space. The main differences lie in: the mapping direction (e.g., latent-to-data for generation, data-to-latent for representation/classification), constraints on the latent space (e.g., a simple prior for generative models, none for others), and the amount of information encoded (e.g., class labels for classification tasks, detailed reconstructions for generative models). To support all these tasks in a single framework, we seek: (1) A unified latent space that captures all necessary information of all tasks; (2) A generative latent space that follows a simple distribution; and (3) Easy mappings between data and latent in both directions.

Framework. To address the above desiderata, our key designs are (Fig. 2):

- • A unified latent space. In existing frameworks, a sample like text can play inconsistent roles– appearing as input latent in text-to-image generation or as output in text generation. This makes it hard to define a unified latent space across tasks. We address this by introducing a hypothetical foundation latent space that represents all possible samples in the world. Each foundation latent is an abstract entity that appears through observations in different data types, such as images, text, and even class labels (e.g., “cat” or “dog”). Importantly, different latents can share the same observation (e.g., multiple cat images all labeled “cat” and described as “a cat image”). As a result, each observed sample defines a latent zone—a subset of the latent space that produces the same observation in that data type. This provides a unified way to represent and connect all data types within the same latent space.
- • A generative latent space. We enforce the latent space to follow a Gaussian distribution, enabling easy unconditional sampling without constraining any data type. Our framework also supports easy conditional sampling from a latent zone induced by an observed sample (e.g., a label).
- • Easy mappings. Given samples of a data type, we compute their latent zones via the corresponding encoder. Conversely, a latent point can be decoded into a data type using its decoder.

Tasks. This design naturally supports a variety of tasks (Fig. 1):

- 3For diffusion models [32, 77, 79], z is the initial Gaussian noise in the sampling process, plus intermediate noise if using SDE sampling [80]. For AR transformers, z can be seen as the randomness in token sampling.
- 4We use “latent”, “representation”, and “embedding” interchangeably in the paper.

- • Single-module tasks. A standalone encoder or decoder can perform specific tasks independently. For instance, the image encoder alone produces image embeddings (representations), while the image decoder alone enables unconditional image generation.
- • Cross-module tasks. Any encoder–decoder pair defines a task. For example, label encoder

+ image decoder enables class-conditional image generation, image encoder + label decoder does classification, and text encoder + text decoder supports text generation.

We expect that tasks can benefit from each other through this unified framework (validated in § 5). Each task contributes its core information to the latent space, making it increasingly expressive and powerful. Conversely, since all tasks interface through the same latent space, improvements in the latent representations can facilitate learning across tasks.

As the latent space partitions into zones, we name this framework Latent Zoning Network (LZN).

#### 2.2 Implementation of Atomic Operations

Training and inference in LZN rely on two operations (Fig. 3): latent computation and latent alignment. We will see in § 3 to 5 that these two operations are sufficient to implement a variety of tasks.

#### 2.2.1 Latent Computation

Desiderata. Latent computation is important in both training and inference of LZN. Given samples X = {x1,...,xn} of the same data type (e.g., images, text, labels), the goal is to sample their latents z1,...,zn = C (X) with a random latent computation function C such that: (1) Prior distribution is Gaussian: the latent z ∼ Uniform{z1,...,zn} follows Gaussian distribution N (0,I), and (2) The latent zones of different samples are disjoint: Supp(zi) ∩ Supp(zj) = ∅ for i ̸= j.

Approach. To achieve this, we first use a deterministic encoder Ex to map each sample to an anchor point ai = Ex (xi). We then apply the seminal flow matching (FM) method [54, 51], which establishes a one-to-one mapping between distributions, to transform these anchor points into latent zones. Specifically, we define a family of distributions πt with endpoints π0 = N(0,I), the desired prior latent distribution, and π1(s) = n1 ni=1 δ(s − ai), the distribution of the anchor points.5 The intermediate distribution πt is induced by linearly interpolating between samples s0 ∼ π0 and s1 ∼ π1 via φ(s0,s1,t) = (1 − t)s0 + ts1 (i.e., πt is (1 − t)π0 + tπ1). The velocity field in FM [54] can then be computed as

i)2 2(1−t)2

n i=1(ai − s)exp −(s−ta

∂φ(s0,s1,t) ∂t |φ(s0,s1,t) = s =

V (s,t) ≜ Es

.

0∼π0,s1∼π1

i)2 2(1−t)2

(1 − t) ni=1 exp −(s−ta

We can obtain st by integrating along the FM trajectory: st = FMx (s0;t) ≜ s0 + τ t=0 V (sτ,τ)dτ for s0 ∼ π0. It has been shown [54] that the distribution of st is πt. Similarly, integrating backward

st = IFMx (s1−g;t) ≜ s1−g + τ t=1−g V (sτ,τ)dτ for s1−g ∼ π1−g ≜ (1 − g)π1 + gN(0,I) also yields πt, where g is a small constant.6 With a slight abuse of notation, we also write st = IFMx (ai,ϵi;t) to represent IFMx (s1−g;t) with s1−g = (1 − g)ai + gϵi. With these setups, we define the latent computation as

zi = C (X)i ≜ IFMx (ai,ϵi;0), where ϵi ∼ N(0,I). (1) Due to the discussed FM properties, we can see that this approach (approximately) satisfies the two desiderata by construction (see § A.1 for more details).

Implementation. Computing C involves an integral, which we approximate using standard numerical solvers such as Euler or DPM-Solver [56, 57] that operates on a finite list of time steps t1,...,tr, as in prior work on diffusion and RF models [54, 80]. A key property of our approach is that all operations are differentiable, allowing gradients to backpropagate all the way from latent zi to the encoders Ex during training. More details are deferred to § A.2.

Efficiency optimization. Computing the velocity V requires access to all samples, making the memory and computation cost of C high. To address this, we introduce several optimization

5δ denotes the Dirac delta function. 6FM is well-defined only when both π0 and π1 have full support. However, in our case, π1 is a mixture of

Dirac deltas and lacks full support. Therefore, we use the full-support distribution π1−g as the starting point.

techniques–latent parallelism, custom gradient checkpointing, and minibatch approximation–that make the training of LZN scalable. See § A.3 for details.

#### 2.2.2 Latent Alignment

Desiderata. Following § 2.2.1, latent zones from different data types are computed independently, which undermines the purpose of a shared latent space. Many applications require these zones to be aligned. We consider two types of alignment: (1) Many-to-one (and one-to-many) alignment: for example, the latent zone of the “cat” label should cover all latent zones of all cat images. (2) One-to-one alignment: for example, in image-text datasets, paired image and text samples should share the same latent zone. Concrete examples will be shown in § 4 and 5.

Formally, let X = {x1,...,xn} and Y = {y1,...,ym} be two datasets from different data types. The pairing is defined by ki, where yi (e.g., a cat image) is paired with xk

(e.g., the “cat” label). We aim to ensure Supp C (X)k

i

⊇ Supp(C (Y)i) for all i ∈ [m], meaning the latent zone of xk

i

covers that of yi. This formulation supports many-to-one alignments directly. For one-to-one alignment, a symmetric constraint can be added with x and y swapped. Approach. Given the FM integral trajectories, alignment reduces to ensuring that the latent of yi, when mapped via the trajectory, matches the anchor point of xk

i

: FMx (C (Y)i ;1) = Ex (xk

).

i

i

Challenge: discrete assignment is non-differentiable. Before introducing our solution, we illustrate why the problem is nontrivial by examining strawman approaches. A natural idea is to directly minimize the distance: d(FMx (C (Y)i ;1),Ex (xk

)), where d(·,·) is a distance metric. This approach fails because FM deterministically maps each latent to exactly one anchor point Ex (xj), so the above objective effectively becomes minimizing the distance between anchor points. However, a latent zone is influenced by all anchor points, not just its own. Therefore, reducing the distance between a pair of anchors does not necessarily improve zone-level alignment. More fundamentally, the core challenge is that FM induces a discrete assignment: each latent deterministically maps to one anchor. This discrete operation is non-differentiable and cannot be directly optimized during training.

i

- Technique 1: Soft approximation of alignment. To address this issue, our key idea is to introduce a

soft approximation of the discrete anchor assignment process. Let us define sit = FMx (C (Y)i ;t) and al = Ex (xl). By construction, the distribution πt is a mixture of Gaussians: πt =

- 1 n

n l=1 N(tal,(1 − t)2I), where the l-th component corresponds to anchor al. We define the

(soft) probability that sit is assigned to al as being proportional to the density of sit under the l-th Gaussian component:

P al|sit = exp −∥s

i t−tal∥2

2(1−t)2 /

n j=1 exp − ∥s

i t−taj∥2

2(1−t)2

.

This formulation provides a smooth, differentiable approximation of the otherwise discrete assignment. When t = 0, the approximation is fully smooth, with P al|si0 = 1/n for all i,l, reflecting a uniform assignment. As t increases toward 1, the assignment becomes sharper. In the limit as t → 1, it converges to the true discrete assignment, where sit deterministically maps to its assigned anchor. From this, a straightforward idea is to maximize the assignment probability over all time steps such

- as t∈{t

1,...,tr} P ak

i|sit (recall that tis are solver time steps; see § 2.2.1). However, our ultimate goal is only to ensure correct assignment at t = 1. Even if this is achieved, the above objective would continue to push the intermediate states st toward ak

i

, which is unnecessary and potentially harmful. Technique 2: Optimizing maximum assignment probability. To avoid this, we propose to maximize maxt∈{t

1,...,tr} P ak

i|sit . This ensures that once the trajectory reaches the correct anchor near t = 1, the objective is maximized (i.e., equals 1) and no further gradient is applied as desired. However, this approach introduces a new issue: if st diverges from ak

i

early on, the maximum probability remains

- at the constant 1/n (attained at t = t1 = 0), yielding no training signal.

- Technique 3: Early step cutoff. To mitigate this, we truncate the set of time steps used in the maxi-

mization, restricting it to the later stages of the trajectory: {tu,...,tr}, where u is a hyperparameter that excludes early time steps. Putting it all together, our proposed alignment objective is:

m

Align(X,Y) ≜ maximize

i|sit . (2) Please see § B for more implementation details.

P ak

max

t∈{tu,...,tr}

i=1

Latent space

[Figure 13]

Training Generator Drawas an aextralatentconditionof the targetinput toimagethe generator

[Figure 14]

[Figure 15]

Draw a latent from the whole latent space (standard Gaussian) as an extra condition input to the generator

Generation

Generator

Figure 4: LZN for unconditional generative modeling (§ 3). During training, the LZN latent of each target image is fed as an extra condition to the rectified flow (RF) model [54], making the RF learn conditional flows based on LZN latents. The objective remains the standard RF loss, and the LZN encoder is trained end-to-end within it. During generation, we sample LZN latents from a standard Gaussian and use them as the extra condition. We illustrate the approach with RF, but since LZN latents require no supervision and are differentiable, the method could apply to other tasks by adding a condition input for LZN latents to the task network.

#### 2.3 Decoder

The decoder Dx maps the LZN latent back to its corresponding sample: Dx (zi) = xi. As we see later, it can be implemented using either a generative model (§ 3 and 5) or FM (§ 5).

#### 2.4 Relationships to Alternatives

A prominent alternative that can also unifies different ML tasks is the use of AR transformers with large-scale pertaining, such as large language models (LLMs) [69, 70, 7, 1]. These models unify generation tasks by representing all data as sequences of tokens and modeling them in an AR manner. Classification tasks are cast as generation problems—for instance, prompting the model to complete the sentence “Which animal is in this image?” [70]. Additionally, prior work has shown that the intermediate outputs in these models can serve as strong representation for downstream tasks [73].

As such, this approach is generation-centric: the core formulation remains a generative modeling problem. Tasks that can be framed as generation–such as classification-can be unified naturally. However, for other tasks like representation learning, this approach must rely on surrogate methods, such as using intermediate model outputs. In contrast, LZN offers a new formulation that seamlessly unifies all these tasks within a single framework, as we will demonstrate in the following sections.

More importantly, AR transformers (and other generative models) should be seen as orthogonal and complementary to LZN, rather than as competitors. In particular, LZN decoders that map latents to data samples can be instantiated using any generative model. We demonstrate this in § 3 and 5. This allows LZN to leverage the strengths of existing generative models within its unified framework.

#### 2.5 Scope of Experiments

While the framework is general, this paper focuses specifically on the image domain. We present three case studies: (1) generation (§ 3), (2) unsupervised representation learning (§ 4), and (3) joint classification and generation (§ 5). These case studies are arranged in order of increasing complexity: the first enhances a single task, the second solves a task using only LZN without any other external objectives, and the third tackles multiple tasks simultaneously within the same framework.

### 3 Case Study 1: Unconditional Generative Modeling

Approach (Fig. 4). Since LZN latents C (X) can be computed using a standalone encoder without introducing new training objectives (§ 2.2.1), they can be easily integrated into existing pipelines as additional network inputs, without modifying the original loss function. We apply this idea to unconditional generative models, where the generator serves as the LZN decoder. We modify the decoder by adding an extra input to accept the LZN latents. Both the encoder and decoder are trained jointly using the original generative loss. In this setup, latent alignment (§ 2.2.2) is not used. Importantly, the encoder is only used during training to compute C (X). During inference, latents are sampled from the Gaussian prior and passed directly to the decoder, maintaining the original model’s inference efficiency. Please see § C.1 for detailed pseudocode of the training and generation processes.

Why it helps. LZN latents provide unique representations for each image. This pushes the generator (decoder) to depend more directly on the latent for reconstruction, making the image distribution conditioned on the latent more deterministic. As a result, the generative objective becomes easier to optimize. Our experiments later confirm that LZN latents indeed capture useful features for generation.

- Table 1: Unconditional image generation quality scores across four datasets. The best results are in gray box . Applying LZN to generative models improves RF on most image quality metrics. RF is a SoTA method; due to space constraints, we omit additional methods—see [54] for extensive comparisons between RF and others. Note that Inception Score (IS) is best suited for natural images like CIFAR10, though we report it for all datasets for completeness.

|Algo.<br><br>|CIFAR10 (32 × 32) FID sFID IS Precision Recall CMMD Recon|AFHQ-Cat (256 × 256) FID sFID IS Precision Recall CMMD Recon|
|---|---|---|
| |↓ ↓ ↑ ↑ ↑ ↓ ↓<br><br>|↓ ↓ ↑ ↑ ↑ ↓ ↓|
|RF RF+LZN|2.76 4.05 9.51 0.70 0.59 0.0360 0.83 2.59 3.95 9.53 0.70 0.59 0.0355 0.41<br><br>|6.08 49.60 1.80 0.86 0.28 0.5145 17.92 5.68 49.32 1.96 0.87 0.30 0.3376 10.29<br><br>|

|Algo.|CelebA-HQ (256 × 256) FID sFID IS Precision Recall CMMD Recon<br><br>|LSUN-Bedroom (256 × 256) FID sFID IS Precision Recall CMMD Recon|
|---|---|---|
| |↓ ↓ ↑ ↑ ↑ ↓ ↓|↓ ↓ ↑ ↑ ↑ ↓ ↓<br><br>|
|RF RF+LZN|6.95 10.61 2.91 0.76 0.42 1.0276 26.20<br><br>7.17 10.33 2.92 0.76 0.45 0.4901 15.90<br><br><br>|6.25 16.22 2.18 0.60 0.40 0.5218 48.72 5.95 17.84 2.22 0.59 0.41 0.4843 37.01<br><br>|

[Figure 16]

[Figure 17]

[Figure 18]

- Figure 5: Generated images of RF+LZN on AFHQ-Cat, CelebA-HQ, LSUN-Bedroom. More in § C.

Related work. RCG [44] and Diffusion Autoencoder (DA) [67] also aim to improve image generation using unsupervised representations. RCG leverages a pre-trained representation model, while DA uses an encoder trained jointly with the generator. Similar to LZN, these representations are provided as additional inputs to the generative model. However, because their representation lacks distributional constraints, both methods require training an auxiliary generative model over the representations to enable unconditional generation. In contrast, LZN enforces a Gaussian representation by construction, eliminating this extra step and allowing everything to be trained end-to-end in one stage.

Results. We plug LZN latents into the seminal rectified flow (RF) models [54], which is closely related to diffusion models [32, 77, 79]. RF achieved SoTA image generation performance on several datasets [54, 41], and has been used in some latest Stable Diffusion [23] and AR models [53, 52].

We follow the experimental setup of RF [54], evaluating on four datasets: CIFAR10, AFHQ-Cat, CelebA-HQ, and LSUN-Bedroom. The latter three are high-resolution (256 × 256). In addition to standard metrics—FID [31], sFID [59], IS [74], precision [40], recall [40], and CMMD [33]—we also report reconstruction error (the ℓ2 distance between an image and its reconstruction), which is relevant for applications like image editing via latent manipulation [76, 50]. Results are shown in Tab. 1, with three key findings: (1) LZN latents improve image quality. During inference, the only difference in RF+LZN is the inclusion of an additional latent drawn from a Gaussian distribution. This improvement indicates that the decoder effectively leverages LZN latents for meaningful generative features. (2) LZN significantly reduces reconstruction error across all datasets, further confirming that its latents capture essential image information. (3) While unconditional generation is important, its image quality often trails that of conditional generation [44]. Compared to RF’s CIFAR10 results in Tab. 4, we find that LZN substantially reduces the FID gap between conditional and unconditional generation by 59%, and even outperforms conditional RF in sFID and reconstruction error.

See Fig. 5 for some generated images. Due to space constraints, please refer to § C for more details on implementation, datasets, metrics, and additional results such as more generated images and ablation studies.

### 4 Case Study 2: Unsupervised Representation Learning

Related work. Contrastive learning is a popular approach to unsupervised representation learning [10, 11, 28, 12, 14, 26, 13], which pulls similar images (e.g., augmentations of the same image) together in the representation space. A central challenge is avoiding collapse, where all inputs are mapped to the same representation. Common solutions include pushing the representations of dissimilar samples from large batches [10] or memory banks [28] away, or adopting architectural designs that prevent collapse [26, 13, 28]. Other unsupervised representation learning approaches [75, 42] include masked image modeling [43, 4, 27, 64] and training on other auxiliary tasks [85, 60, 66, 25].

Approach (Fig. 6). Inspired by contrastive learning, we train an image encoder using Align(X,Y) (§ 2.2.2), where X = {xi}ni=1 and Y = {yi}ni=1 with mapping ki = i contain image pairs (xi,yi) of random augmentations of the same image. Unlike traditional contrastive methods, our approach inherently avoids collapse: different images are mapped to distinct latent zones by design, eliminating the need for large memory banks or specialized architectures. Notably, only a single LZN encoder

[Figure 19]

[Figure 20]

[Figure 21]

Data augmentation

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

|Latent alignment|
|---|

Latent space Latent space

- Figure 6: LZN for unsupervised representation learning (§ 4). During training, each image batch undergoes two sets of data augmentations, and latent zones for each set are computed using the same encoder. We then apply latent alignment (§ 2.2.2) to train the encoder. At inference, we can use the LZN latents, the encoder outputs (i.e., anchor points), or intermediate encoder outputs (§ D.4). The latter two options avoid the costly latent computation process.

Table 3: Classification accuracy on CIFAR10. Baseline results are from [39]. “RF+LZN (no gen)” refers to “RF+LZN” with the RF loss for generation disabled. The horizontal line separates baselines that perform worse or better than our RF+LZN. Note that these results refer to training purely on the CIFAR10 dataset (without pretraining or external data).

- Table 2: Classification accuracy on ImageNet by training a linear classifier on the unsupervised representations. Methods marked with § are based on contrastive learning.7 The horizontal line separates baselines that perform worse or better than our LZN. All methods use the ResNet-50 architecture [29] for fair comparison.8

Algorithm Acc↑

Algorithm Top-1 Acc↑ Top-5 Acc↑

|VGG16 ResNet18 ResNet50 ResNet101<br><br>RegNetX_200MF<br><br>RegNetY_400MF MobileNetV2<br><br><br>|92.64%<br>93.02%<br><br>93.62%<br><br>93.75%<br>94.24%<br><br><br>94.29%<br><br><br>94.43%<br>|
|---|---|
|ResNeXt29(32x4d) ResNeXt29(2x64d) SimpleDLA DenseNet121 PreActResNet18 DPN92 DLA<br><br>|94.73%<br><br>94.82%<br><br>94.89%<br>95.04%<br><br><br>95.11%<br><br><br>95.16% 95.47%<br>|

|InstDisc§ [86] BigBiGAN [21] LocalAgg§ [92] MoCo§ [28] PIRL§ [58] CPC v2§ [30] CMC§ [82] SimSiam§[13] SimCLR§ [10]|54.0 [28] 56.6 [28] 58.8 [28] 60.2 [10] 63.6 [10] 63.8 [10] 66.2 [26]<br><br>68.1 [13]<br><br>69.3 [10]<br><br><br>|85.3 [10] 87.0 [26] 89.0 [10]|
|---|---|---|
|MoCo v2§ [12] SimCLR v2§ [11] BYOL§ [26] DINO§ [9]|71.7 [12] 71.7 [11]<br><br>74.3 [26]<br><br>75.3 [9]<br><br><br>|91.6 [26] -|

RF+LZN (no gen) 93.59% RF+LZN 94.47%

LZN 69.5 89.3

for both X and Y is needed and decoders are not required. See § D.1 for pseudocode of the training process.

Results. We follow the canonical setting [10, 26, 28], where LZN is trained on the unlabelled ImageNet dataset using ResNet-50 [29] to learn representations. A linear classifier is then trained on top of these representations in a supervised manner, and its accuracy is evaluated on the ImageNet test set. The results are shown in Tab. 2. Note that these results are obtained without any pretraining or use of external data. We observe that LZN matches or outperforms several established methods in the field, including seminal approaches such as MoCo [28] (LZN outperforms it by 9.3%) and SimCLR [10] (LZN outperforms it by 0.2%). This is remarkable given that LZN is a new framework capable of supporting not only unsupervised representation learning but also other tasks (§ 3 and 5). However, there remains a significant gap to SoTA performance. We emphasize that our current results are not fully optimized: (1) Training iteration. The performance of LZN continues to improve rapidly with ongoing training (§ D), so we expect the gap to SoTA to narrow with full training. (2) Architecture. Prior work shows that more advanced architectures like ViT can improve the results significantly [14]. We leave these directions for future work.

- Due to space constraints, please refer to § D for more details on implementation, datasets, metrics, and additional results such as representation visualization and ablation studies.

- 7Note that we use the term contrastive learning broadly to refer not only to methods employing the traditional contrastive loss, but to all approaches that encourage relevant images to share similar representations; see § 4.
- 8It is known that better architectures [11] or training on larger datasets [63] can yield stronger results. To ensure a fair comparison, we include only methods reporting results with the ResNet-50 architecture trained on the ImageNet dataset. This excludes potentially stronger methods lacking ResNet-50 results on ImageNet, such as DINOv2 [63]. See Tab. 6 for additional baselines using other architectures.

Table 4: Conditional image generation quality on CIFAR10. The best results are in gray box . Applying LZN to generative models improves or matches RF on all metrics.

|Algo.|FID↓ sFID↓ IS↑ Precision↑ Recall↑ CMMD↓ Recon↓<br><br>|
|---|---|
|RF RF+LZN<br><br>|2.47 4.05 9.77 0.71 0.58 0.0253 0.69 2.40 3.99 9.88 0.71 0.58 0.0229 0.38<br><br>|

### 5 Case Study 3: Conditional Generative Modeling and Classification

Approach (Fig. 3). Building on § 3, we consider X = {xi}ni=1 and Y = {yi}mi=1, where each xi is a class label and yi is an image labeled as xk

. In addition to the image encoder and decoder, we introduce a label encoder-decoder pair. Since labels come from a finite set, both modules share a matrix A ∈ Rq×c of label anchor points, where q is the latent dimension and c is the number of classes. The encoder maps a one-hot label h to its anchor via Ah. The decoder recovers the class ID of a latent g by first applying FM to obtain its anchor FMA (g;1), and then computing its corresponding class latent zone arg maxFMA (g;1)T A, where FMA denotes FM over anchors in

i

- A. The training objective extends that of § 3 by adding Align(X,Y). After training, the model can perform both conditional and unconditional generation, as well as classification by design. See § E.1 for detailed pseudocode of the training, generation, and classification processes.

Related work. Joint classification and conditional generation are often achieved by augmenting generative models with classification losses or networks [61, 74], treating label inputs to the generator and outputs from the classifier as separate components. In contrast, LZN unifies label inputs and outputs within a shared latent space.

Results. Following the setting in § 3, we conduct experiments on CIFAR10. Image quality metrics are shown in Tab. 4, and classification accuracies are shown in Tab. 3. Key observations are: (1) LZN improves both image quality and reconstruction error. Similar to § 3, this confirms that LZN latents capture useful features for generation. (2) LZN achieves classification accuracy on par with SoTA. Tab. 3 includes SoTA classification accuracy from networks trained solely for classification. The fact that LZN, which jointly performs generation and classification and differs significantly from standard classification pipelines, can match SoTA performance is notable. Currently, LZN lags behind the best CIFAR10 result by 1%, potentially due to architectural factors: we use the RF encoder (§ E) without classification-specific optimization. With a better architecture design (as in other methods in Tab. 3), LZN could likely improve further. (3) Joint training on generation and classification improves both. This is evident from: (i) RF+LZN in Tab. 4 showing better generation quality than in Tab. 1; and (ii) “RF+LZN” achieving higher classification accuracy than “RF+LZN (no gen)” in Tab. 3. These results support our motivation from § 1 that different ML tasks can benefit from each other, and demonstrate that LZN is a promising unified framework for achieving this synergy.

- Due to space constraints, please refer to § E for more details on implementation, datasets, metrics, and additional results such as generated images and ablation studies.

### 6 Limitations and Future Work

(1) Training efficiency. Training LZN requires backpropagating through the FM trajectory (§ 2.2.1), which is computationally expensive. In § A, we describe several optimization strategies we implemented to mitigate this cost. To further improve efficiency, we observe an interesting parallel between training LZN and training large language models (LLMs) (see § G), suggesting that some efficient training techniques developed for LLMs may be applicable here. (2) Pure generative modeling. While LZN is fundamentally capable of generative modeling without any auxiliary losses (see § G),

- in § 3, we only demonstrate how it can enhance existing generative models. Exploring how to fully leverage LZN for standalone generative modeling remains an open direction for future work. (3) Improving performance. Although LZN achieves competitive results in unsupervised representation learning (§ 4) and classification (§ 5), there remains a gap to the SoTA. Bridging this gap is an interesting direction. One promising avenue is to incorporate well-established improvements from the literature that we have not yet adopted, such as more advanced architectural designs, as discussed
- in § 4 and 5. (4) Multi-modality and multi-tasks. In this paper, we focus primarily on image-based applications and at most two tasks simultaneously (§ 5). However, LZN is designed to be extensible: by incorporating additional encoders and decoders, it can naturally support more modalities and perform multiple tasks concurrently (Fig. 1). We leave this exploration to future work.

### Acknowledgement

The authors would like to thank Sangyun Lee for suggesting related work and for helpful discussions, as well as the anonymous reviewers for their valuable feedback. Xuefei Ning acknowledges the support by the National Natural Science Foundation of China (No. 62506197). In addition, the authors gratefully acknowledge Cat Cookie and Cat Doudou for graciously lending their adorable faces for Figs. 2 to 4 and 6.9

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Devansh Arpit, Aadyot Bhatnagar, Huan Wang, and Caiming Xiong. Momentum contrastive autoencoder: Using contrastive learning for latent space distribution matching in wae. arXiv preprint arXiv:2110.10303, 2021.
- [3] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a jointembedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023.
- [4] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021.
- [5] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.
- [6] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.
- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [8] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In Proceedings of the European conference on computer vision (ECCV), pages 132–149, 2018.
- [9] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [10] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PmLR, 2020.
- [11] Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E Hinton. Big self-supervised models are strong semi-supervised learners. Advances in neural information processing systems, 33:22243–22255, 2020.
- [12] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020.
- [13] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15750–15758, 2021.

9From https://www.kaggle.com/datasets/fjxmlzn/cat-cookie-doudou released in [47].

- [14] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9640–9649, 2021.
- [15] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8188–8197, 2020.
- [16] Quan Dao, Hao Phung, Binh Nguyen, and Anh Tran. Flow matching in latent space. arXiv preprint arXiv:2307.08698, 2023.
- [17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
- [18] Li Deng. The mnist database of handwritten digit images for machine learning research [best of the web]. IEEE signal processing magazine, 29(6):141–142, 2012.
- [19] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.
- [20] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In Proceedings of the IEEE international conference on computer vision, pages 1422–1430, 2015.
- [21] Jeff Donahue and Karen Simonyan. Large scale adversarial representation learning. Advances in neural information processing systems, 32, 2019.
- [22] Alexey Dosovitskiy, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. Discriminative unsupervised feature learning with convolutional neural networks. Advances in neural information processing systems, 27, 2014.
- [23] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [24] Michael Fuest, Pingchuan Ma, Ming Gui, Johannes Schusterbauer, Vincent Tao Hu, and Bjorn Ommer. Diffusion models and representation learning: A survey. arXiv preprint arXiv:2407.00783, 2024.
- [25] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. arXiv preprint arXiv:1803.07728, 2018.
- [26] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.
- [27] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.
- [28] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.
- [29] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

- [30] Olivier Henaff. Data-efficient image recognition with contrastive predictive coding. In International conference on machine learning, pages 4182–4192. PMLR, 2020.
- [31] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [32] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [33] Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar. Rethinking fid: Towards a better evaluation metric for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9307–9315, 2024.
- [34] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196, 2017.
- [35] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.
- [36] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020.
- [37] Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. Advances in neural information processing systems, 31, 2018.
- [38] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- [39] kuangliu. Train cifar10 with pytorch. https://github.com/kuangliu/pytorch-cifar, 2025.
- [40] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019.
- [41] Sangyun Lee, Zinan Lin, and Giulia Fanti. Improving the training of rectified flows. Advances in Neural Information Processing Systems, 37:63082–63109, 2024.
- [42] Mufan Li, Mihai Nica, and Dan Roy. The neural covariance sde: Shaped infinite depthand-width networks at initialization. Advances in Neural Information Processing Systems, 35:10795–10808, 2022.
- [43] Tianhong Li, Huiwen Chang, Shlok Mishra, Han Zhang, Dina Katabi, and Dilip Krishnan. Mage: Masked generative encoder to unify representation learning and image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2142–2152, 2023.
- [44] Tianhong Li, Dina Katabi, and Kaiming He. Return of unconditional generation: A selfsupervised representation generation method. Advances in Neural Information Processing Systems, 37:125441–125468, 2024.
- [45] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281, 2023.
- [46] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017.

- [47] Zinan Lin, Sivakanth Gopi, Janardhan Kulkarni, Harsha Nori, and Sergey Yekhanin. Differentially private synthetic data via foundation model apis 1: Images. arXiv preprint arXiv:2305.15560, 2023.
- [48] Zinan Lin, Ashish Khetan, Giulia Fanti, and Sewoong Oh. Pacgan: The power of two samples in generative adversarial networks. Advances in neural information processing systems, 31, 2018.
- [49] Zinan Lin, Vyas Sekar, and Giulia Fanti. Why spectral normalization stabilizes gans: Analysis and improvements. Advances in neural information processing systems, 34:9625–9638, 2021.
- [50] Zinan Lin, Kiran Thekumparampil, Giulia Fanti, and Sewoong Oh. Infogan-cr and modelcentrality: Self-supervised model training and selection for disentangling gans. In international conference on machine learning, pages 6127–6139. PMLR, 2020.
- [51] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [52] Enshu Liu, Qian Chen, Xuefei Ning, Shengen Yan, Guohao Dai, Zinan Lin, and Yu Wang. Distilled decoding 2: One-step sampling of image auto-regressive models with conditional score distillation. arXiv preprint arXiv:2510.21003, 2025.
- [53] Enshu Liu, Xuefei Ning, Yu Wang, and Zinan Lin. Distilled decoding 1: One-step sampling of image auto-regressive models with flow matching. arXiv preprint arXiv:2412.17153, 2024.
- [54] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [55] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.
- [56] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022.
- [57] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpmsolver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022.
- [58] Ishan Misra and Laurens van der Maaten. Self-supervised learning of pretext-invariant representations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6707–6717, 2020.
- [59] Charlie Nash, Jacob Menick, Sander Dieleman, and Peter W Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021.
- [60] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In European conference on computer vision, pages 69–84. Springer, 2016.
- [61] Augustus Odena, Christopher Olah, and Jonathon Shlens. Conditional image synthesis with auxiliary classifier gans. In International conference on machine learning, pages 2642–2651. PMLR, 2017.
- [62] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.
- [63] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [64] Druv Pai, Ziyang Wu Wu, Sam Buchanan, Yaodong Yu, and Yi Ma. Masked completion via structured diffusion with white-box transformers. International Conference on Learning Representations, 2023.

- [65] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11410–11420, 2022.
- [66] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2536–2544, 2016.
- [67] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10619–10629, 2022.
- [68] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [69] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.
- [70] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
- [71] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [72] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021.
- [73] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019.
- [74] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.
- [75] Michael Sander, Pierre Ablin, and Gabriel Peyré. Do residual neural networks discretize neural ordinary differential equations? Advances in Neural Information Processing Systems, 35:36520–36532, 2022.
- [76] Yujun Shen, Jinjin Gu, Xiaoou Tang, and Bolei Zhou. Interpreting the latent space of gans for semantic face editing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9243–9252, 2020.
- [77] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.
- [78] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023.
- [79] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.
- [80] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [81] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016.

- [82] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16, pages 776–794. Springer, 2020.
- [83] Arash Vahdat and Jan Kautz. Nvae: A deep hierarchical variational autoencoder. Advances in neural information processing systems, 33:19667–19679, 2020.
- [84] Geoffrey I. Webb. Occam’s Razor, pages 735–735. Springer US, Boston, MA, 2010.
- [85] Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, and Christoph Feichtenhofer. Masked feature prediction for self-supervised visual pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14668–14678, 2022.
- [86] Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3733–3742, 2018.
- [87] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.
- [88] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.
- [89] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14, pages 649–666. Springer, 2016.
- [90] Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, et al. mgte: Generalized long-context text representation and reranking models for multilingual text retrieval. arXiv preprint arXiv:2407.19669, 2024.
- [91] Lin Zhao, Tianchen Zhao, Zinan Lin, Xuefei Ning, Guohao Dai, Huazhong Yang, and Yu Wang. Flasheval: Towards fast and accurate evaluation of text-to-image diffusion generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16122–16131, 2024.
- [92] Chengxu Zhuang, Alex Lin Zhai, and Daniel Yamins. Local aggregation for unsupervised learning of visual embeddings. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6002–6012, 2019.

### Appendix Contents

- A More Details on Latent Computation (§ 2.2.1) 18

- A.1 The Desired Properties . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.2 More Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.3 Efficiency Optimizations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- B More Details on Latent Alignment (§ 2.2.2) 19

- B.1 More Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B.2 Efficiency Optimizations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- C More Details and Results on Case Study 1 19

- C.1 Algorithm Pseudocode . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.2 More Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.3 More Experimental Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.4 More Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- D More Details and Results on Case Study 2 29

- D.1 Algorithm Pseudocode . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- D.2 More Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- D.3 More Experimental Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- D.4 More Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- E More Details and Results on Case Study 3 32

- E.1 Algorithm Pseudocode . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- E.2 More Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- E.3 More Experimental Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- E.4 More Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34

- F Extended Discussions on Related Work 38
- G Extended Discussions on Limitations and Future Work 38

### A More Details on Latent Computation (§ 2.2.1)

#### A.1 The Desired Properties

In this section, we explain in more detail why the construction in § 2.2.1 (approximately) satisfies the two desired properties.

- • Prior distribution is Gaussian. By definition, the distribution of latent z ∼ Uniform{z1,...,zn} is induced by (1) drawing a ∼ Uniform{a1,...,an} and ϵ ∼ N (0,I),

(2) computing s1−g = (1 − g)a + gϵ, and (3) computing IFMx (s1−g;0). In the above process, s1−g ∼ π1−g by definition. Due to the property of FM discussed in § 2.2.1, IFMx (s1−g;0) ∼ π0 when s1−g ∼ π1−g. Therefore, the latent z ∼ π0 = N (0,I).

- • Disjoint latent zones. Each latent point z ∼ N (0,I) can be uniquely map to one of {a1,...,an} through the defined FM. We define the latent zone of i-th sample as the set of latents that map to ai: Zi = {z : FMx (z;1) = ai}. The probability that the latent computed through Eq. (1) falls in the incorrect latent zone can then be defined as P(IFMx (ai,ϵ;0) ∈ Zj) for i ̸= j, where the probability is over the randomness of ϵ ∼ N (0,I). We can see that this probability can be made arbitrarily small by choosing a sufficiently small g. This is because that to make the latent fall in

the incorrect latent zone, we need to have ∥ϵ∥ on the scale of (1−g)(gai−aj) , whose probability → 0 when g → 0. To make this intuition more precise, we give the closed form of this probability for a toy one-dimensional case below.

Theorem 1. Assume that there are n = 2 samples x1,x2, with their anchor points a1 = −1,a2 = 1. We have

g − 1 g

, (3) where Φ is the CDF function of the standard Gaussian distribution.

P(IFMx (a1,ϵ;0) ∈ Z2) = P(IFMx (a2,ϵ;0) ∈ Z1) = Φ

Proof. With a slight abuse of notation, we define FMx (s;t1,t2) = s+ τ t=2 t

V (sτ,τ)dτ as following the FM trajectory from t1 to t2. We generalize the latent zone definition above to all time steps as the latents at time step t that map to the anchor point ai: Zit = {z : FMx (z;t,1) = ai}. Due to symmetry, we know that Z1t = (−∞,0) and Z2t = (0,∞). Therefore, we have

1

1 − g g

g − 1 g

P(IFMx (a1,ϵ;0) ∈ Z2) = P(−(1 − g) + gϵ > 0) = P ϵ >

, where Φ(·) is the CDF function of Gaussian distribution. Similarly, we can get that P(IFMx (a2,ϵ;0) ∈ Z1) = Φ g−g1 .

= Φ

| |
|---|

#### A.2 More Implementation Details

We find that in the training or inference of some tasks benefit from using a more concentrated latent distribution:

zi = C (x1,...,xn)i ≜ IFMx (ai,αϵi;0), (4)

where ϵi ∼ N(0,I) and α ∈ [0,1) is the scaling factor. Similar techniques have been used in prior generative models for improving sample quality [35, 37, 6, 83].

#### A.3 Efficiency Optimizations

We introduce a series of efficiency optimization techniques so that the training of LZN can scale up to large models and large batch sizes.

Minibatch approximation (reducing memory and computation cost). By design, latent computation requires using all samples, which is infeasible for large datasets. In practice, we approximate this by using only the current minibatch as x1,...,xn, which significantly reduces memory and computation cost.

Note that this approximation has nuanced implications on the two desired properties discussed in § 2.2.1.

- • Minibatch approximation still preserves the Gaussian prior. LZN ensures that the latent distribution within each minibatch is approximately N (0,I). As a result, the overall latent distribution becomes a mixture of Gaussians with the same parameters, which is still N (0,I). Therefore, the global prior remains valid under the minibatch approximation.
- • However, minibatch approximation violates the disjoint zone property. This is because the latent zones of each sample now depends on other samples in the same batch, which could change across different batches. Despite this approximation, our experiments show it performs well.
- • In generative modeling (§ 3 and 5), the latent does not need perfect zone disjointness—as long as it provides some information about the input sample, it can help reduce the variance needed to learn by the generative model (rectified flow in our case) and improve the generation quality.
- • In representation learning (§ 4), latent alignment occurs within a single batch. Thus, inconsistency across batches is irrelevant.
- • In classification (§ 5), we only need to map samples within a single batch to the latent zones of labels. Thus, inconsistency across batches is irrelevant.

That said, larger batch sizes can improve the accuracy of latent zones and thus improve performance (§ 5).

#### Custom gradient checkpointing (reducing memory cost). In PyTorch, forward passes store interme-

diate results for use in backpropagation, incurring significant memory cost. Gradient checkpointing10 reduces memory usage (with the cost of extra computation) by selectively discarding intermediates in the forward pass and recomputing them during the backward pass. This technique is typically applied within neural networks. In our case, we discover that the main memory bottleneck lies in latent computation, which has memory complexity O(n2qr), where n is the number of samples, q the latent dimension, and r the solver steps. We design a custom strategy that skips storing velocity computations and retains only the latent trajectories st. This reduces memory complexity to O(nqr), which makes the training far more manageable.

Latent parallelism (making training scalable with multi-GPU). For the same reason discussed above, the main computation overhead also lies in latent computation. A natural idea is to parallelize it with multi-GPU. We partition the data samples across GPUs, and each GPU computes anchor points for its assigned subset. These anchor points are then broadcast to all GPUs, allowing each to compute latents for its own samples using the complete set of anchors. To ensure that gradients can propagate back correctly through the anchor points to the originating GPUs, we use the undocumented PyTorch function torch.distributed.nn.functional.all_gather, which—unlike the standard torch.distributed.all_gather—maintains gradient flow to the original sources.

### B More Details on Latent Alignment (§ 2.2.2)

B.1 More Implementation Details

Optionally, we can apply a logarithm to the assignment probability to make the loss resemble a standard cross-entropy formulation. In that case, our proposed alignment objective is:

m

Align(X,Y) ≜ max

i|sit . (5)

log P ak

max

t∈{tu,...,tr}

i=1

- B.2 Efficiency Optimizations We apply the same efficiency optimizations in § A.3 in latent alignment.

C More Details and Results on Case Study 1

- C.1 Algorithm Pseudocode

Fig. 7 and Fig. 8 show side-by-side comparisons of the training and generation processes of RF and RF+LZN.

10https://pytorch.org/docs/stable/checkpoint.html

Algorithm 2: RF+LZN training Input :Training set: X

Algorithm 1: RF training Input :Training set: X

Decoder: Dx

Encoder: Ex (used by C) Number of iterations: T Batch size: B

Decoder: Dx Number of iterations: T Batch size: B

- 1 for iteration ← 1,...,T do

- 2 x1,...,xB ← Draw samples from X
- 3 z1,...,zB ← C (x1,...,xB)

- 4 ϵ1,...,ϵB ← Gaussian noise
- 5 t1,...,tB ← Random RF timesteps
- 6 ξi ← (1 − ti)ϵi + tixi
- 7 Training using Dx(ξi; zi )

- 1 for iteration ← 1,...,T do

- 2 x1,...,xB ← Draw samples from X
- 3 ϵ1,...,ϵB ← Gaussian noise
- 4 t1,...,tB ← Random RF timesteps
- 5 ξi ← (1 − ti)ϵi + tixi
- 6 Training using Dx (ξi)

Figure 7: Comparison between the training processes of RF and RF+LZN. Left: A simplified illustration of the standard RF [54] training process. In each iteration, a batch of real samples and a batch of Gaussian noise are drawn and interpolated to produce noisy inputs, which are then passed through the decoder network to compute the loss. Right: A simplified illustration of the RF+LZN training process. The key differences are highlighted in gray: we compute LZN latents for the samples using the method in § 2.2.1, and provide these latents as an additional input to the RF decoder.

Algorithm 4: RF+LZN generation Input :Decoder: Dx

Algorithm 3: RF generation Input :Decoder: Dx

- 1 ξ ← Gaussian noise
- 2 z ← Gaussian noise

- 3 Generated sample ← Dx(ξ; z )

- 1 ξ ← Gaussian noise
- 2 Generated sample ← Dx (ξ)

Figure 8: Comparison between the generation processes of RF and RF+LZN. Left: A simple illustration of the standard RF generation process [54]. The decoder takes Gaussian noise as input and generates a sample. The actual process is iterative, but we leave out the steps for simplicity and only show the starting input (Gaussian noise) and the final output (the generated image). Right: A simple illustration of the RF+LZN generation process. The main differences are shown in gray: we sample extra LZN latents from Gaussian noise and use them as additional inputs to the RF decoder during the iterative generation process.

#### C.2 More Implementation Details Architecture.

- • Decoder. The only change to the RF architecture [54] is concatenating the LZN latent with the timestep embedding.
- • Encoder. We extend the UNet encoder in RF [54] by connecting the output of each ResNet block with a latent transformation block. The sum of the outputs of the latent transformation blocks forms the LZN latent. Each latent transformation block consists of: (1) a 1×1 convolution that projects the ResNet output to 20 channels, reducing dimensionality; and (2) a small MLP with a 200-dimensional hidden layer that outputs the latent from the flattened convolution output.

#### C.3 More Experimental Settings Datasets.

- • CIFAR10 (32 × 32) [38] contains 50000 training images and 10000 test images of 10 classes of objects. We only utilize the training set for this experiment.
- • AFHQ-Cat (256 × 256) [15] contains 5153 catimages.

- • CelebA-HQ (256 × 256) [34] contains 30000 face images.
- • LSUN-Bedroom (256 × 256) [87] contains 3033042 bedroom images.

#### Metrics.

- • FID [31] and sFID [59] evaluate the similarity between real and generated images by projecting both into the latent space of a pretrained network (e.g., Inception-v3 [81]), fitting each set of latents with Gaussian distributions, and computing their Wasserstein-2 distance. The key difference between FID and sFID is the feature layer used: FID uses pooled features, while sFID uses intermediate features, making it more sensitive to spatial details.
- • Inception score (IS) [74] measures image quality by assessing both the quality of each image (how confidently a classifier predicts a class) and diversity across all images (coverage over different classes). Since the classifier is trained on ImageNet [17], IS is best suited for natural image datasets like CIFAR10. We report IS for all datasets for completeness.
- • Precision and recall [40] evaluate the quality and coverage of generated images. Intuitively, precision measures the fraction of generated images that are close to real ones, while recall measures the fraction of real images that are close to the generated ones.
- • CMMD [33] measure the MMD distances between the CLIP embeddings of the real images and generated images. Compared to FID, it is reported to align better with human preference and have better sample efficiency.
- • Reconstruction error measures how well a generative model can reconstruct an input image. This reflects the model’s representational power and is crucial for applications like image editing, where edits are made by modifying the image’s latent representation []. For RF, we first apply the inverse ODE to map the image to its latent representation, then use the forward ODE to

reconstruct the image, and compute the ℓ2 distance between the original and reconstructed images. For RF+LZN, we add an initial step: compute the image’s LZN latent C (X) and feed it into the RF latent computation process as an additional input.

Following the convention [54, 78, 48, 91, 49], the metrics are all computed using the training set of the dataset.

For FID, sFID, IS, precision, recall, and CMMD, we subsample the training set and generate the same number of samples to compute the metrics. The number of samples are:

- • CIFAR10: 50000 (the whole training set).
- • AFHQ-Cat: 5120, the largest multiple of the batch size (256) that is less than or equal to the training set size (5153).
- • CelebA-HQ: 29952, the largest multiple of the batch size (256) that is less than or equal to the training set size (30000).
- • LSUN-Bedroom: 29952, the largest multiple of the batch size (256) that is less than or equal to

30000. We limit the number of samples to 30000 so that the computation cost of the metrics are reasonable.

For reconstruction error, we randomly sample a batch of images (2000 for CIFAR10 and 256 for the other datasets) from the training set. Each image is reconstructed 20 times (note that the LZN latents C (X) have randomness). We report the average metric over all reconstructions.

Note that for all the random subsampling procedures mentioned above, we ensure the sampled sets are consistent between RF and RF+LZN, so that the resulting metrics are directly comparable.

Sampler. RF requires a sampler to numerically solve the ODE (integral) trajectory for sample generation. For both RF and RF+LZN, we use the RK45 sampler from RF [54], which adaptively determines the number of steps. In § C.4, we also analyze the effect of varying the number of sampling steps using the Euler sampler [].

Hyperparameters.

- • CIFAR10
- • RF:
- • Batch size: 2000
- • Optimizer: Adam
- • Decoder learning rate: 0.001

- • Gradient clipping: 1.0
- • Number of parameters in decoder: 61804419
- • RF+LZN:
- • Batch size: 2000
- • Optimizer: Adam
- • Decoder learning rate: 0.001
- • Encoder learning rate: 0.000025
- • Gradient clipping: 1.0
- • Latent dimension: 200
- • Number of parameters in decoder: 61906819
- • Number of parameters in encoder: 49790260
- • AFHQ-Cat
- • RF:
- • Batch size: 256
- • Optimizer: Adam
- • Decoder learning rate: 0.0002
- • Gradient clipping: 1.0
- • Number of parameters in decoder: 65574549
- • RF+LZN:
- • Batch size: 256
- • Optimizer: Adam
- • Decoder learning rate: 0.0002
- • Encoder learning rate: 0.000002
- • Gradient clipping: 1.0
- • Latent dimension: 200
- • Number of parameters in decoder: 65676949
- • Number of parameters in encoder: 87768896
- • CelebA-HQ
- • RF:
- • Batch size: 256
- • Optimizer: Adam
- • Decoder learning rate: 0.0002
- • Gradient clipping: 1.0
- • Number of parameters in decoder: 65574549
- • RF+LZN:
- • Batch size: 256
- • Optimizer: Adam
- • Decoder learning rate: 0.0002
- • Encoder learning rate: 0.000004
- • Gradient clipping: 1.0
- • Latent dimension: 200
- • Number of parameters in decoder: 65676949
- • Number of parameters in encoder: 87768896
- • LSUN-Bedroom
- • RF:
- • Batch size: 256
- • Optimizer: Adam
- • Decoder learning rate: 0.0002
- • Gradient clipping: 1.0
- • Number of parameters in decoder: 65574549
- • RF+LZN:
- • Batch size: 256

[Figure 28]

- Figure 9: Generated images of RF on CIFAR10.
- • Optimizer: Adam
- • Decoder learning rate: 0.0002
- • Encoder learning rate: 0.000002
- • Gradient clipping: 1.0
- • Latent dimension: 200
- • Number of parameters in decoder: 65676949
- • Number of parameters in encoder: 87768896

Computation cost. Excluding the computation cost of periodic evaluation (i.e., only counting the computation cost of model training), each RF+LZN experiment takes:

- • CIFAR10: 8 hours on 16 A100 (40 GB) GPUs.
- • AFHQ-Cat: 10 hours on 32 A100 (40 GB) GPUs.
- • CelebA-HQ: 58 hours on 32 A100 (40 GB) GPUs.
- • LSUN-Bedroom: 341 hours on 32 A100 (40 GB) GPUs.

#### C.4 More Results

Generated images. The generated images of RF and RF+LZN are in Figs. 9 to 16.

Ablation studies on FID implementation. It is known that subtle differences in FID implementation can result in different results [65]. In our main experiments, we use the implementation in consistency models [78]. In Tab. 5, we additionally show the FID using two other implementations: RF [54] and clean FID [65]. We can see that, while the numbers are different, the relative ranking across all three implementations is consistent. Especially, RF+LZN achieves the best FID in three out of four datasets.

Ablation studies on sampling steps. In this experiment, we use the Euler sampler with varying numbers of sampling steps. As shown in Fig. 17, RF+LZN generally achieves better FID than the RF baseline across most settings. Notably, in the only case where RF+LZN performs worse than RF in

[Figure 29]

Figure 10: Generated images of RF+LZN on CIFAR10.

Table 5: FID with different implementations for unconditional image generation. “CM” denotes consistency models [78]; “RF” denotes Rectified Flow [54]; “clean” denotes clean FID [65]. The best results are in gray box .

|Algo.<br><br>|CIFAR10 (32 × 32)<br><br>FID (clean) FID (RF) FID (CM)|AFHQ-Cat (256 × 256)<br><br>FID (clean) FID (RF) FID (CM)|
|---|---|---|
| |↓ ↓ ↓|↓ ↓ ↓|
|RF RF+LZN<br><br>|3.18 2.77 2.76 3.05 2.61 2.59<br><br>|5.99 6.20 6.08 5.66 5.69 5.68<br><br>|

|Algo.|CelebA-HQ (256 × 256)<br><br>FID (clean) FID (RF) FID (CM)|LSUN-Bedroom (256 × 256)<br><br>FID (clean) FID (RF) FID (CM)|
|---|---|---|
| |↓ ↓ ↓<br><br>|↓ ↓ ↓|
|RF RF+LZN<br><br>|7.10 7.00 6.95 7.31 7.23 7.17<br><br>|6.39 6.25 6.25 5.88 5.87 5.95<br><br>|

Tab. 1, we observe that the underperformance occurs only at the highest number of sampling steps in the Euler sampler (Fig. 17c).

[Figure 30]

##### Figure 11: Generated images of RF on AFHQ-Cat.

[Figure 31]

##### Figure 12: Generated images of RF+LZN on AFHQ-Cat.

[Figure 32]

##### Figure 13: Generated images of RF on CelebA-HQ.

[Figure 33]

##### Figure 14: Generated images of RF+LZN on CelebA-HQ.

[Figure 34]

##### Figure 15: Generated images of RF on LSUN-Bedroom.

[Figure 35]

##### Figure 16: Generated images of RF+LZN on LSUN-Bedroom.

|Rectified Flow<br><br>Rectified Flow + LZN|
|---|

101

6 × 100

FID

- 3 × 100
- 4 × 100

101 102 103 NFE

|Rectified Flow<br><br>Rectified Flow + LZN|
|---|

- 2 × 101
- 3 × 101

FID

101

6 × 100

101 102 103 NFE

(a) CIFAR10.

|Rectified Flow<br><br>Rectified Flow + LZN|
|---|

FID

101

101 102 103 NFE

###### (b) AFHQ-Cat.

|Rectified Flow<br><br>Rectified Flow + LZN|
|---|

2 × 101

FID

101

6 × 100

101 102 103 NFE

(c) CelebA-HQ.

(d) LSUN-Bedroom.

Figure 17: FID vs. number of sampling steps in the Euler sampler. RF+LZN outperforms RF in most cases.

### D More Details and Results on Case Study 2

- D.1 Algorithm Pseudocode Alg. 5 shows the pseudocode of the training process.

After training, the encoder Ex can be used to obtain image representations. We provide several strategies for extracting these representations. Please see § D.4 for details.

- D.2 More Implementation Details

Architecture. To remain consistent with prior work [28, 10, 26], we use the ResNet-50 architecture [29] as the encoder for LZN. The only modification we make is replacing all batch normalization layers with group normalization. However, our early experiments indicate that this change does not lead to significant performance differences.

For the projection head following the ResNet-50 output, we use an MLP with one hidden layer, as in [10, 11].

Data augmentation. We follow the same data augmentation strategy as in [11].

Representation. Following prior work [10, 11], after training the ResNet-50, we discard the projection head and use only the ResNet-50 backbone to extract representations for training the linear classifier. As a result, obtaining representations from LZN in this way does not require going through the LZN latent computation process C, and thus has the same computational efficiency as baseline methods.

Objective. We use the version with log (Eq. (5)).

- D.3 More Experimental Settings

Datasets. We use the ImageNet dataset, which contains 1281167 training images and 50000 validation images. LZN is trained on the training set, and classification accuracy is evaluated on the validation set.

Hyperparameters.

- • Batch size: 8192
- • Optimizer: Adam
- • Learning rate: 8e-4
- • Gradient clipping: 1.0
- • Latent dimension: 256
- • Number of parameters: 24032832
- • α: 0.45

Computation cost. Excluding the computation cost of periodic evaluation (i.e., only counting the computation cost of model training), each LZN experiment takes 1800 hours on 128 A100 (40 GB) GPUs.

- D.4 More Results

More baselines. Tab. 6 shows the result with more baselines that are not using the ResNet-50 architecture.

Visualizing the learned representations. We take images from randomly selected 20 classes from the validation set of ImageNet and computed their embeddings using the trained LZN model. We chose the validation set to ensure that the results are not influenced by training set overfitting. We then projected these embeddings into a 2D space using t-SNE–a widely used method for visualizing high-dimensional representations, following seminal works such as SimCLR [10]. The resulting

11Note that we use the term contrastive learning broadly to refer not only to methods employing the traditional contrastive loss, but to all approaches that encourage relevant images to share similar representations; see § 4.

Algorithm 5: Unsupervised representation learning with LZN Input :Training set: X

Encoder: Ex Number of iterations: T Batch size: B

- 1 for iteration ← 1,...,T do

- 2 x1,...,xB ← Draw samples from X
- 3 x′i,x′′i ← Two random augmentations of xi
- 4 Training Ex using Align({x′1,...,x′B},{x′′1,...,x′′B})

Table 6: Classification accuracy on ImageNet by training a linear classifier on the unsupervised representations. Methods with § are based on contrastive learning.11 The horizontal line separates baselines that perform worse or better than our LZN. “R” means “ResNet”.

Algorithm Architecture Top-1 Acc↑ Top-5 Acc↑

|Colorization [89] Jigsaw [60] Exemplar [22] DeepCluster [8] CPC v1§ [62] RelativePosition [20]<br><br>InstDisc§ [86] Rotation [25] BigBiGAN [21] LocalAgg§ [92] MoCo§ [28] BigBiGAN [21] PIRL§ [58] CPC v2§ [30] CMC§ [82] SimSiam§[13] SimCLR§ [10]<br><br>|R101<br><br>R50w2×<br><br>R50w3× VGG R101<br><br><br>R50w2× R50 Rv50w4× R50 R50 R50 Rv50w4× R50 R50 R50 R50 R50<br><br>|39.6 [28] 44.6 [28] 46.0 [28] 48.4 [28] 48.7 [28] 51.4 [28]<br><br>54.0 [28]<br>55.4 [28]<br>56.6 [28] 58.8 [28]<br><br><br>60.2 [10]<br>61.3 [28] 63.6 [10] 63.8 [10] 66.2 [26]<br><br><br>68.1 [13]<br>69.3 [10]<br>|81.9 [10] 85.3 [10] 87.0 [26] 89.0 [10]<br><br>|
|---|---|---|---|
|MoCo v2§ [12] SimCLR v2§ [11] BYOL§ [26] DINO§ [9] DINO§ [9] DINO§ [9] DINO§ [9] DINO§ [9] I-JPEA [3] I-JPEA [3] I-JPEA [3] I-JPEA [3]<br><br>|R50 R50 R50 R50 ViT-S ViT-B/16 ViT-S/8 ViT-B/8 ViT-B/16 ViT-L/16 ViT-H/14 ViT-H/16448<br><br>|71.7 [12]<br><br>71.7 [11]<br><br>74.3 [26]<br>75.3 [9]<br><br><br>77.0 [9]<br>78.2 [9]<br>79.7 [9]<br>80.1 [9]<br><br>72.9 [3] 77.5 [3] 79.3 [3]<br><br>81.1 [3]<br><br><br>|91.6 [26] -<br><br>|

LZN R50 69.5 89.3

t-SNE plot is in Fig. 18. We can see that the samples from different classes are well-clustered. This suggests LZN learns meaningful image representations.

Ablation studies on representation choice. Prior work [10, 11] has shown that the choice of feature extraction layer significantly affects downstream performance. In particular, removing the projection head often improves results. Motivated by this, we explore various feature extraction strategies for LZN, which offers more flexibility due to its unique latent computation process (§ 2.2.1). Specifically, we compare the following methods:

- • With latent. Use the latent representation from LZN (see § 2.2.1) to train the classifier.
- • With latent (α = 0). Same as above but with α = 0 in the latent computation.
- • With head. Use the anchor point (i.e., encoder output before FM computation).
- • Without head. Use the ResNet backbone output (before the projection head).

993 859 298 553 672 971 27

30

20

231 306 706 496 558 784 239 578 55

10

0

−10

−20

906 175

−30

14 77

−20 0 20 40

- Figure 18: t-SNE visualization of LZN representations projected into 2D for 20 randomly selected ImageNet validation classes. Images from the same class form distinct clusters, indicating that LZN learns meaningful image representations.

The first two methods are specific to LZN, while the last two follow the design commonly used in prior contrastive learning work [10, 11].

The results are shown in Fig. 19. We observe the following:

- • LZN latent achieves the lowest prediction accuracy. As discussed in § A.2, the LZN latents are reliable only when computed over the full dataset. However, for efficiency, both training and inference rely on minibatches to approximate the latent representations. This approximation increases the size of the latent zones, leading to potential overlap between the zones of different samples across batches, which inevitably degrades downstream classification performance.
- • In comparison, LZN with α = 0 yields significantly higher accuracy. This improvement can be attributed to the reduced likelihood of overlap between latent zones when α = 0, making the resulting representations more distinct and less noisy.
- • The final two methods, “with head” and “without head”, do not involve latent computation and are therefore more efficient. Consistent with findings from prior contrastive learning studies [10], we observe that “without head” performs substantially better. As explained in [10], the projection head often discards important information—such as types of data augmentation—in order to minimize the training loss. In contrast, layers preceding the head might retain richer and more discriminative features, which are more useful for downstream classification tasks.

Ablation studies on the number of training steps. Fig. 20 shows classification accuracy over training iterations. Accuracy continues to improve rapidly at the end of training, suggesting that with more training, the gap between LZN and the SoTA could be further reduced.

65.6

| |11.9<br><br>54.9 55.9| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.6

0.5

Top-1accuracy

0.4

0.3

0.2

0.1

0.0

W/ latent W/ head W/ latent (α = 0)

W/o head Method

- Figure 19: LZN’s linear classification accuracy with different feature extraction methods. Note that this experiment uses fewer iterations (1060000) than the main experiment (5000000) and omits data augmentation when training the linear classifier (used in the main experiment), so the accuracies are lower than the main experiment.

0 1 2 3 4 5 Iterations 1e6

0.58

0.60

0.62

0.64

0.66

0.68

0.70

Top-1accuracy

- Figure 20: LZN’s linear classification accuracy vs. training iteration. The accuracy is still improving at a fast rate at the end of training. More training might further improve the result.

### E More Details and Results on Case Study 3

#### E.1 Algorithm Pseudocode

Alg. 6, Fig. 21, and Alg. 9 show the algorithm pseudocode of the training, generation, and classification process.

#### E.2 More Implementation Details Architecture.

- • Image decoder. For RF, we modify the original architecture [54] to include a one-hot encoding of the class label as an additional input, concatenated with the timestep embedding. For RF+LZN, we apply the same modification on top of the architecture described in § C. For unconditional generation, this one-hot encoding is deterministically derived from the LZN latent: given a LZN latent, we use the class label decoder to predict the class and then encode it as a one-hot vector. As a result, the decoder’s output remains fully determined by the LZN and RF latents, consistent with § 3. For conditional generation, this one-hot encoding is given as a condition, and LZN latents are sampled from the corresponding latent zone (as described in § 2.2.1).
- • Image encoder. Same as that of § C.

Label FM. The method in § 2.2.2 implicitly assumes a uniform distribution over class labels. However, due to sampling randomness during training, each batch may have an imbalanced class

Algorithm 6: RF+LZN training (with class labels) Input :Training set: labels X and images Y

Image decoder: Dy Image encoder: Ey (used by C) Label anchors: A (used by Align) Number of iterations: T Batch size: B

- 1 for iteration ← 1,...,T do

- 2 y1,...,yB ← Draw images from Y
- 3 z1,...,zB ← C (y1,...,yB)
- 4 ϵ1,...,ϵB ← Gaussian noise
- 5 t1,...,tB ← Random RF timesteps
- 6 ξi ← (1 − ti)ϵi + tiyi
- 7 Training using Align(X,{y1,...,yB}) and RF loss on Dy (ξi;zi) (a weighted loss between the two)

Algorithm 8: RF+LZN generation (unconditional)

Algorithm 7: RF+LZN generation (unconditional)

Input :Image decoder: Dy Label set: c1,...,cn Class ID: k (i.e., the class is ck)

Input :Image decoder: Dy

- 1 ξ ← Gaussian noise
- 2 z ← Gaussian noise
- 3 Generated sample ← Dy (ξ;z)

- 1 ξ ← Gaussian noise
- 2 z ← C ({c1,...,cn})k

- 3 Generated sample ← Dy (ξ;z)

Figure 21: The generation process of RF+LZN (with class labels). In this case, RF+LZN can simultaneously support unconditional and conditional generation. Left: Unconditional generation, where the LZN latent is drawn from the prior Gaussian distribution, which is exactly the same as Fig. 8. Right: Conditional generation, where the LZN latent is drawn from the latent zone of the corresponding class. The changes on top of unconditional generation are highlighted in gray.

distribution. To address this, we modify the π1 distribution when computing FMx (·) to be a weighted mixture of Dirac delta functions centered at Ex (xi), with weights corresponding to the fraction of class xi samples in the batch. During testing, we revert to a uniform prior, as the true class distribution of the batch is not available.

Objective. We use the version without log (Eq. (2)). We also tried the version with log (Eq. (5)) and did not observe a large difference in results.

E.3 More Experimental Settings Datasets. We use the CIFAR10 dataset discussed in § C. Metrics. In addition to the metrics discussed in § C, we evaluate on CIFAR10 classification accuracy. The accuracy is evaluated on CIFAR10 test set. Sampler. Same as § C. Hyperparameters.

- • CIFAR10
- • RF:
- • Batch size: 2000
- • Optimizer: Adam
- • Decoder learning rate: 0.002

Algorithm 9: RF+LZN classification Input :Image encoder: Dx

Label decoder: Dx Images: y1,...,yB

1 z1,...,zB ← C (y1,...,yB) 2 ci ← Dx (zi)

- Table 7: FID with different implementations for conditional image generation on CIFAR10. “CM” denotes consistency models [78]; “RF” denotes Rectified Flow [54]; “clean” denotes clean FID [65]. The best results are in gray box .

|Algo.|FID (clean)↓ FID (RF)↓ FID (CM)↓<br><br>|
|---|---|
|RF RF+LZN|2.85 2.50 2.47 2.70 2.42 2.40<br><br>|

- • Gradient clipping: 1.0
- • Number of parameters in decoder: 61809539
- • RF+LZN:
- • Batch size: 2000
- • Optimizer: Adam
- • Decoder learning rate: 0.002
- • Encoder learning rate: 0.00005
- • Label learning rate: 0.0001
- • Gradient clipping: 1.0
- • Latent dimension: 200
- • Number of parameters in decoder: 61911939
- • Number of parameters in encoder: 49790260

Computation cost. Excluding the computation cost of periodic evaluation (i.e., only counting the computation cost of model training), each RF+LZN experiment takes 31 hours on 16 A100 (40 GB) GPUs.

E.4 More Results Generated images. The generated images of RF and RF+LZN are in Figs. 22 and 23.

Ablation studies on FID implementation. Same as § C, we present the FID scores using three different implementations in Tab. 7. We see that, while the numbers are different, the relative ranking across all three implementations is consistent. Especially, RF+LZN achieves the best FID in all implementations.

Ablation studies on sampling steps. Following the experimental settings in § C, we use the Euler sampler with varying numbers of sampling steps. As shown in Fig. 24, RF+LZN generally achieves better FID than the RF baseline across most settings.

Ablation studies on classification techniques. Here, we discuss several techniques for improving the classification results.

- • Recall that latent computation (Eq. (1)) includes randomness from ϵi because each sample corresponds to a latent zone, not a single point. Empirically, for classification tasks, using the “center” of the latent zone yields better performance. Concretely, we set α = 0 in Eq. (4) when computing latents. This is intuitive, as the center is likely farther from zone boundaries and better represents the sample.
- • § A.3 discusses that during training, we use a batch of samples rather than all samples to estimate latents for efficiency. However, during inference, where gradient computation is unnecessary and thus the overhead of large batch sizes is less critical, we can use a larger batch size to improve performance.

[Figure 36]

- Figure 22: Generated images of RF on CIFAR10 (conditional generation). Every 5 rows corresponds to one class in CIFAR10.

Fig. 25 shows that increasing the batch size and decreasing α improve the classification accuracy. The best setting improves the default setting (batch size= 2000 and α = 1.0) by 2.9%.

Ablation studies on latent alignment hyperparameter. In latent alignment (§ 2.2.2), we introduced a hyperparameter u that controls how many time steps are excluded from the latent alignment objective. In our main experiments, we set u = 20 (out of a total of 100 steps). Here, we conduct an ablation study by reducing u to 5 (i.e., 4× smaller). The results are shown in Tab. 8. We can see that u does not affect the results much, and the performance remains better than the baseline RF across most metrics. This is expected. Unlike common hyperparameters (such as loss weights) that influence the optimal solution, u does not alter the optimal solution, which is the perfect alignment

[Figure 37]

- Figure 23: Generated images of RF+LZN on CIFAR10(conditional generation). Every 5 rows corresponds to one class in CIFAR10.

between two latent zones. Instead, this parameter is introduced solely to help avoid getting stuck in local optima (§ 2.2.2). We expect that any small but non-zero value of u should be sufficient in practice.

|Rectified Flow<br><br>Rectified Flow + LZN|
|---|

101

6 × 100

FID

- 3 × 100
- 4 × 100

101 102 103 NFE

- Figure 24: FID vs. number of sampling steps in the Euler sampler on CIFAR10 (conditional generation). RF+LZN outperforms RF in most cases.

[Figure 38]

1.0 0.8 0.6 0.4 0.2 0.0 α

2000

4000

8000

10000

Batchsize

- 91.57 93.35 94.16 94.30 94.29 94.33
- 92.69 93.85 94.25 94.33 94.34 94.40
- 93.70 94.24 94.36 94.31 94.40 94.45

93.85 94.28 94.44 94.47 94.45 94.44

|[Figure 39]| |
|---|---|
| | |
| | |
| | |
| | |
| | |

0.920

0.925

0.930

0.935

0.940

Accuracy

- Figure 25: Classification accuracy with different hyperparameters on CIFAR10. Generally, increasing the batch size and decreasing α improve the classification accuracy.

- Table 8: Conditional image generation quality and classification accuracy on CIFAR10. The best results are in gray box . The hyperparameter u does not impact the results much.

|Algo.|FID↓ sFID↓ IS↑ Precision↑ Recall↑ Recon↓ Accuracy↑<br><br>|
|---|---|
|RF RF+LZN (u = 20) RF+LZN (u = 5)<br><br>|2.47 4.05 9.77 0.71 0.58 0.69 2.40 3.99 9.88 0.71 0.58 0.38 94.47 2.39 3.99 9.76 0.71 0.58 0.36 94.42<br><br>|

### F Extended Discussions on Related Work

[16] proposes to conduct flow matching in the latent space. However, it has quite different goals and techniques from LZN.

- • Goals. The goal of [16] is to improve generation tasks. In contrast, our goal is more ambitious: to develop a unified framework that supports generation, representation learning, and classification. This broader scope requires a different design philosophy and technical approach, as detailed next.
- • Techniques. [16] applies flow matching to the latent space of a pre-trained Stable Diffusion autoencoder, which is reasonable when focusing solely on generation. However, such a latent space is high-dimensional and retains spatial structure, limiting its suitability for classification and compact representation learning. To support our broader objectives, we introduce several novel techniques:
- • Match a discrete distribution (i.e., the anchors) to a continuous one, as opposed to a continuousto-continuous distribution matching in [16].
- • Use an adaptive latent space, since our encoder and decoder are trained end-to-end, as opposed to using a fixed pre-trained autoencoder and fixed latent space in [16].
- • Numerically solve the flow directly, as opposed to training an additional model to learn the flow in [16].
- • Latent alignment between different data types (e.g., image and label), which is new in our paper.

### G Extended Discussions on Limitations and Future Work

Inference efficiency. It is important to note that while the training cost of LZN might be high, at inference time, LZN is often as efficient as existing approaches.

- • For image generation (§ 3 and 5), we do not need to compute the latent during inference. Instead, latents are sampled from the Gaussian prior and passed directly to the decoder, making the generation speed comparable to the base model.
- • For representation learning (§ 4), we find that dropping the final encoder layers during inference improves performance (§ D.4), similar to the observation in prior contrastive learning methods [10]. In this case, inference involves simply passing an image through the encoder without the latent computation process (§ 2.2.1), just like in traditional contrastive learning methods.

Training efficiency. The main training bottleneck stems from the quadratic cost with respect to the batch size. Notably, this is also the case for many contrastive learning methods, including the seminal works MoCo [28] and SimCLR [10], which compute pairwise similarities between all examples in a batch.

The parallel between LLM training and LZN training. We observe an interesting parallel between the training of LLMs and LZN. Specifically, in LLM training, computing attention weights requires O c2dv , where c is the context length, d is the attention dimension, and v is the number of layers. In LZN, computing the latents (§ 2.2.1) requires O(n2qr), where n is the number of samples in a batch, q is the latent dimension, and r is the number of solver steps. Several parallels emerge:

- • Context length in LLMs (c) ↔ Number of samples in LZN (n)
- • Attention dimension in LLMs (d) ↔ Latent dimension in LZN (q)
- • Number of layers in LLMs (v) ↔ Number of solver steps in LZN (r)

Not only do these parameter pairs affect the time complexity in similar ways, but their computation flows are also analogous: in LLMs, the pairwise inner product of token features is computed to derive attention weights, and these weights are computed sequentially across layers. Similarly, in LZN, the pairwise distances between intermediate anchor points of samples are computed to derive velocity, and this velocity is updated sequentially across solver steps.

While LLM training is known to be computationally expensive, recent advances have significantly improved its efficiency. Given the structural similarities, we expect that such advances in LLM training could be adapted to enhance the training efficiency of LZN as well.

Using LZN solely to implement generative modeling. In theory, LZN can be used solely for generative modeling. By construction (§ 3), if the decoder is trained to map latents to the corresponding data

perfectly, then the generative distribution of LZN is exactly n1 ni=1 δ(s − xi), i.e., the empirical distribution of the training set. We explored this approach in our early experiments. It performs well

on simple datasets such as MNIST [18], but generates blurry images on more complex datasets such as CIFAR10. We hypothesize that this may be due to the minibatch approximation (§ A.3), which can break the disjoint latent property, and/or the strict requirement that latent zones have no gaps between them. We leave a deeper exploration of this direction to future work.

Societal impacts. Since LZN can be used to improve ML models, it has the potential for both beneficial and harmful applications. Positive use cases include creative content generation and improved information retrieval, while negative applications may involve the creation of fake or misleading content.

