# arXiv:2604.17078v1[cs.AI]18Apr2026

|…<br><br>Task 𝑡𝑡<br><br>Task 𝑗𝑗<br><br>|
|---|

[Figure 1]

[Figure 2]

### Understanding and Enforcing Weight Disentanglement in Task Arithmetic

𝒟𝒟2

𝒟𝒟1

- 𝑓𝑓(𝑥𝑥;𝜃𝜃0 + 𝛼𝛼1𝜏𝜏1)
- 𝑓𝑓(𝑥𝑥;𝜃𝜃0 + 𝛼𝛼2𝜏𝜏2)

| |
|---|

[Figure 3]

𝒟𝒟1

𝒟𝒟2

Shangge Liu 1, Yuehan Yin 1, Lei Wang 2, Qi Fan 1, Yinghuan Shi 1, Wenbin Li 1*, Yang Gao 1, Dacheng Tao 3 1State Key Laboratory for Novel Software Technology, Nanjing University, China

𝒟𝒟1

𝒟𝒟2

𝑓𝑓(𝑥𝑥;𝜃𝜃0 + 𝛼𝛼1𝜏𝜏1 + 𝛼𝛼2𝜏𝜏2) Weight Disentanglement

2University of Wollongong, Australia 3Nanyang Technological University, Singapore

|任务-特征专业化|
|---|

##### Abstract

Task-Feature Specialization Task 1

…

Task arithmetic provides an efficient, training-free way to edit pre-trained models, yet lacks a fundamental theoretical explanation for its success. The existing concept of “weight disentanglement” describes the ideal outcome of non-interfering task composition but does not reveal its underlying cause. Crucially, what intrinsic properties of the pre-trained model (θ0) or the task vectors (τt) enable this disentanglement remains underexplored. In this paper, we introduce Task-Feature Specialization (TFS), a model’s ability to allocate distinct internal features to different tasks, as the fundamental principle. We first prove that TFS is a sufficient condition for weight disentanglement. More importantly, we find that TFS also gives rise to an observable geometric consequence: weight vector orthogonality. This positions TFS as the common cause for both the desired functional outcome (disentanglement) and a measurable geometric property (orthogonality). This relationship provides the key insight for our method: since the abstract TFS property is intractable to enforce directly, we can instead promote weight disentanglement by shaping its concrete geometric consequence, orthogonality. Therefore, we propose OrthoReg, a simple and effective regularization method that actively enforces an internal orthogonal structure on weight updates (∆W) that constitute τt during fine-tuning. And we theoretically prove that OrthoReg promotes disentanglement. Extensive experiments demonstrate that OrthoReg consistently and significantly enhances the performance of various task arithmetic methods. Code is available at https://github.com/RL-MIND/OrthoReg.

…

Task 𝑡𝑡

…

…

Task 𝑗𝑗

𝑤𝑤𝑖𝑖 𝑤𝑤𝑗𝑗…𝑤𝑤𝑘𝑘

𝒟𝒟2

𝒟𝒟1

- 𝑓𝑓(𝑥𝑥;𝜃𝜃0 + 𝛼𝛼1𝜏𝜏1)
- 𝑓𝑓(𝑥𝑥;𝜃𝜃0 + 𝛼𝛼2𝜏𝜏2)

#### 𝑊𝑊 =

[Figure 4]

𝒟𝒟1

𝒟𝒟2

[Figure 5]

𝒟𝒟1

𝒟𝒟2

∀𝑖𝑖,𝑗𝑗 𝑤𝑤𝑖𝑖𝑤𝑤𝑗𝑗 = 0

𝑓𝑓(𝑥𝑥;𝜃𝜃0 + 𝛼𝛼1𝜏𝜏1 + 𝛼𝛼2𝜏𝜏2) Weight Disentanglement

Weight Vector Orthogonality

Figure 1. Conceptual illustration of our central thesis: TaskFeature Specialization (TFS) is proposed and shown as the common cause that connects the geometric property of Weight Vector Orthogonality (WVO) with the functional property of Weight Disentanglement (WD). This paper establishes this connection in two ways: first, by proving that TFS, which gives rise to inherent orthogonality in the pre-trained model θ0, is a sufficient condition for ideal disentanglement; and second, by proposing a method that actively enforces this structure on weight updates (∆W) that constitute τt to promote disentanglement in realistic scenarios.

quiring new skills [30], personalizing behavior [55], or unlearning harmful capabilities [12, 50]. Conventional methods like joint fine-tuning are often impractical due to prohibitive computational costs, the inaccessibility of all training datasets, and the risk of catastrophic forgetting [23, 27].

To address this challenge, model merging [39, 47, 48] has recently emerged as an efficient, training-free paradigm. Instead of costly retraining, model merging operates posthoc combination of the weights of multiple specialized models, each fine-tuned for a specific task, to create a single, multi-talented model. Among these methods, task arithmetic [16] is particularly elegant. It operates by representing the knowledge for a task t as a task vector, defined

##### 1. Introduction

Large-scale pre-trained models (PTMs) [7, 33, 41] have become powerful foundations for various applications [26, 28]. However, a critical challenge lies in adapting these powerful yet static models to new requirements, such as ac-

*Corresponding Author

by the parameter shift τt = θt − θ0 from the pre-trained weights θ0 to the fine-tuned weights θt. By simply adding or subtracting these task vectors, one can compose, remove, or even draw analogies between different skills, all without the need for costly joint training.

Despite its empirical success, a fundamental question remains: why does task arithmetic work? Answering this is critical to transforming task arithmetic from an empirical curiosity into a reliable engineering tool and to improve it beyond its current limitations, especially in critical applications where predictability and trustworthiness are significant. The concept of Weight Disentanglement, introduced in Tangent Task Arithmetic (TTA) [32], offers a partial answer. It posits that in an ideal scenario, the effects of different task vectors are isolated to their respective data domains, thus preventing destructive interference. However, weight disentanglement is more of a phenomenological description of the desired outcome than an explanation of its fundamental cause. The existing literature does not fully specify what properties of the pre-trained model (θ0) or the task vectors (τt) are necessary to achieve this state.

This gap in understanding motivates our work. To move from description to explanation, we must address two questions. i. What makes a model θ0 good for task arithmetic? ii. How to construct a good τt? The first question asks: what intrinsic properties of a pre-trained model θ0 make it inherently suitable for achieving weight disentanglement? Without answering this question, we will not be able to select or design foundation models that are naturally amenable to effective model editing, leaving performance to chance. The second question addresses the construction of the task vectors themselves: how can we construct task vectors τt that actively promote weight disentanglement? Without answering this, standard fine-tuning offers no guarantee of producing task vectors that compose well and often leads to suboptimal, interference-prone results.

In this paper, we identify Task-Feature Specialization (TFS), the model’s ability to allocate distinct internal features to different tasks, as the key underlying principle. We first prove that TFS is a sufficient condition for Weight Disentanglement (WD) (Theorem 3). More importantly, we find that TFS also gives rise to a geometric consequence: Weight Vector Orthogonality (WVO). This positions TFS as the common cause for both the desired functional outcome (WD) and an observable geometric consequence (WVO), a conceptual relationship shown in Figure 1. We can thus answer Question i: models that achieve TFS are effective at disentanglement, and WVO provides a possible indicator for this abstract property. However, TFS is an ideal property that rarely holds for a pre-trained model θ0 in practice. This challenge motivates us to resort to τt, that is, answering Question ii. To address this, we propose a method that enforces an internal orthogonal structure on weight updates

(∆W) that constitute τt and theoretically show its efficacy (Theorem 4). We also establish a theoretical connection between our approach and TTA [32], showing both converge on the same principle: inter-task vector orthogonality.

Our main contributions can be summarized as follows.

- • We put forward a theory for the success of task arithmetic, identifying task-feature specialization as a sufficient condition for weight disentanglement and then weight vector orthogonality as its geometric consequence.
- • Based on the theory, we propose OrthoReg, a regularization method that actively promotes disentanglement by enforcing orthogonality on weight updates, for which we provide a rigorous theoretical proof of efficacy.
- • We establish a theoretical connection between our method and the existing work TTA and reveal that they both succeed by achieving inter-task vector orthogonality.
- • We experimentally demonstrate that OrthoReg consistently and significantly improves the performance of various task arithmetic methods.

##### 2. Related Work

Model Merging and Task Arithmetic. Task Arithmetic [16] is a model merging technique that combines models by algebraically manipulating their “task vectors”. While this approach avoids costly retraining [47], it often suffers from destructive interference when composing multiple tasks. Existing solutions to this problem can be broadly classified into two categories [47]: during-merging and pre-merging methods. During-merging methods design sophisticated algorithms to combine already-trained models [9, 46, 48]. In contrast, pre-merging methods aim to create more “mergeable” models from the outset by modifying the fine-tuning process [19, 32, 40, 51, 54]. Our proposed method, OrthoReg, belongs to the pre-merging category.

Key theoretical work in this area includes Tangent Task Arithmetic (TTA) [32], which introduces the crucial concept of weight disentanglement and shows that fine-tuning in linearized tangent space promotes it. Work [19] further demonstrates that fine-tuning only the attention modules also enhances it. Concurrent works have provided generalization analyses for nonlinear Transformers based on datadependent task correlations [24] and established theoretical bounds that explicitly require task vectors to be nearly orthogonal [53]. Our work provides a more fundamental explanation through Task-Feature Specialization (TFS) and proposes enforcing its geometric consequence, orthogonality, as a direct mechanism to mitigate interference.

Orthogonality in Neural Networks. The geometric properties of weights, particularly orthogonality, have been wellstudied for their role in improving training stability, generalization, and efficiency [2, 10, 29, 37, 49]. It has been successfully applied in RNNs to prevent vanishing/exploding gradients [2] and in GANs via Spectral Normalization to

stabilize discriminator training [29]. Our work repurposes this powerful geometric constraint for a novel application: task arithmetic. We demonstrate that actively shaping the geometry of weight updates to be orthogonal is a direct and effective way to mitigate interference in task arithmetic.

##### 3. Preliminaries and Problem Formulation

In this section, we first define the basic setup of task arithmetic, then introduce the concept of weight disentanglement. Finally, we introduce Neural Tangent Kernel (NTK) linearization hypothesis.

###### 3.1. Basic Setup and Notation

Let f(x;θ) be a neural network parameterized by weights θ ∈ RP, with initial pre-trained weights denoted as θ0. After fine-tuning on a task t, the new weights are θt∗. Following the literature [16], the task vector τt is defined as the parameter shift,

τt = θt∗ − θ0. (1)

The task vector τt encapsulates the parameter modifications required for the model to adapt to task t. Task arithmetic then performs algebraic operations on these task vectors to create a multi-task model. Specially, combining a set of tasks T = {1,...,T} via task addition is achieved by

T

αtτt, (2)

θMT = θ0 +

t=1

where αt is scalar coefficient of each task. Our theoretical analysis focuses on the parameters of linear layers (e.g., FC layers and attention projections). This simplification is well-justified by their central role in modern architectures [20, 56] and model merging practices [16, 43, 46], with a detailed rationale provided in Appendix A.

###### 3.2. The Weight Disentanglement Property

The concept of weight disentanglement introduced by the seminal work [32] is a key property proposed to explain the success of task arithmetic.

- Definition 1 (Weight Disentanglement). A model f satisfies weight disentanglement at θ0 with respect to a set of tasks T with data domains {Dt}Tt=1 if, for any set of scalar coefficients {αt}Tt=1, the following relationship holds,

T

f(x; θ0 + αiτi), if x ∈ Di f(x; θ0). if x ∈/ Tt=1 Dt

f(x; θ0 +

αtτt) =

t=1

(3)

Intuitively, this property requires that a merged model’s behavior on a specific task’s data depends only on that task’s vector, while reverting to pre-trained behavior for out-ofdomain data. Disentanglement can stem from the inherent properties of the pre-trained model θ0 or from the specific construction of the task vectors τt. Accordingly, our work

investigates both the ideal properties of θ0 and a method for constructing τt to actively promote disentanglement.

###### 3.3. The NTK Linearization Hypothesis

Consistent with the literature [32], our analysis relies on the Neural Tangent Kernel (NTK) [18] linearization hypothesis, which approximates the model’s output for a small parameter change τ with a first-order Taylor expansion around θ0,

f(x; θ0 + τ) ≈ f(x; θ0) + τ⊤∇θf(x; θ0). (4)

Here, J(x) := ∇θf(x;θ0) is the Jacobian of the model’s output with respect to its parameters.

##### 4. The Proposed Framework

In this section, we theorize that Task-Feature Specialization (TFS) is the key principle for task arithmetic. In the first part, we demonstrate that TFS is a sufficient condition for weight disentanglement (WD) and also naturally leads to weight vector orthogonality (WVO). This suggests that WD and WVO are correlated effects of the common cause, TFS. While TFS is rare in practice, its geometric consequence, orthogonality, offers a tangible objective. This motivates our method in second part: actively enforcing orthogonality on weight updates (∆W) to mitigate interference and improve disentanglement.

###### 4.1. An Equivalent Condition for Disentanglement

To clearly reveal the underlying mechanism of weight disentanglement, we first focus our analysis on the interaction between two tasks, t and j. The core in-domain component of the weight disentanglement property (see Definition 1) can be simplified to,

f(x; θ0 + τt + τj) = f(x; θ0 + τt), ∀ x ∈ Dt. (5)

This simplification is sufficient for our analysis because the linearity of interference ensures that pairwise results extend to the multi-task case. Our analysis concentrates on this indomain condition, which is the core challenge, while the out-of-domain case follows from similar logic. A detailed justification for this reframing is provided in Appendix B.

We can now reframe the condition of weight disentanglement into a tractable one using the NTK linearization hypothesis, as formalized in the following lemma.

Lemma 1. Under the NTK linearization hypothesis, weight disentanglement between tasks t and j is equivalent to the interference term from task j being approximately zero on the data domain of task t:

τj⊤J(x) = 0, ∀x ∈ Dt. (6)

Detailed proof in Appendix C. This condition forms the basis for our subsequent analysis, also identified as the key to disentanglement in recent literature [51].

###### 4.2. Our Main Theorem

We now investigate the ideal conditions that enable perfect task arithmetic. We put forward the Task-Feature Specialization (TFS) property and show that it not only guarantees weight disentanglement but also gives rise to the geometric property of weight vector orthogonality.

###### 4.2.1. Task-Feature Specialization (TFS)

To fundamentally explain why task arithmetic can work, we propose a new core concept: Task-Feature Specialization (TFS). It means that an ideal model, when faced with different tasks, intelligently allocates distinct internal features, represented by the column vectors of its weight matrices, to specific tasks. For instance, under the ideal TFS assumption, a task for classifying cars and a task for classifying MNIST digits would rely on two disjoint sets of internal features within the same layer of the model. We posit that this functional separation is the root of perfect task arithmetic. We formalize this intuition with the following definitions.

Definition 2 (Task-Specialized Feature Set). For a given linear layer with weight matrix W, we consider each column vector {wk}dk=1 as extracting a “base feature” whose activation is zk. For a task t with data domain Dt, we define its specialized feature set It ⊆ {1,...,d} as the set of indices for which the model’s final output f(x;θ0) is sensitive to the activation zk for inputs x ∈ Dt. Formally, for any k ∈/ It, we have Ex∼D

[|∂f(x;θ

0)

∂zk |] = 0.

t

We formalize our core assumption for the ideal case.

Assumption 1 (Task-Feature Specialization). For two distinct tasks t and j, their respective specialized feature sets, It and Ij, are disjoint, i.e., It ∩ Ij = ∅.

###### 4.2.2. TFS as a Sufficient Condition for WD

We now prove that TFS property is a sufficient condition for weight disentanglement. This provides a direct explanation for the success of task arithmetic: when the model functionally dedicates distinct features to distinct tasks.

Theorem 1. Under the NTK linearization hypothesis (Section 3.3) and the Task-Feature Specialization property, weight disentanglement between tasks t and j holds.

The proof is detailed in Appendix D.

###### 4.2.3. From TFS to Weight Vector Orthogonality

More interestingly, we find that under the same TFS condition, a geometric property of the model’s parameters can be derived: Weight Vector Orthogonality.

- Definition 3 (Weight Vector Orthogonality). A weight ma-

trix W ∈ Rm×d with column vectors {w1,...,wd} is said to possess column orthogonality if its column vectors are mutually orthogonal, we distinguish between two key forms.

|[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>| | | |[Figure 27]<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |[Figure 28]|
| | | | | | | | | | | |
| | | | | |[Figure 29]|[Figure 30]| | | | |
| | | | | | |[Figure 31]| | | | |
| | | | | |[Figure 32]<br><br>[Figure 33]|[Figure 34]| | | | |
<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]|
|---|

|[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>95.27%<br><br><5%<br><br>|
|---|

(a) The distribution of angles between column vector pairs in a weight matrix.

(b) Statistical summary of angular deviations from 90◦ across all linear layers of the model.

Figure 2. Empirical evidence of weight vector orthogonality in a pre-trained CLIP ViT-B/16.

- (a) Block Orthogonality. Given a partition of the column

indices into disjoint sets {I1,...,IT} (e.g., corresponding to different tasks), the matrix exhibits block orthogonality if any two vectors wk and wl from different sets are orthogonal (i.e., ⟨wk,wl⟩ = 0 for all k ∈ It,l ∈ Ij with t ̸= j).

- (b) Column-wise Orthogonality. The matrix exhibits

column-wise orthogonality if all pairs of distinct column vectors are orthogonal (i.e., ⟨wk,wl⟩ = 0 for all k ̸= l). This can be seen as a special case of block orthogonality where each block contains only a single vector.

The TFS property has a direct geometric consequence on the model’s parameters. We can show that a model exhibiting TFS will naturally develop an orthogonal structure in its weights, which we formalize as the following corollary.

Corollary 1. Given a model that adheres to the TaskFeature Specialization (TFS) property (Assumption 1), its weight matrices will exhibit Block Orthogonality.

The proof is detailed in Appendix E. Empirically, we find that this predicted block orthogonality not only holds, but that the structure is often even stronger, approaching column-wise orthogonality. In a pretrained CLIP ViT-B/16 (Figure 2), the angles between all column vector pairs are sharply peaked at 90◦. This suggests that pre-training pushes the entire weight matrix towards column-wise orthogonality (WVO) by also decorrelating features within the same task. (Full per-layer distributions are provided in Appendix J.1).

###### 4.2.4. Orthogonality as a Clue for Disentanglement

Our analysis thus far provides an answer to the first question posed in Section 1: What makes a pre-trained model θ0 good for task arithmetic? Our theory posits that the sufficient condition is task-feature specialization (TFS), which serves as a common cause for both weight disentanglement and block orthogonality. Although this geometric property should not be seen as a direct cause of disentanglement, it is a geometric consequence of the underlying functional separation (TFS) that effective training produces. TFS is an abstract property, but WVO provides a concrete, measurable signature. This relationship enables us to use WVO

|Feature Overlap<br><br>WVO<br><br>𝝉𝝉 = 𝜽𝜽𝒇𝒇𝒇𝒇 − 𝜽𝜽𝟎𝟎<br><br>𝜃𝜃0 𝜃𝜃𝑓𝑓𝑓𝑓<br><br>|TFS<br><br>|
|---|
<br><br>|Add & LN|
|---|
<br><br>| | |
|---|---|
|FFN| |
<br><br>|𝑊𝑊𝑜𝑜|
|---|
<br><br>Attention<br><br>𝑊𝑊𝑞𝑞 𝑊𝑊𝑘𝑘 𝑊𝑊𝑣𝑣<br><br>Add & LN<br><br>× 𝑁𝑁<br><br>ℒortho =<br><br>𝑙𝑙<br><br>(Δ𝑊𝑊 𝑙𝑙 )𝑇𝑇Δ𝑊𝑊 𝑙𝑙 − 𝐼𝐼 𝐹𝐹2<br><br>𝑊𝑊𝑞𝑞 𝑊𝑊𝑘𝑘 𝑊𝑊𝑣𝑣 𝑊𝑊𝑜𝑜 FFN<br><br>block 1<br><br>𝑊𝑊𝑞𝑞 𝑊𝑊𝑘𝑘 𝑊𝑊𝑣𝑣<br><br>|𝑊𝑊𝑜𝑜|
|---|
<br><br>… FFN<br><br>ℒ + 𝓛𝓛𝐨𝐨𝐨𝐨𝐨𝐨𝐨𝐨𝐨𝐨<br><br>| |
|---|
<br><br>Δ𝑊𝑊 𝑙𝑙 = block 𝑛𝑛<br><br>flatten<br><br>|Δ𝑊𝑊|
|---|
<br><br>|
|---|

| | |
|---|---|
| | |

as a powerful diagnostic clue. As our Bayesian analysis suggests (see Appendix F), observing WVO in a model that has undergone effective training on diverse data strongly increases our belief that it has developed a TFS-like structure, and consequently, will exhibit disentanglement.

|Add & LN|
|---|

Feature Overlap

|TFS<br><br>|
|---|

| | |
|---|---|
|FFN| |

Add & LN

𝑊𝑊𝑜𝑜

###### 4.3. Our Method OrthoReg

|𝑊𝑊𝑜𝑜|
|---|

|𝑊𝑊𝑣𝑣|
|---|

… FFN

Attention

𝑊𝑊𝑜𝑜 FFN

𝑊𝑊𝑞𝑞 𝑊𝑊𝑘𝑘 𝑊𝑊𝑣𝑣

𝑊𝑊𝑞𝑞 𝑊𝑊𝑘𝑘

Section 4.2 shows that task-feature specialization is sufficient for weight disentanglement. However, the TFS property is an idealization that rarely holds in practice. We now address this gap between the theory and realistic scenarios.

Δ𝑊𝑊 𝑙𝑙 = block 𝑛𝑛

block 1

𝑊𝑊𝑞𝑞 𝑊𝑊𝑘𝑘 𝑊𝑊𝑣𝑣

(Δ𝑊𝑊 𝑙𝑙 )𝑇𝑇Δ𝑊𝑊 𝑙𝑙 − 𝐼𝐼 𝐹𝐹2

ℒortho =

flatten

× 𝑁𝑁

𝑙𝑙

| |
|---|

𝝉𝝉 = 𝜽𝜽𝒇𝒇𝒇𝒇 − 𝜽𝜽𝟎𝟎

###### Δ𝑊𝑊

###### 4.3.1. The Challenge of Feature Overlap

𝜃𝜃0 𝜃𝜃𝑓𝑓𝑓𝑓

ℒ + 𝓛𝓛𝐨𝐨𝐨𝐨𝐨𝐨𝐨𝐨𝐨𝐨

WVO

The core assumption underpinning our ideal case is that the specialized feature sets for distinct tasks are disjoint, i.e., It∩Ij = ∅, that is often violated in practice, as distinct tasks could rely on shared underlying features. When this overlap occurs, Theorem 3 does not apply anymore. Specifically, for a shared feature k ∈ It ∩ Ij and an input x ∈ Dt, both the gradient term ∇wk

Figure 3. An overview of the OrthoReg method. It mitigates task interference caused by feature overlap by introducing Lortho. As illustrated for a Transformer block, this loss enforces an orthogonal structure on the weight updates (∆W) during fine-tuning.

f(x;θ0) (since k is now relevant to task t) and the task vector component (τj)k (since k is now relevant to task j) are generally non-zero. Consequently, their inner product ⟨(τj)k,∇wk

fine-tuning a model on a given task t becomes,

L = Ltask(θ0 + ∆θ) + λ · Lortho(∆θ), (7) where Ltask is the original task objective, ∆θ represents all parameter updates, i.e., τ. And λ is a hyperparameter controlling the regularization strength, and Lortho is our proposed orthogonal regularizer.

f(x;θ0)⟩ is highly likely to be non-zero, creating non-zero interference, breaking the disentanglement guarantee.

This reveals the limitations of relying solely on static θ0. In realistic scenarios where the ideal TFS property does not hold, the pre-trained model alone is insufficient to guarantee disentanglement. Consequently, the responsibility shifts towards the second avenue we identified in Section 3.2: the dynamic construction of the task vectors τt themselves. This brings us back to our second question: How can we actively construct “good” task vectors that promote disentanglement, even when the ideal TFS condition is not met?

Definition 4. The orthogonal regularization term is defined as the sum of penalties over all tuned linear layers,

∥(∆W(l))⊤∆W(l) − I∥2F, (8)

Lortho(∆θ) =

l

where the sum is overall linear layers l being updated, ∆W(l) is weight update matrix for that layer, I is the identity matrix, and ∥ · ∥2F denotes the squared Frobenius norm.

###### 4.3.2. Method: Orthogonal Regularizations

This simple, plug-and-play regularizer penalizes the deviation of each update matrix’s Gram matrix from the identity, thereby driving the columns of ∆W(l) to become mutually orthogonal and have unit norm.

Directly enforcing the abstract property TFS is intractable. Our theory suggests a practical alternative: enforcing its geometric consequence, orthogonality. While Corollary 2 proves that TFS leads to block orthogonality, it is hard to enforce this property as the number of feature blocks It is implicitly determined by the specific task and cannot be known beforehand. To handle this, we propose enforcing a simpler and stronger condition: column-wise orthogonality on the weight updates. This forms the core motivation for our method, OrthoReg (Figure 3). As will be proved in Theorem 4, enforcing this condition will minimize crosstask interference during model merging. In addition, from a representation learning perspective, this condition will encourage decorrelated intra-task features, which is a desirable inductive bias as it promotes a more efficient and less redundant feature basis for the task.

###### 4.3.3. Theoretical Justification to OrthoReg

We now present our second main theoretical result, which formalizes the effectiveness of our proposed method. This theorem shows that enforcing an orthogonal structure on weight updates serves as a key mechanism for disentanglement, even in the realistic scenario of feature overlap.

Theorem 2. Under the NTK linearization hypothesis (Section 3.3), even if the Task-Feature Specialization property (Assumption 1) does not hold (i.e., It ∩ Ij ̸= ∅), constraining the task update matrices {∆Wt(l)} to be approximately internally orthogonal (as encouraged by the regularization in Definition 4) actively promotes weight disentanglement between tasks t and j.

To achieve this, we introduce a novel regularization term to the standard fine-tuning loss function. The total loss for

- Table 1. Task addition results on CLIP-based models. Performance of adding 8 task vectors on three architectures. Our proposed orthogonal regularization (+OrthoReg) is applied to several baselines, showing consistent improvements in both Absolute Accuracy (Abs.Acc.) and Normalized Accuracy (Norm.Acc.). An asterisk (*) denotes the best absolute accuracy for each model architecture.

ViT-B-32, 8 tasks ViT-B-16, 8 tasks ViT-L-14, 8 tasks Abs.Acc.(↑) Norm.Acc. (↑) Abs.Acc.(↑) Norm.Acc. (↑) Abs.Acc.(↑) Norm.Acc. (↑) zero-shot 47.74 / 54.22 / 64.54 /

Method

Non-linear Finetuning [16] 70.32 77.56 75.39 75.39 84.07 89.19 Non-lin. FT+OrthoReg (ours) 73.41 93.93 77.68 93.62 88.23 100.08

∆ +3.09 +16.37 +2.29 +18.23 +4.16 +10.89 Tangent Task Arithmetic [32] 74.68 85.27 78.97 87.48 86.19 93.14

###### TTA+OrthoReg (ours) 76.35 91.81 79.85 88.02 87.52 96.44

∆ +1.67 +6.54 +0.88 +0.54 +1.33 +3.30 Attention-Only Fine-tuning [19] 78.07 86.99 80.71 87.64 87.81 93.59

###### ATT-FT+OrthoReg (ours) 80.87* 99.76 83.37* 98.77 90.41* 100.05

∆ +2.80 +12.77 +2.66 +11.13 +2.60 +6.46 LoRA-ATT 73.84 84.29 75.51 83.17 87.02 93.33

LoRA-ATT+OrthoReg (ours) 76.00 86.10 79.67 87.70 89.16 95.50 ∆ +2.16 +1.81 +4.16 +4.53 +2.14 +2.17

Proof Sketch. Our goal is to show that in this case the interference term τj⊤J(x) is approximately 0 for x ∈ Dt. The proof first establishes that for a typical input x from task t’s domain, its Jacobian J(x) is directionally aligned with the task vector τt. This allows us to reframe the interference by approximating the angle involving the Jacobian, ∠(τj,J(x)), with the angle between task vectors, ∠(τj,τt),

|τj⊤J(x)| ≈ ||τj||2 · ||J(x)||2 · | cos ∠(τj, τt)|. (9)

Then we demonstrate that our regularizer implements a dual control mechanism over the resulting interference expression. (1) Norm Control. It inherently bounds the magnitude of the task vector ∥τj∥2. (2) Angle Control. More critically, by enforcing an internal orthogonal structure on each update matrix, it drives the angle between the different task vectors, ∠(τj,τt), statistically towards 90 degrees. By simultaneously bounding the norm and nullifying the angle term, the regularizer ensures the expected interference is negligible, thus establishing weight disentanglement. The full proof is detailed in Appendix G.

Theorem 4 provides a constructive answer to the second question in Section 1: How can we construct good task vectors τt that promote disentanglement? Our analysis shows that actively enforcing an internal orthogonal structure on ∆W serves as a direct and effective mechanism for achieving weight disentanglement.

###### 4.4. Connection between Our Work and TTA

As a seminal theoretical analysis in task arithmetic, Tangent Task Arithmetic (TTA) [32] demonstrates that fine-tuning within the tangent space of the pre-trained model θ0 effectively promotes weight disentanglement. We now connect our analysis with their findings. Our investigation reveals that both methods, despite their different implementations,

derive their effectiveness from a shared underlying mechanism: enforcing orthogonality between different task vectors (⟨τt,τj⟩ ≈ 0), i.e., the “Angle Control” part of our proof for Theorem 4.

Specifically, OrthoReg enforces orthogonality explicitly via a regularizer. In contrast, TTA achieves this implicitly through the model’s NTK geometry, but at a high computational cost. TTA’s reliance on Jacobian calculations can double memory usage and increase training time by 23x [19], posing a significant barrier to adoption. OrthoReg thus offers a more direct, efficient, and scalable alternative. A detailed derivation of this connection and a comparative analysis are provided in Appendix H.

##### 5. Experiments

###### 5.1. Experimental Setup

Datasets and tasks. We follow the evaluation protocol established by [16] and [32]. The primary benchmark consists of eight diverse image classification datasets: Cars [21], DTD [5], EuroSAT [11], GTSRB [38], MNIST [22], RESISC45 [4], SUN397 [44], and SVHN [31].

Models and training methods. In our experiments, we adopt CLIP-pretrained Vision Transformers [33] as pretrained model, including ViT-B-32, ViT-B-16 and ViT-L14. During fine-tuning, the text encoder is frozen, and the image encoder can be updated. The regularization strength is selected via validation within the range [0.1, 100].

Baselines. We evaluate our proposed orthogonal regularization against several state-of-the-art task arithmetic methods. For each baseline, we report the performance when our regularizer is applied, denoted by the “+OrthoReg” suffix. The primary baselines are based on full-parameter fine-tuning. (1) Non-linear Fine-tuning (Nonlin. FT). The standard task

- Table 2. The minimum average Target Accuracy (Tar.Acc.) achievable while maintaining at least 95% of the zero-shot accuracy on the ImageNet control task (Con.Acc.). Our proposed orthogonal regularization (+OrthoReg) shows a consistent and significant improvement in forgetting the target task. An asterisk (*) denotes the best (lowest) target accuracy for each model architecture.

ViT-B-32, 8 tasks ViT-B-16, 8 tasks ViT-L-14, 8 tasks Tar.Acc.(↓) Con.Acc. (↑) Tar.Acc.(↓) Con.Acc. (↑) Tar.Acc.(↓) Con.Acc. (↑) zero-shot 47.74 66.70 54.22 68.34 64.54 77.44

Method

Non-linear Finetuning [16] 25.05 63.91 20.29 66.38 18.09 74.39 Non-lin. FT+OrthoReg (ours) 18.55 64.07 19.51 67.42 16.33 75.39

∆ -6.50 +0.16 -0.78 +1.04 -1.76 +1.00 Tangent Task Arithmetic [32] 11.47 63.99 9.33 66.82 8.36 74.39

###### TTA+OrthoReg (ours) 11.39* 64.07 7.49* 66.73 8.36* 74.87

∆ -0.06 +0.08 -1.84 -0.09 +0.00 +0.48 Attention-Only Fine-tuning [19] 19.39 64.90 19.20 67.75 24.85 76.42

ATT-FT+OrthoReg (ours) 15.67 64.16 14.78 66.81 14.67 75.40 ∆ -3.72 -0.74 -4.42 -0.94 -10.18 -1.02

LoRA-ATT 20.10 64.51 19.44 67.28 22.17 75.81 LoRA-ATT+OrthoReg (ours) 19.19 64.43 17.25 67.08 13.94 74.45 ∆ -0.91 -0.08 -2.19 -0.20 -8.23 -1.36

Zero-shot without OrthoReg with OrthoReg

arithmetic approach [16]. (2) Tangent Task Arithmetic (TTA) [32] that fine-tunes on a linearized version of the model. (3) Attention-Only Fine-tuning (ATT-FT) that finetunes only attention modules [19]. In addition, we investigate the effectiveness of OrthoReg on Parameter-Efficient Fine-Tuning (PEFT) approaches, such as LoRA [14]. Our main result tables include a strong PEFT baseline, LoRAATT, where adapters are applied to the query, key, value, and output projections. A detailed analysis of other LoRA configurations is presented in Appendix J.5.

|[Figure 82]<br><br>[Figure 83]| |
|---|---|
|[Figure 84]| |

|[Figure 85]<br><br>[Figure 86]| |
|---|---|
|[Figure 87]| |

| |[Figure 88]<br><br>[Figure 89]|
|---|---|
| |[Figure 90]|

Non-linear Finetuning Tangent Task Arithmetic Attention-Only Fine-tuning

Figure 4. The accuracy of merged models (ViT-L-14) across the eight benchmark tasks. Each subplot shows the performance for a specific baseline method: zero-shot (gray), the baseline’s merged model (red), and the baseline enhanced with our OrthoReg (blue).

Evaluation metrics. Consistent with [16, 19, 32], we use two metrics to evaluate performance. (1) Absolute Accuracy (Abs.Acc.), the standard classification accuracy of the merged multi-task model. (2) Normalized Accuracy (Norm.Acc.), which measures the performance of the multitask model relative to individually fine-tuned single-task models. The definition is in Appendix I.

Vit-l-14

our proposed orthogonal regularization (OrthoReg) consistently improves performance across all baselines and model scales. For instance, on the ViT-L-14 model, OrthoReg boosts the absolute accuracy of Non-lin. FT [16] by 4.16 points (from 84.07% to 88.23%) and enhances the seminal TTA [32] by 1.33 points (from 86.19% to 87.52%). Similar gains are observed for other baselines, highlighting OrthoReg’s effectiveness as a versatile, plug-and-play regularizer. These results empirically validate our hypothesis: actively enforcing an orthogonal structure on weight updates is a direct mechanism to mitigate interference. Notably, the ATT-FT+OrthoReg combination achieves the highest absolute accuracy across all tested configurations, establishing a new state-of-the-art on this benchmark.

To ensure a fair comparison, a single, uniform scaling coefficient α is applied to the sum of all task vectors (i.e., θMT = θ0 + α τt). This single coefficient is optimized for each method via a grid search on {0.0,0.05,...,1.0}. We emphasize that we do not employ more complex, taskadaptive strategies that would assign a different coefficient αt to each task vector. This approach, which is consistent with the evaluation protocols in prior work [19, 32], allows for a direct and fair comparison focused on the inherent quality of the task vectors produced by each method.

Per-Task Performance Analysis. Figure 4 provides a per-task breakdown of these improvements on the ViT-L-14 model. The blue area, representing the performance with OrthoReg, shows a clear expansion compared to the red area (baseline) across the majority of tasks and methods. This demonstrates that the gains from OrthoReg are not merely an average effect but represent a balanced and widespread

###### 5.2. Main Results on Task Addition

The primary results for the task addition benchmark are summarized in Table 1, where our method is applied to several leading task arithmetic methods across three scales of CLIP-based Vision Transformer models.

Overall Performance Comparison. Table 1 shows that

|𝜆𝜆 Value<br><br>Abs.Acc.<br><br>baseline (𝜆𝜆 = 0)<br><br>𝜆𝜆 Value<br><br>Abs.Acc.<br><br>baseline (𝜆𝜆 = 0)<br><br>| | |𝛼𝛼 Value<br><br>without OrthoReg<br><br>with OrthoReg<br><br>Abs.Acc.<br><br>𝛼𝛼 Value<br><br>without OrthoReg<br><br>with OrthoReg<br><br>Abs.Acc.|
|---|---|---|---|

(a) λ Sensitivity (b) α Sensitivity

Figure 6. Analysis of hyperparameter sensitivity on ViT-B-16. (a) The impact of the regularization strength λ on the performance of LoRA-ATT. (b) The influence of the merging coefficient α on the final accuracy of the merged model. The blue line (TTA+OrthoReg) consistently outperforms the red line (baseline TTA) across a wide range of α values.

(a) Non-lin. FT (b) Non-lin. FT+OrthoReg

Figure 5. Cosine similarity heatmaps of task vectors for ViT-B-16. (a) Task vectors from Non-lin. FT show high similarity for several task pairs. (b) Task vectors trained with OrthoReg are significantly more orthogonal.

performance lift across most individual tasks. The corresponding radar charts for ViT-B-32 and ViT-B-16, which show similar trends, are included in Appendix J.2.

Normalized Accuracy Analysis. The impact of OrthoReg is particularly striking in the Normalized Accuracy.

- As shown in Table 1, our method elevates the Norm.Acc. of Non-lin. FT to 100.08% and ATT-FT to 100.05% on ViT-L-

14. Achieving a normalized accuracy at or above 100% is the functional realization of ideal weight disentanglement, as it signifies that the single merged model performs on par with eight individually specialized models, indicating a near-total absence of task interference. This result provides strong empirical validation for Theorem 4, demonstrating that enforcing an orthogonal geometry on weight updates is an effective mechanism to achieve this state.

5.3. Main Results on Task Negation

Beyond combining capabilities, we also evaluate task negation θ = θ0 − ατt, which aims to make a model forget a specific task. Effective forgetting requires a sharp drop in target task accuracy while preserving performance on a control task (ImageNet) [16, 32].

- As shown in Table 2, our OrthoReg regularizer signifi-

cantly enhances the “forgetting” effect across all baseline methods. For instance, when applied to ATT-FT on the ViT-L-14 model, OrthoReg reduces target task accuracy by an additional 10.18 percentage points. This more thorough forgetting is achieved without compromising the model’s performance on ImageNet. Further details are provided in Appendix J.3. This result validates our theory that OrthoReg produces cleaner task vectors. Consequently, subtracting such a vector acts as a more precise “undo” operation, cleanly removing the target capability with minimal side effects on the model’s other general abilities.

###### 5.4. Validation of Inter-Task Orthogonality

Our theory predicts that OrthoReg promotes inter-task orthogonality (⟨τt,τj⟩ ≈ 0), a mechanism we term “Angle Control” in the proof of Theorem 4. To empirically validate

this, we visualize the pairwise cosine similarity of task vectors in a heatmap. Figure 5 compares the task vectors generated by Non-lin. FT on ViT-B-16. The baseline heatmap (a) shows significant off-diagonal brightness, indicating high correlation between distinct task vectors. In contrast, after applying OrthoReg (b), the heatmap becomes markedly darker. This result provides direct empirical evidence for our theoretical claims, demonstrating that OrthoReg improves task arithmetic by producing more geometrically disentangled task vectors. Additional heatmaps showing similar trends for other methods are in Appendix J.4.

###### 5.5. Parameter Sensitivity Analysis

We analyze the sensitivity to two key hyperparameters: the regularization strength λ and the task vector scaling coefficient α. Figure 6a illustrates that the model’s accuracy steadily improves as λ is increased, demonstrating that the performance gain is a direct and consistent result of the orthogonalization, not sensitive hyperparameter tuning. Figure 6b shows that the model trained with OrthoReg consistently outperforms the baseline across a wide range of α values. This indicates that OrthoReg produces higher-quality task vectors, which not only achieve a higher peak accuracy but also make the task merging process more robust and less sensitive to the choice of the scaling factor.

##### 6. Conclusion

Understanding why task arithmetic works is key to making it a reliable engineering tool. In this paper, we advance this understanding by discovering that Task-Feature Specialization ensures weight disentanglement and creates a geometric consequence: weight vector orthogonality. This insight led us to OrthoReg, a method that promotes disentanglement by enforcing orthogonality on weight updates. We found OrthoReg significantly improves performance by creating more orthogonal task vectors. For future work, we plan to explore more diverse forms of orthogonality constraints for more powerful control over model merging.

##### Acknowledgement

This work is supported in part by the National Natural Science Foundation of China (62576160, 62192783), Young Elite Scientists Sponsorship Program by CAST (2023QNRC001), and the Australian Research Council’s Discovery Project(DP220101784).

##### References

- [1] P.A. Absil, R. Mahony, and R. Sepulchre. Optimization Algorithms on Matrix Manifolds. Princeton University Press,

2009. 22

- [2] Mart´ın Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. In Proceedings of the 33nd International Conference on Machine Learning (ICML), pages 1120–1128, 2016. 2, 19
- [3] Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. CoRR, abs/1607.06450, 2016. 15, 16
- [4] Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proc. IEEE, 105(10):1865–1883, 2017. 6
- [5] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3606–3613, 2014. 6
- [6] H.S.M. Coxeter and S.L. Greitzer. Geometry Revisited. Mathematical Association of America, 1967. 19
- [7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021. 1
- [8] Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211–218,

1936. 19

- [9] Antonio Andrea Gargiulo, Donato Crisostomi, Maria Sofia Bucarelli, Simone Scardapane, Fabrizio Silvestri, and Emanuele Rodol`a. Task singular vectors: Reducing task interference in model merging. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18695–18705, 2025. 2
- [10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In IEEE International Conference on Computer Vision (ICCV), pages 1026–1034,

2015. 2

- [11] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE J. Sel. Top. Appl. Earth Obs. Remote. Sens., 12(7):2217–2226,

2019. 6

- [12] Yihuai Hong, Yuelin Zou, Lijie Hu, Ziqian Zeng, Di Wang, and Haiqin Yang. Dissecting fine-tuning unlearning in large

- language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3933–3941, 2024. 1
- [13] R.A. Horn and C.R. Johnson. Matrix Analysis. Cambridge University Press, 1990. 18, 21
- [14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations (ICLR), 2022. 7, 12
- [15] Lei Huang, Dawei Yang, Bo Lang, and Jia Deng. Decorrelated batch normalization. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages 791–800. Computer Vision Foundation / IEEE Computer Society, 2018. 15, 16
- [16] Gabriel Ilharco, Marco T´ulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In International Conference on Learning Representations (ICLR), 2023. 1, 2, 3, 6, 7, 8, 12, 23, 26, 27
- [17] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 611 July 2015, pages 448–456. JMLR.org, 2015. 15, 16
- [18] Arthur Jacot, Cl´ement Hongler, and Franck Gabriel. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems (NeurIPS), pages 8580–8589, 2018. 3
- [19] Ruochen Jin, Bojian Hou, Jiancong Xiao, Weijie J. Su, and Li Shen. Fine-tuning attention modules only: Enhancing weight disentanglement in task arithmetic. In International Conference on Learning Representations (ICLR), 2025. 2, 6, 7, 23, 26, 27
- [20] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. CoRR, abs/2001.08361, 2020. 3, 12
- [21] Jonathan Krause, Jia Deng, Michael Stark, and Li Fei-Fei. Collecting a large-scale dataset of fine-grained cars. In Proceedings of the IEEE International Conference on Computer Vision (ICCV) Workshops, 2013. 6, 25
- [22] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. 6
- [23] Hongyu Li, Liang Ding, Meng Fang, and Dacheng Tao. Revisiting catastrophic forgetting in large language model tuning. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 4297–4308, 2024. 1
- [24] Hongkang Li, Yihua Zhang, Shuai Zhang, Pin-Yu Chen, Sijia Liu, and Meng Wang. When is task vector provably effective for model editing? A generalization analysis of nonlinear transformers. In International Conference on Learning Representations (ICLR), 2025. 2
- [25] D.G. Luenberger and Y. Ye. Linear and Nonlinear Programming. Springer US, 2008. 20

- [26] Wei Luo and Dihong Gong. Pre-trained large language models for financial sentiment analysis. CoRR, abs/2401.05215,

2024. 1

- [27] Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. CoRR, abs/2308.08747, 2023. 1
- [28] Nikita Makarov, Maria Bordukova, Papichaya Quengdaeng, Daniel Garger, Raul Rodriguez-Esteban, Fabian Schmich, and Michael P. Menden. Large language models forecast patient health trajectories enabling digital twins. npj Digit. Medicine, 8(1), 2025. 1
- [29] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations (ICLR), 2018. 2, 3, 19
- [30] Marius Mosbach, Maksym Andriushchenko, and Dietrich Klakow. On the stability of fine-tuning BERT: misconceptions, explanations, and strong baselines. In 9th International Conference on Learning Representations (ICLR),

2021. 1

- [31] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Baolin Wu, Andrew Y Ng, et al. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervised feature learning, page 7. Granada, 2011. 6
- [32] Guillermo Ortiz-Jimenez, Alessandro Favero, and Pascal Frossard. Task arithmetic in the tangent space: Improved editing of pre-trained models. In Advances in Neural Information Processing Systems, pages 66727–66754, 2023. 2, 3, 6, 7, 8, 12, 22, 23, 26, 27
- [33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), pages 8748–8763, 2021. 1, 6
- [34] T.J. Rivlin. The Chebyshev Polynomials. Wiley, 1974. 19
- [35] W. Rudin. Principles of Mathematical Analysis. McGrawHill, 1976. 19
- [36] Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How does batch normalization help optimization? In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr´eal, Canada, pages 2488–2498, 2018. 16
- [37] Andrew M. Saxe, James L. McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. In International Conference on Learning Representations (ICLR), 2014. 2
- [38] Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. The german traffic sign recognition benchmark: A multi-class classification competition. In The 2011 International Joint Conference on Neural Networks (IJCNN), pages 1453–1460. IEEE, 2011. 6
- [39] George Stoica, Pratik Ramesh, Boglarka Ecsedi, Leshem Choshen, and Judy Hoffman. Model merging with SVD to

- tie the knots. In International Conference on Learning Representations (ICLR), 2025. 1
- [40] Anke Tang, Li Shen, Yong Luo, Yibing Zhan, Han Hu, Bo Du, Yixin Chen, and Dacheng Tao. Parameter-efficient multi-task model fusion with partial linearization. In International Conference on Learning Representations (ICLR),

2024. 2

- [41] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aur´elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023. 1
- [42] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017. 12
- [43] Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International Conference on Machine Learning (ICML), pages 23965–23998,

2022. 3

- [44] Jianxiong Xiao, James Hays, Krista A. Ehinger, Aude Oliva, and Antonio Torralba. SUN database: Large-scale scene recognition from abbey to zoo. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3485– 3492, 2010. 6
- [45] Feng Xiong, Runxi Cheng, Wang Chen, Zhanqiu Zhang, Yiwen Guo, Chun Yuan, and Ruifeng Xu. Multi-task model merging via adaptive weight disentanglement. CoRR, abs/2411.18729, 2024. 14
- [46] Prateek Yadav, Derek Tam, Leshem Choshen, Colin A. Raffel, and Mohit Bansal. Ties-merging: Resolving interference when merging models. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 2, 3, 12
- [47] Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, Jie Zhang, and Dacheng Tao. Model merging in llms, mllms, and beyond: Methods, theories, applications and opportunities. CoRR, abs/2408.07666, 2024. 1, 2
- [48] Enneng Yang, Zhenyi Wang, Li Shen, Shiwei Liu, Guibing Guo, Xingwei Wang, and Dacheng Tao. Adamerging: Adaptive model merging for multi-task learning. In International Conference on Learning Representations (ICLR), 2024. 1, 2
- [49] Yiting Yang, Hao Luo, Yuan Sun, Qingsen Yan, Haokui Zhang, Wei Dong, Guoqing Wang, Peng Wang, Yang Yang, and Hengtao Shen. Efficient adaptation of pre-trained vision transformer underpinned by approximately orthogonal finetuning strategy. CoRR, abs/2507.13260, 2025. 2, 19
- [50] Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. Editing large language models: Problems, methods, and op-

- portunities. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 10222–10240, 2023. 1
- [51] Kotaro Yoshida, Yuji Naraki, Takafumi Horie, Ryosuke Yamaki, Ryotaro Shimizu, Yuki Saito, Julian J. McAuley, and Hiroki Naganuma. Mastering task arithmetic: τjp as a key indicator for weight disentanglement. In International Conference on Learning Representations (ICLR), 2025. 2, 3
- [52] Maxime Zanella and Ismail Ben Ayed. Low-rank few-shot adaptation of vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024 - Workshops, Seattle, WA, USA, June 17-18, 2024, pages 1593–1603. IEEE, 2024. 30
- [53] Siqi Zeng, Yifei He, Weiqiu You, Yifan Hao, Yao-Hung Hubert Tsai, Makoto Yamada, and Han Zhao. Efficient model editing with task vector bases: A theoretical framework and scalable approach. CoRR, abs/2502.01015, 2025. 2
- [54] Haobo Zhang and Jiayu Zhou. Unraveling lora interference: Orthogonal subspaces for robust model merging. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 26459–

26472. Association for Computational Linguistics, 2025. 2

- [55] Hanqing Zhang, Haolin Song, Shaoyu Li, Ming Zhou, and Dawei Song. A survey of controllable text generation using transformer-based pre-trained language models. CoRR, abs/2201.05337, 2022. 1
- [56] Zhengyan Zhang, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. Moefication: Transformer feedforward layers are mixtures of experts. In Findings of the Association for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 877–890. Association for Computational Linguistics, 2022. 3, 12

### Understanding and Enforcing Weight Disentanglement in Task Arithmetic Supplementary Material

##### A. Note on the Scope of Analysis: Why Focus on Linear Layers

Throughout our theoretical analysis, we primarily focus on the parameters of linear layers, such as fully-connected (FC) layers and the projection matrices within attention mechanisms. We omit biases and parameters from normalization layers (e.g., LayerNorm).

This simplification is well-justified, as linear layers constitute the vast majority of parameters [20, 56] in modern large-scale models like Transformers [42], and their behavior consequently dictates the model’s overall functional transformations and capacity for learning taskspecific knowledge. Moreover, this focus aligns with established practices in the model merging literature, where complex strategies are often applied exclusively to linear layers [16, 32, 46], suggesting a secondary role for biases and normalization parameters in the task interference phenomena we aim to mitigate. The centrality of these layers is further underscored by the success of parameter-efficient fine-tuning (PEFT) methods like LoRA [14], which demonstrate that model adaptation for new tasks primarily occurs within these linear components.

Given this convergence of evidence, concentrating our geometric analysis on linear layers allows us to build a tractable yet powerful theoretical framework that captures the core mechanisms of task arithmetic.

##### B. Justification for Two-Task Simplification

In our main analysis (Section 4.1), we simplify the full definition of weight disentanglement (Definition 1) to a twotask, in-domain scenario as,

f(x; θ0 + τt + τj) = f(x; θ0 + τt), ∀ x ∈ Dt. (10)

This appendix provides a detailed justification for why this simplification is sufficient and does not result in a loss of generality. Our simplification is reasonable for two primary reasons.

First, our subsequent proofs focus on demonstrating that the pairwise interference term τj⊤J(x) is approximately zero for any x in the data domain Dt of a different task t. This is the core of the disentanglement mechanism under the NTK linearization hypothesis. Due to the linearity of this interference term with respect to the task vectors, proving the disappearance of pairwise interference is sufficient for the general multi-task case. Specifically, if τj⊤J(x) ≈ 0 for all j ̸= t, then the total interference from all other tasks

in the merged model also vanishes,

αj(τj⊤J(x)) ≈

αj · 0 = 0. (11)

j̸=t

j̸=t

Therefore, focusing on two-task interaction f(x;θ0 + τt + τj) and omitting the scaling coefficients α during the proof does not compromise the generality of our conclusions.

Second, our analysis concentrates on the “in-domain disentanglement” condition because it addresses the central challenge of eliminating crosstalk between actively composed tasks. The“out-of-domain preservation” condition, f(x;θ0 + Tt=1 αtτt) = f(x;θ0) for x ∈/ Tt=1 Dt, can be established using the same underlying logic. For an out-ofdomain sample xood, its processing should ideally not rely on the specialized features of any task t. This implies that the interference term τt⊤J(xood) should be approximately zero for all task vectors τt. This is a direct extension of the principle we prove for the in-domain case. By establishing the core argument for pairwise in-domain disentanglement, we effectively provide the necessary and sufficient reasoning to prove the full weight disentanglement property.

##### C. Proof of Lemma 2

In this part, we provide the detailed proof for Lemma 2, which establishes the equivalence between the functional property of weight disentanglement and a geometric orthogonality condition under the NTK linearization hypothesis.

Lemma 2. Under the NTK linearization hypothesis, weight disentanglement between tasks t and j is equivalent to the interference term from task j being approximately zero on the data domain of task t:

τj⊤J(x) = 0, ∀x ∈ Dt. (12)

Proof. Our starting point is the simplified, two-task definition of weight disentanglement, which states that for any input x from the data domain of task t, the following approximation should hold:

f(x;θ0 + τt + τj) = f(x;θ0 + τt), ∀x ∈ Dt. (13)

We apply the first-order Taylor approximation from the NTK hypothesis to both sides of this equation.

For the left-hand side (LHS), the total parameter perturbation from the pre-trained state θ0 is (τt + τj). The linearization is therefore,

LHS ≈ f(x;θ0) + (τt + τj)⊤J(x)

= f(x;θ0) + τt⊤J(x) + τj⊤J(x). (14)

For the right-hand side (RHS), the perturbation is simply τt. The linearization is,

RHS ≈ f(x;θ0) + τt⊤J(x). (15)

By substituting these approximations from Equation (14) and Equation (15) back into the original weight disentanglement condition (Equation (13)), we obtain,

f(x;θ0)+τt⊤J(x)+τj⊤J(x) ≈ f(x;θ0)+τt⊤J(x). (16)

Canceling the common terms f(x;θ0) and τt⊤J(x) from both sides of the approximation leaves us with the final, equivalent condition:

τj⊤J(x) = 0, ∀x ∈ Dt. (17)

This shows that, under NTK linearization, the functional requirement that task j does not interfere with task t is equivalent to the geometric condition that the task vector τj is orthogonal to the model’s gradient Jacobian J(x) for all data points x in the domain of task t.

| |
|---|

##### D. Detailed Proof of Theorem 3

###### D.1. Proof of Theorem 1

In this section, we provide the formal proof for Theorem 1.

Theorem 3. Under the NTK linearization hypothesis (Section 3.3) and the Task-Feature Specialization property, weight disentanglement between tasks t and j holds.

Proof. According to Lemma 1, our goal is to prove that the interference term τj⊤J(x) is approximately zero for any x ∈ Dt. We can decompose this total interference into contributions from each linear layer. For clarity, we analyze the interference arising from a single weight matrix W ∈ Rm×d and show it is zero. The conclusion generalizes to the entire model by summation.

The interference contributed by W is ⟨(τj)W,JW(x)⟩, where (τj)W and JW(x) are the components of the task vector and Jacobian corresponding to W. By decomposing this along the column vectors {w1,...,wd} of W, we get,

d

f(x;θ0)⟩, (18)

⟨(τj)k,∇wk

InterferenceW(x) =

k=1

where (τj)k is the update applied to column wk. We will show every term in this summation is approximately zero.

f(x;θ0)). For an input x ∈ Dt, the gradient of the model output with respect to a weight column wk can be expressed using the chain rule,

###### Analysis of the gradient term (∇wk

∂f(x;θ0) ∂wk

∇wk

f(x;θ0) =

=

∂f(x;θ0) ∂zk ·

∂zk ∂wk

. (19)

According to Definition 2, if the feature index k is not in the specialized set for task t (i.e., k ∈/ It), the model’s output is insensitive to it, meaning ∂f(x;θ

0)

∂zk ≈ 0. For x ∈ Dt, k ∈/ It =⇒ ∇wk

f(x;θ0) ≈ 0. (20)

Analysis of the task Vector term ((τj)k). The task vector component (τj)k is the accumulated update to weight wk from fine-tuning on task j. By definition, if feature k is not specialized for task j (i.e., k ∈/ Ij), the loss function for task j is insensitive to it. This means the gradients with respect to wk computed on the data domain Dj are consistently negligible. Since (τj)k is the sum of these negligible gradients, it will be approximately zero. (A detailed proof is provided in Appendix D.2 as Proposition 1).

k ∈/ Ij =⇒ (τj)k ≈ 0. (21) Now, we examine each term ⟨(τj)k,∇wk

f(x;θ0)⟩ in the summation for index k ∈ {1,...,d}. There are two mutually exclusive possibilities.

- Case A: k ∈ Ij. By the Task-Feature Specialization

property (It∩Ij = ∅), it must be that k ∈/ It. From gradient analysis (Equation (20)), this implies ∇wk

f(x;θ0) ≈ 0.

- Case B: k ∈/ Ij. From task vector analysis (Equa-

tion (21)), this implies (τj)k ≈ 0. In both cases, the term ⟨(τj)k,∇wk

f(x;θ0)⟩ vanishes. Since this holds for all k, the interference from this layer, InterferenceW(x), is approximately zero. As this applies to all layers, the total interference τj⊤J(x) ≈ 0. By Lemma 2, this proves that weight disentanglement holds.

| |
|---|

###### D.2. Supporting Proposition for Theorem 3

In this part, we provide a detailed proof for the proposition referenced in the proof of Theorem 3. This proposition formalizes the intuition that if a task does not depend on a specific feature, the fine-tuning process for that task will not significantly alter the weights associated with that feature.

Proposition 1. Under the NTK Linearization hypothesis (Section 3.3) and the Task-Feature Specialization property, consider the fine-tuning process for task j on its data domain Dj. If a feature index k does not belong to the specialized feature set for task j (i.e., k ∈/ Ij), then the corresponding component of the resulting task vector, (τj)k, is approximately zero.

k ∈/ Ij =⇒ (τj)k ≈ 0. (22)

Proof. The task vector τj is defined as the total change in parameters after fine-tuning on task j, starting from the pretrained weights θ0,

###### τj = θj∗ − θ0, (23)

where θj∗ are the final fine-tuned parameters. The component (τj)k specifically represents the change in the weight column wk of a given linear layer.

Let’s model the fine-tuning process as a sequence of updates using a gradient-based optimizer, such as Stochastic Gradient Descent (SGD). For a total of S update steps, the weight column wk is updated iteratively. The update rule for wk at step s is,

wk(s+1) = wk(s) − η · Ex∼D

j ∇wkLj(x;θ(s)) , (24)

where η is the learning rate, and θ(s) represents the model parameters at step s, with the initial state being θ(0) = θ0.

Consistent with the perspective from work on Adaptive Weight Disentanglement (AWD) [45] that views the task vector as the sum of accumulated gradients, the total change in the weight column wk, which is the task vector component (τj)k, is the sum of all single-step updates over the course of training,

S−1

(τj)k = wk(S) − wk(0) =

wk(s+1) − wk(s)

s=0

S−1

j ∇wkLj(x;θ(s)) .

Ex∼D

= −η

s=0

(25)

To prove that (τj)k ≈ 0, we need to show that the expected gradient Ex∼D

j ∇wkLj(x;θ(s)) is approximately zero at every step s of the fine-tuning process.

Let’s analyze the gradient for a single data point x ∈ Dj using the chain rule,

∂f(x;θ(s)) ∂zk ·

∂Lj ∂f(x;θ(s)) ·

∇wkLj(x;θ(s)) =

∂zk ∂wk

, (26)

where zk is the activation of the base feature corresponding to wk. We analyze each term in this product.

- • ∂f(∂xL;θj

(s)). This is the derivative of the loss with respect to the model’s final output. Before the model has fully converged, this term is generally non-zero and bounded.

- • ∂zk ∂wk . For a standard linear layer where zk = (wk)⊤In(x), this derivative is simply the input to the layer, In(x). This term is also non-zero and bounded.

- • ∂f(x;θ

(s))

∂zk . It measures the sensitivity of the final model output to the intermediate feature activation zk. Our core assumption is that k ∈/ Ij. By Definiton 2 (TaskSpecialized Feature Set), this means that at the pre-trained state θ0, the model’s output is insensitive to zk in expectation over the data domain Dj,

∂f(x;θ0) ∂zk ≈ 0. (27)

Ex∼D

j

The fine-tuning process occurs in the neighborhood of θ0. Under the NTK linearization hypothesis, the parameter

changes are small, and the model’s Jacobian is assumed to be stable. Therefore, for all steps s in the fine-tuning process, θ(s) remains close to θ0, and the sensitivity of the model’s output to feature zk also remains negligible,

∂f(x;θ(s)) ∂zk ≈ 0 for s = 0,1,...,S − 1.

Ex∼D

j

(28)

Now, let’s take the expectation of the full gradient expression (Equation (26)) over the data domain Dj,

j ∇wkLj(x;θ(s))

Ex∼D





∂f(x;θ(s)) ∂zk Expectation≈0

∂Lj ∂f(x;θ(s))

∂zk ∂wk non-zero,bounded

= Ex∼D

·

·

.

 

 

j

non-zero,bounded

(29) Since the expectation of the sensitivity term ∂z∂f

is approximately zero, and the other terms are bounded, the expectation of their product will also be approximately zero.

k

This holds for every step s of the fine-tuning process. Substituting this result back into Equation (25), we find that the total update (τj)k is a finite sum of near-zero vectors,

S−1

j ∇wkLj(x;θ(s))

≈ 0. (30)

Ex∼D

(τj)k = −η

s=0

≈0

This demonstrates that if a feature is not part of a task’s specialized set, the corresponding weights will remain virtually unchanged during fine-tuning for that task.

This completes the proof.

| |
|---|

##### E. Proof of Corollary 2

This part provides the detailed proof for Corollary 2, which establishes that the Task-Feature Specialization (TFS) property, a functional characteristic of an ideal pre-trained model, gives rise to a specific geometric structure in its parameters, namely, weight vector block-orthogonality. This formalizes the connection, that Weight Vector Orthogonality (WVO) is presented as a geometric consequence of TFS.

Corollary 2. Given a model that adheres to the TaskFeature Specialization (TFS) property, its weight matrices will exhibit Block Orthogonality.

###### E.1. TFS Implies Cross-Task Feature Decorrelation

We begin by proving a key statistical consequence of TFS, which will be instrumental in our main proof. The functional separation defined by TFS has a direct consequence on the statistical properties of the feature activations. We formalize this as the following proposition.

Proposition 2. Under the Task-Feature Specialization (TFS) property, for any two distinct tasks t ̸= j, and for any pair of features with indices k ∈ It and l ∈ Ij, their activations zk and zl are approximately decorrelated over a mixed data distribution µ. That is,

Covµ(zk,zl) ≈ 0. (31)

Proof. Let us assume the contrary. Suppose TFS holds, but two features zk (specialized for task t, i.e., k ∈ It) and zl (specialized for task j, i.e., l ∈ Ij) are statistically correlated. For simplicity, we can model this correlation with an approximate linear relationship,

zk ≈ a · zl + b + ξ, (32)

where a ̸= 0 is a correlation coefficient, b is a bias, and ξ is uncorrelated noise. This model implies that a change in zl systematically induces a change in zk.

Now, consider the total derivative of the model’s final output f(x;θ0) with respect to the activation zl for an input x from task t’s data domain, Dt. Using the chain rule, the change in f with respect to a change in zl has two paths: a direct path (zl → f) and an indirect path through the correlated feature zk (zl → zk → f).

df(x;θ0) dzl

∂f ∂zl

∂f ∂zk

- ∂zk

- ∂zl

(33)

=

+

We analyze each term in the context of TFS for an input x ∈ Dt.

- • ∂z∂f

l

: Since x ∈ Dt and the feature l is specialized for task j (l ∈ Ij), the TFS assumption (It ∩ Ij = ∅) implies l ∈/ It. By Definition 2, the model’s output is insensitive to zl on this data domain. Thus, Ex∼D

t

[|∂z∂f

l

|] ≈ 0.

- • ∂z∂f

k

: Since x ∈ Dt and the feature k is specialized for task t (k ∈ It), the model’s output is sensitive to zk. Thus, Ex∼D

t

[|∂z∂f

k

|] is significantly non-zero.

- • ∂zk ∂zl : From our linear correlation model, this derivative is the correlation coefficient a, which we assumed to be non-zero.

Substituting these into the chain rule expression and taking the expectation over Dt,

Ex∼D

t

∂f ∂zk non-zero

df dzl ≈ Ex∼D

- ∂zk

- ∂zl

∂f ∂zl ≈0

·

+

t

non-zero, a ≈ |a| · Ex∼D

∂f ∂zk

.

t

(34)

Since |a| ̸= 0 and E[|∂z∂f

|] is significantly non-zero, the result is a significantly non-zero value. This means that the model’s output f shows a non-negligible total sensitivity to the activation zl on data from task t.

k

This result, however, directly contradicts the premise of TFS. If a model has truly specialized feature k for task t and feature l for task j, its function for task t should not be affected by perturbations in zl. The total effect of zl on the output, not just the partial derivative, should be negligible.

The contradiction arose from our initial assumption of correlation (a ̸= 0). Therefore, that assumption must be false. We conclude that for TFS to hold, features specialized for different tasks must be statistically decorrelated.

| |
|---|

###### E.2. Detailed proof of Corollary 2

Proof. The proof proceeds by first relating the geometric property of the weight matrix (W⊤W) to a statistical property of the feature activations (the covariance matrix Σz), and then showing that TFS imposes a block-diagonal structure on this covariance matrix.

Step 1: Connecting Weight Geometry to Feature Covariance.

Consider a single linear layer with weight matrix W = [w1,...,wd] ∈ Rm×d, input In(x) ∈ Rm, and feature activations z = W⊤In(x) ∈ Rd. We compute the covariance matrix Σz of the feature activations under a mixed data distribution µ,

Σz = Ex∼µ[(z − µz)(z − µz)⊤], where µz = Ex∼µ[z]. (35)

In modern deep neural networks, the presence of normalization layers like Layer Normalization (LN) [3] or Batch Normalization (BN) [17] is standard practice. A primary function of these layers is to standardize the activations, dynamically regulating their mean and variance [3, 15, 17]. This forces the mean of the layer’s input, µIn = Ex∼µ[In(x)], to be approximately zero.

Consequently, the mean of the output feature activations is also approximately zero,

µz = Ex∼µ[W⊤In(x)] = W⊤Ex∼µ[In(x)] = W⊤µIn ≈ 0.

(36) With this zero-mean property, the covariance matrix Σz simplifies to the second-moment matrix,

Σz = Ex∼µ[zz⊤] = Ex∼µ[W⊤In(x)In(x)⊤W]. (37)

Since the weight matrix W is constant with respect to the input x, we can move it outside the expectation:

Σz = W⊤ Ex∼µ[In(x)In(x)⊤] W. (38)

At this point, we analyze the term Ex∼µ[In(x)In(x)⊤], which represents the second moment matrix of the layer’s input. As argued before, normalization layers standardize activations. Beyond just enforcing a zero mean, this process also regulates variance, driving the covariance matrix of the

layer’s input, ΣIn, towards a whitened state [3, 15, 17, 36]. The covariance matrix of the input is defined as,

ΣIn = Ex∼µ[(In(x) − µIn)(In(x) − µIn)⊤]

- (39)

Given that the input is whitened, we have ΣIn ≈ Im and µIn ≈ 0. Substituting these into the definition gives us the second-moment matrix of the input,

Ex∼µ[In(x)In(x)⊤] = ΣIn + µInµ⊤In ≈ Im + 0 · 0⊤ = Im.

- (40)

= Ex∼µ[In(x)In(x)⊤] − µInµ⊤In

Substituting this result back into the expression for Σz, we arrive at the crucial link between the weights’ geometry and the features’ statistics,

Σz = W⊤ImW = W⊤W. (41) This equation shows that in this case the Gram matrix of the weights, W⊤W, is identical to the covariance matrix of the feature activations, Σz. Proving that W has blockorthogonal columns is now equivalent to proving that its Gram matrix W⊤W is block-diagonal, which in turn is equivalent to proving that Σz is block-diagonal.

Step 2:Proving the Block-Diagonal Structure of Σz. An element (Σz)kl of the covariance matrix is, by def-

inition, the covariance between zk and zl, i.e., (Σz)kl = Covµ(zk,zl).

Let’s consider two distinct feature indices, k ̸= l.

- Case 1: Features are specialized for different tasks. Sup-

pose k ∈ It and l ∈ Ij for two tasks t ̸= j. According to

- Proposition 2, which we derived from the TFS property, the activations of these features are decorrelated over the mixed distribution µ. Therefore, we directly have,

(Σz)kl = Covµ(zk,zl) ≈ 0. (42)

- Case 2: Features are specialized for the same task. Sup-

pose k,l ∈ It for some task t, with k ̸= l. Our theory does not make any assumption about intra-task feature decorrelation. Therefore, the term (Σz)kl = Covµ(zk,zl) is not guaranteed to be zero and may be non-zero in general.

Step 3: Conclusion of Block-Orthogonality

From Step 2, we have shown that the off-diagonal elements of the covariance matrix Σz are approximately zero whenever the indices correspond to different tasks. The elements corresponding to pairs of features within the same task may be non-zero. This means Σz has a block-diagonal structure,

  

  . (43)

###### B1 0 ... 0 0 B2 ... 0

Σz = W⊤W ≈

... .

. .

0 0 ... BT

where Bt is the (generally non-diagonal) covariance submatrix for features whose indices are in the set It, and the 0 blocks represent matrices with near-zero entries.

The (k,l)-th element of the Gram matrix W⊤W is the inner product of the column vectors ⟨wk,wl⟩. The blockdiagonal structure of W⊤W directly implies that if indices k and l belong to different blocks (i.e., k ∈ It and l ∈ Ij with t ̸= j), their corresponding entry in the Gram matrix is approximately zero,

⟨wk,wl⟩ = (W⊤W)kl ≈ 0. for k ∈ It,l ∈ Ij,t ̸= j

(44) This is precisely the definition of block-orthogonality for the columns of the weight matrix W. The set of column vectors {wk}k∈It

forms a subspace that is orthogonal to the subspace spanned by {wl}l∈Ij

for any j ̸= t. This completes the proof.

| |
|---|

##### F. Bayesian Analysis of the Relationship between TFS, WVO, and WD

This part provides a formal Bayesian analysis to justify the claim made in Section 4.2.4, that observing Weight Vector Orthogonality (WVO) in a pre-trained model strongly increases our belief that it will exhibit Weight Disentanglement (WD). This analysis formalizes the intuition that WVO acts as a powerful diagnostic clue for the desirable, yet abstract, property of Task-Feature Specialization (TFS).

Let us define three distinct events.

- • Event A: The model has achieved ideal Task-Feature Specialization (TFS). This represents the underlying, unobservable abstract property where the model allocates disjoint sets of internal features to different tasks.
- • Event B: The model exhibits Weight Disentanglement (WD). This is the desired functional outcome where task vectors can be composed without destructive interference.
- • Event C: The model’s parameters possess Weight Vector Orthogonality (WVO). This is a concrete, measurable geometric property of the model’s weight matrices.

Our core theory, as established in Section 4.2, posits that TFS is a sufficient condition for both WD (Theorem 3) and WVO (Corollary 2). We can formalize this as a logical implication,

###### A =⇒ (B ∧ C). (45)

This means that if Event A is true, then both Event B and Event C must also be true. Consequently, we have the conditional probabilities,

P(B|A) = 1 and P(C|A) = 1. (46)

Our goal is to demonstrate that observing WVO (Event C) provides evidence for WD (Event B). In probabilistic terms, we aim to show that the posterior probability of WD

given WVO is greater than the prior probability of WD,

###### P(B|C) > P(B). (47)

First, we can expand the conditional probability P(B|C) by conditioning on whether TFS (Event A) has occurred, P(B|C) = P(B|A,C)P(A|C) + P(B|¬A,C)P(¬A|C).

(48) Let’s analyze the terms in this expression.

- 1. P(B|A,C): Since Event A (TFS) is a sufficient con-

dition for Event B (WD), if A is true, B must be true, regardless of C. Therefore, P(B|A,C) = 1.

- 2. P(B|¬A,C): This is the probability of observing

WD when TFS is not present, even though WVO is. Without the foundational structure of TFS, WD is not guaranteed. It might occur due to other unknown reasons or by chance, but we can reasonably assume this probability is significantly less than 1. Let’s denote this probability as q, where 0 ≤ q < 1.

Substituting these into the equation, we get,

P(B|C) = 1 · P(A|C) + q · P(¬A|C). (49) Rearranging this gives,

P(B|C) = q + (1 − q)P(A|C). (50)

Now, we examine the crucial term P(A|C), which represents our updated belief in TFS after having observed WVO. Using Bayes’ theorem,

P(C|A)P(A) P(C)

. (51)

P(A|C) =

As established earlier, P(C|A) = 1. This simplifies the expression to,

P(A) P(C)

. (52)

P(A|C) =

Here, P(A) is our prior belief that a model has achieved TFS, and P(C) is the prior probability of observing WVO. WVO is a specific geometric structure that is not guaranteed to occur in any arbitrary neural network; its emergence is non-trivial. Therefore, it is safe to assume that P(C) < 1.

This leads to a key inequality,

P(A) P(C)

> P(A). (53)

P(A|C) =

This inequality formally captures our intuition: observing the geometric signature of WVO (Event C) strengthens our belief that the model has developed the underlying functional structure of TFS (Event A).

To complete the proof, we compare the expression for P(B|C) with the unconditional prior probability of WD, P(B). Using the law of total probability again,

P(B) = P(B|A)P(A) + P(B|¬A)P(¬A). (54)

We know P(B|A) = 1. For the term P(B|¬A), we introduce a reasonable assumption: in the absence of the common cause (TFS), its consequences (WD and WVO) are approximately conditionally independent.

###### P(B|¬A,C) ≈ P(B|¬A). (55)

This assumption is justified because if the fundamental mechanism (TFS) that links WD and WVO is absent, the correlation between them should vanish or be significantly diminished. Any residual correlation would be a minor influence. Under this assumption, P(B|¬A) ≈ P(B|¬A,C) = q.

Substituting this into the expression for P(B):

P(B) ≈ 1·P(A)+q·(1−P(A)) = q+(1−q)P(A). (56)

We now have two expressions to compare:

- 1. P(B|C) = q + (1 − q)P(A|C);
- 2. P(B) ≈ q + (1 − q)P(A). We have proved that P(A|C) > P(A). Since q < 1, the

term (1 − q) is positive. It therefore follows directly that,

###### P(B|C) > P(B) (57)

This result provides a rigorous probabilistic foundation for our central thesis. It demonstrates that observing the measurable geometric property of Weight Vector Orthogonality is a strong piece of evidence that increases the likelihood that the model also possesses the desired functional property of Weight Disentanglement. This justifies using WVO as a diagnostic tool to assess a model’s suitability for task arithmetic.

##### G. Detailed Proof of Theorem 4

###### G.1. Proof of Theorem 4

Theorem 4. Under the NTK linearization hypothesis (Section 3.3), even if the Task-Feature Specialization property does not hold (i.e., It∩Ij ̸= ∅), constraining the task update matrices {∆Wt(l)} to be approximately internally orthogonal (as encouraged by the regularization in Definition 4) actively promotes weight disentanglement between tasks t and j.

Proof. According to Lemma 2, our goal is to demonstrate that the interference from task j on the data domain of task t is approximately zero, i.e., τj⊤J(x) ≈ 0. The interference term’s magnitude can be expressed as,

|τj⊤J(x)| = ||τj||2 · ||J(x)||2 · |cos∠(τj,J(x))|. (58) The proof proceeds in four steps. We first reframe the

angle term, then demonstrate how our regularizer controls both the norm and angle terms, and finally synthesize the results.

Step 1: Directional Alignment.

First, we establish that for a typical input x ∈ Dt, its Jacobian J(x) is directionally aligned with the task vector τt. The direction of τt is determined by the average Jacobian over the task’s data domain, µJ := Ex∈D

[J(x)]. Under a reasonable data consistency assumption, the gradients of different samples are statistically consistent rather than random, the direction of a typical J(x) aligns with that of µJ and, by extension, with τt. This alignment, rigorously proven in Appendix G.2, allows us to reframe the term’s angle using the angle between the two task vectors,

t

|τj⊤J(x)| ≈ ||τj||2 · ||J(x)||2 · |cos∠(τj,τt)|. (59) Step 2: Norm Control.

Our second step is to show that the orthogonal regularization term Lortho effectively bounds the norm of the task vectors. The regularizer penalizes the deviation of each update matrix ∆W from the identity. By solving a constrained optimization problem, we can prove that the Frobenius norm of an update matrix ∆W is strictly bounded by its deviation from orthogonality. As formalized in Proposition

- Proposition 3 (see Appendix G.3), if ∥∆W⊤∆W − I∥2F ≤ ξ, then the norm is bounded by,

∥∆W∥2F ≤ d + dξ, (60)

where d is the number of columns. As the task vector’s total norm is determined by the norms of its constituent update

matrices, ∥τj∥22 = l ∥∆Wj(l)∥2F, our regularizer effectively constrains the overall magnitude of τj.

Step 3: Angle Control. Our third and most critical step is to demonstrate that the

regularization statistically promotes orthogonality between different task vectors, i.e., E[|cos∠(τj,τt)|] ≈ 0.

The core mechanism is that the internal orthogonal structure imposed on each update matrix ∆W induces inter-task statistical orthogonality between the resulting task vectors τt and τj. This can be understood through the lens of Polar Decomposition [13], which allows us to express any approximately orthogonal update matrix ∆W as ∆W = QP, where Q is a strictly orthonormal matrix (an element of the Stiefel manifold Vd(Rm)) and P is a symmetric positive semi-definite matrix that is very close to the identity (as formalized in Proposition 4 in Appendix G.4.1).

Consequently, the inner product of two task vectors, ⟨τt,τj⟩, which is a sum of layer-wise inner products l⟨vec(∆Wt(l)),vec(∆Wj(l))⟩, is dominated by the sum of inner products of their orthonormal components,

l⟨vec(Q(tl)),vec(Q(jl))⟩ (see Appendix G.4 for a detailed derivation). As established in Lemma 3 (Appendix G.4.3), two matrices independently and uniformly drawn from the Stiefel manifold are, when vectorized, statistically orthogonal. Their inner product has an expected value of zero and

its probability distribution is sharply peaked at zero. This strong statistical tendency towards orthogonality at each layer propagates to the entire task vectors, ensuring that τt and τj are highly likely to be nearly orthogonal. The detailed proof can be seen in Appendix G.4.2

Step 4: Completing the Proof. We now synthesize the results. The magnitude of the interference term is given by,

· |cos∠(τj,τt)| Statistically near zero

|τj⊤J(x)| ≈ ||τj||2 Bounded

· ||J(x)||2

Inherently Bounded

(61)

The dual control mechanism of our regularization ensures that this product is approximately zero in expectation. The norm ∥τj∥ is bounded (Step 2), ∥J(x)∥ is bounded for any given input and model, and the cosine of the angle between task vectors is statistically driven towards zero (Step 3). Consequently, the expected interference is negligible,

E[|τj⊤J(x)|] ≈ 0. (62)

By Lemma 2, this establishes that weight disentanglement is approximately achieved. This completes the proof.

| |
|---|

###### G.2. Proof of Directional Alignment (Step 1)

In this section, we provide a rigorous proof for the claim that for a typical input x ∈ Dt, its Jacobian vector J(x) is directionally aligned with the task vector τt.

Proof. The proof proceeds in two parts: first, relating the direction of τt to the average Jacobian µJ, and second, relating the direction of an individual J(x) to µJ.

Part 1: Direction of the Task Vector τt.

As clarified in Equation (25), the task vector τt is the result of accumulated gradients during fine-tuning. In the initial phase of fine-tuning, where the parameters θ are close to θ0, the direction of τt is dominated by the average gradient of the task loss Lt over the data domain Dt, evaluated at θ0.

[∇θLt(f(x;θ0),y)]. (63) Using the chain rule, ∇θLt = ∂L

τt ∝ −E(x,y)∼D

t

∂f ·∇θf = ∂L

∂f ·J(x). The expression becomes,

t

t

∂Lt

∂f · J(x) . (64) For a well-posed learning task, the loss derivative ∂Lt

τt ∝ −Ex∼D

t

∂f

(which indicates how the loss changes with respect to the model’s output) can be assumed to be an approximately constant scalar kt across the dataset. This yields,

[J(x)]. (65) Let µJ := Ex∈D

τt ∝ −kt · Ex∈D

t

[J(x)] be the average Jacobian vector over the data domain of task t. We thus establish the first directional link,

t

Direction(τt) = Direction(µJ). (66)

Part 2: Direction of an Individual Jacobian J(x).

Next, we formalize an intuitive hypothesis. For a welldefined, non-random machine learning task, the loss function’s gradient directions for different samples within its data domain should exhibit statistical consistency, rather than pointing randomly in all directions throughout the parameter space. This consistency is fundamental to the model’s ability to learn generalizable patterns from data. Applied to our scenario, this implies that the distribution of the Jacobian vectors J(x) should not be overly dispersed.

We formalize this as the Data Consistency Assumption.

Assumption 2 (Data Consistency Assumption). For a well-defined task, the Jacobian vectors of individual samples are statistically concentrated around their mean. This means the variance of the Jacobians, σJ2 := Ex∈D

t ∥J(x) − µJ∥22 , is significantly smaller than the squared norm of their mean,

σJ2 ≪ ∥µJ∥22. (67) By Chebyshev’s inequality [34], for any constant C > 1,

we have,

E ∥J(x) − µJ∥22 C2σJ2

P ∥J(x) − µJ∥22 ≥ C2σJ2 ≤

σJ2 C2σJ2

1 C2

=

=

.

(68)

This implies that the squared Euclidean distance between the random vector J(x) and its mean µJ is bounded by C2σJ2 with a probability of at least 1 − 1/C2. In other words, for a “typical” (i.e., high-probability) sample x′, its Jacobian vector J(x) satisfies,

∥J(x′) − µJ∥2 < CσJ. (69)

Now, we bound the angle θx′ = ∠(J(x′),µJ) for such a typical sample. Consider the triangle formed by the origin and the endpoints of the vectors J(x′) and µJ. By the properties of vector geometry (related to the Law of Sines [6]), the sine of the angle θx′ is bounded by the ratio of the length of the opposing side to the length of the adjacent side,

∥J(x′) − µJ∥2 ∥J(x′)∥2

. (70)

sin(θx′) ≤

We have an upper bound for the numerator, ∥J(x)−µJ∥2 < CσJ. For the denominator, we use the reverse triangle inequality [35] to find a lower bound,

∥J(x′)∥2 = ∥µJ + (J(x′) − µJ)∥2 ≥ ∥µJ∥2 − ∥J(x′) − µJ∥2 > ∥µJ∥2 − CσJ.

(71)

Substituting these bounds, we get,

C(σJ/∥µJ∥2) 1 − C(σJ/∥µJ∥2)

CσJ ∥µJ∥2 − CσJ

. (72)

sin(θx′) <

=

Given Assumption 2, the ratio σJ/∥µJ∥2 is a value much smaller than 1. Therefore, the right-hand side of the inequality is a very small positive number. Since sin(θx′) is very small, the angle θx′ must also be very close to zero. This establishes our second directional link,

Direction(J(x)) ≈ Direction(µJ), for a typical x ∈ Dt. (73)

Combining the two parts, we have shown that for a typical sample x ∈ Dt,

Direction(J(x)) ≈ Direction(µJ) ≈ Direction(τt). (74)

This directional alignment justifies the approximation used in the main proof, allowing the angle between τj and J(x) to be replaced by the angle between τj and τt. This completes the proof.

| |
|---|

###### G.3. Proposition 3 and Proof (Norm Control)

Proposition 3. The Frobenius norm of a matrix is bounded by its deviation from orthonormality. Specifically, for a matrix W ∈ Rm×d, if its deviation from being identity is bounded by ∥W⊤W − Id∥2F ≤ ξ for some constant ξ ≥ 0, then its squared Frobenius norm is bounded by,

∥W∥2F ≤ d + dξ. (75)

Several prior works have implicitly or explicitly leveraged the norm-controlling property of orthogonality [2, 29, 49]. Here, we provide a formal and rigorous proof to establish this principle.

Proof. We aim to find the maximum possible value of ∥W∥2F under the given constraint. This can be formulated as a constrained optimization problem,

∥W∥2F.

max

(76)

W

s.t. ∥W⊤W − Id∥2F ≤ ξ

To solve this, we use the Singular Value Decomposition (SVD) [8] of W. Let W = UΣV ⊤, where U ∈ Rm×m and V ∈ Rd×d are orthogonal matrices, and Σ ∈ Rm×d is a rectangular diagonal matrix with non-negative singular values {σ1,σ2,...,σd} on its diagonal.

First, we rewrite the objective function in terms of the singular values. Because Frobenius norm is invariant under orthogonal transformations, we can get,

∥W∥2F = ∥UΣV ⊤∥2F = ∥Σ∥2F =

d

σi2. (77)

i=1

Next, we rewrite the constraint. We have W⊤W = (UΣV ⊤)⊤(UΣV ⊤) = V Σ⊤U⊤UΣV ⊤ = V (Σ⊤Σ)V ⊤. Let D = Σ⊤Σ, which is a d × d diagonal matrix with diagonal elements Dii = σi2. Again, using the orthogonal invariance of the Frobenius norm,

∥W⊤W − Id∥2F = ∥V DV ⊤ − V IdV ⊤∥2F

= ∥V (D − Id)V ⊤∥2F = ∥D − Id∥2F.

(78) Since D − Id is a diagonal matrix, its squared Frobenius norm is the sum of the squares of its diagonal elements,

∥D − Id∥2F =

d

(σi2 − 1)2. (79)

i=1

The original problem is now equivalent to a simpler optimization problem over the squared singular values. Let xi = σi2 ≥ 0,

d

max

xi.

i=1

(80)

d

(xi − 1)2 ≤ ξ

s.t.

i=1

To find the maximum, the constraint must be active, i.e., tipliers [25]. The Lagrangian is,

- d i=1(xi − 1)2 = ξ. We use the method of Lagrange mul-

d

L(x,λ) =

xi − λ

i=1

d

(xi − 1)2 − ξ . (81)

i=1

Taking the partial derivative with respect to xj and setting it to zero,

∂L ∂xj

= 1 − λ · 2(xj − 1) = 0. (82)

- 1

- 2λ

- 1

- 2λ

. (83)

xj − 1 =

=⇒ xj = 1 +

This shows that at the optimal point, all xj must be equal. Let x1 = x2 = ··· = xd = x∗.

Substituting xi = x∗ into the active constraint,

d

(x∗ − 1)2 = d(x∗ − 1)2 = ξ. (84)

i=1

Solving for x∗, we get,

ξ d

(x∗ − 1)2 =

=⇒ x∗ − 1 = ±

ξ d

, (85)

x∗ = 1 ±

ξ d

. (86)

To maximize the objective function xi = d·x∗, we must choose the positive root,

x∗max = 1 +

ξ d

. (87)

Finally, the maximum value of the objective function is,

d

x∗max = d · x∗max

max∥W∥2F =

i=1

ξ d

= d 1 +

= d + dξ.

(88)

This establishes the upper bound and completes the proof.

| |
|---|

###### G.4. Detailed Proof of Angle Control Mechanism

This section provides the full proof for Step 3 of Theorem 4, showing that our orthogonal regularization statistically promotes orthogonality between different task vectors.

G.4.1. Proposition 4 and Detailed Proof Proposition 4. Let P ∈ Rd×d be a symmetric positive semi-definite matrix. If ∥P2 −Id∥F ≤

√ξ, then ∥P −Id∥F is also bounded, and specifically satisfies,

∥P − Id∥F ≤ ∥P2 − Id∥F. (89)

Proof. Since P is symmetric, it has an eigenvalue decomposition P = UΛU⊤, where U is an orthogonal matrix and Λ is a diagonal matrix of non-negative eigenvalues λ1,...,λd ≥ 0. The Frobenius norm is invariant under orthogonal transformations. Thus, we can express the norms in terms of the eigenvalues,

∥P − Id∥2F = ∥UΛU⊤ − UIdU⊤∥2F

= ∥U(Λ − Id)U⊤∥2F

= ∥Λ − Id∥2F

d

(λi − 1)2.

=

i=1

(90)

Similarly, since P2 = (UΛU⊤)(UΛU⊤) = UΛ2U⊤,

∥P2 − Id∥2F = ∥U(Λ2 − Id)U⊤∥2F

= ∥Λ2 − Id∥2F

d

(λ2i − 1)2

=

i=1

Now we compare the terms for each eigenvalue,

(91)

###### (λ2i −1)2 = ((λi−1)(λi+1))2 = (λi−1)2·(λi+1)2 (92)

Since P is positive semi-definite, λi ≥ 0. This implies λi + 1 ≥ 1, and therefore (λi + 1)2 ≥ 1. Multiplying both sides by the non-negative quantity (λi − 1)2, we get

###### (λi − 1)2 · (λi + 1)2 ≥ (λi − 1)2 · 1. (93)

This means (λ2i − 1)2 ≥ (λi − 1)2 for all i. Summing over all i,

d

d

(λ2i − 1)2 ≥

(λi − 1)2. (94)

i=1

i=1

Substituting the norm expressions back, we have,

∥P2 − Id∥2F ≥ ∥P − Id∥2F. (95) Taking the square root of both sides yields the desired result,

###### ∥P − Id∥F ≤ ∥P2 − Id∥F. (96)

| |
|---|

- G.4.2. Proof of Angle Control Proof. Our goal is to show that enforcing an internal orthogonal structure on the update matrices ∆Wt and ∆Wj statistically drives their corresponding task vectors τt and τj towards orthogonality. That is, E[|cos∠(τt,τj)|] ≈ 0. This is equivalent to showing that the inner product ⟨τt,τj⟩ is statistically concentrated around zero.

The total inner product is the sum of layer-wise inner products,

###### ⟨vec(∆Wt(l)),vec(∆Wj(l))⟩. (97)

###### ⟨τt,τj⟩ =

l∈Layers

We analyze the inner product for a single layer, dropping the superscript (l) for clarity: ⟨vec(∆Wt),vec(∆Wj)⟩.

Our Lortho = ∥∆W⊤∆W − I∥2F encourages the resulting update matrix ∆W∗ to be approximately orthogonal, satisfying ∥(∆W∗)⊤∆W∗ − I∥2F ≤ ξ for a small ξ.

Using Polar Decomposition [13], any such matrix ∆W∗ can be uniquely decomposed into ∆W∗ = QP, where Q ∈ Vd(Rm) is a matrix with orthonormal columns (an element of the Stiefel manifold) and P = (∆W∗)⊤∆W∗ is a symmetric positive semi-definite matrix.

Substituting this relation into our regularization con-

straint, ∥(∆W∗)⊤∆W∗ −I∥2F ≤ ξ, we have ∥P2 −I∥2F ≤ ξ. By Proposition 4, this implies that P is close to the iden-

tity matrix, i.e., ∥P − I∥F is also small. We can thus write small Frobenius norm ∥E∥F.

- P = I + E, where E = P − I is an “error” matrix with a

Therefore, the update matrices for tasks t and j can be written as,

∆Wt = Qt + QtEt, (98)

∆Wj = Qj + QjEj, (99)

where Qt,Qj are matrices on Stiefel manifold, and Et,Ej are error matrices with small norms controlled by ξ.

Now, we analyze the inner product of their vectorized forms,

⟨vec(∆Wt),vec(∆Wj)⟩

= ⟨vec(Qt + QtEt),vec(Qj + QjEj)⟩.

Expanding this expression yields four terms,

(100)

###### +⟨vec(Qt),vec(QjEj)⟩

###### =⟨vec(Qt),vec(Qj)⟩

Main Term

Error Term 1

###### + ⟨vec(QtEt),vec(Qj)⟩

###### +⟨vec(QtEt),vec(QjEj)⟩

###### .

Error Term 2

Error Term 3

(101) We analyze the expectation of each term, assuming that the fine-tuning processes for distinct tasks t and j result in independently sampled matrices from the space of approximately orthogonal matrices.

Main Term. Qt and Qj are independent, random matrices from the Stiefel manifold Vd(Rm). According to Lemma 3 (proven in Appendix G.4.3), the expected value of their inner product is zero,

###### E[⟨vec(Qt),vec(Qj)⟩] = 0. (102)

And, Lemma 3 states that the probability distribution of this inner product is sharply concentrated around zero. Error Terms. We bound the magnitude of the error terms using the Cauchy-Schwarz inequality.

For Error Term 1,

###### |⟨vec(Qt),vec(QjEj)⟩| ≤ ∥vec(Qt)∥2 · ∥vec(QjEj)∥2.

(103) Since Qt has d orthonormal columns, ∥vec(Qt)∥22 = ∥Qt∥2F = d. Since Qj is an orthogonal transformation, ∥vec(QjEj)∥2 = ∥QjEj∥F = ∥Ej∥F. Thus, the term is bounded by √

d · ∥Ej∥F. As ∥Ej∥F is a small value controlled by the regularizer, this error term is negligible.

Error Term 2 is similarly bounded by √

d · ∥Et∥F and is also negligible.

Error Term 3 is bounded by ∥vec(QtEt)∥2 · ∥vec(QjEj)∥2 = ∥Et∥F · ∥Ej∥F, which is a second-order small term and even more negligible.

Since the main term has an expected value of zero and the error terms are negligible, the expected inner product for a single layer is approximately zero.

###### E[⟨vec(∆Wt),vec(∆Wj)⟩] ≈ 0. (104)

By linearity of expectation, the expected inner product of the full task vectors is also approximately zero,

E[⟨τt,τj⟩] =

l

E[⟨vec(∆Wt(l)),vec(∆Wj(l))⟩] ≈ 0.

(105)

Because the distribution of the main term at each layer is sharply peaked at zero, the distribution of the sum (the total inner product) will also be sharply peaked at zero. This implies that τt and τj are statistically very likely to be orthogonal, and thus E[|cos∠(τt,τj)|] ≈ 0. This completes the proof of the angle control mechanism.

| |
|---|

- G.4.3. Lemma 2 and Detailed Proof: Stiefel Manifold Inner Product

Lemma 3. Let A and B be two matrices independently and uniformly sampled from the Stiefel manifold Vd(Rm) [1] (the set of m × d matrices with orthonormal columns). Let Z = ⟨vec(A),vec(B)⟩. Then,

- (1) The expected value of the inner product is zero:

E[Z] = 0.

- (2)The probability distribution of Z is sharply concen-

trated around 0. Proof. Part 1: Proof of Zero Expectation.

The inner product can be written as the trace of the matrix product: Z = Tr(A⊤B). Due to the independence of A and B, the expectation of the product is the product of expectations,

E[Z] = EA[EB[Tr(A⊤B)|A]] = EA[Tr(A⊤EB[B])].

(106) Let’s compute E[B]. The distribution of B is the uniform (Haar) measure on Vd(Rm). This distribution is invariant under left-multiplication by any orthogonal matrix

- Q ∈ O(m), where O(m) is the group of m×m orthogonal matrices. This means that for any Q ∈ O(m), the random matrix QB has the same distribution as B. Therefore,

###### E[B] = E[QB] = QE[B]. (107)

This equality must hold for all Q ∈ O(m). Let’s consider a specific reflection matrix Q that negates the first coordinate,

- e.g., Q = diag(−1,1,...,1). If the first row of E[B] were a non-zero vector r, then the first row of QE[B] would be −r. The equality E[B] = QE[B] would imply r = −r, which is only possible if r = 0. This logic applies to every row by choosing appropriate reflection matrices. Therefore, the only matrix that satisfies this condition for all Q ∈ O(m) is the zero matrix.

E[B] = 0. (108) Substituting this back into the expectation for Z, we get,

E[Z] = Tr(E[A⊤] · 0) = 0. (109) This proves the first part of the lemma.

###### Part 2: Proof of Concentration around Zero.

This is a geometric argument. The vectors vec(A) and vec(B) are not arbitrary vectors in Rm×d. They are constrained to lie on the submanifold vec(Vd(Rm)). The condition A⊤A = Id imposes d(d2+1) independent constraints

on the elements of A. This means the dimension of the Stiefel manifold Vd(Rm) is dim(V ) = md − d(d2+1).

The co-dimension of this submanifold within the ambi-

ent space Rm×d is d(d2+1), which is positive for d ≥ 1. The condition for orthogonality, ⟨vec(A),vec(B)⟩ = 0, defines

a hyperplane in the product space. The probability density of Z at a value z0 is proportional to the “volume” of the surface defined by ⟨vec(A),vec(B)⟩ = z0 on the product manifold Vd(Rm) × Vd(Rm).

Intuitively, because the vectors are already living in a lower-dimensional space due to the internal orthogonality constraints, the additional constraint of being orthogonal to another such vector is “easier” to satisfy. The intersection of the hyperplane ⟨a,b⟩ = 0 with the product manifold is larger than its intersection with the product of two spheres of the same dimension. This geometric fact leads to a higher probability density at Z = 0, creating a sharp peak in the distribution. This indicates that two random matrices from Stiefel manifold are much more likely to be nearly orthogonal than two completely random unit vectors in Rm×d.

| |
|---|

##### H. Comparative Analysis with TTA

###### H.1. Theoretical Connection

In this section, we establish a theoretical connection between our proposed method (OrthoReg) and Tangent Task Arithmetic (TTA) [32]. We demonstrate that both methods, despite their different implementations, derive their effectiveness from a shared underlying mechanism: promoting orthogonality between different task vectors (i.e., ⟨τt,τj⟩ ≈ 0 for t ̸= j). This inter-task vector orthogonality is a key driver for achieving weight disentanglement.

As proven in Theorem 4 (specifically, the Angle Control mechanism in Appendix G.4.2), our OrthoReg achieves this goal explicitly. By enforcing an internal orthogonal structure on each update matrix ∆W, it statistically drives the resulting full task vectors towards orthogonality.

In contrast, TTA achieves this goal implicitly by leveraging the geometric properties of the pre-trained model’s Neural Tangent Kernel (NTK). We now provide a detailed derivation to formalize this connection.

TTA operates by performing fine-tuning in the tangent space of the pre-trained model θ0. The model’s output is approximated by its first-order Taylor expansion,

f(x;θ0 + τ) ≈ f(x;θ0) + τ⊤J(x), (110)

where J(x) = ∇θf(x;θ0) is the Jacobian. The optimization is performed over the task vector τ directly. For a task t with data {(xi,yi)}N

i=1 from domain Dt, the TTA objective can be formulated as a regularized empirical risk minimization problem, for instance, using a mean-squared error loss:

t

Nt

1 Nt

min

τt

i=1

(f(xi;θ0) + τt⊤J(xi)) − yi 22 + λ∥τt∥22.

- Table 3. Computational cost comparison on the Cars dataset using a ViT-L-14 model. The table highlights the efficiency of OrthoReg. The final column shows the Absolute Accuracy from the task addition benchmark (as seen in Table 1 of the main paper). While applying OrthoReg to Non-linear Fine-tuning (Non-lin. FT) achieves performance that is superior to Tangent Task Arithmetic (TTA) and significantly better than the baseline Non-lin. FT, this table further demonstrates its superior computational efficiency. As seen, TTA incurs substantial overhead in both training time and memory, whereas OrthoReg adds only a modest cost to the baseline. The colored cells visually emphasize the significant difference in computational cost between TTA and our proposed method.

Fine-tuning Method Total Trainable Training Peak GPU Abs. Acc.

Params (M) Params (M) Time (Min) Mem (MB) (%) Full Fine-tuning Methods

Non-lin. FT [16] (Baseline) 342.56 342.56 158.21 42589.22 84.07 TTA [32] (Linearized) 685.12 342.56 280.86 68031.34 86.19 Non-lin. FT + OrthoReg (ours) 342.56 342.56 177.04 44500.27 88.23

Parameter-Efficient Fine-tuning (Attention-Only)

ATT-FT [19] 342.56 100.66 126.28 36591.06 87.81 ATT-FT + OrthoReg (ours) 342.56 100.66 132.96 36976.50 90.41

This is a linear ridge regression problem in the variable τt. According to the Representer Theorem, the optimal solution

τt∗ must lie in the subspace spanned by the Jacobians of the training data points. Therefore, τt∗ can be expressed as a linear combination of these Jacobians,

τt∗ =

Nt

αiJ(xi), (111)

i=1

where {αi} are scalar coefficients determined by the optimization.

Now, consider the inner product between two task vec-

tors, τt∗ and τj∗, obtained by applying TTA to two different tasks, t and j,

⟨τt∗,τj∗⟩ =

Nj

Nt

αiJ(xi),

i=1

k=1

βkJ(xk) , (112)

where {xi} ⊂ Dt and {xk} ⊂ Dj. By linearity of the inner product, this becomes,

Nt

⟨τt∗,τj∗⟩ =

i=1

Nj

αiβk⟨J(xi),J(xk)⟩. (113)

k=1

The term ⟨J(xi),J(xk)⟩ is precisely the definition of the Neural Tangent Kernel (NTK) evaluated at the pair of inputs (xi,xk),

kNTK(xi,xk) = J(xi)⊤J(xk). (114)

Therefore, the inner product of the task vectors is a weighted sum of NTK values between the data points of the two tasks,

Nt

⟨τt∗,τj∗⟩ =

i=1

Nj

αiβk kNTK(xi,xk). (115)

k=1

A central empirical finding of the TTA paper [32] is that the NTK of large pre-trained models, such as CLIP, exhibits a strong localization property. This property means that the kernel function value is significant only when both inputs are from the same task domain and decays rapidly to nearzero when the inputs are from different, unrelated task domains. Formally, for distinct tasks t ̸= j,

kNTK(xi,xk) ≈ 0 for all xi ∈ Dt and xk ∈ Dj. (116)

Substituting this result into our expression for the inner product, we find that every term in the double summation is approximately zero. Consequently, the entire sum is approximately zero,

Nt

⟨τt∗,τj∗⟩ ≈

i=1

Nj

αiβk · 0 ≈ 0. (117)

k=1

This derivation shows that TTA’s effectiveness in promoting weight disentanglement stems from its ability to implicitly construct task vectors that are nearly orthogonal to each other. This orthogonality is not an explicit constraint but rather an emergent property arising from the localized structure of the pre-trained model’s NTK.

Our analysis thus unifies our method and TTA under a common principle: inter-task vector orthogonality is a core mechanism for achieving weight disentanglement. Our OrthoReg method provides a more direct, explicit to enforce this geometric property, which explains its ability to further enhance the performance of TTA and other task arithmetic methods, as demonstrated in our experiments.

###### H.2. Experimental Performance Comparison and Analysis

As established in Section 4.4 and Appendix H.1, both our OrthoReg method and Tangent Task Arithmetic (TTA) [32]

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

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

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

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

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

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
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

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

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
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
| | | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

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

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
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
| | | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

###### Figure 7. Angle distributions of weight matrix columns for all layers in ViT-B/16. Each subplot displays a histogram of the angles (in degrees) between all pairs of column vectors for a specific weight matrix. The red dashed line indicates the 90◦ point of perfect orthogonality. The plots are ordered sequentially, starting with the embedding layers, followed by the 12 transformer blocks.

Zero-shot without OrthoReg with OrthoReg

|[Figure 91]<br><br>[Figure 92]| |
|---|---|
|[Figure 93]| |

| |[Figure 94]<br><br>[Figure 95]|
|---|---|
| |[Figure 96]|

|[Figure 97]<br><br>[Figure 98]| |
|---|---|
|[Figure 99]| |

|[Figure 100]<br><br>[Figure 101]| |
|---|---|
| |[Figure 102]|

Non-linear Finetuning Tangent Task Arithmetic Attention-Only Fine-tuning LoRA-ATT

(a) ViT-B-16

Zero-shot without OrthoReg with OrthoReg

|[Figure 103]<br><br>[Figure 104]| |
|---|---|
| |[Figure 105]|

| |[Figure 106]<br><br>[Figure 107]|
|---|---|
| |[Figure 108]|

|[Figure 109]<br><br>[Figure 110]| |
|---|---|
|Vit-b-16<br><br>[Figure 111]| |

|[Figure 112]<br><br>[Figure 113]| |
|---|---|
| |[Figure 114]|

V

Non-linear Finetuning Tangent Task Arithmetic Attention-Only Fine-tuning LoRA-ATT

(b) ViT-B-32

Zero-shot without OrthoReg with OrthoReg

|[Figure 115]<br><br>[Figure 116]| |
|---|---|
| |[Figure 117]|

| |[Figure 118]<br><br>[Figure 119]|
|---|---|
| |[Figure 120]|

|Vit-b-32<br><br>[Figure 121]<br><br>[Figure 122]| |
|---|---|
| |[Figure 123]|

|[Figure 124]<br><br>[Figure 125]| |
|---|---|
| |[Figure 126]|

V

Non-linear Finetuning Tangent Task Arithmetic Attention-Only Fine-tuning LoRA-ATT

(c) ViT-L-14

- Figure 8. The accuracy of merged models across the eight benchmark tasks for different ViT architectures. Each subplot shows the performance for a specific baseline method: zero-shot (gray), the baseline’s merged model (red), and the baseline enhanced with our orthogonal regularization (blue). The rows correspond to models: (a) ViT-B-16, (b) ViT-B-32, and (c) ViT-L-14.

Vit-l-14

succeed by promoting inter-task vector orthogonality. However, we posited that OrthoReg offers a more direct, efficient, and scalable approach by avoiding the costly Jacobian computations inherent to TTA. This section provides an empirical analysis to validate this claim by comparing the computational costs, specifically training time and peak

GPU memory usage of TTA against standard fine-tuning methods enhanced with our OrthoReg regularizer.

Experimental Setup. We conduct a controlled experiment on the Cars dataset [21] using the ViT-L-14 model architecture. We measure the wall-clock training time and peak GPU memory consumption for a single fine-tuning run.

- Table 4. The minimum average Target Accuracy (Tar.Acc.) achievable while maintaining at least 90% of the zero-shot accuracy on the ImageNet control task (Con.Acc.). Our proposed orthogonal regularization (+OrthoReg) shows a consistent and significant improvement in forgetting the target task. An asterisk (*) denotes the best (lowest) target accuracy for each model architecture.

Method

ViT-B-32, 8 tasks ViT-B-16, 8 tasks ViT-L-14, 8 tasks Tar.Acc.(↓) Con.Acc. (↑) Tar.Acc.(↓) Con.Acc. (↑) Tar.Acc.(↓) Con.Acc. (↑) zero-shot 47.74 66.70 54.22 68.34 64.54 77.44

Non-linear Finetuning [16] 17.34 60.80 14.92 63.63 13.51 72.51 Non-lin. FT+OrthoReg (ours) 14.14 60.84 13.78 65.69 12.69 74.17

∆ -3.20 +0.04 -1.14 +2.06 -0.82 +1.66 Tangent Task Arithmetic [32] 7.36 62.08 6.68 65.49 5.07 72.51

TTA+OrthoReg (ours) 6.66 62.19 4.77 65.13 3.83 72.87

∆ -0.70 +0.11 -1.91 -0.36 -1.24 +0.36 Attention-Only Fine-tuning [19] 19.11 64.82 19.01 67.67 24.85 76.42

ATT-FT+OrthoReg (ours) 10.75 62.18 10.63 64.10 11.47 73.17 ∆ -8.36 -2.64 -8.38 -3.57 -13.38 -3.25

LoRA-ATT 16.85 63.23 19.44 67.28 21.23 75.41 LoRA-ATT+OrthoReg (ours) 14.59 61.68 17.25 67.08 10.10 72.19 ∆ -2.26 -1.55 -2.19 -0.20 -11.13 -3.22

- Table 5. The minimum average Target Accuracy (Tar.Acc.) achievable while maintaining at least 80% of the zero-shot accuracy on the ImageNet control task (Con.Acc.). Our proposed orthogonal regularization (+OrthoReg) shows a consistent and significant improvement in forgetting the target task. An asterisk (*) denotes the best (lowest) target accuracy for each model architecture.

ViT-B-32, 8 tasks ViT-B-16, 8 tasks ViT-L-14, 8 tasks Tar.Acc.(↓) Con.Acc. (↑) Tar.Acc.(↓) Con.Acc. (↑) Tar.Acc.(↓) Con.Acc. (↑) zero-shot 47.74 66.70 54.22 68.34 64.54 77.44

Method

Non-linear Finetuning [16] 11.97 54.43 11.65 59.20 12.67 70.59 Non-lin. FT+OrthoReg (ours) 10.24 57.06 10.40 61.39 9.30 72.33

∆ -1.73 +2.63 -1.25 +2.19 -3.37 +1.74 Tangent Task Arithmetic [32] 5.70 60.76 5.61 64.53 2.84 70.81

TTA+OrthoReg (ours) 3.26 59.26 2.10 62.61 1.86 70.23 ∆ -2.44 -1.50 -3.51 -1.92 -0.98 -0.58

Attention-Only Fine-tuning [19] 19.11 64.82 19.01 67.67 24.85 76.42 ATT-FT+OrthoReg (ours) 7.23 58.38 8.08 61.21 8.12 68.46 ∆ -11.88 -6.44 -10.93 -6.46 -16.73 -7.96

LoRA-ATT 15.58 62.4 15.83 62.40 21.23 75.41 LoRA-ATT+OrthoReg (ours) 11.00 58.47 9.19 60.41 7.68 69.83 ∆ -4.58 -3.93 -6.64 -1.99 -13.55 -5.58

Results and Analysis. The results, summarized in Table 3 are organized to highlight the efficiency trade-offs between different full-parameter fine-tuning strategies and their parameter-efficient counterparts.

The primary comparison focuses on the full fine-tuning methods. Standard Non-linear Fine-tuning (Non-lin. FT) serves as our baseline, completing training in 158.21 minutes and consuming 42589.22 MB of peak GPU memory. In stark contrast, TTA [32], which operates on a linearized model, is substantially more resource-intensive. It requires 280.86 minutes (a 77.5% increase in time) and 68031.34 MB of memory (a 59.7% increase), confirming that its re-

liance on Jacobian computations imposes a significant computational burden.

Our proposed OrthoReg, when applied to Non-lin. FT, introduces only a moderate overhead for its regularization calculations, resulting in a total cost of 177.04 minutes and 44500.27 MB of memory during the training phase. Crucially, this is significantly more efficient than TTA in both time and memory, while achieving superior or comparable task-addition performance as shown in the main text and the last column of Table 3 (e.g., for ViT-L-14, Non-lin. FT + OrthoReg achieves 88.23% Abs.Acc. vs. TTA’s 86.19%). This demonstrates that OrthoReg provides a more efficient

path to enforcing the properties that benefit task arithmetic.

This efficiency advantage is also evident in the parameter-efficient setting. As shown in the lower section of Table 3, applying OrthoReg to ATT-FT baseline results in only a minimal increase in computational cost. The training time rises modestly from 126.28 to 132.96 minutes, and peak memory usage increases marginally from 36591.06 MB to 36976.50 MB. However, the performance increases considerably from 87.81% to 90.41%. This demonstrates that the substantial performance improvements gained from OrthoReg come at a very low computational price, further highlighting its practicality.

In conclusion, these experiments provide strong empirical evidence that OrthoReg achieves the goal of promoting task vector orthogonality more efficiently than TTA. This efficiency, combined with the superior performance demonstrated in our main results, establishes OrthoReg as a more effective and accessible tool for reliable task arithmetic.

##### I. Experiments Details

The Normalized Accuracy (Norm.Acc.) metric evaluates the performance of the merged multi-task model (θMT) relative to individually fine-tuned single-task models (θt∗). It is defined as the average of the performance ratios across all T tasks. A score of 100% indicates that the merged model performs, on average, on par with the individual specialist models, suggesting a successful composition with minimal negative interference.

The formula is given by,

Norm.Acc. =

T

acc(θMT,Dt) acc(θ∗

1 T

t ,Dt) × 100%, (118)

t=1

where T is the total number of tasks being merged, acc(θMT,Dt) is the accuracy of the merged model on test set for task t and acc(θt∗,Dt) is the accuracy of the model fine-tuned only on task t, evaluated on its own test set.

This definition is consistent with the evaluation protocol established in prior work [16, 19, 32].

##### J. More Experimental Results J.1. Detailed Visualization of Orthogonality

To provide comprehensive empirical support for the claim made in Section 4.2.3, this part presents a detailed visualization of the weight vector angle distributions for all linear layers within the pre-trained CLIP ViT-B/16 model. Figure 7 displays the histograms for each weight matrix.

As illustrated in Figure 7, a clear and consistent pattern emerges across the model’s layers. We observe two distinct behaviors. (1) Embedding Layers. The first two subplots correspond to the patch embedding and pos embedding

layers. These layers show broader, more Gaussian-like distributions, which is understandable given their unique function of mapping raw inputs into the initial embedding space. As our analysis primarily concerns the transformation dynamics within the main model body, these layers are not the central focus of our study. (2) Transformer Blocks. In stark contrast, nearly all subsequent weight matrices, which constitute the core computational machinery of the model, including the query, key, value (QKV) projections, attention output projections (proj), and MLP layers within all 12 transformer blocks, exhibit angle distributions that are sharply and narrowly peaked at 90 degrees.

This detailed, per-layer visualization provides robust evidence that near-orthogonality is not an isolated occurrence but a pervasive geometric property of the pre-trained model’s core processing blocks.

###### J.2. Detailed Per-Task Performance Visualization

This section supplements the analysis in Section 5.2 by providing the comprehensive per-task performance radar charts for all evaluated architectures: ViT-L-14, ViT-B-16, and ViT-B-32. The results shown in Figure 8 reinforce and expand upon the findings presented in the main body. We consistently observe that applying OrthoReg (the blue area) leads to a larger performance footprint compared to the baselines (the red area) across the vast majority of tasks, methods, and architectures. This further corroborates our claim that OrthoReg is a model-agnostic regularizer that effectively mitigates task interference, leading to broad performance gains in multi-task scenarios.

###### J.3. Details About Task Negation

In this section, we provide additional details for the task negation experiments discussed in Section 5.3. When the accuracy requirement on the control task is further relaxed, such as to 90% (see Table 4) or 80% (see Table 5), the effect of task negation becomes progressively stronger, resulting in lower accuracy on the target task. Moreover, our OrthoReg regularizer can further enhance the negation effect while still meeting the control-task accuracy threshold. In some cases, it even improves control-task accuracy while reducing target-task accuracy. These results demonstrate that our method effectively disentangles task-specific feature information, substantially reducing undesired interference with non-target tasks during the task negation process.

###### J.4. Visualization of Task Vector Similarity

To supplement the analysis in Section 5.4, this section provides additional task vector similarity heatmaps. These figures (Figure 9, Figure 10, Figure 11) illustrate the effect of OrthoReg across different baseline methods and model architectures, consistently demonstrating that our method produces more orthogonal task vectors.

(e) Non-lin. FT+OrthoReg (f) TTA+OrthoReg (g) ATT-FT+OrthoReg (h) LoRA-ATT+OrthoReg

- Figure 9. Pairwise cosine similarity heatmaps of task vectors for ViT-B-16 across different methods. The top row shows the baseline methods, where significant off-diagonal correlation (brighter colors) is visible. The bottom row shows the same methods with our OrthoReg regularizer. The consistently darker off-diagonal values in the bottom row provide strong empirical validation that OrthoReg successfully produces more orthogonal task vectors, mitigating a key source of task interference.

(a) Non-lin. FT (b) TTA (c) ATT-FT (d) LoRA-ATT

(e) Non-lin. FT+OrthoReg (f) TTA+OrthoReg (g) ATT-FT+OrthoReg (h) LoRA-ATT+OrthoReg

- Figure 10. Pairwise cosine similarity heatmaps of task vectors for VIT-B-32 across different methods. The top row shows the baseline methods, where significant off-diagonal correlation (brighter colors) is visible. The bottom row shows the same methods with our OrthoReg regularizer. The consistently darker off-diagonal values in the bottom row provide strong empirical validation that OrthoReg successfully produces more orthogonal task vectors, mitigating a key source of task interference.

###### J.5. Detailed Ablation Study on LoRA Components

This part provides additional details and results to supplement the LoRA ablation study presented in Section 5.1.

###### J.5.1. Rationale for Module Selection

The selection of different module subsets for our LoRAbased ablation study was designed to systematically probe

(e) Non-lin. FT+OrthoReg (f) TTA+OrthoReg (g) ATT-FT+OrthoReg (h) LoRA-ATT+OrthoReg

- Figure 11. Pairwise cosine similarity heatmaps of task vectors for ViT-L-14 across different methods. The top row shows the baseline methods, where significant off-diagonal correlation (brighter colors) is visible. The bottom row shows the same methods with our OrthoReg regularizer. The consistently darker off-diagonal values in the bottom row provide strong empirical validation that OrthoReg successfully produces more orthogonal task vectors, mitigating a key source of task interference.

- Table 6. Performance comparison of different LoRA module configurations with and without orthogonality regularization. The last row under each module shows the improvement (∆) from OrthoReg.

Finetuning ViT-B-32, 8 tasks ViT-B-16, 8 tasks ViT-L-14, 8 tasks Mode Abs.Acc.(↑) Norm.Acc.(↑) Abs.Acc.(↑) Norm.Acc.(↑) Abs.Acc.(↑) Norm.Acc.(↑)

LoRA Modules

LoRA 73.03 81.89 75.18 81.83 85.44 90.98 +OrthoReg 74.71 84.31 78.07 85.23 87.69 93.67

qkvofp (All)

###### ∆ +1.68 +2.42 +2.89 +3.40 +2.25 +2.69

LoRA 73.95 84.19 76.31 84.04 87.13 93.49 +OrthoReg 76.20 86.55 80.48 91.97 89.14 95.49

qkvo– (Attn All)

###### ∆ +2.25 +2.36 +4.17 +7.93 +2.01 +2.00

LoRA 70.14 80.98 74.69 82.82 85.03 91.67 +OrthoReg 73.68 84.40 78.10 86.27 87.56 93.97

qkv— (Q,K,V)

###### ∆ +3.54 +3.42 +3.41 +3.45 +2.53 +2.30

LoRA 69.25 80.30 75.15 83.35 84.39 91.11 +OrthoReg 72.71 83.77 77.03 85.37 86.58 93.29 ∆ +3.46 +3.47 +1.88 +2.02 +2.19 +2.18

q-v— (Q,V only)

LoRA 69.19 78.01 71.24 78.02 81.98 87.78 +OrthoReg 68.92 77.77 72.05 78.72 82.80 88.13 ∆ -0.27 -0.24 +0.81 +0.70 +0.82 +0.35

—-fp (MLP only)

the effect of OrthoReg on distinct functional components of the Vision Transformer.

• All Tunable Layers. qkvofp: This represents the most comprehensive PEFT approach, applying LoRA to all available linear layers (attention and MLP). It serves as a baseline to evaluate the effect of tuning the entire model in a parameter-efficient manner.

- • MLP Layers Only. —fp: This configuration isolates the FFN or MLP blocks. By tuning only these layers, we can assess their specific contribution to task adaptation and how OrthoReg influences them in isolation.
- • Attention Subsets. qkvo–, qkv—, and q-v—: These configurations focus on the multi-head self-attention mechanism, which is widely considered crucial for capturing

task-specific patterns.

- – qkvo– tunes all four projection matrices (query, key, value, and output), representing a full intervention within the attention block.
- – qkv— omits the output projection, allowing us to gauge its importance.
- – q-v— is a particularly important configuration. Prior work [52] has identified that fine-tuning only the query and value matrices can be a highly effective and parameter-efficient strategy.

By comparing these configurations, we can draw nuanced conclusions about where task-specific knowledge is stored and how promoting orthogonality in different components contributes to the final performance of task arithmetic.

###### J.5.2. results

Table 6 summarizes the effect of applying OrthoReg across different LoRA module configurations. Overall, OrthoReg consistently improves performance in all settings except the MLP-only configuration. The largest gains appear in attention-related modules , such as qkvo– , with improvements up to +4.17 points on ViT-B-16. This aligns with the common understanding that attention layers carry most of the task-specific information, and orthogonalizing their updates most effectively reduces feature entanglement.

Full-layer tuning (qkvofp) also benefits substantially from OrthoReg, indicating that larger tunable subspaces allow orthogonality constraints to better isolate task-relevant directions. The Q,V-only configuration (q-v—), previously identified as an efficient tuning strategy, also shows stable improvements when combined with OrthoReg.

The only exception is the MLP-only setup, where OrthoReg slightly reduces accuracy on smaller models. This suggests that MLP layers contribute less task-specific variation, and enforcing orthogonality may occasionally restrict useful shared representations.

Overall, the results confirm that OrthoReg most strongly enhances the components responsible for taskdiscriminative behavior, leading to more accurate task vectors and more reliable task arithmetic.

