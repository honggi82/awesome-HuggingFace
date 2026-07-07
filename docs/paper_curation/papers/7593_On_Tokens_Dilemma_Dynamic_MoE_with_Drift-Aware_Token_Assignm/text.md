# arXiv:2603.27481v1[cs.LG]29Mar2026

## On Token’s Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models

Chongyang Zhao Mingsong Li Haodong Lu Dong Gong* University of New South Wales (UNSW Sydney)

{chongyang.zhao, dong.gong}@unsw.edu.au

#### Abstract

Multimodal Continual Instruction Tuning aims to continually enhance Large Vision Language Models (LVLMs) by learning from new data without forgetting previously acquired knowledge. Mixture of Experts (MoE) architectures naturally facilitate this by incrementally adding new experts and expanding routers while keeping the existing ones frozen. However, despite expert isolation, MoE-based continual learners still suffer from forgetting due to routingdrift: old-task tokens become mistakenly attracted to newly added experts, degrading performance on prior tasks. We analyze the failure mode at the token level and reveal the token’s dilemma: ambiguous and old tokens in new-task data offer minimal learning benefit yet induce forgetting when routed to new experts, due to their ambiguous routing assignment during training. Motivated by this, we propose LLaVA-DyMoE, a dynamic MoE framework that incrementally expands the MoE with drift-aware token assignment. We characterize token types via their routing score distributions and apply targeted regularization. Specifically, a token-level assignment guidance steers ambiguous and old tokens away from new experts to preserve established routing patterns and alleviate routing-drift, while complementary routing score regularizations enforce expert-group separation and promote new-expert specialization. Extensive experiments demonstrate that our LLaVA-DyMoE effectively mitigates routing-drift-induced forgetting, achieving over a 7% gain in mean final accuracy and a 12% reduction in forgetting compared to baselines. The project page is zhaoc5.github.io/DyMoE.

#### 1. Introduction

Large Vision Language Models (LVLMs) [3, 12, 38, 64] have recently achieved remarkable performance across a wide range of vision-language tasks [2, 37, 44] by extending Large Language Models (LLMs) [28, 59, 60] to pro-

*D. Gong is the corresponding author.

###### MFN

VQAOCR

SQA

LoRA

MoELoRA

VQAText

VQAv2

EWC

LwF

IncLoRA

O-LoRA

IncMoELoRA

LLaVA-DyMoE (Ours)

ImgNet

REF

GQA VizWiz

Figure 1. Performance comparison on the CoIN benchmark, showing per-task final accuracy and mean final accuracy (MFN).

cess multimodal information. Central to their success is a two-phase development pipeline: pre-training for visionlanguage alignment, followed by instruction tuning to adapt the model to specific domains and tasks. While these models are trained on fixed datasets and remain largely static, new instruction-following requirements often arise dynamically in real-world applications. This motivates continual learning capabilities that allow the model to assimilate new knowledge while preserving performance on previously learned tasks, overcoming catastrophic forgetting [17, 30, 36].

As naively retraining on a combined set of old and new data is resource-intensive, Multimodal Continual Instruction Tuning (MCIT) [5, 7, 8, 23, 73] has emerged to address this need, aiming to incrementally instruction-tune LVLMs on new tasks while maintaining proficiency on previously learned ones. Common strategies include regularizationbased methods [8, 23, 31, 46], which impose parameter constraints to prevent forgetting, and rehearsal-based methods [32, 43, 73], which rely on replaying old data. However, these approaches often introduce non-trivial computational overhead or storage constraints. An attractive alternative is to isolate task-specific parameters via parameter-efficient tuning (PEFT) approaches [8, 16, 24, 75]. Among these, the Mixture of Experts (MoE) paradigm [16, 65, 69, 73, 74] has become a prevalent solution owing to its dynamic modular architecture, superior scalability, and inference efficiency.

This modular structure inherently facilitates flexible expert allocation and parameter isolation across tasks, which is crucial for mitigating catastrophic forgetting.

Despite these advantages, existing MoE-based MCIT approaches still exhibit significant forgetting. Training a fixed-size MoE without parameter isolation across shared experts and routers leads to inter-task interference and degraded knowledge retention [7]. Some works [20, 25, 73, 81] address this by incrementally expanding model experts while freezing previous ones, and introducing taskspecific routers to identify tasks and assign experts accordingly. However, reliable task identification may require heavy computation and can become unreliable when tasks are diverse and complex. Moreover, task-level expert assignment reduces the combinatorial flexibility of MoE, sacrificing its inherent token-level routing adaptability.

In this paper, we introduce LLaVA-DyMoE, a Dynamic MoE framework with Drift-Aware Token Assignment for continual learning of LVLMs. Unlike prior works that bypass routing instability via task-specific routing [20, 25, 73], we focus on directly addressing the underlying token-level cause of forgetting during dynamic MoE expansion. Even with old experts and their router parameters frozen, training the newly added components on new-task data still causes forgetting: the updated routing parameters distort the assignment of old-task tokens to their established experts. This distortion constitutes routing-drift, a corruption of the router’s learned policy for old tasks that drives catastrophic forgetting at the token level. We analyze how routingdrift arises during training of newly added components and reveal that not all tokens in the new-task data contribute equally (Sec. 3.3 and Fig. 2). Beyond new tokens that carry genuinely novel patterns, we identify two types that pose a forgetting risk: ambiguous tokens, which exhibit similar routing affinity for both old and new expert groups; and old tokens, whose patterns closely resemble old tasks yet receive non-negligible new-expert weight from the underoptimized router. Both types offer minimal benefit for newtask learning, yet when routed to new experts, they inadvertently train the new router to attract old-task patternscausing old-task tokens to be mis-routed at inference time and inducing forgetting. This is the token’s dilemma: minimal learning value, yet a direct forgetting cost when left unguided; compounded by the inherent ambiguity of their routing and expert assignment.

Motivated by this analysis, LLaVA-DyMoE mitigates forgetting in dynamic MoE expansion through a twofold regularization comprising Token Assignment Guidance (TAG) and Routing Score Regularization (RSR). TAG identifies token types from their routing scores and guides their assignment by adjusting routing scores during training, directly tackling the tokens’ dilemma and steering ambiguous tokens away from new experts. As a complemen-

tary soft regularization, RSR encourages exclusive tokento-group routing and promotes new-expert specialization on genuinely new-task tokens. Extensive experiments on the CoIN benchmark [7] across eight VQA-based tasks demonstrate the effectiveness of LLaVA-DyMoE, achieving over a 7% gain in MFN and a 12% reduction in forgetting compared to baseline methods.

Moreover, LLaVA-DyMoE is orthogonal to and compatible with existing MCIT paradigms, including databased methods [8, 51, 73] and task-specific routing approaches [20, 72, 73, 81], and can be combined with them for further performance gains.

Our main contributions are summarized as follows:

- • We identify the token-level cause of routing-drift: the token’s dilemma. Through controlled analysis, we show that ambiguous tokens and old tokens in new-task data offer minimal new-task benefit yet induce forgetting when routed to new experts. Ambiguous tokens are especially challenging, as their ambiguous affinity makes them difficult to identify and prone to unstable routing. (Sec. 3.3)
- • Motivated by this, we introduce LLaVA-DyMoE, a twofold regularization framework. It comprises a Token Assignment Guidance (TAG) mechanism that identifies and redirects ambiguous tokens away from new experts, and a Routing Score Regularization (RSR) that encourages exclusive token-to-group routing and promotes new-expert specialization. (Sec. 3.4)
- • Extensive experiments demonstrate that our method significantly outperforms baseline methods, achieving a superior balance between knowledge retention and new-task acquisition. (Sec. 4)

#### 2. Related Work

Continual Learning (CL) investigates methods for training models on non-stationary data distributions, typically presented as a sequence of tasks, with the primary goal of overcoming catastrophic forgetting [14, 30, 49, 62, 71, 80]. CL methods can be broadly categorized by their core strategy to mitigate catastrophic forgetting. Rehearsal-based methods [4, 6, 39, 49, 50, 52, 57] store and generate a small subset of previous samples or features during training on new tasks, thereby approximating the data distribution of the past. Regularization-based methods [1, 27, 30, 36, 45, 76, 78, 79] mitigate catastrophic forgetting by penalizing updates to parameters deemed critical for performance on previous tasks. Architecture-based methods [35, 40, 41, 55, 61, 68, 70–72] allocate new parameters for each task, either by physically expanding the network or by functionally isolating parameter subsets via masking.

Continual Learning for LVLMs and LLMs. Continually expanding the capabilities of LLMs [28, 59, 60] and LVLMs [3, 12, 38, 64] presents unique challenges, as the immense computational cost of retraining makes continual

instruction tuning a necessity. In the vision-language domain, recent efforts [5, 7, 18, 20, 23, 25, 46, 65, 77] focus on continual instruction-tuning LVLMs with sequential tasks, avoiding the expensive process of retraining from scratch. MoELoRA [7] proposes the CoIN benchmark and adopts the framework of Mixture of Experts (MoE) with LoRA experts. SEFE [8] incrementally learns new LoRA matrices and regularizes key parameter updates to retain prior knowledge. ProgLoRA [73] proposes a progressive LoRA pool that mitigates task interference by isolating knowledge in separate LoRA blocks. In the language domain, similar efforts have been applied to either regularize learning [48, 63] or expand the capacity of the model [48].

Mixture of Experts (MoE) with LoRA. The MoE paradigm enhances model capacity by replacing the Transformer’s dense feed-forward layer with multiple expert subnetworks and a routing network [13, 16, 21, 33, 56, 69]. This framework dynamically routes each input to a sparse subset of experts, employing auxiliary loadbalancing losses [16] to ensure balanced expert utilization. This paradigm has been adopted in conjunction with LoRA [24] for standard fine-tuning [10, 15, 34] and for continual learning [7, 72], where low-rank adapters are treated as experts. Our formulation adopts this MoE with LoRA paradigm, where we add new LoRA experts for each task to expand the knowledge base of the foundation model.

#### 3. Methodology: LLaVA-DyMoE with DriftAware Token Assignment

##### 3.1. Problem Setup

Continual Learning aims to enable models to continually acquire new knowledge without catastrophic forgetting. Within the broader CL paradigm, Multimodal Continual Instruction Tuning (MCIT) enables Large Vision-Language Models (LVLMs) to incrementally adapt to new tasks and maintain strong performance on previously learned tasks, without full retraining. Let {D1,...,Dt,...,DT} denote the training data of a sequence of T tasks arriving as a stream. The dataset Dt = {X}S

t for the t-th task consists of St samples. Each sample is a multimodal instruction-response triplet X = (xv,xq,xa). Here, xv, xq, and xa denote the image, instruction, and answer tokens, respectively. We focus on the MCIT setting [7] based on LLaVA [38].

##### 3.2. Dynamic MoE Layers

MoE Layers with LoRA. Given a pre-trained LLaVA, learning on new instruction tuning tasks is achieved through fine-tuning with LoRA [24]. A LoRA module parameterizes a low-rank update to a pre-trained weight matrix W0l,m ∈ Rd

out×din (at layer l, module m in the Transformer of LLaVA) by introducing two factors Bl,m ∈ Rd

out×r and Al,m ∈ Rr×d

in, such that ∆Wl,m = Bl,mAl,m, where

r ≪ min(din,dout). The updated weight matrix is then defined as: Wl,m = W0l,m + ∆Wl,m = W0l,m + Bl,mAl,m.

Instead of relying on a single continually trained LoRA adapter or merging task-specific adapters into the backbone, we develop an MoE architecture with LoRA modules as experts to augment each module with weight matrix W0l,m in the pre-trained LLM of LLaVA. An MoE layer is composed of multiple experts {el,mi ()}Ni=1 and a router Rl,m() : Rd

→ RN to assign each input token representation to specific experts. Each expert ei() is a LoRA module parameterized by (Al,mi ,Bl,mi ). Given an input multimodal token’s representation hl,m ∈ Rd

in

in, the output hl,mout is computed as:

N i=1

wiBl,mi Al,mi hl,m, wi = wi′/

hl,mout = W0l,mhl,m +

N i=1

wi′, wi′ = exp(si)[i ∈ TopK K(s)],

(1)

where Bl,mi Al,mi hl,m = ei(hl,m), s = Rl,m(hl,m) is the logits vector produced by the router, TopKK(s) denotes the set comprising the indices of the K highest affinity scores among all N experts, () is the indicator function, s i and wi are the i-th elements of s and w, respectively. The router assigns the token to the corresponding K experts with topK highest scores. The resulting routing weight w is sparse, indicating that only K out of N gate values are nonzero. This sparsity property encourages the tokens to be assigned to specialized experts at each layer.

Dynamic MoE with incrementally added experts. We implement dynamic MoE (DyMoE) with LoRA experts, incrementally adding new experts and expanding the router as each new task arrives. In MCIT, when the t-th task (t > 1) arrives, we assume existing experts Et−1 with a router producing scores wt−1,st−1 ∈ R|E

t−1|, indexed by St−1. For a new task t, we add Nt new experts Et,new = {el,mi }N

t

i=1 and expand the router to produce Nt new output routing scores wt,new,st,new ∈ RN

t. All old existing parameters are frozen; only the newly added experts Et,new and their associated router parameters are trained, while input tokens from task t can be routed to both old frozen experts Et−1 and new trainable experts Et,new. After adding and training new experts, the expert set and router output are expanded: Et = Et−1 ∪ Et,new, wt = [wt−1;wt,new], and st = [st−1;st,new], and the index set is updated as St = St−1 ∪ St,new.

Forgetting as routing-drift in DyMoE. During training, newly added experts and their router parameters are updated while existing experts remain frozen, allowing new-task tokens to reuse previously learned knowledge and keeping experts isolated across tasks. However, despite this isolation, DyMoE still exhibits catastrophic forgetting in MCIT due to routing-drift. After updating the new router parameters, old-task tokens may be mis-routed to newly added experts that were never trained on them, resulting in performance

82

54

Task2(TextVQA)Accuracy(%)

Task1(SQA)Accuracy(%)

80

52

78

50

76

74

48

72

46

70

- Task 1 (SQA)

- Task 2 (TextVQA)

Only Retain New Tokens

44

Default Training

68

50 100 150 200 250

Training Step

(a) Only retain new tokens

82

54

Task2(TextVQA)Accuracy(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Task1(SQA)Accuracy(%)

80

| |
|---|

52

78

76

50

74

48

72

70

46

- Task 1 (SQA)

- Task 2 (TextVQA)

Mask Out Old Tokens

| |
|---|

Default Training

68

| |
|---|

50 100 150 200 250

Training Step

(b) Mask out old tokens

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

80

Task2(TextVQA)Accuracy(%)

70

Task1(SQA)Accuracy(%)

60

50

- Task 1 (SQA)

- Task 2 (TextVQA)

40

Only Retain Ambiguous Tokens

30

20

10

0

−1

50 100 150 200 250

Training Step

(c) Only retain ambiguous tokens

- Figure 2. Routing-drift analysis in a controlled two-task learning experiment. After learning on the 1st task (SQA), we conduct the 2nd task (TextVQA) training using the baseline (default training) and three different token masking strategies based on token type. Throughout the training stages, we evaluate forgetting (decrease in task 1 accuracy) and new-task learning (improvement of task 2 accuracy). Polynomial regression-fitted curves are used for better visualization and readability of performance changes. The baseline (default training) refers to a scenario where each input token is assigned to all experts (including old frozen and new learnable ones). We then examine the role of each token group based on its routing score: (a) We only retain the contribution of tokens that have high affinity to the new expert group (termed “new tokens”). (b) We mask out the contribution of tokens with a high affinity to the old expert group (termed “old tokens”). (c) We only retain the contribution of tokens with a small affinity difference between the old and new expert group (termed “ambiguous tokens”).

degradation on old tasks, i.e., forgetting.

##### 3.3. Token’s Dilemma: Analyses on Routing-drift Associated with Token Assignment

Although many CL and MCIT methods with incrementally added network components [9, 61, 73, 77, 81] attempt to handle or bypass the forgetting caused by routingdrift, they typically rely on auxiliary mechanisms, such as task-specific router predictors or auxiliary regularizers. Instead, we aim to investigate and tackle the inherent cause of routing-drift in the dynamic MoE expansion process. We analyze how routing-drift is caused at the token level when only newly added parameters are updated on new-task data while old parameters remain frozen. Even when trained only on new-task tokens, the newly added router parameters can still attract old-task tokens (i.e., high routing scores st,new) and route them to new experts. Since MoE routers operate on and are also trained by individual token-expert assignments, we ask: how does token assignment during new-task training lead to routing confusion?

We investigate how different tokens from new-task data are assigned to experts during training and how this finally influences routing and performance on both tasks in testing. To closely examine the token–router dynamics when training newly added experts and router parameters, we conduct a controlled two-task experiment at the second incremental step (Fig. 2). When the second (i.e., new) task arrives, only the newly added LoRA experts and router are updated in a default way (i.e., basic IncMoELoRA; Sec. 3.2), and we measure accuracy on both new and old tasks as indicators of new-knowledge acquisition and forgetting. During training, each token is assigned to all experts (including old frozen ones and new learnable ones) according to routing scores s = [sold,snew]. Since routing-drift occurs with old-task tokens attracted by newly trained components, even when training only on new-task tokens, we hypothesize that dif-

ferent tokens exhibit varying degrees of new patterns—not all new-task tokens carry genuinely new patterns—and that freely assigning all of them to both old and new experts during training causes routing confusion between tasks. After investigating the token-expert assignment pattern, we dynamically categorize new-task tokens into three groups based on the relative dominance of snew vs. sold in their routing scores s = [sold,snew]: new, old, and ambiguous tokens. We analyze how each token type influences forgetting and new-task learning (Fig. 2), yielding three key observations.

- Observation 1: New tokens (with high affinity to the new expert group) primarily drive new-task knowledge acquisition and cause less forgetting. As shown in Fig. 2(a), training only on new tokens yields strong new-task performance with minimal forgetting, as these tokens carry patterns distinct from old tasks and are naturally routed to newly added experts, leaving the old router policy uncorrupted.
- Observation 2: Old tokens contribute less to new-task learning. Masking them from accessing newly added parameters yields similar new-task performance and forgetting as the baseline (Fig. 2(b)), suggesting they are best handled by old frozen experts and do not need to contribute to new-task learning. When assigned small but non-negligible weight toward new experts (by an under-optimized router), this inadvertently biases the new router toward old-task patterns, causing routing-drift despite limited learning value.
- Observation 3: Ambiguous tokens offer minimal new-task learning benefit while posing a direct forgetting risk. Identified by their small affinity difference between old and new expert groups, these tokens capture ambiguous patterns across tasks. Their ambiguity makes them particularly difficult to handle correctly. As shown in Fig. 2(c), training solely on these tokens neither improves new-task acquisition nor preserves old-task performance, confirming their minimal learning value and direct forgetting risk.

These controlled experiments reveal how different token

LLaVA-DyMoE

× 𝐿 × 𝐿

Task t Expansion

[Figure 1]

Language Model

Transformer Block

MLP

old group new group

✅

[Figure 2]

…

[Figure 3]

[Figure 4]

[Figure 5]

🔥

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

RMS Norm

VL Adapter

…

…

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Pretrained Weight

Text Tokenizer

Vision Encoder

[Figure 17]

❌

Multi-head Attention

? ?

…

|Driving well takes practice, following trait inherited or acquired? Darnel can drive a car. A. acquired B. inherited|
|---|

|What is the object in the image?<br><br>[Figure 18]|
|---|

[Figure 19]

Router 🔥

[Figure 20]

RMS Norm

[Figure 21]

| | | |
|---|---|---|
| | | |

Task 1 Task t

…

…

- Figure 3. Overview of our LLaVA-DyMoE method, which applies a dynamic MoE with LoRA experts to each layer of the language backbone in LLaVA. It is a two-fold regularization approach designed to resolve routing-drift-induced forgetting, based on our analysis of different token types in Sec. 3.3. The right panel illustrates this high-level approach: as new tasks (Task t) arrive, the router and experts expand, creating a frozen “old group” and a trainable “new group”. Our Token Assignment Guidance (TAG) prevents routing-drift (red dashed arrow) by directing tokens to appropriate expert–router groups, complemented by our Routing Score Regularization (RSR) that encourage exclusive token-to-group routing and new-expert specialization. Our method regularizes the router behavior during training and imposes no constraints at inference, allowing seamless combination with other continual learning methods.

###### 3.4.1. Token Assignment Guidance (TAG)

types affect new-task learning and contribute to routingdrift, exposing the link between the plasticity–stability dilemma in CL and the token’s dilemma: the inherent assignment ambiguity and trade-off between learning new tasks and inducing routing-drift. Building on this insight, we design regularization strategies for DyMoE that identify token types and guide their assignment during training, enabling us to leverage all tokens while mitigating routingdrift-induced forgetting.

During training, router behavior in MoE is iteratively updated through backpropagation, and the token assignments made by an under-optimized router directly affect the subsequent learning of both experts and routers, influencing whether the model develops the desired expert specializations and routing patterns [66, 67, 69]. As routers and experts are jointly trained, under-optimized routing weights may generate misleading gradients through token-expert assignment, contaminating the training of both. Our analysis in Sec. 3.3 shows that different tokens influence training differently w.r.t. old and new expert groups: new tokens carry clear new patterns and route naturally to new experts; old tokens gravitate toward old experts but their residual affinity for new experts should be suppressed to prevent routing corruption; ambiguous tokens exhibit uncertain routing between both groups and require careful handling. TAG dynamically identifies token types via their routing score ambiguity w.r.t. old and new expert groups and guides their assignment during training to mitigate routing-drift.

##### 3.4. Drift-Aware Token Assignment Regularization for Alleviating Forgetting

To resolve routing-drift-induced forgetting in DyMoE for MCIT, we design a two-fold regularization approach in our proposed LLaVA-DyMoE. As analyzed in Sec. 3.3, different new-task tokens from new component training affect new-task learning and old-task forgetting differently. Motivated by this, our proposed regularization guides token routing between old frozen and newly added experts during training to mitigate routing-drift, which relies solely on tokens’ routing scores without additional assumptions. The regularization operates on intermediate token representations across all MoE layers. The two-fold regularization comprises Token Assignment Guidance (TAG) and Routing Score Regularization (RSR). TAG identifies token types from their routing scores and guides their assignment by adjusting routing scores during training, shaping the router to avoid drift. It directly tackles the tokens’ dilemma and specifically handles ambiguous tokens. As a complementary soft regularization, RSR directly regularizes the routing score values to enforce discrepancy and specialty.

Token assignment ambiguity w.r.t. router-expert group. In LLaVA-DyMoE, when a new task t arrives in MCIT, we add new experts together with their router parameters, resulting in two router–expert groups: the old group St−1 (also denoted St,old) and the new group St,new. For the representation of a given token at any layer, we denote the router logits over all experts as st = [st−1;st,new] ∈ R|E

t|. We extract the confidence score from each expert group by taking the maximum logit within that group:

cold = max(st−1), cnew = max(st,new). (2) We quantify token assignment ambiguity by the relative dif-

ference between the two group-wise confidence scores:

Drel = |cnew − cold| max(|cnew|,|cold|) + ϵ

, (3)

where ϵ is a small constant (e.g., 1e − 9) for numerical stability. Drel characterizes the token type. By introducing the ambiguity hyperparameter τ, a token (i.e., intermediate representation) is identified as ambiguous if Drel ≤ τ, indicating no clear routing preference between old and new expert groups.

Token assignment guidance. The TAG mechanism routes a token to new expert group St,new only if it meets two conditions simultaneously: (1) it is not ambiguous (Drel > τ), and (2) it is new-dominant (cnew > cold).

We formalize this decision with a binary mask Mnew ∈ {0,1}: Mnew = ((c new > cold) ∧ (Drel > τ)), where ()

is the indicator function, Mold = 1−Mnew. This ensures that any token which is old-dominant (cold ≥ cnew) or ambiguous (Drel ≤ τ) is automatically assigned to the safe old expert group. TAG applies the mask to produce the final logits s′t ∈ R|E

t| that are passed to the Softmax function. For expert i, the final routing score s′t,i is defined as:

st,i, if mt,i = 1, −∞, otherwise,

s′t,i =

(4)

where mt,i = (i ∈ S t−1)Mold+(i ∈ S t,new)Mnew. TAG routes low-ambiguity tokens according to their inherentpattern-decided routing preference: new tokens to new experts to promote new-task learning, and old tokens to old experts with their residual affinity for new experts suppressed. Considering their low benefit for learning new knowledge (Sec. 3.3), high-ambiguity tokens are routed safely to old experts to prevent potential forgetting, when no additional prior knowledge is given.

###### 3.4.2. Routing Score Regularization (RSR)

As a complement to TAG, we introduce Routing Score Regularization (RSR) to directly regularize the routing score weights. RSR comprises two terms: (1) an exclusivity loss (Lexc) that enforces clean separation between old and new expert groups; and (2) a specialization loss (Lspe) that promotes new-expert utilization and specialization to enhance new-task learning.

Given a token representation at a specific layer, we define the collective gate output, i.e., the total routing probability mass assigned to each expert group, in terms of routing weights wi (from Eq. (1)) as:

wi. (5)

gold =

wi, gnew =

i∈St−1

i∈St,new

Exclusivity loss. This loss enforces clean routing separation by preventing a token from strongly activating both expert groups simultaneously, directly working on the routing

scores. Minimizing the product of their collective gate outputs encourages exclusive routing to one group, reinforcing the conditional routing decision of TAG:

Lexc = gold gnew. (6)

Specialization loss. Complementing TAG and Lexc that mitigate routing-drift and forgetting for stability, we introduce the specialization loss Lspe to promote and balance plasticity by encouraging higher routing weight toward new experts. We first define a soft target y which is close to 1 if no old experts are selected by the router (i.e., g˜old = max{wi}i∈St−1

is close to zero): y ≜ 1 − g˜old. Relying on g˜old, y approaches 1, reflecting the varying activity level of old experts. Lspe is formulated as a BCE loss between the collective new expert routing weight gnew and target y to encourage usage of new experts:

Lspe = −y log gnew − (1 − y)log 1 − gnew . (7)

Lspe works in synergy with Lexc and TAG to balance newtask learning and forgetting mitigation.

##### 3.5. Training of LLaVA-DyMoE

Training objectives. Our total training objective is a weighted combination of the primary task-learning loss, a standard auxiliary load balancing loss for MoE, and our proposed routing regularization terms:

L = LNTP + λLaux + α(Lexc + Lspe), (8)

where LNTP is the standard instruction-tuning loss (e.g., autoregressive cross-entropy), and λ and α are scalar hyperparameters. We adopt the standard auxiliary load balancing loss (Laux) [16] to ensure balanced utilization, applied to the set of newly added experts. We use a unified hyperparameter α to control the contributions of the two complementary regularization losses, Lexc and Lspe.

Integration with other methods. Our LLaVA-DyMoE, which focuses on rectifying micro-level token routingdrift, is inherently orthogonal to and compatible with other macro-level MCIT paradigms. It is also compatible with data-based approaches [8, 49, 73], as LLaVA-DyMoE enhances the router’s robustness in handling the mixed stream of old and new data, regardless of its source. Our method can also be seamlessly integrated into architectures that employ task-level routing methods [20, 73, 81]. These methods first decide which group of experts to activate at a task level, while our LLaVA-DyMoE then optimizes the token assignment within that activated group, mitigating the intra-group routing drifts we identified. These combinations offer the potential for further enhanced performance.

Table 1. Comparison with continual learning models on the CoIN benchmark.

Accuracy on Each Task (%) Aggregate Results (%)

Method

SQA VQAText ImgNet GQA VizWiz REF VQAv2 VQAOCR MFN↑ MAA↑ BWT↑

LoRA 52.56 48.12 39.27 44.47 37.46 1.22 56.10 55.11 41.79 43.99 -23.12 MoELoRA [7] 72.01 46.89 44.75 42.79 28.22 3.31 55.74 57.72 43.93 43.92 -22.18 EWC [54] 59.11 47.21 39.88 45.12 35.33 2.72 56.29 41.21 40.86 43.75 -21.76 LWF [36] 62.32 48.66 51.45 45.84 43.76 0.24 54.96 44.63 43.98 44.89 -19.69 IncLoRA 73.33 44.32 54.59 44.07 25.93 4.49 54.91 58.55 45.02 43.12 -23.21 O-LoRA [63] 75.61 49.98 78.24 44.18 30.70 4.66 55.51 57.37 49.53 46.65 -17.54 IncMoELoRA 68.43 50.31 68.42 47.97 39.46 4.56 57.31 60.95 49.68 49.50 -16.67

LLaVA-DyMoE (Ours) 76.25 53.86 95.80 48.40 52.35 9.25 58.30 62.00 57.03 57.70 -4.67

Table 2. Ablations on main components.

Aggregate Results (%) MFN↑ MAA↑ BWT↑

Configuration

IncMoELoRA 49.68 49.50 -16.67 + Laux 50.76 51.17 -15.44

+ TAG 54.44 52.18 -7.04 + Lexc 55.18 54.38 -6.83 + Lspe (Ours) 57.03 57.70 -4.67

Table 3. Ablations on ambiguity threshold.

Aggregate Results (%) MFN↑ MAA↑ BWT↑

τ

10% 56.87 57.23 -4.94 20% 57.03 57.70 -4.67 30% 56.27 55.65 -5.21 50% 55.32 53.51 -5.54

Table 4. Ablations on loss weights α.

Aggregate Results (%) MFN↑ MAA↑ BWT↑

α

- 1e-2 55.43 57.73 -5.81

- 5e-3 56.87 57.50 -5.32

1e-3 57.03 57.70 -4.67

- 5e-4 56.32 57.63 -4.94

#### 4. Experiments

##### 4.1. Experimental Setup

Datasets. We evaluate our method on the CoIN [7] benchmark, which encompasses a series of eight VQA tasks. These tasks include ScienceQA (SQA) [42], TextVQA [58], ImageNet [53], GQA [26], VizWiz [22], RefCOCO (REF) [29], VQAv2 [19], and OCR-VQA [44]. Each task varies in terms of the number of data samples, stylistic features, and domain characteristics. The training set comprises a total of 569k samples, while the testing set contains 261k samples.

Evaluation metrics. We adopt the metrics introduced in CoIN [7, 8], which measure the discrepancy between the model’s output and the ground truth. Specifically, we report: (1) Mean Final Accuracy MFN = T1 Ti=1 AT,i, assessing the average accuracy across all tasks after the complete incremental training sequence, where AT,i represents the accuracy on task i after learning on the final task T; (2) Mean Average Accuracy MAA = T1 Tj=1 1j ji=1(Aj,i), representing the mean of the average accuracies on all learned tasks after each incremental training step; and (3) Backward Transfer BWT = T1 Ti=1(AT,i − Ai,i), assessing the degree of forgetting.

Implementation details. In our experiments, we utilize the pre-trained, instruction-untuned LLaVA-v1.5-7B [38] as the backbone model for continual learning. The model integrates Vicuna [11] as its language backbone and a pretrained CLIP ViT-L/14 visual encoder [47] to extract visual embeddings. Only the newly added modules are trainable,

while the other components remain frozen throughout the continual learning process. Please refer to the Appendix for details on the network architectures, hyperparameters, and implementation settings.

##### 4.2. Main Results

We evaluate the proposed method on the CoIN [7] benchmark, as presented in Table 1. Previous methods that continually train a static model without expanding new parameters for new tasks, e.g., LoRA, MoELoRA [7], EWC [54], and LWF [36], show significant forgetting and inferior accuracy as the knowledge obtained from previous tasks gets repeatedly overwritten.

To mitigate such overwriting, new modules are expanded to explicitly capture this new knowledge. We conduct baseline experiments with IncLoRA (adding new LoRA modules for each task), and O-LoRA [63], which further adds orthogonal regularization to the LoRA weight matrix to help mitigate forgetting by regularizing parameter updates in orthogonal directions. To ensure that LoRA experts learned from previous tasks can be reused while new task knowledge can be absorbed without excessive overwriting, we further implement the per-task expansion baseline in the MoE framework (IncMoELoRA), where new LoRA experts are added for each new task. This baseline exhibits comparable to or better performance than O-LoRA without any regularization techniques. However, it suffers from ambiguous token-expert assignment (see Fig. 2), where ambiguous cross-task tokens activate and train newly added experts, largely hindering old expert utilization on these tokens.

Our proposed method significantly outperforms these

Table 5. LLaVA-DyMoE is compatible with data-based continual learning strategies.

Accuracy on Each Task (%) Aggregate Results (%)

Method

SQA VQAText ImgNet GQA VizWiz REF VQAv2 VQAOCR MFN↑ MAA↑ BWT↑ None

MoELoRA [7] 72.01 46.89 44.75 42.79 28.22 3.31 55.74 57.72 43.93 43.92 -22.18 O-LoRA [63] 75.61 49.98 78.24 44.18 30.70 4.66 55.51 57.37 49.53 46.65 -17.54

SEFE [8] 75.35 58.66 83.10 54.25 48.85 16.75 65.35 66.25 58.57 63.04 -10.45

+ ASD [8]

- LLaVA-DyMoE (Ours) 74.60 55.24 93.80 53.45 55.00 25.50 63.95 62.85 60.55 62.26 -4.75

+ Replay

ProgLoRA [73] 74.84 51.83 83.90 49.93 53.87 31.19 62.71 64.44 59.09 62.38 -6.59

- LLaVA-DyMoE (Ours) 75.55 56.88 96.50 54.75 55.90 29.15 63.65 64.25 62.08 61.93 -1.55

methods through our token-level routing regularization. This enforces the assignment of ambiguous cross-task tokens to old, frozen experts, bringing improvements of 7.35%, 8.20%, and 12.00% on MFN, MAA, and BWT, respectively. We provide qualitative studies in the Appendix.

##### 4.3. Ablation Study

The effect of main components. In Table 2, we ablate the effects of each component of our method. Starting from our implemented baseline (IncMoELoRA), adopting a standard auxiliary load balancing loss (Laux) yields a slight performance improvement by balancing the expert usage for the new incoming task. Notably, the proposed TAG module, which applies group-wise routing guidance based on token assignment ambiguity, significantly improves the final accuracy and mitigates forgetting. This is because TAG prevents ambiguous tokens from contributing to the learning of the new task (aligned with our observations in Sec. 3.3). Building on the TAG mechanism, we further regularize the raw router logits. Adding Lexc, which regularizes the activations of ambiguous tokens on new task routers, brings a further improvement. Finally, to ensure effective learning on the new task, we enforce the utilization of new task routers on (potentially) new task-specific tokens through Lspe. This component largely improves new task learning while maintaining low forgetting, bringing the full model’s performance to the state-of-the-art level.

The effect of ambiguity threshold τ for token assignment routing guidance. We ablate the effects of different ambiguity thresholds in Table 3, where we sweep the threshold over {10%,20%,30%,50%}. The ambiguity threshold controls the token-expert assignment to the old and new groups of experts. A higher ambiguity threshold encourages ambiguous tokens to be assigned to frozen old experts, which helps mitigate forgetting (as these experts do not contribute to the learning of the new router and experts) but could limit learning on new tasks, and vice versa. Through our experiments, we find that controlling the ambiguity threshold between 10% and 20% brings the best tradeoff between learning new tasks and forgetting old tasks. Using a larger ambiguity threshold hinders effective learning on the new task, as shown by the decreased MFN and MAA.

The effect of token assignment regularization weighting factor α. We conduct an ablation study on the hyperparameter α, which controls the unified weighting of our proposed regularization losses (Lexc and Lspe). As shown in Table 4, we evaluate model performance by sweeping α across the values {1e-2,5e-3,1e-3,5e-4}. The results indicate that our proposed method is relatively robust to the choice of this weighting factor, and we adopt α = 1e-3 by default.

##### 4.4. Discussion

Study of complementary data-based techniques. Our proposed method is orthogonal to and compatible with data-based techniques. As shown in Table 5, incorporating these techniques with our method yields consistent improvements across metrics, demonstrating their complementary nature. We first compare against SEFE [8], which adopts the ASD data paradigm and achieves a BWT of 10.45%. By complementing this same ASD paradigm with our proposed LLaVA-DyMoE, forgetting is significantly mitigated, improving the BWT to -4.75%. We evaluate against replay-based methods. When combined with a standard replay buffer [51] (using a buffer size comparable to ProgLoRA [73]), our approach again outperforms the competitor: LLaVA-DyMoE with a small replay buffer achieves a BWT of -1.55%, substantially better than ProgLoRA’s 6.59%, while also attaining a higher mean final accuracy.

#### 5. Conclusion

We analyze routing-drift in dynamic MoE expansion and trace its cause to the token level: ambiguous and old tokens in new-task data offer minimal learning value yet corrupt the router’s policy for old tasks when left unguided. We propose LLaVA-DyMoE, comprising a Token Assignment Guidance mechanism that detects and redirects highdrift tokens, and complementary routing losses that enforce expert-group separation. Experiments show that LLaVADyMoE effectively mitigates routing-drift. As it targets the inherent routing mechanism, it is complementary to existing MCIT paradigms for further gains.

Limitations and future work. Future work could further investigate the scalability of our approach on larger-scale models and in more realistic scenarios.

#### Acknowledgments

This work was partially supported by the ARC DECRA Fellowship (DE230101591), the ARC Discovery Project Grant (DP260103379), and the NVIDIA Academic Grant Program.

#### References

- [1] Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. Memory aware synapses: Learning what (not) to forget. In Proceedings of the European conference on computer vision (ECCV), pages 139–154, 2018. 2
- [2] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425– 2433, 2015. 1
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 2
- [4] Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. Advances in neural information processing systems, 33:15920–15930,

2020. 2

- [5] Meng Cao, Yuyang Liu, Yingfei Liu, Tiancai Wang, Jiahua Dong, Henghui Ding, Xiangyu Zhang, Ian Reid, and Xiaodan Liang. Continual llava: Continual instruction tuning in large vision-language models. arXiv preprint arXiv:2411.02564, 2024. 1, 3
- [6] Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS Torr, and Marc’Aurelio Ranzato. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486,

2019. 2

- [7] Cheng Chen, Junchen Zhu, Xu Luo, Heng T Shen, Jingkuan Song, and Lianli Gao. Coin: A benchmark of continual instruction tuning for multimodel large language models. Advances in Neural Information Processing Systems, 37: 57817–57840, 2024. 1, 2, 3, 7, 8, 13, 14, 18
- [8] Jinpeng Chen, Runmin Cong, Yuzhi Zhao, Hongzheng Yang, Guangneng Hu, Horace Ho Shing Ip, and Sam Kwong. Sefe: Superficial and essential forgetting eliminator for multimodal continual instruction tuning. In ICML, 2025. 1, 2, 3, 6, 7, 8
- [9] Wuyang Chen, Yanqi Zhou, Nan Du, Yanping Huang, James Laudon, Zhifeng Chen, and Claire Cui. Lifelong language pretraining with distribution-specialized experts. In International Conference on Machine Learning, pages 5383–5395. PMLR, 2023. 4
- [10] Zeren Chen, Ziqin Wang, Zhen Wang, Huayang Liu, Zhenfei Yin, Si Liu, Lu Sheng, Wanli Ouyang, Yu Qiao, and Jing Shao. Octavius: Mitigating task interference in mllms via lora-moe. arXiv preprint arXiv:2311.02684, 2023. 3

- [11] Wei-Lin Chiang, Zhuohan Li, Ziqing Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023. 7, 13
- [12] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 1, 2
- [13] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024. 3
- [14] Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleˇs Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE transactions on pattern analysis and machine intelligence, 44(7):3366–3385, 2021. 2
- [15] Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Jun Zhao, Wei Shen, Yuhao Zhou, Zhiheng Xi, Xiao Wang, Xiaoran Fan, et al. Loramoe: Alleviate world knowledge forgetting in large language models via moe-style plugin. arXiv preprint arXiv:2312.09979, 2023. 3
- [16] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022. 1, 3, 6
- [17] Robert M French. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128–135, 1999. 1
- [18] Chendi Ge, Xin Wang, Zeyang Zhang, Hong Chen, Jiapei Fan, Longtao Huang, Hui Xue, and Wenwu Zhu. Dynamic mixture of curriculum lora experts for continual multimodal instruction tuning. arXiv preprint arXiv:2506.11672, 2025. 3
- [19] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017. 7, 13
- [20] Haiyang Guo, Fanhu Zeng, Ziwei Xiang, Fei Zhu, DaHan Wang, Xu-Yao Zhang, and Cheng-Lin Liu. Hidellava: Hierarchical decoupling for continual instruction tuning of multimodal large language model. arXiv preprint arXiv:2503.12941, 2025. 2, 3, 6, 15
- [21] Yongxin Guo, Zhenglin Cheng, Xiaoying Tang, Zhaopeng Tu, and Tao Lin. Dynamic mixture of experts: An autotuning approach for efficient transformer models. arXiv preprint arXiv:2405.14297, 2024. 3
- [22] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham.

- Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617, 2018. 7, 13
- [23] Jinghan He, Haiyun Guo, Ming Tang, and Jinqiao Wang. Continual instruction tuning for large multimodal models. arXiv preprint arXiv:2311.16206, 2023. 1, 3
- [24] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 1, 3
- [25] Tianyu Huai, Jie Zhou, Xingjiao Wu, Qin Chen, Qingchun Bai, Ze Zhou, and Liang He. Cl-moe: Enhancing multimodal large language model with dual momentum mixtureof-experts for continual visual question answering. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19608–19617, 2025. 2, 3
- [26] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 7, 13
- [27] Saurav Jha, Dong Gong, and Lina Yao. Clap4clip: Continual learning with probabilistic finetuning for vision-language models. Advances in neural information processing systems, 37:129146–129186, 2024. 2
- [28] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. 1, 2
- [29] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 787–798, Doha, Qatar, 2014. Association for Computational Linguistics. 7, 13
- [30] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka GrabskaBarwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017. 1, 2
- [31] Mingrui Lao, Nan Pu, Yu Liu, Zhun Zhong, Erwin M Bakker, Nicu Sebe, and Michael S Lew. Multi-domain lifelong visual question answering via self-critical distillation. In Proceedings of the 31st ACM International Conference on Multimedia, pages 4747–4758, 2023. 1
- [32] Stan Weixian Lei, Difei Gao, Jay Zhangjie Wu, Yuxuan Wang, Wei Liu, Mengmi Zhang, and Mike Zheng Shou. Symbolic replay: Scene graph as prompt for continual learning on vqa task. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1250–1259, 2023. 1
- [33] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam

- Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020. 3
- [34] Dengchun Li, Yingzi Ma, Naizheng Wang, Zhengmao Ye, Zhiyuan Cheng, Yinghao Tang, Yan Zhang, Lei Duan, Jie Zuo, Cal Yang, et al. Mixlora: Enhancing large language models fine-tuning with lora-based mixture of experts. arXiv preprint arXiv:2404.15159, 2024. 3
- [35] Xilai Li, Yingbo Zhou, Tianfu Wu, Richard Socher, and Caiming Xiong. Learn to grow: A continual structure learning framework for overcoming catastrophic forgetting. In International conference on machine learning, pages 3925–

3934. PMLR, 2019. 2

- [36] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947, 2017. 1, 2, 7
- [37] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 1
- [38] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1, 2, 3, 7, 13
- [39] David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30, 2017. 2
- [40] Haodong Lu, Chongyang Zhao, Jason Xue, Lina Yao, Kristen Moore, and Dong Gong. Adaptive rank, reduced forgetting: Knowledge retention in continual learning visionlanguage models with dynamic rank-selective lora. arXiv preprint arXiv:2412.01004, 2024. 2
- [41] Haodong Lu, Chongyang Zhao, Jason Xue, Lina Yao, Kristen Moore, and Dong Gong. Little by little: Continual learning via incremental mixture of rank-1 associative memory experts. arXiv preprint arXiv:2506.21035, 2025. 2
- [42] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022. 7, 13
- [43] Imad Eddine Marouf, Enzo Tartaglione, St´ephane Lathuili`ere, and Joost Van De Weijer. Ask and remember: A questions-only replay strategy for continual visual question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18078–18089, 2025. 1
- [44] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pages 947–

952. IEEE, 2019. 1, 7, 13

- [45] Cuong V Nguyen, Yingzhen Li, Thang D Bui, and Richard E Turner. Variational continual learning. arXiv preprint arXiv:1710.10628, 2017. 2

- [46] Jingyang Qiao, Zhizhong Zhang, Xin Tan, Yanyun Qu, Shouhong Ding, and Yuan Xie. Large continual instruction assistant. arXiv preprint arXiv:2410.10868, 2024. 1, 3
- [47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 7, 13
- [48] Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis, and Amjad Almahairi. Progressive prompts: Continual learning for language models. arXiv preprint arXiv:2301.12314, 2023. 3
- [49] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 2001–2010, 2017. 2, 6
- [50] Matthew Riemer, Tim Klinger, Djallel Bouneffouf, and Michele Franceschini. Scalable recollections for continual lifelong learning. Proceedings of the AAAI conference on artificial intelligence, 33(01):1352–1359, 2019. 2
- [51] David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy Lillicrap, and Gregory Wayne. Experience replay for continual learning. Advances in neural information processing systems, 32, 2019. 2, 8, 15
- [52] Mohammad Rostami, Soheil Kolouri, and Praveen K Pilly. Complementary learning for overcoming catastrophic forgetting using experience replay. arXiv preprint arXiv:1903.04566, 2019. 2
- [53] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115

(3):211–252, 2015. 7, 13

- [54] Jonathan Schwarz, Wojciech Czarnecki, Jelena Luketina, Agnieszka Grabska-Barwinska, Yee Whye Teh, Razvan Pascanu, and Raia Hadsell. Progress & compress: A scalable framework for continual learning. In International conference on machine learning, pages 4528–4537. PMLR, 2018. 7
- [55] Joan Serra, Didac Suris, Marius Miron, and Alexandros Karatzoglou. Overcoming catastrophic forgetting with hard attention to the task. In International conference on machine learning, pages 4548–4557. PMLR, 2018. 2
- [56] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixtureof-experts layer. arXiv preprint arXiv:1701.06538, 2017. 3
- [57] Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim. Continual learning with deep generative replay. Advances in neural information processing systems, 30, 2017. 2
- [58] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8317–8326, 2019. 7, 13

- [59] Qwen Team. Qwen2.5: A party of foundation models, 2024. 1, 2
- [60] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1, 2
- [61] Huiyi Wang, Haodong Lu, Lina Yao, and Dong Gong. Selfexpansion of pre-trained models with mixture of adapters for continual learning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10087–10098,

2025. 2, 4

- [62] Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: theory, method and application. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 2
- [63] Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuan-Jing Huang. Orthogonal subspace learning for language model continual learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10658–10671, 2023. 3, 7, 8
- [64] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1, 2
- [65] Ziqi Wang, Chang Che, Qi Wang, Yangyang Li, Zenglin Shi, and Meng Wang. Smolora: Exploring and defying dual catastrophic forgetting in continual visual instruction tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 177–186, 2025. 1, 3
- [66] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Yujin Han, Zhekai Chen, Jiayu Wang, Difan Zou, Xihui Liu, Yingya Zhang, Yu Liu, et al. Routing matters in moe: Scaling diffusion transformers with explicit routing guidance. arXiv preprint arXiv:2510.24711, 2025. 5
- [67] Yuan Xie, Shaohan Huang, Tianyu Chen, and Furu Wei. Moec: Mixture of expert clusters. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 13807– 13815, 2023. 5
- [68] Shipeng Yan, Jiangwei Xie, and Xuming He. Der: Dynamically expandable representation for class incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3014–3023,

2021. 2

- [69] Longrong Yang, Dong Shen, Chaoxiang Cai, Fan Yang, Tingting Gao, Di ZHANG, and Xi Li. Solving token gradient conflict in mixture-of-experts for large vision-language model. In The Thirteenth International Conference on Learning Representations, 2025. 1, 3, 5
- [70] Fei Ye and Adrian G Bors. Self-evolved dynamic expansion model for task-free continual learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22102–22112, 2023. 2
- [71] Jaehong Yoon, Eunho Yang, Jeongtae Lee, and Sung Ju Hwang. Lifelong learning with dynamically expandable networks. arXiv preprint arXiv:1708.01547, 2017. 2

- [72] Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Ping Hu, Dong Wang, Huchuan Lu, and You He. Boosting continual learning of vision-language models via mixture-of-experts adapters. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23219–23230, 2024. 2, 3, 15
- [73] Yahan Yu, Duzhen Zhang, Yong Ren, Xuanle Zhao, Xiuyi Chen, and Chenhui Chu. Progressive lora for multimodal continual instruction tuning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 2779– 2796, 2025. 1, 2, 3, 4, 6, 8, 15
- [74] Ted Zadouri, Ahmet Ust¨¨ un, Arash Ahmadian, Beyza Ermis, Acyr Locatelli, and Sara Hooker. Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning. In The Twelfth International Conference on Learning Representations, 2024. 1
- [75] Fanhu Zeng, Fei Zhu, Haiyang Guo, Xu-Yao Zhang, and Cheng-Lin Liu. Modalprompt: Towards efficient multimodal continual instruction tuning with dual-modality guided prompt. arXiv preprint arXiv:2410.05849, 2024. 1
- [76] Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In International conference on machine learning, pages 3987–3995. PMLR,

2017. 2

- [77] Duzhen Zhang, Yong Ren, Zhong-Zhi Li, Yahan Yu, Jiahua Dong, Chenxing Li, Zhilong Ji, and Jinfeng Bai. Enhancing multimodal continual instruction tuning with branchlora. arXiv preprint arXiv:2506.02041, 2025. 3, 4
- [78] Gengwei Zhang, Liyuan Wang, Guoliang Kang, Ling Chen, and Yunchao Wei. Slca: Slow learner with classifier alignment for continual learning on a pre-trained model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19148–19158, 2023. 2
- [79] Junting Zhang, Jie Zhang, Shalini Ghosh, Dawei Li, Serafettin Tasci, Larry Heck, Heming Zhang, and C-C Jay Kuo. Class-incremental learning via deep model consolidation. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1131–1140, 2020. 2
- [80] Chongyang Zhao and Dong Gong. Learning mamba as a continual learner: Meta-learning selective state space models for efficient continual learning. arXiv preprint arXiv:2412.00776, 2024. 2
- [81] Hengyuan Zhao, Ziqin Wang, Qixin Sun, Kaiyou Song, Yilin Li, Xiaolin Hu, Qingpei Guo, and Si Liu. Llava-cmoe: Towards continual mixture of experts for large vision-language models. arXiv preprint arXiv:2503.21227, 2025. 2, 4, 6, 15

## On Token’s Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models

### Supplementary Material

#### A. Additional Experiment Details

##### A.1. More Implementation Details

In our experiments, we utilize the pre-trained, instructionuntuned LLaVA-v1.5 [38] as the backbone model for continual learning. The model employs Vicuna [11] as its language backbone and a pre-trained CLIP ViT-L/14 visual encoder [47] for visual feature extraction. During continual learning, only the newly added modules are trainable, while all other components remain frozen. In the default setting, 16 rank-4 LoRA experts are added when each new task starts, and K = 16 is applied for top-K routing over all trained experts. Different configurations are evaluated in ablation studies. The same configurations are also applied to the baseline model IncMoELoRA in experiments. All experiments are conducted on a compute node equipped with four NVIDIA H100 GPUs. Following the official LLaVAv1.5 configuration, we adopt a global batch size of 128 and a learning rate of 2 × 10−4. We set the warmup ratio to 0.03 and use the AdamW optimizer for training. The model is trained in PyTorch with BF16 precision and DeepSpeed ZeRO-2. We set the weights of the load balancing, exclusivity, and specialization losses to 1 × 10−3. For all compared methods, we follow the default configurations from their original papers. Other remaining settings are consistent with those specified for LLaVA-v1.5 [38].

##### A.2. Details of Datasets

The eight tasks in the CoIN [7] benchmark are as follows:

ScienceQA (SQA) [42] is a multimodal science questionanswering dataset designed to assess models’ reasoning over integrated visual and textual information. The training set contains 12,726 samples (6,218 image–text and 6,508 text-only), and the test set includes 4,241 samples (2,017 image–text and 2,224 text-only).

TextVQA [58] focuses on text recognition within visual question-answering. It features real-world images with diverse textual content. The training set includes 34,602 image–text samples, and the test set comprises 5,000 image–text samples.

ImageNet [53] is a large-scale benchmark for image classification. The training set contains 129,833 image–text samples, and the test set includes 5,050 image–text samples.

GQA [26] emphasizes real-world visual reasoning, requiring understanding of object relationships and multi-step inference based on both synthetic and real images with scene graphs. The training and test sets include 72,140 and 12,578

Table 6. Performance under varying task orders.

Aggregate Results (%) MFN↑ MAA↑ BWT↑ Origin 57.03 57.70 -4.67

Task Order

Reverse 56.67 57.34 -4.71 Alphabet 56.44 56.98 -4.92

image–text samples, respectively.

VizWiz [22] is designed for visual question-answering in assistive contexts for visually impaired users. It provides 20,523 training samples and 4,319 test samples, all in the image–text modality.

Grounding (Ref) [29] evaluates grounding of naturallanguage expressions in images. It contains image–text pairs requiring models to predict bounding boxes aligned with textual descriptions. The training set includes 55,885 samples, and the test set includes 30,969 samples.

VQAv2 [19], a visual question-answering benchmark, features balanced answer distributions and broad topical coverage. It provides 82,783 training samples and 214,354 test samples, all in the image–text modality.

OCRVQA [44] integrates OCR with visual questionanswering to assess models’ ability to extract and reason over textual content in images. The dataset includes 165,348 training samples and 99,926 test samples, all image–text.

#### B. More Experiments

##### B.1. Experiments on Different Task Orders

To assess order sensitivity, we trained our method under multiple task orderings on the eight CoIN datasets. We conduct experiments using three task orderings: “Origin”, the original task ordering proposed in the CoIN benchmark; “Reverse”, the reversed version of the original ordering; and “Alphabet”, where tasks are ordered alphabetically. As summarized in Table 6, our method exhibits excellent stability across different task orderings; the amount of forgetting remains consistently low with minimal variation across orderings. Without techniques like experience replay, our proposed token-level expert assignment regularization within this incremental MoE with LoRA approach consistently learns new task knowledge with minimal forgetting, regardless of the task ordering. This is because our method effectively prevents ambiguous tokens from con-

Table 7. Performance under distinct training instruction types.

Aggregate Results (%) MFN↑ MAA↑ BWT↑

Instruction

Origin 57.03 57.70 -4.67 Diverse 56.90 57.44 -5.12 10Type 56.77 57.25 -4.87

tributing to the learning of new routers and experts, ensuring that effective new knowledge is absorbed into the new experts without any assumptions about the incoming order of the data.

##### B.2. Experiments on Distinct Training Instruction Types

To validate the reliability of our proposed method against distinct instruction templates, we conduct experiments with different template types, reported in Table 7. There are three types of instruction templates in the CoIN benchmark [7]. Following the default setting, the experiments in the main paper are based on the “Origin” type. We further conduct experiments with the other two types of instruction templates, Diverse and 10Type, in [7]: 1) Diverse: Distinct instruction templates tailored to different tasks. 2) 10Type: Randomly sampled from 10 distinct instruction templates. (Details can be found in Table 16.) The results show that forgetting and accuracy on all three metrics are nearly identical across instruction types, indicating the method’s stability. This result is significant, as it indicates that our method’s token-level routing mechanism is not overfitting to superficial, task-specific prompt formats.

##### B.3. Ablation Studies on MoE Configurations

In this section, we validate the proposed method on different MoE configurations, including the top-K value, the number of experts, and the expert capacity, and show the results in Table 8, 9, and 10. Experiments with baseline IncMoELoRA are conducted as a reference. The ablation studies demonstrate that the proposed method performs robustly across different MoE configurations and consistently delivers improvements.

Ablations on top-K value. First, we conduct an ablation study on the top-K value, comparing top-8 and top-16 over all experts. As shown in Table 8, the overall performance is relatively insensitive to the choice of top-K, and under both configurations our LLaVA-DyMoE consistently outperforms the baseline method, demonstrating the effectiveness of the proposed approach.

Ablations on expert number. Second, we conduct an ablation study on the number of newly added experts, comparing 8 and 16 experts per task. In the 8-expert setting, we

Table 8. Ablations on top-K value.

Aggregate Results (%) MFN↑ MAA↑ BWT↑ 8

Top-K Method

IncMoELoRA 48.55 49.87 -16.28

- LLaVA-DyMoE 56.89 57.71 -5.12

16

IncMoELoRA 49.68 49.50 -16.67

- LLaVA-DyMoE 57.03 57.70 -4.67

Table 9. Ablations on expert number.

Aggregate Results (%) MFN↑ MAA↑ BWT↑ 8

Expert Number

Method

IncMoELoRA 48.39 50.62 -17.93

- LLaVA-DyMoE 56.97 58.37 -5.78

16

IncMoELoRA 49.68 49.50 -16.67

- LLaVA-DyMoE 57.03 57.70 -4.67

Table 10. Ablations on expert capacity.

Aggregate Results (%) MFN↑ MAA↑ BWT↑

Expert Capacity

Method

- 1

IncMoELoRA 48.23 49.14 -15.79 LLaVA-DyMoE 56.78 57.34 -4.13

- 2

IncMoELoRA 49.08 49.25 -16.58

- LLaVA-DyMoE 56.91 57.48 -4.62

4

IncMoELoRA 49.68 49.50 -16.67

- LLaVA-DyMoE 57.03 57.70 -4.67

increase the parameters of each expert to maintain a comparable total capacity. As shown in Table 9, LLaVA-DyMoE consistently outperforms the baseline under both configurations, demonstrating its effectiveness.

Ablations on expert capacity. Furthermore, we conduct an ablation study on the impact of expert capacity by varying the LoRA experts’ rank to 1, 2, and 4. As shown in Table 10, models with larger expert capacity achieve better performance, and across all capacity settings LLaVA-DyMoE consistently outperforms the baseline, further demonstrating the effectiveness of the proposed approach.

Overall, these ablations show that LLaVA-DyMoE is robust to variations in the MoE configuration: it consistently outperforms the baseline across different choices of top-K, number of experts, and expert capacity.

##### B.4. Experiments with Different Backbone Sizes

Besides the 7B model, we further validate our method using the larger 13B LLaVA backbone, as shown in Table 11.

Table 11. Performance across different model sizes.

Accuracy on Each Task (%) Aggregate Results (%)

Size Method

SQA VQAText ImgNet GQA VizWiz REF VQAv2 VQAOCR MFN↑ MAA↑ BWT↑ 7B

IncMoELoRA 68.43 50.31 68.42 47.97 39.46 4.56 57.31 60.95 49.68 49.50 -16.67 LLaVA-DyMoE (Ours) 76.25 53.86 95.80 48.40 52.35 9.25 58.30 62.00 57.03 57.70 -4.67

IncMoELoRA 68.75 51.69 85.80 48.10 40.20 6.55 58.85 64.60 53.07 53.20 -14.23 LLaVA-DyMoE (Ours) 78.75 56.24 96.05 55.85 53.20 13.85 64.05 65.15 60.39 61.25 -4.64

13B

Table 12. LLaVA-DyMoE is compatible with task-level routing methods.

Accuracy on Each Task (%) Aggregate Results (%)

Method

SQA VQAText ImgNet GQA VizWiz REF VQAv2 VQAOCR MFN↑ MAA↑ BWT↑

LLaVA-DyMoE 76.25 53.86 95.80 48.40 52.35 9.25 58.30 62.00 57.03 57.70 -4.67 + Task Router 78.18 53.36 95.63 54.63 53.92 24.46 59.54 60.40 60.02 60.78 -1.73

Table 13. LLaVA-DyMoE is compatible with data-based continual learning strategies. Table 5 in the main paper shows that the proposed token assignment regularization can work with replay techniques. This table shows the performance across different replay buffer sizes, with ProgLoRA [73] (containing replay) as a reference.

Aggregate Results (%) MFN↑ MAA↑ BWT↑ 200

Replay Size

Method

ProgLoRA 59.09 62.38 -6.59 LLaVA-DyMoE 62.08 61.93 -1.55

ProgLoRA 59.14 62.74 -6.47

500

- LLaVA-DyMoE 62.55 62.17 -1.00

1000

ProgLoRA 59.66 63.23 -6.21

- LLaVA-DyMoE 63.19 62.95 -0.64

Scaling to a larger and stronger backbone model yields improved continual learning performance while preserving a low forgetting rate. LLaVA-DyMoE demonstrates robust scalability, effectively leveraging the increased capacity to achieve a higher MFN of 60.39% while maintaining a consistently low forgetting rate (-4.64%). This confirms that our drift-aware token assignment mechanism remains effective regardless of the underlying model size.

##### B.5. Additional Results on Data-based Strategies

The proposed drift-aware token assignment regularization in LLaVA-DyMoE is orthogonal and compatible with databased strategies such as replay and data augmentation. By focusing on the core router training, our method can improve performance when combined with these techniques. In the main paper, we have provided the experiments in Table 5. In this section, we provide additional details on the results of replay techniques under different replay buffer

sizes. We compare our method, LLaVA-DyMoE equipped with a standard replay buffer [51], against ProgLoRA [73]. This configuration serves as a basic replay-based variant of our dynamic MoE architecture. Following ProgLoRA, we vary the buffer size (200, 500, 1000) to match comparable replay budgets. As shown in Table 13, LLaVA-DyMoE consistently achieves competitive or better performance across different replay buffer sizes.

##### B.6. Compatibility with Additional Task-specific Router

Our LLaVA-DyMoE, which focuses on rectifying microlevel token routing drifts, is inherently orthogonal to and compatible with macro-level MCIT paradigms based on task-specific routing. In particular, our method can be seamlessly integrated into architectures that employ tasklevel routing strategies [20, 72, 73, 81]. These approaches first decide which group of experts to activate at the task level, while LLaVA-DyMoE then optimizes the token assignments within the selected group, mitigating the intragroup routing drifts we identified and thus providing complementary benefits.

To verify this compatibility, we equip LLaVA-DyMoE with a task-specific router. In this setup, the task router determines which experts are activated for each task, while our dynamic MoE component can further regularize token-level routing. As shown in Table 12, this combination yields improved performance over vanilla LLaVA-DyMoE, demonstrating that LLaVA-DyMoE can provide additive gains when integrated with task-specific routing methods.

##### B.7. LLaVA-DyMoE with Expert Pruning

We evaluate the performance of LLaVA-DyMoE under different MoE configurations. In particular, we investigate expert pruning, which removes potentially unnecessary ex-

Table 14. Performance of LLaVA-DyMoE with expert pruning.

Accuracy on Each Task (%) Aggregate Results (%)

Method

SQA VQAText ImgNet GQA VizWiz REF VQAv2 VQAOCR MFN↑ MAA↑ BWT↑

LLaVA-DyMoE 76.25 53.86 95.80 48.40 52.35 9.25 58.30 62.00 57.03 57.70 -4.67 + Pruning 1/8 76.04 53.91 96.10 48.16 52.51 9.27 58.21 61.79 57.00 57.62 -4.63 + Pruning 1/4 75.59 53.79 95.11 47.78 52.51 9.26 57.94 61.39 56.67 57.37 -4.48

Visualization of Expert Activation

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

ExpertGroupID

0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 Layer

(a) LLaVA-DyMoE after training on the 8-th task.

Visualization of Expert Activation

- 1

- 2

- 3

- 4

- 5

ExpertGroupID

0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 Layer

(b) LLaVA-DyMoE after training on the 5-th task.

- Figure 4. Layer-wise expert activation on the CoIN benchmark. Activation frequency is shown for each expert group across layers, and circle size reflects how often an expert is activated.

perts from the MoE. Table 14 reports the performance of LLaVA-DyMoE after pruning either 1/8 or 1/4 of the experts with the lowest activation frequencies following the training of each task. The results show that our method remains robust even with expert pruning. Note that we apply only a simple, naive pruning strategy, which leads to a slight performance drop. Although pruning is not the main focus of this work, this experiment demonstrates the potential of our proposed techniques to remain effective under more complex MoE training pipelines.

##### B.8. Visualization of Expert Activation

In Fig. 4, we present layer-wise expert activation frequencies of LLaVA-DyMoE across the eight tasks in the CoIN benchmark. For clarity, the activation frequencies of newly added experts for each task are merged into one single ex-

pert group. The routing scores are aggregated as the expert activation strength. The visualization shows that all experts are activated with varying strengths, exhibiting diverse utilization patterns across layers.

##### B.9. Qualitative Result Examples

We provide a qualitative comparison in Fig. 5 by randomly sampling data from previous tasks (ScienceQA and ImageNet) after the model has finished training on the final task. As illustrated, LLaVA-DyMoE successfully retains fine-grained knowledge that the baseline often forgets. Specifically, the IncMoELoRA baseline tends to regress to coarse-grained or incorrect labels, such as simplifying a “Bernese mountain dog” to a generic “Dog”, or confusing a “Sloth bear” with a visually similar “Otter”. In contrast, our method accurately recalls specific species and reason-

|[Figure 22]|
|---|

What is the object in the image?

|[Figure 23]|
|---|

[Figure 24]

IncMoELoRA: Otter ❌ LLaVA-DyMoE: Sloth bear ✅ Ground Truth: Sloth bear

[Figure 25]

|[Figure 26]|
|---|

A tomato plant can grow seeds. Which part of the tomato plant makes the seeds? A. the fruit; B. the flowers.

|[Figure 27]|
|---|

What is the object in the image?

[Figure 28]

IncMoELoRA: Dog ❌ LLaVA-DyMoE: Bernese mountain dog ✅ Ground Truth: Bernese mountain dog

[Figure 29]

- IncMoELoRA: A ❌

[Figure 30]

- LLaVA-DyMoE: B ✅

[Figure 31]

- Ground Truth: B

What is the capital of Louisiana? A. Frankfort; B. Salem; C. Baton Rouge; D. New Orleans

IncMoELoRA: C ✅ LLaVA-DyMoE: C ✅

[Figure 32]

[Figure 33]

- Ground Truth: C

|[Figure 34]|
|---|

Which country is highlighted?

- A. Antigua and Barbuda;
- B. Saint Kitts and Nevis;
- C. Grenada; D. Saint Lucia.

IncMoELoRA: B ❌ LLaVA-DyMoE: D ✅

[Figure 35]

[Figure 36]

- Ground Truth: D

|[Figure 37]|
|---|

What is the object in the image?

[Figure 38]

IncMoELoRA: Desk ❌ LLaVA-DyMoE: Computer ❌ Ground Truth: Mouse

[Figure 39]

- Figure 5. Comparisons between baseline IncMoELoRA and LLaVA-DyMoE on cases after training on the final task. The first column shows cases from ScienceQA, the second column shows cases from ImageNet.

Table 15. Training time of LLaVA-DyMoE during sequential training.

Training Time on Each Task (min)

Method

Average SQA VQAText ImgNet GQA VizWiz REF VQAv2 VQAOCR

IncMoELoRA 7.4 13.6 92.4 123.4 14.9 99.6 105.0 137.1 74.18 LLaVA-DyMoE (Ours) 7.4 14.6 95.5 127.6 15.8 103.2 109.6 145.7 77.43

ing details, such as identifying the correct biological part of a “Tomato” plant. While complex scenes with small objects remain challenging for both models (e.g., the ambiguous Mouse case), our approach exhibits improved knowledge retention compared to the baseline across diverse domains.

ing a sparse MoE architecture, we minimize the computational energy required for long-term learning compared to dense retraining methods, aligning with the goals of Green AI. Regarding ethical considerations, we note that our model builds upon the pre-trained LLaVA backbone and standard datasets within the open-source CoIN benchmark. As with general data-driven LVLMs, our model naturally reflects the data distributions and characteristics of these foundational resources. While our current work focuses on optimizing knowledge retention and plasticity, we encourage future research to continue exploring safety alignment and fairness as integral components of the continual learning process for real-world applications.

##### B.10. Efficiency

The proposed drift-aware token assignment regularization is applied during training with minor additional computations. To validate the efficiency, we report the training time of the baseline (IncMoELoRA) and our LLaVA-DyMoE in Table 15. Our method incurs only a small training-time overhead of 4.4% (from 74.18 minutes to 77.43 minutes), while leaving inference efficiency unaffected.

#### C. Ethical and Social Impacts

This work advances MCIT by effectively enabling LVLMs to incrementally perform instruction tuning on new tasks while maintaining proficiency on previously learned ones. A key social benefit of our proposed LLaVA-DyMoE is its emphasis on parameter and inference efficiency. By utiliz-

Table 16. The list of instruction templates for each task [7].

Task Original Diverse 10Type

Answer with the option’s letter from the given choices directly Select the correct answer from the given choices and respond with the letter of the chosen option Determine the correct option from the provided choices and reply with its corresponding letter Pick the correct answer from the listed options and provide the letter of the selected option Identify the correct choice from the options below and respond with the letter of the correct option From the given choices, choose the correct answer and respond with the letter of that choice Choose the right answer from the options and respond with its letter Select the correct answer from the provided options and reply with the letter associated with it From the given choices, select the correct answer and reply with the letter of the chosen option Identify the correct option from the choices provided and respond with the letter of the correct option From the given choices, pick the correct answer and respond by indicating the letter of the correct option

Answer with the option’s letter from the given choices directly

Answer with the option’s letter from the given choices directly

SQA

Answer the question with just one word or a brief phrase Use one word or a concise phrase to respond to the question Answer using only one word or a short, descriptive phrase Provide your answer in the form of a single word or a brief phrase Use a single word or a short phrase to respond to the question Summarize your response in one word or a concise phrase Respond to the question using a single word or a brief phrase Provide your answer in one word or a short, descriptive phrase Answer the question with a single word or a brief, descriptive phrase Capture the essence of your response in one word or a short phrase Capture the essence of your response in a single word or a concise phrase

Capture the essence of your response in a single word or a concise phrase

Answer the question using a single word or phrase

VQAText

Express your answer in a single word or a short, descriptive phrase Provide your answer using a single word or a brief phrase Describe the content of the image using one word or a concise phrase Respond to the question with a single word or a short, descriptive phrase Classify the image content using only one word or a brief phrase Give your answer in the form of a single word or a concise phrase Use a single word or a short phrase to categorize the image content Express your answer with one word or a short, descriptive phrase Identify the type of content in the image using one word or a concise phrase Summarize your response in a single word or a brief phrase Use one word or a short phrase to classify the content of the image

Express your answer in a single word or a short, descriptive phrase

Answer the question using a single word or phrase

ImgNet

Respond to the question with a single word or a short phrase Respond to the question using only one word or a concise phrase Answer the question with a single word or a brief phrase Respond with one word or a short phrase Provide your answer in the form of a single word or a concise phrase Respond to the question with just one word or a brief phrase Answer the question using a single word or a concise phrase Provide your response using only one word or a short phrase Respond to the question with a single word or a brief phrase Respond to the question using just one word or a concise phrase Answer the question with one word or a short phrase

Answer the question using a single word or phrase

Respond to the question briefly, using only one word or a phrase

GQA

Answer the question using only one word or a concise phrase Respond to the question using only one word or a concise phrase Respond to the question with a single word or a brief phrase Provide your answer using just one word or a short phrase Respond with one word or a concise phrase Answer the question with just one word or a brief phrase Use a single word or a short phrase to answer the question Provide your answer in the form of one word or a brief phrase Reply to the question using one word or a concise phrase Answer with a single word or a short phrase Use one word or a brief phrase to answer the question

Answer the question using a single word or phrase

Provide a succinct response with a single word or phrase

VizWiz

Identify and provide the bounding box coordinates that match the description given in this sentence Extract and provide the bounding box coordinates based on the region described in the sentence Please provide the bounding box coordinate of the region this sentence describes Find and provide the bounding box coordinates for the region mentioned in the sentence Provide the coordinates of the bounding box that correspond to the region described in the sentence Give the bounding box coordinates as described in the sentence Determine and provide the bounding box coordinates based on the description in the sentence Identify and provide the coordinates of the bounding box described in the sentence Provide the coordinates for the bounding box based on the region described in the sentence Extract and provide the coordinates for the bounding box described in the sentence Identify and give the coordinates of the bounding box as described by the sentence

Please provide the bounding box coordinate of the region this sentence describes

Please provide the bounding box coordinate of the region this sentence describes

REF

Answer the question using a single word or phrase Answer the question with a single word or a brief phrase Use one word or a short phrase to respond to the question Answer the question using just one word or a concise phrase Provide your answer to the question using only one word or a brief phrase Respond to the question with a single word or a short phrase Use a single word or phrase to answer the question Provide an answer using only one word or a brief phrase Answer the question succinctly with one word or a brief phrase Answer the question with just one word or a short phrase Respond to the question using a single word or a concise phrase

Answer the question using a single word or phrase

Answer the question using a single word or phrase

VQAv2

Answer with the option’s letter from the given choices directly Select the correct answer from the given choices and respond with the letter of the chosen option Determine the correct option from the provided choices and reply with its corresponding letter Pick the correct answer from the listed options and provide the letter of the selected option Identify the correct choice from the options below and respond with the letter of the correct option From the given choices, choose the correct answer and respond with the letter of that choice Choose the right answer from the options and respond with its letter Select the correct answer from the provided options and reply with the letter associated with it From the given choices, select the correct answer and reply with the letter of the chosen option Identify the correct option from the choices provided and respond with the letter of the correct option From the given choices, pick the correct answer and respond by indicating the letter of the correct option

Condense your answer for each question into a single word or concise phrase

Answer the question using a single word or phrase

VQAOCR

