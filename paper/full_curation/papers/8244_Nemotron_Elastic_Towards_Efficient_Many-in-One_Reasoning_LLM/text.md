# arXiv:2511.16664v1[cs.CL]20Nov2025

[Figure 1]

2025-11-21

## Nemotron Elastic: Towards Efficient Many-in-One Reasoning LLMs

Ali Taghibakhshi*, Sharath Turuvekere Sreenivas*, Saurav Muralidharan*, Ruisi Cai†, Marcin Chochowski, Ameya Sunil Mahabaleshwarkar, Yoshi Suhara, Oluwatobi Olabiyi, Daniel Korzekwa, Mostofa Patwary, Mohammad Shoeybi, Jan Kautz, Bryan Catanzaro, Ashwath Aithal, Nima Tajbakhsh, Pavlo Molchanov

Abstract: Training a family of large language models targeting multiple scales and deployment objectives is prohibitively expensive, requiring separate training runs for each different size. Recent work on model compression through pruning and knowledge distillation has reduced this cost; however, this process still incurs hundreds of billions of tokens worth of training cost per compressed model. In this paper, we present Nemotron Elastic, a framework for building reasoning-oriented LLMs, including hybrid Mamba-Attention architectures, that embed multiple nested submodels within a single parent model, each optimized for different deployment configurations and budgets. Each of these submodels shares weights with the parent model and can be extracted zero-shot during deployment without additional training or fine-tuning. We enable this functionality through an end-to-end trained router, tightly coupled to a two-stage training curriculum designed specifically for reasoning models. We additionally introduce group-aware SSM elastification that preserves Mamba’s structural constraints, heterogeneous MLP elastification, normalized MSE-based layer importance for improved depth selection, and knowledge distillation enabling simultaneous multi-budget optimization. We apply Nemotron Elastic to the Nemotron Nano V2 12B model, simultaneously producing a 9B and a 6B model using only 110B training tokens; this results in over 360× cost reduction compared to training model families from scratch, and around 7x compared to SoTA compression techniques. Each of the nested models performs on par or better than the SoTA in accuracy. Moreover, unlike other compression methods, the nested capability of our approach allows having a many-in-one reasoning model that has constant deployment memory against the number of models in the family.

Models on Hugging Face

Nemotron-Elastic

[Figure 2]

### Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse natural language tasks [1, 2, 3], achieving state-of-the-art performance through massive parameter scaling. However, this scaling comes at a significant cost: training LLM families with multiple model sizes—each targeting different deployment scenarios—requires training each variant from scratch, resulting in prohibitively expensive computational budgets. For instance, the Llama-3.1 family [3] spans 8B, 70B, and 405B parameters, each trained independently on trillions of tokens. This repeated full-scale training not only multiplies infrastructure costs but also limits practitioners’ ability to efficiently deploy models tailored to specific resource constraints.

Recent advances in model compression have sought to address this challenge through structured pruning and knowledge distillation [4, 5]. These methods train only

the largest model from scratch, then derive smaller variants through pruning and retraining. While effective, they still require hundreds of billions of training tokens per compressed model, keeping overall training costs high. A promising alternative to model compression is elastic or Matryoshka-style nested networks [6, 7]; here, an “elastic” or nested model is produced either from scratch or after continued training from an existing model - these elastic models have two special properties: (1) sub-networks meeting specific deployment objectives can be extracted from the parent model “for free” (i.e., without any additional training/fine-tuning), and (2) all sub-networks share the same weights with the parent model.

Concurrently, we observe two recent trends that are relevant to the above discussion: the first is the rise of hybrid models that combine attention mechanisms with State Space Models (SSMs) such as Mamba [8, 9]. These hybrid architectures, exemplified by models like Jamba [10], Zamba [11], and Nemotron-H [12], achieve

* Equal contribution. † Work done during an internship at NVIDIA. © 2025 NVIDIA. All rights reserved.

[Figure 3]

[Figure 4]

- Figure 1 | Left: Accuracy across key reasoning and mathematical benchmarks. The accuracy shown is the average across all benchmarks: MATH-500, AIME-2024, AIME-2025, GPQA, LiveCodeBench v5, and MMLU-Pro. Right: Scaling analysis comparing Nemotron Elastic and Minitron-SSM as model family size grows. Nemotron Elastic maintains constant cost for tokens and deployment memory, while Minitron-SSM scales linearly.

superior efficiency through reduced KV cache requirements and linear-time sequence processing while maintaining competitive accuracy. Unfortunately, there is very limited work targeting the elastification and compression of hybrid models [13]. Second is the transition from base and instruct-tuned models to reasoning models. Modern reasoning-capable LLMs generate extended chains of thought to solve complex problems, requiring substantial token budgets for intermediate reasoning steps. This creates a fundamental tension: reasoning models demand both architectural flexibility to handle variable computational budgets and the capacity to process long-context sequences where multi-step inference unfolds. Existing compression techniques fail to address this dual requirement, as they neither support elastic deployment across diverse constraints nor optimize for the longcontext reasoning scenarios critical to these models’ performance.

In this work, we present Nemotron Elastic, a framework for training hybrid LLMs that simultaneously support multiple deployment configurations via an end-to-end trained router. Our approach produces multiple nested sub-networks at different parameter budgets from a single elastic training run, each optimized for reasoning through a two-stage curriculum prioritizing long-context capability. We demonstrate that reasoning models require fundamentally different elastic training strategies compared to standard LLMs, with extended-context training (49K tokens) critical for multi-step inference. We achieve up to 40× reduction in training tokens compared to training model families from scratch, while enabling simultaneous training of multiple budgets within the memory footprint of the largest model alone. Our framework achieves this efficiency through: (1) importancebased component ranking establishing architecture

priority orderings, (2) frozen teacher knowledge distillation enabling joint sub-network optimization, (3) two-stage curriculum balancing router stabilization with reasoning-specific long-context adaptation, and (4) end-to-end router learning ensuring architecture decisions respond to actual task difficulty rather than post-hoc search heuristics.

We validate our approach by training elastic variants of Nemotron NanoV2 12B reasoning model [14], producing both homogeneous and heterogeneous 9B configurations plus a 6B variant, all from a single training run. We notice that the resulting nested models achieve competitive or superior accuracy compared to independently trained baselines while delivering significantly faster inference. This work provides an efficient path toward democratizing access to high-performance reasoning models across diverse deployment scenarios.

This paper makes the following key contributions:

- • First elastic reasoning model: we introduce the first elastic architecture specifically designed for reasoning LLMs, incorporating twostage training with extended-context optimization (49K tokens) critical for multi-step inference.
- • Depth elastification: We add depth reduction to elastification via iterative layer removal guided by normalized MSE to the full model’s predictions—resulting in more reliable layer ranking than single-shot or perplexity-based methods.
- • Knowledge distillation guided elastification: during elastic training, we treat the nonelastified model as a fixed teacher, guiding compression using teacher-aligned signals rather than CE loss alone. This results in elastified variants that more closely track the behavior of the origi-

- nal model.
- • Significant training cost reduction: Our approach requires only 110B tokens to derive 6B and 9B variants from a 12B parent—a 7× reduction compared to NanoV2 Compression (Minitron-SSM) and 360× more efficient than NanoV2 Pretraining from scratch.
- • Memory-efficient multi-budget training: Elastic training with nested weight-sharing requires memory overhead of only the largest model plus router parameters (<2% additional memory), enabling simultaneous training and deployment of multiple sizes without incurring a linear increase in memory costs.
- • Heterogeneous elastification: Our routerbased search enables layer-wise heterogeneous configurations (e.g., varying FFN dimensions across layers), whereas previous elastic methods support only homogeneous configurations. This allows for more granular and potentially more optimal model candidate exploration.

### Methodology

In this section, we describe the core components of Nemotron Elastic: importance estimation to establish component priority rankings, elastic formulation enabling flexible width and depth selection, two-stage training that couples router learning to task-specific constraints, and the dynamic masking implementation that enables efficient multi-budget training. Figure 2 illustrates and overview of the Nemotron Elastic Pipeline.

Importance Estimation and Model Preparation

Component importance guides the architectural search by identifying which elements contribute most to model performance. We follow an activation-based approach similar to prior work, establishing a foundation upon which the router makes selection decisions.

#### Width

We employ activation-based importance scoring to rank model components along each width dimension using layer activation magnitudes. For each axis—embedding channels, Mamba heads, Mamba head channels, attention heads, and FFN intermediate neurons—we compute importance scores from forward propagation only, keeping this phase lightweight.

For embedding channels, we aggregate normalized input activations across the sequence and batch di-

mensions:

Importance(emb𝑖) =

∑︁

|LN(𝑋)|𝑖 (1)

𝐵,𝐿

For FFN neurons, we score based on the output of the first linear layer (the intermediate activations after projection):

∑︁

Importance(neuron𝑖) =

|𝑋(W1𝑖)𝑇| (2)

𝐵,𝐿

where W1𝑖 refers to the 𝑖-th row of the first weight matrix in the FFN layer.

For Mamba components, we extract scores from projection matrix outputs (specifically W𝑥) and apply nested procedures that respect group-aware constraints. First, head channels are scored by aggregating across all heads:

∑︁

s:,𝑑

𝑠𝑑 =

(3)

⃦

⃦

𝐵,𝐿

2

where s = LN(𝑋)(W𝑥)𝑇. Then head-wise scores are computed using top-ranked channels 𝒟top:

𝑓ℎ = ⃦sℎ,𝒟

2 , ∀ℎ ∈ {1,...,𝑚ℎ} (4)

⃦

top

Finally, group-constrained ranking is applied to preserve SSM structure, ensuring heads within each Mamba group 𝒢𝑔 are ranked independently. For attention heads, importance is computed from head-wise activation magnitudes aggregated across query projections. Components are then sorted in decreasing order of importance, establishing a ranking permutation 𝜎(𝑤) that orders components by their contribution to model behavior. This sorted ordering serves as a preference structure guiding the router’s selection of which components to retain at different compression budgets.

#### Depth

Layer importance is estimated iteratively using normalized mean squared error (MSE) between the full model’s predictions and predictions with specific layers removed. For each layer 𝑗, we compute:

𝑠𝑗 = ∑︀

𝐵,𝐿(ℳfull − ℳ−𝑗)2 ∑︀

(5)

𝐵,𝐿 ℳ2full

where ℳfull represents logits from the full model and ℳ−𝑗 represents logits with layer 𝑗 ablated. The normalization by the full model’s energy ensures that importance scores are comparable across different calibration datasets. This yields per-layer importance

[Figure 5]

- Figure 2 | Overview of the Nemotron-Elastic training and deployment pipeline. Training: For each training sample, data flows to both teacher and student models. A budget (parameter size: 6B, 9B, or 12B) is selected and passed to the router, which generates differentiable masks for the student model. Knowledge distillation from the model prior to elastification enables simultaneous optimization across all budget variants. Deployment: After training, all models are extracted zero-shot from a single elastic checkpoint: the full 12B model and nested sub-networks (9B and 6B) are immediately available without additional fine-tuning or re-training.

scores {𝑠𝑗}𝑁𝑗=0−1 that quantify each layer’s contribution to model predictions. Layers are sorted in decreasing

order of importance, yielding a depth ranking permutation 𝜎(𝑑) that establishes a preference order over the layer stack. This ordering ensures that when the router selects a target depth, the most critical layers—those with highest normalized MSE—are preferentially retained through the binary depth coefficient 𝛾𝑗 = 1. This metric-driven approach captures the actual importance structure specific to the model and dataset, enabling principled depth selection during elastic training.

#### Elastic Formulation

We build upon a nested weight-sharing architecture that enables a single hybrid LLM to dynamically adapt across multiple resource constraints. The model architecture can be resized along both width dimensions (embedding size, attention heads, FFN intermediate dimensions, Mamba heads and head channels) and depth (number of layers), enabling instantaneous generation of sub-networks with different parameter budgets without additional fine-tuning.

Elastic Width. For width dimensions, we define a set of elastic choices for each component: embedding dimension 𝑑𝑒, FFN intermediate dimension 𝑑𝑓, attention heads 𝑛ℎ, Mamba heads 𝑚ℎ, and Mamba head channels 𝑚𝑑. At training time, sub-networks are constructed by selecting values from these dimension ranges according to a target budget. For a given objective for a sub-network (e.g., latency, memory, model size, etc.), the router selects appropriate dimensions (𝑑𝑗𝑒,𝑑𝑗𝑓,𝑛𝑗ℎ,𝑚𝑗ℎ,𝑚𝑗𝑑) to satisfy that objective. The nested structure ensures that smaller sub-networks always use a contiguous subset of the neurons, heads, and channels retained by larger variants, achieved through the importance-based ranking established during model preparation. Specifically, embeddings are selected via ℳemb masking, FFN neurons via ℳffn, attention heads via ℳattn_head, and Mamba components via ℳmamba, maintaining consistency with the dynamic masking operators defined in the Implementation section.

Elastic Depth. Depth elasticity is controlled through a binary selection vector 𝛾𝑗 = [𝛾0𝑗,𝛾1𝑗,...,𝛾𝑁𝑗 −1] where 𝛾𝑖𝑗 ∈ {0,1} determines whether layer 𝑖 is active in sub-network 𝑗.

Layers with 𝛾𝑖𝑗 = 0 are bypassed through residual skip connections, maintaining gradient flow while

reducing computation. The importance-based layer ranking ensures that critical layers are preferentially retained at lower budgets.

Hybrid Architecture Considerations. For hybrid models combining Mamba and attention, the elastic formulation must respect the structural constraints of both components. Mamba layers require group-aware pruning and channel consistency to preserve SSM computation and attention layers require head-wise selection. The router jointly optimizes selections across both layer types and all width dimensions to discover architectures that balance the complementary strengths of Mamba’s efficient sequence processing and attention’s contextual reasoning capabilities.

Elastic Training Router Architecture and Design

For each dynamic dimension 𝑘 ∈ {emb,mamba,attn_head,ffn,depth}, we introduce a dedicated router network that performs architecture search over the target configuration space. Each router consists of two fully connected layers with leaky ReLU activation applied between them.

Router Input Representation. The input to router 𝑘 is a one-hot encoded vector representing the target compression level:

u(𝑘) = eℓ ∈ R𝑛

targets (6)

where eℓ is the ℓ-th standard basis vector and 𝑛targets is the number of target model configurations.

Router Architecture. Each router is parameterized as:

h(𝑘) = LeakyReLU(︁W1(𝑘)u(𝑘) + b(1𝑘))︁ (7)

- where W1(𝑘) ∈ R𝑑

router×𝑛targets and b(1𝑘) ∈ R𝑑

router are the first layer weights and bias, and 𝑑router is the intermediate hidden dimension. The router output is:

z(𝑘) = W2(𝑘)h(𝑘) + b(2𝑘) (8)

- where W2(𝑘) ∈ R𝑛

out×𝑑router and b(2𝑘) ∈ R𝑛

(𝑘)

(𝑘)

out. The

output dimension 𝑛(out𝑘) varies by axis and configuration mode.

Router Output Dimensions. For homogeneous configuration modes where all instances of a component type share the same compression ratio:

𝑛(emb)out = |ℰ| 𝑛(mamba,hom)out = |ℳ| 𝑛(attn,hom)out = |𝒜|

𝑛(ffn,hom)out = |ℱ| 𝑛(depth)out = 𝑁

(9)

where |ℰ|,|ℳ|,|𝒜|,|ℱ| denote the cardinality of target configuration sets for each dimension. For heterogeneous configuration modes where each layer can independently select its compression ratio:

𝑛(mamba,het)out = |ℳ| × 𝑁M

𝑛(attn,het)out = |𝒜| × 𝑁A 𝑛(ffn,het)out = |ℱ| × 𝑁F

(10)

where 𝑁M,𝑁A,𝑁F denote the total counts of Mamba, attention, and FFN layers respectively. Embedding remains homogeneous as its channels are globally indexed.

#### Loss Formulation

The router outputs are passed through GumbelSoftmax with temperature 𝜏 to produce soft probability distributions over configuration choices. At each training iteration, we sample from these distributions to obtain relaxed discrete selections that enable gradient flow to the router parameters.

Gumbel-Softmax Relaxation. Let z(𝑘) denote the raw logits output by router 𝑘. The GumbelSoftmax relaxation is:

exp(︂

)︂

z(𝑖𝑘)+𝑔𝑖 𝜏

𝜋𝑖(𝑘) =

)︂ (11)

𝑗 exp(︂

z(𝑗𝑘)+𝑔𝑗 𝜏

∑︀

where 𝑔𝑖 ∼ Gumbel(0,1) are i.i.d. Gumbel noise samples and 𝜏 > 0 is a temperature parameter that is annealed from high values (soft exploration) to low values (sharp decisions) during training.

Router Objective Function. The router is jointly trained to optimize a resource-aware objective that maps selected configurations to hardware and com-

putational constraints. Let 𝑎𝑘 ∈ {1,...,𝑛(out𝑘)} denote the configuration selected by router 𝑘. The resource

cost of configuration 𝑎𝑘 is denoted 𝒞(𝑘)(𝑎𝑘), where

possible cost metrics include parameter count, memory usage (including model parameters, KV cache, Mamba cache, and activations), latency, or throughput. The router loss is:

ℒrouter = ⃦𝒞(𝑘)(𝑎𝑘) − 𝒞^(𝑘)⃦ (12)

where 𝒞^(𝑘) is the target constraint for dimension 𝑘. This enables the router to autonomously search through the joint architecture space, balancing multiple objectives and discovering Pareto-optimal configurations. The hybrid approach of combining importance-based sorting with learned router policies is particularly beneficial for hybrid architectures where the interplay between Mamba’s linear-time properties and attention’s expressiveness creates nonobvious accuracy-efficiency trade-offs.

#### Versatile Training Options

The model and router are jointly optimized during training, enabling the architecture search to directly respond to task-specific learning signals. The model parameters are updated to minimize the primary loss, while the router parameters are updated to discover configurations that satisfy resource constraints while maintaining model accuracy. The training framework supports multiple loss formulations, allowing flexible combinations depending on the training regime and available teacher models.

Cross Entropy Loss The model can be trained using standard cross-entropy loss over the training corpus without external supervision:

ℒCE = −E(𝑥,𝑦)∼𝒟 [log 𝑝𝜃(𝑦 | 𝑥)] (13)

where 𝒟 is the training dataset, 𝜃 represents model parameters, and 𝑝𝜃(𝑦 | 𝑥) is the model’s predicted probability distribution. This loss can be used independently or combined with other training objectives.

Knowledge Distillation Knowledge Distillation (KD) improves model accuracy by transferring knowledge from a teacher model. Let 𝑝𝜃(𝑥;𝜏) denote the student model’s softmax-normalized logits at temperature 𝜏, and 𝑝𝜑(𝑥;𝜏) denote the teacher’s corresponding distribution. The distillation loss using forward KL divergence is:

ℒKD = 𝐷KL (𝑝𝜑(𝑥;𝜏)‖𝑝𝜃(𝑥;𝜏)) (14)

Trainable Teacher: In this mode, the full-budget model (100% across all dimensions) simultaneously serves as the teacher and is updated during training.

Both student and teacher parameters are optimized jointly:

ℒteacher = ℒKD(𝜃student,𝜃teacher) + 𝛼 · ℒCE(𝜃teacher)

(15) where 𝜃teacher corresponds to model parameters with full budget allocation and 𝛼 > 0 is a weighting factor. This enables the teacher to adapt to the training distribution while providing moving supervision targets. The Cross-Entropy loss is added in this case so that the model doesn’t collapse to itself during self distillation.

Frozen Teacher: In this mode, the teacher model parameters are frozen throughout training and do not receive gradient updates. The teacher can either be the original pre-trained full model or an alternative model architecture:

ℒfrozen = ℒKD(𝜃student,𝜑fixed) (16)

where 𝜑fixed are static teacher parameters. This approach reduces computational overhead and provides stable, consistent supervision throughout training.

Mixed Training Modes The framework supports flexible combinations of these losses. For example, a training run can employ trainable teachers for initial phases (capturing distribution-specific knowledge) and transition to frozen teachers for final stages (stabilizing convergence). Different sub-models (e.g., elastic variants at different compression levels) can simultaneously use different teacher modes and loss combinations, enabling rich multi-objective training scenarios.

#### Final Optimization Target

The joint optimization of the model and router is achieved through a combined objective:

ℒtotal = ℒtask(𝜃) + 𝜆 · ℒrouter(𝜓) (17)

where ℒtask(𝜃) is the primary learning objective (either cross-entropy, knowledge distillation, or their combination), 𝜓 denotes router parameters, and 𝜆 > 0 is a weighting coefficient that balances task accuracy against resource constraints.

The task loss ℒtask directly incorporates the chosen supervision signal—whether from standard language modeling, knowledge distillation from a teacher, or a hybrid of both. Critically, this end-to-end optimization enables the router to make architecture decisions that are aware of the actual training signal (crossentropy, distillation loss, or combined), rather than optimizing purely for zero-shot proxy metrics in posthoc search phases. This tight coupling between NAS and the training objective represents a key distinction

from prior methods such as Minitron and MinitronSSM, which decouple architecture search (performed via importance scoring on frozen checkpoints) from the final training objective. Our approach integrates architecture discovery directly into the learning process, allowing the router to dynamically adapt configurations in response to the loss landscape of the chosen training regime.

Two-Stage Training with Curriculum-Based Sampling

Multi-budget elastic training requires carefully orchestrated data allocation across budget targets to prevent training imbalance and maintain performance across all sub-networks. This is particularly critical for reasoning models where task complexity demands sophisticated architectural trade-offs.

Multi-Budget Training Mechanics. In the multi-budget setting, each training sample is assigned to one of 𝑛𝑏 target budgets, and the corresponding router output determines which subset of parameters participates in the forward pass. This requires careful sampling of data across budgets to ensure balanced learning signals for all model variants. The choice of budget distribution directly influences architecture discovery and performance characteristics of the resulting model family.

The Role of Extended-Context Training for Reasoning. Standard elastic training approaches optimize for general knowledge recovery and parameter efficiency. However, reasoning tasks impose fundamentally different constraints: complex multi-step inference—from mathematical reasoning to code generation—requires substantial token budget for thinking traces and intermediate steps. Short-context training alone is insufficient for developing genuine reasoning capability; the model must adapt its architecture to support extended sequences where reasoning paths unfold. Extended-context training (with sequence length 𝐿2) exposes all elastic variants to problems requiring longer inference chains, forcing the router to discover configurations that maintain coherence and performance across extended contexts. This necessity motivated our two-stage approach: Stage 1 establishes foundational architecture patterns, while Stage 2 enforces reasoning-specific constraints on the final elastic configuration.

Stage 1: Uniform Budget Sampling (Short Context). During the initial short-context phase (sequence length 𝐿1, total tokens 𝑇1), we employ

uniform budget sampling. For 𝑛𝑏 target budgets, each training batch receives equal allocation:

1 𝑛𝑏

𝑝1(𝑏) =

, ∀𝑏 ∈ {1,...,𝑛𝑏} (18)

Uniform sampling ensures all sub-networks receive balanced training signal during router stabilization, allowing architecture discovery without budget-specific bias. This allocation establishes diverse architectural patterns before reasoning becomes the dominant bottleneck.

Stage 2: Curriculum-Based Non-Uniform Sampling (Extended Context). During extendedcontext training (sequence length 𝐿2, total tokens 𝑇2), we transition to non-uniform sampling that prioritizes full-budget models. For 𝑛𝑏 target budgets with sampling weights {𝛼1,𝛼2,...,𝛼𝑛

𝑏} normalized to

∑︀𝑛

𝑖=1 𝛼𝑖 = 1: 𝑝2(𝑏) = 𝛼𝑏, ∀𝑏 ∈ {1,...,𝑛𝑏} (19)

𝑏

The curriculum-based distribution addresses training imbalance observed empirically: uniform sampling in extended-context causes performance degradation in the full model while smaller budgets improve, indicating gradient competition. Non-uniform weighting biases updates toward full-model performance—critical when the full model serves as teacher in frozen distillation—while still training smaller variants. This approach prioritizes long-context reasoning capability across all sub-networks, with weights typically skewed toward larger budgets to prevent collapse of the largest model.

Training Signal Coupling to Architecture Search. The two-stage sampling strategy directly couples multi-budget training to the router’s architecture discovery process. During Stage 1, uniform sampling encourages exploration of diverse configurations across budgets. During Stage 2, non-uniform sampling provides stronger gradients for the full model, guiding the router toward configurations that preserve reasoning capability on extended contexts. This coupling ensures that architecture decisions evolve in response to the actual difficulty of training tasks at each stage, rather than being independently determined by importance scores alone.

#### Implementation

The elastic architecture is instantiated through structured masking applied to the hybrid MambaAttention-MLP model. Rather than modifying network topology or creating distinct sub-networks, we

apply dimension-specific binary masks that dynamically select active components. This masking-based approach enables efficient training of multiple budgets simultaneously while maintaining architectural transparency and enabling straightforward deployment of any sub-network without architectural recompilation.

#### Dynamic Model Formulation

We present a flexible architecture framework for Nemotron Elastic that enables dynamic adjustment of model dimensions during training through a structured masking approach. Our method builds upon the hybrid Mamba-Attention-MLP architecture and extends the elastic training paradigm to support comprehensive width and depth flexibility for hybrid architectures.

A dynamic model is obtained by making the stack of layers dynamic, and then making each layer type dynamic across different dimensions. If the original LLM is defined as 𝑦 = ℒ𝑁0 (𝑥) where ℒ𝑗0(𝑥) = ℒ𝑗0−1(𝑥)+ℒ𝑗(ℒ𝑗0−1(𝑥)), a dynamic layer stack is noted

- as 𝒟 ∘ ℒ𝑁0 where the operator 𝒟 is applied to each layer and makes it dynamic. For example:

𝒟 ∘ ℒ𝑗 = (𝒟 ∘ ℒ𝑗) · 𝛾𝑗 (20)

where 𝛾𝑗 ∈ {0,1} controls layer retention (depth adaptation) and 𝒟 ∘ ℒ𝑗 represents a dynamic Mamba, Attention, or MLP layer.

The dynamic operator 𝒟 applies dimension-specific binary masks m to the output activations of each layer component, enabling selective feature retention (width adaptation):

𝒟(ℒ(𝑥)) = ℒ(𝑥) ⊙ m (21)

where ⊙ denotes element-wise multiplication and m ∈ {0,1}𝑑 is a binary mask vector that determines which dimensions remain active. Depth adaptation is controlled through the binary coefficient vector 𝛾 = [𝛾0,𝛾1,...,𝛾𝑁−1], while width adaptation is managed through dimension-specific masks applied within each layer type.

#### Dynamic Mamba

For Mamba-2 components in the hybrid architecture, we apply group-aware masking following permutationpreserving constraints to maintain structural integrity of state-space computations. The elastic Mamba layer applies the dynamic operator to its output:

𝒟(Mambaℓ(𝑦)) = Mambaℓ(𝑦) ⊙ mmamba (22) where mmamba ∈ {0,1}𝑑

𝑒 is the output mask constructed from dynamic embedding and Mambaspecific constraints.

Dynamic Embedding Mask Operator. The operator ℳemb applies to any activation or weight matrix with the hidden size 𝑑𝑒 as one dimension. For a matrix W ∈ R𝑑

𝑒×𝑘, the masked operation is:

ℳemb(W) = W ⊙ (I𝑒 ⊗ 1𝑘) (23) where I𝑒 ∈ {0,1}𝑑

𝑒 with I𝑒[0 : 𝑖] = 1 and I𝑒[𝑖 + 1 : 𝑑𝑒] = 0 for some 𝑖 ∈ [0,𝑑𝑒], and ⊗ denotes outer product broadcasting across dimension 𝑘. For matrices W ∈ R𝑘×𝑑

𝑒, the mask broadcasts similarly: ℳemb(W) = W⊙(1𝑘 ⊗I𝑒). This operator is applied to layer normalization outputs and all weight matrices interfacing with the embedding dimension.

Dynamic Mamba Mask Operator. The operator ℳmamba applies to matrices where dimensions derive from Mamba heads 𝑚ℎ or head channels 𝑚𝑑. For a matrix W ∈ R𝑓(𝑚

ℎ,𝑚𝑑)×𝑘 where 𝑓 represents a dimension function (typically 𝑓(𝑚ℎ,𝑚𝑑) = 𝑚ℎ · 𝑚𝑑), the masked operation is:

ℳmamba(W) = W ⊙ (I𝑚 ⊗ 1𝑘) (24) where I𝑚 ∈ {0,1}𝑓(𝑚

ℎ,𝑚𝑑) is constructed to satisfy:

I𝑚[𝜑(ℎ,𝑐)] = {︃1 if ℎ ≤ ℎ* and 𝑐 ≤ 𝑐*

(25)

0 otherwise

with 𝜑(ℎ,𝑐) mapping head ℎ and channel 𝑐 to flat index, ℎ* ∈ [0,𝑚ℎ] and 𝑐* ∈ [0,𝑚𝑑] defining active dimensions. This construction preserves group-aware permutation structure: for heads ℎ,ℎ′ ∈ 𝒢𝑔 belonging to group 𝑔, I𝑚[𝜑(ℎ,·)] = I𝑚[𝜑(ℎ′,·)], and maintains head channel consistency: I𝑚[𝜑(·,𝑐)] is uniform across all heads for each channel 𝑐.

Forward Pass. The dynamic Mamba layer processes input through projection matrices following masked layer normalization. First, we apply the embedding mask to the layer norm output:

𝑦ln = ℳemb(LN(𝑦)) (26)

Then, projections are computed from the masked normalized input:

𝑧 = W𝑧 · 𝑦ln, 𝑥 = W𝑥 · 𝑦ln, 𝐵 = W𝐵 · 𝑦ln, 𝐶 = W𝐶 · 𝑦ln, 𝑑𝑡 = W𝑑𝑡 · 𝑦ln

(27) where W𝑧,W𝑥 ∈ R(𝑚

ℎ·𝑚𝑑)×𝑑𝑒, W𝐵,W𝐶 ∈ R(𝑔·𝑑

𝑠)×𝑑𝑒, and W𝑑𝑡 ∈ R𝑚

ℎ×𝑑𝑒. Here, 𝑑𝑒 is the embedding dimension, 𝑚ℎ denotes Mamba heads, 𝑚𝑑 is the head channel dimension, 𝑔 represents the number of Mamba groups, and 𝑑𝑠 is the SSM state dimension.

We apply the Mamba-specific mask to 𝑧, 𝑥, and 𝑑𝑡:

𝑧 ← ℳmamba(𝑧), 𝑥 ← ℳmamba(𝑥),

(28)

𝑑𝑡 ← ℳmamba(𝑑𝑡)

The intermediate activations 𝑥, 𝐵, and 𝐶 undergo causal convolution:

𝑥^ = conv1d(𝑥), 𝐵^ = conv1d(𝐵), 𝐶^ = conv1d(𝐶)

(29) where the conv1d operation on 𝑥^ implicitly respects the Mamba mask structure.

The selective state-space model update computes:

𝑦˜ = SSM(^𝑥,𝐵,^ 𝐶,^ A,D,𝑑𝑡) (30) Followed by gated RMSNorm and output projection:

𝑦pre = W𝑂 · RMSNorm(˜𝑦 ⊙ silu(𝑧)) (31) where W𝑂 ∈ R𝑑

𝑒×(𝑚ℎ·𝑚𝑑).

Finally, both dynamic masks are applied to the layer output:

𝑦out = ℳemb(ℳmamba(𝑦pre)) (32) The complete Mamba layer output is thus

𝒟(Mambaℓ(𝑦)) = 𝑦out. Dynamic Attention

For multi-head attention layers in the hybrid architecture, we apply head-wise and embedding dimension masking to control capacity. The elastic attention layer applies the dynamic operator to its output:

𝒟(Attnℓ(𝑦)) = Attnℓ(𝑦) ⊙ mattn (33) where mattn ∈ {0,1}𝑑

𝑒 is the output mask constructed from dynamic embedding and attention head constraints.

#### Dynamic Attention Head Mask Operator.

The operator ℳattn_head applies to matrices where one dimension derives from attention heads 𝑛ℎ or head dimension 𝑑ℎ. For a matrix W ∈ R𝑓(𝑛

ℎ,𝑑ℎ)×𝑘 where 𝑓(𝑛ℎ,𝑑ℎ) = 𝑛ℎ · 𝑑ℎ, the masked operation is:

ℳattn_head(W) = W ⊙ (I𝑎 ⊗ 1𝑘) (34) where I𝑎 ∈ {0,1}𝑛

ℎ·𝑑ℎ satisfies:

I𝑎[𝜓(𝑛,𝑑)] = {︃1 if 𝑛 ≤ 𝑛* and 𝑑 ≤ 𝑑*

(35)

0 otherwise

with 𝜓(𝑛,𝑑) mapping head 𝑛 and head dimension 𝑑 to flat index, 𝑛* ∈ [0,𝑛ℎ] and 𝑑* ∈ [0,𝑑ℎ] defining active dimensions.

Forward Pass. The dynamic attention layer processes input through masked layer normalization:

𝑦ln = ℳemb(LN(𝑦)) (36)

Projections for query, key, and value are computed as:

Q = ℳattn_head(W𝑄) · 𝑦ln, K = ℳattn_head(W𝐾) · 𝑦ln, V = ℳattn_head(W𝑉 ) · 𝑦ln

(37)

where W𝑄,W𝐾,W𝑉 ∈ R(𝑛

ℎ·𝑑ℎ)×𝑑𝑒, with 𝑛ℎ denoting attention heads and 𝑑ℎ the head dimension. The attention computation follows:

Attn = softmax(︂

)︂V (38)

QK𝑇 √𝑑ℎ

Followed by output projection:

𝑦pre = ℳemb(W𝑂) · Attn (39) where W𝑂 ∈ R𝑑

𝑒×(𝑛ℎ·𝑑ℎ).

Finally, both dynamic masks are applied to the layer output:

𝑦out = ℳemb(ℳattn_head(𝑦pre)) (40)

The complete attention layer output is thus 𝒟(Attnℓ(𝑦)) = 𝑦out.

#### Dynamic FFN

For feed-forward network layers, we apply masking to both embedding and intermediate dimensions. The elastic FFN layer applies the dynamic operator to its output:

𝒟(FFNℓ(𝑦)) = FFNℓ(𝑦) ⊙ mffn (41) where mffn ∈ {0,1}𝑑

𝑒 is the output mask constructed from dynamic embedding and FFN intermediate dimension constraints.

Dynamic FFN Mask Operator. The operator ℳffn applies to matrices where one dimension derives from the FFN intermediate dimension 𝑑int. For a matrix W ∈ R𝑑

int×𝑘, the masked operation is:

ℳffn(W) = W ⊙ (I𝑓 ⊗ 1𝑘) (42) where I𝑓 ∈ {0,1}𝑑

int with I𝑓[0 : 𝑗] = 1 and I𝑓[𝑗 + 1 : 𝑑int] = 0 for some 𝑗 ∈ [0,𝑑int]. For matrices W ∈ R𝑘×𝑑

int, the mask broadcasts similarly.

Forward Pass. The dynamic FFN layer processes input through masked layer normalization:

𝑦ln = ℳemb(LN(𝑦)) (43) The first linear transformation with dynamic masking: ℎ = ℳffn(W1) · 𝑦ln (44)

- where W1 ∈ R𝑑

int×𝑑𝑒 and 𝑑int is the intermediate

dimension. Followed by activation and second linear transformation:

𝑦pre = ℳemb(W2) · 𝜎(ℎ) (45)

- where W2 ∈ R𝑑

𝑒×𝑑int and 𝜎(·) denotes the activation

function. Finally, both dynamic masks are applied to the layer output:

𝑦out = ℳemb(ℳffn(𝑦pre)) (46)

The complete FFN layer output is thus 𝒟(FFNℓ(𝑦)) = 𝑦out.

Depth Adaptation. Layer-wise depth adaptation is achieved through selective layer retention controlled by 𝛾. The set of active layers is:

𝒜 = {𝑗 | 𝛾𝑗 = 1,𝑗 ∈ [0,𝑁 − 1]} (47)

where |𝒜| = 𝑁target specifies the target model depth. Skipped layers are bypassed via residual connections:

𝑦𝑗+1 = {︃𝑦𝑗 + 𝒟 ∘ ℒ𝑗(𝑦𝑗) if 𝛾𝑗 = 1 𝑦𝑗 if 𝛾𝑗 = 0

(48)

This maintains signal propagation while reducing computation. For hybrid architectures, selective layer retention enables leveraging the complementary strengths of Mamba and attention components

- at different model scales.

#### Mask Generation

Mask Generation from Router Output. The router outputs z(𝑘) are processed through GumbelSoftmax to produce relaxed discrete selections. The selected configuration index is determined by 𝑎^𝑘 = arg max𝑖 𝜋𝑖(𝑘), where 𝜋(𝑘) is the Gumbel-Softmax probability distribution. In homogeneous mode, if dimension 𝑘 selects configuration index 𝑎^𝑘, the corresponding target count is 𝑐𝑎^

(e.g., number of active embedding channels, depth, or head counts per layer). The binary mask is then constructed by selecting the top 𝑐𝑎^

𝑘

components according to the importancebased ranking 𝜎(𝑤) or 𝜎(𝑑):

𝑘

I(𝑘) = I[𝜎(𝑘)(𝑗) ≤ 𝑐𝑎^

], 𝑗 = 1,...,size(𝑘) (49)

𝑘

In heterogeneous mode, the router output is reshaped into per-layer selections: z(𝑘) is partitioned into 𝑁X segments of size |𝒳|, where each segment determines the configuration for one layer. Per-layer masks are constructed similarly, allowing each layer to have distinct compression ratios. For depth selection, if the router outputs 𝐿target ∈ [1,𝑁], the top 𝐿target layers from the importance ranking 𝜎(𝑑) are activated via 𝛾𝑗 = 1 for the selected layers.

The generated masks are then applied to the dynamic model operators ℳemb,ℳmamba,ℳattn_head,ℳffn, and depth retention coefficients 𝛾 as defined in the Dynamic Model Formulation section, enabling the model to dynamically adjust capacity.

Mask Integration Strategies. The GumbelSoftmax probabilities provide differentiable signals for router optimization. We support two mask integration modes:

- Mode 1: Hard Selection via Argmax Logits. The

discrete selection is obtained by ^𝑖𝑘 = arg max𝑖 𝜋𝑖(𝑘), and a hard mask is applied using the corresponding

logit:

I(train𝑘) = z^(𝑖𝑘)

𝑘

· I^𝑖

𝑘

(50)

This directly applies the mask from the selected configuration, scaled by its logit magnitude to provide task-relevant gradient signals.

- Mode 2: Soft Masking via Probabilistic Combination. Alternatively, masks from all candidate configurations are combined proportionally to their probabilities:

∑︁

I(train𝑘) =

𝜋𝑖(𝑘) · I𝑖 (51)

𝑖

During training, this soft mask is applied in the dynamic operators, allowing gradients to flow through all configuration options. At inference time, the discrete mask corresponding to ^𝑖𝑘 from the argmax mode is used and the logit, z^(𝑖𝑘)

is set to 1.

𝑘

#### Elastic Model Deployment

A key advantage of the elastic architecture is the ability to extract multiple model variants from a single trained checkpoint without requiring separate training or fine-tuning. This is achieved through a learned slicing mechanism that leverages the router module trained during the elastic training phase.

After training converges, the router has learned optimal budget-aware decisions for every layer and component (attention heads, Mamba, FFN, embeddings). At deployment time, to extract a model for any target budget 𝐵 that was seen during training, we invoke the router with the budget specification. The

router’s learned decisions are used to determine which components should be pruned from the full model. These components are then permanently removed (sliced out) from the checkpoint, effectively extracting a nested sub-network that corresponds to the desired parameter count.

Formally, given a trained full model with parameter set Θmax and a target budget 𝐵 ∈ ℬ (where ℬ is the set of budgets used during training), the router ℛ produces a pruning specification that identifies the parameters to retain. The sliced model parameters are then:

Θ𝐵 = {𝜃 ∈ Θmax : 𝜃 is retained for budget 𝐵}

This zero-shot slicing operation is computationally negligible and produces an inference-ready model immediately, with no retraining, fine-tuning, or additional distillation required. Crucially, any budget 𝐵 ∈ ℬ—whether the largest, smallest, or any intermediate size explored during training—can be deployed directly from the single full-model checkpoint.

The practical benefit is substantial: practitioners need to deploy and maintain only a single full-size model checkpoint, yet at inference time can select any of the trained budget variants on-the-fly without cost. This enables dynamic model selection based on per-request latency or resource constraints. Furthermore, all extracted variants share the same learned representations and architectural decisions, ensuring consistency across the model family and eliminating the need for separate fine-tuning or calibration for each size.

### Experiments and Results

We evaluate Nemotron Elastic by compressing the NVIDIA Nemotron Nano V2 12B hybrid model [14] across both base and reasoning variants. We simultaneously target two nested models: a 9B, and a 6B model, representing 25% and 50% compression, respectively. This multi-target setting showcases the Nemotron-Elasticibility of our elastic framework to serve multiple deployment scenarios from a single trained model.

#### Experimental Setup

Training data. All experiments utilize the same compression data blend that was used to train Nemotron NanoV2 9B (both base and reasoning variants) [14]. This dataset is employed for both importance estimation of network components, and knowledge distillation-based retraining. Using this standardized data blend ensures fair comparison with the Minitron-SSM baseline [15] and maintains consistency across base and reasoning model variants.

Evaluation tasks. We evaluate Nemotron-Elastic across a comprehensive suite of downstream reasoning and knowledge benchmarks. For general knowledge and language understanding, we use MMLUPro [16] (college-level multiple-choice reasoning) and GPQA [17] (graduate-level science questions). For mathematical and algorithmic reasoning, we employ MATH-500 [18] (pre-calculus through competitionlevel mathematics), AIME-2024 and AIME-2025 [19] (American Invitational Mathematics Examination), and LiveCodeBench v5 [20] (code generation and problem-solving). All evaluations use pass@1 metrics with reasoning enabled, averaging results over 4 to 16 shots as appropriate for each benchmark. This diverse evaluation set allows us to assess the quality-efficiency tradeoff of Nemotron-Elastic against baseline compression methods.

Nested compression. We simultaneously train three nested models (6B, 9B, and 12B) from a single 12B parent architecture using multi-budget elastic compression. The 12B model serves as the frozen teacher model for knowledge distillation, providing stable supervision signals throughout training. As described in the Two-Stage Training with CurriculumBased Sampling subsection of the Methodology section, training proceeds in two stages: an initial shortcontext phase (sequence length 8192) followed by an extended-context phase (sequence length 49152).

Hyperparameters and training setup. For importance estimation (see the Importance Estimation and Model Preparation of the Methodology section), we process 1024 calibration samples with a sequence length of 8192. Knowledge distillation training is conducted in two phases:

- Phase 1 (Short Context): Batch size 1536, sequence length 8192, trained for approximately 65B tokens.
- Phase 2 (Extended Context): Batch size 512, sequence length 49152, trained for approximately 45B tokens.

Model parameters are optimized at a learning rate 9e5, while router parameters are optimized at 1e-2. A 60-step linear learning rate warmup is applied to both. The Gumbel-Softmax temperature 𝜏 is initialized at 1.0 and annealed to 0.05. The router weight 𝜆 is set to 1.0, and the linear scaling coefficient for router logits is initialized at 1.0 and linearly increased to 10.0. The router intermediate hidden dimension is 𝑑router = 256.

Budget sampling strategy. For our NemotronElastic family, we target three nested budgets (6B,

9B, 12B). During the short-context phase, we employ uniform budget sampling:

1 3

𝑝(budget) =

for each of {6B,9B,12B}

This ensures that each budget receives an equal training signal, with approximately one-third of each batch assigned to each model variant.

In the extended-context phase, we transition to weighted non-uniform sampling to prevent accuracy degradation in larger models:

𝑝(12B) = 0.5, 𝑝(9B) = 0.3, 𝑝(6B) = 0.2

The non-uniform distribution biases training toward the full-budget model, addressing an observed empirical phenomenon: under uniform sampling in extendedcontext training, the 12B model’s accuracy substantially degrades while the 6B model improves, indicating a training imbalance. The adjusted weighting recovers this balance, allowing all model variants to maintain strong performance.

#### Results

Our multi-budget elastic compression strategy yields three model variants from a single training run, each operating at different parameter budgets while sharing a common foundation. As shown in Table 1, the Nemotron-Elastic-12B model achieves performance comparable to NanoV2-12B on most reasoning benchmarks, achieving an average score of 77.41 compared to 77.38 for NanoV2-12B, despite the complexity of simultaneously optimizing three nested budget targets. Notably, the two-stage training approach with adjusted budget sampling prevents accuracy degradation in larger models that would occur under naive uniform sampling. The extended-context phase training (49k sequence length) demonstrates that the router can adapt architecture decisions to support longer contexts while maintaining multi-budget compatibility. The ability to derive three distinct model deployments from a single training process provides significant practical advantages: a unified model infrastructure can serve heterogeneous hardware constraints and latency requirements through dynamic budget selection without retraining or managing multiple checkpoints.

Cost savings. As shown in Tables 2 and 3, Nemotron Elastic achieves substantial reductions in both training token requirements and deployment memory compared to prior compression approaches. These savings become increasingly significant as model family size grows, demonstrating the practical advantages of elastic training over sequential compression methods.

Training token efficiency. A key advantage of Nemotron Elastic is the elimination of exploratory knowledge distillation runs required by prior methods such as Minitron [4] and Minitron-SSM [15]. These methods perform architecture search by pruning and distilling candidate configurations to identify optimal architectures for each target size, then perform final knowledge distillation with the selected architecture. This two-phase approach incurs substantial token costs that scale linearly with the number of models in the family: each model size requires both exploratory search runs and final distillation.

In contrast, Nemotron Elastic performs end-to-end router-guided architecture search during a single elastic training run, where all target budgets are optimized simultaneously. The router learns to select optimal configurations for each budget as part of the unified training objective, eliminating the need for separate exploratory runs. Table 2 compares training token requirements for deriving 6B and 9B models from a 12B parent.

For prior methods like Minitron-SSM, token cost scales as:

TokensMinitron(𝑛) = 𝑛 · (Tokensexplore + TokensKD)

(52) where 𝑛 is the number of target model sizes. In contrast, Nemotron Elastic requires:

TokensElastic(𝑛) = Tokenselastic-KD ≈ constant (53)

This constant-cost property stems from simultaneous multi-budget optimization: all nested sub-networks share gradient information and are trained together, with marginal overhead for additional target budgets.

Deployment memory efficiency. Elastic models with nested weight-sharing provide significant memory advantages for deployment scenarios requiring multiple model sizes. Since all sub-networks share the same parameter space with only routing metadata differentiating them, deploying all budget variants requires memory equivalent to the largest model alone. In contrast, traditional compression methods produce separate checkpoints for each model size, requiring cumulative storage.

The memory advantage scales with family size. For prior approaches, memory requirements scale linearly:

∑︁𝑛

MemorySeparate(𝑛) =

Size(Model𝑖) (54)

𝑖=1

For elastic models with nested weight-sharing: MemoryNested(𝑛) = Size(Modelmax) + 𝜖router (55)

|Model<br><br>|Math-500 AIME-2024 AIME-2025 GPQA LiveCodeBench MMLU-Pro<br><br>|Average|
|---|---|---|
|Nemotron-Elastic-6B Nemotron-Elastic-9B Nemotron-Elastic-12B<br><br>|96.50 77.64 68.13 53.78 60.95 66.65<br><br>97.25 80.26 75.42 62.50 66.82 73.45 97.70 83.44 75.83 63.25 68.01 76.20<br><br><br>|70.61 75.95 77.41|
|NanoV2-9B NanoV2-12B<br><br>|97.30 80.89 71.43 63.01 67.30 73.61 97.50 82.90 72.50 65.28 67.61 78.47<br><br>|75.99 77.38|
|QWen3-8B<br><br>|96.3 75.83 69.31 59.61 59.5 75.50<br><br>|72.68|

- Table 1 | Multi-budget nested compression results on comprehensive reasoning benchmarks. All three Nemotron-Elastic variants (6B, 9B, 12B) are obtained from a single training run with a frozen 12B teacher. Nemotron-Elastic-12B achieves competitive performance (77.41) compared to NanoV2-12B baseline (77.38), while simultaneously enabling efficient 9B and 6B deployments.

|Method|Model Sizes|Exploratory Final|Total|
|---|---|---|---|
|NanoV2 Pretraining NanoV2 Compression (Minitron-SSM) Nemotron Elastic|6B + 9B 6B + 9B 6B + 9B<br><br>|0 B 40 T 480 B 270 B<br><br>0 B 110 B<br><br>|40 T 750 B 110 B|

- Table 2 | Token budget comparison for deriving 6B and 9B models. Nemotron Elastic eliminates exploratory runs and requires only a single elastic distillation phase, achieving around 7X token reduction compared to Minitron-SSM (NanoV2 Compression). Note: Token budgets for Minitron-SSM 6B, pretraining 9B, and 6B NanoV2 models are estimated based on the token counts for pretraining and compressing the NanoV2 12B model [14].

|Config|Models Memory|
|---|---|
|Nemotron Elastic NanoV2|6B + 9B + 12B 24 GB<br><br>9B + 12B 42 GB|

- Table 3 | Deployment memory comparison (BF16 weights). Despite storing three models, Nemotron Elastic uses 43% less memory than NanoV2’s two models.

sub-models, while extended-context improves the long context reasoning capability of the model, necessary for achieving competitive results on reasoning benchmarks.

We evaluate the impact of sampling strategy on downstream performance:

The ablation demonstrates that adjusted sampling substantially improves performance for the full-budget model. For instance, on AIME-2025, the 12B model gains 3.54 percentage points with adjusted sampling, while maintaining competitive performance on other budgets. This suggests that multi-budget training requires careful load balancing to prevent negative transfer between budget targets.

where 𝜖router < 0.02·Size(Modelmax) represents router parameter overhead (typically <1 GB). The nested architecture is particularly valuable for edge deployment scenarios where multiple model sizes must be available to handle varying workloads or user-selected quality-latency tradeoffs.

#### Effects of Two-Stage Training

The necessity of two-stage training is demonstrated through comparisons between Stage 1 (short-context) and Stage 2 (extended-context) performance:

The results in Table 4 reveal a clear pattern: Stage 2 extended-context training delivers disproportionate improvements on complex reasoning benchmarks (AIME-2025), especially for smaller models. The 6B model gains 19.8% on AIME-2025, while the 12B model gains 4.0%, indicating that smaller models particularly benefit from extended-context adaptation for multi-step reasoning. These gains justify the twostage curriculum: short-context training stabilizes the router and helps initial recovery of the compressed

Impact of Budget Sampling Strategy. We investigate the effect of budget sampling distribution through an ablation study comparing uniform budget allocation against our adjusted non-uniform sampling (Table 5). Results demonstrate that uniform sampling leads to performance imbalance during extendedcontext training: the 12B model’s accuracy degrades significantly on challenging benchmarks, while smaller variants remain competitive. Our adjusted weighting (𝑝(12B) = 0.5,𝑝(9B) = 0.3,𝑝(6B) = 0.2) recovers fullmodel performance by prioritizing gradients toward the largest variant, where reasoning capability demands greater architectural sophistication. The 12B model shows substantial improvements across multiple benchmarks, while smaller variants maintain sta-

|Model (Benchmark)|Performance Stage 1 Stage 2<br><br>|Absolute Gain<br><br>|Relative Improvement|
|---|---|---|---|
|Nemotron-Elastic-6B (Math-500) Nemotron-Elastic-6B (AIME-2025) Nemotron-Elastic-6B (GPQA)<br><br>|95.15 96.50 56.88 68.13 49.12 53.78|+1.35 +11.25 +4.66<br><br>|+1.4% +19.8% +9.5%|
|Nemotron-Elastic-9B (Math-500) Nemotron-Elastic-9B (AIME-2025) Nemotron-Elastic-9B (GPQA)<br><br>|97.13 97.25 68.75 75.42 59.43 62.50|+0.12 +6.67 +3.07<br><br>|+0.1% +9.7% +5.2%|
|Nemotron-Elastic-12B (Math-500) Nemotron-Elastic-12B (AIME-2025) Nemotron-Elastic-12B (GPQA)<br><br>|97.27 97.70 72.92 75.83 62.50 63.25<br><br>|+0.43 +2.91 +0.75|+0.4%<br><br>+4.0%<br><br>+1.2%<br>|

- Table 4 | Two-stage training improvements across model sizes and benchmarks. Stage 2 (extended-context) provides substantial gains on reasoning benchmarks, particularly on AIME-2025, where smaller models benefit significantly (6B: +19.8%, 9B: +9.7%). These improvements demonstrate that reasoning tasks require extended-context training to achieve competitive performance, validating the two-stage approach.

|Model|Math-500 Uniform Adjusted<br><br>|AIME-2025 Uniform Adjusted<br><br>|GPQA Uniform Adjusted|
|---|---|---|---|
| | | | |
|Nemotron-Elastic-6B Nemotron-Elastic-9B Nemotron-Elastic-12B<br><br>|96.40 96.50<br><br>97.40 97.25 97.33 97.70<br><br><br>|67.71 68.13 75.00 75.42 72.29 75.83|55.30 53.78 62.75 62.50 61.11 63.25<br><br>|
|NanoV2-9B NanoV2-12B<br><br>|97.30 97.50|71.43 72.50<br><br>|63.01 65.28|

- Table 5 | Budget sampling ablation. Adjusted non-uniform sampling (0.5, 0.3, 0.2 for 12B, 9B, 6B) achieves better balance across model sizes, particularly improving 12B accuracy on challenging benchmarks (AIME-2025:

+3.54%, GPQA: +2.14%) compared to uniform sampling.

ble performance. This ablation confirms that budgetaware curriculum design is essential for balanced multi-target elastic compression, and that NemotronElastic-12B achieves competitive performance relative to baseline NanoV2 models while enabling zero-shot deployment across all budget variants.

### Related Work

Model Compression and Pruning. Structured pruning has emerged as a powerful technique for LLM compression [21, 22, 5]. Recent work combines pruning with knowledge distillation for accuracy recovery [4], achieving strong results on Transformer models. Group-aware SSM pruning preserves structural constraints critical for sequence modeling while enabling hybrid model compression [23]. Unfortunately, these approaches require separate distillation for each target size.

Hybrid SSM-Transformer Models. Hybrid architectures combining Transformers with SSMs have shown promise for efficient long-context modeling [8, 9, 10, 11, 12]. Nemotron-H [12] replaces 92% of attention layers with Mamba2 blocks, achieving 3× inference speedup. Concurrent compression work [15]

introduces group-aware Mamba pruning but requires separate distillation per model size.

Elastic and Nested Architectures. MatFormer [7] and Flextron [6] pioneered nested weight-sharing for Transformers, training multiple sub-networks simultaneously. Extending the MatFormer methodology to SSMs, MatMamba [13] introduces Matryoshka-style sub-block architecture for Mamba layers. MatFormer introduces MixnMatch heuristics for sub-network selection, while Flextron adds input-adaptive routing for attention and MLP dimensions. However, neither supports: (1) hybrid Mamba-Attention architectures, (2) reasoning-focused two-stage training with extended context, or (3) heterogeneous layer-wise architecture selection via end-to-end learned routing. Google’s Gemma 3n [24] recently demonstrated MatFormerstyle nested models with conditional parameter loading, validating the practical deployment value of elastic architectures. Our work extends these foundations to reasoning models and hybrid architectures.

Reasoning Model Training. Reasoning-capable LLMs generate extended thought chains for complex problem-solving [25, 26], requiring long-context support for intermediate steps. Prior work on reasoning model optimization focuses on prompting strategies

or reinforcement learning from reasoning traces [27], but does not address architectural efficiency or elastic deployment. We demonstrate that reasoning models have fundamentally different training requirements—specifically, extended-context training (49K tokens) is critical for maintaining reasoning performance in compressed variants, a requirement not present in standard LLM compression.

### Conclusions

This paper has presented Nemotron Elastic, the first elastic training framework for reasoning-capable LLMs. We demonstrate that elastic compression of reasoning models requires fundamentally different approaches than standard LLM compression, with extended-context training playing a critical role in preserving reasoning performance across deployment scales. Nemotron Elastic achieves strong results: deriving a model family from a single 12B parent requires only 110B training tokens—a 360x reduction versus training from scratch and a 7x reduction compared to sequential compression. This efficiency is achieved without compromising accuracy or introducing memory overhead, and having a constant memory footprint at deployment against the number of models in the family. During deployment, all the nested submodels can be extracted from the biggest model using zero-shot slicing. Our approach makes elastic reasoning model training practical for organizations with modest computational budgets. Future directions for this work include scaling to larger model families, task-specific architecture selection, dynamic inferencetime routing, and integration with quantization for extreme parameter reduction.

### Acknowledgments

We would like to thank our colleagues and leaders at NVIDIA for their valuable input and support, including Akhiad Bercovich, Alex Fit-Florea, Joey Conway, Jonah Alben, Jonathan Cohen, Luis Vega, Michael Lightstone, Nave Assaf, Oleksii Kuchaiev, Ran Zilberstein, Terry Kong, Udi Karpas, and Zijia Chen.

### References

- [1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [2] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. ArXiv, abs/2307.09288, 2023.
- [3] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [4] Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, et al. Compact language models via pruning and knowledge distillation. arXiv preprint arXiv:2407.14679, 2024.
- [5] Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. Sheared llama: Accelerating language model pre-training via structured pruning. arXiv preprint arXiv:2310.06694, 2023.
- [6] Ruisi Cai, Saurav Muralidharan, Greg Heinrich, et al. Flextron: Many-in-one flexible large language model. arXiv preprint arXiv:2406.10260, 2024.
- [7] Sneha Kudugunta, Aditya Kusupati, Tim Dettmers, et al. Matformer: Nested transformer for elastic inference. arXiv preprint arXiv:2310.07707, 2023.
- [8] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.
- [9] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.

- [10] Opher Lieber, Barak Lenz, Hofit Bata, et al. Jamba: A hybrid transformer-mamba language model. arXiv preprint arXiv:2403.19887, 2024.
- [11] Paolo Glorioso, Quentin Anthony, and Yury Tokpanov. Zamba: A compact 7b ssm hybrid model. arxiv preprint arXiv:2405.16712, 2024.
- [12] Aaron Blakeman, Aarti Basant, et al. Nemotronh: A family of accurate and efficient hybrid mamba-transformer models. arXiv preprint arXiv:2504.03624, 2025.
- [13] Abhinav Shukla, Sai Vemprala, Aditya Kusupati, and Ashish Kapoor. MatMamba: A Matryoshka State Space Model, 2024.
- [14] NVIDIA Nemotron Nano. Efficient hybrid mambatransformer reasoning model. arXiv preprint arXiv:2508.14444, 2025.
- [15] Ali Taghibakhshi, Sharath Turuvekere Sreenivas, Saurav Muralidharan, et al. Minitron-SSM: Efficient Hybrid Language Model Compression through Group-Aware SSM Pruning. arXiv preprint arXiv:2504.11409, 2025.
- [16] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlupro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024.
- [17] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.
- [18] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [19] Mathematical Association of America. American invitational mathematics examination. https:// www.maa.org/math-competitions/aime, 2024.
- [20] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.
- [21] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llmpruner: On the structural pruning of large language models. Advances in neural information processing systems, 36, 2023.
- [22] Saleh Ashkboos, Maximilian L Croci, Marcelo Gennari do Nascimento, et al. Slicegpt: Compress large language models by deleting rows and columns. arXiv preprint arXiv:2401.15024, 2024.

- [23] Ali Taghibakhshi, Sharath Turuvekere Sreenivas, Saurav Muralidharan, Marcin Chochowski, Yashaswi Karnati, Raviraj Joshi, Ameya Sunil Mahabaleshwarkar, Zijia Chen, Yoshi Suhara, Oluwatobi Olabiyi, et al. Efficient hybrid language model compression through group-aware ssm pruning. arXiv preprint arXiv:2504.11409, 2025.
- [24] Google AI. Gemma 3n model overview. https:// ai.google.dev/gemma/docs/gemma-3n, 2024.
- [25] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.
- [26] Shunyu Yao, Dian Yu, Jeffrey Zhao, et al. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36, 2023.
- [27] Hunter Lightman, Vineet Kosaraju, Yura Burda, et al. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

