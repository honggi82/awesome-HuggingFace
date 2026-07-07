### CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing

# arXiv:2603.08589v1[cs.CV]9Mar2026

Yucheng Wang* Zedong Wang* Yuetong Wu Yue Ma Dan Xu The Hong Kong University of Science and Technology

{ywangls, zwangmw, ywufe}@connect.ust.hk mayuefighting@gmail.com danxu@cse.ust.hk

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Object Removal Object Replacement

Instruction-driven Editing Subject-driven Editing

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

- A
- B

[Figure 13]

[Figure 14]

[Figure 15]

A+B (ref+base) B+A (ref+base)

Human + Human Contextual Editing Human + Object Contextual Editing Editing with Style Transfer

Figure 1. CARE-Edit routes diffusion tokens to heterogeneous experts conditioned on text, mask, and reference signals, enabling high-fidelity editing across diverse tasks: instruction-driven modification, subject removal and replacement, scene reconfiguration, and style transfer.

#### Abstract

user-defined masks for precise spatial guidance; (ii) the router applies sparse top-K selection to dynamically allocate computation to the most relevant experts; (iii) a Latent Mixture module subsequently fuses expert outputs, coherently integrating semantic, spatial, and stylistic information to the base images. Experiments validate CARE-Edit’s strong performance on contextual editing tasks, including erasure, replacement, text-driven edits, and style transfer. Empirical analysis further reveals task-specific behavior of specialized experts, showcasing the importance of dynamic, conditionaware processing to mitigate multi-condition conflicts. The project page and source code are available at LINK.

Unified diffusion editors often rely on a fixed, shared backbone for diverse tasks, suffering from task interference and poor adaptation to heterogeneous demands (e.g., local vs global, semantic vs photometric). In particular, prevalent ControlNet and OmniControl variants combine multiple conditioning signals (e.g., text, mask, reference) via static concatenation or additive adapters which cannot dynamically prioritize or suppress conflicting modalities, thus resulting in artifacts like color bleeding across mask boundaries, identity or style drift, and unpredictable behavior under multicondition inputs. To address this, we propose ConditionAware Routing of Experts (CARE-Edit) that aligns model computation with specific editing competencies. At its core, a lightweight latent-attention router assigns encoded diffusion tokens to four specialized experts–Text, Mask, Reference, and Base–based on multi-modal conditions and diffusion timesteps: (i) a Mask Repaint module first refines coarse

#### 1. Introduction

Diffusion-based models have radically transformed image editing, making diverse tasks like localized object replacement, global style adjustment, and text-driven content insertion more accessible and with higher quality [10, 34, 36, 39]. However, most unified editors process all edits with a fixed,

*Equal contribution. Corresponding author.

shared backbone, which struggles to adapt to heterogeneous demands in practice. This could potentially result in artifacts such as color bleeding at mask boundaries, identity or style drift when using reference, or inconsistent behavior when multiple, conflicting conditions must be satisfied [43, 44].

One of the key reasons for this limitation lies in the static fusion of control signals. Popular methods like ControlNet and OmniControl-style variants integrate multi-modal inputs (e.g., text prompts, masks, reference images, or sketches) via simple concatenation or additive adapters attached to the backbone [26, 41, 56]. While effective for single-condition guidance, it is not condition-aware, i.e., it cannot allocate the finite model capacity adaptively to multiple, heterogeneous input conditions. As such, competing signals might not be properly resolved, inducing issues appeared in Figure 4: text semantics may override mask constraints, reference identity or style may be misapplied, and global adjustments can spill into regions that should be preserved. More importantly, the importance of these signals changes over the diffusion trajectory, from semantic layout in early steps to boundary refinement and style consistency in later ones [9, 10, 27], yet static methods offer little mechanism to adapt this balance.

In this paper, we present CARE-Edit, a Condition-Aware Routing of Experts framework to tackle above challenges. Instead of forcing all signals through one shared pathway, a lightweight latent-attention router conditions on the prompt, mask statistics, reference features, and diffusion timestep to dynamically dispatch tokens to four specialized experts: (i) Text for semantic reasoning and synthesis, (ii) Mask for spatial precision and boundary refinement, (iii) Reference for identity/style transfer, and (iv) Base for global coherence and background preservation. Sparse top-K routing enables token-wise and timestep-aware prioritization, while a persistent shared expert stabilizes training and prevents routing collapse, following practices in previous studies [7, 17, 33].

To further push its limits, we introduce two complementary designs as in Figure 2. First, we facilitate interactions via Mask Repaint (Sec. 3.3) for precise spatial guidance and Latent Mixture (Sec. 3.4) which coherently fuses expert outputs, propagating semantic cues to spatial refinement while mitigating conflicts. Second, our training curriculum is set to be progressive. The model is initially exposed to basic, single-task training data, then graduates to complex multitask samples. This allows the experts to evolve from generic representations to specialization, mitigating mode collapse and improving generalization over diverse editing behaviors. As such, compared to static-fusion editors like OmniControl, or DiT-adapter variants [30, 41, 56], CARE-Edit provides selective, condition-aware compute that better resolves the conflicts among text, mask, and reference signals (Sec. 4.5).

We evaluate CARE-Edit on instruction-based (Sec. 4.2) and subject-driven (Sec. 4.3) settings, covering diverse tasks such as object erasure, replacement, text-driven edits, and

localized style transfer. CARE-Edit achieves strong results, improving edit faithfulness, boundary cleanliness, and identity/style preservation over unified editors and swapstyle pipelines [1, 19, 29, 53]. Training dynamics and taskexpert relationsip analysis demonstrate the effectiveness of condition-aware processing to reduce task interference.

Our contributions can thus be summarized as follows:

- • We identify static, model capacity-agnostic fusion of conditions as a major source of conflicts in existing image editors, and thus propose CARE-Edit that employs conditionaware routing of experts for dynamic compute allocation.
- • We introduce three complementary modules to maximize CARE-Edit’s fidelity: (i) Mask Repaint to refine arbitrary, user-defined masks for precise spatial control; (ii) Latent Mixture to coherently aggregate expert outputs; and (iii) Routing Select that implements top-K activation, ensuring only the most relevant experts process each specific token.
- • We demonstrate competitive performance of CARE-Edit on diverse editing tasks, with empirical analysis showcasing that dynamic, condition-aware experts are effective for resolving conflicts in multi-condition image editing.

#### 2. Related Work

Instruction-based Editing. Text-driven diffusion image editing can be classified into two groups. (i) Task-specific methods [12, 55, 58]: global refinement via re-sampling (SDEdit [23]); semantic steering via attention or PnP [8] and NTI [25]; instruction-following editing [2]; localized editing via mask discovery [6] or conditional control [41, 56]; Recent pipelines like EMU-Edit [38]. These methods excel under single-signal guidance but struggle with multi-source, conflicting constraints due to static fusion. (ii) Unified editors [24, 42, 52, 57]: systems that consolidate heterogeneous objectives within a single interface, exemplified by ACE++ [22, 45], OmniGen2 [50, 51], and AnyEdit [54]. Despite the progress, resolving multi-condition conflicts remains a significant challenge in contextual image editing.

Subject-driven Editing. Subject-based image editing spans embedding-based or adapter-based methods (e.g., DreamBooth [35], LoRA [11]) that learn subject/style concepts. It risks overfitting or unintended editing outside target regions. Recent methods extend reference conditioning, including BLIP-Diffusion [18], OmniControl [41], UNO [47], and unified editors such as OmniGen2 [50, 51]; earlier subject-centric editors (e.g., MimicBrush [4], AnyDoor [5]) also explore appearance transfer with varying locality and generalization. In contrast, we treat reference guidance as a conditional competency handled by specialized experts.

Mixture-of-Experts for Image Editing. Sparse MoE models scale capacity via routed specialization [7, 17, 33, 37], and the recent diffusion MoE (e.g., EC-DiT [40] with

[Figure 16]

[Figure 17]

[Figure 18]

T 𝜀𝜀text B 𝜀𝜀image

T :Text Prompt B :Base Image R :Reference Image M :Mask

DiT Blocks xN

LoRA

## …

[Figure 19]

[Figure 20]

[Figure 21]

T 𝜀𝜀text B 𝜀𝜀image R 𝜀𝜀image M 𝜀𝜀mask

| | |
|---|---|
| | |
| | |
| | |

[Figure 22]

LatentMixture

Heterogeneous

RoutingSelect

MaskRepaint

- A) Instruction-based Editing
- B) Subject-based Editing

[Figure 23]

Experts

LoRA

DiT Blocks

[Figure 24]

[Figure 25]

[Figure 26]

B 𝜀𝜀image R 𝜀𝜀image

…

[Figure 27]

DiT Blocks xN

LoRA

…

[Figure 28]

[Figure 29]

xN

C) CARE-Edit (Ours)

- Figure 2. Overview of contextual image editing paradigm. (A) Instruction-based editing guides modifications via a text prompt T and a base image B. (B) Subject-based editing uses the base B and a references image R to preserve identity or style. (C) CARE-Edit incorporates all these modalities (T, B, R) and the user-defined mask M in a diffusion transformer (DiT) backbone with condition-aware routing of experts.

adaptive expert-choice routing) shows timestep-aware token routing is effective. Unlike these application of homogenous experts, we employ heterogeneous experts (text, mask, reference, and base) to solve multi-condition conflicts. A timestep-aware router selectively activates these experts along denoising trajectory, which adaptively allocates model capacity for different conditions in contexual image editing.

incorporates (T,B,R) to confine edits to specific regions. CARE-Edit unifies these modalities (T,B,R,M) within a single framework for flexible, condition-aware generation.

Modality Encoders. Each input modality is first mapped to a latent token sequence by a specialized, frozen encoder. The text prompt is processed by a Text Encoder Etext(·), producing contextual embeddings Cp = Etext(T) ∈ RN

t×d, where Nt denotes the number of text tokens and d indicates the feature dimension of each token embedding. The Image Encoder Eimage(·) (e.g., DINO [3] or VAE [15]) extracts latent representations Zb = Eimage(B) for the base image and Zr = Eimage(R) for the reference image. A Mask Encoder Emask(·) converts the spatial mask into aligned latent tokens Zm = Emask(M), which enables explicit region control.

#### 3. Methodology

We propose CARE-Edit, a diffusion-based editor that routes computation to condition-aware experts. As aforementioned, the key challenge in unified image editing is task interference, where conflicting conditions from text, masks, and reference images lead to artifacts like style bleeding or identity loss. Unlike static-fusion approaches that process all conditions with a shared backbone, CARE-Edit performs fine-grained condition-aware routing over a set of heterogeneous experts, each specialized in processing a particular modality or function. They communicate through cross-modal interactions, enabling the model to selectively integrate multi-condition information across the denoising process. This specializethen-fuse manner allows CARE-Edit to dynamically allocate computation, prioritize relevant modalities, and thus potentially mitigate conflicts between competing edit instructions.

Latent Composition and DiT Backbone. These latents are projected to a shared embedding space and concatenated:

h0 = [Cp;Zb;Zr;Zm] ∈ RN×d, where N = Nt +3Nv, where Nv indicates the number of visual tokens from each image- or mask-related latent. This unified token sequence is then propagated through the diffusion transformer (DiT) backbone [30] which is parameterized by θ:

ht = DDiT, θ(h0, t), where t denotes the diffusion timestep.

Fine-Tuning and Optimization. We apply LoRA-style fine-tuning [11] on a pretrained FLUX diffusion model [16] to adapt the DiT backbone [30] to multi-modal conditioning while preserving pretrained generative priors. The model is optimized via the standard denoising diffusion objective as:

##### 3.1. Preliminaries

As shown in Figure 2, different image editing paradigms can be categorized by the input modalities used for conditioning. Let T denote the text prompt, B∈RH×W×3 the base image, R ∈ RH×W×3 the reference image, and M ∈ [0,1]H×W a binary or soft mask indicating editable regions. Four representative settings are considered: Instruction-based editing conditions on (T,B), where textual guidance directs modifications to the base image; Subject-based ones condition on (B,R) to maintain identity or style; Contextual editing

0,ϵ ∥ϵ − ϵθ(ht,t)∥2 ,

Ldiff = Et,h

where ϵ∼N(0,I) denotes the Gaussian noise. This formulation unifies all modalities into a shared latent representation, laying the groundwork for the condition-aware expert routing discussed in the following sections.

Projection Layer with Experts

DiT Blocks xN

[Figure 30]

B sits beside R holding “CARE”

𝐿𝐿𝑚𝑚𝑑𝑑𝑚𝑚 𝐿𝐿𝑙𝑙𝑙𝑙𝑚𝑚𝑑𝑑

𝜀𝜀𝜀𝜀𝜀𝜀𝜀𝜀imageimagemasktext

Heterogeneous Experts

T

𝐶𝐶𝑝𝑝

| |𝑇𝑇<br><br>|
|---|---|
| |𝑍𝑍𝑏𝑏<br><br>|
| |𝑍𝑍𝑟𝑟<br><br>|
| |𝑍𝑍𝑟𝑟<br><br>|
| | |

[Figure 31]

[Figure 32]

[Figure 33]

ℎ𝑡𝑡′ ℎ𝑡𝑡′

| |
|---|

Top-K

+

Text

LatentMixture(LatMix)

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

LinearProjection

MLP MM

ProjectionLayer

RoutingSelect

| |
|---|

[Figure 39]

ℎ𝑏𝑏′ ℎ𝑚𝑚𝑑𝑑𝑚𝑚′

TokenizeFuse

B

| |
|---|

LayerNorm

𝑍𝑍𝑏𝑏

Norm+Mod

Base

ℎ𝑑𝑑𝑓𝑓𝑚𝑚𝑓𝑓′

concatenate

K = 3

+

𝑠𝑠

[Figure 40]

[Figure 41]

Time

ℎ𝑟𝑟′ ℎ𝑟𝑟′

Attention

[Figure 42]

[Figure 43]

Ref

[Figure 44]

R

𝑍𝑍𝑟𝑟

[Figure 45]

ℎ𝑚𝑚′

ℎ𝑚𝑚′

+

𝑠𝑠

Mask

| |
|---|

Policy

| |
|---|

[Figure 46]

[Figure 47]

𝐿𝐿𝑑𝑑𝑑𝑑𝑑𝑑𝑑𝑑

|[Figure 48]<br><br>[Figure 49]<br><br>Freeze<br><br>LoRA|
|---|

𝑀𝑀 𝑡𝑡 𝐿𝐿𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚

|𝑍𝑍 Latent ℎ Tokens<br><br>𝑇𝑇 Task 𝑠𝑠 Time<br><br>|
|---|

[Figure 50]

M

[Figure 51]

ℎ′

𝑍𝑍𝑚𝑚 ℎ

…

Mask Repaint

- Figure 3. CARE-Edit introduces condition-aware specialized experts within the frozen DiT backbone. Given multimodal conditions, inputs are tokenized and projected to heterogeneous expert branches. The router assigns confidence scores and selects the top-K experts to process each token. Expert outputs are normalized, modulated, and fused through the Latent Mixture module, yielding denoised representations h′ refined by Mask Repaint. Only lightweight adapters, the router, and fusion layers are trainable. This enables CARE-Edit to dynamically allocate computation, mitigates conflicts between heterogeneous conditions (e.g., text vs mask) and enables high-fidelity, coherent edits.

##### 3.2. Overview of CARE-Edit

Unified Dimensionality. Despite differences in structure, all experts share the backbone’s input and output dimensionality d, with each defined as fe: RN×d→RN×d. The expert outputs pass through a LayerNorm followed by a Linear projection layer to maintain feature-scale consistency, which is proven significant for residual stability during training.

As illustrated in Figure 3, CARE-Edit is built around three core components: (1) Mask Repaint, which refines the coarse, user-provided masks into spatially accurate ones guided by reference geometry; (2) Latent Mixture, which fuses the reference and base features within the masked region with per-token and per-timestep awareness; and (3) Routing Select, which dynamically activates the top-K modality experts to allocate task-specific capacity during the diffusion process. Together, these enable semantically consistent and spatially coherent multimodal editing in a unified DiT framework.

Token-wise Top-K Routing. After obtaining the latent tokens h′ from each DiT block, CARE-Edit performs token-wise routing to determine which modality experts should process each token. For every token h′i, a router computes a probability distribution πi,e over the four experts by combining local content features and global task context. Following the Mixture-of-Experts formulation in [7, 37], a token-specific key ki = Wkh′i encodes local information, while a global conditioning query q = ϕ(T) summarizes the current editing objective, where T denotes the task-condition embedding representing removal, replacement, text-driven edits, or localized style transfer. The routing temperature τ is gradually annealed during training, and router logits are smoothed with an exponential moving average to reduce variance, which stabilizes dynamic expert selection. As such, the expert-specific logits then can be computed as:

LoRA Fine-Tuning. Following the previous studies like OmniControl [41], we integrate LoRA adapters [11] into the self-attention and projection layers of each DiT block while keeping the pretrained encoders frozen. This configuration, as shown in Figure 3, enables flexible multi-modal modulation and expert routing without fullly retraining the model. Within each DiT block, we denote by h

′

∈RN×d the latent token set immediately before the Projection Layer.

Specialized Experts. CARE-Edit employs four heterogeneous experts corresponding to the text, mask, reference, and base modalities. The text expert performs semantic reasoning and object synthesis through cross-attention with text tokens. The mask expert focuses on spatial precision and boundary refinement guided by the edit mask. The reference expert learns identity- and style-consistent transformations from reference features. The base expert enforces global coherence and background consistency. Each expert functions as a lightweight adapter embedded in a DiT block, introducing modality-specific inductive bias with minimal additional parameters. h

αi,e = MLPe([ki∥q]) + be, (1)

where αi,e denotes the pre-softmax activation (logit) for expert e, and be is a learnable bias that captures prior expert preferences. The softmax-based routing probabilities and Top-K sparsification are jointly expressed as:

exp(αi,e/τ)1[e∈Si] j∈Si exp(αi,j/τ)

, (2)

π˜i,e =

where Si is the index set of the K experts with the highest scores for token i. In practice, K is set to 3, achieving a

′

′

′

′

m represent the tokens from the base, reference, text, and mask experts, respectively.

b, h

r, h

t, and h

favorable balance between representational diversity and computational efficiency. This scheme allows each token to adaptively attend to the most relevant experts (with specific modalities) underthe spatial–semantic–task joint guidance.

Residual Aggregation and Stability. At this stage, each expert refines h′ through their specific compact adapter as:

′

t = h′ + Atext(CrossAttn(h′,T)), (3) h

h

′

b = h′ + Abase(CrossAttn(h′,Zb)), (4) h

′

r = h′ + Aref(FiLM(h′,Zr)), (5) h

′

m = h′ + Amask Conv h′ ⊙ Up(Zm ⊙ Mˆ (t)) , (6)

where A(·) indicates the lightweight convolutional adapters. Cross-attention follows the conditioning paradigm of latent diffusion [30, 34, 46]. To prevent routing collapse, a fixed fraction λshared of tokens is always routed through a shared expert, which preserves the representation continuity:

π˜i,e+ = (1 − λshared)˜πi,e + λshared1[e = share], (7) and the final residual aggregation is computed as:

h′′i = h′i +

e

π˜i,e+ fe(h′i,e) − h′i . (8)

This convex residual fusion aims to stabilize the gradient propagation and thus ensure balanced expert influence.

Overall Training. CARE-Edit is trained jointly with the diffusion reconstruction objective. To encourage balanced expert utilization and stabilize the routing, we add a load-balancing regularizer following prior MoE work [40]:

Lload =

e

1 N i

1 |E|

πi,e −

2

. (9)

##### 3.3. Mask Repaint

The user-defined masks M could misalign with object boundaries, thereby causing artifacts and color bleeding. To address this, Mask Repaint module refines M at each diffusion step t by exploiting geometric correspondence between current latent and reference features. It predicts a soft, boundaryaware mask that adapts to object contours and thus promotes smooth transitions between edited and preserved regions.

Reference–guided Refinement. At timestep t, the module takes the current latent h′t, the reference encoding Zr, and the previous-step predicted mask latent Mˆ (t−1). A residual mask field ∆m is estimated from concatenated features:

∆m = σ W2 Conv [h′(t) ∥Up(Zr)∥Up(Mˆ (t−1))] ,

(10)

where Wm is a projection layer, and σ(·) denotes sigmoid activation. The refined soft mask is then given by:

###### Mˆ (t) = clip M ˆ (t−1) + ∆m, 0, 1 , (11)

which adaptively aligns to boundaries in both h′t and Zr. Since the update operates in latent space, it yields spatially coherent masks without explicit pixel-wise supervision.

Integration with Diffusion. The refined mask Mˆ (t) is fed back into the routing process of the next CARE-Edit diffusion block, modulating the mask and base experts. In practice, it serves as the dynamic spatial prior within Eq. (6), updating the mask interaction Up(Mˆ (t)) that gates token features at the subsequent denoising step. This iterative refinement enables progressively sharper boundary control and seamless region blending across diffusion steps.

Training. Mask Repaint is trained jointly with the diffusion objective using a boundary-consistency loss [14]:

Lmask = ∇Mˆ (t) − ∇Mgt 1 + λsmooth ∇2Mˆ (t) 1, (12)

where Mgt is a ground-truth or pseudo mask. The first term enforces accurate boundary localization, while the smoothness term suppresses spurious oscillations, producing clean and temporally stable mask evolution throughout denoising.

##### 3.4. Latent Mixture

The output of above specialized experts must be coherently aggregated to the final denoised latent. Naive concatenation or averaging could blur local details and weaken modality-specific information. As such, we employ a Latent Mixture module that performs per-token and per-timestep processing based on routing confidence and contextual cues.

′

Token-wise Fusion. At each diffusion step, let h

e denote the output of active experts e. From the normalized routing weights π˜i,e+ (Eq. 7), we can compute a probability distribution map we∈RN×d for each expert satisfying e we = 1. The fused latent h

′

fuse is then obtained through a convex combination of expert outputs, as:

′

h

fuse =

e

′

e, (13)

we ⊙ h

′

where each channel of h

fuse integrates text, semantic, and mask cues according to the router’s attention pattern.

Timestep-adaptive Mixture. To maintain global coherence, we blend h′fuse with the base expert’s output via a learned, timestep-dependent gate. A modulation network computes an adaptive gating coefficient γ(hb,s) from the base-expert feature and the timestep embedding as:

###### γ = σ (Wγ[GAP(hb)||ψ(s)]), (14)

Text Base Ours (T, M) OmniGen2 (T,M) Swap-Anything (M) Flux .1 Pro (T)

Ref UNO (T,M)

Mask

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

| |
|---|

[Figure 58]

[Figure 59]

Exchange(DB) Merge+

A toy is on the sofa

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Merge(DB)

A vase and A bowl are place together

| |
|---|

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Text(DB)

A stuffed dog cuddles a purple can

| |
|---|

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

|[Figure 83]|
|---|

Style(OOD)

A girl stands in the farm

Merge+

- Figure 4. Qualitative comparison of subject-driven contextual editing. Green annotations indicate data belonging to the benchmark, while Blue annotations denote whether mask inputs are provided. CARE-Edit ensures the preservation of subject identity and coherent contexts.

##### 3.6. Training Dataset

where GAP denotes global average pooling and ψ(s) is a sinusoidal timestep encoding. The final latent integrates global background and context-dependent foreground edits:

Our training data is drawn from MagicBrush [55] and OmniEdit [49], supplemented with selections from UNO [47] to broaden object categories and AnyEdit [54] for edit types. To target complex contextual editing with identity preservation, we curate a 20K multi-paired corpus (Image+Mask+Prompt, optional reference) from Subjects200K [41]. It offers samples focusing on diverse objects and humans within shared backgrounds. For mask supervision, we introduce a pipeline to generate precise multi-paired masks for high-quality spatial control. Please refer to Appendix A.1 for more details.

′

′

′

b (15)

mix = (1 − γ)h

h

fuse + γh

This anchors output to the base’s global structure while sharpening attention in regions driven by semantic/mask signals.

Training Objective. The Latent Mixture module is optimized jointly with the overall diffusion objective. To encourage spatial coherence of the mixture maps, we introduce a total-variation regularizer from previous studies [13] as:

Lmix = λtv

e

∥∇we∥1, (16)

where ∇we computes finite-difference spatial gradients of the weight map, and λtv controls the trade-off between mixture smoothness and routing flexibility. Larger λtv enforces stronger spatial continuity in selection, while a smaller one allows sharper expert transitions for fine-grained boundaries.

##### 3.5. Training Loss

CARE-Edit is trained end-to-end by combining the diffusion reconstruction loss (Sec. 3.1) with the three aforementioned auxiliary regularizers: a load-balancing loss Lload (Sec. 3.2), a mask-boundary consistency loss Lmask (Sec. 3.3), and a latent-mixture smoothness loss Lmix (Sec. 3.4) as:

LCARE = Ldiff +λload Lload +λmask Lmask +λmix Lmix (17)

Throughout training, only the expert adapters, router parameters, and lightweight fusion modules are optimized, while the pretrained denoising backbone remains frozen.

#### 4. Experiments

We test CARE-Edit on three mainstream benchmarks, covering (i) instruction-based editing on EMU-Edit [38] and MagicBrush [55], and (ii) subject-driven contextual editing on recent DreamBench++ [31], including its challenging multiobject settings to ensure rigorous evaluation. We first introduce experimental details and evaluation metrics (Sec. 4.1), followed by direct comparisons on both instruction-based (Sec. 4.2) and contexual (Sec. 4.3) tasks. We then present ablation studies (Sec. 4.4) and empirical analysis (Sec. 4.5) that validate CARE-Edit beyond benchmarking results.

##### 4.1. Experimental Setup

Implementation Details. We adopt a DiT-style backbone, interleaving routed layers every 2–3 blocks to balance the overhead and representation power. Each routed layer uses top-K expert routing with K ∈ {2,3,4}, with a shared expert activated for stability and global coherence [7, 17, 33]. The model is first finetuned following Ominicontrol [41],

- Table 1. Quantitative results on EMU-Edit [38] and MagicBrush [55] test sets. All included editors are classified into Task-specific and Unified models. Best and second-best results per metric are marked in bold and underline. ↑ (↓) indicates higher (lower) values are better.

Category Method Venue Backbone

EMU-Edit [38] Test MagicBrush [55] Test CLIPim ↑ CLIPout ↑ L1 ↓ DINO ↑ CLIPim ↑ CLIPout ↑ L1 ↓ DINO ↑

Task-specific Models

PnP [8] CVPR’23 SD1.5 0.521 0.089 0.089 0.304 0.568 0.101 0.289 0.220 Null-Text [25] CVPR’23 SD1.5 0.761 0.236 0.075 0.678 0.752 0.263 0.077 0.664 InstructPix2Pix [2] CVPR’23 SD1.5 0.834 0.219 0.121 0.762 0.837 0.245 0.093 0.767 EMU-Edit [38] CVPR’24 – 0.859 0.231 0.094 0.819 0.897 0.261 0.052 0.879

Unified Models

FLUX.1 Fill [16] HuggingFace FLUX.1 Fill 0.663 0.205 0.176 0.674 0.725 0.235 0.208 0.661 ACE (ACE++) [22, 45] ICLR’26 FLUX.1 Fill 0.831 0.256 0.073 0.802 0.818 0.268 0.042 0.823 OmniGen2 [50, 51] CVPR’25 FLUX.1 Dev 0.865 0.306 0.088 0.832 0.905 0.306 0.055 0.889 AnyEdit [54] CVPR’25 SD1.5 0.866 0.284 0.095 0.812 0.892 0.273 0.057 0.877 CARE-Edit (Ours) – FLUX.1 Dev 0.868 0.313 0.082 0.835 0.894 0.324 0.052 0.885

- Table 2. Quantitative results on DreamBench++ [31]. The best and second-best results are marked in bold and underlined, respectively.

Ours (with mask)

Ours (w/o mask)

OmniGen2 (Flux .1 Dev)

ACE++ (Flux .1 Fill) Flux .1 Dev

Text Base

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

The toy is holding a "CARE" tag

Single-Object Multiple-Object DINO-I ↑ CLIP-I ↑ CLIP-T ↑ DINO-I ↑ CLIP-I ↑ CLIP-T ↑

Human(DB)OOD

Method

DreamBooth [35] 0.552 0.544 0.301 0.359 0.495 0.305 BLIP-Diffusion [18] 0.610 0.649 0.293 0.462 0.592 0.289 OmniControl [41] 0.770 0.704 0.312 0.501 0.641 0.316 UNO [47] 0.782 0.713 0.304 0.508 0.649 0.303 OmniGen2 [50, 51] 0.861 0.784 0.318 0.560 0.713 0.319

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

A girl strolling through a vibrant marketplace

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

a raccoon in the fallen leaves of a forest during autumn

CARE-Edit (Ours) 0.874 0.792 0.325 0.568 0.720 0.327

Animal(DB)

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

and then jointly optimize with sparse routing to specialize experts. We use AdamW optimizer with a learning rate of 1 × 10−4, a batch size of 16, and weight decay of 0.01. All models are trained for 100K iterations on 8× NVIDIA L20 GPUs using a cosine-decay learning rate scheduler.

[Figure 107]

A cherry blossom tree in the Ukiyo-e style

Style(DB)

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Alter the style of the image to painting

Style(AE)

Curriculum Training. We adopt a cross-task curriculum that gradually increases task difficulty, which promotes stable training and guides specialized expert learning. The model is first trained for 40K iterations on basic, single-task samples, and then switches to complex multi-task data for the remaining 60K iterations. This progressive schedule allows the routing layers to evolve from generic representations to specialized functions, mitigating mode collapse and improving generalization over diverse editing behaviors.

Figure 5. Qualitative comparison of instruction-based editing.

120K training data, which is much lower than that of OmniGen2 [50]. On EMU-Edit [38], CARE-Edit yields the best CLIPim, CLIPout, and DINO scores, matching or even surpassing task-specific methods. On MagicBrush [55], CAREEdit hits the highest CLIPout and DINO scores while keeping a competitive L1. The per-category results demonstrate CARE-Edit’s robustness across both local edits (e.g., object removal) and global appearance changes (e.g., style transfer).

Evaluation Metrics. We follow standard protocols to evaluate performance and report image-level subject consistency using DINO-I [28] and CLIP-I [32], and text–image alignment using CLIP-T [32]. For all three metrics, higher scores indicate better subject fidelity and semantic coherence.

Qualitative Results. Figure 5 visually confirms our quantitative results. CARE-Edit produces cleaner, more instructionfaithful edits with sharper boundaries and fewer artifacts than competing editors. Please view Appendix B for details.

##### 4.2. Instruction-based Image Editing

Baselines. We compare CARE-Edit against two classes of baselines: (i) Task-specific methods, including PnP [8], NullText [25], InstructPix2Pix [2], and EMU-Edit [38]; (ii) Unified editors, including FLUX.1Fill [16], ACE/ACE++ [22, 45], OmniGen2 [50, 51], and AnyEdit [54]. All methods are tested using their official checkpoints or configurations.

Quantitative Results. As illustrated in Table 1, CAREEdit achieves competitive performance over unified editors on EMU-Edit [38] and MagicBrush [55] with approximately

##### 4.3. Subject-driven Contextual Image Editing

Baselines. We evaluate subject-driven contextual editing against strong personalization and unified editors, including DreamBooth [35], BLIP-Diffusion [18], OmniControl [41], UNO [47], and OmniGen2 [50, 51]. All methods use official implementations. We follow the evaluation protocol in UNO [47] for the DreamBench++ [31] multi-object setting.

DreamBench++ Results. Table 2 shows that CARE-Edit achieves the best performance across all metrics on DreamBench++ [31] in both single- and multiple-object settings.

Table 3. Ablation studies of CARE-Edit. ↑ indicates higher values are better. The best results per metric are highlighted in bold.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Variant DINO-I ↑ CLIP-I ↑ CLIP-T ↑

| |
|---|

| |
|---|

| |
|---|

w/o Experts 0.485 0.652 0.296 w/o LatentMixture 0.509 0.678 0.301 w/o MaskRepaint 0.523 0.693 0.304 K = 2 0.541 0.707 0.312 K = 4 0.562 0.716 0.325 Full Model (K=3) 0.568 0.720 0.327

T=0 T=30K T=70K T=100K

- Figure 6. Visualization of base expert attention map over iterations.

[Figure 120]

[Figure 121]

- Figure 7. Task-Expert Activation Analysis and Loss Curves.

[Figure 122]

[Figure 123]

activation patterns across four distinct tasks. As shown in Figure 7, different tasks exhibit clear, distinct demands for specific conditions/modalities, which are met by corresponding expert specialization. We observe two key patterns: (i) Base Expert maintains robust activation across all tasks, ensuring consistent global representation of the source image. (ii) Other experts exhibit clear specialization: Mask Expert dominates structure-aware edits (e.g., removal and replacement), while the Reference Expert is heavily activated during style transfer to maintain fidelity. This dynamic, task-aware activation highlights the limitations of static fusion, which struggle to adaptively allocate resources for multi-condition information. It also demonstrates that CARE-Edit achieves the dynamic multi-condition processing it was designed for. Expert Latent Attention. We visualize Base Expert’s latent attention maps to understand how it processes the source image over time. Figure 6 shows a clear evolution: In midphase (30K–100K steps), the updated mask information is injected via Latent Mixture and inter-expert interactions. This leads to progressively sharper, structured attention on masked regions. Interestingly, in the late stage, Base Expert evolves beyond merely copying spatial signals and begins to semantically refine masked areas, adjusting its focus based on the editing intent rather than just spatial cues. The training dynamics (Figure 7, right) corroborate this finding: the mask loss decreases in tandem with the total loss, indicating the model is successfully learning to integrate mask-aware information into a semantically-meaningful representation.

Notably, it slightly but consistently outperforms the recent strong OmniGen2 [50]. This demonstrates that CARE-Edit effectively preserve subject identity and structure, even when accommodating complex multi-object contextual changes.

Qualitative Results. Figure 4 shows that CARE-Edit produces edits with more faithful subject appearance and more coherent backgrounds, reducing artifacts and improving the integration between foreground objects and their surrounding context. Please view Appendix B for more details.

##### 4.4. Ablation Study

We conduct ablation studies on challenging multiple-object setting, where preserving subject identity and contextual consistency is most difficult. Table 3 isolates the contributions of main components (e.g., expert routing, Latent Mixture, Mask Repaint) and the number of activated experts K.

Impact of Core Components. Removing expert routing (w/o Experts) causes large performance drop, indicating that dynamically routing tokens to specialized experts is crucial for handling diverse editing behaviors. Disabling Latent Mixture (w/o Latent Mixture) and Mask Repaint (w/o Mask Repaint) also incur degrades, underscoring their vital roles in aggregating expert outputs and achieving precise edits.

#### 5. Conclusion and Discussion

We present CARE-Edit that addresses multi-condition conflicts by employing heterogeneous experts with efficient routing for versatile, high-fidelity image editing. It improves controllability via masks and references and scales with modest overhead. We will release our code, models, and the dataset to facilitate research in controllable, multimodal editing.

Impact of K in Routing. We observe that setting K = 3 yields optimal results. Using fewer experts may underexpress the model while more experts slightly hurts performance, which we attribute to reduced expert specialization. Note that the variations between K = 2,3,4 are relatively minimal, which demonstrates the robustness of CARE-Edit.

Limitations and Future Work. There are several limitations in this work: (1) The additional hyperparameters (e.g., top-K) inherent to CARE-Edit. (2) While the expert set covers most common tasks and modalities, we aim to explore extending it to handle broader edit types in our future work, potentially through dynamic expert loading or expansion.

##### 4.5. Empirical Analysis

Task-Expert Activation. To investigate the relative contributions of experts for different tasks, we analyze their

#### References

- [1] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. In ICLR, 2019. 2
- [2] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 2, 7
- [3] Mathilde Caron, Hugo Touvron, Ishan Misra, et al. Emerging properties in self-supervised vision transformers. In ICCV,

2021. 3

- [4] Xi Chen, Yutong Feng, Mengting Chen, Yiyang Wang, Shilong Zhang, Yu Liu, Yujun Shen, and Hengshuang Zhao. Zero-shot image editing with reference imitation, 2024. 2
- [5] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In CVPR, 2024. 2
- [6] Guillaume Couairon et al. Diffedit: Diffusion-based semantic image editing with mask guidance. In ICLR, 2023. 2
- [7] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. In JMLR, 2022. 2, 4, 6
- [8] Amir Hertz, Ron Mokady, Jonathan Tenenbaum, et al. Prompt-to-prompt image editing with cross attention control. In ICLR, 2023. 2, 7
- [9] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS Workshop, 2021. 2
- [10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NIPS, 2020. 1, 2
- [11] Edward Hu, Yelong Shen, Phillip Wallis, et al. Lora: Lowrank adaptation of large language models. In ICLR, 2022. 2, 3, 4, 12
- [12] Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing, 2024. 2
- [13] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, 2016. 6
- [14] Hoel Kervadec, Jawad Bouchtiba, Christian Desrosiers, Eric Granger, Jose Dolz, and Ismail Ben Ayed. Boundary loss for highly unbalanced segmentation. In MIDL, 2019. 5
- [15] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014. 3
- [16] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 3, 7, 12, 13
- [17] Dmitry Lepikhin et al. Gshard: Scaling giant models with conditional computation and automatic sharding. In ICLR,

2021. 2, 6

- [18] Junnan Li, Dongxu Li, Hexiang Hu, Zhe Yang, Kaixuan Yao, Boyang Gao, Yinfei Wang, Lisa Amini, and Steven C. H. Hoi. Blip-diffusion: Pre-trained vision-language models for zero-shot image-to-image translation. In NIPS, 2023. 2, 7
- [19] Xingchao Liu, Chengyue Gong, Lemeng Wu, Shujian Zhang, Hao Su, and Qiang Liu. Fusedream: Training-free text-toimage generation with improved clip+gan space optimization,

2021. 2

- [20] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. In SIGGRAPH Asia, 2024. 12
- [21] Yue Ma, Kunyu Feng, Zhongyuan Hu, Xinyu Wang, Yucheng Wang, Mingzhe Zheng, Xuanhua He, Chenyang Zhu, Hongyu Liu, Yingqing He, et al. Controllable video generation: A survey. arXiv preprint arXiv:2507.16869, 2025. 12
- [22] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instruction-based image creation and editing via context-aware content filling,

2025. 2, 7, 13

- [23] Chenlin Meng, Jonathan Ho, Chitwan Saharia, et al. Sdedit: Image synthesis and editing with stochastic differential equations. In ICLR, 2022. 2
- [24] Zichong Meng, Changdi Yang, Jun Liu, Hao Tang, Pu Zhao, and Yanzhi Wang. Instructgie: Towards generalizable image editing. In ECCV, 2024. 2
- [25] Ron Mokady, Amir Hertz, Kfir Aberman, et al. Null-text inversion for editing real images using guided diffusion models. In CVPR, 2023. 2, 7
- [26] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In AAAI, 2024. 2
- [27] Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICML, 2021. 2
- [28] Maxime Oquab, Tristan Darcet, Theo Moutakanni, Huy Vo, Marc Szafraniec Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Russell Howes, Wojciech Galuba, Piotr Bojanowski, Natalia Neverova, Andrea Vedaldi, Mike Rabbat, Yann LeCun, and Mathilde Caron. Dinov2: Learning robust visual features without supervision, 2023. 7
- [29] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. Semantic image synthesis with spade. In CVPR, 2019. 2
- [30] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 2, 3, 5
- [31] Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. Dreambench++: A human-aligned benchmark for personalized image generation. In ICLR, 2025. 6, 7, 12, 13
- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 7
- [33] Stephen Roller et al. Hash layers for large sparse models. In NeurIPS, 2021. 2, 6
- [34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 5
- [35] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine

- tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 2, 7
- [36] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In NIPS, 2022. 1
- [37] Noam Shazeer et al. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In ICLR, 2017. 2, 4
- [38] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu-edit: Precise image editing via emu diffusion models. In CVPR, 2024. 2, 6, 7, 13
- [39] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 1
- [40] Haotian Sun, Tao Lei, Bowen Zhang, Yanghao Li, Haoshuo Huang, Ruoming Pang, Bo Dai, and Nan Du. Ec-dit: Scaling diffusion transformers with adaptive expert-choice routing. In ICLR, 2025. 2, 5, 12
- [41] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. In ICCV, 2025. 2, 4, 6, 7, 11, 12, 13
- [42] Xueyun Tian, Wei Li, Bingbing Xu, Yige Yuan, Yuanzhuo Wang, and Huawei Shen. Mige: Multi-instruction guided editing via diffusion models. In ACMMM, 2025. 2
- [43] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-toimage translation. In CVPR, 2023. 2
- [44] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds, 2024. 2
- [45] Ruipeng Wang, Junfeng Fang, Jiaqi Li, Hao Chen, Jie Shi, Kun Wang, and Xiang Wang. Ace: All-round creator and editor following instructions via diffusion transformer. In ICLR, 2026. 2, 7, 16
- [46] Yucheng Wang and Dan Xu. Modit: Learning highly consistent 3d motion coefficients with diffusion transformer for talking head generation, 2025. 5
- [47] Yilun Wang, Ziyi Chen, Hao Zhou, Wenqi Tang, and Zongxin Lin. Uno: Unified neural operator for any-to-any generation. In ICCV, 2025. 2, 6, 7, 11, 12, 13
- [48] Zedong Wang, Siyuan Li, and Dan Xu. Rep-mtl: Unleashing the power of representation-level task saliency for multi-task learning. In ICCV, 2025. 12
- [49] Cong Wei, Zheyang Xiong, Weiming Ren, Xinrun Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision, 2025. 6, 11
- [50] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation, 2025. 2, 7, 8, 12, 13, 16

- [51] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In CVPR, 2025. 2, 7
- [52] Yu Xu, Fan Tang, You Wu, Lin Gao, Oliver Deussen, Hongbin Yan, Jintao Li, Juan Cao, and Tong-Yee Lee. In-context brush: Zero-shot customized subject insertion with context-aware latent space manipulation, 2025. 2
- [53] Yang Xu et al. Styleswin: Transformer-based gan for highresolution image generation. In CVPR, 2022. 2
- [54] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In CVPR, 2025. 2, 6, 7, 11
- [55] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. In NIPS, 2023. 2, 6, 7, 11, 13
- [56] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICLR, 2023. 2
- [57] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large-scale diffusion transformers. In NIPS, 2025. 2
- [58] Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. In NIPS, 2024. 2

### CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing Supplementary Material

This appendix provides complete supplementary material to the main manuscript and is organized as follows:

- • Appendix A: Dataset and Implementation Details. We detail the construction of training dataset in Appendix A.1, including the targeted design of mask-aware image pairs and the generation pipeline. We also provide a systematic comparison with existing public datasets. Appendix A.2 presents implementation specifications, including network architectures, optimization protocols, training schedules, and hyperparameter configurations in our experiments.
- • Appendix B: Extended Qualitative Comparisons. We report additional qualitative comparisons for instructionbased (Appendix B.1) and subject-driven (Appendix B.2) editing that could not be included in main text due to space limitations, encompassing mainstream tasks such as object removal, replacement, and style transfer. For each task, we include detailed per-category and per-edit-type samples and further discuss the behavior and capabilities of CAREEdit. We provide a project page, where we host more qualitative examples and interactive visualizations.
- • Appendix C: Additional Empirical Analysis. We present the empirical analysis and visualizations of latent attention maps in Appendix C to validate the efficacy of specialized experts. Figure 11 demonstrates how each expert in CAREEdit, including condition-aware routing, reference-guided subject preservation, and mask-aware control, contributes to effective task-specific learning over diffusion timesteps.

#### A. Dataset and Implementation Details

##### A.1. Training Dataset

Base corpora. To equip CARE-Edit with diverse editing capabilities, we collect data from several high-quality sources, targeting four key editing tasks: instruction-based editing, object removal/replacement, and style transfer. (i) Instruction-based edits are drawn from MagicBrush [55] and OmniEdit [49], which provide rich natural-language instructions paired with real-world images. (ii) and (iii) Removal and replacement samples are sourced from UNO [47] subset. (iv) Style transfer data is enriched using AnyEdit [54], which contains both the fine-grained appearance- and style-level instructions (e.g., “convert to watercolor painting”).

Motivation for Subjects200K. A critical limitation of purely instruction-based datasets is the spatial underspecification of edits. Models are often forced to infer the edit location solely from language, leading to ambiguity. To address this, we require a dataset with precise object masks

[Figure 124]

Figure 8. Pipeline for Mask-aware Image-Pair Generation. We use a GPT-Image-1 and VLM-based pipeline to create our training data. Starting with a reference subject on a white background, we generate diverse scene descriptions and corresponding images using an image-to-image model. This yields high-quality image pairs with consistent backgrounds but varying foregrounds, annotated with precise segmentation masks (M) and bounding boxes (B).

to explicitly teach the model to align edits with spatial constraints. We thus select Subjects200K [41] as our foundation for two reasons: (i) it offers high-quality foreground masks for a diverse taxonomy of objects and humans; and (ii) reference images are captured on clean white backgrounds, which simplifies downstream composition and in-context editing. From this source, we construct a 20K subset where each sample comprises an image, a fine-grained mask, an instruction, and an optional reference image, enabling CARE-Edit to learn region-specific editing while keeping subject identity.

Mask-Aware Image-Pair Generation. As shown in Figure 8, we construct background-consistent image pairs with varying foregrounds using a GPT-based generation pipeline. Starting with a Subjects200K [41] reference subject, we (i) sample a scene template and a set of subjects, and (ii) query the GPT-image-1 model to synthesize images that share a consistent background but feature diverse foreground objects. (iii) Extract a high-resolution fine mask M ∈ {0,1}H×W for foreground using an off-the-shelf segmentation model, followed by manual filtering to ensure quality. In practice, we instantiate these templates with category names (e.g., swan, boat, flamingo) and short descriptions of target background (e.g., “floating on a calm river near the shore”). To improve robustness against imperfect user inputs at inference time, we also derive a coarse mask B in training, defined as the tight axis-aligned bounding box of the fine mask M:

Ω = {(x,y) | M(x,y) = 1}, (18) xmin = min

x, (19) ymin = min

x, xmax = max

(x,y)∈Ω

(x,y)∈Ω

y, (20)

y, ymax = max

(x,y)∈Ω

(x,y)∈Ω

1, xmin ≤ x ≤ xmax ∧ ymin ≤ y ≤ ymax, 0, otherwise,

B(x,y) =

(21)

where xmin,xmax,ymin,ymax are the extrema of coordinates in Ω. We utilize M for the pixel-accurate supervision (e.g., boundary consistency loss) and B as a coarse spatial prior for the condition-aware expert routing in CARE-Edit.

Prompt Taxonomy. To systematically construct diverse yet controllable training triplets [20, 21] for our CARE-Edit, we organize the generation pipeline along two axes:

###### (1) Category and Operation. We factorize the synthetic

space into (i) Categories (people, animals, everyday objects, stylized assets) and (ii) Operation Types (instruction-based, removal, replacement, style transfer, multi-subject cases).

###### (2) Scene-level Templates. Conditioned on a subject

category, we query the LLM to generate multiple scene descriptions. Crucially, the scene prompt is constrained to a single line and constrains the generation to modify only the environment, lighting, camera view, or overall atmosphere, while strictly preserving the core subject identity or attributes. For each Category, we sample five such scene descriptions. This separation of subject and scene allows us to reuse identical subjects across heterogeneous contexts, and in turn to build foreground-consistent but background-varying pairs for mask-aware training. Please refer to Figure 9 for details.

(3) Task-level Templates. Given two subject descriptions a and b, a scene description s, and a style phrase p (e.g., “in a retro vintage style”), we instantiate three mask-friendly templates: (i) Replacement (Trep) keeps the background unchanged while swapping the main subject from a to b; (ii) Addition (Tadd) forms a diptych where one panel contains only a and the other contains both a and b under the same scene s; (iii) style / attribute change (Tsty / Tattr) preserves the subject identity and scene but alters only high-level appearance attributes p; All templates explicitly ask the generator to keep background layout, lighting, and camera viewpoint as similar as possible across paired images, so that the resulting pairs differ primarily in well-localized foreground regions. This naturally aligns with our fine masks M and coarse boxes B, and provides clean supervision for subject-centric, mask-aware editing. Please view Figure 10 for details.

Data Efficiency vs Previous Training Setups. Despite being trained on a significantly smaller data corpus (∼ 120K triplets in total) compared to recent state-of-the-art editing and personalization baselines, CARE-Edit achieves superior performance. Table 4 reports multiple-object results on DreamBench++ [31], together with the approximate scale of the training data used by each method. The results show that CARE-Edit outperforms strong baselines such as OmniControl [41], UNO [47], and OmniGen2 [50] on all multipleobject metrics, despite relying on substantially fewer training samples. This suggests that our mask-aware, subject-centric curriculum applied in CARE-Edit and the curated multipaired construction are substantially more data-efficient than

Table 4. Quantitative results on the multiple-object subset of DreamBench++ [31]. We report three metrics along with the approximate number of training examples used by each method. The best and second-best results are marked in bold and underlined, respectively.

Method #Train data DINO-I ↑ CLIP-I ↑ CLIP-T ↑

OmniControl [41] ˜1M 0.501 0.641 0.316 UNO [47] ˜1M 0.508 0.649 0.303 OmniGen2 [50] ≥ 533K 0.560 0.713 0.319

CARE-Edit (Ours) 120K 0.568 0.720 0.327

simply scaling up instruction-only datasets, particularly for applications with complicated multi-object compositions.

##### A.2. Implementation Details

Backbone and Training Setup. CARE-Edit is built upon the FLUX.1 [16] variant of the Rectified Flow Transformer family. Unless otherwise specified, we select FLUX.1-dev as the backbone for all experiments, as it offers a good balance between visual quality and training stability in the editing setup. Following the design of OmniControl [41], we employ condition-aware modules via LoRA [11] on top of the base model, and keep the original backbone weights frozen during fine-tuning. We adopt a standard LoRA rank [11] of r = 4 for all attention modules, and only enable the LoRA branches [11] when processing condition-related tokens. For regular text-only tokens, the LoRA scale is set to zero so that the backbone behaves identically to the original FLUX.1 [16] model. All models are trained on 8×NVIDIA L20 GPUs, which corresponds to roughly 800 GPU hours in total. We use a per-GPU batch size of 1 with gradient accumulation over 8 steps (effective batch size of 8). For most experiments, we follow a two-stage training schedule: the model is first trained for 40K iterations on basic, singlesubject samples, and then switches to complex multi-subject data for the remaining 60K iterations. We also apply EMA to the LoRA parameters [11] with a decay rate of 0.999.

Loss Functions and Hyperparameters. The total training objective LCARE combines the standard diffusion reconstruction loss Ldiff (Sec. 3.1) with three regularization terms as:

LCARE = Ldiff +λload Lload +λmask Lmask +λmix Lmix (22)

where Lload ensures balanced expert utilization (Sec. 3.2), Lmask enforces the boundary consistency (Sec. 3.3), and Lmix

encourages spatial smoothness in the mixture map (Sec. 3.4). To prioritize the reconstruction term while maintaining regularization, we empirically set small weights to regularizers:

###### (λload, λmask, λmix) = (0.01, 0.1, 0.05). (23)

where the hyperparameters were identified according to prior works [40, 48] and fixed for all experiments w/o extra tuning.

[Figure 125]

Role: You are an expert scene designer for image-to-image generation. Goal: Given a brief subject prompt [asset], generate 5 different Scene Descriptions that place the same subject in diverse environments. Rules:

- 1. Each Scene Description must reuse the identical subject [asset] but change the background, environment, camera view, lighting, or atmosphere.
- 2. Focus on describing ONLY the background and global conditions; do NOT alter the core identity of the subject.
- 3. Each Scene Description must be one line and should not exceed ~65 tokens.
- 4. Be highly creative and cover a wide range of locations, times, and moods. Format: [asset]: Koala

- [SceneDescription1]: [A koala] sitting on a wooden railing of a city rooftop garden at sunset, with skyscrapers in the distance.
- [SceneDescription2]: [A koala] sitting on a picnic blanket in a sunny park, surrounded by scattered fruits and leaves.
- [SceneDescription3]: [A koala] sitting on a tree trunk above a small creek in a lush tropical valley after rain.
- [SceneDescription4]: [A koala] clinging to a tall eucalyptus tree in a misty forest at dawn.
- [SceneDescription5]: [A koala] resting on a branch inside a glass-walled rainforest dome in a modern zoo.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

- Figure 9. GPT Prompting for Scene Diversity. We explicitly instruct the LLM to generate varied scene descriptions (e.g., “sunny park”) while keeping the subject (e.g., “Koala”) constant. As such, by synthesizing multiple environments for the identical subject entity, we create training data that compels the image editing model to learn robust subject grounding independent of the background correlations.

##### B.1. Instruction-based Image Editing

Evaluation. We evaluate CARE-Edit on four representative image editing tasks across three diverse benchmarks that probe different aspects of contextual image editing:

In this subsection, we introduce and discuss qualitative results on instruction-based image editing on EMU-Edit [38] and MagicBrush [55]. Figure 12 shows a large-scale visual comparison between our method and several strong baselines, including OmniGen2 [50], ACE++ [22], and the vanilla FLUX.1-Dev [16] backbone. The examples cover typical instruction-based edits such as style changes, attribute modifications, and cases involving visible text (e.g., a toy holding a “CARE” label).

- • EMU-Edit [38]: this benchmark tests both the objectlevel and attribute-level modifications on real-world photos using fine-grained text prompt descriptions.
- • MagicBrush [55]: this dataset involves complex, regionbased editing tasks guided by free-form natural language instructions and the user-provided masks.
- • DreamBench++ [31]: this benchmark evaluates personalized subject-driven image editing and composition, covering single-object and complex multiple-object scenarios.

Qualitatively, CARE-Edit (especially the masked variant) follows the textual instructions while better preserving unedited content and fine-grained structures. Compared to SOTA methods, our method produces fewer spurious background changes and sharper, more localized boundaries at the edited regions. These trends are consistent with the quantitative gains reported in the main paper.

We follow the official data splits and evaluation subsets for each benchmark whenever available. Following the data processing pipelines in OmniControl [41] and UNO [47], we resize images to 512×512 while preserving aspect ratio.

#### B. Extended Qualitative Comparisons

##### B.2. Subject-driven Contextual Image Editing

In this subsection, we provide more results on subject-driven contextual editing, primarily on DreamBench++ [31]. The goal is not only to preserve subject identity (e.g., a particular person, pet, or product) but also to compose the subject into new contexts with complex surroundings and interactions.

Appendix B reports additional experimental results that could not be included in the main paper due to space limitations. In particular, we organize these results by task to demonstrate the model’s robustness in handling diverse semantic demands, from subtle attribute changes to complex scene re-contextualization. For each task, we include representative per-category and per-edit-type samples and briefly discuss the behavior of CARE-Edit across different settings.

A key motivation behind our design is that, for this class of tasks, it is often difficult to resolve the relative size and placement of the reference objects in the base image using

[Figure 132]

[Figure 133]

Replacement (𝑇𝑇𝑟𝑟𝑟𝑟𝑟𝑟) : "High-quality photo of the same scene as before, but the main subject is [b] instead of [a]. The background, lighting, and camera angle should remain as similar as possible to the previous image."

Add (𝑻𝑻𝒂𝒂𝒂𝒂𝒂𝒂) : " High-quality photo of [a] together with [b] (on the left/right) in the same scene, keeping the background, lighting, and overall layout as consistent as possible between the two panels."

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

𝑇𝑇𝑟𝑟𝑟𝑟𝑟𝑟

𝑻𝑻𝒂𝒂𝒂𝒂𝒂𝒂

[Figure 140]

[Figure 141]

Replacement (𝑻𝑻𝒔𝒔𝒔𝒔𝒔𝒔) : "High-quality photo of the same [a] in the exact same scene, but now style is [p]. Keep the background, layout, and lighting as similar as possible; only change the style.”

Replacement (𝑻𝑻𝒂𝒂𝒔𝒔𝒔𝒔𝒂𝒂) : "High-quality photo of the same [a] in the exact same scene, but now attribute is [p]. Keep the background, layout, and lighting as similar as possible; only change the attribute.”

|A purple car|𝑻𝑻𝒂𝒂𝒔𝒔𝒔𝒔𝒂𝒂|
|---|---|
| | |

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

𝑻𝑻𝒔𝒔𝒔𝒔𝒔𝒔

- Figure 10. Task-Specific GPT Prompt Templates. We visualize the templates used to construct training triplets for: (i) Replacement (Top-Left); (ii) Addition (Top-Right); and (iii) Style/Attribute Change (Bottom). By constraining the LLM to modify specific slots (e.g., subject identity) while holding scene descriptions constant, it ensures the resulting image pairs possess consistent backgrounds.

text prompts and the backbone model alone. Instructions such as “The man is holding a camera.” or “Add a Rubik’s Cube next to the sneaker.” do not uniquely determine how large the inserted object should be or where it should appear. To address this, CARE-Edit incorporates a user-provided mask as an additional control signal. Even when the mask is coarse, it specifies the intended location and approximate size of the edited content, disambiguating the spatial relationship between the reference objects and the base subject.

Figure 13 illustrates this design with representative cases such as “The man is holding a cup.”, “Add a watch next to the drink.”, and “Add a toy bear next to the cat.”. In all these examples, CARE-Edit produces edits where the inserted objects have plausible geometry and scale, while the main subject’s identity, pose, and global lighting are preserved. This mask-guided formulation enables reliable subject-driven contextual editing in scenarios such as personalized product shots and multi-object layout design, where precise control over relative size and placement is crucial.

##### B.3. More Results on Diverse Editing Tasks

In this concluding subsection, we present extensive visual evidence of CARE-Edit across different editing tasks and summarize how they map to practical usage scenarios. Figures 14–16 show extended qualitative results for object re-

moval, object addition and replacement, and style transfer.

Object Removal. Given an input image, the model removes the selected object and synthesizes background content consistent with the surrounding regions. Figure 14 shows that our CARE-Edit can inpaint relatively large masked areas without obvious seams or blur, while leaving unedited regions nearly unchanged.

Addition and Replacement. The user provides a short text instruction (e.g., “Add ...”, “Replace ...”) and a coarse mask indicating desired location and approximate size of the edited object. Figure 15 shows that CARE-Edit uses this mask to control scale and placement, filling the region with an object that matches the text and blends with the scene.

Style Transfer. The image content is largely preserved, while global appearance is modified according to a target style. Figure 16 shows that CARE-Edit maintains scene structure and object boundaries, avoiding severe detail loss.

These results show that CARE-Edit can handle removal, addition, replacement, and stylistic changes.

#### C. Analysis of Expert Lattent Attention Maps

To complement the qualitative results and provide a mechanistic understanding of CARE-Edit, we conduct a deep

diagnostic empirical analysis of the model’s internal expert learning behavior. A core hypothesis of this work is that different editing conditions (e.g., text semantics vs. spatial masks vs. reference style) impose different learning dynamics on a shared backbone. CARE-Edit resolves this via an explicit routing of heterogenous, specialized experts.

The main paper only visualizes the Base Expert due to the space limitations. To validate that these experts indeed evolve distinct, complementary roles rather than collapsing into a uniform average, we visualize the attention maps of all three condition-aware experts, (i) Base, (ii) Mask, and (iii) Reference, throughout the training process. Figure 11 illustrates the evolution of these attention map distributions at different training iterations (T = 0, T = 30K, T = 70K, and T = 100K). This visualization effectively opens the black box of these experts, revealing how the model learns to disentangle complex editing objectives over time.

Base Expert: The Global Anchor. As shown in Figure 11, the Base Expert (Top Row) maintains a robust, spatially widespread activation pattern across the entire training trajectory. Even at late training stages (T = 100K), its attention map covers the majority of the image canvas with high intensity. This confirms its role as the task-agnostic anchor, which is responsible for preserving the intrinsic structure, lighting, and layout of the original image, while incorporating conditional information. By handling the global coherence, the Base Expert frees the other experts to focus purely on differential changes, ensuring that the unedited regions remain perceptually and semantically consistent with base images.

Mask Expert: Spatial and Geometric Specialization. The Mask Expert (mid-row in Figure 11) displays the most dramatic evolution, demonstrating the emergence of spatial intelligence. At early training stages (T = 0), the attention is diffuse, noisy, and object-unaware. However, as training progresses through the mid-phase (T = 30K to 70K), the entire attention becomes aggressively focused, concentrating strictly within and immediately around the user-provided input editing regions. By T = 100K, the Mask Expert exhibits fine-grained, binary-like activation boundaries that align perfectly with the intended edit objects. This trajectory indicates that the Mask Expert successfully learns to exploit the Mask Repaint module’s signals, delegating geometric restructuring (e.g., object removal, shape modification) exclusively to this expert while suppressing its influence on the original background to prevent potential leakage or artifacts.

Reference Expert: Semantic and Stylistic Injection. The Reference Expert (Bottom Row in Figure 11) exhibits a distinct pattern of ”semantic sparsity.” Unlike the Mask expert, which aligns with geometry, the Reference expert aligns with content relevant to the style or identity transfer.

MaskRefBase

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

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

T=0 T=30K T=70K T=100K

Figure 11. Expert Specialization During Training. We visualize the latent attention maps for the Base, Mask, and Reference experts at increasing training iterations (T). (Top) Base Expert: Maintains consistent, global activation throughout training (T = 0 → 100K), acting as a foundation to preserve image structure. (Middle) Mask Expert: Learns to progressively suppress background signals, evolving from a noisy initialization to a highly localized, structureaware attention map that precisely targets the edit region. (Bottom) Reference Expert: Gradually increases engagement in regions requiring semantic or stylistic modification. This distinct separation of concerns confirms that CARE-Edit effectively disentangles conflicting editing signals into specialized processing pathways.

Initially inactive, its attention grows as the model learns to map features from the reference image encoder (Zr) to the generated latent space. At convergence stage (T = 100K), we observe heightened activation in regions that require texture synthesis or photometric adjustment (e.g., the surface of an object changing material, or the entire scene during style transfer). More importantly, its activation map is orthogonal to the Base expert. It injects fine-grained appearance cues (e.g., color, texture) without overwriting the original structural geometry maintained by the Base and Mask experts.

Overall, these distinct activation signatures observed in Figure 11 validate the efficacy of our Condition-Aware Routing design. Instead of forcing a single set of weights to compromise between preserving identity and changing style, CARE-Edit dynamically distributes the workload: the Mask expert handles the “where,” the Reference expert handles the “what” (appearance), and the Base expert ensures global consistency of the image to be edited. This learned specialization serves as the key factor enabling our CARE-Edit to minimize task interference from mulitple inputs and thus achieve highfidelity editing in the challening multi-condition scenarios.

Ours (w/o mask)

Ours (with mask)

OmniGen2 (Flux .1 Dev)

ACE++ (Flux .1 Fill) Flux .1 Dev

Base

Text

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

The toy is holding a "CARE" tag

Human(DB)Animal(DB)OOD

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

A girl strolling through a vibrant marketplace

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

a raccoon in the fallen leaves of a forest during autumn

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

A cherry blossom tree in the Ukiyo-e style

Style(DB)

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Alter the style of the image to painting

Style(AE)

- Figure 12. Instruction-based Editing on EMU-Edit and MagicBrush. We compare CARE-Edit against existing unified editors [45, 50]. Rows exhibit challenging scenarios such as text rendering (“CARE tag”), global style transfer (“Ukiyo-e”), and complex object insertions.

(1) Text and Geometric Fidelity (Row 1): The geometric rigidity of the tag and the legibility of the text are paramount. While ACE++ and OmniGen2 correctly interpret the semantic intent, they suffer from structural drift, resulting in warped tag boundaries and deformed glyphs. CARE-Edit, particularly the masked variant, utilizes the Mask Expert to enforce spatial constraints, producing orthogonal tag geometry and crisp, readable text. (2) Identity Preservation (Row 2): CARE-Edit preserves the subject’s facial identity and hair texture significantly better than baselines, which tend to over-blend the subject into the crowd (identity dilution) or generate a generic face. (3) Style Disentanglement (Row 4 & 5): In the “Ukiyo-e” and “Painting” style transfer tasks, baselines often hallucinate new objects or flatten the entire scene into a texture map. CARE-Edit disentangles style from structure. The Base Expert maintains the complex branching of the cherry blossom tree and the mechanical details of the tractor, while Reference Expert strictly applies the artistic texture to the environments.

Ref Base Mask Result

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

“The man is holding a cup.”

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

| |
|---|

“Add a Rubik's Cube next to the sneaker.”

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

“The man is holding a camera.”

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

“Add a watch next to the drink.”

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

| |
|---|

“Add a toy bear next to the cat.”

- Figure 13. Complex Contextual Editing Results. Multi-condition examples requiring harmonization of subject identity, mask constraints, and text prompts. CARE-Edit successfully composes subjects into disparate environments while respecting the user-provided spatial layout.

|Input Output|Input Output|
|---|---|
|[Figure 215]<br><br>[Figure 216]|[Figure 217]<br><br>[Figure 218]|
|[Figure 219]<br><br>[Figure 220]|[Figure 221]<br><br>[Figure 222]|
|[Figure 223]<br><br>[Figure 224]|[Figure 225]<br><br>[Figure 226]|

- Figure 14. Object Removal Results. This task requires the model to remove the foreground object and hallucinate a plausible background (e.g., sofa fabric, wall patterns) that is consistent with surrounding scene context. (i) Top and Middle: CARE-Edit successfully handles stochastic textures, such as natural water ripples and uneven stone surfaces, filling the void with spatially coherent content. (ii) Bottom-Left: A stress test for structural consistency. Removing the television requires reconstructing the rigid grid pattern of the wallpaper. CARE-Edit accurately hallucinates the missing tiles, maintaining the correct perspective and alignment without the blurring or geometric warping.

|Object Addition|Object Replacement|
|---|---|
|Input Output|Input Output|
|[Figure 227]<br><br>[Figure 228]<br><br>Add<br><br>[Figure 229]<br><br>| |
|---|
<br><br>[Figure 230]|[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>Replace|
|[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]<br><br>[Figure 238]<br><br>Add|[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]<br><br>Replace<br><br>|
|[Figure 243]<br><br>[Figure 244]<br><br>[Figure 245]<br><br>[Figure 246]<br><br>Add|[Figure 247]<br><br>[Figure 248]<br><br>|[Figure 249]|
|---|
<br><br>Replace<br><br>[Figure 250]|

- Figure 15. Object Addition and Replacement Results. CARE-Edit demonstrates precise control over scene composition, inserting or swapping objects while rigorously adhering to environmental constraints. (i) Object Addition (Left): CARE-Edit introduces new elements that respect physical laws. Note that how the added rubber duck (middle) is generated with accurate water reflections and surface interaction. (ii) Object Replacement (Right): CARE-Edit handles drastic changes in structure and material while maintaining lighting consistency. In the top-right example, replacing a furry pokemon creature with a pokemon ball (metallic sphere), CARE-Edit correctly renders specular highlights and casts realistic shadows onto the complex dirt terrain, ensuring the newly added object sits naturally within the depth of field.

SketchesVincentvanGoghWatercolor

|Input Output|Input Output|
|---|---|
|[Figure 251]<br><br>[Figure 252]|[Figure 253]<br><br>[Figure 254]|
|[Figure 255]<br><br>[Figure 256]|[Figure 257]<br><br>[Figure 258]|
|[Figure 259]<br><br>[Figure 260]|[Figure 261]<br><br>[Figure 262]|

- Figure 16. Reference-guided Style Transfer with Structure Preservation. A key advantage of CARE-Edit is the ability to decouple style from structure via expert routing. Unlike holistic transfer methods that often deform the underlying geometry, our approach injects the target aesthetic while strictly anchoring the scene layout. (i) Top: The results show that CARE-Edit’s Reference Expert successfully translates the scene into the swirling impasto of Van Gogh or a watercolor wash, yet the Base Expert ensures the architectural rigidity of the cabin, preserving the straight lines of the roof and window frames. (ii) Bottom: The subject’s fur texture is re-rendered to match the artistic medium, demonstrating CARE-Edit’s successful fine-grained textural adaptation without distorting the animal’s original silhouette or pose.

