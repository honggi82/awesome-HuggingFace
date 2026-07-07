# arXiv:2508.07981v4[cs.CV]17Dec2025

### Omni-Effects: Unified and Spatially-Controllable Visual Effects Generation

##### Fangyuan Mao1*, Aiming Hao1*, Jintao Chen1,2, Dongxia Liu1,3, Xiaokun Feng1,4, Jiashu Zhu1, Meiqi Wu1,4, Chubin Chen1,3, Jiahong Wu1†, Xiangxiang Chu1

1AMAP, Alibaba Group 2Peking University 3Tsinghua University 4Institute of Automation, Chinese Academy of Sciences fangyuanmaocs@gmail.com, {haoaiming.ham, hongxi.wjh}@alibaba.inc.com

###### Abstract

Visual effects (VFX) are essential visual enhancements fundamental to modern cinematic production. Although video generation models offer cost-efficient solutions for VFX production, current methods are constrained by per-effect LoRA training, which limits generation to single effects. This fundamental limitation impedes applications that require spatially controllable composite effects, i.e., the concurrent generation of multiple effects at designated locations. However, integrating diverse effects into a unified framework faces major challenges: interference from effect variations and spatial uncontrollability during multi-VFX joint training. To tackle these challenges, we propose Omni-Effects, a first unified framework capable of generating prompt-guided effects and spatially controllable composite effects. The core of our framework comprises two key innovations: (1) LoRA-based Mixture of Experts (LoRA-MoE), which employs a group of expert LoRAs, integrating diverse effects within a unified model while effectively mitigating cross-task interference. (2) Spatial-Aware Prompt (SAP) incorporates spatial mask information into the text token, enabling precise spatial control. Furthermore, we introduce an Independent-Information Flow (IIF) module integrated within the SAP, isolating the control signals corresponding to individual effects to prevent any unwanted blending. To facilitate this research, we construct a comprehensive VFX dataset Omni-VFX via a novel data collection pipeline combining image editing and First-Last Frame-to-Video (FLF2V) synthesis, and introduce a dedicated VFX evaluation framework for validating model performance. Extensive experiments demonstrate that Omni-Effects achieves precise spatial control and diverse effect generation, enabling users to specify both the category and location of desired effects.

Code — https://amap-ml.github.io/Omni-Effects.github.io Extended version — https://arxiv.org/abs/2508.07981

##### Introduction

Visual effects (VFX) play a crucial role in modern filmmaking, enabling the creation of immersive narratives and fantastical worlds. While traditional VFX pipelines, especially

*These authors contributed equally. †Corresponding Author

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

[Figure 1]

Figure 1: Capabilities for diverse customized visual effects. Omni-Effects supports both (a) single-VFX and (b) multiVFX generation through pure prompt-guided generation. Integrated with the Spatial-Aware Prompt, Omni-Effects enables (c) precise spatial VFX control and (d) intricate objectbased visual effects with targeted environmental transformations.

for composite effects requiring simultaneous coordination across different spatial locations, are notoriously complex and resource-intensive (Chabanova 2022). Recent rapid advances in video generation technologies (Yang et al. 2024; Bao et al. 2024; Kong et al. 2024; Wan, Wang et al. 2025) are driving a paradigm shift in VFX creation—transitioning from conventional methods to generative model-powered dynamic, efficient synthesis.

The inherent scarcity of VFX data and pronounced variability in dynamic characteristics across effects pose significant challenges to training generative models. Consequently, current methods (Liu et al. 2025b) focus on singleeffect generation, employing dedicated Low-Rank Adaptation (LoRA) (Hu et al. 2022) tailored to individual effects. However, this paradigm struggles with multi-VFX scenes, exhibiting two critical limitations: Limitation 1. Cross-Adapter Interference where joint multi-LoRA activation (Ding et al. 2023) induces spatial occlusion artifacts (Figure 2 (a)) and shared-subspace hybrid training triggers fidelity-degrading cross-effect confusion via task interfer-

[Figure 2]

- Figure 2: Defects in standard video generation models. (a) VFX disappearance, (b) quality degradation, (c) confusion between VFX elements, and (d) spatial uncontrollability.

ence (Zhang et al. 2025a) (Figure 2 (b, c)). Limitation 2. Spatial-Semantic Misalignment wherein the text-pixel space gap prevents precise spatial cue encoding for VFX placement (Figure 2 (d)). These limitations fundamentally constrain conventional video generation adaptation to complex multiple VFX compositions (multi-VFX).

To address these limitations, we propose Omni-Effects, a unified framework modeling multi-VFX generation as a multi-condition video generation problem, where textual prompts specify effect categories while spatial masks define their precise locations. First, to tackle Limitation 1, we introduce a LoRA-based Mixture of Experts (LoRAMoE) module (Shazeer et al. 2017; Dou et al. 2023; Zhang et al. 2025c), which partitions effects into specialized subspaces, each optimized by dedicated expert branches, with a gating router dynamically activating relevant subspaces to minimize cross-task interference and enhance effect fidelity. Second, to overcome Limitation 2, we present the Spatial-Aware Prompt (SAP), which integrates explicit mask-based spatial conditioning with textual inputs via full-attention mechanisms for accurate effect placement. To mitigate SAP cross-interference during concurrent application, we introduce the Independent-Information Flow (IIF) module, which isolates condition-specific information flows through IIF Attention Mask, preventing unintended effect blending. Collectively, these innovations enable OmniEffects—to our knowledge, the first VFX framework—to achieve high-fidelity multi-VFX compositions with pixellevel spatial control. (Figure 1).

To advance this research, we construct a high-quality VFX dataset Omni-VFX and develop a specialized VFX data pipeline. This pipeline utilises image editing models (Liu et al. 2025a) to generate source image pairs depicting ini-

tial/final effect states, which are then synthesised into VFX videos via the FLF2V framework (built on Wan2.1 (Wan, Wang et al. 2025)). Rigorous manual filtering ensures quality while expanding coverage to 55 distinct effect categories. Further, we introduce an evaluation framework specifically designed for controllable VFX generation tasks. Comprehensive experiments validate the Omni-Effects framework’s superior performance across three core capabilities: singleVFX, multi-VFX, and controllable VFX generation. In summary, the major contributions of our work are as follows:

- • A unified VFX framework, Omni-Effects, is proposed to enable high-fidelity spatial controllable multi-VFX generation through a dual-core architecture: (1) LoRAMoE modules unifying multi-VFX training, and (2) IIF-augmented SAP mechanisms enabling independent multi-condition control without interference.
- • The most comprehensive VFX dataset Omni-VFX is developed with an automated production pipeline to support the generation of diverse high-quality VFX video, complemented by a comprehensive evaluation framework for rigorous controllable VFX assessment.
- • Extensive experiments demonstrate that our OmniEffects achieves precise spatial control and enables diverse VFX generation, thereby allowing users to specify both the category and location of desired effects.

##### Related Works

###### Video Generation Models

Recent advances in diffusion-based video generation (Chen et al. 2023; He et al. 2022; Zeng et al. 2024; Kong et al. 2024; Yang et al. 2024; HaCohen et al. 2024; Bao et al.

- 2024; Polyak et al. 2024; Wan, Wang et al. 2025; Teng et al.
- 2025; Seawead et al. 2025; Ling et al. 2025; Chen et al.

2025) have enabled text-to-video (T2V) and image-to-video (I2V) synthesis, where input images establish spatial context. As a critical I2V application, VFX generation creates unrealizable fantastical visuals. However, VFX data scarcity forces reliance on Low-Rank Adaptation (LoRA) (Hu et al. 2022) for limited-data fine-tuning (Liu et al. 2025b). This necessitates separate LoRA models per effect, while combined training causes performance degradation, fundamentally limiting multi-VFX generation within a single video.

Our architecture unifies multi-VFX training, avoids degradation, and enables concurrent multi-VFX generation.

###### Conditional Video Generation

Condition-guided diffusion models leverage auxiliary inputs for precise output control, falling into two paradigms: spatial-fusion guidance and high-level semantic guidance. Spatial-fusion methods, exemplified by ControlNet (Zhang, Rao, and Agrawala 2023), integrate condition inputs with denoising inputs. These methods (Li et al. 2024; Bai et al. 2024; Bian et al. 2025; Xu et al. 2025; Lei et al. 2025; Jiang et al. 2025) enable fine-grained spatial alignment while preserving the generation quality of pre-trained diffusion models. High-level semantic methods exploit latent interactions between conditions and the denoising process. Techniques

[Figure 3]

- Figure 3: Flowchart of proposed Omni-Effects. Given a reference image and composite conditions of arbitrary length, OmniEffects first encodes each input into corresponding tokens. These tokens are concatenated and processed sequentially through downstream DiT blocks. These blocks incorporate two key technologies: (a) LoRA-MoE, a MoE plugin replacing standard FFN linear layers to enable collaborative expert task-solving and (b) SAP, which fuses effect descriptors with spatial trigger information during the attention stage while mitigating cross-condition information leakage via an IIF mechanism. Note that, in the IIF, dashed lines represent blocked information flow, while solid lines indicate active information transmission.

include cross-attention based mechanisms (Ye et al. 2023; Zhang et al. 2024; Zhou et al. 2025; Yuan et al. 2025) and conditional token concatenation strategies (Wang et al. 2024; Huang et al. 2024a; Tan et al. 2025a,b; Zhang et al. 2025b), dynamically modulate generation through semantic embeddings. Crucially, while these methods effectively handle individual conditions, they struggle to simultaneously and independently control multiple conditions, a critical requirement for professional VFX generation.

Our model employs IIF-powered SAP control mechanisms to support independent, non-interfering control of multiple conditions within the same video.

##### Method

###### Preliminaries

Video Diffusion Models Video generation models usually utilize the diffusion paradigm (Ho, Jain, and Abbeel 2020; Lipman et al. 2022), which generates samples from a data distribution p(x0) by progressively denoising samples that are initially drawn from a Gaussian distribution p(xT). During training, clean samples x0 ∼ p(x0) undergo iterative corruption through T diffusion steps:

xt = αtx0 + σtϵ, ϵ ∼ N (0,I), t = 1,...,T (1)

where αt,σt > 0 are scalars that jointly define the Signal-toNoise Ratio (SNR). The denoiser with parameter θ is optimized to predict the target noise ϵ. The optimization process

is defined as:

t,t,c,ϵ ∥ϵ − ϵθ (xt,t,τ (c))∥22 (2)

Lmse (θ) = Ex

where c is conditions (e.g., text and spatial location), and τ denotes the condition encoder. By replacing target noise ϵ with v, which is a weighted combination of x0 and ϵ, as the prediction target, the v-prediction (Salimans and Ho 2022) is derived, which is adopted in our Omni-Effects framework. Moreover, mainstream video generation leverages Diffusion Transformer (DiT) (Peebles and Xie 2023) architecture by employing attention mechanisms to model spatiotemporal consistency while aligning conditional inputs with visual outputs (Zheng et al. 2024). By integrating diffusion processes with Transformer architectures, the video generation performance is improved, leading to high-quality and accurate video synthesis results.

Spatially Controllable Multi-VFX Generation In practical applications, it’s often necessary to display different VFX at distinct locations throughout a video. We formalize this task as multi-conditional video generation, wherein video diffusion models take a reference image and a set of

N control signals C = {ci}Ni=1 as inputs. Each condition ci = (ei,si) couples an effect descriptor ei with a spatial trigger si, whereby the generated video x0 applies effect ei at the location specified by si. Specifically, we use a text prompt to describe the VFX, while using a spatial mask

[Figure 4]

- Figure 4: FVD scores for diverse VFX trained with a shared LoRA. VFX performance exhibits an initial improvement followed by progressive degradation with increasing numbers of co-trained effects. This indicates inherent effect clustering: synergistic groups (e.g., Melt-like effects) improve co-training performance, while incompatible combinations (e.g., Deflate + Squish) suffer from mode collapse and underperform relative to compatible sets. Note that, lower FVD values indicate superior performance, with optimal VFX results uniformly achieved when the number of cotrained VFX equals 4.

m ∈ RH×W to serve as the spatial trigger. A set of conditions is incorporated into the denoising process, and the denoiser prediction becomes:

vˆ = ϵθ xt,t, τe(i) (ei)

N i=1

, τs(i) (si)

N i=1

, (3)

where τe and τs denote the text and spatial mask encoder, respectively. Notice that, when N = 1 and the spatial trigger is empty, the above task reduces to the traditional singleVFX generation task (Liu et al. 2025b).

###### Omni-Effects

To model the above task, video diffusion models require simultaneous support for multi-VFX inference and spatial control capabilities. We accordingly propose Omni-Effects, building upon the CogVideoX (Yang et al. 2024) architecture and incorporating two core components: LoRA-MoE and Spatial-Aware Prompt. The overview is illustrated in Figure 3, and the details are as follows.

LoRA-MoE As mentioned in Figure 2, both multi-LoRA parallel inference and single-LoRA unified training degrade performance. Crucially, we observe the synergistic mechanism in VFX training: compatible VFX-combination training enhances single-VFX generation quality (Figure 4). This discovery motivates our adaptive task-space partitioning strategy: inspired by MoE (Shazeer et al. 2017) architectures, we partition distinct effects into specialized subspaces and deploy a gating router for adaptive subspace selection.

Specifically, LoRA-MoE (Dou et al. 2023) integrates MoE with LoRA (Figure 3 (a)), which employs an expert ensemble where each LoRA specializes in distinct VFX manifolds. Formally, for input token x ∈ Rd, a weight is obtained by a gating network G : Rd  → Rn for each expert, resulting in G(x) = [G(x)1 ,G(x)2 ,...,G(x)n], where n represents the number of experts. Each expert Ei implements

[Figure 5]

Figure 5: Visualization of controllable VFX performance and attention maps. (a) Position description lacks spatial control; (b) ControlNet faces inter-condition interference, leading to VFX leakage and artifacts; (c) Proposed SAP+IIF achieves precise positional controllability while preventing mutual interference between multi-VFX.

LoRA decomposition:

α r

xAiBi, Ai ∈ Rd×r, Bi ∈ Rr×d, (4)

Ei(x) =

where r denotes the low-rank and α is a scaling factor. The final prediction combines base model and expert outputs:

y = Base(x) +

n

G(x)i ⊙ Ei (x), (5)

i=1

This MoE-structured plugin replaces standard FFN linear layers, enabling collaborative expert task-solving. During training, Top-k routing strategy (k ≤ n) is adopted to enforce exactly k non-zero entries in G(x). At inference, all experts are activated to prevent effect suppression caused by Top-k filtering, omitting critical experts, which is essential for multi-VFX combination generation. Moreover, to mitigate workload imbalance caused by the gating network favoring a few experts during training, we also employ a balanced routing auxiliary loss Laux (Fedus, Zoph, and Shazeer 2022). Comprehensive details are provided in Supplement A. The final training objective is expressed as: L = Lmse + βLaux, where β is hyperparameter.

Spatial-Aware Prompt For condition ci = (ei,si), embedding positional descriptors within text prompts proves insufficient for precise spatial control. To investigate this phenomenon, we visualize attention maps across diverse prompts. Crucially, attention consistently activates identical regions regardless of prompt semantics (Figure 5 (a)), evidencing textual position cues’ failure to direct activation toward specified targets. Prior work (Liu et al. 2025b; Jiang et al. 2025) mitigates this via ControlNet (Zhang, Rao, and Agrawala 2023) to extract a mask sequence for generation guidance. However, this solution suffers from two critical limitations:

1. Significant parameter overhead: ControlNet duplicates a portion of the base model’s parameters (typically half), requiring substantial extra trainable weights;

2. Severe cross-condition interference: During multiVFX generation, parallel ControlNet inference suffers from information leakage, manifesting as erroneous cooccurrence of effect ei and ej at positions si and sj respectively (Figure 5 (b)).

In summary, signals within composite conditions must be integrated while preventing cross-condition interference to ensure robust performance. We address these challenges by proposing the Spatial-Aware Prompt to directly inject spatial information into prompts tokens via enhanced spatialtext condition token interactions within attention mechanism, enabling controllable generation with minimal parameter/computational overhead. Building on this, we introduce Information-Independent Flow, which utilizes a designed attention mask to restrict cross-condition information exchange, thereby preventing interference between distinct control streams. Formally, given a set of conditions C, en-

N i=1

coder processing yields text condition tokens τe(i) (ei)

N i=1

and spatial condition tokens τs(i) (si)

, which are se-

quentially concatenated with the noisy latent xt to form the inputs Q, K and V . Then we define an attention mask

M ∈ {0,−∞}l×l (l is the total sequence length) to regulate attention flow (Figure 3 (b), details are in Supplement A) that blocks condition-to-condition and noise-to-condition interactions, eliminating cross-condition leakage to prevent effect misalignment or blending. The final output of attention is expressed as:

y = Softmax QKT/ dk + M V , (6)

where dk denotes the feature dimension. To enhance spatial conditioning alignment with noisy latents, we inject posi-

tional embeddings from xt’s initial frame into τs(i) (si), coupled with a dedicated Spatial-Condition LoRA. Crucially, all spatial conditions share identical LoRA parameters while maintaining a common base LoRA across other branches, ensuring efficient conditional injection without disrupting pretrained representations. Each text condition is individually processed through text encoder while sharing identical positional encoding. As shown in Figure 5 (c), our SAP+IIF achieves precise VFX targeting in target regions with nonoverlapping activation zones.

##### Data and Training

###### Dataset Collection

VFX fundamentally manifest as radical spatio-temporal state transformations (e.g., explosion). Despite modern techniques like animation and Computer Graphics Interface (CGI) (Chabanova 2022), modeling such dynamics remains challenging. We introduce a novel pipeline: for any input image, Step1X-Edit (Liu et al. 2025a) produces its modified counterpart to establish boundary frames defining a VFX’s initial and terminal states. This constraint provides strong transformation priors for generative models. The FLF2V framework (Wan, Wang et al. 2025) then synthesizes the final video by compressing VFX production into boundary-

constrained state-transition path search, significantly reducing modeling complexity. Through curated manual selection, we build a comprehensive dataset Omni-VFX spanning 55 distinct VFX across instantaneous environmental shifts, artistic styles, human emotions, and so on, enabling diverse creative applications. For more data details, please refer to Supplement B.

###### Training

Since our training dataset contains only single-VFX without multi-VFX data, empirical observations reveal that standard training fails to achieve controllable multi-VFX generation. We overcome this with a tri-level solution. At the data level, through random cropping and splicing with two videos, and random temporal freezing, we generate pseudo multiVFX videos with corresponding masks. At the scheduler level, Non-Uniform Sampling prioritizes denoising steps ∈ [900,1000](early stage) for spatial control learning with increased batch allocation, while dedicating fewer batches to detail refinement in lower steps ∈ [0,900], motivated by empirical findings that enhanced focus on early denoising accelerates model convergence. At the training strategy level, iterative single to multi-VFX (N = 2) fine-tuning ensures stable convergence and performance gains. For more training details, please refer to Supplement C.

##### Experiments

###### Experimental Setup

Evaluation Metrics Following previous work (Liu et al. 2025b), for single-VFX evaluation, we employ two established metrics: Fr´echet Video Distance (FVD) (Unterthiner et al. 2018) for overall fidelity and Dynamic Degree (Huang et al. 2024b) for motion dynamics. For controllable VFX, we introduce three novel metrics. Regional Dynamic Degree (RDD), which utilizes optical flow and masks, quantifies the strength of motion within the target region, thereby quantifies motion strength within target regions to measure visual impression. Effect Occurrence Rate (EOR), which is computed by inputting both the video and a given prompt template into Gemini2.5 (Comanici et al. 2025) to obtain the answer, measures intended effect trigger frequency, indicating generation reliability. Building upon EOR, Effect Controllability Rate (ECR) assesses spatial precision by verifying VFX confinement to designated areas. Complete metric details appear in Supplement D.

Implementation Details Training employs a CogVideoX5B backbone with LoRA rank of 128 with a total of n = 8 experts, generating 49 × 480 × 720 resolution videos. For loss, β is set to 0.01. We utilize 8 H20 GPUs (96GB) with a batch size of 1 per GPU. We use AdamW (Loshchilov and Hutter 2017) at a constant 10−4 learning rate for 5,000 steps. During inference, DDIM (Nichol and Dhariwal 2021) sampling (50 steps) with CFG (Ho and Salimans 2022) scale 6.0 is applied, which can be perform on a single GPU. Extended details are in Supplement C.

Metrics Methods Cake-ify Crumble Crush Decapitate Deflate Dissolve Eye-pop Harley Inflate Levitate Melt Squish Ta-da Venom Avg. Param.#

Single LoRA 2138 2947 1496 1190 913 1770 1995 3576 1505 1401 2827 1415 1053 4146 2026 132.1M Mix LoRA 1674 2552 1772 1299 886 1995 1725 4496 2042 1006 2748 1225 1240 3923 2041 9.4M

FVD↓

LoRA-MoE 1506 1641 1213 1177 839 1118 1460 3330 1304 736 2512 1561 1064 3339 1628 28.5M

Single LoRA 0.8 0.8 0.0 0.6 0.0 0.8 0.0 1.0 0.8 0.0 0.6 1.0 1.0 1.0 0.60 132.1M Mix LoRA 0.8 0.8 0.0 0.6 0.0 0.8 0.0 1.0 0.8 0.0 0.6 1.0 1.0 1.0 0.60 9.4M

Dynamic Degree ↑

LoRA-MoE 1.0 1.0 0.6 0.6 0.0 0.4 0.0 1.0 1.0 0.0 0.6 1.0 1.0 1.0 0.66 28.5M

- Table 1: Performance comparison on OpenVFX dataset. Param.# represents the average training parameters per effect. And the highest metric values are highlighted in bold, with the second-best underlined.

Methods

RDD↑ EOR↑ ECR↑ Melt Levitate Explode Avg. Melt Levitate Explode Avg. Melt Levitate Explode Avg.

CogVideoX 0.91 0.99 1.11 1.00 0.06 0.09 0.11 0.09 0.00 0.00 0.00 0.00 LTX-Video 0.12 0.11 0.14 0.12 0.05 0.02 0.05 0.04 0.00 0.00 0.00 0.00 Wan-2.1 2.06 1.57 2.38 2.00 0.11 0.02 0.03 0.05 0.02 0.00 0.00 0.01 CogV+CN 3.80 2.39 2.09 2.76 0.95 0.80 0.82 0.86 0.56 0.36 0.70 0.54

Ours 2.69 2.22 3.87 2.93 0.99 0.94 0.99 0.97 0.93 0.83 0.89 0.88

- Table 2: Quantitative results of Single-VFX control generation. We compare Omni-Effects with representative open-source video generation models under three controllable VFX scenarios: Melt, Levitate, and Explode.

###### Quantitative Results

In the following, we evaluate the effectiveness of OmniEffects by comparing it with baseline models on unified and controllable VFX generation tasks.

Unified VFX Generation We evaluate the LoRA-MoE against VFX-specific training LoRA and mix training LoRA for all VFX on the public OpenVFX dataset as detailed in Table 1. LoRA-MoE achieves the best performance across different types of VFX, while significantly reducing the number of trainable parameters. This demonstrates the effectiveness of the designed VFX task-subspace partitioning strategy. Qualitative results are shown in Supplement E.

Controllable VFX Generation To evaluate our model, we perform comprehensive experiments for single- and multi-VFX control, comparing with state-of-the-art methods including (a) CogVideoX (Yang et al. 2024), (b) LTXVideo (HaCohen et al. 2024), (c) Wan2.1 (Wan, Wang et al. 2025), and (d) CogVideoX integrated with ControlNet (CogV+CN). Evaluation targets three spatially localized VFX types—Explode, Melt, and Levitate—to ensure contamination-free assessment.

Single-VFX Control. Table 2 demonstrates baseline methods’ fundamental limitations in synthesizing target VFX and achieving precise spatial control. While CogV+CN can synthesize VFX, it exhibits limited controllability. In comparison, Omni-Effects achieves the best performance with 0.97 EOR and 0.88 ECR, significantly outperforming all baselines in both generation quality and spatial control precision. This validates that our proposed SAP effectively integrates VFX descriptors with spatial triggers without introducing substantial additional training parame-

Melt+Levitate Melt+Explode RDD↑ EOR↑ ECR↑ RDD↑ EOR↑ ECR↑

Methods

CogVideoX 1.80 0.00 0.00 0.96 0.03 0.00 LTX-Video 0.11 0.00 0.00 0.13 0.00 0.00 Wan-2.1 1.93 0.01 0.00 3.10 0.01 0.00 CogV+CN 3.18 0.09 0.05 3.60 0.08 0.08

###### Ours 2.63 0.68 0.41 4.59 0.62 0.50

Table 3: Quantitative results of Multi-VFX generation. Omni-Effects achieves a high success rate of independent control over Multi-VFX.

ters. Qualitative comparisons are shown in Supplement E.

Multi-VFX Control. For multi-VFX generation using two effects combinations, Table 3 shows baseline models consistently failing to generate or spatially control VFX. Omni-Effects achieves precise spatial control over simultaneous VFX. Moreover, Figure 6 demonstrates Omni-Effects’ superiority: when instructed to melt the left chair while levitating the right chair, CogVideoX erroneously applies melting to both objects; CogV+CN correctly renders melting but fails to generate levitation; whereas Omni-Effects simultaneously executes both VFX through spatial condition. This performance validates our proposed IIF’s efficacy in mitigating cross-condition interference. The user study is shown in Supplement E.

Generalization Despite being trained with only N = 2 effects, our model generalizes to diverse mask conditions during inference using the shared Spatial-Condition LoRA,

[Figure 6]

- Figure 6: Qualitative comparison of Multi-VFX generation. The desired outcome requires the left chair to melt while the right levitates simultaneously.

[Figure 7]

- Figure 7: Scalable VFX augmentation. Omni-Effects supports inference-time extension to diverse VFX composition.

thereby extending to the generation of more concurrent control VFX (N > 2). Omni-Effects demonstrates robust extensibility, successfully handling complex effect combinations (Figures 1 (d) and 7), validating its test-time scalable VFX control capability.

###### Ablation Studies

LoRA-MoE Ablation study on expert count n and Topk selection (Table 4) reveals that scaling experts improves generation quality at increased parameter cost. Crucially, our MoE architecture with minimal experts surpasses LoRA baselines (Table 1), demonstrating efficient VFX adaptation through parameter-optimized expert aggregation.

SAP+IIF Ablation study on SAP+IIF reveal critical insights in attention mechanisms corresponding to information flow. Removing SAP attention masks from regions {➁,➅,➆} causes melting artifacts on levitating objects (Figure 8 (b, c)), exposing information leakage, while complete attention induces uncontrolled object melting, demonstrating excessive information interaction degrades control. Strategic masking of {➁,➅,➆} prevents leakage while preserving independent information flow in target regions. Additional ablation studies are detailed in the Supplement F.

##### Conclusion

In this paper, we propose Omni-Effects, a unified framework for generating customized VFX videos. It supports the

Metrics Model Avg. Param.# FVD↓

4 Experts+Top1 1762 18.9 8 Experts+Top2 1628 28.5

Dynamic Degree ↑

4 Experts+Top1 0.65 18.9 8 Experts+Top2 0.66 28.5

Table 4: Ablation study on LoRA MoE settings. Scaling experts improves generation quality, at the expense of more parameters.

[Figure 8]

Figure 8: Effect of different attention masks in SAP. Attention Masks are progressively removed while information flow constraints are relaxed from top to bottom.

creation of diverse VFX, ranging from single-VFX, multiVFX to spatially controllable multi-VFX. To achieve these, our framework integrates two core modules: LoRA-MoE and SAP-IIF. Specifically, the LoRA-MoE module mitigates cross-condition interference arising during mix training of multi-VFX. The SAP module, on the other hand, fuses VFX descriptors with spatial trigger information and tackles cross-condition information leakage via an IIF mechanism. Through the synergistic integration of LoRA-MoE and SAPIIF, Omni-Effects enables precise spatial control and produces high-fidelity multi-VFX composites. We also develop a comprehensive VFX dataset Omni-VFX with a specialized data production pipeline and an evaluation framework tailored for controllable VFX generation to further validate our approach. Extensive experiments demonstrate the robustness of Omni-Effects across complex, multi-condition VFX generation scenarios. Multi-VFX generation represents a domain of substantial practical value coupled with persistent technical challenges. To the best of our knowledge, this work pioneers the first comprehensive framework explicitly addressing this complex problem. Our methodology substantively advances controllable multi-VFX synthesis capabilities while unlocking novel applications across film production, game development, and advertising creatives.

##### References

Bai, J.; Bai, S.; et al. 2023. Qwen Technical Report. arXiv preprint arXiv:2309.16609.

Bai, J.; Xia, M.; Wang, X.; Yuan, Z.; Fu, X.; Liu, Z.; Hu, H.; Wan, P.; and Zhang, D. 2024. Syncammaster: Synchronizing multi-camera video generation from diverse viewpoints. arXiv preprint arXiv:2412.07760.

Bao, F.; Xiang, C.; Yue, G.; He, G.; Zhu, H.; Zheng, K.; Zhao, M.; Liu, S.; Wang, Y.; and Zhu, J. 2024. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233.

Bian, Y.; Zhang, Z.; Ju, X.; Cao, M.; Xie, L.; Shan, Y.; and Xu, Q. 2025. Videopainter: Any-length video inpainting and editing with plug-and-play context control. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, 1–12.

Chabanova, A. 2022. VFX–A new frontier: The impact of innovative technology on visual effects. Ph.D. thesis, University of Westminster.

Chen, C.; Zhu, J.; Feng, X.; Huang, N.; Wu, M.; Mao, F.; Wu, J.; Chu, X.; and Li, X. 2025. S2-Guidance: Stochastic Self Guidance for Training-Free Enhancement of Diffusion Models. arXiv preprint arXiv:2508.12880.

Chen, H.; Xia, M.; He, Y.; Zhang, Y.; Cun, X.; Yang, S.; Xing, J.; Liu, Y.; Chen, Q.; Wang, X.; et al. 2023. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512.

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Ding, N.; Qin, Y.; Yang, G.; Wei, F.; Yang, Z.; Su, Y.; Hu, S.; Chen, Y.; Chan, C.-M.; Chen, W.; et al. 2023. Parameter-efficient fine-tuning of large-scale pre-trained language models. Nature machine intelligence, 5(3): 220– 235.

Dou, S.; Zhou, E.; Liu, Y.; Gao, S.; Zhao, J.; Shen, W.; Zhou, Y.; Xi, Z.; Wang, X.; Fan, X.; et al. 2023. LoRAMoE: Alleviate world knowledge forgetting in large language models via MoE-style plugin. arXiv preprint arXiv:2312.09979.

Fedus, W.; Zoph, B.; and Shazeer, N. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120): 1–39.

HaCohen, Y.; Chiprut, N.; Brazowski, B.; Shalem, D.; Moshe, D.; Richardson, E.; Levin, E.; Shiran, G.; Zabari, N.; Gordon, O.; et al. 2024. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103.

He, Y.; Yang, T.; Zhang, Y.; Shan, Y.; and Chen, Q. 2022. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33: 6840–6851.

Ho, J.; and Salimans, T. 2022. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; Chen, W.; et al. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2): 3.

Huang, L.; Wang, W.; Wu, Z.-F.; Shi, Y.; Dou, H.; Liang, C.; Feng, Y.; Liu, Y.; and Zhou, J. 2024a. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775.

Huang, Z.; He, Y.; Yu, J.; Zhang, F.; Si, C.; Jiang, Y.; Zhang, Y.; Wu, T.; Jin, Q.; Chanpaisit, N.; et al. 2024b. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21807–21818.

Jiang, Z.; Han, Z.; Mao, C.; Zhang, J.; Pan, Y.; and Liu, Y. 2025. Vace: All-in-one video creation and editing. arXiv

- preprint arXiv:2503.07598.

Kong, W.; Tian, Q.; Zhang, Z.; Min, R.; Dai, Z.; Zhou, J.; Xiong, J.; Li, X.; Wu, B.; Zhang, J.; et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603.

Lei, G.; Wang, C.; Zhang, R.; Wang, Y.; Li, H.; and Xu, W. 2025. Animateanything: Consistent and controllable animation for video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 27946–27956.

Li, M.; Yang, T.; Kuang, H.; Wu, J.; Wang, Z.; Xiao, X.; and Chen, C. 2024. ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback. In European Conference on Computer Vision, 129–147. Springer.

Ling, X.; Zhu, C.; Wu, M.; Li, H.; Feng, X.; Yang, C.; Hao, A.; Zhu, J.; Wu, J.; and Chu, X. 2025. Vmbench: A benchmark for perception-aligned video motion generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 13087–13098.

Lipman, Y.; Chen, R. T.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2022. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747.

Liu, S.; Han, Y.; Xing, P.; Yin, F.; Wang, R.; Cheng, W.; Liao, J.; Wang, Y.; Fu, H.; Han, C.; et al. 2025a. Step1xedit: A practical framework for general image editing. arXiv

- preprint arXiv:2504.17761.

Liu, X.; Zeng, A.; Xue, W.; Yang, H.; Luo, W.; Liu, Q.; and Guo, Y. 2025b. VFX Creator: Animated Visual Effect Generation with Controllable Diffusion Transformer. arXiv preprint arXiv:2502.05979.

Loshchilov, I.; and Hutter, F. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Nichol, A. Q.; and Dhariwal, P. 2021. Improved denoising diffusion probabilistic models. In International conference on machine learning, 8162–8171. PMLR.

Peebles, W.; and Xie, S. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, 4195–4205.

Polyak, A.; Zohar, A.; Brown, A.; Tjandra, A.; Sinha, A.; Lee, A.; Vyas, A.; Shi, B.; Ma, C.-Y.; Chuang, C.-Y.; et al. 2024. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720.

Salimans, T.; and Ho, J. 2022. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512.

Seawead, T.; Yang, C.; Lin, Z.; Zhao, Y.; Lin, S.; Ma, Z.; Guo, H.; Chen, H.; Qi, L.; Wang, S.; et al. 2025. Seaweed7b: Cost-effective training of video generation foundation model. arXiv preprint arXiv:2504.08685.

Shazeer, N.; Mirhoseini, A.; Maziarz, K.; Davis, A.; Le, Q.; Hinton, G.; and Dean, J. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538.

Tan, Z.; Liu, S.; Yang, X.; Xue, Q.; and Wang, X. 2025a. Ominicontrol: Minimal and universal control for diffusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 14940–14950.

Tan, Z.; Xue, Q.; Yang, X.; Liu, S.; and Wang, X. 2025b. Ominicontrol2: Efficient conditioning for diffusion transformers. arXiv preprint arXiv:2503.08280.

Teed, Z.; and Deng, J. 2020. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, 402–419. Springer.

Teng, H.; Jia, H.; Sun, L.; Li, L.; Li, M.; Tang, M.; Han, S.; Zhang, T.; Zhang, W.; Luo, W.; et al. 2025. MAGI-1: Autoregressive Video Generation at Scale. arXiv preprint arXiv:2505.13211.

Unterthiner, T.; Van Steenkiste, S.; Kurach, K.; Marinier, R.; Michalski, M.; and Gelly, S. 2018. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717.

Wan, T.; Wang, A.; et al. 2025. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314.

Wang, J.; Zhang, Y.; Zou, J.; Zeng, Y.; Wei, G.; Yuan, L.; and Li, H. 2024. Boximator: Generating rich and controllable motions for video synthesis. arXiv preprint arXiv:2402.01566.

Xu, Z.; Yu, Z.; Zhou, Z.; Zhou, J.; Jin, X.; Hong, F.-T.; Ji, X.; Zhu, J.; Cai, C.; Tang, S.; et al. 2025. Hunyuanportrait: Implicit condition control for enhanced portrait animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 15909–15919.

Yang, Z.; Teng, J.; Zheng, W.; Ding, M.; Huang, S.; Xu, J.; Yang, Y.; Hong, W.; Zhang, X.; Feng, G.; Yin, D.; Gu, X.; Zhang, Y.; Wang, W.; Cheng, Y.; Liu, T.; Xu, B.; Dong, Y.; and Tang, J. 2024. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. ArXiv, abs/2408.06072.

Ye, H.; Zhang, J.; Liu, S.; Han, X.; and Yang, W. 2023. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721.

Yuan, S.; Huang, J.; He, X.; Ge, Y.; Shi, Y.; Chen, L.; Luo, J.; and Yuan, L. 2025. Identity-preserving text-to-video generation by frequency decomposition. In Proceedings of the Computer Vision and Pattern Recognition Conference, 12978–12988.

Zeng, Y.; Wei, G.; Zheng, J.; Zou, J.; Wei, Y.; Zhang, Y.; and Li, H. 2024. Make pixels dance: High-dynamic video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8850–8860.

Zhang, J.; You, J.; Panda, A.; and Goldstein, T. 2025a. Lori: Reducing cross-task interference in multi-task lowrank adaptation. arXiv preprint arXiv:2504.07448.

Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding Conditional Control to Text-to-Image Diffusion Models. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 3813–3824.

Zhang, Y.; Song, Y.; Liu, J.; Wang, R.; Yu, J.; Tang, H.; Li, H.; Tang, X.; Hu, Y.; Pan, H.; et al. 2024. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8069–8078.

- Zhang, Y.; Yuan, Y.; Song, Y.; Wang, H.; and Liu, J. 2025b. Easycontrol: Adding efficient and flexible control for diffusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 19513–19524.
- Zhang, Z.; Xie, J.; Lu, Y.; Yang, Z.; and Yang, Y. 2025c. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer. arXiv preprint arXiv:2504.20690.

Zheng, Z.; Peng, X.; Yang, T.; Shen, C.; Li, S.; Liu, H.; Zhou, Y.; Li, T.; and You, Y. 2024. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404.

Zhou, Y.; Bu, J.; Ling, P.; Zhang, P.; Wu, T.; Huang, Q.; Li, J.; Dong, X.; Zang, Y.; Cao, Y.; et al. 2025. Light-a-video: Training-free video relighting via progressive light fusion. arXiv preprint arXiv:2502.08590.

##### Method

###### Balanced Routing Auxiliary Loss Laux

Drawing inspiration from Switch Transformers (Fedus, Zoph, and Shazeer 2022), we integrate a balanced routing auxiliary loss Laux into LoRA MoE training. Specifically, for a batch B with T tokens, define fi as the fraction of tokens routed to expert i:

1 T x∈B

I{arg maxp(x) = i}, (7)

fi =

and Pi is the mean router probability for expert i:

1 T x∈B

pi (x). (8)

Pi =

The auxiliary loss encourages load balancing through:

n

fi · Pi, (9)

Laux = n

i=1

which reaches its theoretical minimum of 1 when pi(x) = 1/n uniformly (∀x,i). When the gating network outputs an average probability distribution of [1/n,··· ,1/n] for tokens in a batch, Laux achieves its minimum value as n ni=1 1/n · 1/n = 1

###### IIF Attention Mask

The IIF Attention Mask is partitioned into two primary components:

- • Condition Interaction: Within each condition pair ci, comprising text (τe(i) (ei)) and spatial condition

(τs(i) (si)) tokens, tokens attend to each other. However, tokens within one condition pair are masked from all tokens in other condition pairs and from the noisy latent token xt to prevent information leakage.

- • Information Aggregation: The noisy latent token xt at-

N k=1

tends to all text tokens ( τe(k) (ek)

) and to itself. This aggregates textual information for updating its representation. Crucially, xt is masked from all spatial condition tokens ( τs(k) (sk)

N k=1

), preventing direct access and avoiding redundancy.

Formally, the IIF Attention Mask Mij for tokens xi and xj is defined as:

 

0, if xi and xj belong to the same condition ck 0, if xi ∈ xt and xj ∈ xt ∪ τe(k) (ek)

N k=1

Mij =



−∞, otherwise

(10)

##### Dataset

###### Dataset Collection

To augment our dataset, we employ three strategies: a novel first-last frame generation method and integrating external datasets.

Our dataset construction pipeline, outlined in Figure 9, follows a multi-stage generative approach. Firstly, we define target VFX categories spanning diverse styles (e.g., seasonal transformations, claymation, 3D doll rendering). An initial semantic analysis using Qwen (Bai, Bai et al. 2023) classifies input images by content and style. Secondly, for each category, Step1X-Edit (Liu et al. 2025a) generates stylistically consistent frame pairs (initial-final) conditioned on dynamically constructed prompts. These pairs are then analyzed by Qwen to produce descriptive text prompts, which undergo iterative refinement through Wan2.1-14B’s Video Prompt Extender (Wan, Wang et al. 2025) for temporal coherence. Thirdly, the optimized prompts and frames jointly drive Wan2.1-14B to synthesize augmented video sequences. Through this pipeline, we achieve coverage of 55 effect categories while maintaining quality via: (1) automated style-consistency checks, and (2) manual validation of visual fidelity.

###### Dataset Visualization

Our dataset consists of 55 diverse VFX samples, systematically categorized into five groups:

- 1. Environmental Shifts: Including seasonal transitions (spring, summer, autumn, winter) and weather variations (e.g., rain).
- 2. Dynamic Transformations: Featuring simulated phenomena such as explosions and fluid dynamics.
- 3. Artistic Styles: Comprising stylized renderings (e.g., oil painting effects, 3D doll aesthetics).
- 4. Human Emotion Depictions: Capturing facial expressions (e.g., smiles, crying).
- 5. Complex Effects: Integrating multiple visual elements across categories to produce sophisticated composites.

The distribution of samples across categories is illustrated in Figure 12, while representative visualizations are provided in Figure 11.

##### Implementation Details

###### Training Data Augmentation

To address the scarcity of multi-VFX (Multiple Visual Effects) data, we perform data augmentation on single-VFX data. We sample single-VFX data along with their corresponding complete mask data with a probability of 20%; with a 40% probability, we sample single-VFX data and perform random cropping and splicing; and with a 40% probability, we sample two types of VFX and perform random cropping and splicing. During the splicing process, there is a 20% probability of applying temporal freezing to any segment of the spliced video (turning it into a static video with the corresponding mask set to empty, simulating the condition where no VFX is generated). The example of data augmentation is shown in Figure 13.

###### Non-Uniform Timestep Sampling

Through experiments, we observe that for video generation with strict control requirements, the denoising steps in the

(a) First Frame Prompt

Determine what type of subject is in the image

Prompt

[Figure 9]

[Figure 10]

## Step1X-Edit

- (a)Scene transform the setting to a snowy scene
- (b)Object transform the setting to a snowy scene,object covered with snow
- (c)People transform the setting to a snowy scene,change clothes to winter clothes

Question

[Figure 11]

[Figure 12]

(b) Last Frame

(a)Scene (b)Object (c)People

[Figure 13]

[Figure 14]

Answer

(a) First Frame

[Figure 15]

[Figure 16]

(b) Last Frame

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Wan2.1

(c) Prompt

(a) First Frame

[Figure 21]

[Figure 22]

A vibrant woodland path featuring a cheerful gnome standing………and creating a wintry atmosphere.

Wan2.1 Video Prompt Extender

(b) Last Frame

[Figure 23]

###### Figure 9: Synthetic VFX Video Generation via Keyframe Editing and WAN 2.1 Interpolation.

- （a) scene

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

[Figure 40]

[Figure 41]

（c) people

- （b) object

###### Figure 10: Some examples of our dataset curation pipeline.

[Figure 42]

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

[Figure 61]

[Figure 62]

[Figure 63]

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

[Figure 81]

[Figure 82]

[Figure 83]

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

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

###### Figure 11: Some examples of our Omni-VFX dataset.

[Figure 106]

###### Figure 12: Distribution of our Omni-VFX dataset.

early stages of the diffusion model play a critical role, as they determine whether the target region achieves precise control. Traditional uniform timestep sampling requires extensive training to sufficiently optimize the model’s accuracy in the initial steps. To accelerate training convergence, we enhance the focus on early denoising by allocating 3/4 of the batch to the crucial initial steps ([900,1000]), while dedicating the remaining 1/4 to refining details in later steps ([0,900]).

###### Dual-phase Training Strategy

To enhance the model’s capability in controlling both singleVFX and multi-VFX videos’ generation, we adopt a dualstage progressive training strategy during the training phase:

- • Stage 1: We train the model using single-VFX videos with single masks, allowing it to learn how to control a single VFX. This stage runs for 2,000 steps.
- • Stage 2: In addition to single-VFX videos, we introduce multi-VFX videos by combining two VFXs (as described in Sec. C.1) and perform data augmentation on these samples. The model is fine-tuned for an additional 3,000 steps under this setting.

This training approach improves the model’s robustness and enables it to generalize to a larger number of control VFXs (N > 2) during inference.

##### Metrics

Our evaluation framework comprises three components: VFX detection, VFX controllable estimation, and video motion estimation, with the specific architecture illustrated in the Figure 14. These components correspond to the calculation processes of the metrics Effect Occurrence Rate (EOR), Effect Controllability Rate (ECR), and Regional Dynamic Degree (RDD).

###### Visual Effects Detection

To determine whether the desired visual effect is present in the generated videos, we leverage the Gemini 2.5 (Comanici

20%Single-VFX40%Multi-VFXSplicing40%Single-VFXSplicing

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

"Explode it"

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

"Melt it" "Melt it"

"Levitate it" "Levitate it"

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

20% Temporal Freezing

|[Figure 125]<br><br>"Change it anime"|
|---|

"Levitate it"

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

"Melt it"

20% Temporal

"Change it anime"

Freezing

###### Figure 13: Visualization of Data augmentation.

et al. 2025) large multimodal model as an evaluation assistant. Specifically, for each test video, we pair it with a predefined prompt template that explicitly describes the target effect, and input both into Gemini. We then request Gemini to provide a binary answer as to whether the specified effect has appeared in the video. Each query is repeated three times, and the most frequently occurring answer among the three is selected as the final result.

By applying this process to all videos in the evaluation set, we compute the Effect Occurrence Rate (EOR), defined as the percentage of videos in which Gemini confirms the occurrence of the designated visual effect.

###### Visual Effects Controllability Estimation

Controllability estimation is performed only on videos that are identified as having the target effect by the Visual Effects Detection step. For these videos, we compute the absolute pixel-wise difference between the first and last frames and compute the resulting difference binary map. In the masked region, we select the top 80% of difference values; for the non-masked region, the bottom 80%. The mean squared error (MSE) is then computed as Inner Diff and Outer Diff. An effect is regarded as controllable if Inner Diff is below a threshold of 0.5 and Outer Diff is below a threshold of 0.1. The Effect Controllability Rate (ECR) is defined as the fraction of detected-effect videos that meet these criteria.

###### Video Motion Detection

The Regional Dynamic Degree (RDD) measures the strength of motion caused by VFX within mask-specified region of a video. We use the RAFT algorithm (Teed and Deng 2020) to estimate optical flow between consecutive frames. Given a binary mask s for the region of interest, we calculate the mean motion magnitude within this region as follows:

N

1 |s|

1 N

∥Flow(It,It+1)x,y∥2 , (11)

RDD =

t=1

(x,y)∈s

where N is the number of consecutive frame pairs and |s| is the number of pixels in the masked region. A higher RDD

VFX Detection

Input "Melt the right apple"

VLM Prompt:

Input

[Figure 137]

[Figure 138]

You are given a sequence of video

|[Figure 139]| |
|---|---|
| | |

frames, as well as a specified instruction for evaluation......

#### VFX Controllable Estimation

Please respond {“VFX exists”: True/False}......

[Figure 140]

Inner Diff

∩

80% change?

[Figure 141]

| | |
|---|---|
| | |
| | |

[Figure 142]

Outer Diff

∩

80% unchange?

Metric1: Effect Occurrence Rate (EOR)

Metric2: Effect Control Rate (ECR)

Video Motion Estimation

[Figure 143]

∩ Metric3: Regional Dynamic Degreee (RDD)

Optimal Flow Extractor

Input

| | |
|---|---|
| | |
| | |

AND Gate Binarization

###### Figure 14: Flowchart of the metric design for controllable visual effect.

indicates stronger or more dynamic effects within the specified area, enabling precise, region-specific evaluation of effect intensity.

##### Experiments Results Details

###### Qualitative Results of LoRA MoE

Qualitative results of Different LoRA settings are visualized in Figure 15. LoRA-MoE demonstrates superior visual performance.

###### Qualitative Results of single-VFX control

Qualitative results of single-VFX control are visualized in Figure 16. CogV+CN incorrectly causes both cups to explode, while our proposed Omni-Effects explodes the correct cup while keeping the other cup intact.

###### User Study of Multi-VFX Generation

To achieve more reliable evaluation results, we select a subset of videos from the test set for a user study. We choose the state-of-the-art open-source Wan2.1-I2V model as the representative base model, and evaluate it alongside CogV+CN and our proposed method on multi-effect videos. We design two questions: one regarding user preference (i.e., which video the user considers to be of the highest overall quality?), and another asking whether each video demonstrates precise controllability of the specified visual effect, which can directly reflect the effectiveness of our approach. Six professional raters participated in the evaluation, and the results are shown in Figure 17

##### More Ablation Study

###### VFX-combination training

Ablation studies on diverse VFX-combination training regimes (Table 5 and Table 6) reveal inherent effect clustering, demonstrating that compatible VFX-combination training enhances single-VFX generation quality. The proposed LoRA-MoE framework effectively leverages this property to boost performance across all VFX types.

###### Data Augmentation

Ablation study on data augmentation (Figure 18) reveals that models struggle to achieve spatial controllability using single-VFX data alone without augmentation. Our customdesigned data augmentation protocol enables effective controllability for both single- and multi-VFX generation.

###### Timestep Scheduler

Ablation study on different time schedulers (Figure 18) reveals that traditional uniform sampling necessitates extensive training efforts to sufficiently optimize model accuracy in initial steps, whereas our employed non-uniform timestep sampling empirically accelerates model convergence.

###### Training Strategy

Ablation study on different training strategies (Figure 19) reveals that our employed Dual-phase Training Strategy effectively enhances model robustness for multi-VFX generation, with Stage 2 demonstrably mitigating interference artifacts compared to Stage 1.

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

(c)LoRA-MoE(b)Mix-LoRA(a)Single-LoRA(g)LoRA-MoE(f)Mix-LoRA(e)Single-LoRA(d)GT(PiKa)(h)GT(PiKa)

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Melt-like fusion

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Prompt: "Dissolve it"

Prompt: "Levitate it"

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Wrong Feature!

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

Prompt: "Transform into a black Venom"

Prompt: "Eye-pop it"

Figure 15: Qualitative Comparison of Different LoRA Settings.

VFX Nums

Method

Deflate Melt Crumble Dissolve Ta-da Squish Crush Cake-ify

- 1 913 2827 2947 1770 1053 1415 1496 2138

- 2 857 3606 - - - - - 4 779 3957 2324 1559 - - - 8 787 3456 2756 2103 1206 1699 1540 1355

FVD↓ LoRA

14 949 2030 2718 1722 1501 1527 1570 1388 LoRA MoE 14 839 2512 1641 1118 1064 1561 1213 1506

Table 5: Ablation Study on Co-trained with Different VFX Number.

Deflate Deflate+Melt Deflate+Squish FVD↓ 913 857(↓ 6%) 1611(↑ 76%)

Table 6: Ablation Study on Co-trained with Different VFX combination.

crease. We attribute this phenomenon to the lack of real multi-VFX data, which leads to a bias between training and inference. In future work, we plan to collect more high-quality multi-VFX data to improve the stability of our method.

##### Limitation

As the number of combined VFX categories N increases (especially when N > 2), the performance tends to de-

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

###### (d)CogV+CN(c)CogVideoX(b)Wan2.1(a)LTX-Video(e)OmniEffects

"Explode right cup"

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

"Explode right cup"

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

"Explode right cup"

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

"Explode right cup"

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

"Explode right cup"

- Figure 16: Qualitative Comparison of Single-VFX Generation. The desired outcome requires the right cup to explode while the left stays static.

| |20.8%<br><br>1.8% 0.0% 0.0%<br><br>79.2%<br><br>45.5%<br><br>CogV+CN<br><br>Wan2.1-I2V<br><br>Omni-Effects| | |
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

Overall Quality Precise and Controllable

0

10

20

30

40

50

60

70

80

ChoosingPercentage(%)

- Figure 17: User Study for Multi-VFX Generation. OmniEffects exceeds other baseline.

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

"Melt it"

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

- (a)Omni-Effects

"Melt it"

- (b)w/oNUS(c)w/oDA

[Figure 244]

"Melt it"

- Figure 18: Ablation study on Non-uniform Timestep Sampling Steps. NUS stands for Non-Uniform Sampling, and DA stands for Data Augmentation. The training of first stage N = 1 with epoch=70.

[Figure 245]

[Figure 246]

[Figure 247]

"Levitate it" "Melt it"

(b)N=1Input(a)OmniEffects(c)N=1,w/omask⑥

Semantic Leakage

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

###### Figure 19: Ablation Study on Different Training Strategy.

