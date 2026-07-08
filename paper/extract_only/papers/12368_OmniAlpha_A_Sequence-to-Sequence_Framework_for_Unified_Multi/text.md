# arXiv:2511.20211v2[cs.CV]28Apr2026

## OMNIALPHA: Aligning Transparency-Aware Generation via Multi-Task Unified Reinforcement Learning

Hao Yu1∗ Jinglin Wang12∗ Jiabo Zhan1∗ Rui Chen3 Zile Wang1 Huaisong Zhang1 Hongyu Li3 Xinrui Chen1 Yongxian Wei1 Chun Yuan1†

1Tsinghua University 2Beijing University of Posts and Telecommunications 3Beihang University yuh24@mails.tsinghua.edu.cn, yuanc@sz.tsinghua.edu.cn https://github.com/Longin-Yu/OmniAlpha

[Figure 1]

Figure 1. Demonstrating OMNIALPHA’s versatility across a range of RGBA tasks. Our unified model handles: text-to-image generation (Row 1); layer decomposition and object removal (Row 2); referring matting and automatic matting (Row 3).

### Abstract

Transparency-aware generation requires modeling not only RGB appearance but also alpha-based opacity and cross-

∗Equal contribution. †Corresponding author. This work was done when Jinglin was a research intern at Tsinghua University.

layer composition, which are essential for tasks such as image matting, object removal, layer decomposition, and multilayer content creation. However, existing RGBA-related methods remain largely fragmented, with separate pipelines designed for individual tasks. While a unified model is desirable, supervised fine-tuning alone is insufficient, as localized regression objectives cannot directly optimize the compo-

sitional fidelity, alpha-boundary precision, and structural consistency required for high-quality RGBA generation. To address this, we propose OMNIALPHA, a unified multi-task reinforcement learning framework for transparency-aware generation and manipulation. OMNIALPHA combines an end-to-end alpha-aware VAE and a sequence-to-sequence Diffusion Transformer, with a bi-directional layer axis in positional encoding to jointly model multiple RGBA inputs and outputs within a single forward pass. Built on a multi-task SFT cold start, it further performs GRPO-style post-training with layer-aware rewards defined on decoded RGBA outputs, enabling direct optimization of cross-layer coherence and fine transparency details. Experiments across five categories of transparency-aware tasks show that OMNIALPHA consistently outperforms its unified SFT baseline and achieves strong performance against specialized expert models, including a 9.07% relative reduction in RGB L1 on layer decomposition and 74%/68% improvements over conventional matting tools on SAD/Grad for automatic matting.

### 1. Introduction

Diffusion-based generative models [8, 28, 33] have become the dominant paradigm for high-fidelity image synthesis. Powered by scalable latent autoencoding and large-scale text-to-image pre-training, modern systems such as Stable Diffusion [33], FLUX [15], and Qwen-Image [41] have established strong foundations for RGB image generation.

However, many real-world visual creation workflows require representations beyond opaque RGB pixels. In applications such as visual effects, graphic design, image editing, and multi-layer content creation, transparency is a first-class necessity rather than an optional attribute. The RGBA format explicitly models per-pixel opacity through an additional alpha channel, making it essential for preserving fine structures such as hair and fur, representing semi-transparent materials such as glass, smoke, and water, and supporting flexible layer composition.

To address these needs, prior work has largely developed specialized models for individual RGBA-related tasks. Image matting techniques [10, 50] focus on accurate alpha estimation; layer decomposition pipelines [49] separate an image into its constituent foreground and background layers; object removal systems [40] remove specified objects and reconstruct occluded content; and text-to-image workflows like LayerDiffuse [54] explore text-conditioned RGBA synthesis. Despite their effectiveness, these approaches are inherently fragmented. Each task is addressed by a dedicated pipeline, which is not only inefficient but also fails to capture the shared layer-aware structure underlying diverse RGBA workflows.

This fragmentation motivates a unified foundation model for transparency-aware generation and manipulation, anal-

ogous to the recent trend towards unified modeling in the RGB domain [6, 23, 45]. To achieve this unification, scaling up Supervised Fine-Tuning (SFT) across diverse tasks serves as the intuitive next step. Yet, pushing this strategy to its limits exposes a fundamental bottleneck: while effective for instilling basic multi-task capabilities, its localized regression objectives do not directly optimize the properties that matter most in transparency-aware generation, such as cross-layer consistency, alpha-boundary precision, and compositional faithfulness. Consequently, an SFT-only model often plateaus, struggling to capture the complex cross-layer physics and precise boundary fidelity essential for seamless transparency-aware generation.

This limitation suggests moving from purely SFT to outcome-oriented post-training. Reinforcement learning (RL) has recently shown strong promise for aligning RGB diffusion models, with methods such as DanceGRPO [48] and Flow-GRPO [25] demonstrating the value of optimizing generation quality at the trajectory level. However, this emerging RL paradigm remains strictly confined to the RGB space. Fundamentally geared toward opaque pixels, their alignment mechanisms lack the structural awareness needed to process the alpha channel or evaluate multi-layer compositions. This renders existing RL approaches incapable of addressing the complex, layer-based workflows inherent to the RGBA domain, leaving transparency-aware generation largely unexplored.

To bridge this gap, we propose OMNIALPHA, a unified multi-task reinforcement learning framework designed to align transparency-aware generation. Our framework combines an end-to-end alpha-aware VAE with a sequenceto-sequence Diffusion Transformer (DiT), and uses a bidirectional layer coordinate in positional embeddings to jointly process multiple RGBA inputs and outputs within a single forward pass. We first use multi-task SFT as a cold-start stage to establish broad task competence, and then perform Group Relative Policy Optimization (GRPO)-style post-training, driven by rewards tailored specifically to the decoded RGBA layer structures. By harnessing RL to orchestrate the unified generative process, OMNIALPHA explicitly optimizes structural fidelity, compositional consistency, and fine alpha details that are difficult to capture through SFT alone.

Our experiments show that the RL alignment in OMNIALPHA unlocks significant gains over standard supervised training, yielding a 9.07% relative improvement in RGB L1 for layer decomposition over its SFT counterpart. Furthermore, by unifying fragmented task-specific solutions into a single aligned policy, OMNIALPHA eclipses conventional auto-matting baselines by reducing SAD and Grad by 74% and 68%, respectively. These results highlight the strong generalizability of our approach, delivering robust, state-ofthe-art performance across diverse datasets.

In summary, our contributions are as follows:

- • We propose OMNIALPHA, the first reinforcement learning framework for unified multi-task RGBA generation. By shifting the paradigm from localized supervised finetuning to holistic RL alignment, we enable a single policy to master diverse transparency-aware workflows.
- • We introduce structurally-grounded rewards for multilayer compositional alignment, establishing a posttraining paradigm that successfully transcends the performance ceiling of SFT.
- • Experiments show that OMNIALPHA consistently achieves state-of-the-art results across multiple benchmarks, outperforming both its multi-task SFT baseline and specialized expert models.

### 2. Related Work

Transparency-aware Image Generation. Large diffusion models, such as Stable Diffusion [33], have achieved highquality image synthesis for T2I tasks, but their outputs are limited to single-layer RGB images. Recent advances have extended these models to generate RGBA content directly, often using diffusion transformer [28] architectures. Numerous studies have demonstrated the feasibility of multi-layer RGBA generation (PSDiffusion [11], LayerFusion [3], LayerDiffuse [54], ART [30], DreamLayer [12]) and end-to-end generation (AlphaVAE [39], Alfie [32]) using diffusion models. However, these works focus mainly on T2I generation, with limited exploration of more complex visual generation tasks.

Task-Specific and Multi-Task Models. Some methods have extended the RGB/RGBA generation capability of diffusion models, with representative approaches emerging for specific tasks. For instance, dedicated models have emerged for object removal (ObjectDrop [40], ObjectClear [56], PowerPaint [57], DesignEdit [13]), image matting (MAM [18], Matte Anything [51], TeachDiffusionMatting [44], ViTMatte [50], DiffMatte [10], DRIP [22]), and generative layer decomposition (LayerDecomp [49]). While powerful, these approaches are inherently siloed, lacking the flexibility to generalize across multiple generation scenarios. The pursuit of such generalization has led to several unified multi-task models (OmniGen [45], OmniGen2 [42], VisualCloze [23], DreamOmni [43], PUMA [6]), but these frameworks are confined to RGB generation. Consequently, a framework that jointly addresses multi-task RGBA generation remains a relatively underexplored area.

Reinforcement Learning for Diffusion Post-training Generative diffusion models have increasingly adopted posttraining to better align their outputs with target objectives. Early efforts (DPOK [5], DDPO [1]) formulated denoising as a sequential decision problem and optimized sampling

with PPO-style methods [34]. This line of work was later developed under the GRPO paradigm [35] (DanceGRPO [48], Flow-GRPO [25], MixGRPO [20]), with subsequent efforts further improving its efficiency and applicability. However, confined to RGB-centric rewards, these paradigms lack the structural priors necessary for alpha-channel optimization.

### 3. Methodology

In this section, we introduce our proposed OMNIALPHA. Our architecture is built upon the latent diffusion paradigm and comprises two core components. The first is an end-toend, transparency-aware VAE, which is efficiently initialized from a pre-trained RGB autoencoder using an opaque initialization strategy. The second is the denoising backbone, a diffusion transformer. To support our sequence-to-sequence task formulation, we extend the standard 3D rotary positional encoding with a bi-directional layer coordinate (z-axis). This structural adaptation enables the DiT to differentiate and process multiple input and output images concurrently within a single forward pass.

#### 3.1. Formulation and Unified Architecture

Task Formulation. We formulate the multi-layer visual manipulation task as a conditional sequence-to-sequence generation problem. Given a text instruction T and n input RGBA images x1,··· ,xn ∈ RH×W×4, the model M predicts m target RGBA images yˆ1,··· ,yˆm ∈ RH×W×4:

yˆ1,··· ,yˆm = M(x1,··· ,xn;m,T) (1) Operating in a lower-dimensional latent space Z, this generative process is parameterized by a velocity predictor vθ(zt,t,T,{xk}nk=1) within the flow-matching framework, which estimates the velocity field of the latent variable zt at timestep t. In practice, we instantiate this unified paradigm across five categories of transparency-aware tasks: text-toimage generation, object removal, automatic matting, referring matting, and layer decomposition.

End-to-End Transparency-aware VAE. To map the pixel space RH×W×4 to the latent space Z, we build the transparency-aware autoencoder ⟨E,D⟩ following AlphaVAE. It is initialized from a pre-trained RGB VAE via an opaque initialization strategy: the first layer of E zeros out the alpha channel weights to focus initially on RGB features, while the final layer of D is deterministically initialized to predict a fully opaque alpha channel (α = 1.0). Detailed formulations are provided in Appendix A.1.

Sequence-to-Sequence Diffusion Transformer. The denoising backbone vθ is a Multimodal Diffusion Transformer. Following a dual-stream paradigm [41], the n input images are simultaneously processed through two pathways: fed into a Vision-Language Model alongside instruction T for semantic context, and compressed by E into spatial conditions E(xk). To concurrently manage these n inputs and m

[Figure 2]

Figure 2. Overview of the OMNIALPHA Diffusion Transformer architecture and the Cold-Start Stage. Conditioned on a task instruction and n RGBA images, the model simultaneously denoises m target images. To process multiple layers concurrently, we extend positional embeddings with a bi-directional z-axis.

noisy targets, we extend 2D rotary positional embeddings with a bi-directional layer coordinate (z-axis). As illustrated in Figure 2, this z-axis spatially differentiates the sequence by assigning distinct layer indices to the input latents, target latents, and language embeddings (details in Appendix A.2).

#### 3.2. Supervised Cold-Start

Transparency-Aware Alignment of VAE. Before training the diffusion backbone, we align our autoencoder to the 4channel RGBA space. Following AlphaVAE, the training objective incorporates reconstruction, perceptual, two KL divergence terms, and adversarial losses:

L(E,D) = λrecLrec(E,D) + λpercLperc(E,D)

+ λklLkl(E;Eref) + λrefLkl(E;N(0;I))

+ λGANLGAN({E,D};Pd)

(2)

Multi-Task Joint Fine-tuning. With the VAE aligned and the diverse transparency-aware tasks unified under a single sequence-to-sequence paradigm, we train the DiT using the Flow Matching objective [4, 24]. Let T denote the set of all tasks and Dτ the data distribution for task τ ∈ T . For a sampled instance (c,Y ) ∼ Dτ, where c = (T,{xk}nk=1) represents the multimodal condition and Y = (y1,...,ym) is the sequence of m target RGBA images, we encode the targets into a latent sequence Z0 = (z0,1,...,z0,m) via z0,k = E(yk). Given a standard Gaussian noise sequence Z1 = (z1,1,...,z1,m) where z1,k ∼ N(0,I), we define the intermediate state at timestep t ∼ U[0,1] as Zt = (1 − t)Z0 + tZ1. The model vθ is trained to predict the marginal velocity field. To normalize the loss across variable output

lengths m, we average the ℓ2 errors:

LSFT = Eτ∼T ,(c,Y )∼D

τ,t,Z1

1 m

2 2

m k=1 vθ(k)(Zt,t,c) − (z1,k − z0,k)

, (3)

where vθ(k) denotes the velocity prediction corresponding to the k-th target latent.

#### 3.3. Layer-aware Alignment via RL

While SFT provides a foundational generative policy, its density-based regression objective often fails to capture spatially sparse yet visually critical structures, such as finegrained hair boundaries. To address this, we introduce a post-training stage that shifts from localized velocity regression to direct output-space supervision.

Formulating RGBA Generation as an MDP. Our flowmatching framework generates samples by integrating the velocity field backwards from t = 1 to t = 0. We formulate this generative process as a Markov Decision Process (MDP), where each episode corresponds to a complete ordinary differential equation (ODE) rollout from an initial noise sequence Z1 ∼ N(0,I) to the final target latent. Specifically, we discretize the integration trajectory into N steps, denoted by 1 = t0 > t1 > ··· > tN = 0. At step i ∈ {0,...,N − 1}, the state is defined as si ≜ (c,ti,Zt

), comprising the multimodal condition c, the current integration time ti, and the intermediate latent sequence Zt

i

. The action ai is derived from a numerical ODE step guided by the predicted velocity vθ(Zt

i

,ti,c). The transition policy is thus formalized as:

i

πθ(ai | si) ≜ pθ(Zt

,c), (4)

i+1 | Zt

i

[Figure 3]

Figure 3. Demonstration of Layer-Aware Reward Shaping.

which is parameterized by the unified DiT backbone. We adopt an outcome reward formulation, assigning non-zero feedback exclusively at the end of the rollout:

R(si,ai) =

r D(Zt

);c , i = N − 1, 0, otherwise.

N

(5)

Policy Optimization Algorithm. Motivated by DeepSeekMath [35] and DanceGRPO, for each condition c, we sample

a group of G rollouts {Zt(g)

}Gg=1 from the current policy πθ

N

. We optimize the policy model πθ by maximizing the following group-relative objective function:

old

##### J (θ) = E{Z(g)

tN }Gg=1

1 G · N

N−1

G

A(Ag,ρ(ig);ϵ) , (6)

g=1

i=0

where A(A,ρ;ϵ) = min(ρA, clip(ρ,1 − ϵ,1 + ϵ)A). The term ρ(ig) = πθ(a

(g) i |s(ig))

denotes the importance sampling ratio, enabling off-policy optimization from the sampled trajectories. To align the model without a value network, we compute the advantage Ag by normalizing the outcome rewards rg = r D(Zt(g)

πθold(a(ig)|s(ig))

);c within each group:

N

rg − mean({rk}Gk=1) std({rk}Gk=1) + ε

. (7)

Ag =

By emphasizing relative quality, this formulation allows OMNIALPHA to effectively prioritize superior compositional structures and refine sparse visual details across diverse RGBA tasks.

Task-Agnostic Layer-Aware Rewards. To preserve the structural generality of OMNIALPHA, we eschew taskspecific reward engineering in favor of four universal objectives defined on the decoded RGBA sequence Yˆ = (ˆy1,...,yˆm), as shown in Figure 3. Let Y = (y1,...,ym)

denote the ground-truth sequence, and B(f1,··· ,fk;b) represent the alpha-blending operator that composites an ordered sequence of foreground f1,··· ,fk over a background b. To evaluate transparency across diverse contexts, we introduce a background set B = {0,1}, consisting of constant pure black and pure white images. For a given task τ, let K = {1,...,m}, and Kfg,Kbg ⊆ K denote the index sets of foreground and background layers, respectively. The reward suite is formulated as follows:

- • Layer Fidelity (rlayer): Measures RGB similarity ϕ ∈ {SSIM,PSNR} across all valid layers:

rlayer = |K|1

k∈K

E

ϕ,b∈B

ϕ(B(ˆyk;b), B(yk;b))

maxI1,I2 ϕ(I1,I2) . (8)

- • Composition Fidelity (rcomp): Evaluates the globally composited prediction against full-image ground truth:

rcomp = E

ϕ,b∈B

ϕ(B(ˆy1,··· ,yˆm;b), B(y1,··· ,ym;b))

maxI1,I2 ϕ(I1,I2) . (9)

- • Background Structure (rbg): Penalizes structural deviations in background layers using LPIPS [55]:

rbg = (|K 1

bg|+ε)

k∈Kbg

E

b∈B

1−LPIPS(B(ˆyk;b), B(yk;b)) . (10)

- • Foreground Boundary (rfg): Enforces alpha boundary precision and recall for foreground layers:

rfg = |K 1

fg|+ε

k∈Kfg

m(bdryk) ⊙(ˆαk−αk)

1 −

1 m(bdryk)

+ε

1

, (11)

where αˆk,αk represents the alpha channel of yˆk,yk respectively, and m(bdryk) denotes the soft boundary mask constructed from both blurred alpha-band masks given the transparency threshold τ = [τlow,τhigh]:

##### m(bdryk) = clip Blur(I[ˆαk ∈ τ]) + Blur(I[αk ∈ τ]),0,1 . (12)

Table 1. Mapping of tasks to layer-aware reward components. T denotes the text prompt, x and y denote the input and output layers respectively. Kfg and Kbg denote the indices of foreground and background layers in the output sequence.

Task Input Output Kfg Kbg χlayer χcomp χbg χfg

Text-to-Image Generation T y1 {1} ∅ 0 0 0 0 Object Removal x1,x2 y1 ∅ {1} 1 0 1 0 Referring Matting T,x1 y1 {1} ∅ 1 0 0 1 Automatic Matting x1 y1 {1} ∅ 1 0 0 1 Layer Decomposition x1 y1,y2 {2} {1} 1 1 1 1

Adaptive Reward Aggregation. To handle variable output structures across diverse RGBA tasks, we introduce a taskspecific indicator vector χτ = [χlayer,χcomp,χbg,χfg]⊤ ∈

{0,1}4, where χj = 1 if the j-th reward component is applicable to task τ. As detailed in Table 1, OMNIALPHA dynamically activates these reward terms based on the specific multi-layer requirements of each supervised task (e.g., setting χbg = 1 only if Kbg ̸= ∅). In particular, since text-toimage generation lacks objective, pixel-aligned ground-truth references, we set χT2I = 0, effectively bypassing the RL phase for this task. For each sampled trajectory, we independently apply Z-score normalization to all active reward terms to align their scales, yielding normalized scores r˜j. The final trajectory-level outcome reward is then computed as a normalized weighted sum:

λjχjr˜j j∈R λjχj + ε

r = j∈R

, (13)

where R = {layer,comp,bg,fg} and λj are the corresponding weights. In our implementation, we weight all enabled components equally, setting λlayer = λcomp = λbg = λfg = 1.0.

### 4. Experiments 4.1. Setup

Preparation of Training Data. To support both supervised fine-tuning and reinforcement learning, we construct a unified layer-aware training corpus covering the included tasks. For text-to-image generation, we follow AlphaVAE to construct 8,124 high-quality RGBA images using the same data preparation pipeline. Specifically, the data are derived from the same ten public image matting datasets used in AlphaVAE and converted into four-channel RGBA format by combining each foreground image with its corresponding alpha matte. For object removal, we use the training split of OBER-Dataset, which is introduced in ObjectClear [56]. For automatic matting, we adopt the data construction protocol of AIM [17] and construct a training set of 8,710 images. The data are derived from Composition-1k [47], HAtt [31], and AM-2k [16], together with BG-20k [16] backgrounds under the composition strategy of AIM. For referring matting, we adopt RefMatte [19], which contains 47,500 images and 118,749 expression-region pairs with high-quality alpha mattes. For layer decomposition, we use PrismLayersReal and PrismLayersPlus [2]. Among the approximately 100,000 cases provided by the two datasets as shown in Figure 4, we uniformly sample 12,000 cases for our training.

Training Implementation. We adopt Qwen-ImageEdit [41] as our base model and employ a three-stage training paradigm. We first adapt the pre-trained RGB VAE from Qwen-Image-Edit into an alpha-aware (RGBA) VAE by finetuning it on our RGBA image dataset for 32k steps. This stage uses a global batch size of 16 and the AdamW optimizer [26] (β1=0.9,β2=0.999). We use a base learning rate of 1.5 × 10−5 with a 5% linear warmup, followed by

Task Distribution

Text-to-Image (7.1%)

Layer Decomposition (10.5%)

Object Removal (33.2%)

Referring Matting (41.5%)

Automatic Matting (7.6%)

Figure 4. Distribution of the training data across the five tasks in OmniAlpha. The numbers of training samples for Text-to-Image, Object Removal, Automatic Matting, Referring Matting, and Layer Decomposition are 8,124, 37,994, 8,710, 47,500, and 12,000, respectively.

a cosine decay schedule. With the fine-tuned RGBA VAE weights frozen, we train the DiT backbone, also initialized from Qwen-Image-Edit, for 15k cold-start steps and 150 RL steps. This stage uses a batch size of 64 with the AdamW optimizer (β1=0.9,β2=0.999). For parameter-efficient finetuning, we apply LoRA [9] with a rank of 256 to all attention weights and MLP layers, using a constant learning rate of 5 × 10−5. All models are trained on 64 NVIDIA H20 GPUs.

#### 4.2. Main Experimental Results

OMNIALPHA is evaluated on five transparency-aware tasks, including text-to-image generation, referring matting, object removal, automatic matting, and layer decomposition, using both public benchmarks and our curated test split. Across these heterogeneous settings, the model delivers state-of-theart or competitive performance against strong task-specific baselines, with especially clear advantages on transparencysensitive tasks such as referring and automatic matting, while remaining robust on synthesis, removal, and decomposition. Overall, the results indicate that a shared layer-aware RGBA representation transfers effectively across tasks within a single unified architecture, and that RL further strengthens the model on tasks requiring stronger structural alignment and perceptual fidelity.

Text-to-Image. Table 3 reports transparent text-to-image generation results on AIM-500 [17]. Compared with specialized baselines such as LayerDiffuse [54] and AlphaVAE [39], our unified OMNIALPHA remains highly competitive. The SFT variant achieves the best FID (67.891) and CLIPScore [7] (0.852), while the RL variant, despite unseen during RL post-training, still achieves the best ImageReward [46] (1.060). This shows that OMNIALPHA generalizes well without task-specific architectures or alignment tax.

Referring Matting. Referring matting is evaluated on RefMatte-RW100 [19] against specialized baselines such as CLIPMat [19]. As shown in Table 4, both variants of OMNIALPHA outperform all prior methods by a large margin. In particular, RL further improves upon SFT and achieves

Table 2. Object removal results on RORD-Val, OBER-Test, and OBER-Wild.

RORD-Val OBER-Test OBER-Wild PSNR ↑ PSNR-BG ↑ LPIPS ↓ CLIP ↓ PSNR ↑ PSNR-BG ↑ LPIPS ↓ CLIP ↓ ReMOVE ↑

Method

SDXL-INP [29] 20.230 24.830 0.204 0.087 22.420 25.770 0.143 0.077 0.804 PowerPaint [57] 21.460 24.620 0.180 0.065 22.760 24.670 0.154 0.073 0.854 Attentive Eraser [37] 22.170 24.590 0.188 0.064 25.700 27.080 0.120 0.044 0.901 RORem [21] 22.490 24.100 0.294 0.063 24.510 25.280 0.129 0.046 0.903

ObjectClear [56] 26.400 30.750 0.080 0.035 32.980 35.140 0.035 0.010 0.917 Ours (SFT) 26.214 29.005 0.141 0.036 30.736 32.315 0.098 0.019 0.914 Ours (RL) 26.315 29.367 0.147 0.046 31.389 32.903 0.095 0.016 0.915

[Figure 4]

GT

AM

input

LD

best result

second best

main w/o rlayer w/o rcomp w/o rbg w/o rfg

Figure 5. Qualitative ablation cases on Automatic Matting (AM) and Layer Decomposition (LD). The legend entries ”best result” and ”second best” indicate the variants with the highest and second-highest metric values, respectively. Removing individual reward terms leads to visible performance degradation on both tasks.

Table 3. Text-to-image results on AIM-500.

AIM-500 FID ↓ CLIP-Score ↑ LAION-AES ↑ ImageReward ↑

Method

GT - 0.837 5.234 0.611 LayerDiffuse [54] 78.499 0.836 5.405 0.853 AlphaVAE [39] 72.080 0.842 5.453 0.931 Ours (SFT) 67.891 0.852 5.317 1.050 Ours (RL) 68.829 0.850 5.323 1.060

the best results on all three metrics, with 14.768 SAD, 0.022 MSE, and 0.026 MAD. These results demonstrate that OMNIALPHA can accurately localize the referred target and recover high-quality alpha mattes directly from language.

Object Removal. Object removal is evaluated on RORDVal, OBER-Test, and OBER-Wild against representative specialized baselines. Table 2 shows that both our SFT and RL models consistently outperform a wide range of established methods, while RL further improves over SFT on

Table 4. Referring matting results on RefMatte-RW100.

RefMatte-RW100 SAD ↓ MSE ↓ MAD ↓

Method

MDETR [14] 131.580 0.068 0.075 CLIPSeg [27] 211.860 0.118 0.122 MAM [18] 120.100 0.066 0.068 CLIPMat [19] 85.830 0.047 0.050

Ours (SFT) 15.029 0.027 0.030 Ours (RL) 14.768 0.022 0.026

the more challenging OBER-Test benchmark and on the out-of-domain OBER-Wild set. Although the highly specialized ObjectClear model achieves the best overall performance, our unified approach remains highly competitive across benchmarks. On OBER-Wild, SFT and RL achieve

0.914 and 0.915 ReMOVE, respectively, remaining close to the task-specific ObjectClear (0.917), which indicates strong generalization under distribution shift.

Automatic Matting. Automatic matting is evaluated on AIM-500 [17] in the mask-free setting, where the model directly identifies the visible subject and predicts its alpha matte without trimap or mask guidance. Table 5 shows that both variants outperform all prior methods by a large margin. RL achieves the best overall performance, while SFT obtains the second best score. Compared with the strongest baseline, SMat [52], OMNIALPHA significantly improves foreground localization and alpha reconstruction.

Table 5. Automatic matting results on AIM-500.

AIM-500 SAD ↓ MSE ↓ MAD ↓ Grad ↓ Conn ↓

Method

GFM [16] 52.660 0.0213 0.0313 46.110 52.690 AIM [17] 48.090 0.0183 0.0285 47.580 21.740 SMat [52] 34.300 0.0129 0.0203 31.490 13.980

Ours (SFT) 9.245 0.0109 0.0185 10.677 9.071 Ours (RL) 9.089 0.0105 0.0182 10.099 8.930

Layer Decomposition. Layer Decomposition is evaluated on OBER-Test-decompose, derived from OBERTest [56], where each composed image is decomposed into editable foreground and background layers with explicit transparency. Table 6 shows that RL achieves the best Alpha soft IoU (0.7069), while SFT also attains a strong result of 0.6831, outperforming both LayerD [38] and Qwen-ImageLayered [53] on this metric. In terms of RGB L1, LayerD remains the strongest, and Qwen-Image-Layered also outperforms our model. Nevertheless, RL substantially improves over SFT, indicating that RL post-training improves reconstruction quality while further strengthening alpha-aware decomposition.

Table 6. Decomposition results on OBER-Test-decompose.

OBER-Test-decompose

Method

RGB L1 ↓ Alpha soft IoU ↑

LayerD [38] 0.0173 0.5645 Qwen-Image-Layered [53] 0.0977 0.6589

Ours (SFT) 0.1135 0.6831 Ours (RL) 0.1032 0.7069

#### 4.3. Ablation Study

To better understand the role of each component in OmniAlpha, we perform reward ablations on all RL objectives.

Text-to-Image Generation. Table 7 shows that the full RL model performs best overall on AIM-500 [17], achieving the best FID, CLIP-Score [7], and ImageReward [46]. Removing any reward term degrades performance, confirming

Table 7. Ablation study on text-to-image generation.

AIM-500 FID ↓ CLIP-Score ↑ LAION-AES ↑ ImageReward ↑ OmniAlpha RL 68.829 0.850 5.323 1.060 w/o rlayer 69.663

Method

0.846

5.315

1.056

(+0.834)

(-0.004)

(-0.008)

(-0.004)

###### 5.328

w/o rfg 68.904

0.848

1.059

(+0.075)

(-0.002)

(+0.005)

(-0.001)

w/o rbg 69.558

0.848

5.313

1.054

(+0.729)

(-0.002)

(-0.010)

(-0.006)

w/o rcomp 69.627

0.849

5.317

1.058

(+0.798)

(-0.001)

(-0.006)

(-0.002)

the effectiveness of the full reward design. In particular, rlayer is the most critical, while the degradation caused by removing rbg and rcomp highlights the importance of boundarybackground interaction and compositional realism in transparent image generation.

Referring Matting. Table 8 shows that the full RL model achieves the best performance on RefMatte-RW100 across all metrics, confirming the effectiveness of the full reward design for text-guided alpha prediction. In particular, rlayer and rfg are the most critical, as their removal causes the largest drops, while the degradation caused by removing rbg is also substantial. Although removing rcomp has a relatively smaller effect, it remains consistently worse than the full model.

Table 8. Ablation study on RefMatte-RW100.

RefMatte-RW100 SAD ↓ MSE ↓ MAD ↓ OmniAlpha RL 14.768 0.0217 0.0257 w/o rlayer 16.251

Method

0.0307

0.0326

(+1.483)

(+0.0090)

(+0.0069)

w/o rfg 16.273

0.0305

0.0327

(+1.505)

(+0.0088)

(+0.0070)

w/o rbg 16.268

0.0307

0.0327

(+1.500)

(+0.0090)

(+0.0070)

0.0303

0.0322

w/o rcomp 16.048

(+1.280)

(+0.0086)

(+0.0065)

Object Removal. Table 9 shows that the full RL model performs best overall on object removal, confirming the effectiveness of the full reward design. Removing any reward term degrades performance, with rlayer and rbg being the most critical. Removing rfg also causes consistent degradation, whereas removing rcomp has a relatively smaller effect but is still worse than the full model on the paired benchmarks.

Automatic Matting. Table 10 shows that the full OMNIALPHA RL model achieves the best performance on AIM-500 across all metrics, confirming the effectiveness of the full reward design. Removing any reward term degrades performance. In particular, removing rfg, rbg, or rcomp causes more noticeable drops on SAD, Grad, and Conn, while removing rlayer leads to the smallest degradation but remains consistently worse than the full model. Figure 5 provides qualitative examples that are consistent with these quanti-

Table 9. Ablation study on object removal.

RORD-Val OBER-Test OBER-Wild

Method

PSNR ↑ PSNR-BG ↑ LPIPS ↓ CLIP ↓ PSNR ↑ PSNR-BG ↑ LPIPS ↓ CLIP ↓ ReMOVE ↑ OmniAlpha RL 26.315 29.367 0.1467 0.0460 31.389 32.903 0.0946 0.0157 0.91481 w/o rlayer 26.048

0.0466

31.193

32.719

0.0948

0.0160

0.91377

29.142

0.1469

(-0.00104) w/o rfg 26.032

(+0.0002)

(+0.0006)

(-0.196)

(-0.184)

(+0.0002)

(+0.0003)

(-0.267)

(-0.225)

29.120

0.1471

0.0468

31.243

32.778

0.0950

0.0160

0.91372

(-0.00109) w/o rbg 26.020

(-0.283)

(-0.247)

(+0.0004)

(+0.0008)

(-0.146)

(-0.125)

(+0.0004)

(+0.0003)

0.91376

29.106

0.1477

0.0464

31.263

32.804

0.0953

0.0158

(-0.00105) w/o rcomp 26.058

(+0.0001)

(-0.295)

(-0.261)

(+0.0010)

(+0.0004)

(-0.126)

(-0.099)

(+0.0007)

29.142

0.1473

0.0462

31.269

32.805

0.0952

0.0159

0.91390

(-0.257)

(-0.225)

(+0.0006)

(+0.0002)

(-0.120)

(-0.098)

(+0.0006)

(+0.0002)

(-0.00091)

Table 10. Ablation study on automatic matting.

AIM-500 SAD ↓ MSE ↓ MAD ↓ Grad ↓ Conn ↓ OmniAlpha RL 9.089 0.0105 0.0182 10.099 8.930 w/o rlayer 9.114

Method

0.01063

0.0183

10.396

8.957

(+0.0250)

(+0.00013)

(+0.00005)

(+0.2977)

(+0.0267)

w/o rfg 9.187

0.01064

0.0184

10.323

9.020

(+0.0984)

(+0.00014)

(+0.00020)

(+0.2239)

(+0.0901)

w/o rbg 9.152

0.01067

0.0183

10.412

8.998

(+0.0634)

(+0.00017)

(+0.00013)

(+0.3127)

(+0.0680)

w/o rcomp 9.180

0.01071

0.0184

10.399

9.028

(+0.0913)

(+0.00020)

(+0.00018)

(+0.3004)

(+0.0976)

tative results. These results indicate that all reward terms contribute to accurate alpha prediction.

Layer Decomposition. Table 11 shows that the full OMNIALPHA RL model performs best on OBER-TestDecompose for both RGB L1 and Alpha soft IoU. Removing any reward term degrades performance, indicating that all components are beneficial. In particular, rlayer is the most critical, while rfg and rcomp also have clear impact. Figure 5 provides qualitative examples consistent with these trends.

Table 11. Ablation study on layer decomposition. Method

###### OBER-Test-Decompose

RGB L1 ↓ Alpha soft IoU ↑ OmniAlpha RL 0.1032 0.7069 w/o rlayer 0.1096

0.6918

(+0.0065)

(-0.0151)

w/o rfg 0.1069

0.6927

(+0.0037)

(-0.0141)

0.6965

w/o rbg 0.1065

(+0.0034)

(-0.0103)

w/o rcomp 0.1078

0.6943

(+0.0046)

(-0.0126)

Ablation results reveal that the four reward terms are complementary and jointly necessary for robust multi-task RGBA alignment. Although their importance varies across tasks, removing any single term consistently degrades performance on at least one benchmark or metric, while the full reward design yields the strongest overall objective. In particular, rlayer is essential for structural consistency, rfg is especially beneficial for foreground-sensitive tasks such as text-guided matting, rbg is crucial for background recovery

in object removal, and rcomp further improves cross-layer compositional fidelity.

### 5. Conclusion

In this paper, we introduced OMNIALPHA, a unified multitask reinforcement learning framework for transparencyaware generation and manipulation. By combining an alpha-aware VAE, a sequence-to-sequence Diffusion Transformer, and GRPO-style post-training with layer-aware rewards, OMNIALPHA enables a single model to handle diverse RGBA workflows in a unified manner. Experiments across five categories of transparency-aware tasks show that unified RL alignment consistently improves over multi-task supervised fine-tuning and achieves strong performance against specialized expert systems. These results highlight the promise of moving from task-specific RGBA pipelines to general-purpose transparency-aware foundation models. We hope this work motivates future research on reinforcement learning for layer-aware generation, compositional reasoning, and more flexible multi-layer visual creation.

### References

- [1] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning, 2024. 3
- [2] Junwen Chen, Heyang Jiang, Yanbin Wang, Keming Wu, Ji Li, Chao Zhang, Keiji Yanai, Dong Chen, and Yuhui Yuan. Prismlayers: Open data for high-quality multi-layer transparent image generative models, 2025. 6
- [3] Yusuf Dalva, Yijun Li, Qing Liu, Nanxuan Zhao, Jianming Zhang, Zhe Lin, and Pinar Yanardag. Layerfusion: Harmonized multi-layer text-to-image generation with generative priors, 2024. 3
- [4] Patrick Esser, Sumith Kulal, A. Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for highresolution image synthesis. International Conference on Machine Learning, 2024. 4
- [5] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Moham-

- mad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models, 2023. 3
- [6] Rongyao Fang, Chengqi Duan, Kun Wang, Hao Li, Hao Tian, Xingyu Zeng, Rui Zhao, Jifeng Dai, Hongsheng Li, and Xihui Liu. Puma: Empowering unified mllm with multi-granular visual generation, 2024. 2, 3
- [7] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning, 2022. 6, 8
- [8] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 2
- [9] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. 6
- [10] Yihan Hu, Yiheng Lin, Wei Wang, Yao Zhao, Yunchao Wei, and Humphrey Shi. Diffusion for natural image matting, 2024.

- 2, 3

[11] Dingbang Huang, Wenbo Li, Yifei Zhao, Xinyu Pan, Yanhong Zeng, and Bo Dai. Psdiffusion: Harmonized multi-layer image generation via layout and appearance alignment, 2025.

- 3

- [12] Junjia Huang, Pengxiang Yan, Jinhang Cai, Jiyang Liu, Zhao Wang, Yitong Wang, Xinglong Wu, and Guanbin Li. Dreamlayer: Simultaneous multi-layer generation via diffusion mode, 2025. 3
- [13] Yueru Jia, Yuhui Yuan, Aosong Cheng, Chuke Wang, Ji Li, Huizhu Jia, and Shanghang Zhang. Designedit: Multi-layered latent decomposition and fusion for unified & accurate image editing, 2024. 3
- [14] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetr – modulated detection for end-to-end multi-modal understanding, 2021. 7
- [15] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. 2
- [16] Jizhizi Li, Jing Zhang, Stephen J. Maybank, and Dacheng Tao. Bridging composite and real: Towards end-to-end deep image matting, 2021. 6, 8
- [17] Jizhizi Li, Jing Zhang, and Dacheng Tao. Deep automatic natural image matting, 2021. 6, 8
- [18] Jiachen Li, Jitesh Jain, and Humphrey Shi. Matting anything,

2023. 3, 7

- [19] Jizhizi Li, Jing Zhang, and Dacheng Tao. Referring image matting, 2023. 6, 7
- [20] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Yiming Cheng, Miles Yang, Zhao Zhong, and Liefeng Bo. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde, 2026. 3
- [21] Ruibin Li, Tao Yang, Song Guo, and Lei Zhang. Rorem: Training a robust object remover with human-in-the-loop,

2025. 7

- [22] Xiaodi Li, Zongxin Yang, Ruijie Quan, and Yi Yang. Drip: Unleashing diffusion priors for joint foreground and alpha prediction in image matting. Advances in Neural Information Processing Systems 37, 2024. 3
- [23] Zhong-Yu Li, Ruoyi Du, Juncheng Yan, Le Zhuo, Zhen Li, Peng Gao, Zhanyu Ma, and Ming-Ming Cheng. Visualcloze: A universal image generation framework via visual in-context learning, 2025. 2, 3
- [24] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. 4
- [25] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl, 2025. 2, 3
- [26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv: 1711.05101, 2017. 6
- [27] Timo Lüddecke and Alexander S. Ecker. Image segmentation using text and image prompts, 2022. 7
- [28] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. 2, 3
- [29] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023. 7
- [30] Yifan Pu, Yiming Zhao, Zhicong Tang, Ruihong Yin, Haoxing Ye, Yuhui Yuan, Dong Chen, Jianmin Bao, Sirui Zhang, Yanbin Wang, Lin Liang, Lijuan Wang, Ji Li, Xiu Li, Zhouhui Lian, Gao Huang, and Baining Guo. Art: Anonymous region transformer for variable multi-layer transparent image generation, 2025. 3
- [31] Yu Qiao, Yuhao Liu, Xin Yang, Dongsheng Zhou, Mingliang Xu, Qiang Zhang, and Xiaopeng Wei. Attention-guided hierarchical structure aggregation for image matting. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 6
- [32] Fabio Quattrini, Vittorio Pippi, Silvia Cascianelli, and Rita Cucchiara. Alfie: Democratising rgba image generation with no $$$, 2024. 3
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 2, 3
- [34] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. 3
- [35] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. 3, 5
- [36] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 1

- [37] Wenhao Sun, Benlei Cui, Xue-Mei Dong, and Jingqun Tang. Attentive eraser: Unleashing diffusion model’s object removal potential via self-attention redirection guidance, 2025. 7

- [38] Tomoyuki Suzuki, Kang-Jun Liu, Naoto Inoue, and Kota Yamaguchi. Layerd: Decomposing raster graphic designs into layers, 2025. 8
- [39] Zile Wang, Hao Yu, Jiabo Zhan, and Chun Yuan. Alphavae: Unified end-to-end rgba image reconstruction and generation with alpha-aware representation learning. arXiv preprint arXiv: 2507.09308, 2025. 3, 6, 7
- [40] Daniel Winter, Matan Cohen, Shlomi Fruchter, Yael Pritch, Alex Rav-Acha, and Yedid Hoshen. Objectdrop: Bootstrapping counterfactuals for photorealistic object removal and insertion, 2024. 2, 3
- [41] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. 2, 3, 6, 1
- [42] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation, 2025. 3
- [43] Bin Xia, Yuechen Zhang, Jingyao Li, Chengyao Wang, Yitong Wang, Xinglong Wu, Bei Yu, and Jiaya Jia. Dreamomni: Unified image generation and editing, 2025. 3
- [44] Tianyi Xiang, Weiying Zheng, Yutao Jiang, Tingrui Shen, Hewei Yu, Yangyang Xu, and Shengfeng He. Teaching diffusion models to ground alpha matte. Transactions on Machine Learning Research, 2025. 3
- [45] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation,

2024. 2, 3

- [46] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: learning and evaluating human preferences for text-to-image generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 15903–15935, 2023. 6, 8
- [47] Ning Xu, Brian Price, Scott Cohen, and Thomas Huang. Deep image matting, 2017. 6
- [48] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, and Ping Luo. Dancegrpo: Unleashing grpo on visual generation, 2025. 2, 3
- [49] Jinrui Yang, Qing Liu, Yijun Li, Soo Ye Kim, Daniil Pakhomov, Mengwei Ren, Jianming Zhang, Zhe Lin, Cihang Xie, and Yuyin Zhou. Generative image layer decomposition with visual effects, 2024. 2, 3
- [50] Jingfeng Yao, Xinggang Wang, Shusheng Yang, and Baoyuan Wang. Vitmatte: Boosting image matting with pretrained plain vision transformers, 2023. 2, 3

- [51] Jingfeng Yao, Xinggang Wang, Lang Ye, and Wenyu Liu. Matte anything: Interactive natural image matting with segment anything models, 2024. 3
- [52] Zixuan Ye, Wenze Liu, He Guo, Yujia Liang, Chaoyi Hong, Hao Lu, and Zhiguo Cao. Unifying automatic and interactive matting with pretrained vits. In Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 8
- [53] Shengming Yin, Zekai Zhang, Zecheng Tang, Kaiyuan Gao, Xiao Xu, Kun Yan, Jiahao Li, Yilei Chen, Yuxiang Chen, Heung-Yeung Shum, Lionel M. Ni, Jingren Zhou, Junyang Lin, and Chenfei Wu. Qwen-image-layered: Towards inherent editability via layer decomposition, 2025. 8
- [54] Lvmin Zhang and Maneesh Agrawala. Transparent image layer diffusion using latent transparency, 2024. 2, 3, 6, 7
- [55] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 5
- [56] Jixin Zhao, Zhouxia Wang, Peiqing Yang, and Shangchen Zhou. Precise object and effect removal with adaptive targetaware attention, 2026. 3, 6, 7, 8
- [57] Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting, 2024. 3, 7

### A. Details of Model Architecture

#### A.1. Opaque Initialization of VAE

Formally, let (WE

0 ,bE

0 ) be the parameters of the first convolutional layer of Eref, with WE

ref

ref

c and bE

0 ∈ Rk×k×3×D

ref

c. Let (WD

L ,bD

L ) be the parameters of the final convolutional layer of Dref, with WD

0 ∈ RD

ref

ref

ref

′×k′×Df×3 and bD

L ∈ Rk

ref

L ∈ R3. The corresponding layers of the new 4-channel VAE, (W0E,bE0) and (WLD,bDL), are initialized as specified in Equation 14:

ref



W0E[:,:,1 : 3,:] = WE

0 (Copy RGB weights) W0E[:,:,4,:] = 0 (0-init alpha weights) bE0 = bE

ref

0 (Copy bias) WLD[:,:,:,1 : 3] = WD

ref



(14)

L (Copy RGB weights) WLD[:,:,:,4] = 0 (0-init alpha weights) bDL[1 : 3] = bD

ref

L (Copy RGB biases) bDL[4] = 1 (Set as opaque)

ref



where W0E and WLD have new shapes Rk×k×4×D

c and Rk

′×k′×Df×4, respectively. The bias bDL is a 4-dimensional vector. This opaque initialization provides a stable starting point for fine-tuning.

#### A.2. RoPE with Bidirectional Z-axis

Inspired by [41], to unify multimodal inputs and multiple outputs into a single sequence, we equip a bi-directionally extendable layer axis to the standard 2D Rotary Position Embedding (RoPE) [36]. This z-axis differentiates components by assigning unique indices. The n input latents occupy nonnegative indices from z = 0 to n − 1. The m target latents are assigned negative indices from z = −1 to −m. The contextual embeddings output by the language encoder are assigned distinct positive indices z ≥ n, spatially separating them from the input image latents. This coordinate formulation effectively transforms the multi-image generation task into a unified sequence-to-sequence denoising paradigm.

The implementation leverages the inherent translation invariance property of Rotary Positional Embeddings. Fundamentally, the dot product of two RoPE-encoded feature vectors, q and k, at positions pi = (xi,yi,zi) and pj = (xj,yj,zj), depends solely on their relative spatial and layer distances:

⟨R(q,pi),R(k,pj)⟩ = g(q,k,pi − pj). (15)

Consequently, shifting the layer index z by a constant scalar S for all tokens preserves the relative positional relationships and, by extension, the attention scores. This property can be formally expressed as:

R(h,x,y,z) · R(h′,x′,y′,z′)−1

= R(h,x,y,z + S) · R(h′,x′,y′,z′ + S)−1,

(16)

where h represents the hidden states.

In our sequence-to-sequence formulation, target images are assigned negative layer indices z ∈ {−m,...,−1}, where m is the number of generated target frames. However, the pre-trained Qwen-Image-Edit backbone typically utilizes pre-computed frequency tables defined over a non-negative domain. To align our bi-directional indexing strategy with the pre-defined frequency configurations of the base model, we implement a global index shift operation.

Let zraw denote the logical layer index defined in Section 3. Specifically:

 

{−m,...,−1} for target latents, {0,...,n − 1} for input latents, {n,...} for VLM condition tokens.

zraw ∈



(17) To ensure all indices map to the valid domain of the pretrained frequency encodings, we define the implementation index zimpl as:

zimpl = zraw + Soffset, where Soffset ≥ m. (18)

In practice, we set Soffset = m. This transformation allows OMNIALPHA to concurrently process multiple inputs and outputs while seamlessly utilizing the optimized RoPE functions and frequency computations inherited from QwenImage-Edit.

### B. More Results

Figures 6–64 present a set of randomly sampled results. Each caption corresponds to the prompt provided to OMNIALPHA. Images outlined in blue denote inputs, while those outlined in purple represent predictions.

- • Figures 6–19 showcase the text-to-image task category.
- • Figures 20–29 show results for the referring matting task category.
- • Figures 30–44 display results for the object removal task category.
- • Figures 45–54 show results for the automatic matting task category.
- • Figures 55–64 illustrate the layer decomposition task category.

[Figure 5]

###### Figure 6. A light brown plush teddy bear with a red plaid bow tie sits upright with arms slightly spread and legs apart.

[Figure 6]

###### Figure 7. A decorative ceramic teacup with a swirling handle, adorned with floral patterns in purple, green, and blue, overflowing with fresh red strawberries topped with green leaves.

[Figure 7]

###### Figure 8. A bright orange modern armchair with a solid blocky design and four angled legs.

[Figure 8]

###### Figure 9. A decorative arrangement featuring black scissors, a cream candle, red berries on a twig, a pinecone, charcoal pieces, and white sticks bound together with twine.

[Figure 9]

###### Figure 10. A rustic wooden armchair with a simple rectangular backrest, thick armrests, and sturdy legs showing natural wood grain and knots.

[Figure 10]

###### Figure 11. A small potted plant with vibrant green grass-like leaves in a simple white ceramic pot.

[Figure 11]

###### Figure 12. A close-up of an antelope’s head and neck, showing its brown fur, large ears, and spiraled horns.

[Figure 12]

###### Figure 13. A single pink rose with slightly wilted petals and green leaves, standing upright in a clear glass vase filled with water.

[Figure 13]

###### Figure 14. Vivid orange and yellow flames rising upward with dynamic, flowing shapes.

[Figure 14]

###### Figure 15. A clear, round water droplet with a smooth, glossy surface.

[Figure 15]

###### Figure 16. A modern wooden cabinet with light-colored drawers and minimalist handles.

[Figure 16]

###### Figure 17. A tiger with orange fur and black stripes stares directly forward, its head and upper torso visible.

[Figure 17]

###### Figure 18. A person with shoulder-length wavy reddish-blonde hair, wearing a blue and white patterned shirt, carries a floral-patterned backpack with brown leather straps.

[Figure 18]

###### Figure 19. A modern white plastic chair with gray metal legs and cut-out sides.

[Figure 19]

###### Figure 20. Extract the object described by the text, preserving fine details and transparency: the human arm on the left edge of the image

[Figure 20]

###### Figure 21. Extract the object described by the text, preserving fine details and transparency: a girl with medium-length brown hair in yellow located on the center right of the photo

[Figure 21]

###### Figure 22. Extract the object described by the text, preserving fine details and transparency: the white phone that is on the middle of the image

[Figure 22]

###### Figure 23. Extract the object described by the text, preserving fine details and transparency: a silver metal water bottle that is on the left bottom of the photo

[Figure 23]

###### Figure 24. Extract the object described by the text, preserving fine details and transparency: the leather belt on the middle of the image

[Figure 24]

###### Figure 25. Extract the object described by the text, preserving fine details and transparency: the silver iphone is in the middle of the image

[Figure 25]

###### Figure 26. Extract the object described by the text, preserving fine details and transparency: A black man in a black shirt and blue pants is on the right side of the photo

[Figure 26]

###### Figure 27. Extract the object described by the text, preserving fine details and transparency: the male human-being on the right side of the image

[Figure 27]

###### Figure 28. Extract the object described by the text, preserving fine details and transparency: the girl is sitting at the rightside of the photo

[Figure 28]

###### Figure 29. Extract the object described by the text, preserving fine details and transparency: the male baby in white t-shirt locates at the left of the picture

[Figure 29]

###### Figure 30. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 30]

###### Figure 31. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 31]

###### Figure 32. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 32]

###### Figure 33. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 33]

###### Figure 34. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 34]

###### Figure 35. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 35]

###### Figure 36. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 36]

###### Figure 37. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 37]

###### Figure 38. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 38]

###### Figure 39. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 39]

###### Figure 40. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 40]

###### Figure 41. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 41]

###### Figure 42. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 42]

###### Figure 43. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 43]

###### Figure 44. Use the first image as the source scene and the second image as the object mask. Remove the masked object and all of its associated effects, including shadows, reflections, highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. Reconstruct the clean base background as if the object had never been present.

[Figure 44]

###### Figure 45. Automatically matte this image and extract the foreground with a physically accurate alpha channel that preserves true transparency and fine details.

[Figure 45]

###### Figure 46. Automatically matte this image and extract the foreground with a physically accurate alpha channel that preserves true transparency and fine details.

[Figure 46]

###### Figure 47. Automatically matte this image and extract the foreground with a physically accurate alpha channel that preserves true transparency and fine details.

[Figure 47]

###### Figure 48. Automatically matte this image and extract the foreground with a physically accurate alpha channel that preserves true transparency and fine details.

[Figure 48]

###### Figure 49. Automatically matte this image and extract the foreground with a physically accurate alpha channel that preserves true transparency and fine details.

[Figure 49]

###### Figure 50. Automatically matte this image and extract the foreground with a physically accurate alpha channel that preserves true transparency and fine details.

[Figure 50]

###### Figure 51. Automatically matte this image and extract the foreground with a physically accurate alpha channel that preserves true transparency and fine details.

[Figure 51]

###### Figure 52. Automatically matte this image and extract the foreground with a physically accurate alpha channel that preserves true transparency and fine details.

[Figure 52]

###### Figure 53. Automatically matte this image and extract the foreground with a physically accurate alpha channel that preserves true transparency and fine details.

[Figure 53]

###### Figure 54. Automatically matte this image and extract the foreground with a physically accurate alpha channel that preserves true transparency and fine details.

###### Figure 55. Qualitative results of image decomposition. From left to right: reconstructed input image, predicted base layer, and predicted object layer visualized on a checkerboard background.

[Figure 55]

###### Figure 56. Qualitative results of image decomposition. From left to right: reconstructed input image, predicted base layer, and predicted object layer visualized on a checkerboard background.

[Figure 56]

###### Figure 57. Qualitative results of image decomposition. From left to right: reconstructed input image, predicted base layer, and predicted

###### Figure 58. Qualitative results of image decomposition. From left to right: reconstructed input image, predicted base layer, and predicted object layer visualized on a checkerboard background.

[Figure 58]

###### Figure 59. Qualitative results of image decomposition. From left to right: reconstructed input image, predicted base layer, and predicted object layer visualized on a checkerboard background.

[Figure 59]

###### Figure 60. Qualitative results of image decomposition. From left to right: reconstructed input image, predicted base layer, and predicted

###### Figure 61. Qualitative results of image decomposition. From left to right: reconstructed input image, predicted base layer, and predicted object layer visualized on a checkerboard background.

[Figure 61]

###### Figure 62. Qualitative results of image decomposition. From left to right: reconstructed input image, predicted base layer, and predicted object layer visualized on a checkerboard background.

[Figure 62]

###### Figure 63. Qualitative results of image decomposition. From left to right: reconstructed input image, predicted base layer, and predicted

[Figure 63]

###### Figure 64. Qualitative results of image decomposition. From left to right: reconstructed input image, predicted base layer, and predicted object layer visualized on a checkerboard background.

