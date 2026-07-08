## PromptRL: Prompt Matters in RL for Flow-Based Image Generation

Fu-Yun Wang12* Han Zhang3 Micha¨el Gharbi2 Hongsheng Li1 Taesung Park2

# arXiv:2602.01382v1[cs.CV]1Feb2026

### Abstract

Flow matching models (FMs) have revolutionized text-to-image (T2I) generation, with reinforcement learning (RL) serving as a critical posttraining strategy for alignment with reward objectives. In this research, we show that current RL pipelines for FMs suffer from two underappreciated yet important limitations: sample inefficiency due to insufficient generation diversity, and pronounced prompt overfitting, where models memorize specific training formulations and exhibit dramatic performance collapse when evaluated on semantically equivalent but stylistically varied prompts. We present PromptRL (Prompt Matters in RL for Flow-Based Image Generation), a framework that incorporates language models (LMs) as trainable prompt refinement agents directly within the flow-based RL optimization loop. This design yields two complementary benefits: rapid development of sophisticated prompt rewriting capabilities and, critically, a synergistic training regime that reshapes the optimization dynamics. PromptRL achieves state-of-the-art performance across multiple benchmarks, obtaining scores of 0.97 on GenEval, 0.98 on OCR accuracy, and 24.05 on PickScore. Furthermore, we validate the effectiveness of our RL approach on large-scale image editing models, improving the EditReward of FLUX.1-Kontext from 1.19 to 1.43 with only 0.06 million rollouts, surpassing Gemini 2.5 Flash Image (also known as Nano Banana), which scores 1.37, and achieving comparable performance with ReasonNet (1.44), which relied on fine-grained data annotations along with a complex multi-stage training. Our extensive experiments empirically demonstrate that PromptRL consistently achieves higher performance ceilings while requiring over 2× fewer rollouts compared

*Work done during an internship at Reve. 1The Chinese University of Hong Kong, Hong Kong 2Reve, USA 3Meta Superintelligence Labs, USA. Correspondence to: Han Zhang <hanzhang.ai@gmail.com>, Hongsheng Li <hsli@ee.cuhk.edu.hk>.

Preprint. February 3, 2026.

to naive flow-only RL. Our code is available at https://github.com/G-U-N/UniRL.

### 1. Introduction

The advent of flow matching models (FMs) (Liu et al., 2022; Lipman et al., 2022; Labs, 2024; Song et al., 2020) has transformed text-to-image (T2I) generation, enabling photorealistic synthesis from natural language descriptions. To align these models with human preferences and specific reward objectives, reinforcement learning (RL) (Sutton, 2018; Fan et al., 2023; Black et al., 2023) has become the standard posttraining mechanism, refining model behavior beyond the scope of supervised pretraining. Despite these advances, applying RL to FMs remains prohibitively sample-inefficient.

Our investigation reveals two underappreciated yet critical failure modes in current flow-based RL pipelines. (i) First, we observe a counterintuitive exploration paradox: as T2I models improve at following prompts precisely, they simultaneously lose generative diversity under identical prompts. This increased prompt adherence constrains the behavioral variation necessary for effective RL exploration, causing optimization to stagnate in narrow modes of the generation space. (ii) Second, we identify severe prompt overfitting, where models learn to exploit superficial linguistic patterns in training prompts rather than developing genuine visual understanding. This overfitting manifests as dramatic performance collapse when models encounter semantically equivalent prompts phrased with different syntax at test time. This makes prompt enhancement (PE) (Hao et al., 2023; Rosenman et al., 2024; Ma˜nas et al., 2024; Mo et al., 2024), a crucial technique for improving generation quality, ineffective or even counterproductive for RL-finetuned FMs. We provide detailed empirical evidence for both phenomena in Section 3.

These limitations expose a fundamental design oversight in existing approaches: treating prompts as fixed inputs rather than malleable components of the optimization process. Naive augmentation techniques such as random synonym substitution or rule-based paraphrasing prove inadequate, often failing to generate semantically coherent variations at scale. In this paper, we explore the hypothesis that large language models (LMs), when trained as adaptive co-learners

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Editing (Optional)

Prompt 1

Image 1

Reward 1

Reference Image

[Figure 5]

[Figure 6]

[Figure 7]

GenEval

Copy

(Optional)

Editing

Prompt 𝑚

Image 𝑚

Reward 𝑚

Original Prompt

###### 𝑛 × Noise

HPS

- Prompt m + 1

- Prompt m + 2

Image 𝑚 + 1

Reward 𝑚 + 1

Flow Matching Models

Language Model

OCR

Image m + 2

Reward m + 2

[Figure 8]

[Figure 9]

[Figure 10]

Refine

Edit Reward

Prompt 𝑛

Image 𝑛

Reward 𝑛

Prompt Variants

Reward Functions

RL loop update

- Figure 1. Overview of the PromptRL framework. PromptRL jointly trains a language model and a flow-matching image generator within a unified RL loop. Given an original prompt (and optionally a reference image), the LM produces semantically grounded prompt variants that expand the exploration space beyond fixed-prompt training. These prompts are paired with independent noise samples and passed to the flow-matching model to generate diverse images. A mixture of reward functions evaluates each image and guides the evolution of the LM (for improved prompt rewriting) and the FM (for improved visual generation).

### 2. Related works

via joint RL, can generate semantically grounded prompt variations that enhance exploration efficiency and serve as a co-trained PE module in practical deployment.

RL for image generation. Reinforcement learning for flow-based image generation has evolved through several paradigms. Early differentiable reward methods (e.g., DRaFT (Clark et al., 2023), AlignProp (Prabhudesai et al.,

We introduce PromptRL (Prompt Matters in RL), a framework that integrates LMs as adaptive co-learners within flow-based RL training loops, as illustrated in Fig. 1. Rather than employing LMs as static preprocessors, we train them to generate prompt variations that simultaneously preserve semantic intent and maximize downstream image generation rewards. This creates a mutually beneficial training dynamic: diverse LM-generated prompts expand the exploration space for FMs, accelerating policy improvement, while reward signals from flow model outputs guide LMs toward discovering linguistically varied yet contextually appropriate reformulations.

- 2023), ReFL (Xu et al., 2023)) backpropagate gradients from pre-trained reward models like ImageReward, offering simplicity but prone to reward hacking such as oversaturation. RL-based approaches (e.g., DDPO (Black et al., 2023), DPOK (Fan et al., 2023)) treat denoising as an MDP and apply PPO or variants for alignment, with scaled versions showing promise (Zhang et al., 2024). Direct preference optimization advanced with Diffusion-DPO (Wallace et al.,
- 2024) for simulation-free training on preference pairs, later extended by D3PO (Yang et al., 2024) and SPO (Liang et al.,

- 2024) to combine RL and direct objectives without absolute scores. DiffusionNPO (Wang et al., 2025a), which uses negative/reversed preferences to avoid undesirable modes and boost details, lighting, and structure. Flow-GRPO (Liu et al.,
- 2025a), DanceGRPO (Xue et al., 2025)) extends DDPMbased RL approaches (Black et al., 2023) to flow-matching models by converting flow ODEs to SDEs. Complementary works include DiffusionNFT (Zheng et al., 2025), which performs efficient online RL on the forward noising process via flow matching—contrasting positive/negative samples without reverse gradients, trajectory storage, or solver constraints. MixGRPO (Li et al., 2025) further improves GRPO efficiency in flow settings through hybrid ODE–SDE sampling and sliding-window scheduling.

Our experimental results demonstrate that PromptRL achieves state-of-the-art performance across multiple benchmarks, obtaining scores of 0.97 on GenEval (Ghosh et al.,

- 2023), 0.98 on OCR accuracy (Cui et al., 2025), and 24.05 on PickScore (Kirstain et al., 2023). Furthermore, we validate the effectiveness of our RL approach on large-scale image editing models, improving EditReward (Wu et al., 2025b) of FLUX.1-Kontext (Labs et al., 2025) from 1.19 to 1.43 with only 0.06 million rollouts. Extensive experiments show that PromptRL consistently achieves higher performance ceilings while requiring up to 2× fewer rollouts compared to existing methods, all while maintaining robust generalization to diverse prompt formulations. These findings establish language-vision co-optimization as a foundational principle for efficient and robust preference learning in generative models.

PE for image generation Prompt enhancement (PE) has become essential for improving text-to-image (T2I) generation quality and alignment. While early approaches relied on manual refinement, recent methods leverage LMs for automated prompt optimization. Promptist (Hao et al.,

(c) FLUX.1-dev + LM refinement

Prompt (a) Stable Diffusion v1-5 (b) FLUX.1-dev

[Figure 11]

Mini bee with nebula wings harvesting honey inside a floating dewdrop kingdom.

- TI-Sim 0.28↑, P.S. 19.4↑, II-Sim 0.72↓ TI-Sim 0.32↑, P.S. 21.9↑ II-Sim 0.93↓ TI-Sim 0.34↑, P.S. 22.3↑, II-Sim 0.85↓

[Figure 12]

[Figure 13]

A tiny koala in liquid mercury spacesuit riding a singing peppermint comet.

[Figure 14]

- TI-Sim 0.29↑, P.S. 18.9↑, II-Sim 0.58↓ TI-Sim 0.35↑, P.S. 23.0↑ II-Sim 0.92↓ TI-Sim 0.36↑, P.S. 22.9↑, II-Sim 0.84↓

[Figure 15]

[Figure 16]

- Figure 2. The quality-diversity dilemma in flow-based T2I models and its mitigation through prompt refinement. As models advance from Stable Diffusion v1-5 (a) to FLUX.1-dev (b), they achieve higher text-image alignment (TI-Sim) and aesthetic quality (P.S.) but suffer from dramatically reduced output diversity (II-Sim), creating an exploration bottleneck for RL optimization. LM-based prompt refinement (c) partially restores diversity while maintaining quality, demonstrating that linguistic variations can expand the exploration space. All images in each row share identical random seeds to isolate the effect of prompt conditioning.

- 2023) combines supervised fine-tuning with RL to optimize prompts for aesthetic appeal while preserving user intent. NeuroPrompts (Rosenman et al., 2024) introduces constrained text decoding for automatic prompt enhancement with user-controllable styles. OPT2I (Ma˜nas et al., 2024) iteratively refines prompts using LMs to maximize consistency scores. RePrompt (Wu et al., 2025c) incorporates chain-of-thought reasoning and reward-guided training for structured reprompting. PAE (Mo et al., 2024) dynamically optimizes word weights and injection timesteps via online RL with multi-objective rewards. However, these methods are limited to prompt-level RL with modest gains—for instance, RePrompt only improves FLUX.1-dev’s GenEval score from 0.66 to 0.76—and lack exploration of synergistic prompt-level and flow-level RL. Beyond generation, recent works adopt LMs for instruction-based image editing. MGIE (Fu et al., 2023) transforms brief user instructions into expressive editing guidance. SmartEdit (Huang et al.,
- 2024) introduces a Bidirectional Interaction Module for complex reasoning scenarios. Emu Edit (Sheynin et al.,

- 2024) incorporates the LM to output detailed editing instruction and target image caption.

### 3. Understanding flow RL inefficiencies

#### 3.1. Generation dilemma: quality vs. diversity

We observe a fundamental tension between generation quality and output diversity in modern T2I FMs. As models advance in their capacity to precisely follow textual prompts, they simultaneously sacrifice the generative variability essential for effective RL exploration. To quantify this phenomenon, we measure text-image similarity (TI-Sim) using CLIP ViT-g-14 (Radford et al., 2021) for prompt alignment, PickScore (P.S.) for aesthetic quality, and image-image similarity (II-Sim) via CLIP ViT-g-14 for output diversity under identical prompts. Fig. 2 demonstrates this quality-diversity trade-off: Stable Diffusion v1-5 (column a) produces generations with moderate TI-Sim and P.S., but maintains substantial diversity (II-Sim of 0.58–0.72). In contrast, FLUX.1-dev (column b) achieves notably higher aesthetic scores, yet generates outputs with considerably reduced diversity (II-Sim of 0.92–0.93). This pattern holds consistently across semantically diverse prompts, from fantastical creatures to surreal compositional scenes. Notably, LM-based prompt refinement (column c) partially mitigates this collapse, reducing II-Sim to 0.84–0.85 while preserving quality metrics.

This quality-diversity dilemma directly undermines RL op-

Table 1. Performance comparison across different models and evaluation metrics. All metrics are tested with 20 Euler steps with resolution 1024. PE denotes prompt enhancement with Qwen-2.5VL. Green and red numbers indicate performance changes after applying PE. P.S. and U.R. denotes PickScore and UnifiedReward, respectively.

Model w/ PE GenEval OCR P.S. HPS U.R. SD3 ✗ 0.58 0.48 22.30 26.95 2.982 SD3 ✓ 0.63 (+0.05) 0.53 (+0.05) 22.34 (+0.04) 27.67 (+0.72) 3.140 (+0.158) DiffusionNFT ✗ 0.88 0.89 23.63 31.78 3.392 DiffusionNFT ✓ 0.77 (-0.11) 0.86 (-0.03) 23.21 (-0.42) 30.66 (-1.12) 3.268 (-0.124)

FlowGRPO ✗ 0.92 0.89 23.33 — FlowGRPO ✓ 0.81 (-0.11) 0.86 (-0.03) 23.13 (-0.20) — —

timization in flow-based models. As generation policies become increasingly deterministic, rollout trajectories collapse into narrow modes of the output space, causing reward signals to degenerate: when all samples cluster around similar high-quality outputs, advantage estimators lose the comparative information necessary for policy improvement. This exploration bottleneck is further compounded by severe prompt overfitting, wherein models exploit superficial lexical patterns rather than semantic understanding, which we investigate in the following subsection.

#### 3.2. Prompt linguistic hacking

Beyond the quality-diversity dilemma, we identify a second critical failure mode: prompt linguistic hacking, where RL-trained models exploit superficial lexical patterns rather than developing robust semantic understanding. We evaluate this by testing models on both original prompts and semantically-preserved paraphrases generated by Qwen-2.5VL (i.e., PE) (Bai et al., 2025).

As shown in Table 1, the pretrained SD3 (Esser et al., 2024) demonstrates linguistic robustness, with consistent or improved performance under paraphrasing across all metrics. However, flow-only RL models exhibit severe prompt overfitting. DiffusionNFT achieves strong performance on original prompts but suffers catastrophic degradation under paraphrasing. Similarly, FlowGRPO trained on GenEval drops from 0.92 to 0.81 when prompts are paraphrased. This indicates that learned policies memorize superficial linguistic features rather than understanding underlying visual concepts. This overfitting occurs at the prompt-conditioning level and cannot be resolved through standard regularization. More critically, PE techniques that benefit pretrained FMs become ineffective or even harmful after flow-only RL, as fine-tuned models overfit to specific prompt distributions. This observation motivates our joint LM-FM optimization approach: rather than applying PE as a fixed preprocessing step, we co-evolve the prompt enhancer and generator in a symbiotic manner.

### 4. PromptRL

#### 4.1. Incorporating LMs as dynamic prompt refiner

To address the dual challenges of exploration collapse and prompt overfitting identified in Section 3, we propose incorporating LMs as adaptive prompt refiners within the RL training loop. Unlike static augmentation techniques that rely on rule-based transformations or synonym substitution, our approach leverages the semantic understanding and compositional flexibility of pretrained LMs to generate contextually grounded prompt variations.

Formally, given an original prompt p0 from the training distribution, we deploy an LM πLM(·|p0) to generate a set of refined prompts {p1,p2,...,pk} that preserve the core semantic intent while introducing linguistic diversity. Each refined prompt pi is then paired with an independent noise sample ϵi ∼ N(0,I) and fed into the flowmatching model πFM(·|pi,ϵi) to produce diverse image samples {x1,x2,...,xk}. This architecture creates a hierarchical exploration mechanism: the LM explores the linguistic manifold of semantically equivalent descriptions, while the FM explores the visual manifold conditioned on each prompt variant.

Critically, we introduce a prompt retention mechanism during training: for each batch of n total samples, we retain m < n samples that use the original prompt p0 without LM refinement, while the remaining n−m samples undergo LMbased augmentation. This design serves two complementary purposes. First, unmodified prompts provide a strong baseline for advantage estimation—augmented prompts that yield lower rewards than p0 are effectively pruned during policy optimization, preventing wasteful exploration in lowreward regions of the prompt space. Second, consistent exposure to original prompts ensures that the FM maintains robust performance on the training distribution, preventing the model from becoming overly dependent on LM refinements at inference time.

#### 4.2. Joint RL training on disjoint LMs and FMs

Having established the LM as a dynamic prompt refiner, we now describe the joint RL training procedure that simultaneously optimizes both πLM and πFM within a unified policy gradient framework. Crucially, while the two models share reward signals, they remain architecturally disjoint—gradients do not propagate between the LM and FM, preserving modularity and computational efficiency.

At each training iteration, we sample a batch of B original prompts {p(1)0 ,...,p(0B)} from the training distribution. For each prompt p(0j), we generate n samples using the procedure described in Section 4.1: m samples are generated directly from p(0j) with different noise seeds, while the re-

###### Reference Image

###### Prompt  ano Banana FLUX.1-Kontext

Improved Prompt

PromptRL

|[Figure 17]|
|---|

[Figure 18]

[Figure 19]

[Figure 20]

Enhance the image by changing the color of the ice skates to purple. Ensure that the skates are the only objects in the image that are altered, and that the rest of the scene remains unchanged. The reflection of the skates in the water should also be preserved. The final image should maintain the original intent and aesthetic of the original photo.

turn the color of a set of ice skates to purple

|[Figure 21]|
|---|

[Figure 22]

[Figure 23]

[Figure 24]

Remove the small village with several houses with red roofs in the background. Ensure the remaining elements, such as the cows, the bridge, and the landscape, are still present and maintain the original scene's atmosphere and style.

Remove a small village with several houses with red roofs in the background.

- Figure 3. Qualitative comparison on instructional image editing tasks. Our method enables the LM to leverage the original image’s visual signals to transform vague editing instructions into more explicit and image-specific prompts, ultimately improving editing performance.

maining n−m samples use LM-refined prompts {p(ij)}ni=1−m

sampled from πLM(·|p(0j)). Each prompt (original or refined) is paired with an independent latent noise vector ϵ ∼ N(0,I) to produce images via the FM.

The resulting images are evaluated using a composite reward function R(·) that aggregates format reward and image generation reward:

R(x,p) = λFormatRFormat(p) + λGenRGen(x,p), (1) where RFormat(p) is a binary reward that requires the LM to output refined prompt variants enclosed within XML tags <answer> and </answer>. This format constraint ensures structured output parsing and penalizes malformed generations, with RFormat(p) = 1 if the output conforms to the required format and 0 otherwise. The term RGen(x,p) denotes a image generation reward models (e.g., GenEval for compositional accuracy). The hyperparameters λFormat and λGen balance format compliance and generation quality; in practice, we set λFormat = 1.0 and λGen = 1.0 for simplicity. Following the GRPO, we compute advantages through group-wise normalization. For each prompt p(0j) and its n associated samples {x(1j),...,x(nj)}, we calculate:

R(x(ij),p(ij)) − µj σj + ϵ

A(x(ij),p(ij)) =

, (2)

where µj and σj are the mean and standard deviation of rewards within the j-th group, and ϵ is a small constant for numerical stability. This group-wise normalization makes advantage signals invariant to reward scale across different prompts, enabling stable optimization across diverse semantic categories. It also inherently implements a form of self-competition where samples generated from the same prompt compete against each other, encouraging withingroup diversity.

Crucially, while the LM and FM share reward signals in PromptRL, they remain architecturally disjoint—gradients do not propagate between the LM and FM, preserving modularity and computational efficiency. For the language model, we only retain advantages corresponding to the n − m LMrefined samples, excluding those generated from original prompts. The LM policy gradient is:

n

∇θLMJLM = Ep

A(xi,pi) · ∇θLM

log πLM(pi|p0) .

0

i=m+1

(3) This selective advantage assignment ensures the LM learns to generate prompt variants that improve upon the baseline. Variants that underperform receive negative advantages and are down-weighted, preventing the LM from introducing counterproductive linguistic changes.

For the flow matching model, we utilize advantages from all n samples (both original and refined prompts) to update the image generation policy:

n

∇θFMJFM = Ep

A(xi,pi) · ∇θFM

log πFM(xi|pi,ϵi) .

0

i=1

(4) The inclusion of advantages from original prompts ensures the FM maintains strong performance on the base distribution while benefiting from the expanded exploration space provided by LM refinements.

Notably, this joint optimization requires no architectural/algorithmic modifications to either model—we simply backpropagate separate policy gradients through their respective parameters using shared rewards as the update signal. Our framework is agnostic to the specific RL algorithm; while we use GRPO (Shao et al., 2024) as the baseline implementation for its simplicity, the components

A high-fashion runway with a sleek, modern backdrop displaying "Spring Collection 2024". Models walk confidently on the catwalk, showcasing vibrant, floral prints and pastel tones, under soft, ambient lighting that enhances the fresh, spring vibe.

An ancient Egyptian painting depicting an argument over whose turn it is to take out the trash.

A 1960s yearbook photo with animals dressed as humans.

Prompt ahorsephotoofanelephantbelowa aphotooftwocars

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

FlowGRPO

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Diffusion FT

A high-fashion runway with a sleek, modern backdrop displaying "Spring Collection 2024". Models walk confidently on the catwalk, showcasing vibrant, floral prints and pastel tones, under soft, ambient lighting that enhances the fresh, spring vibe. The text "Spring Collection 2024" is clear and legible, ensuring that viewers can easily read the event's name...

Generate a 1960s yearbook photo with animals dressed as humans. The photo should have a vintage feel, with a focus on the animals' human-like attire and the overall composition of the yearbook layout. The animals should be depicted in a realistic and detailed manner, with attention to their expressions and the surrounding environment...

Generate an ancient Egyptian painting depicting a detailed argument over whose turn it is to take out the trash. The artwork should feature realistic and lifelike characters, with intricate details in the Egyptian art style, including hieroglyphs, traditional clothing, and the environment...

an image of two distinct cars positioned side by side on a clear, sunny day... The background features a wellmaintained road with a gentle curve, and the sky is a soft, pastel blue with a few wispy clouds.

an image featuring a majestic elephant positioned below and a graceful horse directly above, with precise alignment of both animals and their surroundings.

Improved Prompt

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

###### PromptRL

- Figure 4. Qualitative comparison on text-to-image generation. The first two prompts are from GenEval. The third prompt is from OCR-1k. And the last two prompts are from Drawbench.

of PromptRL should be able to integrated with other online RL approaches for LMs (e.g., ReMax (Li et al., 2023)) and FMs (e.g., DiffusionNFT (Zheng et al., 2025)).

#### 4.3. Multi-reward training via reward tagging

Beyond single-reward scenarios, we validate PromptRL under multi-reward optimization with GenEval, PickScore, and OCR. A known challenge is that different reward models exhibit vastly different scales and variances, requiring cumbersome manual tuning of reward coefficients. Alternative approaches adopt multi-stage training pipelines (e.g., DiffusionNFT first trains on PickScore before transitioning to additional rewards), introducing scheduling complexity. We propose a simple yet effective solution: single-rewardper-sample training. Rather than computing weighted reward sums, we assign each training prompt a categorical

tag indicating which reward function evaluates its generated images. During group-wise advantage computation, normalization is performed within each reward category, ensuring comparable advantage signals despite differing scales. This eliminates reward coefficient tuning entirely—each reward model operates in its native scale without interference. Empirically, this simple tagging mechanism achieves strong performance across all objectives simultaneously without explicit reward engineering or multi-stage curricula.

### 5. Experiments

#### 5.1. Experimental setup

Base models. To comprehensively validate PromptRL on flow-based image generation, we evaluate on both T2I synthesis and instructional image editing. We adopt FLUX.1-

dev as the base flow-matching model for T2I generation and FLUX.1-Kontext for image editing. For the language model component, we use Qwen2.5-VL-3B-Instruct as the prompt refiner.

Datasets and benchmarks. For text-to-image generation, following FlowGRPO, we evaluate across three complementary dimensions: compositional accuracy (GenEval), text rendering capability (OCR), and human preference alignment (PickScore). For GenEval and OCR objectives, we use the corresponding training sets from FlowGRPO. For PickScore optimization, we train on the Pick-a-Pic dataset. For instructional image editing, we randomly sample 10,000 examples from the OmniEdit (Wei et al., 2024) training set, retaining only the editing instructions and corresponding reference images. For T2I generation, we use the official GenEval validation set for compositional evaluation, DrawBench (Saharia et al., 2022) for aesthetic quality assessment (PickScore, HPS (Wu et al., 2023), UnifiedReward (Wang et al., 2025b)), and three OCR benchmarks: the FlowGRPO OCR-1k validation set, TMDB, and OpenLib from MARIOEval (Chen et al., 2023). For image editing, we directly evaluate on the OmniEdit validation set, which covers six editing categories: Object Swap, Object Addition, Object Removal, Attribute Modification, Environment Change, and Style Transfer.

Training details. For computational efficiency, we conduct rollouts at 512×512 resolution with 20 inference steps when training FLUX.1-dev. However, we observe that reducing resolution significantly degrades the editing capabilities of FLUX.1-Kontext; therefore, we maintain 1024×1024 resolution for editing experiments. To improve training efficiency under the high computational demands of highresolution training, we reduce the number of inference steps to 8 and apply SDE solving only during the first 4 denoising steps of each rollout, following prior work (Li et al., 2025) demonstrating that image structure largely stabilizes in the early timesteps. All models are trained using the GRPO algorithm with a group size of n = 8 samples per prompt and a prompt retention number of m = 2.

#### 5.2. Quantitative comparison

Text-to-image generation. As shown in Table 2, PromptRL achieves state-of-the-art performance on GenEval with an overall score of 0.97, outperforming FlowGRPO at 0.92 and DiffusionNFT at 0.88. PromptRL w/ PE attains near-perfect scores of 0.99 on both Position and Counting, demonstrating exceptional compositional accuracy. Notably, even without prompt enhancement at inference, PromptRL w/o PE achieves 0.94, confirming that joint training instills robust visual understanding independent of LM refinements. Table 3 further validates PromptRL across aesthetic and OCR metrics. Our method achieves a PickScore of 24.04, HPS

- Table 2. Performance comparison on GenEval benchmark across different models. Higher scores indicate better performance. Best results are shown in bold. Metrics of models with * are obtained from the Qwen-Image paper.

Model 1 Obj. 2 Obj. Cnt. Clr. Pos. Attr. Avg.↑

Show-o* (Xie et al., 2024b) 0.95 0.52 0.49 0.82 0.11 0.28 0.53 Emu3-Gen* (Wang et al., 2024) 0.98 0.71 0.34 0.81 0.17 0.21 0.54 SD3 Medium* (Esser et al., 2024) 0.98 0.74 0.63 0.67 0.34 0.36 0.62 FLUX.1-dev* (Labs, 2024) 0.98 0.81 0.74 0.79 0.22 0.45 0.66 SD3.5 Large* 0.98 0.89 0.73 0.83 0.34 0.47 0.71 JanusFlow* (Ma et al., 2025) 0.97 0.59 0.45 0.83 0.53 0.42 0.63 Janus-Pro-7B* (Chen et al., 2025) 0.99 0.89 0.59 0.90 0.79 0.66 0.80 HiDream (Cai et al., 2025b) 1.00 0.98 0.79 0.91 0.60 0.72 0.83 Seedream 3.0* (Gao et al., 2025) 0.99 0.96 0.91 0.93 0.47 0.80 0.84 Qwen-Image* (Wu et al., 2025a) 0.99 0.92 0.89 0.88 0.76 0.77 0.87

RL-based

RePrompt (Wu et al., 2025c) 0.98 0.87 0.77 0.85 0.62 0.49 0.76 FlowGRPO (Liu et al., 2025a) 1.00 0.99 0.91 0.89 0.95 0.80 0.92 DiffusionNFT (Zheng et al., 2025) 1.00 0.98 0.74 0.92 0.85 0.80 0.88 PromptRL w/o PE 1.00 0.96 0.95 0.95 0.93 0.85 0.94 PromptRL w/ PE 1.00 0.99 0.99 0.96 0.99 0.90 0.97

- Table 3. Performance comparison across different models on aesthetic and OCR metrics.

Aesthetic OCR

###### Model P.S. HPS U.R. OCR-1k TMDB EOpenLib

SD1.5 (Rombach et al., 2022) 20.92 23.71 2.00 0.05 0.13 0.08 SDXL (Podell et al., 2023) 22.14 26.67 2.78 0.13 0.20 0.09 SD3 Medium (Esser et al., 2024) 22.38 28.56 3.09 — 0.44 0.33

- FLUX.1-schnell (Labs, 2024) 22.64 29.39 3.25 0.54 0.66 0.50
- FLUX.2-klein (Labs, 2025) 22.79 29.03 3.29 0.55 0.22 0.46 Z-Image (Cai et al., 2025a) 20.14 28.22 3.51 0.70 0.71 0.83 Qwen-Image (Wu et al., 2025a) 23.05 30.40 3.53 0.65 0.79 0.94 Qwen-Image-2512 23.16 30.79 3.40 0.72 0.81 0.87 RL-based

FlowGRPO 23.33 29.80 3.33 0.89 0.83 0.73 DiffusionNFT 23.63 31.79 3.39 0.89 0.91 0.86 PromptRL w/o PE 24.01 31.79 3.38 0.97 0.92 0.95 PromptRL w/ PE 24.05 32.03 3.44 0.98 0.91 0.95

of 32.03, and OCR accuracy of 0.98 on OCR-1k, consistently surpassing prior RL-based approaches. These improvements across diverse metrics demonstrate that joint LM-FM optimization generalizes beyond single-objective reward optimization.

Instructional image editing. For instructional image editing, as reported in Table 5, PromptRL w/ PE improves upon the FLUX.1-Kontext baseline from 1.19 to 1.43, approaching ReasonEdit-Think at 1.44, which relied on fine-grained and dense data annotations along with a complex multistage training pipeline. Improvements are most pronounced on Removal (+0.69) and Environment (+0.28). Notably, naively applying prompt enhancement without joint training degrades FLUX.1-Kontext performance from 1.19 to 1.01, demonstrating that PromptRL’s joint optimization is essential for effective prompt refinement.

Multi-reward training. We evaluate our tag-based multireward training strategy in Table 4. Despite using no reward coefficient tuning or multi-stage curriculum, the multireward model achieves competitive performance across all objectives (GenEval: 0.93, OCR: 0.96, PickScore: 23.94), with only modest degradation compared to single-reward specialists.

- Table 4. Comparison of single-reward and multi-reward training. Single-reward trains separately on each objective, while multireward uses tag-based joint optimization.

Training GenEval↑ OCR↑ PickScore↑

Single-reward 0.97 0.98 24.05 Multi-reward 0.93 0.96 23.94

- Table 5. Performance comparison on image editing tasks measured by EditReward. We evaluate models across six editing categories: Swap, Style, Addition (Add.), Attribute Modification (attr.), Environment (Env.), and Removal. The editing instructions of FLUX.1Kontext w/ PE are refined by pretrained Qwen2.5-VL.

Model Swap Style Add. Attr. Env. Removal Avg.↑

InstructPix2Pix (Brooks et al., 2023) -0.24 0.91 -0.45 0.45 0.48 -0.80 0.02 MagicBrush (Zhang et al., 2023) -0.38 0.36 -0.78 -0.80 0.91 -0.85 -0.27 LEDITS++ (Brack et al., 2024) -0.81 -0.32 -0.30 -0.60 -0.37 -0.97 -0.60 Qwen-Image-Edit 1.11 1.14 0.95 0.90 1.39 0.61 1.03 FLUX.2-klein 1.42 1.73 1.29 1.42 1.80 0.32 1.34 Nano Banana 1.58 1.20 1.28 1.18 1.61 1.13 1.37 Step1X-Edit (Liu et al., 2025b) 1.39 1.58 1.19 1.34 1.57 0.22 1.24 ReasonEdit (Yin et al., 2025) 1.51 1.43 1.19 1.47 1.58 1.14 1.40 ReasonEdit-Think 1.52 1.47 1.19 1.44 1.69 1.27 1.44 FLUX.1-Kontext 1.35 1.36 1.16 1.15 1.44 0.55 1.19 FLUX.1-Kontext w/ PE 1.35 0.97 1.04 0.48 1.22 0.65 1.01

PromptRL w/o PE 1.45 1.46 1.28 1.35 1.56 0.98 1.36 PromptRL w/ PE 1.47 1.43 1.29 1.39 1.72 1.24 1.43

1.0

0.9

###### GenEvalReward

0.8

0.7

0.6

0.5

PromptRL

0.4

Flow-Only RL

0.3

0 100 200 300 400 500

Number of Rollouts (K)

(a) Training curve comparison on GenEval reward.

1.0

0.9

###### OCRReward

0.8

0.7

0.6

0.5

PromptRL

0.4

Flow-Only RL

0.3

0 10 20 30 40 50 60 70 80

Number of Rollouts (K)

(b) Training curve comparison on OCR reward. Figure 5. Training curve comparison on different rewards.

#### 5.3. Qualitative comparison

Figs. 3 and 4 presents qualitative comparisons across T2I generation and image editing tasks. For compositional prompts, PromptRL correctly renders objects with accurate colors and spatial arrangements where baselines exhibit color leakage or omission. On OCR tasks, PromptRL produces legible, correctly spelled text while maintaining aesthetic quality. For instructional editing, PromptRL’s jointlytrained LM produces image-aware prompt refinements by leveraging visual signals from reference images, transforming vague instructions into precise, image-specific prompts that preserve foreground subjects while modifying only intended regions.

#### 5.4. Ablation study

Joint RL boosts training efficiency. We compare training dynamics as rollout counts increase, with both methods using FlowGRPO for FM optimization and single-updateper-sample for stability. As shown in Fig. 5a and Fig. 5b, PromptRL consistently achieves higher rewards with fewer rollouts across both GenEval and OCR objectives. On GenEval, PromptRL reaches comparable performance to FlowGRPO’s convergence point using approximately 50% fewer rollouts. Similar trends are observed on OCR, where PromptRL demonstrates faster initial improvement and superior final performance. These results validate that the expanded exploration space provided by LM-generated prompt variations improves sample efficiency.

Prompt retention mechanism. We investigate the effect of retaining original prompts during training on model

Table 6. Ablation on prompt retention mechanism. We report GenEval scores without PE at inference with varying numbers of retained original prompts m (group size n = 8).

Retention 1 Obj. 2 Obj. Cnt. Clr. Pos. Attr. Avg.↑

- m = 0 0.99 0.94 0.84 0.89 0.51 0.83 0.83

- m = 1 0.98 0.88 0.83 0.85 0.37 0.64 0.77

- m = 2 1.00 0.96 0.95 0.95 0.93 0.85 0.94 m = 4 1.00 0.96 0.94 0.88 0.93 0.81 0.92

performance without PE at inference. With group size n = 8, we vary the number of retained original prompts m ∈ {0,1,2,4} and evaluate GenEval performance without PE (Table 6). When m = 0, the model never sees original prompts during training, resulting in degraded performance on unmodified test prompts (0.83). m = 1 yields even worse results (0.76). We empirically find that the LM quickly discovers prompt variants more suitable for the FM, causing the single original prompt to consistently receive negative advantages and thus ineffective gradient updates. Setting m = 2 significantly improves performance to 0.94, as multiple original prompts compete within the group, enabling meaningful advantage signals and robust learning on the original prompt distribution.

### 6. Conclusion

We presented PromptRL, a framework that jointly trains language models and flow-matching models within a unified reinforcement learning loop for text-to-image generation. Our approach leverages LM-generated prompt variations to expand the exploration space while simultaneously co-

evolving a prompt enhancement module that improves generation quality at inference time. Extensive experiments demonstrate that PromptRL achieves state-of-the-art performance across multiple benchmarks.

### Impact statement

This paper presents work whose goal is to advance the field of joint RL training on both language models and diffusion models. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Black, K., Janner, M., Du, Y., Kostrikov, I., and Levine, S. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

Brack, M., Friedrich, F., Kornmeier, K., Tsaban, L., Schramowski, P., Kersting, K., and Passos, A. Ledits++: Limitless image editing using text-to-image models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8861–8870, 2024.

Brooks, T., Holynski, A., and Efros, A. A. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18392–18402, 2023.

Cai, H., Cao, S., Du, R., Gao, P., Hoi, S., Hou, Z., Huang, S., Jiang, D., Jin, X., Li, L., et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699,

- 2025a.

Cai, Q., Chen, J., Chen, Y., Li, Y., Long, F., Pan, Y., Qiu, Z., Zhang, Y., Gao, F., Xu, P., et al. Hidream-i1: A highefficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705,

- 2025b.

Chen, J., Huang, Y., Lv, T., Cui, L., Chen, Q., and Wei, F. Textdiffuser: Diffusion models as text painters. Advances in Neural Information Processing Systems, 36: 9353–9387, 2023.

Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., and Ruan, C. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

Clark, K., Vicol, P., Swersky, K., and Fleet, D. J. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023.

Cui, C., Sun, T., Lin, M., Gao, T., Zhang, Y., Liu, J., Wang, X., Zhang, Z., Zhou, C., Liu, H., et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., and Lee, K. Reinforcement learning for fine-tuning text-to-image diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS) 2023. Neural Information Processing Systems Foundation, 2023.

Fu, T.-J., Hu, W., Du, X., Wang, W. Y., Yang, Y., and Gan, Z. Guiding instruction-based image editing via multimodal large language models. arXiv preprint arXiv:2309.17102, 2023.

Gao, Y., Gong, L., Guo, Q., Hou, X., Lai, Z., Li, F., Li, L., Lian, X., Liao, C., Liu, L., et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Hao, Y., Chi, Z., Dong, L., and Wei, F. Optimizing prompts for text-to-image generation. Advances in Neural Information Processing Systems, 36:66923–66939, 2023.

Huang, Y., Xie, L., Wang, X., Yuan, Z., Cun, X., Ge, Y., Zhou, J., Dong, C., Huang, R., Zhang, R., et al. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8362–8371, 2024.

Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., and Levy, O. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:36652–36663, 2023.

Labs, B. F. Flux. https://github.com/ black-forest-labs/flux, 2024.

Labs, B. F. FLUX.2: Frontier Visual Intelligence. https: //bfl.ai/blog/flux-2, 2025.

Labs, B. F., Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Li, J., Cui, Y., Huang, T., Ma, Y., Fan, C., Yang, M., and Zhong, Z. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025.

Li, Z., Xu, T., Zhang, Y., Lin, Z., Yu, Y., Sun, R., and Luo, Z.-Q. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. arXiv preprint arXiv:2310.10505, 2023.

Liang, Z., Yuan, Y., Gu, S., Chen, B., Hang, T., Li, J., and Zheng, L. Step-aware preference optimization: Aligning preference with denoising performance at each step. arXiv preprint arXiv:2406.04314, 2024.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025a.

Liu, S., Han, Y., Xing, P., Yin, F., Wang, R., Cheng, W., Liao, J., Wang, Y., Fu, H., Han, C., et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025b.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Ma, Y., Liu, X., Chen, X., Liu, W., Wu, C., Wu, Z., Pan, Z., Xie, Z., Zhang, H., Yu, X., et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7739–7751, 2025.

Ma˜nas, O., Astolfi, P., Hall, M., Ross, C., Urbanek, J., Williams, A., Agrawal, A., Romero-Soriano, A., and Drozdzal, M. Improving text-to-image consistency via automatic prompt optimization. arXiv preprint arXiv:2403.17804, 2024.

Mo, W., Zhang, T., Bai, Y., Su, B., Wen, J.-R., and Yang, Q. Dynamic prompt optimizing for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26627–26636, 2024.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Prabhudesai, M., Goyal, A., Pathak, D., and Fragkiadaki, K. Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Rosenman, S., Lal, V., and Howard, P. Neuroprompts: An adaptive framework to optimize prompts for text-toimage generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pp. 159–167, 2024.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35: 36479–36494, 2022.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheynin, S., Polyak, A., Singer, U., Kirstain, Y., Zohar, A., Ashual, O., Parikh, D., and Taigman, Y. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8871–8879, 2024.

Song, J., Meng, C., and Ermon, S. Denoising diffusion

implicit models. arXiv preprint arXiv:2010.02502, 2020. Sutton, R. S. Reinforcement learning: An introduction. A

Bradford Book, 2018.

Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., and Naik, N. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024.

Wang, F.-Y., Shui, Y., Piao, J., Sun, K., and Li, H. Diffusionnpo: Negative preference optimization for better preference aligned generation of diffusion models. arXiv preprint arXiv:2505.11245, 2025a.

- Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- Wang, Y., Zang, Y., Li, H., Jin, C., and Wang, J. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025b.

Wei, C., Xiong, Z., Ren, W., Du, X., Zhang, G., and Chen, W. Omniedit: Building image editing generalist models through specialist supervision. In The Thirteenth International Conference on Learning Representations, 2024.

Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.-m., Bai, S., Xu, X., Chen, Y., et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Wu, K., Jiang, S., Ku, M., Nie, P., Liu, M., and Chen, W. Editreward: A human-aligned reward model for instructionguided image editing. arXiv preprint arXiv:2509.26346, 2025b.

Wu, M., Wang, L., Zhao, P., Yang, F., Zhang, J., Liu, J., Zhan, Y., Han, W., Sun, H., Ji, J., et al. Reprompt: Reasoning-augmented reprompting for text-toimage generation via reinforcement learning. arXiv preprint arXiv:2505.17540, 2025c.

Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., and Li, H. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

Xie, E., Chen, J., Chen, J., Cai, H., Tang, H., Lin, Y., Zhang, Z., Li, M., Zhu, L., Lu, Y., et al. Sana: Efficient highresolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024a.

Xie, J., Mao, W., Bai, Z., Zhang, D. J., Wang, W., Lin, K. Q., Gu, Y., Chen, Z., Yang, Z., and Shou, M. Z. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024b.

Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., and Dong, Y. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36: 15903–15935, 2023.

Xue, Z., Wu, J., Gao, Y., Kong, F., Zhu, L., Chen, M., Liu, Z., Liu, W., Guo, Q., Huang, W., et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Yang, K., Tao, J., Lyu, J., Ge, C., Chen, J., Shen, W., Zhu, X., and Li, X. Using human feedback to fine-tune diffusion models without any reward model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8941–8951, 2024.

Yin, F., Liu, S., Han, Y., Wang, Z., Xing, P., Wang, R., Cheng, W., Wang, Y., Li, A., Yin, Z., et al. Reasonedit: Towards reasoning-enhanced image editing models. arXiv preprint arXiv:2511.22625, 2025.

Zhang, K., Mo, L., Chen, W., Sun, H., and Su, Y. Magicbrush: A manually annotated dataset for instructionguided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.

Zhang, Y., Tzeng, E., Du, Y., and Kislyuk, D. Large-scale reinforcement learning for diffusion models. arXiv preprint arXiv:2401.12244, 2024.

Zheng, K., Chen, H., Ye, H., Wang, H., Zhang, Q., Jiang, K., Su, H., Ermon, S., Zhu, J., and Liu, M.-Y. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

Algorithm 1 PromptRL: Joint RL Training of LM and FM Require: Training prompts D, LM πLM, FM πFM, reward function R, group size n, retention number m Ensure: Optimized πLM and πFM

- 1: for each iteration do
- 2: Sample batch of prompts {p(0j)}Bj=1 from D
- 3: for each prompt p(0j) do
- 4: // Generate prompt variants
- 5: P(j) ← {p(0j)}m // Retain m original prompts
- 6: for i = m + 1 to n do
- 7: p(ij) ∼ πLM(· | p(0j)) // LM refinement
- 8: P(j) ← P(j) ∪ {p(ij)}
- 9: end for
- 10: // Generate images and compute rewards
- 11: for each pi ∈ P(j) do
- 12: Sample ϵi ∼ N(0,I)
- 13: xi ← πFM(· | pi,ϵi)
- 14: ri ← R(xi,pi)
- 15: end for
- 16: // Group-wise advantage normalization
- 17: µ(j),σ(j) ← mean({ri}),std({ri})
- 18: for i = 1 to n do
- 19: A(ij) ← (ri − µ(j))/(σ(j) + ϵ)
- 20: end for
- 21: end for
- 22: // Update LM (only on refined prompts)
- 23: θLM ← θLM + αLM j,i>m A(ij)∇log πLM(p(ij) | p(0j))
- 24: // Update FM (on all samples)
- 25: θFM ← θFM + αFM j,i A(ij)∇log πFM(x(ij) | p(ij),ϵi)
- 26: end for

### A. Training details

Pseudo code. Algorithm 1 presents the complete PromptRL training procedure. The algorithm alternates between generating prompt variants via the LM, producing images through the FM, and updating both models using group-normalized advantages. Key design choices include the prompt retention mechanism (lines 4–9) that maintains m original prompts per group, and the selective gradient updates that train the LM only on refined prompts while the FM learns from all samples.

Training configurations. Table 7 summarizes the hyperparameters and computational resources used across all experiments. We adopt consistent settings where possible to ensure fair comparisons, with task-specific adjustments for learning rates and KL coefficients based on preliminary experiments. For image editing, we use higher resolution (1024×1024) and fewer SDE steps to balance quality and efficiency, as discussed in Section 4.

### B. Discussion

Generalization of prompt enhancer to unseen FMs. We investigate whether the prompt enhancement module trained with PromptRL generalizes to flow models not seen during training. Specifically, we evaluate our prompt enhancer on SANA (Xie et al., 2024a) and SD3 using GenEval at 1024 resolution with 20 inference steps (Table 8). Although our prompt enhancer does not match the performance of model-specific prompt enhancers trained via prompt-only RL for each FM, it significantly improves over the original model performance. For SANA, our enhancer improves GenEval from 0.62 to 0.70, and for SD3 from 0.58 to 0.77. These results demonstrate that PromptRL learns generalizable linguistic refinements that benefit diverse flow models, suggesting the LM captures semantic patterns broadly useful for T2I generation rather than overfitting to FLUX.1-dev-specific characteristics.

- Table 7. Training configurations and hyperparameters for PromptRL across different reward objectives. All experiments use GRPO as the base RL algorithm. “—” indicates not applicable or same as default.

Configuration

Text-to-Image Multi-Reward Image Editing GenEval OCR PickScore EditReward

Model Setup

Base FM FLUX.1-dev FLUX.1-dev FLUX.1-dev FLUX.1-dev FLUX.1-Kontext Language Model Qwen2.5-VL-3B-Instruct FM Parameters 12B 12B LM Parameters 3B

Training Details

Training Dataset FlowGRPO-GenEval FlowGRPO-OCR Pick-a-Pic Mixed OmniEdit (10k) Training Samples 50,000 19,653 25,432 95,085 10,000 Training Resolution 512 × 512 512 × 512 512 × 512 512 × 512 1024 × 1024 Training Precision bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 Inference Steps (Rollout) 20 20 20 20 8 SDE Steps 20 20 20 10 4 Group Size (n) 8 8 8 8 8 Prompt Retention (m) 2 2 2 2 2 Batch Size 1 1 1 1 1 K epochs 1 1 1 1 1 Learning Rate (FM) 3×10−7 3×10−7 3×10−7 10−7 2×10−7 Learning Rate (LM) 10−6 10−6 10−6 10−6 4 × 10−7 Optimizer AdamW AdamW AdamW AdamW AdamW KL Coefficient (FM) 4×10−3 4×10−3 2×10−3 2×10−3 10−2 KL Coefficient (LM) 10−2 10−2 10−2 10−2 10−2

Reward Configuration Reward Model GenEval OCR Accuracy PickScore Tag-based EditReward

Format Reward (λFormat) 1.0 1.0 1.0 1.0 1.0 Generation Reward (λGen) 1.0 1.0 1.0 1.0 1.0 Reward Normalization Group-wise Group-wise Group-wise Per-tag Group-wise

Training Cost

Number of GPUs 8 8 8 8 8 GPU Type H100 H100 H100 H100 H100 Training Rollouts 0.2M 0.05M 0.13M 0.5M 0.06M

Comparison to flow-only RL. A natural question is whether the gains from PromptRL can be matched by simply scaling up flow-only RL training. To investigate this, we train a flow-only baseline with twice the number of rollouts and compare it against PromptRL in Table 9. Despite the 2× computational advantage, flow-only RL still underperforms PromptRL across all metrics (GenEval: 0.93 vs. 0.97, OCR: 0.93 vs. 0.98, PickScore: 23.85 vs. 24.05). This result suggests that the limitations of flow-only RL—namely exploration collapse and prompt overfitting identified in Section 3—cannot be overcome by additional rollouts alone. Joint LM-FM optimization fundamentally reshapes the optimization landscape by continuously

- Table 8. Generalization of PromptRL’s prompt enhancer (PE) to unseen flow models on GenEval. All evaluations are conducted at 1024×1024 resolution with 20 inference steps. We compare official pretrained weights without PE, with our PE, and with model-specific RL-tuned PE. Both PE modules use Qwen-2.5-VL-Instruct-3B. Our learned PE demonstrates strong generalization capability, effectively improving pretrained models that were not seen during training.

Model Original Ours’ PE Prompt-only RL

SANA 0.62 0.70 0.76 SD3 0.58 0.77 0.83

- Table 9. Comparison between PromptRL and flow-only RL. Flow-only RL is trained with 2× the number of rollouts to control for computational budget.

#### Method Rollouts GenEval↑ OCR↑ PickScore↑

Flow-only RL 2× 0.93 0.93 23.85 PromptRL 1× 0.97 0.98 24.05

injecting linguistic diversity, enabling the FM to escape narrow reward modes and achieve higher performance ceilings.

Limitations. Although PromptRL achieves strong performance through joint LM-FM optimization, the FM and LM develop a degree of co-adaptation during training. Specifically, when replacing our co-trained prompt enhancer with a different LM (e.g., Qwen-3) at inference time, we observe a performance drop on GenEval from 0.97 to 0.88. This indicates that the FM becomes partially specialized to the linguistic patterns produced by its training-time LM partner. Fortunately, this co-adaptation is by design rather than a fundamental flaw—PromptRL explicitly aims to jointly optimize both components for deployment as a unified system. In practical scenarios where the co-trained LM and FM are used together, this tight coupling translates into the state-of-the-art performance we report. Future work could explore techniques such as multi-LM training or regularization strategies to further improve cross-LM generalization when broader compatibility is desired.

Why JointRL matters? As T2I models become more powerful, they also become more sensitive to how prompts are phrased—small linguistic changes can lead to significant differences in generation quality. This makes the PE module just as important as the FM itself. Yet existing methods typically train these two components separately, either through independent SFT or RL. We believe this isolated approach misses a crucial point: the best way to refine prompts depends on the current FM, and vice versa. By treating PE and FM as a unified system, PromptRL allows both to improve together—the LM discovers prompt variants that yield higher rewards, while the expanded prompt diversity provides richer exploration signals that accelerate FM training efficiency.

